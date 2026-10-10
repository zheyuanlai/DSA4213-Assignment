"""Cost measurements (TASK §4.4): environment record, inference workloads A (scoring) and B (generation).

Workload A: score the entire LM-test block set with batch size 64 (inputs/targets resident on the
device; timed region = forward + log-softmax + gather + masked sum for all batches); 2 warm-up
runs then 5 timed repeats with torch.cuda.synchronize() before each timer read; median tokens/s
over non-pad targets. The n-gram runs the same blocks on CPU in a single process
(OMP_NUM_THREADS=1) and records the load average.
Workload B: batch 1, the 5 prompts, 60 new tokens each, greedy; every step is timed with a
synchronize; median ms/token over the 300 steps (one untimed warm-up generation first). The
Transformer recomputes the full prefix at each step (no KV cache); RNN/LSTM carry their state.
An nvidia-smi snapshot is taken immediately before every GPU timing run; if the selected GPU is
not idle the run waits and retries (user rule D67).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time

import numpy as np
import torch

from src.common import PAD, ROOT, cpu_model, env_info, read_json, write_json
from src.evaluate import make_blocks
from src.generate import EOS, PROMPTS, NeuralGenerator, NgramGenerator, choose, load_neural
from src.ngram import KneserNeyLM
from src.vocab import Vocab


def gpu_snapshot() -> list[dict]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, u, m, t = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "util_pct": int(u), "mem_used_mib": int(m), "mem_total_mib": int(t)})
    return rows


def other_process_memory_mib(phys_index: int) -> int:
    """GPU memory (MiB) used on the given physical GPU by processes other than this one."""
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    uuid = next(l.split(",")[1].strip() for l in uuid.strip().splitlines() if int(l.split(",")[0]) == phys_index)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    total = 0
    for line in apps.strip().splitlines():
        u, pid, mem = [x.strip() for x in line.split(",")]
        if u == uuid and int(pid) != os.getpid():
            total += int(mem)
    return total


def wait_for_idle_gpu(max_util: int = 5, max_other_mem_mib: int = 1500, retries: int = 40, sleep_s: int = 30) -> dict:
    """Wait until the single visible GPU (physical index from CUDA_VISIBLE_DEVICES, default 2) is idle.

    Idle = utilisation <= max_util % and memory held by *other* processes <= max_other_mem_mib
    (another user's idle 528 MiB context is always present on this node). The snapshot is taken
    after this process's own work has been synchronised, so its own memory is excluded.
    """
    phys = int(os.environ.get("CUDA_VISIBLE_DEVICES", "2").split(",")[0])
    for attempt in range(retries):
        time.sleep(1.0)  # let utilisation counters settle after our own synchronised work
        snap = gpu_snapshot()
        me = next(r for r in snap if r["index"] == phys)
        other = other_process_memory_mib(phys)
        idle = me["util_pct"] <= max_util and other <= max_other_mem_mib
        if idle:
            return {"gpu_index": phys, "snapshot": snap, "other_process_mem_mib": other, "loadavg": os.getloadavg(),
                    "attempts": attempt + 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"GPU {phys} busy (util {me['util_pct']} %, other processes {other} MiB); waiting {sleep_s}s ...", flush=True)
        time.sleep(sleep_s)
    raise RuntimeError(f"GPU {phys} did not become idle after {retries} retries")


@torch.no_grad()
def workload_a_neural(model, inputs: torch.Tensor, targets: torch.Tensor, bs: int = 64, warm: int = 2, reps: int = 5) -> dict:
    N = inputs.shape[0]
    n_targets = int((targets != PAD).sum().item())
    def run():
        total = torch.zeros((), device=inputs.device, dtype=torch.float64)
        for b in range(0, N, bs):
            x, y = inputs[b:b + bs], targets[b:b + bs]
            lp = torch.log_softmax(model(x).float(), dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            total += -(lp * (y != PAD)).double().sum()
        return total
    for _ in range(warm):
        run()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        total = run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return {"n_targets": n_targets, "n_blocks": int(N), "batch_size": bs, "times_s": times,
            "median_tokens_per_s": n_targets / statistics.median(times), "ce_check": float(total.item()) / n_targets}


def workload_a_ngram(lm: KneserNeyLM, inputs: np.ndarray, targets: np.ndarray, bs: int = 64, warm: int = 2, reps: int = 5) -> dict:
    N = inputs.shape[0]
    mask = targets != PAD
    n_targets = int(mask.sum())
    def run():
        total = 0.0
        for b in range(0, N, bs):
            lp = lm.logprob_blocks(inputs[b:b + bs], targets[b:b + bs])
            total += float(-(lp * mask[b:b + bs]).sum())
        return total
    for _ in range(warm):
        run()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        total = run()
        times.append(time.perf_counter() - t)
    return {"n_targets": n_targets, "n_blocks": int(N), "batch_size": bs, "times_s": times,
            "median_tokens_per_s": n_targets / statistics.median(times), "ce_check": total / n_targets,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "loadavg_before": os.getloadavg()}


def workload_b(g, vocab: Vocab, cuda: bool, n_new: int = 60) -> dict:
    def gen(prompt, timed):
        pid = vocab.encode(prompt.split())
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logp = g.start([EOS] + pid)
        if cuda:
            torch.cuda.synchronize()
        step_times, prompt_time = [], time.perf_counter() - t0
        for _ in range(n_new):
            t = time.perf_counter()
            tok = choose(logp, "greedy", None)
            logp = g.step(tok)
            if cuda:
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t)
        return step_times, prompt_time
    gen(PROMPTS[0], False)  # warm-up
    all_steps, prompt_times = [], []
    for p in PROMPTS:
        st, pt = gen(p, True)
        all_steps.extend(st)
        prompt_times.append(pt)
    return {"n_steps_timed": len(all_steps), "median_ms_per_token": 1000 * statistics.median(all_steps),
            "mean_ms_per_token": 1000 * float(np.mean(all_steps)), "prompt_encode_ms_mean": 1000 * float(np.mean(prompt_times)),
            "decoding": "greedy", "n_new_tokens": n_new}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["part1/rnn_s0", "part1/lstm_s0", "part1/transformer_s0"])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--ngram", default="checkpoints/ngram/kn_selected.npz")
    ap.add_argument("--out", default="results/part1/timing.json")
    ap.add_argument("--env-out", default="results/env.json")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    vocab = Vocab.load("data/streams/vocab.json")
    test = np.load(ROOT / "data/streams/lm_test.npy")
    inputs_np, targets_np = make_blocks(test, 256)
    inputs = torch.as_tensor(inputs_np, device=device)
    targets = torch.as_tensor(targets_np, device=device)

    env = env_info()
    write_json(env, args.env_out)
    results = {"env": env, "cpu_model": cpu_model(), "workload_a": {}, "workload_b": {}, "notes": {
        "workload_a": "entire LM-test (2138 blocks of 256) in batches of 64, tokens/s over non-pad targets; 2 warm-ups, 5 timed repeats, median; data resident on device; n-gram on CPU single process",
        "workload_b": "batch 1, 5 prompts x 60 new tokens, greedy; per-step synchronize; median ms/token over 300 steps; Transformer recomputes the full prefix each step (no KV cache); RNN/LSTM carry hidden state; n-gram on CPU",
        "idle_rule": "nvidia-smi snapshot before each GPU timing; wait while util > 5% or memory held by other processes > 1500 MiB on the selected GPU"}}

    labels = args.labels or [None] * len(args.runs)
    for run, label in zip(args.runs, labels):
        model, ckpt = load_neural(run, device)
        name = label or model.kind
        pre = wait_for_idle_gpu()
        a = workload_a_neural(model, inputs, targets)
        a["gpu_before"] = pre
        results["workload_a"][name] = {"run": run, "device": torch.cuda.get_device_name(0), **a}
        print(f"[A] {name}: {a['median_tokens_per_s']:,.0f} tokens/s (CE check {a['ce_check']:.4f})", flush=True)
        pre = wait_for_idle_gpu()
        b = workload_b(NeuralGenerator(model, device), vocab, cuda=True)
        b["gpu_before"] = pre
        results["workload_b"][name] = {"run": run, "device": torch.cuda.get_device_name(0), **b}
        print(f"[B] {name}: {b['median_ms_per_token']:.2f} ms/token", flush=True)
        del model
        torch.cuda.empty_cache()

    if args.ngram != "none":
        lm = KneserNeyLM.load(args.ngram)
        a = workload_a_ngram(lm, inputs_np, targets_np)
        results["workload_a"]["ngram"] = {"run": args.ngram, "device": f"CPU: {cpu_model()}", **a}
        print(f"[A] ngram: {a['median_tokens_per_s']:,.0f} tokens/s (CE check {a['ce_check']:.4f}, loadavg {a['loadavg_before']})", flush=True)
        la = os.getloadavg()
        b = workload_b(NgramGenerator(lm), vocab, cuda=False)
        b["loadavg_before"] = la
        results["workload_b"]["ngram"] = {"run": args.ngram, "device": f"CPU: {cpu_model()}", **b}
        print(f"[B] ngram: {b['median_ms_per_token']:.2f} ms/token", flush=True)
    write_json(results, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
