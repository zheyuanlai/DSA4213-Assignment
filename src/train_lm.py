"""Config-driven language-model training (TASK §4.3), identical for every neural run.

Per epoch: random start offset in [0, L), non-overlapping L-blocks of the LM-train stream (final
incomplete block and final incomplete batch dropped), shuffled, batches of 64. AdamW (weight decay
on ndim>=2 parameters only), linear warm-up then cosine decay to 10 % of peak over
`schedule_epochs` epochs, gradient-norm clipping at 1.0 with the pre-clip norm logged every step.
LM-val after every epoch, early stopping with patience 3, best-val checkpoint kept. JSONL log per
run plus a per-step array (loss, grad norm). A non-finite loss aborts the run (user rule D31).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common import (PAD, ROOT, JsonlLogger, env_info, fmt_seconds, load_config, set_seed,
                        sha256_array, write_json)
from src.evaluate import evaluate_stream
from src.models import build_model, load_embedding_table, logprob_fn_for, param_breakdown


def lr_at(step: int, peak: float, warmup: int, total: int, final_frac: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    final = peak * final_frac
    return final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model, ocfg: dict):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": ocfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=ocfg["lr"], betas=tuple(ocfg.get("betas", (0.9, 0.999))),
                             eps=ocfg.get("eps", 1e-8)), {"decay_params": sum(p.numel() for p in decay),
                                                          "no_decay_params": sum(p.numel() for p in no_decay)}


def epoch_batches(stream: torch.Tensor, L: int, bs: int, steps_per_epoch: int, rng: np.random.Generator):
    """Yield (inputs, targets) batches for one epoch: random offset, non-overlapping blocks, shuffled."""
    n = len(stream)
    offset = int(rng.integers(0, L))
    n_blocks = (n - offset - 1) // L
    starts = offset + L * np.arange(n_blocks)
    rng.shuffle(starts)
    starts = starts[: steps_per_epoch * bs]        # constant number of steps per epoch
    idx = torch.as_tensor(starts, device=stream.device)
    ar = torch.arange(L, device=stream.device)
    for b in range(steps_per_epoch):
        s = idx[b * bs:(b + 1) * bs]
        pos = s[:, None] + ar[None, :]
        yield stream[pos], stream[pos + 1]


def count_spikes(losses: np.ndarray, window: int = 50, factor: float = 1.5) -> int:
    """Spike = per-step loss > factor × trailing mean of the previous `window` steps, from step window+1."""
    if len(losses) <= window:
        return 0
    cs = np.cumsum(np.insert(losses, 0, 0.0))
    trailing = (cs[window:-1] - cs[:-window - 1]) / window   # mean of losses[i-window:i] for i >= window
    return int((losses[window:] > factor * trailing).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    run = cfg["run_name"]
    run_dir = ROOT / "results/runs" / run
    ckpt_dir = ROOT / "checkpoints" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        log_path.unlink()
    logger = JsonlLogger(log_path)

    # --- data
    dcfg, ocfg, tcfg = cfg["data"], cfg["optim"], cfg["train"]
    L, bs = dcfg["block_length"], ocfg["batch_size"]
    streams = ROOT / dcfg["streams_dir"]
    train_np = np.load(streams / "lm_train.npy")
    if dcfg.get("train_fraction", 1.0) < 1.0:
        train_np = train_np[: int(len(train_np) * dcfg["train_fraction"])]
    val_np = np.load(streams / "lm_val.npy")
    stream = torch.from_numpy(train_np.astype(np.int64)).to(device)
    vocab_size = len(__import__("json").load(open(streams / "vocab.json"))["itos"])
    n = len(stream)
    steps_per_epoch = ((n - (L - 1) - 1) // L) // bs      # worst-case offset so every epoch has the same length
    total_steps = tcfg.get("schedule_epochs", tcfg["max_epochs"]) * steps_per_epoch

    # --- model
    model = build_model(cfg["model"], vocab_size).to(device)
    ecfg = cfg.get("embedding", {"init": "random", "freeze": False})
    emb_info = {"init": ecfg.get("init", "random"), "freeze": bool(ecfg.get("freeze", False))}
    if ecfg.get("init") == "file":
        table = np.load(ROOT / ecfg["path"])
        load_embedding_table(model, table, freeze=emb_info["freeze"])
        emb_info["path"] = ecfg["path"]
    emb_info["sha256_before"] = sha256_array(model.emb.weight)
    emb_info["per_dim_std_before"] = float(model.emb.weight.detach().float().std(dim=0).mean().item())
    breakdown = param_breakdown(model)
    optimizer, opt_info = make_optimizer(model, ocfg)

    env = env_info()
    logger.log({"record": "config", "config": cfg, "env": env, "device": str(device), "vocab_size": vocab_size,
                "train_tokens": n, "steps_per_epoch": steps_per_epoch, "total_schedule_steps": total_steps,
                "param_breakdown": breakdown, "optimizer_groups": opt_info, "embedding": emb_info})
    print(f"[{run}] {cfg['model']['type']} params {breakdown['total']:,} (trainable {breakdown['trainable']:,}); "
          f"{n:,} train tokens, {steps_per_epoch} steps/epoch, schedule {total_steps} steps, device {device}", flush=True)

    # --- training loop
    step_losses, step_gnorms = [], []
    best_val, best_epoch, epochs_since_best = float("inf"), 0, 0
    train_time, epoch_times, val_history = 0.0, [], []
    step = 0
    log_every = tcfg.get("log_every", 50)
    status = "completed"
    failure = None
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter()
        for x, y in epoch_batches(stream, L, bs, steps_per_epoch, rng):
            step += 1
            lr = lr_at(step, ocfg["lr"], ocfg["warmup_steps"], total_steps, ocfg.get("final_lr_fraction", 0.1))
            for g in optimizer.param_groups:
                g["lr"] = lr
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.reshape(-1), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
            optimizer.step()
            li, gi = loss.item(), gnorm.item()
            step_losses.append(li)
            step_gnorms.append(gi)
            if not math.isfinite(li) or not math.isfinite(gi):
                status, failure = "failed", {"reason": "non-finite loss or grad norm", "step": step, "epoch": epoch,
                                             "loss": li, "grad_norm": gi}
                break
            if step % log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = train_time + (time.perf_counter() - t_epoch)
                logger.log({"record": "step", "step": step, "epoch": epoch,
                            "train_loss_last50": float(np.mean(step_losses[-log_every:])), "lr": lr,
                            "grad_norm": gi, "grad_norm_mean_last50": float(np.mean(step_gnorms[-log_every:])),
                            "train_time_s": elapsed, "tokens_seen": step * bs * L})
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t_epoch
        train_time += epoch_time
        epoch_times.append(epoch_time)
        if status == "failed":
            logger.log({"record": "failure", **failure})
            print(f"[{run}] FAILED: {failure}", flush=True)
            break

        model.eval()
        t_val = time.perf_counter()
        val = evaluate_stream(logprob_fn_for(model, device), val_np, L, bs)
        val_time = time.perf_counter() - t_val
        improved = val["ce"] < best_val
        if improved:
            best_val, best_epoch, epochs_since_best = val["ce"], epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_ce": val["ce"],
                        "param_breakdown": breakdown, "vocab_size": vocab_size}, ckpt_dir / "best.pt")
        else:
            epochs_since_best += 1
        val_history.append(val["ce"])
        logger.log({"record": "epoch", "epoch": epoch, "step": step, "val_ce": val["ce"], "val_ppl": val["ppl"],
                    "val_unk_target_fraction": val["unk_target_fraction"], "train_loss_epoch_mean": float(np.mean(step_losses[-steps_per_epoch:])),
                    "epoch_train_time_s": epoch_time, "cumulative_train_time_s": train_time, "val_time_s": val_time,
                    "improved": improved, "best_epoch": best_epoch, "lr_end_of_epoch": lr})
        print(f"[{run}] epoch {epoch:2d} | train {np.mean(step_losses[-steps_per_epoch:]):.4f} | val CE {val['ce']:.4f} PPL {val['ppl']:.2f} "
              f"| {fmt_seconds(epoch_time)} train, {val_time:.1f}s val | best epoch {best_epoch}{' *' if improved else ''}", flush=True)
        if epochs_since_best >= tcfg["patience"]:
            status = "early_stopped"
            break

    # --- per-step arrays and stability statistics
    losses, gnorms = np.asarray(step_losses, dtype=np.float32), np.asarray(step_gnorms, dtype=np.float32)
    np.savez(run_dir / "steps.npz", loss=losses, grad_norm=gnorms)
    finite = np.isfinite(gnorms)
    stability = {"max_grad_norm": float(gnorms[finite].max()) if finite.any() else None,
                 "mean_grad_norm": float(gnorms[finite].mean()) if finite.any() else None,
                 "loss_spikes": count_spikes(losses[np.isfinite(losses)]),
                 "spike_definition": "per-step train loss > 1.5 x trailing mean of the previous 50 steps, counted from step 51",
                 "non_finite_steps": int((~np.isfinite(losses)).sum() + (~finite).sum())}

    summary = {"record": "summary", "status": status, "epochs_run": len(epoch_times), "best_epoch": best_epoch,
               "best_val_ce": best_val if best_epoch else None, "val_ce_history": val_history,
               "train_time_s_total": train_time, "train_time_s_per_epoch": epoch_times,
               "train_time_s_to_best": float(sum(epoch_times[:best_epoch])) if best_epoch else None,
               "train_tokens_per_s": (step * bs * L) / train_time if train_time > 0 else None,
               "steps": step, "stability": stability, "embedding": emb_info}

    # --- frozen-embedding checksum and optional single LM-test evaluation of the best checkpoint
    if best_epoch and status != "failed":
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        emb_info["sha256_after"] = sha256_array(model.emb.weight)
        emb_info["per_dim_std_after"] = float(model.emb.weight.detach().float().std(dim=0).mean().item())
        emb_info["frozen_table_unchanged"] = (emb_info["sha256_after"] == emb_info["sha256_before"]) if emb_info["freeze"] else None
        if emb_info["freeze"]:
            assert emb_info["frozen_table_unchanged"], "frozen embedding table changed during training!"
        if tcfg.get("evaluate_test", False):
            model.eval()
            test_np = np.load(streams / "lm_test.npy")
            test, ce, targets = evaluate_stream(logprob_fn_for(model, device), test_np, L, bs, return_ce=True)
            derived = ROOT / "data/derived" / run
            derived.mkdir(parents=True, exist_ok=True)
            np.save(derived / "test_ce.npy", ce.astype(np.float32))
            logger.log({"record": "test", "split": "lm_test", "ce": test["ce"], "ppl": test["ppl"],
                        "n_targets": test["n_targets"], "unk_target_fraction": test["unk_target_fraction"],
                        "best_epoch": best_epoch, "checkpoint": str(ckpt_dir / "best.pt"),
                        "ce_by_position": test["ce_by_position"], "test_ce_path": str(derived / "test_ce.npy")})
            summary["test_ce"], summary["test_ppl"] = test["ce"], test["ppl"]
            print(f"[{run}] LM-test CE {test['ce']:.4f} PPL {test['ppl']:.2f} (best epoch {best_epoch})", flush=True)
    logger.log(summary)
    logger.close()
    print(f"[{run}] {status}: best val CE {best_val:.4f} at epoch {best_epoch}; train time {fmt_seconds(train_time)}; "
          f"max grad norm {stability['max_grad_norm']}; spikes {stability['loss_spikes']}", flush=True)
    sys.exit(1 if status == "failed" else 0)


if __name__ == "__main__":
    main()
