"""Text generation with matched decoding for every model (TASK §4.5).

Prompts are prefixed with <eos>. Decoding is identical for all models: (i) greedy, (ii) sampling
with temperature 0.8 and top-k 50. <pad> and <unk> are suppressed (log-probability set to -inf)
for every model; exactly 60 new tokens are generated and generation does not stop at <eos>
(rendered as ⏎). The n-gram's next-token distribution goes through the very same torch sampler.
RNN/LSTM decode incrementally (mathematically identical to re-reading the prefix); the Transformer
recomputes the full prefix at every step (no KV cache).

Metrics (NOTES D13): distinct-1/2 = distinct n-grams / total n-grams pooled over all generated
tokens of a model's samples (corpus level); rep-4 = fraction of a sample's 4-grams that already
occurred earlier in the same sample, averaged over samples.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.common import EOS, PAD, UNK, ROOT, write_json
from src.models import build_model
from src.ngram import KneserNeyLM
from src.vocab import Vocab

PROMPTS = ["this movie was", "i have never seen", "the acting in this film", "if you are looking for", "the plot of this"]
N_NEW = 60
TEMPERATURE, TOP_K = 0.8, 50
N_SAMPLES = 20


def load_neural(run_name: str, device: torch.device):
    ckpt = torch.load(ROOT / "checkpoints" / run_name / "best.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"], ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


class NeuralGenerator:
    def __init__(self, model, device):
        self.model, self.device, self.kind = model, device, model.kind

    @torch.no_grad()
    def start(self, prefix: list[int]) -> torch.Tensor:
        """Consume the prompt; return log-probabilities (float32, [V]) of the next token."""
        x = torch.as_tensor([prefix], device=self.device)
        if self.kind == "transformer":
            self.prefix = list(prefix)
            logits = self.model(x)[0, -1]
        else:
            z, self.state = self.model.features(x, None, return_state=True)
            logits = self.model.head(z[:, -1])[0]
        return torch.log_softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def step(self, token: int) -> torch.Tensor:
        if self.kind == "transformer":
            self.prefix.append(token)
            x = torch.as_tensor([self.prefix], device=self.device)
            logits = self.model(x)[0, -1]                       # full-prefix recompute, no KV cache
        else:
            x = torch.as_tensor([[token]], device=self.device)
            logits, self.state = self.model.step(x, self.state)
            logits = logits[0, -1]
        return torch.log_softmax(logits.float(), dim=-1)


class NgramGenerator:
    def __init__(self, lm: KneserNeyLM):
        self.lm, self.kind = lm, "ngram"

    def start(self, prefix: list[int]) -> torch.Tensor:
        self.hist = list(prefix)
        return torch.log(torch.as_tensor(self.lm.next_distribution(self.hist)))

    def step(self, token: int) -> torch.Tensor:
        self.hist.append(token)
        return torch.log(torch.as_tensor(self.lm.next_distribution(self.hist)))


def choose(logp: torch.Tensor, mode: str, gen: torch.Generator | None) -> int:
    logp = logp.clone().float()
    logp[PAD] = -math.inf
    logp[UNK] = -math.inf
    if mode == "greedy":
        return int(torch.argmax(logp).item())
    scaled = logp / TEMPERATURE
    topv, topi = torch.topk(scaled, TOP_K)
    probs = torch.softmax(topv, dim=-1)
    j = torch.multinomial(probs.cpu(), 1, generator=gen).item()
    return int(topi[j].item())


def generate(g, prompt_ids: list[int], mode: str, seed: int | None = None, n_new: int = N_NEW) -> list[int]:
    gen = torch.Generator().manual_seed(seed) if mode == "sample" else None
    logp = g.start([EOS] + prompt_ids)
    out = []
    for _ in range(n_new):
        t = choose(logp, mode, gen)
        out.append(t)
        logp = g.step(t)
    return out


def distinct_n(samples: list[list[int]], n: int) -> float:
    grams = [tuple(s[i:i + n]) for s in samples for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def rep_n(sample: list[int], n: int = 4) -> float:
    grams = [tuple(sample[i:i + n]) for i in range(len(sample) - n + 1)]
    seen, rep = set(), 0
    for gr in grams:
        if gr in seen:
            rep += 1
        seen.add(gr)
    return rep / len(grams) if grams else float("nan")


def render(vocab: Vocab, ids: list[int]) -> str:
    return " ".join("⏎" if t == EOS else vocab.itos[t] for t in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["part1/rnn_s0", "part1/lstm_s0", "part1/transformer_s0"])
    ap.add_argument("--labels", nargs="*", default=None, help="display labels for --runs (default: model type)")
    ap.add_argument("--ngram", default="checkpoints/ngram/kn_selected.npz", help="path or 'none'")
    ap.add_argument("--out", default="results/part1")
    ap.add_argument("--vocab", default="data/streams/vocab.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(args.vocab)
    for p in PROMPTS:
        missing = [t for t in p.split() if t not in vocab.stoi]
        assert not missing, f"prompt tokens not in vocabulary: {missing}"

    gens: dict[str, tuple] = {}
    if args.ngram != "none":
        gens["ngram"] = (NgramGenerator(KneserNeyLM.load(args.ngram)), args.ngram)
    labels = args.labels or [None] * len(args.runs)
    for run, label in zip(args.runs, labels):
        model, ckpt = load_neural(run, device)
        gens[label or model.kind] = (NeuralGenerator(model, device), f"checkpoints/{run}/best.pt (epoch {ckpt['epoch']})")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    records, metrics = [], []
    md = ["# Generations (TASK §4.5)\n",
          "Prompts are prefixed with `<eos>`; `<pad>` and `<unk>` are suppressed for every model; exactly "
          f"{N_NEW} new tokens; `⏎` = `<eos>`. Greedy, and sampling with temperature {TEMPERATURE}, top-k {TOP_K} "
          "(sampling seeds 0 and 1 shown; 20 seeds per prompt feed `generation_metrics.csv`). "
          "RNN/LSTM decode incrementally; the Transformer recomputes the full prefix each step (no KV cache).\n",
          "Checkpoints: " + "; ".join(f"**{k}** = `{src}`" for k, (_, src) in gens.items()) + "\n"]
    for name, (g, src) in gens.items():
        greedy_all, samples_all, reps_g, reps_s = [], [], [], []
        for pi, prompt in enumerate(PROMPTS):
            pid = vocab.encode(prompt.split())
            greedy = generate(g, pid, "greedy")
            greedy_all.append(greedy)
            reps_g.append(rep_n(greedy))
            records.append({"model": name, "prompt": prompt, "mode": "greedy", "seed": None, "tokens": render(vocab, greedy)})
            for s in range(N_SAMPLES):
                smp = generate(g, pid, "sample", seed=s)
                samples_all.append(smp)
                reps_s.append(rep_n(smp))
                records.append({"model": name, "prompt": prompt, "mode": "sample", "seed": s, "tokens": render(vocab, smp)})
        metrics.append({"model": name, "n_samples": len(samples_all), "sample_distinct1": distinct_n(samples_all, 1),
                        "sample_distinct2": distinct_n(samples_all, 2), "sample_rep4_mean": float(np.mean(reps_s)),
                        "greedy_distinct1": distinct_n(greedy_all, 1), "greedy_distinct2": distinct_n(greedy_all, 2),
                        "greedy_rep4_mean": float(np.mean(reps_g)), "checkpoint": src})
        print(f"{name}: distinct-1 {metrics[-1]['sample_distinct1']:.3f} distinct-2 {metrics[-1]['sample_distinct2']:.3f} rep-4 {metrics[-1]['sample_rep4_mean']:.3f} | greedy rep-4 {metrics[-1]['greedy_rep4_mean']:.3f}", flush=True)

    df = pd.DataFrame(records)
    df.to_json(out_dir / "generations_all.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame(metrics).to_csv(out_dir / "generation_metrics.csv", index=False)
    for prompt in PROMPTS:
        md.append(f"\n## Prompt: `{prompt}`\n")
        md.append("| Model | Greedy | Sample (seed 0) | Sample (seed 1) |\n|---|---|---|---|")
        for name in gens:
            d = df[(df.model == name) & (df.prompt == prompt)]
            g0 = d[d["mode"] == "greedy"].tokens.iloc[0]
            s0 = d[(d["mode"] == "sample") & (d.seed == 0)].tokens.iloc[0]
            s1 = d[(d["mode"] == "sample") & (d.seed == 1)].tokens.iloc[0]
            md.append(f"| {name} | {g0} | {s0} | {s1} |")
    (out_dir / "generations.md").write_text("\n".join(md) + "\n")
    print("wrote", out_dir / "generations.md", out_dir / "generation_metrics.csv")


if __name__ == "__main__":
    main()
