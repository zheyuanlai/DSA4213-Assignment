"""Fixed-length representations from frozen LMs for Part III (TASK §6).

Input per review: <eos> + the first (or, for the ablation, last) 255 review tokens = at most 256
positions, right-padded with <pad> inside a batch. Representations are the 300-d pre-head vector of
each LM (D30): `last` = the vector at the last non-pad position, `mean` = the mean over non-pad
positions. LMs are frozen, in eval mode, seed-0 best checkpoints. Also writes the shared input
matrices (for the TF-IDF and GloVe-mean baselines) and results/part3/input_stats.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.common import EOS, PAD, UNK, ROOT, load_config, read_json, smoke_indices, write_json
from src.generate import load_neural


def build_inputs(ids: np.ndarray, offsets: np.ndarray, max_tokens: int, truncate: str, sel: np.ndarray) -> dict:
    N = len(sel)
    L = max_tokens + 1
    X = np.full((N, L), PAD, dtype=np.int64)
    full_len = np.diff(offsets)[sel]
    n_input = np.minimum(full_len, max_tokens) + 1
    for i, j in enumerate(sel):
        r = ids[offsets[j]:offsets[j + 1]]
        r = r[:max_tokens] if truncate == "first" else r[-max_tokens:]
        X[i, 0] = EOS
        X[i, 1:1 + len(r)] = r
    return {"ids": X, "n_input": n_input, "full_len": full_len, "truncated": full_len > max_tokens}


def input_stats(inp: dict) -> dict:
    X, n = inp["ids"], inp["n_input"]
    mask = np.arange(X.shape[1])[None, :] < n[:, None]
    review_tok = mask.copy(); review_tok[:, 0] = False       # exclude the leading <eos>
    return {"n_reviews": int(len(n)), "n_truncated": int(inp["truncated"].sum()), "pct_truncated": 100.0 * float(inp["truncated"].mean()),
            "mean_full_length_tokens": float(inp["full_len"].mean()), "median_full_length_tokens": float(np.median(inp["full_len"])),
            "mean_input_positions_incl_eos": float(n.mean()),
            "unk_rate_of_inputs": float((X[review_tok] == UNK).mean()), "input_review_tokens": int(review_tok.sum())}


@torch.no_grad()
def extract(model, X: np.ndarray, n_input: np.ndarray, device, bs: int) -> dict[str, np.ndarray]:
    N, L = X.shape
    last = np.zeros((N, model.d_model), dtype=np.float32)
    mean = np.zeros((N, model.d_model), dtype=np.float32)
    for b in range(0, N, bs):
        x = torch.as_tensor(X[b:b + bs], device=device)
        n = torch.as_tensor(n_input[b:b + bs], device=device)
        h = model.features(x).float()                                  # [B, L, 300] pre-head vectors
        mask = (torch.arange(L, device=device)[None, :] < n[:, None]).float()
        last[b:b + bs] = h[torch.arange(len(n), device=device), n - 1].cpu().numpy()
        mean[b:b + bs] = ((h * mask[:, :, None]).sum(1) / n[:, None].float()).cpu().numpy()
    return {"last": last, "mean": mean}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part3/classify.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N train and N test reviews, outputs under smoke dirs")
    ap.add_argument("--truncate", choices=["first", "last"], default=None)
    ap.add_argument("--only-lm", nargs="*", default=None, help="subset of LM names (used by the ablation)")
    args = ap.parse_args()
    cfg = load_config(args.config, {"limit_reviews": args.limit})   # limit slices the full data (no stream redirect)
    if cfg.get("limit_reviews"):
        cfg["derived_dir"], cfg["results_dir"] = "data/smoke/derived/part3", "results/smoke/part3"
    truncate = args.truncate or cfg["truncate"]
    suffix = "" if truncate == "first" else "_last255"
    derived = ROOT / cfg["derived_dir"]; derived.mkdir(parents=True, exist_ok=True)
    results = ROOT / cfg["results_dir"]; results.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = {"truncate": truncate, "max_review_tokens": cfg["max_review_tokens"], "device": str(device)}
    inputs = {}
    for split in ("train", "test"):
        z = np.load(ROOT / cfg["streams_dir"] / f"imdb_{split}.npz")
        ids, offsets, labels = z["ids"], z["offsets"], z["labels"]
        orig_idx = smoke_indices(labels, cfg.get("limit_reviews"))      # all reviews unless --limit (stratified smoke subset)
        inp = build_inputs(ids, offsets, cfg["max_review_tokens"], truncate, orig_idx)
        inp["labels"] = labels[orig_idx]
        inp["orig_idx"] = orig_idx
        inputs[split] = inp
        stats[split] = input_stats(inp)
        np.savez(derived / f"inputs_{split}{suffix}.npz", **inp)
    write_json(stats, results / f"input_stats{suffix}.json")
    print(json.dumps({k: v for k, v in stats.items() if k in ("train", "test")}, indent=1))

    lms = {k: v for k, v in cfg["lms"].items() if not args.only_lm or k in args.only_lm}
    timing = {}
    for name, run in lms.items():
        if not (ROOT / "checkpoints" / run / "best.pt").exists():
            print(f"!! checkpoint missing for {name} ({run}); skipped"); continue
        model, ckpt = load_neural(run, device)
        t0 = time.perf_counter()
        for split, inp in inputs.items():
            feats = extract(model, inp["ids"], inp["n_input"], device, cfg["feature_batch_size"])
            for pool, F in feats.items():
                np.save(derived / f"{name}_{pool}_{split}{suffix}.npy", F)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing[name] = {"run": run, "epoch": ckpt["epoch"], "seconds_train_and_test": time.perf_counter() - t0}
        print(f"{name}: features from {run} (epoch {ckpt['epoch']}) in {timing[name]['seconds_train_and_test']:.1f}s", flush=True)
        del model; torch.cuda.empty_cache() if device.type == "cuda" else None
    write_json({"stats": stats, "feature_extraction": timing, "representation": "300-d pre-head vector (RNN/LSTM: projection output; Transformer: final-LayerNorm output); last = last non-pad position; mean = masked mean"},
               results / f"features{suffix}.json")


if __name__ == "__main__":
    main()
