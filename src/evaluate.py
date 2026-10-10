"""Common evaluation protocol (TASK §3) — used unchanged for every model in Parts I and II.

Blocks of length L=256: inputs = s[kL : kL+L], targets = s[kL+1 : kL+L+1]; the final incomplete
block is right-padded with <pad>; pad targets are excluded from every loss and average.
Context rule: the per-token log-probability of the target at position j may depend only on
inputs[0..j] of the same block. Each model supplies `logprob_fn(inputs, targets) -> logp` of shape
[B, L] (natural log, float64/float32); this module turns it into masked cross-entropy statistics.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.common import EOS, PAD, UNK

LogProbFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def make_blocks(stream: np.ndarray, L: int = 256, pad: int = PAD) -> tuple[np.ndarray, np.ndarray]:
    """Split a token stream into [N, L] inputs and targets exactly as TASK §3 specifies."""
    s = np.asarray(stream, dtype=np.int64)
    n = len(s)
    if n < 2:
        raise ValueError("stream must contain at least two tokens")
    n_blocks = math.ceil((n - 1) / L)
    padded = np.full(n_blocks * L + 1, pad, dtype=np.int64)
    padded[:n] = s
    inputs = padded[:-1].reshape(n_blocks, L)
    targets = padded[1:].reshape(n_blocks, L)
    return inputs, targets


def score_blocks(logprob_fn: LogProbFn, inputs: np.ndarray, targets: np.ndarray,
                 batch_size: int = 64) -> np.ndarray:
    """Per-token cross-entropy (nats) of shape [N, L]; NaN where the target is <pad>."""
    N = inputs.shape[0]
    ce = np.full(inputs.shape, np.nan, dtype=np.float64)
    for b in range(0, N, batch_size):
        x, y = inputs[b:b + batch_size], targets[b:b + batch_size]
        lp = np.asarray(logprob_fn(x, y), dtype=np.float64)
        assert lp.shape == x.shape, (lp.shape, x.shape)
        mask = y != PAD
        ce[b:b + batch_size][mask] = -lp[mask]
    return ce


def summarize(ce: np.ndarray, targets: np.ndarray) -> dict:
    """Mean CE / PPL over non-pad targets, <unk>-target fraction, and per-position mean CE."""
    mask = targets != PAD
    vals = ce[mask]
    assert not np.isnan(vals).any(), "NaN cross-entropy at a non-pad target"
    mean_ce = float(vals.mean())
    per_pos = np.where(mask.sum(0) > 0, np.nansum(np.where(mask, ce, 0.0), axis=0) / np.maximum(mask.sum(0), 1), np.nan)
    return {
        "ce": mean_ce,
        "ppl": float(math.exp(mean_ce)),
        "n_targets": int(mask.sum()),
        "n_blocks": int(targets.shape[0]),
        "unk_target_fraction": float((targets[mask] == UNK).mean()),
        "eos_target_fraction": float((targets[mask] == EOS).mean()),
        "ce_by_position": per_pos.tolist(),
    }


def evaluate_stream(logprob_fn: LogProbFn, stream: np.ndarray, L: int = 256, batch_size: int = 64,
                    return_ce: bool = False):
    inputs, targets = make_blocks(stream, L)
    ce = score_blocks(logprob_fn, inputs, targets, batch_size)
    summary = summarize(ce, targets)
    return (summary, ce, targets) if return_ce else summary


# --- frequency buckets (Part II analysis; defined here so every model uses the same buckets)
BUCKETS = [("1-100", 1, 100), ("101-1k", 101, 1000), ("1k-5k", 1001, 5000), ("5k-25k", 5001, 10**9)]


def frequency_ranks(vocab_counts: np.ndarray) -> np.ndarray:
    """Rank (1 = most frequent) of every vocabulary id by LM-train count; <pad> and <unk> get rank 0."""
    counts = np.asarray(vocab_counts, dtype=np.int64).copy()
    ids = np.arange(len(counts))
    eligible = ids >= 2  # <eos> and words are ranked; <pad>, <unk> excluded
    order = np.lexsort((ids[eligible], -counts[eligible]))
    ranks = np.zeros(len(counts), dtype=np.int64)
    ranks[ids[eligible][order]] = np.arange(1, eligible.sum() + 1)
    return ranks


def ce_by_frequency_bucket(ce: np.ndarray, targets: np.ndarray, vocab_counts: np.ndarray) -> dict:
    ranks = frequency_ranks(vocab_counts)
    mask = targets != PAD
    t, c = targets[mask], ce[mask]
    r = ranks[t]
    out = {}
    for name, lo, hi in BUCKETS:
        sel = (r >= lo) & (r <= hi)
        out[name] = {"ce": float(c[sel].mean()) if sel.any() else None, "n_targets": int(sel.sum())}
    sel = t == UNK
    out["<unk>"] = {"ce": float(c[sel].mean()) if sel.any() else None, "n_targets": int(sel.sum())}
    return out


def uniform_logprob_fn(vocab_size: int) -> LogProbFn:
    """Harness check: log P = -ln|V| for every target, so CE must equal ln|V| and PPL must equal |V|."""
    def fn(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.full(inputs.shape, -math.log(vocab_size), dtype=np.float64)
    return fn


def ce_by_context_frequency_bucket(ce: np.ndarray, inputs: np.ndarray, targets: np.ndarray, vocab_counts: np.ndarray) -> dict:
    """Same buckets, but keyed by the LM-train frequency rank of the immediately preceding input token
    (`inputs[:, j]` for the target at position j); the `<unk>` bucket = targets whose preceding token is <unk>."""
    ranks = frequency_ranks(vocab_counts)
    mask = targets != PAD
    ctx, c = inputs[mask], ce[mask]
    r = ranks[ctx]
    out = {}
    for name, lo, hi in BUCKETS:
        sel = (r >= lo) & (r <= hi)
        out[name] = {"ce": float(c[sel].mean()) if sel.any() else None, "n_targets": int(sel.sum())}
    sel = ctx == UNK
    out["<unk>"] = {"ce": float(c[sel].mean()) if sel.any() else None, "n_targets": int(sel.sum())}
    return out
