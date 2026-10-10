"""TASK §3 test (c): adding extra pad positions does not change the loss.

A sequence of T < 256 tokens is scored once as a single unpadded block of length T and once right-
padded to 256; the masked mean CE must be identical. Neural models are added to MODELS in Stage 3.
"""
import numpy as np
import pytest

from src.evaluate import make_blocks, score_blocks, summarize
from src.ngram import KneserNeyLM


def _ce_two_ways(logprob_fn, seq, L=256):
    T = len(seq)
    x1, y1 = make_blocks(seq, L=T - 1)             # exactly T-1 targets, no pad
    assert (y1 != 0).all() and x1.shape == (1, T - 1)
    x2, y2 = make_blocks(seq, L=L)                 # one block, right-padded to L
    assert x2.shape == (1, L) and (y2 == 0).sum() == L - (T - 1)
    s1 = summarize(score_blocks(logprob_fn, x1, y1), y1)
    s2 = summarize(score_blocks(logprob_fn, x2, y2), y2)
    return s1, s2


def test_padding_ngram(small_kn, zipf_stream):
    seq = zipf_stream[100:100 + 97]
    s1, s2 = _ce_two_ways(small_kn.logprob_blocks, seq)
    assert s1["n_targets"] == s2["n_targets"] == 96
    assert abs(s1["ce"] - s2["ce"]) < 1e-12


def test_make_blocks_layout():
    s = np.arange(3, 3 + 10)
    x, y = make_blocks(s, L=4)
    assert x.shape == (3, 4)
    assert x.tolist() == [[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 0, 0]]
    assert y.tolist() == [[4, 5, 6, 7], [8, 9, 10, 11], [12, 0, 0, 0]]
    s = np.arange(3, 3 + 9)  # (n-1) % L == 0 → no padding
    x, y = make_blocks(s, L=4)
    assert x.shape == (2, 4) and (y != 0).all()
