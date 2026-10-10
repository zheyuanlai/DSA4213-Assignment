"""TASK §3 test (b): KN probabilities sum to 1 over the vocabulary for 20 random histories of each length."""
import numpy as np
import pytest

from src.common import ROOT
from src.ngram import KneserNeyLM
from tests.conftest import SMALL_V


def _check(lm, stream, V, n_hist=20, tol=1e-6):
    rng = np.random.default_rng(1)
    for length in range(0, lm.order):  # 0 tokens (unigram) .. order-1 tokens
        for i in range(n_hist):
            if length == 0:
                hist = []
            elif i % 2 == 0:  # observed history taken from the stream
                start = rng.integers(0, len(stream) - length)
                hist = [int(t) for t in stream[start:start + length]]
            else:              # random, mostly unseen history
                hist = [int(t) for t in rng.integers(0, V, size=length)]
            P = lm.next_distribution(hist)
            assert P.shape == (V,)
            assert P.min() >= 0.0, (length, hist, P.min())
            assert abs(P.sum() - 1.0) < tol, (length, hist, P.sum())
    return True


def test_normalisation_synthetic(small_kn, zipf_stream):
    assert _check(small_kn, zipf_stream, SMALL_V)


def test_dense_matches_blockwise_scoring(small_kn, zipf_stream):
    """The dense distribution and the vectorised block scorer must agree on P(target | history)."""
    rng = np.random.default_rng(2)
    L = 16
    start = int(rng.integers(0, len(zipf_stream) - L - 1))
    inputs = zipf_stream[start:start + L][None].astype(np.int64)
    targets = zipf_stream[start + 1:start + L + 1][None].astype(np.int64)
    lp = small_kn.logprob_blocks(inputs, targets)[0]
    for j in range(L):
        hist = [int(t) for t in inputs[0, max(0, j - small_kn.order + 2):j + 1]]
        P = small_kn.next_distribution(hist)
        assert abs(np.log(P[targets[0, j]]) - lp[j]) < 1e-9, (j, hist)


@pytest.mark.skipif(not (ROOT / "checkpoints/ngram/kn_selected.npz").exists(), reason="selected real model not built yet")
def test_normalisation_real_model():
    lm = KneserNeyLM.load("checkpoints/ngram/kn_selected.npz")
    stream = np.load(ROOT / "data/streams/lm_val.npy")
    assert _check(lm, stream, lm.V)
