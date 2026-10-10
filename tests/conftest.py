"""Shared fixtures: a small synthetic Zipf stream and KN tables (self-contained, no data needed)."""
import numpy as np
import pytest

from src.ngram import KneserNeyLM, build_tables

SMALL_V = 60
PAD_ID, UNK_ID, EOS_ID = 0, 1, 2


@pytest.fixture(scope="session")
def zipf_stream():
    rng = np.random.default_rng(0)
    ranks = np.arange(1, SMALL_V - 2)  # token ids 3..V-1 (never emit pad/unk here)
    p = 1.0 / ranks
    p /= p.sum()
    toks = rng.choice(np.arange(3, SMALL_V), size=20000, p=p)
    # sprinkle <unk> and <eos> so they are observed like in the real streams
    toks[rng.choice(len(toks), 300, replace=False)] = UNK_ID
    toks[rng.choice(len(toks), 400, replace=False)] = EOS_ID
    return np.concatenate([[EOS_ID], toks]).astype(np.int32)


@pytest.fixture(scope="session")
def zipf_tables(zipf_stream):
    return build_tables(zipf_stream, SMALL_V, 5)


@pytest.fixture(scope="session", params=[
    ("single", 0.75, 4), ("single", 0.5, 3), ("modified", None, 4), ("modified", None, 5), ("modified", None, 2), ("modified", None, 1)])
def small_kn(request, zipf_tables):
    typ, D, order = request.param
    spec = {"type": "single", "D": D} if typ == "single" else {"type": "modified", "fallback": [0.5, 1.0, 1.5]}
    return KneserNeyLM(zipf_tables, order, spec)
