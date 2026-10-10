"""Harness check requested at the Stage-1 review: a uniform model must give CE = ln|V| and PPL = |V|."""
import math

import numpy as np

from src.evaluate import evaluate_stream, uniform_logprob_fn


def test_uniform_model_ce_and_ppl():
    V = 25003
    rng = np.random.default_rng(0)
    stream = np.concatenate([[2], rng.integers(3, V, size=5000)])  # arbitrary content, with a partial last block
    s = evaluate_stream(uniform_logprob_fn(V), stream, L=256)
    assert s["n_targets"] == 5000
    assert math.isclose(s["ce"], math.log(V), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(s["ppl"], V, rel_tol=1e-12)
