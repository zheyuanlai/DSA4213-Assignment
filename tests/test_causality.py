"""TASK §3 test (a): changing tokens after position j leaves logits at positions <= j unchanged (tol 1e-5).

Runs on CPU in fp32 (TF32 is CUDA-only), eval mode, with the real architectures and vocabulary size.
"""
import numpy as np
import pytest
import torch

from src.models import build_model

V = 25003
CONFIGS = {
    "rnn": {"type": "rnn", "hidden": 1072, "dropout": 0.3},
    "lstm": {"type": "lstm", "hidden": 512, "dropout": 0.3},
    "transformer": {"type": "transformer", "dropout": 0.1},
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_causality(name):
    torch.manual_seed(0)
    model = build_model(CONFIGS[name], V).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(3, V, (2, 256), generator=g)
    with torch.no_grad():
        ref = model(x)
    for j in (0, 1, 7, 100, 254):
        x2 = x.clone()
        x2[:, j + 1:] = torch.randint(3, V, (2, 256 - j - 1), generator=g)
        assert not torch.equal(x2, x)
        with torch.no_grad():
            out = model(x2)
        diff = (out[:, :j + 1] - ref[:, :j + 1]).abs().max().item()
        assert diff <= 1e-5, f"{name}: logits at positions <= {j} changed by {diff}"
        # and positions after j do change (sanity that the test is not vacuous)
        assert (out[:, j + 1:] - ref[:, j + 1:]).abs().max().item() > 1e-3
