"""TASK §3 test (d): each neural model can overfit a single batch.

Fixture (NOTES D25): dropout 0, one fixed seeded batch of 4 blocks x 64 tokens from LM-train (synthetic
if the stream is absent), Adam lr 1e-3, no weight decay / clipping / schedule, up to 2,000 full-batch
steps; pass when the loss drops below 0.5 nats. The step-1 loss is checked to be near ln|V| = 10.13.
Results (step-1 loss, steps to < 0.5) are written to results/part1/test_overfit.json for the report.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.common import ROOT, write_json
from src.models import build_model, set_dropout

V = 25003
CONFIGS = {
    "rnn": {"type": "rnn", "hidden": 1072, "dropout": 0.3},
    "lstm": {"type": "lstm", "hidden": 512, "dropout": 0.3},
    "transformer": {"type": "transformer", "dropout": 0.1},
}
RESULTS = ROOT / "results/part1/test_overfit.json"


def _batch(n_blocks=4, T=64):
    path = ROOT / "data/streams/lm_train.npy"
    rng = np.random.default_rng(0)
    if path.exists():
        s = np.load(path)
        starts = rng.integers(0, len(s) - T - 1, size=n_blocks)
        x = np.stack([s[a:a + T] for a in starts]); y = np.stack([s[a + 1:a + T + 1] for a in starts])
        source = "lm_train"
    else:
        x = rng.integers(3, V, size=(n_blocks, T)); y = np.concatenate([x[:, 1:], rng.integers(3, V, size=(n_blocks, 1))], 1)
        source = "synthetic"
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long), source


@pytest.mark.parametrize("name", list(CONFIGS))
def test_overfit_single_batch(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_model(CONFIGS[name], V)
    set_dropout(model, 0.0)
    model = model.to(device).train()
    x, y, source = _batch()
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    first_loss, steps_to_threshold, final = None, None, None
    for step in range(1, 2001):
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        if step == 1:
            first_loss = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = loss.item()
        if final < 0.5:
            steps_to_threshold = step
            break
    rec = {"model": name, "device": str(device), "batch_source": source, "step1_loss": first_loss,
           "ln_vocab": math.log(V), "steps_to_below_0.5": steps_to_threshold, "final_loss": final, "passed": steps_to_threshold is not None}
    existing = json.load(open(RESULTS)) if RESULTS.exists() else {}
    existing[name] = rec
    write_json(existing, RESULTS)
    assert abs(first_loss - math.log(V)) < 0.5, f"{name}: step-1 loss {first_loss:.3f} far from ln|V| {math.log(V):.3f}"
    assert steps_to_threshold is not None, f"{name}: loss {final:.3f} after 2000 steps, did not reach < 0.5"
