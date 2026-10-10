"""Part III representation checks (added after an external code review, NOTES D123).

- build_inputs: <eos> + first / last 255 review tokens, right-padding, truncation flags and lengths.
- pooling: `last` takes the vector at the last non-pad position and `mean` averages only the non-pad positions
  (checked exactly with a stub model whose features encode token id and position).
- padding / batching invariance: with the real architectures (CPU fp32, eval mode), the features of a review do not
  depend on how much right-padding it receives or on which other reviews share its batch (tolerance 1e-5).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.common import EOS, PAD
from src.features import build_inputs, extract
from src.models import build_model

V = 25003
NEURAL = {
    "rnn": {"type": "rnn", "hidden": 1072, "dropout": 0.3},
    "lstm": {"type": "lstm", "hidden": 512, "dropout": 0.3},
    "transformer": {"type": "transformer", "dropout": 0.1},
}


def test_build_inputs_first_and_last():
    reviews = [np.array([10, 11, 12, 13, 14]), np.array([20, 21])]
    ids = np.concatenate(reviews).astype(np.int32)
    offsets = np.array([0, 5, 7])
    first = build_inputs(ids, offsets, max_tokens=3, truncate="first", sel=np.arange(2))
    assert first["ids"].tolist() == [[EOS, 10, 11, 12], [EOS, 20, 21, PAD]]
    assert first["n_input"].tolist() == [4, 3] and first["full_len"].tolist() == [5, 2]
    assert first["truncated"].tolist() == [True, False]
    last = build_inputs(ids, offsets, max_tokens=3, truncate="last", sel=np.arange(2))
    assert last["ids"].tolist() == [[EOS, 12, 13, 14], [EOS, 20, 21, PAD]]
    assert last["n_input"].tolist() == [4, 3]


class StubLM(nn.Module):
    """features(x)[b, t] = (token id, position, 1) — makes the pooled vectors exactly predictable."""
    d_model = 3

    def features(self, x):
        pos = torch.arange(x.shape[1]).expand_as(x)
        return torch.stack([x.float(), pos.float(), torch.ones_like(x, dtype=torch.float)], dim=-1)


def test_last_and_mean_pooling_use_only_valid_positions():
    X = np.array([[EOS, 5, 6, 7, PAD, PAD], [EOS, 9, PAD, PAD, PAD, PAD], [EOS, 3, 4, 8, 1, 6]])
    n = np.array([4, 2, 6])
    out = extract(StubLM(), X, n, torch.device("cpu"), bs=2)               # bs=2 also exercises batch boundaries
    assert out["last"].tolist() == [[7, 3, 1], [9, 1, 1], [6, 5, 1]]
    expected_mean = [[np.mean([EOS, 5, 6, 7]), 1.5, 1], [np.mean([EOS, 9]), 0.5, 1], [np.mean([EOS, 3, 4, 8, 1, 6]), 2.5, 1]]
    assert np.allclose(out["mean"], expected_mean)


@pytest.mark.parametrize("name", list(NEURAL))
def test_features_invariant_to_padding_and_batch(name):
    torch.manual_seed(0)
    model = build_model(NEURAL[name], V).eval()
    rng = np.random.default_rng(5)
    lengths = [256, 37, 120]
    X = np.full((3, 256), PAD, dtype=np.int64)
    for i, L in enumerate(lengths):
        X[i, 0] = EOS
        X[i, 1:L] = rng.integers(3, V, size=L - 1)
    batched = extract(model, X, np.array(lengths), torch.device("cpu"), bs=3)
    for i, L in enumerate(lengths):
        alone = extract(model, X[i:i + 1, :L], np.array([L]), torch.device("cpu"), bs=1)  # no padding at all
        for pool in ("last", "mean"):
            diff = np.abs(batched[pool][i] - alone[pool][0]).max()
            assert diff <= 1e-5, (name, pool, L, diff)
