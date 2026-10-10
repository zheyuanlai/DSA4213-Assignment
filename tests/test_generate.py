"""Generation checks (added after an external code review, NOTES D123).

- Incremental decoding: the log-probabilities produced step by step by NeuralGenerator (RNN/LSTM carry their
  recurrent state; the Transformer re-encodes the prefix) equal a full-prefix forward pass at every step
  (CPU fp32, eval mode, tolerance 1e-5).
- Decoding rules: <pad> and <unk> can never be emitted; greedy = argmax; sampling stays inside the top-k.
"""
import numpy as np
import pytest
import torch

from src.common import EOS, PAD, UNK
from src.generate import TOP_K, NeuralGenerator, choose
from src.models import build_model

V = 25003
NEURAL = {
    "rnn": {"type": "rnn", "hidden": 1072, "dropout": 0.3},
    "lstm": {"type": "lstm", "hidden": 512, "dropout": 0.3},
    "transformer": {"type": "transformer", "dropout": 0.1},
}


@pytest.mark.parametrize("name", list(NEURAL))
def test_incremental_decoding_matches_full_prefix(name):
    torch.manual_seed(0)
    model = build_model(NEURAL[name], V).eval()
    rng = np.random.default_rng(7)
    prefix = [EOS] + rng.integers(3, V, size=4).tolist()
    continuation = rng.integers(3, V, size=10).tolist()
    g = NeuralGenerator(model, torch.device("cpu"))

    def full(seq):
        with torch.no_grad():
            return torch.log_softmax(model(torch.tensor([seq]))[0, -1].float(), dim=-1)

    lp = g.start(prefix)
    assert torch.allclose(lp, full(prefix), atol=1e-5)
    seq = list(prefix)
    for t in continuation:
        lp = g.step(t)
        seq.append(t)
        assert torch.allclose(lp, full(seq), atol=1e-5), (name, len(seq), (lp - full(seq)).abs().max().item())


def test_choose_rules():
    logp = torch.full((V,), -20.0)
    logp[PAD], logp[UNK] = 5.0, 4.0                       # the two best tokens are the suppressed specials
    logp[100] = 3.0
    logp[200:200 + TOP_K + 20] = torch.linspace(2.0, -2.0, TOP_K + 20)
    assert choose(logp, "greedy", None) == 100
    topk = set(torch.topk(torch.where(torch.arange(V) < 2, torch.tensor(-1e9), logp), TOP_K).indices.tolist())
    gen = torch.Generator().manual_seed(0)
    draws = {choose(logp, "sample", gen) for _ in range(300)}
    assert PAD not in draws and UNK not in draws and draws <= topk
