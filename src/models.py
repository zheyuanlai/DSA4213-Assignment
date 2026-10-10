"""Neural language models (TASK §4.2, with the user's Q2 change).

Shared structure: Embedding(|V|, 300) → core → 300-d pre-head vector → Linear(300, |V|); no weight
tying. RNN/LSTM: dropout on the embedding output, between the two recurrent layers and on the
top-layer output, then Linear(H, 300) (bias, no nonlinearity) → head. Transformer: GPT-style
decoder (pre-LN, learned absolute positions, causal SDPA, final LayerNorm) whose final-LN output is
the pre-head vector. `features(x)` returns that 300-d vector for Part III.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import PAD


class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, kind: str = "lstm", d_model: int = 300, hidden: int = 512,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert kind in ("rnn", "lstm")
        self.kind, self.vocab_size, self.d_model, self.hidden = kind, vocab_size, d_model, hidden
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=None)
        self.drop = nn.Dropout(dropout)
        if kind == "lstm":
            self.core = nn.LSTM(d_model, hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        else:
            self.core = nn.RNN(d_model, hidden, num_layers=n_layers, nonlinearity="tanh", dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)  # PyTorch word-LM convention; core/proj/head keep defaults

    def features(self, x: torch.Tensor, state=None, return_state: bool = False):
        h, state = self.core(self.drop(self.emb(x)), state)   # state=None → zero initial state (stateless per block)
        z = self.proj(self.drop(h))
        return (z, state) if return_state else z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def step(self, x_t: torch.Tensor, state):
        """Incremental decoding: x_t [B, 1] → logits [B, 1, V], new state (equivalent to re-reading the prefix)."""
        z, state = self.features(x_t, state, return_state=True)
        return self.head(z), state


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads, self.p = n_heads, dropout
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 300, n_layers: int = 4, n_heads: int = 6,
                 ffn: int = 1200, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.kind, self.vocab_size, self.d_model, self.max_len = "transformer", vocab_size, d_model, max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.core = nn.ModuleDict({
            "pos": nn.Embedding(max_len, d_model),
            "blocks": nn.ModuleList([Block(d_model, n_heads, ffn, dropout) for _ in range(n_layers)]),
            "ln_f": nn.LayerNorm(d_model),
        })
        self.proj = None
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init)
        for blk in self.core["blocks"]:  # GPT-2: scale residual projections by 1/sqrt(2 * n_layers)
            for lin in (blk.proj, blk.fc2):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.emb(x) + self.core["pos"](pos))
        for blk in self.core["blocks"]:
            h = blk(h)
        return self.core["ln_f"](h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(mcfg: dict, vocab_size: int) -> nn.Module:
    t = mcfg["type"]
    if t in ("rnn", "lstm"):
        return RNNLM(vocab_size, kind=t, d_model=mcfg.get("d_model", 300), hidden=mcfg["hidden"],
                     n_layers=mcfg.get("n_layers", 2), dropout=mcfg.get("dropout", 0.3))
    if t == "transformer":
        return TransformerLM(vocab_size, d_model=mcfg.get("d_model", 300), n_layers=mcfg.get("n_layers", 4),
                             n_heads=mcfg.get("n_heads", 6), ffn=mcfg.get("ffn", 1200),
                             dropout=mcfg.get("dropout", 0.1), max_len=mcfg.get("max_len", 256))
    raise ValueError(f"unknown model type {t}")


def _count(mod, trainable_only=False) -> int:
    if mod is None:
        return 0
    return sum(p.numel() for p in mod.parameters() if (p.requires_grad or not trainable_only))


def param_breakdown(model: nn.Module) -> dict:
    emb, core, proj, head = (_count(getattr(model, n, None)) for n in ("emb", "core", "proj", "head"))
    total = _count(model)
    assert total == emb + core + proj + head, "parameter groups do not partition the model"
    return {"embedding": emb, "core": core, "projection": proj, "head": head, "total": total,
            "trainable": _count(model, trainable_only=True)}


def set_dropout(model: nn.Module, p: float) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
        if isinstance(m, (nn.RNN, nn.LSTM)):
            m.dropout = p
        if isinstance(m, Block):
            m.p = p


def load_embedding_table(model: nn.Module, table: np.ndarray, freeze: bool) -> None:
    """Copy a [|V|, d] table into the input embedding; freeze the entire table if requested."""
    t = torch.as_tensor(np.asarray(table, dtype=np.float32))
    assert t.shape == model.emb.weight.shape, (t.shape, model.emb.weight.shape)
    with torch.no_grad():
        model.emb.weight.copy_(t)
    model.emb.weight.requires_grad_(not freeze)


def logprob_fn_for(model: nn.Module, device: torch.device):
    """Adapter to the common evaluation protocol: log P(target_j | inputs[0..j]) per position (eval mode)."""
    @torch.no_grad()
    def fn(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(inputs, dtype=np.int64), device=device)
        y = torch.as_tensor(np.asarray(targets, dtype=np.int64), device=device).clamp(min=0)
        logits = model(x)
        lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        return lp.double().cpu().numpy()
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="print the parameter breakdown of the three Part I models")
    ap.add_argument("--vocab-size", type=int, default=25003)
    ap.add_argument("--rnn-hidden", type=int, default=1072)
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()
    rows = []
    for name, mcfg in [("rnn", {"type": "rnn", "hidden": args.rnn_hidden, "dropout": 0.3}),
                       ("lstm", {"type": "lstm", "hidden": 512, "dropout": 0.3}),
                       ("transformer", {"type": "transformer", "dropout": 0.1})]:
        bd = param_breakdown(build_model(mcfg, args.vocab_size))
        rows.append({"model": name, **bd})
    print(f"{'model':12s}{'embedding':>12s}{'core':>12s}{'projection':>12s}{'head':>12s}{'total':>12s}{'trainable':>12s}")
    for r in rows:
        print(f"{r['model']:12s}{r['embedding']:12,d}{r['core']:12,d}{r['projection']:12,d}{r['head']:12,d}{r['total']:12,d}{r['trainable']:12,d}")
    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
