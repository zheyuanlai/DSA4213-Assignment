"""Shared figure style: one fixed colour-blind-safe colour (Okabe–Ito), line style, marker and bar hatch per
model across every figure (A-random is the same model as the Part I LSTM and shares its colour); half-page
width; PDF + PNG; every plotted series is also dumped to CSV next to the tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import ROOT  # noqa: E402

HALF_PAGE_IN = 3.3

# key -> (colour, line style, marker, hatch)
STYLE = {
    "ngram":       ("#000000", "--",              "s", ""),
    "rnn":         ("#D55E00", "-",               "o", ""),
    "lstm":        ("#0072B2", "-.",              "^", ""),
    "transformer": ("#009E73", ":",               "D", ""),
    "A-random":    ("#0072B2", "-.",              "^", ""),       # = Part I LSTM (same checkpoints)
    "B-own":       ("#E69F00", (0, (5, 1)),       "v", "//"),
    "C-glove":     ("#CC79A7", (0, (3, 1, 1, 1)), "P", ".."),
    "C-scaled":    ("#56B4E9", (0, (1, 1)),       "X", "xx"),
    "lstm_own":    ("#E69F00", (0, (5, 1)),       "v", "//"),     # Part III names for B / C
    "lstm_glove":  ("#CC79A7", (0, (3, 1, 1, 1)), "P", ".."),
}
COLORS = {k: v[0] for k, v in STYLE.items()}
LABELS = {"ngram": "KN 4-gram", "rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer",
          "A-random": "A: random-trainable (LSTM)", "B-own": "B: own word2vec, frozen", "C-glove": "C: GloVe, frozen",
          "C-scaled": "C-scaled: GloVe rescaled, frozen", "lstm_own": "LSTM-B (own word2vec)", "lstm_glove": "LSTM-C (GloVe)"}


def style(key: str) -> dict:
    c, ls, mk, _ = STYLE[key]
    return {"color": c, "ls": ls, "marker": mk}


def hatch(key: str) -> str:
    return STYLE[key][3]


def setup() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "lines.markersize": 3,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "legend.framealpha": 0.85,
        "figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 0.5,
    })


def new_figure(width: float = HALF_PAGE_IN, height: float = 2.4, **kw):
    setup()
    return plt.subplots(figsize=(width, height), **kw)


def save(fig, stem: str, subdir: str = "figures") -> list[str]:
    out_dir = ROOT / "results" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def dump(df: pd.DataFrame, rel_path: str) -> str:
    """Write the plotted values of a figure to CSV (traceability, TASK §8)."""
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return rel_path
