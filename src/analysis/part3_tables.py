"""Part III figure: seed-0 LM-test perplexity vs IMDb-test accuracy (one point per LM and pooling), with 95 % CI
error bars, point labels and dashed baseline lines (tfidf_uni_bi, w2v_mean). Values dumped to CSV."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis import plots
from src.common import ROOT

LM_KEYS = {"rnn": "rnn", "lstm": "lstm", "transformer": "transformer", "lstm_own": "lstm_own", "lstm_glove": "lstm_glove"}
SHORT = {"rnn": "RNN", "lstm": "LSTM-A", "transformer": "Transformer", "lstm_own": "LSTM-B", "lstm_glove": "LSTM-C"}
POOL_MARKER = {"mean": "o", "last": "s"}
# label offsets in points and horizontal alignment, chosen so that no two labels overlap in the crowded 75–78 PPL region
OFFSETS = {("transformer", "mean"): (-5, -10, "right"), ("transformer", "last"): (-5, -10, "right"),
           ("lstm", "mean"): (-5, -10, "right"), ("lstm", "last"): (5, -9, "left"),
           ("lstm_own", "mean"): (-5, 5, "right"), ("lstm_own", "last"): (-5, -3, "right"),
           ("lstm_glove", "mean"): (6, 3, "left"), ("lstm_glove", "last"): (6, -8, "left"),
           ("rnn", "mean"): (5, 3, "left"), ("rnn", "last"): (5, -9, "left")}


def fig_ppl_vs_accuracy(stem: str = "part3_ppl_vs_accuracy") -> list[str]:
    res = pd.read_csv(ROOT / "results/part3/results_table.csv").set_index("feature_set")
    lmd = pd.read_csv(ROOT / "results/part3/lm_vs_downstream.csv").set_index("lm")
    rows = []
    fig, ax = plots.new_figure(height=2.6)
    for lm, key in LM_KEYS.items():
        for pool, mk in POOL_MARKER.items():
            fs = f"{lm}_{pool}"
            if fs not in res.index:
                continue
            r = res.loc[fs]
            lo, hi = json.loads(str(r.acc_ci95).replace("'", '"'))
            x, y = float(lmd.loc[lm, "lm_test_ppl_seed0"]), float(r.test_acc)
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=plots.COLORS[key], ms=4.5, capsize=2, elinewidth=0.8,
                        markeredgecolor="black", markeredgewidth=0.4)
            dx, dy, ha = OFFSETS.get((lm, pool), ((4, 3, "left") if pool == "mean" else (4, -9, "left")))
            ax.annotate(f"{SHORT[lm]} {pool}", (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=5.5, color=plots.COLORS[key])
            rows.append({"lm": lm, "pooling": pool, "feature_set": fs, "lm_test_ppl_seed0": x, "test_acc": y, "acc_ci95_lo": lo, "acc_ci95_hi": hi})
    for base, ls in (("tfidf_uni_bi", "--"), ("w2v_mean", (0, (3, 1, 1, 1)))):
        if base in res.index:
            yb = float(res.loc[base, "test_acc"])
            ax.axhline(yb, color="#555555", ls=ls, lw=0.8, label=f"{base} ({yb:.3f})")
            rows.append({"lm": base, "pooling": "baseline", "feature_set": base, "lm_test_ppl_seed0": None, "test_acc": yb, "acc_ci95_lo": None, "acc_ci95_hi": None})
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=mk, color="black", ls="", ms=4.5, label=f"{pool} pooling") for pool, mk in POOL_MARKER.items()]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles + h, [x.get_label() for x in handles] + l, fontsize=6, loc="lower left")
    ax.set_xlabel("LM-test perplexity (lower is better)"); ax.set_ylabel("IMDb-test accuracy")
    ax.set_title("Part III: LM perplexity vs downstream accuracy (95 % CI)")
    ax.set_xlim(58, 106)
    plots.dump(pd.DataFrame(rows), "results/part3/ppl_vs_accuracy.csv")
    return plots.save(fig, stem)


if __name__ == "__main__":
    print("figures:", *fig_ppl_vs_accuracy(), sep="\n  ")
