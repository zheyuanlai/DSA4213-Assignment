"""Part III figure: LM-test perplexity of the checkpoint the features come from vs IMDb-test accuracy (one point per
LM and pooling; circles = mean, squares = last) with 95 % CI error bars, point labels and dashed baseline lines.
Broken y-axis: the upper panel zooms on the 0.83-0.89 cluster, the lower panel holds the low-accuracy points.
Label positions are fixed in data coordinates (with leader lines in the crowded cluster) so that no label overlaps
another label, a marker or a dashed line. Plotted values are dumped to CSV."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis import plots
from src.common import ROOT

LMS = ["rnn", "lstm", "transformer", "lstm_own", "lstm_glove"]
SHORT = {"rnn": "RNN", "lstm": "LSTM-A", "transformer": "Transformer", "lstm_own": "LSTM-B", "lstm_glove": "LSTM-C"}
POOL_MARKER = {"mean": "o", "last": "s"}
BASELINES = {"tfidf_uni_bi": ("TF-IDF uni+bigram", "--"), "w2v_mean": ("word2vec mean", (0, (3, 1, 1, 1)))}
TOP_YLIM, BOT_YLIM, XLIM = (0.830, 0.890), (0.636, 0.815), (61.0, 103.0)
# label anchor (x, y in data units), horizontal alignment, leader line from the point? — upper panel cluster labels sit
# in the empty area right of the LSTM points, ordered like the points and kept between / away from the dashed lines
LABELS = {
    ("transformer", "mean"): (65.6, 0.8745, "left", False),
    ("transformer", "last"): (65.8, 0.8394, "left", False),
    ("lstm_own", "mean"): (81.0, 0.8775, "left", True),
    ("lstm_glove", "mean"): (81.0, 0.8720, "left", True),
    ("lstm_own", "last"): (75.6, 0.8669, "right", True),
    ("lstm_glove", "last"): (81.0, 0.8628, "left", True),
    ("lstm", "mean"): (81.0, 0.8479, "left", True),
    ("lstm", "last"): (77.2, 0.7942, "left", False),
    ("rnn", "mean"): (97.6, 0.7997, "right", False),
    ("rnn", "last"): (97.6, 0.6510, "right", False),
}


def fig_ppl_vs_accuracy(stem: str = "part3_ppl_vs_accuracy") -> list[str]:
    res = pd.read_csv(ROOT / "results/part3/results_table.csv").set_index("feature_set")
    lmd = pd.read_csv(ROOT / "results/part3/lm_vs_downstream.csv").set_index("lm")
    fig, (top, bot) = plots.new_figure(height=3.1, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.7, 1.0], "hspace": 0.07})
    rows = []
    for lm in LMS:
        for pool, mk in POOL_MARKER.items():
            fs = f"{lm}_{pool}"
            r = res.loc[fs]
            lo, hi = json.loads(str(r.acc_ci95).replace("'", '"'))
            x, y = float(lmd.loc[lm, "lm_test_ppl_seed0"]), float(r.test_acc)
            for ax in (top, bot):
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=plots.COLORS[lm], ms=3.6, capsize=1.6,
                            elinewidth=0.7, markeredgecolor="black", markeredgewidth=0.35, zorder=3)
            lx, ly, ha, leader = LABELS[(lm, pool)]
            ax = top if TOP_YLIM[0] <= y <= TOP_YLIM[1] else bot
            ax.annotate(f"{SHORT[lm]} {pool}", xy=(x, y), xytext=(lx, ly), textcoords="data", ha=ha, va="center",
                        fontsize=5.5, color=plots.COLORS[lm], zorder=4,
                        arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.5, "shrinkA": 0, "shrinkB": 2.5} if leader else None)
            rows.append({"lm": lm, "pooling": pool, "feature_set": fs, "label": f"{SHORT[lm]} {pool}",
                         "lm_test_ppl": x, "test_acc": y, "acc_ci95_lo": lo, "acc_ci95_hi": hi, "panel": "upper" if ax is top else "lower"})
    handles = [Line2D([], [], marker=mk, color="black", ls="", ms=3.6, label=f"{pool} pooling") for pool, mk in POOL_MARKER.items()]
    for fs, (name, ls) in BASELINES.items():
        yb = float(res.loc[fs, "test_acc"])
        top.axhline(yb, color="#555555", ls=ls, lw=0.8, zorder=1)
        handles.append(Line2D([], [], color="#555555", ls=ls, lw=0.8, label=f"{name} ({yb:.3f})"))
        rows.append({"lm": fs, "pooling": "baseline", "feature_set": fs, "label": name, "lm_test_ppl": None, "test_acc": yb,
                     "acc_ci95_lo": None, "acc_ci95_hi": None, "panel": "upper"})
    top.set_ylim(*TOP_YLIM); bot.set_ylim(*BOT_YLIM); bot.set_xlim(*XLIM)
    top.spines["bottom"].set_visible(False); top.tick_params(bottom=False)
    bot.set_yticks([0.65, 0.70, 0.75, 0.80])
    for ax, yy in ((top, 0.0), (bot, 1.0)):                                    # broken-axis marks on the left spine
        ax.plot([0], [yy], transform=ax.transAxes, marker=[(-1, -0.5), (1, 0.5)], ms=6, ls="none", color="black",
                mec="black", mew=0.8, clip_on=False)
    bot.set_xlabel("LM-test perplexity (lower is better)")
    fig.supylabel("IMDb-test accuracy", fontsize=8, x=0.0)
    bot.legend(handles=handles, loc="lower left", fontsize=5.5, ncol=1, handlelength=1.8, borderpad=0.4, labelspacing=0.3)
    plots.dump(pd.DataFrame(rows), "results/part3/ppl_vs_accuracy.csv")
    return plots.save(fig, stem)


if __name__ == "__main__":
    print("figures:", *fig_ppl_vs_accuracy(), sep="\n  ")
