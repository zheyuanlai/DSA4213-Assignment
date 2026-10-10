"""Part I tables and figures (TASK §4.7, user additions D57/D67).

`probes`  -> results/part1/lr_probes.csv, lr_selected.json (Stage 3)
`main`    -> results/part1/run_summary.csv (per run: status, early stopping, best epoch, flags,
             probe consistency, train-val gap), main_table.csv, loss_by_position.csv and the figures:
             part1_curves_{model}, part1_val_curves, part1_loss_by_position, part1_ngram_order.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ROOT, read_json, read_jsonl, write_json
from src.analysis import plots

MODELS = ["rnn", "lstm", "transformer"]
PROBE_LRS = [3e-4, 1e-3, 2e-3]
SEEDS = [0, 1, 2]
TIE_TOL = 0.01  # nats (user rule D58)
MAX_EPOCHS = 12


# ----------------------------------------------------------------------------- Stage 3: probes
def probe_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for lr in PROBE_LRS:
            run = f"probes/{m}_lr{lr:g}"
            path = ROOT / "results/runs" / run / "log.jsonl"
            if not path.exists():
                rows.append({"model": m, "lr": lr, "status": "missing"}); continue
            recs = read_jsonl(path)
            summ = next((r for r in recs if r["record"] == "summary"), None)
            ep = [r for r in recs if r["record"] == "epoch"]
            fail = next((r for r in recs if r["record"] == "failure"), None)
            rows.append({"model": m, "lr": lr, "status": summ["status"] if summ else "incomplete",
                         "val_ce_after_epoch1": ep[0]["val_ce"] if ep else None,
                         "val_ppl_after_epoch1": ep[0]["val_ppl"] if ep else None,
                         "train_loss_epoch_mean": ep[0]["train_loss_epoch_mean"] if ep else None,
                         "max_grad_norm": summ["stability"]["max_grad_norm"] if summ else None,
                         "mean_grad_norm": summ["stability"]["mean_grad_norm"] if summ else None,
                         "loss_spikes": summ["stability"]["loss_spikes"] if summ else None,
                         "non_finite_steps": summ["stability"]["non_finite_steps"] if summ else None,
                         "nan_or_inf": bool(fail) or bool(summ and summ["stability"]["non_finite_steps"]),
                         "steps": summ["steps"] if summ else None,
                         "train_time_s": summ["train_time_s_total"] if summ else None, "run": run})
    return pd.DataFrame(rows)


def select_lrs(df: pd.DataFrame) -> dict:
    out = {}
    for m in MODELS:
        d = df[(df.model == m) & df.val_ce_after_epoch1.notna() & ~df.nan_or_inf].sort_values("val_ce_after_epoch1")
        if len(d) == 0:
            continue
        best, rule = d.iloc[0], "lowest val CE"
        if len(d) > 1 and (d.iloc[1].val_ce_after_epoch1 - best.val_ce_after_epoch1) <= TIE_TOL:
            pair = d.iloc[:2]
            best, rule = pair.loc[pair.lr.idxmin()], f"tie within {TIE_TOL} nats -> smaller LR"
        out[m] = {"lr": float(best.lr), "val_ce_after_epoch1": float(best.val_ce_after_epoch1), "rule": rule,
                  "candidates": {f"{r.lr:g}": float(r.val_ce_after_epoch1) for r in d.itertuples()}}
    return out


# ----------------------------------------------------------------------------- Stage 4: runs
def load_run(run: str) -> dict | None:
    path = ROOT / "results/runs" / run / "log.jsonl"
    if not path.exists():
        return None
    recs = read_jsonl(path)
    d = {"run": run, "config": next(r for r in recs if r["record"] == "config"),
         "steps": [r for r in recs if r["record"] == "step"], "epochs": [r for r in recs if r["record"] == "epoch"],
         "test": next((r for r in recs if r["record"] == "test"), None),
         "summary": next((r for r in recs if r["record"] == "summary"), None),
         "failure": next((r for r in recs if r["record"] == "failure"), None)}
    arr = ROOT / "results/runs" / run / "steps.npz"
    if arr.exists():
        z = np.load(arr)
        d["loss"], d["gnorm"] = z["loss"], z["grad_norm"]
    return d


def run_summary(prefix: str = "part1", models=MODELS, seeds=SEEDS, probe_sel: dict | None = None) -> pd.DataFrame:
    rows = []
    for m in models:
        for s in seeds:
            r = load_run(f"{prefix}/{m}_s{s}")
            if r is None:
                rows.append({"model": m, "seed": s, "status": "missing"}); continue
            summ, eps, test = r["summary"], r["epochs"], r["test"]
            best = summ["best_epoch"] if summ else None
            best_ep = next((e for e in eps if e["epoch"] == best), None)
            row = {"model": m, "seed": s, "status": summ["status"] if summ else ("failed" if r["failure"] else "incomplete"),
                   "lr": r["config"]["config"]["optim"]["lr"], "epochs_run": len(eps),
                   "early_stopped": bool(summ and summ["status"] == "early_stopped"),
                   "best_epoch": best, "best_is_last_epoch": bool(best == MAX_EPOCHS),
                   "best_val_ce": summ["best_val_ce"] if summ else None,
                   "train_loss_at_best_epoch": best_ep["train_loss_epoch_mean"] if best_ep else None,
                   "train_val_gap_at_best": (best_ep["val_ce"] - best_ep["train_loss_epoch_mean"]) if best_ep else None,
                   "epoch1_val_ce": eps[0]["val_ce"] if eps else None,
                   "test_ce": test["ce"] if test else None, "test_ppl": test["ppl"] if test else None,
                   "test_unk_target_fraction": test["unk_target_fraction"] if test else None,
                   "train_time_s_total": summ["train_time_s_total"] if summ else None,
                   "train_time_s_to_best": summ["train_time_s_to_best"] if summ else None,
                   "train_tokens_per_s": summ["train_tokens_per_s"] if summ else None,
                   "max_grad_norm": summ["stability"]["max_grad_norm"] if summ else None,
                   "mean_grad_norm": summ["stability"]["mean_grad_norm"] if summ else None,
                   "loss_spikes": summ["stability"]["loss_spikes"] if summ else None,
                   "non_finite_steps": summ["stability"]["non_finite_steps"] if summ else None,
                   "val_ce_history": json.dumps([round(e["val_ce"], 4) for e in eps])}
            if s == 0 and probe_sel and m in probe_sel:
                row["probe_val_ce_epoch1"] = probe_sel[m]["val_ce_after_epoch1"]
                row["probe_minus_full_epoch1"] = probe_sel[m]["val_ce_after_epoch1"] - row["epoch1_val_ce"] if row["epoch1_val_ce"] is not None else None
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_std(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def main_table(rs: pd.DataFrame, timing: dict | None, ngram: dict | None, breakdown: pd.DataFrame, prefix="part1",
               models=MODELS, labels=None) -> pd.DataFrame:
    labels = labels or {m: m for m in models}
    rows = []
    if ngram is not None:
        tA = timing["workload_a"].get("ngram") if timing else None
        tB = timing["workload_b"].get("ngram") if timing else None
        rows.append({"model": f"KN {ngram['selected']['order']}-gram ({ngram['selected']['discount']})",
                     "params_embedding": None, "params_core": None, "params_projection": None, "params_head": None,
                     "params_total": ngram["n_stored_total"], "params_trainable": None,
                     "ngrams_per_order": ";".join(f"{k}:{v}" for k, v in ngram["n_stored_per_order"].items()),
                     "memory_mb": ngram["memory_mb"], "n_seeds": 1,
                     "test_ce_mean": ngram["test"]["ce"], "test_ce_std": None, "test_ppl_mean": ngram["test"]["ppl"], "test_ppl_std": None,
                     "unk_target_fraction": ngram["test"]["unk_target_fraction"],
                     "best_epoch": None, "early_stopped": None, "epochs_run": None,
                     "train_time_s_total_mean": ngram["count_build_seconds"], "train_time_s_to_best_mean": ngram["count_build_seconds"],
                     "train_tokens_per_s": ngram["lm_train_tokens"] / ngram["count_build_seconds"],
                     "workloadA_tokens_per_s": tA["median_tokens_per_s"] if tA else None,
                     "workloadB_ms_per_token": tB["median_ms_per_token"] if tB else None,
                     "device": f"CPU ({ngram['cpu_model']})"})
    for m in models:
        d = rs[(rs.model == m) & (rs.status.isin(["completed", "early_stopped"]))]
        bd = breakdown[breakdown.model == m].iloc[0] if (breakdown is not None and (breakdown.model == m).any()) else None
        ce_m, ce_s = _mean_std(d.test_ce)
        ppl_m, ppl_s = _mean_std(d.test_ppl)
        tA = timing["workload_a"].get(labels[m]) if timing else None
        tB = timing["workload_b"].get(labels[m]) if timing else None
        row = {"model": labels[m], "n_seeds": int(len(d)),
               "test_ce_mean": ce_m, "test_ce_std": ce_s, "test_ppl_mean": ppl_m, "test_ppl_std": ppl_s,
               "unk_target_fraction": float(d.test_unk_target_fraction.mean()) if len(d) else None,
               "best_epoch": ";".join(str(int(x)) for x in d.best_epoch), "early_stopped": ";".join(str(bool(x)) for x in d.early_stopped),
               "epochs_run": ";".join(str(int(x)) for x in d.epochs_run),
               "train_time_s_total_mean": float(d.train_time_s_total.mean()) if len(d) else None,
               "train_time_s_to_best_mean": float(d.train_time_s_to_best.mean()) if len(d) else None,
               "train_tokens_per_s": float(d.train_tokens_per_s.mean()) if len(d) else None,
               "workloadA_tokens_per_s": tA["median_tokens_per_s"] if tA else None,
               "workloadB_ms_per_token": tB["median_ms_per_token"] if tB else None,
               "device": tA["device"] if tA else None}
        for k in SEEDS:
            dd = d[d.seed == k]
            row[f"test_ce_s{k}"] = float(dd.test_ce.iloc[0]) if len(dd) else None
        if bd is not None:
            row.update({"params_embedding": int(bd["embedding"]), "params_core": int(bd["core"]), "params_projection": int(bd["projection"]),
                        "params_head": int(bd["head"]), "params_total": int(bd["total"]), "params_trainable": int(bd["trainable"]),
                        "memory_mb": 4 * int(bd["total"]) / 2**20})
        rows.append(row)
    cols = ["model", "params_embedding", "params_core", "params_projection", "params_head", "params_total", "params_trainable",
            "ngrams_per_order", "memory_mb", "n_seeds", "test_ce_mean", "test_ce_std", "test_ce_s0", "test_ce_s1", "test_ce_s2",
            "test_ppl_mean", "test_ppl_std", "unk_target_fraction", "best_epoch", "early_stopped", "epochs_run",
            "train_time_s_total_mean", "train_time_s_to_best_mean", "train_tokens_per_s", "workloadA_tokens_per_s",
            "workloadB_ms_per_token", "device"]
    df = pd.DataFrame(rows)
    return df.reindex(columns=[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols])


# ----------------------------------------------------------------------------- figures
def _stack_runs(runs: list[dict]):
    """Align the 50-step train records and the per-epoch val records of several runs; returns (x_train, train[n,T], epochs, val[n,E])."""
    spe = runs[0]["config"]["steps_per_epoch"]
    T = min(len(r["steps"]) for r in runs)
    x_train = np.array([s["step"] for s in runs[0]["steps"][:T]]) / spe
    train = np.array([[s["train_loss_last50"] for s in r["steps"][:T]] for r in runs])
    E = min(len(r["epochs"]) for r in runs)
    epochs = np.arange(1, E + 1)
    val = np.array([[e["val_ce"] for e in r["epochs"][:E]] for r in runs])
    return x_train, train, epochs, val


def _band(ax, x, Y, key: str, label: str, lw=1.0, marker=True, ms=3, alpha_band=0.25, band="std", ls=None, color=None):
    """Mean line with a band: ± std over runs (ddof=1) or the min–max envelope."""
    m = Y.mean(axis=0)
    if band == "std":
        s = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(m)
        lo, hi = m - s, m + s
    else:
        lo, hi = Y.min(axis=0), Y.max(axis=0)
    st = plots.style(key)
    color = color or st["color"]; ls = ls if ls is not None else st["ls"]
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0)
    ax.plot(x, m, ls=ls, color=color, lw=lw, marker=st["marker"] if marker else None, ms=ms,
            markevery=max(1, len(x) // 12) if marker else None, label=label)
    return m, lo, hi


def fig_curves(prefix: str, m: str, label: str, stem: str) -> list[str]:
    """Per-model training loss (50-step means) and LM-val CE: mean ± std over runs; both in the model colour."""
    runs = [r for r in (load_run(f"{prefix}/{m}_s{s}") for s in SEEDS) if r is not None and r["steps"]]
    fig, ax = plots.new_figure(height=2.5)
    rows = []
    if runs:
        x_tr, tr, ep, val = _stack_runs(runs)
        mt, lt, ht = _band(ax, x_tr, tr, label, "training loss (50-step mean, dropout active)", lw=0.9, marker=False, ls="-")
        mv, lv, hv = _band(ax, ep, val, label, "LM-val cross-entropy", lw=1.2, ls="--")
        rows = [pd.DataFrame({"series": "train_loss_50step", "epoch": x_tr, "mean": mt, "std": tr.std(axis=0, ddof=1) if len(runs) > 1 else 0.0}),
                pd.DataFrame({"series": "val_ce", "epoch": ep, "mean": mv, "std": val.std(axis=0, ddof=1) if len(runs) > 1 else 0.0})]
        plots.dump(pd.concat(rows, ignore_index=True), f"results/{prefix}/curves_{m}.csv")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title(f"{plots.LABELS.get(label, label)}: training and validation loss (mean ± std)")
    ax.legend()
    return plots.save(fig, stem)


def fig_val_curves(prefix: str, models, labels, stem: str, ngram_val_ce: float | None = None) -> list[str]:
    fig, ax = plots.new_figure(height=2.5)
    n_runs, rows = 0, []
    for m in models:
        lab = labels[m]
        runs = [r for r in (load_run(f"{prefix}/{m}_s{s}") for s in SEEDS) if r is not None]
        if not runs:
            continue
        n_runs = max(n_runs, len(runs))
        _, _, ep, val = _stack_runs(runs)
        mv, lo, hi = _band(ax, ep, val, lab, plots.LABELS.get(lab, lab))
        rows.append(pd.DataFrame({"model": lab, "epoch": ep, "val_ce_mean": mv, "val_ce_std": val.std(axis=0, ddof=1) if len(runs) > 1 else 0.0}))
    if ngram_val_ce is not None:
        ax.axhline(ngram_val_ce, **{k: v for k, v in plots.style("ngram").items() if k != "marker"}, lw=0.9, label=plots.LABELS["ngram"])
        rows.append(pd.DataFrame({"model": "ngram", "epoch": [None], "val_ce_mean": [ngram_val_ce], "val_ce_std": [None]}))
    plots.dump(pd.concat(rows, ignore_index=True), f"results/{prefix}/val_curves.csv")
    ax.set_xlabel("epoch"); ax.set_ylabel("LM-val cross-entropy (nats/token)")
    ax.set_title(f"LM-val cross-entropy per epoch (mean ± std)")
    ax.legend()
    return plots.save(fig, stem)


def fig_learning_curves(prefix: str, models, labels, stem: str, ngram_val_ce: float | None) -> list[str]:
    """1×3 panels (RNN, LSTM, Transformer), shared y: training loss (50-step means) and LM-val CE per epoch as the
    mean over runs with a min–max band, plus the n-gram LM-val CE as a dashed reference line."""
    fig, axes = plots.new_figure(width=plots.HALF_PAGE_IN, height=1.9, ncols=len(models), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    rows = []
    for ax, m in zip(np.atleast_1d(axes), models):
        lab = labels[m]
        runs = [r for r in (load_run(f"{prefix}/{m}_s{s}") for s in SEEDS) if r is not None and r["steps"]]
        if not runs:
            continue
        x_tr, tr, ep, val = _stack_runs(runs)
        mt, lt, ht = _band(ax, x_tr, tr, lab, "train loss", lw=0.8, marker=False, band="minmax", ls="-", alpha_band=0.3)
        mv, lv, hv = _band(ax, ep, val, lab, "LM-val CE", lw=1.0, band="minmax", ls="--", alpha_band=0.3)
        if ngram_val_ce is not None:
            ax.axhline(ngram_val_ce, color=plots.COLORS["ngram"], ls="--", lw=0.7)
        ax.set_title(plots.LABELS.get(lab, lab), fontsize=7)
        ax.set_xlabel("epoch", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_xticks([1, 4, 8, 12])
        rows += [pd.DataFrame({"model": lab, "series": "train_loss_50step", "epoch": x_tr, "mean": mt, "min": lt, "max": ht}),
                 pd.DataFrame({"model": lab, "series": "val_ce", "epoch": ep, "mean": mv, "min": lv, "max": hv})]
    np.atleast_1d(axes)[0].set_ylabel("cross-entropy (nats/token)", fontsize=7)
    np.atleast_1d(axes)[0].set_ylim(3.6, 7.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#777777", ls="-", lw=0.9, label="training loss (50-step mean)"),
               Line2D([], [], color="#777777", ls="--", lw=1.0, marker="o", ms=3, label="LM-val cross-entropy (markers)"),
               Line2D([], [], color=plots.COLORS["ngram"], ls="--", lw=0.7, label=plots.LABELS["ngram"] + " LM-val CE")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.06))
    plots.dump(pd.concat(rows, ignore_index=True), f"results/{prefix}/learning_curves.csv")
    return plots.save(fig, stem)


def loss_by_position(prefix: str, models, labels, ngram_ce_path: str | None, out_csv: str, stem: str) -> list[str]:
    curves, bands = {}, {}
    if ngram_ce_path and (ROOT / ngram_ce_path).exists():
        ce = np.load(ROOT / ngram_ce_path)
        curves["ngram"] = np.nanmean(ce, axis=0)
    n_runs = 0
    for m in models:
        per_seed = []
        for s in SEEDS:
            p = ROOT / "data/derived" / prefix / f"{m}_s{s}" / "test_ce.npy"
            if p.exists():
                per_seed.append(np.nanmean(np.load(p), axis=0))
        if per_seed:
            A = np.array(per_seed); n_runs = max(n_runs, len(per_seed))
            curves[labels[m]] = A.mean(axis=0)
            bands[labels[m]] = A.std(axis=0, ddof=1) if len(per_seed) > 1 else np.zeros(A.shape[1])
    df = pd.DataFrame({"position": np.arange(1, 257)} | {k: v for k, v in curves.items()} | {f"{k}_std": v for k, v in bands.items()})
    df.to_csv(ROOT / out_csv, index=False)
    fig, ax = plots.new_figure(height=2.5)
    for k, v in curves.items():
        st = plots.style(k)
        if k in bands:
            ax.fill_between(df.position, v - bands[k], v + bands[k], color=st["color"], alpha=0.25, linewidth=0)
        ax.plot(df.position, v, color=st["color"], ls=st["ls"], lw=0.9, marker=st["marker"], ms=2.5, markevery=[0, 1, 3, 7, 15, 31, 63, 127, 255], label=plots.LABELS.get(k, k))
    ax.set_xscale("log"); ax.set_xlabel("position in 256-token block (log scale)"); ax.set_ylabel("mean LM-test CE (nats/token)")
    ax.set_title(f"Loss by position on LM-test (neural models: mean ± std)")
    ax.legend()
    return plots.save(fig, stem), df


def fig_ngram_order(stem: str) -> list[str]:
    df = pd.read_csv(ROOT / "results/part1/ngram_selection.csv")
    d = df[df.discount == "modifiedKN"].sort_values("order")
    fig, ax = plots.new_figure(height=2.4)
    st = plots.style("ngram")
    ax.plot(d.order, d.val_ppl, color=st["color"], ls="-", marker=st["marker"], ms=4, label="LM-val perplexity")
    ax.set_xlabel("n-gram order (modified Kneser–Ney)"); ax.set_ylabel("LM-val perplexity")
    ax.set_xticks(d.order); ax.set_yscale("log")
    for o, pp in zip(d.order, d.val_ppl):
        ax.annotate(f"{pp:.0f}", (o, pp), textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax2 = ax.twinx()
    ax2.bar(d.order, d.memory_mb, width=0.35, alpha=0.35, color="#56B4E9", hatch="//", edgecolor="#56B4E9", label="memory (MB)")
    ax2.set_ylabel("model memory (MB)"); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center")
    ax.set_title("n-gram order vs LM-val perplexity and memory")
    plots.dump(d[["order", "val_ce", "val_ppl", "memory_mb", "n_stored_total"]], "results/part1/ngram_order_figure.csv")
    return plots.save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probes", "main"])
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    if args.what == "probes":
        df = probe_table()
        df.to_csv(ROOT / "results/part1/lr_probes.csv", index=False)
        sel = select_lrs(df)
        write_json(sel, ROOT / "results/part1/lr_selected.json")
        print(df.drop(columns=["run"]).to_string(index=False)); print("\nselected:", json.dumps(sel, indent=1))
        return
    probe_sel = read_json("results/part1/lr_selected.json") if (ROOT / "results/part1/lr_selected.json").exists() else None
    rs = run_summary("part1", probe_sel=probe_sel)
    rs.to_csv(ROOT / "results/part1/run_summary.csv", index=False)
    timing = read_json("results/part1/timing.json") if (ROOT / "results/part1/timing.json").exists() else None
    ngram = read_json("results/part1/ngram_model.json") if (ROOT / "results/part1/ngram_model.json").exists() else None
    bd = pd.read_csv(ROOT / "results/part1/param_breakdown.csv")
    mt = main_table(rs, timing, ngram, bd)
    mt.to_csv(ROOT / "results/part1/main_table.csv", index=False)
    figs = []
    for m in MODELS:
        figs += fig_curves("part1", m, m, f"part1_curves_{m}")
    figs += fig_val_curves("part1", MODELS, {m: m for m in MODELS}, "part1_val_curves", ngram["val"]["ce"] if ngram else None)
    figs += fig_learning_curves("part1", MODELS, {m: m for m in MODELS}, "part1_learning_curves", ngram["val"]["ce"] if ngram else None)
    f, _ = loss_by_position("part1", MODELS, {m: m for m in MODELS}, "data/derived/part1/ngram_test_ce.npy",
                            "results/part1/loss_by_position.csv", "part1_loss_by_position")
    figs += f
    figs += fig_ngram_order("part1_ngram_order")
    print(rs.drop(columns=["val_ce_history"]).to_string(index=False))
    print(); print(mt.to_string(index=False))
    print("\nfigures:", *figs, sep="\n  ")


if __name__ == "__main__":
    main()
