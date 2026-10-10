"""Part II tables and figures (TASK §5, user additions D34/D57/D94/D95/D108/D115).

Outputs under results/part2/: main_table.csv, stability.csv, embedding_std.csv, freq_buckets.csv,
context_freq_buckets.csv, *_diff_vs_A.csv, config_check.json, nearest_neighbours.md, the C-scaled control
tables (ablation_glove_scaled.csv, stability_glove_scaled.csv, *_with_scaled.csv) and the figure-value CSVs;
figures part2_val_curves, part2_train_curves, part2_freq_buckets / part2_context_freq_buckets (difference to A)
and their *_abs versions (no in-figure titles).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis import plots
from src.analysis.part1_tables import SEEDS, _band, _mean_std, _stack_runs, load_run
from src.common import ROOT, read_json, write_json
from src.evaluate import ce_by_context_frequency_bucket, ce_by_frequency_bucket, make_blocks

SETTINGS = {"A-random": "part1/lstm_s{s}", "B-own": "part2/lstm_ownfrozen_s{s}", "C-glove": "part2/lstm_glovefrozen_s{s}"}
EXTRA = {"C-scaled": "part2/lstm_glovescaled_s{s}"}          # user D95 control, reported outside the main table
ALL_SETTINGS = {**SETTINGS, **EXTRA}
NN_WORDS = ["good", "bad", "boring", "actor", "plot"]


def runs_for(setting: str) -> list[dict]:
    return [r for r in (load_run(ALL_SETTINGS[setting].format(s=s)) for s in SEEDS) if r is not None]


def config_check() -> dict:
    """Part I LSTM configs must equal the Part II configs except for the embedding fields (TASK §5 A)."""
    a = load_run(SETTINGS["A-random"].format(s=0))["config"]["config"]
    out = {}
    for setting in ("B-own", "C-glove"):
        b = load_run(SETTINGS[setting].format(s=0))["config"]["config"]
        diffs = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if k not in ("embedding", "run_name", "_config_path") and a.get(k) != b.get(k)}
        out[setting] = {"identical_except_embedding_and_run_name": not diffs, "differences": diffs,
                        "embedding_A": a["embedding"], f"embedding_{setting}": b["embedding"]}
    return out


def main_table(settings=None) -> pd.DataFrame:
    rows = []
    for setting in (settings or SETTINGS):
        rs = runs_for(setting)
        if not rs:
            continue
        tests = [r["test"] for r in rs if r["test"]]
        ce_m, ce_s = _mean_std([t["ce"] for t in tests]); ppl_m, ppl_s = _mean_std([t["ppl"] for t in tests])
        summ = [r["summary"] for r in rs]
        row = {"setting": setting, "runs": ";".join(r["run"] for r in rs), "n_seeds": len(rs),
               "params_total": rs[0]["config"]["param_breakdown"]["total"], "params_trainable": rs[0]["config"]["param_breakdown"]["trainable"],
               "embedding_frozen": rs[0]["config"]["embedding"]["freeze"],
               "frozen_checksum_unchanged": ";".join(str(s["embedding"].get("frozen_table_unchanged")) for s in summ),
               "test_ce_mean": ce_m, "test_ce_std": ce_s, "test_ppl_mean": ppl_m, "test_ppl_std": ppl_s,
               "unk_target_fraction": float(np.mean([t["unk_target_fraction"] for t in tests])) if tests else None,
               "best_epoch": ";".join(str(s["best_epoch"]) for s in summ), "early_stopped": ";".join(str(s["status"] == "early_stopped") for s in summ),
               "epochs_run": ";".join(str(s["epochs_run"]) for s in summ),
               "train_time_s_total_mean": float(np.mean([s["train_time_s_total"] for s in summ])),
               "train_time_s_to_best_mean": float(np.mean([s["train_time_s_to_best"] for s in summ])),
               "train_tokens_per_s": float(np.mean([s["train_tokens_per_s"] for s in summ])),
               "emb_per_dim_std_init": float(np.mean([s["embedding"]["per_dim_std_before"] for s in summ])),
               "emb_per_dim_std_after": float(np.mean([s["embedding"]["per_dim_std_after"] for s in summ])),
               "lr": rs[0]["config"]["config"]["optim"]["lr"], "gpu": ";".join(str(r["config"]["env"].get("CUDA_VISIBLE_DEVICES")) for r in rs)}
        for r in rs:
            row[f"test_ce_s{r['config']['config']['seed']}"] = r["test"]["ce"] if r["test"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(settings=None) -> pd.DataFrame:
    rows = []
    for setting in (settings or SETTINGS):
        rs = runs_for(setting)
        if not rs:
            continue
        summ = [r["summary"] for r in rs]
        rows.append({"setting": setting, "n_seeds": len(rs),
                     "across_seed_std_test_ce": _mean_std([r["test"]["ce"] for r in rs if r["test"]])[1],
                     "across_seed_std_best_val_ce": _mean_std([s["best_val_ce"] for s in summ])[1],
                     "max_grad_norm_max_over_seeds": max(s["stability"]["max_grad_norm"] for s in summ),
                     "max_grad_norm_per_seed": ";".join(f"{s['stability']['max_grad_norm']:.2f}" for s in summ),
                     "mean_grad_norm_mean_over_seeds": float(np.mean([s["stability"]["mean_grad_norm"] for s in summ])),
                     "loss_spikes_total": int(sum(s["stability"]["loss_spikes"] for s in summ)),
                     "loss_spikes_per_seed": ";".join(str(s["stability"]["loss_spikes"]) for s in summ),
                     "non_finite_steps_total": int(sum(s["stability"]["non_finite_steps"] for s in summ)),
                     "spike_definition": summ[0]["stability"]["spike_definition"]})
    return pd.DataFrame(rows)


def embedding_std_table() -> pd.DataFrame:
    rows = []
    for setting in SETTINGS:
        for r in runs_for(setting):
            e = r["summary"]["embedding"]
            rows.append({"setting": setting, "seed": r["config"]["config"]["seed"], "frozen": e["freeze"],
                         "per_dim_std_at_init": e["per_dim_std_before"], "per_dim_std_after_training": e["per_dim_std_after"],
                         "sha256_unchanged": e.get("frozen_table_unchanged")})
    g = read_json("results/part2/glove_coverage.json") if (ROOT / "results/part2/glove_coverage.json").exists() else None
    w = read_json("results/part2/word2vec_training.json") if (ROOT / "results/part2/word2vec_training.json").exists() else None
    extra = []
    if w:
        extra.append({"setting": "B-own table", "seed": None, "frozen": True, "per_dim_std_at_init": w["per_dim_std_mean"], "per_dim_std_after_training": w["per_dim_std_mean"], "sha256_unchanged": None})
    if g:
        extra.append({"setting": "C-glove table (covered vectors)", "seed": None, "frozen": True, "per_dim_std_at_init": g["per_dim_std_mean"]["covered_vectors"], "per_dim_std_after_training": g["per_dim_std_mean"]["covered_vectors"], "sha256_unchanged": None})
        extra.append({"setting": "C-glove table (random-init missing words)", "seed": None, "frozen": True, "per_dim_std_at_init": g["per_dim_std_mean"]["random_init_vectors"], "per_dim_std_after_training": g["per_dim_std_mean"]["random_init_vectors"], "sha256_unchanged": None})
    return pd.DataFrame(rows + extra)


def freq_buckets(kind: str = "target", settings=None) -> pd.DataFrame:
    """kind = 'target': bucket by the target's LM-train rank; 'context': by the preceding input token's rank (D94)."""
    vocab = read_json("data/streams/vocab.json")
    counts = np.asarray(vocab["counts"])
    inputs, targets = make_blocks(np.load(ROOT / "data/streams/lm_test.npy"), 256)
    rows = []
    for setting in (settings or SETTINGS):
        per_seed = []
        for s in SEEDS:
            p = ROOT / "data/derived" / ALL_SETTINGS[setting].format(s=s) / "test_ce.npy"
            if p.exists():
                ce = np.load(p).astype(np.float64)
                per_seed.append(ce_by_frequency_bucket(ce, targets, counts) if kind == "target"
                                else ce_by_context_frequency_bucket(ce, inputs, targets, counts))
        if not per_seed:
            continue
        for b in per_seed[0]:
            vals = [x[b]["ce"] for x in per_seed]
            rows.append({"setting": setting, "bucket": b, "ce_mean_over_seeds": float(np.mean(vals)), "ce_std_over_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                         "n_targets": per_seed[0][b]["n_targets"], "n_seeds": len(per_seed)})
    return pd.DataFrame(rows)


def nearest_neighbours() -> str:
    """10 cosine nearest neighbours in the word2vec table (B), the GloVe table (C) and the trained input-embedding
    table of the Part I LSTM seed 0 (A after training); specials excluded."""
    import torch
    vocab = read_json("data/streams/vocab.json"); itos = vocab["itos"]; stoi = {t: i for i, t in enumerate(itos)}
    tables = {"word2vec (B)": np.load(ROOT / "data/embeddings/word2vec_300.npy"),
              "GloVe 6B 300d (C)": np.load(ROOT / "data/embeddings/glove_300.npy")}
    ck = ROOT / "checkpoints/part1/lstm_s0/best.pt"
    if ck.exists():
        tables["trained LSTM input embedding, seed 0 (A)"] = torch.load(ck, map_location="cpu", weights_only=False)["model"]["emb.weight"].float().numpy()
    units = {}
    for name, T in tables.items():
        T = T.astype(np.float64).copy(); T[:3] = 0.0
        norms = np.linalg.norm(T, axis=1); norms[norms == 0] = np.inf
        units[name] = T / norms[:, None]
    lines = ["# Nearest neighbours (cosine, within the 25,003-token vocabulary, specials excluded)\n",
             "| query | " + " | ".join(units) + " |", "|---|" + "---|" * len(units)]
    for w in NN_WORDS:
        i = stoi[w]; cells = []
        for name, U in units.items():
            sims = U @ U[i]; sims[i] = -np.inf; sims[:3] = -np.inf
            top = np.argsort(-sims)[:10]
            cells.append(", ".join(f"{itos[j]} ({sims[j]:.2f})" for j in top))
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fig_curves(stem_val: str, stem_train: str) -> list[str]:
    figs, n_runs, rows_v, rows_t = [], 0, [], []
    fig, ax = plots.new_figure(height=2.5)
    for setting in SETTINGS:
        runs = runs_for(setting)
        if not runs:
            continue
        n_runs = max(n_runs, len(runs))
        _, _, ep, val = _stack_runs(runs)
        mv, _, _ = _band(ax, ep, val, setting, plots.LABELS[setting])
        rows_v.append(pd.DataFrame({"setting": setting, "epoch": ep, "val_ce_mean": mv, "val_ce_std": val.std(axis=0, ddof=1) if len(runs) > 1 else 0.0}))
    ax.set_xlabel("epoch"); ax.set_ylabel("LM-val cross-entropy (nats/token)")
    ax.legend()
    plots.dump(pd.concat(rows_v, ignore_index=True), "results/part2/val_curves.csv")
    figs += plots.save(fig, stem_val)
    fig, ax = plots.new_figure(height=2.5)
    for setting in SETTINGS:
        runs = [r for r in runs_for(setting) if r["steps"]]
        if not runs:
            continue
        x_tr, tr, _, _ = _stack_runs(runs)
        mt, _, _ = _band(ax, x_tr, tr, setting, plots.LABELS[setting], lw=0.8, marker=True)
        rows_t.append(pd.DataFrame({"setting": setting, "epoch": x_tr, "train_loss_mean": mt, "train_loss_std": tr.std(axis=0, ddof=1) if len(runs) > 1 else 0.0}))
    ax.set_xlabel("epoch"); ax.set_ylabel("train CE, 50-step mean (nats/token)"); ax.set_ylim(top=7.0)
    ax.legend()
    plots.dump(pd.concat(rows_t, ignore_index=True), "results/part2/train_curves.csv")
    figs += plots.save(fig, stem_train)
    return figs


def fig_freq(df: pd.DataFrame, stem: str, kind: str = "target") -> list[str]:
    buckets = ["1-100", "101-1k", "1k-5k", "5k-25k", "<unk>"]
    fig, ax = plots.new_figure(height=2.5)
    w = 0.26
    for k, setting in enumerate(SETTINGS):
        d = df[df.setting == setting].set_index("bucket").reindex(buckets)
        ax.bar(np.arange(len(buckets)) + (k - 1) * w, d.ce_mean_over_seeds, w, yerr=d.ce_std_over_seeds, capsize=1.5,
               color=plots.COLORS[setting], hatch=plots.hatch(setting), edgecolor="white", linewidth=0.4, label=plots.LABELS[setting])
    ax.set_xticks(np.arange(len(buckets))); ax.set_xticklabels(buckets)
    ax.set_xlabel("LM-train frequency rank of the target" if kind == "target" else "LM-train frequency rank of the preceding input token")
    ax.set_ylabel("mean LM-test CE (nats/token)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.42)                      # headroom: legend sits above the bars
    ax.legend(fontsize=6, loc="upper left", ncol=1, handlelength=1.4)
    return plots.save(fig, stem)


BUCKET_ORDER = ["1-100", "101-1k", "1k-5k", "5k-25k", "<unk>"]


def freq_diff_vs_A(df: pd.DataFrame) -> pd.DataFrame:
    """Difference of the mean bucket CE to setting A, with a combined std sqrt(std_A^2 + std_X^2) (settings are
    separate training runs, so their run-to-run spreads are combined as independent)."""
    a = df[df.setting == "A-random"].set_index("bucket")
    rows = []
    for setting in [s for s in ("B-own", "C-glove", "C-scaled") if (df.setting == s).any()]:
        x = df[df.setting == setting].set_index("bucket")
        for b in BUCKET_ORDER:
            rows.append({"setting": setting, "bucket": b, "diff_vs_A_mean": float(x.loc[b, "ce_mean_over_seeds"] - a.loc[b, "ce_mean_over_seeds"]),
                         "diff_vs_A_std": float(np.hypot(x.loc[b, "ce_std_over_seeds"], a.loc[b, "ce_std_over_seeds"])),
                         "mean_ce_setting": float(x.loc[b, "ce_mean_over_seeds"]), "mean_ce_A": float(a.loc[b, "ce_mean_over_seeds"]),
                         "n_targets": int(x.loc[b, "n_targets"])})
    return pd.DataFrame(rows)


def fig_freq_diff(df: pd.DataFrame, stem: str, kind: str, out_csv: str) -> list[str]:
    """Grouped bars of (setting − A) per bucket with ± combined-std error bars and a zero line; C-scaled lighter."""
    d = freq_diff_vs_A(df)
    plots.dump(d, out_csv)
    settings = list(dict.fromkeys(d.setting))
    legend = {"B-own": "B − A (own word2vec, frozen)", "C-glove": "C − A (GloVe, frozen)", "C-scaled": "C-scaled − A (extra control)"}
    fig, ax = plots.new_figure(height=2.4)
    w = 0.8 / len(settings)
    for k, setting in enumerate(settings):
        dd = d[d.setting == setting].set_index("bucket").reindex(BUCKET_ORDER)
        extra = setting == "C-scaled"
        ax.bar(np.arange(len(BUCKET_ORDER)) + (k - (len(settings) - 1) / 2) * w, dd.diff_vs_A_mean, w,
               yerr=dd.diff_vs_A_std, capsize=1.5, error_kw={"elinewidth": 0.7},
               color=plots.COLORS[setting], alpha=0.45 if extra else 1.0, hatch=plots.hatch(setting),
               edgecolor="white", linewidth=0.4, label=legend[setting])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(BUCKET_ORDER))); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_xlabel("LM-train frequency rank of the target" if kind == "target" else "LM-train frequency rank of the preceding input token")
    ax.set_ylabel("Δ mean LM-test CE vs A (nats/token)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(fontsize=6, loc="upper left")
    return plots.save(fig, stem)


def main() -> None:
    out = ROOT / "results/part2"; out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    cc = config_check(); write_json(cc, out / "config_check.json")
    print("config check:", json.dumps({k: v["identical_except_embedding_and_run_name"] for k, v in cc.items()}))
    mt = main_table(); mt.to_csv(out / "main_table.csv", index=False); print(mt.drop(columns=["runs"]).to_string(index=False))
    st = stability_table(); st.to_csv(out / "stability.csv", index=False); print(st.drop(columns=["spike_definition"]).to_string(index=False))
    es = embedding_std_table(); es.to_csv(out / "embedding_std.csv", index=False); print(es.to_string(index=False))
    fb = freq_buckets("target"); fb.to_csv(out / "freq_buckets.csv", index=False); print(fb.to_string(index=False))
    cb = freq_buckets("context"); cb.to_csv(out / "context_freq_buckets.csv", index=False); print(cb.to_string(index=False))
    (out / "nearest_neighbours.md").write_text(nearest_neighbours())
    figs = fig_curves("part2_val_curves", "part2_train_curves") + fig_freq(fb, "part2_freq_buckets_abs", "target") + fig_freq(cb, "part2_context_freq_buckets_abs", "context")
    fbs, cbs = fb, cb
    # extra control C-scaled (D95): own files, never in the main table
    if runs_for("C-scaled"):
        ext = ["B-own", "C-glove", "C-scaled"]
        main_table(["A-random"] + ext).to_csv(out / "ablation_glove_scaled.csv", index=False)
        stability_table(["A-random"] + ext).to_csv(out / "stability_glove_scaled.csv", index=False)
        fbs = freq_buckets("target", ["A-random"] + ext); fbs.to_csv(out / "freq_buckets_with_scaled.csv", index=False)
        cbs = freq_buckets("context", ["A-random"] + ext); cbs.to_csv(out / "context_freq_buckets_with_scaled.csv", index=False)
        print("C-scaled extra tables written")
    figs += fig_freq_diff(fbs, "part2_freq_buckets", "target", "results/part2/freq_buckets_diff_vs_A.csv")
    figs += fig_freq_diff(cbs, "part2_context_freq_buckets", "context", "results/part2/context_freq_buckets_diff_vs_A.csv")
    print("figures:", *figs, sep="\n  ")


if __name__ == "__main__":
    main()
