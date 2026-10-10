"""Part III classifiers (TASK §6, user additions D32/D33/D46/D91–D93/D107 and the TASK §6 optional ablation).

Every feature set uses the same procedure: Pipeline(StandardScaler, LogisticRegression(lbfgs,
max_iter 2000, C)) with C from the grid in configs/part3/classify.yaml ({0.01, 0.1, 1, 10, 100, 1000})
chosen by accuracy on the 2,500-review classifier-val split (ties -> smaller C, D80), fit on the 22,500
classifier-train reviews, evaluated once on the 25,000 IMDb-test reviews. Baselines over the very same
inputs: TF-IDF (uni+bigrams, top 100k, no scaler because rows are L2-normalised, D32) and the masked means
of three embedding tables (frozen GloVe; own word2vec and the trained Part I LSTM seed-0 table via
--mode extra_baselines). Metrics: accuracy and macro-F1 with 95 % bootstrap CIs (1,000 resamples, seed 0,
identical resample indices for every set); paired bootstrap of accuracy differences (10,000 resamples);
D46 robustness subset = test reviews whose cleaned text does not occur in IMDb-train; accuracy by
full-review-length bucket; LM-vs-downstream table; error-analysis exports for the two LM feature sets with
the highest classifier-val accuracy (ties keep table order) and for pairs chosen by design.
Modes: main | ablation (last 255 tokens, best LM) | extra_baselines | exports (rebuild the error exports
from the saved predictions, no fitting).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, load_config, read_json, read_jsonl, write_json

PART1_LMS = {"rnn": "part1/rnn_s0", "lstm": "part1/lstm_s0", "transformer": "part1/transformer_s0"}


# ----------------------------------------------------------------------------- data
def load_inputs(derived, suffix: str) -> dict:
    return {split: dict(np.load(derived / f"inputs_{split}{suffix}.npz")) for split in ("train", "test")}


def clf_indices(cfg, orig_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions (within the loaded train subset) of the classifier-train / classifier-val reviews."""
    sp = read_json(ROOT / cfg["processed_dir"] / "clf_split.json")
    pos = {int(o): p for p, o in enumerate(orig_idx)}
    tr = np.asarray([pos[i] for i in sp["clf_train"] if i in pos]); va = np.asarray([pos[i] for i in sp["clf_val"] if i in pos])
    return tr, va


def texts_for_tfidf(inp: dict, itos: list[str]) -> list[str]:
    X, n = inp["ids"], inp["n_input"]
    return [" ".join(itos[t] for t in X[i, :n[i]]) for i in range(len(n))]


def glove_mean(inp: dict, table: np.ndarray) -> np.ndarray:
    X, n = inp["ids"], inp["n_input"]
    mask = (np.arange(X.shape[1])[None, :] < n[:, None]).astype(np.float32)
    return (table[X] * mask[:, :, None]).sum(1) / n[:, None]


def robustness_mask(cfg, orig_idx: np.ndarray) -> np.ndarray:
    """D46: True for test reviews whose cleaned text does NOT appear in IMDb-train."""
    train_clean = {r["clean"] for r in read_jsonl(ROOT / cfg["processed_dir"] / "train.jsonl")}
    test = read_jsonl(ROOT / cfg["processed_dir"] / "test.jsonl")
    return np.asarray([test[i]["clean"] not in train_clean for i in orig_idx])


# ----------------------------------------------------------------------------- classifier
def make_pipeline(kind: str, C: float, cfg) -> Pipeline:
    lr = LogisticRegression(C=C, solver=cfg["classifier"]["solver"], max_iter=cfg["classifier"]["max_iter"])
    if kind == "tfidf":
        vec = TfidfVectorizer(ngram_range=tuple(cfg["tfidf"]["ngram_range"]), max_features=cfg["tfidf"]["max_features"],
                              tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None)
        return Pipeline([("tfidf", vec), ("logreg", lr)])          # no StandardScaler (D32)
    return Pipeline([("scaler", StandardScaler()), ("logreg", lr)])


def fit_select(kind: str, Xtr, ytr, Xva, yva, cfg) -> tuple[dict, Pipeline]:
    recs, pipes = [], {}
    for C in cfg["classifier"]["C_grid"]:
        pipe = make_pipeline(kind, C, cfg)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(Xtr, ytr)
            conv_warn = any(issubclass(x.category, ConvergenceWarning) for x in w)
        n_iter = int(np.max(pipe.named_steps["logreg"].n_iter_))
        recs.append({"C": C, "val_acc": float(accuracy_score(yva, pipe.predict(Xva))), "n_iter": n_iter,
                     "converged": (not conv_warn) and n_iter < cfg["classifier"]["max_iter"], "fit_seconds": time.perf_counter() - t0})
        pipes[C] = pipe
    best_acc = max(r["val_acc"] for r in recs)
    tied = [r["C"] for r in recs if r["val_acc"] == best_acc]
    C_star = min(tied)                                                # tie rule: smaller C (D80)
    return {"per_C": recs, "chosen_C": C_star, "val_acc": best_acc, "tie": len(tied) > 1}, pipes[C_star]


def bootstrap(y, pred, idx: np.ndarray) -> dict:
    accs = np.array([accuracy_score(y[i], pred[i]) for i in idx]); f1s = np.array([f1_score(y[i], pred[i], average="macro") for i in idx])
    return {"acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def evaluate(pipe, Xte, yte, boot_idx, robust, lengths, buckets) -> tuple[dict, np.ndarray]:
    prob = pipe.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {"test_acc": float(accuracy_score(yte, pred)), "test_macro_f1": float(f1_score(yte, pred, average="macro")), **bootstrap(yte, pred, boot_idx),
           "robust_n": int(robust.sum()), "robust_excluded": int((~robust).sum()),
           "robust_acc": float(accuracy_score(yte[robust], pred[robust])), "robust_macro_f1": float(f1_score(yte[robust], pred[robust], average="macro"))}
    edges = [0] + list(buckets) + [10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lengths > lo) & (lengths <= hi)
        name = f"len_{lo + 1}-{hi}" if hi < 10**9 else f"len_>{lo}"
        out[f"acc_{name}"] = float(accuracy_score(yte[sel], pred[sel])) if sel.any() else None
        out[f"n_{name}"] = int(sel.sum())
    return out, prob


# ----------------------------------------------------------------------------- tables
def lm_vs_downstream(results: pd.DataFrame, cfg) -> pd.DataFrame:
    p1 = pd.read_csv(ROOT / "results/part1/main_table.csv") if (ROOT / "results/part1/main_table.csv").exists() else None
    p2 = pd.read_csv(ROOT / "results/part2/main_table.csv") if (ROOT / "results/part2/main_table.csv").exists() else None
    rows = []
    for lm, run in cfg["lms"].items():
        log = ROOT / "results/runs" / run / "log.jsonl"
        if not log.exists() or not (results.feature_set == f"{lm}_{cfg['poolings'][0]}").any():
            continue
        test = next((r for r in read_jsonl(log) if r["record"] == "test"), None)
        row = {"lm": lm, "run": run, "lm_test_ppl_seed0": test["ppl"] if test else None, "lm_test_ce_seed0": test["ce"] if test else None}
        src = None
        if lm in ("rnn", "lstm", "transformer") and p1 is not None:
            src = p1[p1.model == lm]
        elif p2 is not None:
            src = p2[p2.setting == {"lstm_own": "B-own", "lstm_glove": "C-glove"}[lm]]
        if src is not None and len(src):
            row["lm_test_ppl_3seed_mean"], row["lm_test_ppl_3seed_std"] = float(src.test_ppl_mean.iloc[0]), float(src.test_ppl_std.iloc[0])
        for pool in cfg["poolings"]:
            r = results[results.feature_set == f"{lm}_{pool}"]
            if len(r):
                row[f"acc_{pool}"] = float(r.test_acc.iloc[0]); row[f"macro_f1_{pool}"] = float(r.test_macro_f1.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def load_embedding_table(spec: str) -> np.ndarray:
    """A .npy table, or `checkpoint:<run>` for the trained input-embedding table of that run's best checkpoint."""
    if spec.startswith("checkpoint:"):
        import torch
        ckpt = torch.load(ROOT / "checkpoints" / spec.split(":", 1)[1] / "best.pt", map_location="cpu", weights_only=False)
        return ckpt["model"]["emb.weight"].float().numpy()
    return np.load(ROOT / spec).astype(np.float32)


def paired_bootstrap(cfg, probs: dict, yte: np.ndarray, lms: dict) -> pd.DataFrame:
    """Accuracy difference A - B with a percentile 95 % CI from paired bootstrap resamples of the test set
    (identical resampled indices for both models and for every pair; user D92)."""
    pb = cfg["paired_bootstrap"]
    pairs = [tuple(p) for p in pb["pairs"]]
    if pb.get("mean_vs_last_for_every_lm"):
        pairs += [(f"{lm}_mean", f"{lm}_last") for lm in lms]
    pairs += [tuple(p) for p in cfg.get("extra_paired_pairs", []) if all(x in probs for x in p)]
    rng = np.random.default_rng(pb["seed"])
    n, R = len(yte), pb["n_resamples"]
    correct = {k: ((p >= 0.5).astype(int) == yte).astype(np.int8) for k, p in probs.items()}
    diffs = {pair: np.empty(R) for pair in pairs if pair[0] in correct and pair[1] in correct}
    chunk = 500
    for start in range(0, R, chunk):
        idx = rng.integers(0, n, size=(min(chunk, R - start), n))
        for (a, b) in diffs:
            d = (correct[a] - correct[b]).astype(np.float64)
            diffs[(a, b)][start:start + idx.shape[0]] = d[idx].mean(axis=1)
    rows = []
    for (a, b), v in diffs.items():
        rows.append({"model_a": a, "model_b": b, "acc_a": float(correct[a].mean()), "acc_b": float(correct[b].mean()),
                     "acc_diff_a_minus_b": float(correct[a].mean() - correct[b].mean()),
                     "ci95_lo": float(np.percentile(v, 2.5)), "ci95_hi": float(np.percentile(v, 97.5)),
                     "ci_excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0),
                     "n_a_right_b_wrong": int(((correct[a] == 1) & (correct[b] == 0)).sum()),
                     "n_b_right_a_wrong": int(((correct[b] == 1) & (correct[a] == 0)).sum()),
                     "n_resamples": R, "seed": pb["seed"], "n_test": n})
    return pd.DataFrame(rows)


def error_exports(cfg, results: pd.DataFrame, probs: dict, inp_test: dict, out_dir, labels, pair=None, tag=None) -> dict:
    """Error-analysis CSV + markdown samples for a pair (default: the two LM sets with the highest classifier-val accuracy;
    ties keep the table order — the val macro-F1 tie-break written in D22 was never implemented and no tie occurred;
    an explicit `pair` = (A, B) chosen by design writes files suffixed with the pair names)."""
    lm_sets = results[results.kind == "lm"].sort_values(["val_acc"], ascending=False, kind="stable")   # val accuracy only (D22)
    ranking = lm_sets[["feature_set", "val_acc"]].to_dict("records")
    if pair is None:
        A, B = ranking[0]["feature_set"], ranking[1]["feature_set"]
        md_name, json_name, rule = "error_samples.md", "best_pair.json", "highest classifier-val accuracy among the 10 LM feature sets; ties keep the table order (stable sort)"
    else:
        A, B = pair
        md_name, json_name, rule = f"error_samples_{A}_vs_{B}.md", f"pair_{A}_vs_{B}.json", f"pair chosen by design ({tag}), not by results"
    pA, pB = probs[A], probs[B]
    predA, predB = (pA >= 0.5).astype(int), (pB >= 0.5).astype(int)
    df = pd.DataFrame({"test_idx": inp_test["orig_idx"], "label": labels, f"prob_pos_{A}": pA, f"prob_pos_{B}": pB,
                       f"pred_{A}": predA, f"pred_{B}": predB, "review_length_tokens": inp_test["full_len"], "truncated": inp_test["truncated"]})
    df.to_csv(out_dir / f"error_analysis_{A}_vs_{B}.csv", index=False)
    all_test = read_jsonl(ROOT / cfg["processed_dir"] / "test.jsonl")
    test_recs = [all_test[i] for i in inp_test["orig_idx"]]
    groups = {"both wrong": (predA != labels) & (predB != labels), f"A ({A}) right, B ({B}) wrong": (predA == labels) & (predB != labels),
              f"B ({B}) right, A ({A}) wrong": (predB == labels) & (predA != labels)}
    rng = np.random.default_rng(cfg["error_sample_seed"])
    md = [f"# Error-analysis samples: A = `{A}`, B = `{B}` ({rule})\n",
          f"{cfg['error_samples_per_group']} reviews per group sampled with seed {cfg['error_sample_seed']}; first {cfg['error_sample_words']} words of the cleaned text. "
          "Columns: true label (1 = positive), P(positive) under A and B, review length in tokens, truncated flag.\n"]
    counts = {}
    for gname, mask in groups.items():
        idx = np.flatnonzero(mask); counts[gname] = int(len(idx))
        pick = rng.choice(idx, size=min(cfg["error_samples_per_group"], len(idx)), replace=False) if len(idx) else []
        md.append(f"\n## {gname} — {len(idx)} test reviews\n")
        for i in sorted(pick):
            words = test_recs[i]["clean"].split()[: cfg["error_sample_words"]]
            md.append(f"- **test_idx {i}** | label {labels[i]} | P_A {pA[i]:.3f} | P_B {pB[i]:.3f} | length {inp_test['full_len'][i]} | truncated {bool(inp_test['truncated'][i])}\n  > {' '.join(words)}\n")
    (out_dir / md_name).write_text("\n".join(md))
    info = {"A": A, "B": B, "rule": rule, "ranking_by_val_acc": ranking, "group_counts": counts,
            "csv": f"error_analysis_{A}_vs_{B}.csv", "markdown": md_name}
    write_json(info, out_dir / json_name)
    return info


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part3/classify.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=["main", "ablation", "extra_baselines", "exports"], default="main")
    args = ap.parse_args()
    cfg = load_config(args.config, {"limit_reviews": args.limit})   # limit slices the full data (no stream redirect)
    if cfg.get("limit_reviews"):
        cfg["derived_dir"], cfg["results_dir"] = "data/smoke/derived/part3", "results/smoke/part3"
    derived, out_dir = ROOT / cfg["derived_dir"], ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    itos = read_json(ROOT / cfg["streams_dir"] / "vocab.json")["itos"]

    if args.mode == "exports":
        # rebuild the best-pair and design-chosen error exports from results_table.csv and the saved predictions
        inputs = load_inputs(derived, "")
        yte = inputs["test"]["labels"].astype(int)
        results = pd.read_csv(out_dir / "results_table.csv")
        probs = {p.stem: np.load(p) for p in (out_dir / "predictions").glob("*.npy") if not p.stem.endswith("_last255")}
        pair = error_exports(cfg, results, probs, inputs["test"], out_dir, yte)
        print("best pair:", pair["A"], pair["B"], pair["group_counts"])
        for extra in cfg.get("extra_error_pairs", []):
            e = error_exports(cfg, results, probs, inputs["test"], out_dir, yte, pair=tuple(extra), tag="Part II embedding question, D93")
            print("extra error export:", e["A"], "vs", e["B"], e["group_counts"])
        return
    if args.mode == "extra_baselines":
        # fit only the extra embedding-mean baselines, append them to the main tables, recompute the paired bootstrap
        inputs = load_inputs(derived, "")
        ytr_all, yte = inputs["train"]["labels"].astype(int), inputs["test"]["labels"].astype(int)
        tr, va = clf_indices(cfg, inputs["train"]["orig_idx"])
        rng = np.random.default_rng(cfg["bootstrap"]["seed"])
        boot_idx = rng.integers(0, len(yte), size=(cfg["bootstrap"]["n_resamples"], len(yte)))
        robust = robustness_mask(cfg, inputs["test"]["orig_idx"])
        lengths = inputs["test"]["full_len"]
        rows = []
        for name, spec in cfg["extra_embedding_mean_baselines"].items():
            table = load_embedding_table(spec)
            Xtr_all, Xte = glove_mean(inputs["train"], table), glove_mean(inputs["test"], table)
            t0 = time.perf_counter()
            sel, pipe = fit_select("dense", Xtr_all[tr], ytr_all[tr], Xtr_all[va], ytr_all[va], cfg)
            ev, prob = evaluate(pipe, Xte, yte, boot_idx, robust, lengths, cfg["length_buckets"])
            np.save(out_dir / "predictions" / f"{name}.npy", prob.astype(np.float32))
            rows.append({"feature_set": name, "kind": "baseline", "truncate": "first", "dim": Xtr_all.shape[1], "chosen_C": sel["chosen_C"],
                         "val_acc": sel["val_acc"], "val_acc_tie": sel["tie"], **{f"val_acc_C{r['C']:g}": r["val_acc"] for r in sel["per_C"]},
                         **{f"n_iter_C{r['C']:g}": r["n_iter"] for r in sel["per_C"]}, "all_converged": all(r["converged"] for r in sel["per_C"]),
                         "chosen_converged": next(r["converged"] for r in sel["per_C"] if r["C"] == sel["chosen_C"]), **ev, "seconds": time.perf_counter() - t0,
                         "embedding_table": spec})
            print(f"{name:18s} C={sel['chosen_C']:<5g} val {sel['val_acc']:.4f} | test acc {ev['test_acc']:.4f} [{ev['acc_ci95'][0]:.4f},{ev['acc_ci95'][1]:.4f}] F1 {ev['test_macro_f1']:.4f} | conv {rows[-1]['all_converged']}", flush=True)
        new = pd.DataFrame(rows)
        main = pd.read_csv(out_dir / "results_table.csv")
        main = pd.concat([main[~main.feature_set.isin(new.feature_set)], new], ignore_index=True)
        main.to_csv(out_dir / "results_table.csv", index=False)
        lb_cols = ["feature_set"] + [c for c in main.columns if c.startswith("acc_len") or c.startswith("n_len")]
        main[lb_cols].to_csv(out_dir / "length_buckets.csv", index=False)
        probs = {p.stem: np.load(p) for p in (out_dir / "predictions").glob("*.npy") if not p.stem.endswith("_last255")}
        pbdf = paired_bootstrap(cfg, probs, yte, cfg["lms"])
        pbdf.to_csv(out_dir / "paired_bootstrap.csv", index=False)
        print(pbdf[["model_a", "model_b", "acc_diff_a_minus_b", "ci95_lo", "ci95_hi", "ci_excludes_zero"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return
    if args.mode == "ablation":
        pair = read_json(out_dir / "best_pair.json")
        best_lm = pair["A"].rsplit("_", 1)[0]
        suffix, lms = "_last255", {best_lm: cfg["lms"][best_lm]}
    else:
        suffix, lms = "", cfg["lms"]
    inputs = load_inputs(derived, suffix)
    ytr_all, yte = inputs["train"]["labels"].astype(int), inputs["test"]["labels"].astype(int)
    tr, va = clf_indices(cfg, inputs["train"]["orig_idx"])
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    boot_idx = rng.integers(0, len(yte), size=(cfg["bootstrap"]["n_resamples"], len(yte)))
    robust = robustness_mask(cfg, inputs["test"]["orig_idx"])
    lengths = inputs["test"]["full_len"]
    print(f"classifier-train {len(tr)}, classifier-val {len(va)}, test {len(yte)}; D46 subset keeps {robust.sum()} of {len(yte)}", flush=True)

    sets = []
    for lm in lms:
        for pool in cfg["poolings"]:
            sets.append((f"{lm}_{pool}", "lm", lambda lm=lm, pool=pool: (np.load(derived / f"{lm}_{pool}_train{suffix}.npy"), np.load(derived / f"{lm}_{pool}_test{suffix}.npy"))))
    if args.mode == "main":
        sets.append(("tfidf_uni_bi", "tfidf", lambda: (np.asarray(texts_for_tfidf(inputs["train"], itos), dtype=object), np.asarray(texts_for_tfidf(inputs["test"], itos), dtype=object))))
        table = np.load(ROOT / cfg["glove_table"]).astype(np.float32)
        sets.append(("glove_mean", "baseline", lambda: (glove_mean(inputs["train"], table), glove_mean(inputs["test"], table))))

    rows, probs = [], {}
    for name, kind, loader in sets:
        if kind == "lm" and not (derived / f"{name}_train{suffix}.npy").exists():
            print(f"!! features missing for {name} (checkpoint not trained yet?); skipped", flush=True); continue
        Xtr_all, Xte = loader()
        t0 = time.perf_counter()
        sel, pipe = fit_select("tfidf" if kind == "tfidf" else "dense", Xtr_all[tr], ytr_all[tr], Xtr_all[va], ytr_all[va], cfg)
        ev, prob = evaluate(pipe, Xte, yte, boot_idx, robust, lengths, cfg["length_buckets"])
        probs[name] = prob
        np.save(out_dir / "predictions" / f"{name}{suffix}.npy", prob.astype(np.float32))
        row = {"feature_set": name, "kind": kind, "truncate": "last" if suffix else "first", "dim": (Xtr_all.shape[1] if kind != "tfidf" else cfg["tfidf"]["max_features"]),
               "chosen_C": sel["chosen_C"], "val_acc": sel["val_acc"], "val_acc_tie": sel["tie"],
               **{f"val_acc_C{r['C']:g}": r["val_acc"] for r in sel["per_C"]},
               **{f"n_iter_C{r['C']:g}": r["n_iter"] for r in sel["per_C"]},
               "all_converged": all(r["converged"] for r in sel["per_C"]),
               "chosen_converged": next(r["converged"] for r in sel["per_C"] if r["C"] == sel["chosen_C"]),
               **ev, "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(f"{name:18s} C={row['chosen_C']:<5g} val {row['val_acc']:.4f} | test acc {row['test_acc']:.4f} [{row['acc_ci95'][0]:.4f},{row['acc_ci95'][1]:.4f}] "
              f"F1 {row['test_macro_f1']:.4f} | robust acc {row['robust_acc']:.4f} | conv {row['all_converged']} ({row['seconds']:.0f}s)", flush=True)
    results = pd.DataFrame(rows)

    if args.mode == "ablation":
        first = pd.read_csv(out_dir / "results_table.csv")
        first = first[first.feature_set.isin(results.feature_set)]
        abl = pd.concat([first.assign(truncate="first"), results], ignore_index=True)
        abl.to_csv(out_dir / "ablation_last255.csv", index=False)
        print(abl[["feature_set", "truncate", "chosen_C", "val_acc", "test_acc", "test_macro_f1"]].to_string(index=False))
        return

    results.to_csv(out_dir / "results_table.csv", index=False)
    lm_vs_downstream(results, cfg).to_csv(out_dir / "lm_vs_downstream.csv", index=False)
    lb_cols = ["feature_set"] + [c for c in results.columns if c.startswith("acc_len") or c.startswith("n_len")]
    results[lb_cols].to_csv(out_dir / "length_buckets.csv", index=False)
    pair = error_exports(cfg, results, probs, inputs["test"], out_dir, yte)
    for extra in cfg.get("extra_error_pairs", []):
        e = error_exports(cfg, results, probs, inputs["test"], out_dir, yte, pair=tuple(extra), tag="Part II embedding question, D93")
        print("extra error export:", e["A"], "vs", e["B"], e["group_counts"])
    if "paired_bootstrap" in cfg:
        pbdf = paired_bootstrap(cfg, probs, yte, lms)
        pbdf.to_csv(out_dir / "paired_bootstrap.csv", index=False)
        print(pbdf[["model_a", "model_b", "acc_diff_a_minus_b", "ci95_lo", "ci95_hi", "ci_excludes_zero"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    write_json({"classifier_train": int(len(tr)), "classifier_val": int(len(va)), "test": int(len(yte)), "robust_subset_n": int(robust.sum()),
                "robust_excluded_n": int((~robust).sum()), "bootstrap": cfg["bootstrap"], "C_grid": cfg["classifier"]["C_grid"],
                "tie_rule": cfg["classifier"]["tie_rule"], "best_pair": pair["A"] + " / " + pair["B"]}, out_dir / "classify_meta.json")
    print("best pair:", pair["A"], pair["B"], "| group counts:", pair["group_counts"])


if __name__ == "__main__":
    main()
