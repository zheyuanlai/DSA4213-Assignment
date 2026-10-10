"""Follow-up quantitative checks requested after the overnight run (NOTES D97–D99); existing outputs only.

1. GloVe-uncovered words in Part III: IMDb-test split by whether the model input contains >= 1 of the
   310 vocabulary words missing from GloVe; accuracy of lstm_mean / lstm_own_mean / lstm_glove_mean per
   subset and a paired-bootstrap 95 % CI of (lstm_own_mean - lstm_glove_mean).
2. Truncation and borderline cases in the error groups of each error-analysis CSV.
3. Bracket and quote closure in the 100 sampled generations per model (Part I and Part II).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import ROOT, read_json, write_json


def paired_ci(ca: np.ndarray, cb: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = (ca.astype(np.int8) - cb.astype(np.int8)).astype(np.float64)
    n = len(d)
    vals = np.empty(n_resamples)
    for s in range(0, n_resamples, 500):
        idx = rng.integers(0, n, size=(min(500, n_resamples - s), n))
        vals[s:s + idx.shape[0]] = d[idx].mean(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def glove_uncovered_subsets(part3: str = "results/part3") -> pd.DataFrame:
    covered = np.load(ROOT / "data/embeddings/glove_300_covered_mask.npy")
    uncovered_ids = np.flatnonzero(~covered)
    uncovered_ids = uncovered_ids[uncovered_ids > 2]                  # words only (310)
    inp = np.load(ROOT / "data/derived/part3/inputs_test.npz")
    X, n_input, y = inp["ids"], inp["n_input"], inp["labels"].astype(int)
    mask_pos = np.arange(X.shape[1])[None, :] < n_input[:, None]
    has = np.isin(X, uncovered_ids) & mask_pos
    contains = has.any(axis=1)
    rows = []
    preds = {m: (np.load(ROOT / part3 / "predictions" / f"{m}.npy") >= 0.5).astype(int) for m in ("lstm_mean", "lstm_own_mean", "lstm_glove_mean")}
    for name, sel in (("contains >= 1 uncovered word", contains), ("no uncovered word", ~contains), ("all test reviews", np.ones_like(contains))):
        row = {"subset": name, "n": int(sel.sum()), "n_uncovered_vocab_words": int(len(uncovered_ids)),
               "mean_uncovered_tokens_per_review": float(has[sel].sum(axis=1).mean())}
        for m, p in preds.items():
            row[f"acc_{m}"] = float((p[sel] == y[sel]).mean())
        lo, hi = paired_ci(preds["lstm_own_mean"][sel] == y[sel], preds["lstm_glove_mean"][sel] == y[sel])
        row["diff_own_minus_glove"] = row["acc_lstm_own_mean"] - row["acc_lstm_glove_mean"]
        row["diff_ci95_lo"], row["diff_ci95_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def error_group_truncation(csv_path: str, overall_truncated_share: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / csv_path)
    pa_col = next(c for c in df.columns if c.startswith("prob_pos_"))
    pb_col = [c for c in df.columns if c.startswith("prob_pos_")][1]
    A, B = pa_col[len("prob_pos_"):], pb_col[len("prob_pos_"):]
    predA, predB, y = df[f"pred_{A}"].values, df[f"pred_{B}"].values, df["label"].values
    pA, pB = df[pa_col].values, df[pb_col].values
    overall = float(df["truncated"].mean()) if overall_truncated_share is None else overall_truncated_share
    groups = {"both wrong": (predA != y) & (predB != y), f"A right, B wrong": (predA == y) & (predB != y),
              f"B right, A wrong": (predB == y) & (predA != y), "both right": (predA == y) & (predB == y)}
    rows = []
    for g, sel in groups.items():
        row = {"pair": f"A={A} vs B={B}", "group": g, "n": int(sel.sum()), "truncated_share": float(df["truncated"][sel].mean()),
               "overall_test_truncated_share": overall}
        if g == "A right, B wrong":
            row["wrong_model_borderline_share_0.4_0.6"] = float(((pB[sel] >= 0.4) & (pB[sel] <= 0.6)).mean())
        elif g == "B right, A wrong":
            row["wrong_model_borderline_share_0.4_0.6"] = float(((pA[sel] >= 0.4) & (pA[sel] <= 0.6)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bracket_quote_closure(jsonl_path: str, part: str) -> pd.DataFrame:
    g = pd.read_json(ROOT / jsonl_path, lines=True)
    g = g[g["mode"] == "sample"]
    rows = []
    for model, d in g.groupby("model", sort=False):
        opened = closed = 0
        even_quotes = 0
        for toks in d["tokens"]:
            t = toks.split(" ")
            stack = []                      # indices of unmatched "("
            closed_flags = {}
            for i, tok in enumerate(t):
                if tok == "(":
                    stack.append(i); closed_flags[i] = False
                elif tok == ")" and stack:
                    closed_flags[stack.pop()] = True
            early = [i for i in closed_flags if i < 40]
            opened += len(early); closed += sum(closed_flags[i] for i in early)
            even_quotes += (t.count('"') % 2 == 0)
        rows.append({"part": part, "model": model, "n_samples": int(len(d)), "open_parens_first40": opened,
                     "fraction_closed_later": (closed / opened) if opened else None,
                     "fraction_samples_even_double_quotes": even_quotes / len(d)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["glove_subsets", "error_groups", "brackets", "all"])
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    if args.what in ("glove_subsets", "all"):
        df = glove_uncovered_subsets(); df.to_csv(ROOT / "results/part3/glove_uncovered_subsets.csv", index=False)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.what in ("error_groups", "all"):
        stats = read_json("results/part3/input_stats.json")["test"]["pct_truncated"] / 100
        frames = []
        for p in sorted((ROOT / "results/part3").glob("error_analysis_*_vs_*.csv")):
            if "Cgrid4" in p.name:
                continue
            frames.append(error_group_truncation(str(p.relative_to(ROOT)), stats))
        df = pd.concat(frames, ignore_index=True); df.to_csv(ROOT / "results/part3/error_group_truncation.csv", index=False)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.what in ("brackets", "all"):
        df = pd.concat([bracket_quote_closure("results/part1/generations_all.jsonl", "Part I"),
                        bracket_quote_closure("results/part2/generations_all.jsonl", "Part II")], ignore_index=True)
        df.to_csv(ROOT / "results/part1/generation_brackets.csv", index=False)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
