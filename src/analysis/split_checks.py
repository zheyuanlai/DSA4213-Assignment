"""Split-integrity diagnostics added after an external code review (NOTES D122).

overlaps()      Index and cleaned-text intersections between LM-train / LM-val / LM-test, the labelled IMDb splits,
                and classifier-train / classifier-val / IMDb-test -> results/part3/split_overlap.json.
sensitivity()   26 classifier-val reviews have an identical (same-label) copy in classifier-train, so classifier-val
                accuracy could be optimistic for C selection. For every feature set this refits the classifiers with the
                unchanged procedure and checks whether the chosen C changes when those reviews (and, separately, also
                the val reviews whose text occurs in IMDb-test) are removed from classifier-val. The final classifier
                depends only on classifier-train and the chosen C, so an unchanged choice means the overlap did not
                influence any reported test number. Refits are not bit-reproducible (multi-threaded lbfgs), so the
                committed choice is listed next to the refit choice. -> results/part3/val_dedup_sensitivity.csv
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.common import ROOT, load_config, read_json, read_jsonl, write_json


def _texts(split: str) -> list[dict]:
    return read_jsonl(ROOT / "data/processed" / f"{split}.jsonl")


def overlaps() -> dict:
    train, test, unsup = _texts("train"), _texts("test"), _texts("unsupervised")
    lm = read_json(ROOT / "data/processed/lm_split.json")
    clf = read_json(ROOT / "data/processed/clf_split.json")
    utext = {r["idx"]: r["clean"] for r in unsup}
    lm_sets = {k: set(v) for k, v in lm.items()}
    lm_text = {k: {utext[i] for i in v} for k, v in lm.items()}
    train_text, test_text = {r["clean"] for r in train}, {r["clean"] for r in test}
    groups = defaultdict(list)
    for r in train:
        groups[r["clean"]].append(r["idx"])
    tr, va = set(clf["clf_train"]), set(clf["clf_val"])
    split_groups = [ids for ids in groups.values() if len(ids) > 1 and tr & set(ids) and va & set(ids)]
    labels = {r["idx"]: r["label"] for r in train}
    val_dup_train = sorted(i for ids in split_groups for i in ids if i in va)
    val_dup_test = sorted(i for i in va if train[i]["clean"] in test_text)
    return {
        "lm_splits": {
            "sizes": {k: len(v) for k, v in lm_sets.items()},
            "index_overlaps": {f"{a}&{b}": len(lm_sets[a] & lm_sets[b]) for a, b in (("lm_train", "lm_val"), ("lm_train", "lm_test"), ("lm_val", "lm_test"))},
            "text_overlaps": {f"{a}&{b}": len(lm_text[a] & lm_text[b]) for a, b in (("lm_train", "lm_val"), ("lm_train", "lm_test"), ("lm_val", "lm_test"))},
            "text_overlap_with_labelled_train_or_test": {k: len(v & (train_text | test_text)) for k, v in lm_text.items()},
        },
        "labelled": {
            "imdb_train_duplicate_text_groups": sum(1 for v in groups.values() if len(v) > 1),
            "imdb_train_extra_copies": sum(len(v) - 1 for v in groups.values()),
            "imdb_test_extra_copies": len(test) - len(test_text),
            "imdb_train_test_shared_texts": len(train_text & test_text),
            "imdb_test_reviews_with_text_in_imdb_train": sum(1 for r in test if r["clean"] in train_text),
        },
        "classifier_split": {
            "sizes": {"clf_train": len(tr), "clf_val": len(va)},
            "index_overlap": len(tr & va), "union": len(tr | va),
            "duplicate_groups_split_across_train_and_val": len(split_groups),
            "clf_val_reviews_with_text_in_clf_train": len(val_dup_train),
            "labels_agree_within_split_groups": all(len({labels[i] for i in ids}) == 1 for ids in split_groups),
            "clf_val_reviews_with_text_in_imdb_test": len(val_dup_test),
            "clf_train_reviews_with_text_in_imdb_test": sum(1 for i in tr if train[i]["clean"] in test_text),
            "clf_val_idx_with_text_in_clf_train": val_dup_train,
            "clf_val_idx_with_text_in_imdb_test": val_dup_test,
        },
    }


def sensitivity(ov: dict) -> pd.DataFrame:
    from src.classify import (clf_indices, fit_select, glove_mean, load_embedding_table, load_inputs, make_pipeline,
                              texts_for_tfidf)
    from sklearn.metrics import accuracy_score
    cfg = load_config("configs/part3/classify.yaml")
    derived = ROOT / cfg["derived_dir"]
    inputs = load_inputs(derived, "")
    y = inputs["train"]["labels"].astype(int)
    tr, va = clf_indices(cfg, inputs["train"]["orig_idx"])
    orig = inputs["train"]["orig_idx"]
    drop_train = set(ov["classifier_split"]["clf_val_idx_with_text_in_clf_train"])
    drop_both = drop_train | set(ov["classifier_split"]["clf_val_idx_with_text_in_imdb_test"])
    keep1 = np.array([orig[p] not in drop_train for p in va]); keep2 = np.array([orig[p] not in drop_both for p in va])
    itos = read_json(ROOT / cfg["streams_dir"] / "vocab.json")["itos"]
    committed = pd.read_csv(ROOT / cfg["results_dir"] / "results_table.csv").set_index("feature_set")
    sets = [(f"{lm}_{pool}", "dense", lambda lm=lm, pool=pool: np.load(derived / f"{lm}_{pool}_train.npy"))
            for lm in cfg["lms"] for pool in cfg["poolings"]]
    sets.append(("tfidf_uni_bi", "tfidf", lambda: np.asarray(texts_for_tfidf(inputs["train"], itos), dtype=object)))
    tables = {"glove_mean": cfg["glove_table"], **cfg.get("extra_embedding_mean_baselines", {})}
    for name, spec in tables.items():
        sets.append((name, "dense", lambda spec=spec: glove_mean(inputs["train"], load_embedding_table(spec))))
    grid = cfg["classifier"]["C_grid"]
    pick = lambda accs: min(c for c, a in zip(grid, accs) if a == max(accs))           # ties -> smaller C (D80)
    rows = []
    for name, kind, loader in sets:
        X = loader(); t0 = time.perf_counter()
        full, ex1, ex2 = [], [], []
        for C in grid:
            pipe = make_pipeline(kind, C, cfg).fit(X[tr], y[tr])
            pred = pipe.predict(X[va])
            full.append(accuracy_score(y[va], pred)); ex1.append(accuracy_score(y[va][keep1], pred[keep1])); ex2.append(accuracy_score(y[va][keep2], pred[keep2]))
        row = {"feature_set": name, "committed_chosen_C": float(committed.loc[name, "chosen_C"]),
               "refit_chosen_C_full_val": pick(full), "chosen_C_excl_val_dups_of_train": pick(ex1),
               "chosen_C_excl_val_dups_of_train_or_test": pick(ex2),
               "n_val_full": int(len(va)), "n_val_excl_train_dups": int(keep1.sum()), "n_val_excl_train_or_test_dups": int(keep2.sum())}
        row["choice_unchanged_excl_train_dups"] = row["chosen_C_excl_val_dups_of_train"] == row["refit_chosen_C_full_val"]
        row["choice_unchanged_excl_train_or_test_dups"] = row["chosen_C_excl_val_dups_of_train_or_test"] == row["refit_chosen_C_full_val"]
        row["refit_matches_committed"] = row["refit_chosen_C_full_val"] == row["committed_chosen_C"]
        for C, a, b, c in zip(grid, full, ex1, ex2):
            row[f"val_acc_full_C{C:g}"], row[f"val_acc_excl_train_dups_C{C:g}"], row[f"val_acc_excl_both_C{C:g}"] = a, b, c
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        print(f"{name:16s} committed C={row['committed_chosen_C']:<6g} refit C={row['refit_chosen_C_full_val']:<6g} "
              f"excl train dups C={row['chosen_C_excl_val_dups_of_train']:<6g} excl train+test dups C={row['chosen_C_excl_val_dups_of_train_or_test']:<6g} "
              f"({row['seconds']:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["overlaps", "sensitivity", "all"])
    args = ap.parse_args()
    ov = overlaps()
    if args.what in ("overlaps", "all"):
        write_json(ov, "results/part3/split_overlap.json")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("_idx_with_text_in_clf_train") and not kk.endswith("_idx_with_text_in_imdb_test")} for k, v in ov.items()}, indent=1))
    if args.what in ("sensitivity", "all"):
        df = sensitivity(ov)
        df.to_csv(ROOT / "results/part3/val_dedup_sensitivity.csv", index=False)
        print("\nchoice unchanged for every set (excl. train dups):", bool(df.choice_unchanged_excl_train_dups.all()),
              "| (excl. train+test dups):", bool(df.choice_unchanged_excl_train_or_test_dups.all()))


if __name__ == "__main__":
    main()
