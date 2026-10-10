"""Data pipeline (TASK §2): download IMDb, clean + tokenise, de-duplicate, LM splits, classifier split.

Outputs (under cfg.processed_dir):
  {train,test,unsupervised}.jsonl   one record per review: idx, label, clean, tokens (space-joined)
  lm_split.json                     indices (into unsupervised.jsonl) of LM-train / LM-val / LM-test
  clf_split.json                    indices (into train.jsonl) of classifier-train / classifier-val
Outputs (under cfg.results_dir):
  data_stats.json                   raw sizes, de-dup counts, split sizes (vocab.py adds more)
  tokenisation_examples.md          5 random reviews raw vs tokenised
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.common import ROOT, apply_limit, load_config, write_json
from src.tokenize import clean_and_tokenize


_NONASCII_RE = re.compile(r"[^\x00-\x7f]")
_TAG_RE = re.compile(r"<[a-z/][^>]*>")
_MOJIBAKE = ("ã©", "ã¨", "ã¡", "ã³", "ã±", "ã¶", "ã¼", "â€™", "â€œ", "â€“")


class Census:
    """Character census: non-ASCII characters before/after cleaning, residual tag-like patterns, mojibake."""

    def __init__(self):
        self.raw_chars, self.clean_chars, self.tags, self.mojibake = Counter(), Counter(), Counter(), Counter()
        self.reviews = self.reviews_nonascii_raw = self.reviews_nonascii_clean = self.reviews_with_tag = 0

    def add(self, raw: str, clean: str) -> None:
        self.reviews += 1
        r = _NONASCII_RE.findall(raw)
        c = _NONASCII_RE.findall(clean)
        self.raw_chars.update(r)
        self.clean_chars.update(c)
        self.reviews_nonascii_raw += bool(r)
        self.reviews_nonascii_clean += bool(c)
        tags = _TAG_RE.findall(clean)
        if tags:
            self.reviews_with_tag += 1
            self.tags.update(re.sub(r"\s+", " ", t)[:40] for t in tags)
        for pat in _MOJIBAKE:
            k = clean.count(pat)
            if k:
                self.mojibake[pat] += k

    @staticmethod
    def _top(counter: Counter, k: int = 30) -> list[dict]:
        return [{"char": ch, "codepoint": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "<control>"),
                 "category": unicodedata.category(ch), "count": n} for ch, n in counter.most_common(k)]

    def summary(self) -> dict:
        return {
            "reviews": self.reviews,
            "non_ascii_raw": {"distinct_chars": len(self.raw_chars), "occurrences": sum(self.raw_chars.values()),
                              "reviews_affected": self.reviews_nonascii_raw, "top30": self._top(self.raw_chars)},
            "non_ascii_after_cleaning": {"distinct_chars": len(self.clean_chars), "occurrences": sum(self.clean_chars.values()),
                                         "reviews_affected": self.reviews_nonascii_clean, "top30": self._top(self.clean_chars)},
            "control_or_format_chars_after_cleaning": sum(n for ch, n in self.clean_chars.items()
                                                          if unicodedata.category(ch) in ("Cc", "Cf")),
            "tag_like_patterns_after_cleaning": {"reviews_affected": self.reviews_with_tag,
                                                 "reviews_affected_pct": round(100 * self.reviews_with_tag / max(1, self.reviews), 4),
                                                 "top10": self.tags.most_common(10)},
            "mojibake_sequences_after_cleaning": dict(self.mojibake),
        }


def process_split(ds, name: str, out_path: Path, limit: int | None, census: Census) -> list[dict]:
    n = len(ds) if not limit else min(limit, len(ds))
    records = []
    with open(out_path, "w") as f:
        for i in tqdm(range(n), desc=f"tokenise {name}", mininterval=5):
            ex = ds[i]
            clean, toks = clean_and_tokenize(ex["text"])
            census.add(ex["text"], clean)
            rec = {"idx": i, "label": int(ex["label"]), "clean": clean, "tokens": " ".join(toks)}
            f.write(json.dumps(rec) + "\n")
            records.append(rec)
    return records


def dedup(unsup: list[dict], train: list[dict], test: list[dict]) -> tuple[list[int], dict]:
    """Return the kept unsupervised indices and de-duplication counts (TASK §2 De-duplication)."""
    train_set = {r["clean"] for r in train}
    test_set = {r["clean"] for r in test}
    labelled = train_set | test_set
    seen: set[str] = set()
    kept, n_in_labelled, n_within = [], 0, 0
    for r in unsup:
        c = r["clean"]
        if c in labelled:
            n_in_labelled += 1
            continue
        if c in seen:
            n_within += 1
            continue
        seen.add(c)
        kept.append(r["idx"])
    train_counts = Counter(r["clean"] for r in train)
    test_counts = Counter(r["clean"] for r in test)
    stats = {
        "unsupervised_raw": len(unsup),
        "unsupervised_removed_in_labelled_train_or_test": n_in_labelled,
        "unsupervised_removed_exact_duplicates_within": n_within,
        "unsupervised_kept": len(kept),
        "labelled_train_test_shared_duplicates_distinct_texts": len(train_set & test_set),
        "labelled_train_test_shared_duplicates_review_pairs": sum(
            train_counts[c] * test_counts[c] for c in train_set & test_set),
        "labelled_train_within_duplicate_reviews": len(train) - len(train_set),
        "labelled_test_within_duplicate_reviews": len(test) - len(test_set),
        "note": "labelled splits are not modified; duplicates in them are only reported",
    }
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke mode: first N reviews per split")
    args = ap.parse_args()
    cfg = apply_limit(load_config(args.config, {"limit_reviews": args.limit}))
    limit = cfg.get("limit_reviews")
    t0 = time.time()

    processed = ROOT / cfg["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    results = ROOT / cfg["results_dir"]
    results.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg["hf_name"], cache_dir=str(ROOT / cfg["hf_cache"]))
    raw_sizes = {k: len(ds[k]) for k in ds}
    census = Census()
    splits = {name: process_split(ds[name], name, processed / f"{name}.jsonl", limit, census)
              for name in ("train", "test", "unsupervised")}

    # --- de-duplication and LM splits (by review, never by sentence)
    kept, dd = dedup(splits["unsupervised"], splits["train"], splits["test"])
    rng = np.random.default_rng(cfg["split_seed"])
    perm = rng.permutation(len(kept))
    kept_arr = np.asarray(kept)[perm]
    n_val, n_test = cfg["lm_val_reviews"], cfg["lm_test_reviews"]
    if limit:  # smoke: scale the held-out sizes down
        n_val = n_test = max(1, len(kept_arr) // 10)
    lm_split = {
        "lm_val": kept_arr[:n_val].tolist(),
        "lm_test": kept_arr[n_val:n_val + n_test].tolist(),
        "lm_train": kept_arr[n_val + n_test:].tolist(),
    }
    write_json(lm_split, processed / "lm_split.json")

    # --- classifier split of IMDb train (stratified, seed 0)
    labels = np.array([r["label"] for r in splits["train"]])
    idx = np.arange(len(labels))
    n_clf_val = cfg["classifier_val_reviews"] if not limit else max(2, len(labels) // 10)
    tr_idx, va_idx = train_test_split(idx, test_size=n_clf_val, stratify=labels,
                                      random_state=cfg["classifier_split_seed"])
    write_json({"clf_train": sorted(tr_idx.tolist()), "clf_val": sorted(va_idx.tolist())},
               processed / "clf_split.json")

    # --- statistics
    unsup_by_idx = {r["idx"]: r for r in splits["unsupervised"]}
    def n_tokens(ids):
        return int(sum(len(unsup_by_idx[i]["tokens"].split()) for i in ids))
    stats = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "hf_raw_sizes": raw_sizes,
        "processed_sizes": {k: len(v) for k, v in splits.items()},
        "dedup": dd,
        "lm_splits_reviews": {k: len(v) for k, v in lm_split.items()},
        "lm_splits_review_tokens_excl_eos": {k: n_tokens(v) for k, v in lm_split.items()},
        "classifier_split_reviews": {"clf_train": len(tr_idx), "clf_val": len(va_idx),
                                     "clf_val_pos_fraction": float(labels[va_idx].mean())},
        "imdb_review_tokens": {k: int(sum(len(r["tokens"].split()) for r in splits[k])) for k in ("train", "test")},
        "character_census_all_splits": census.summary(),
        "data_py_seconds": round(time.time() - t0, 1),
    }
    write_json(stats, results / "data_stats.json")

    # --- tokenisation examples
    ex_rng = np.random.default_rng(cfg["example_seed"])
    ex_ids = ex_rng.choice(len(splits["unsupervised"]), size=min(cfg["n_examples"], len(splits["unsupervised"])), replace=False)
    lines = ["# Tokenisation examples (unsupervised split, seed %d)\n" % cfg["example_seed"]]
    for i in ex_ids:
        raw = ds["unsupervised"][int(i)]["text"]
        rec = splits["unsupervised"][int(i)]
        lines += [f"## Review idx {int(i)}\n", "**Raw**\n", "```", raw, "```\n",
                  f"**Tokenised** ({len(rec['tokens'].split())} tokens)\n", "```", rec["tokens"], "```\n"]
    (results / "tokenisation_examples.md").write_text("\n".join(lines))

    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=1))
    print("wrote", processed, "and", results)


if __name__ == "__main__":
    main()
