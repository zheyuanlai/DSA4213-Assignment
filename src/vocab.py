"""Vocabulary and token streams (TASK §2 Vocabulary / Streams).

Vocabulary: the 25,000 most frequent LM-train tokens (ties broken alphabetically) plus the specials
<pad>=0, <unk>=1, <eos>=2 → |V| = 25,003. Streams: <eos>, then each review followed by <eos>, as
int32 .npy. IMDb train/test are stored per review (ragged: ids + offsets + labels) for Part III.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.common import EOS, PAD, UNK, ROOT, apply_limit, load_config, read_json, write_json


def read_jsonl_tokens(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class Vocab:
    def __init__(self, itos: list[str], counts: list[int]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.counts = np.asarray(counts, dtype=np.int64)  # LM-train counts (0 for specials except <eos>)
        assert self.itos[PAD] == "<pad>" and self.itos[UNK] == "<unk>" and self.itos[EOS] == "<eos>"

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        s = self.stoi
        return [s.get(t, UNK) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        d = read_json(path)
        return cls(d["itos"], d["counts"])

    def save(self, path: str | Path) -> None:
        write_json({"itos": self.itos, "counts": self.counts.tolist()}, path)


def build_vocab(train_token_lists: list[list[str]], size: int, specials: list[str]) -> Vocab:
    counter: Counter[str] = Counter()
    for toks in train_token_lists:
        counter.update(toks)
    # deterministic: most frequent first, ties alphabetical
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:size]
    itos = list(specials) + [t for t, _ in ranked]
    counts = [0] * len(specials) + [c for _, c in ranked]
    n_eos = len(train_token_lists) + 1  # <eos> at the stream start and after every review
    counts[EOS] = n_eos
    return Vocab(itos, counts)


def make_stream(vocab: Vocab, token_lists: list[list[str]]) -> np.ndarray:
    out = [EOS]
    for toks in token_lists:
        out.extend(vocab.encode(toks))
        out.append(EOS)
    return np.asarray(out, dtype=np.int32)


def unk_rate(vocab: Vocab, token_lists: list[list[str]]) -> dict:
    n = sum(len(t) for t in token_lists)
    unk = sum(1 for toks in token_lists for t in toks if t not in vocab.stoi)
    return {"tokens": int(n), "unk": int(unk), "unk_rate": (unk / n) if n else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = apply_limit(load_config(args.config, {"limit_reviews": args.limit}))
    t0 = time.time()
    processed = ROOT / cfg["processed_dir"]
    streams = ROOT / cfg["streams_dir"]
    streams.mkdir(parents=True, exist_ok=True)
    results = ROOT / cfg["results_dir"]

    unsup = read_jsonl_tokens(processed / "unsupervised.jsonl")
    by_idx = {r["idx"]: r["tokens"].split() for r in unsup}
    lm_split = read_json(processed / "lm_split.json")
    lm = {k: [by_idx[i] for i in v] for k, v in lm_split.items()}

    vocab = build_vocab(lm["lm_train"], cfg["vocab_size"], cfg["specials"])
    vocab.save(streams / "vocab.json")
    with open(results / "vocab.tsv", "w") as f:
        f.write("id\ttoken\tlm_train_count\n")
        for i, (t, c) in enumerate(zip(vocab.itos, vocab.counts)):
            f.write(f"{i}\t{t}\t{int(c)}\n")

    stream_stats = {}
    for k in ("lm_train", "lm_val", "lm_test"):
        s = make_stream(vocab, lm[k])
        np.save(streams / f"{k}.npy", s)
        stream_stats[k] = {"reviews": len(lm[k]), "stream_tokens_incl_eos": int(len(s)),
                           "unk_targets_in_stream": int((s == UNK).sum())}

    imdb = {}
    for k in ("train", "test"):
        recs = read_jsonl_tokens(processed / f"{k}.jsonl")
        toks = [r["tokens"].split() for r in recs]
        ids = [np.asarray(vocab.encode(t), dtype=np.int32) for t in toks]
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in ids])
        np.savez(streams / f"imdb_{k}.npz", ids=np.concatenate(ids) if ids else np.zeros(0, np.int32),
                 offsets=offsets, labels=np.asarray([r["label"] for r in recs], dtype=np.int8))
        imdb[k] = toks

    unk = {
        "lm_train": unk_rate(vocab, lm["lm_train"]),
        "lm_val": unk_rate(vocab, lm["lm_val"]),
        "lm_test": unk_rate(vocab, lm["lm_test"]),
        "imdb_train": unk_rate(vocab, imdb["train"]),
        "imdb_test": unk_rate(vocab, imdb["test"]),
    }
    stats = read_json(results / "data_stats.json")
    stats["vocab"] = {"size_incl_specials": len(vocab), "specials": {t: i for i, t in enumerate(cfg["specials"])},
                      "distinct_lm_train_types": None, "min_count_in_vocab": int(vocab.counts[3:].min()) if len(vocab) > 3 else None,
                      "last_vocab_token": vocab.itos[-1]}
    # distinct types in LM-train (for the record)
    stats["vocab"]["distinct_lm_train_types"] = len({t for toks in lm["lm_train"] for t in toks})
    stats["unk_rates_token_level_excl_eos"] = unk
    stats["streams"] = stream_stats
    stats["vocab_py_seconds"] = round(time.time() - t0, 1)
    write_json(stats, results / "data_stats.json")
    print(json.dumps({"vocab": stats["vocab"], "unk": unk, "streams": stream_stats}, indent=1))


if __name__ == "__main__":
    main()
