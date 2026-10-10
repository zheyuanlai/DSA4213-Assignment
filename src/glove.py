"""Public GloVe 6B 300d table for Part II setting C (TASK §5).

Coverage is reported as the % of vocabulary word types and the % of LM-train tokens found in GloVe,
with the 30 most frequent uncovered words. Missing words are initialised from N(0, sigma_d^2) per
dimension, where sigma_d is the per-dimension std of the covered GloVe vectors (seeded). Special
tokens: <pad> = zeros; <unk> and <eos> are treated as missing words (random init as above) even if a
literal "<unk>" string exists in the GloVe file (recorded in the JSON).
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile

import numpy as np

from src.common import EOS, PAD, UNK, ROOT, load_config, read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2/glove.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    vocab = read_json(ROOT / cfg["streams_dir"] / "vocab.json")
    itos, counts = vocab["itos"], np.asarray(vocab["counts"], dtype=np.int64)
    V = len(itos)
    stoi = {t: i for i, t in enumerate(itos)}
    d = 300
    table = np.zeros((V, d), dtype=np.float32)
    covered = np.zeros(V, dtype=bool)
    literal_specials_in_file = {}
    t0 = time.perf_counter()
    n_lines = 0
    with zipfile.ZipFile(ROOT / cfg["zip_path"]) as zf, zf.open(cfg["member"]) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            n_lines += 1
            sp = line.rstrip("\n").split(" ")
            w = sp[0]
            if w in ("<unk>", "<pad>", "<eos>", "<s>", "</s>"):
                literal_specials_in_file[w] = True
            i = stoi.get(w)
            if i is not None and i > EOS:          # words only; specials handled below
                table[i] = np.asarray(sp[1:], dtype=np.float32)
                covered[i] = True
    read_s = time.perf_counter() - t0
    words = np.arange(V) > EOS
    sigma = table[covered].std(axis=0)             # per-dimension std of the covered vectors
    rng = np.random.default_rng(cfg["init_seed"])
    missing = ~covered
    missing[PAD] = False
    for i in np.flatnonzero(missing):              # includes <unk> and <eos>
        table[i] = rng.normal(0.0, sigma).astype(np.float32)
    table[PAD] = 0.0

    word_counts = counts.copy(); word_counts[:3] = 0
    tok_total_words = int(word_counts.sum())                # LM-train word tokens (excl. specials)
    tok_covered = int(word_counts[covered].sum())
    unk_tokens = int(counts[UNK])
    uncovered_ids = [i for i in np.argsort(-word_counts) if words[i] and not covered[i]][: cfg["n_uncovered_listed"]]
    info = {
        "source": "glove.6B.300d (Wikipedia 2014 + Gigaword 5, 6B tokens, 400k uncased vocabulary)", "zip": cfg["zip_path"],
        "glove_lines_read": n_lines, "read_seconds": read_s,
        "coverage": {
            "vocab_word_types": int(words.sum()), "covered_types": int(covered.sum()),
            "pct_types_covered": 100.0 * covered.sum() / words.sum(),
            "lm_train_word_tokens_excl_specials": tok_total_words, "covered_tokens": tok_covered,
            "pct_lm_train_word_tokens_covered": 100.0 * tok_covered / tok_total_words,
            "pct_lm_train_tokens_covered_counting_unk_as_uncovered": 100.0 * tok_covered / (tok_total_words + unk_tokens),
            "unk_tokens_in_lm_train": unk_tokens,
        },
        "most_frequent_uncovered_words": [{"word": itos[i], "lm_train_count": int(counts[i])} for i in uncovered_ids],
        "n_missing_words_random_init": int(missing.sum() - 2), "init": {
            "rule": "missing word w ~ N(0, sigma_d^2) independently per dimension d; sigma_d = std over covered vectors of dimension d; numpy default_rng(init_seed)",
            "init_seed": cfg["init_seed"], "sigma_mean_over_dims": float(sigma.mean()), "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()),
            "<pad>": "zeros", "<unk>": "random init as a missing word (not a GloVe word)", "<eos>": "random init as a missing word (not a GloVe word)",
            "literal_special_strings_present_in_glove_file": literal_specials_in_file},
        "per_dim_std_mean": {"covered_vectors": float(sigma.mean()), "random_init_vectors": float(table[missing].std(axis=0).mean()) if missing.any() else None,
                             "whole_table_excl_pad": float(table[1:].std(axis=0).mean())},
        "mean_vector_norm": {"covered": float(np.linalg.norm(table[covered], axis=1).mean()), "random_init": float(np.linalg.norm(table[missing], axis=1).mean()) if missing.any() else None},
        "out_table": cfg["out_table"],
    }
    (ROOT / cfg["out_table"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(ROOT / cfg["out_table"], table)
    np.save(ROOT / cfg["out_table"].replace(".npy", "_covered_mask.npy"), covered)
    write_json(info, cfg["out_json"])
    c = info["coverage"]
    print(f"GloVe: {c['covered_types']}/{c['vocab_word_types']} types ({c['pct_types_covered']:.2f} %), "
          f"{c['pct_lm_train_word_tokens_covered']:.2f} % of LM-train word tokens; read in {read_s:.0f}s; sigma mean {sigma.mean():.3f}")
    print("most frequent uncovered:", [(x['word'], x['lm_train_count']) for x in info['most_frequent_uncovered_words']])
    print("literal specials in file:", literal_specials_in_file)


if __name__ == "__main__":
    main()
