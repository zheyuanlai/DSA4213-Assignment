"""Own word embeddings for Part II setting B (TASK §5): gensim Skip-gram on the vocabulary-mapped LM-train
stream. Sentences = reviews (each ending in <eos>), tokens = vocabulary ids as strings, so <unk> and <eos>
receive vectors and coverage is 100 % by construction; <pad> never occurs and is set to zeros. Training
time is recorded separately from LM training. No LM-val or LM-test text is used.
"""
from __future__ import annotations

import argparse
import time

import gensim
import numpy as np
from gensim.models import Word2Vec

from src.common import EOS, PAD, ROOT, cpu_model, env_info, load_config, read_json, write_json


def stream_to_sentences(stream: np.ndarray) -> list[list[str]]:
    sents, cur = [], []
    for t in stream.tolist():
        cur.append(str(t))
        if t == EOS:
            if len(cur) > 1:          # skip the lone leading <eos>
                sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2/word2vec.yaml")
    ap.add_argument("--limit", type=float, default=None, help="smoke: fraction of the LM-train stream, 1 epoch, outputs under data/smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    stream = np.load(ROOT / cfg["streams_dir"] / "lm_train.npy")
    vocab = read_json(ROOT / cfg["streams_dir"] / "vocab.json")
    V = len(vocab["itos"])
    if args.limit:
        stream = stream[: int(len(stream) * args.limit)]
        cfg["epochs"] = 1
        cfg["out_table"], cfg["out_json"] = "data/smoke/embeddings/word2vec_300.npy", "results/smoke/word2vec_training.json"
    sents = stream_to_sentences(stream)
    n_tokens = sum(len(s) for s in sents)
    print(f"{len(sents):,} sentences, {n_tokens:,} tokens, |V|={V}; training skip-gram ...", flush=True)
    t0 = time.perf_counter()
    model = Word2Vec(sentences=sents, vector_size=cfg["vector_size"], window=cfg["window"], negative=cfg["negative"],
                     sg=cfg["sg"], epochs=cfg["epochs"], min_count=cfg["min_count"], seed=cfg["seed"], workers=cfg["workers"])
    train_s = time.perf_counter() - t0
    table = np.zeros((V, cfg["vector_size"]), dtype=np.float32)
    present = 0
    for i in range(V):
        key = str(i)
        if key in model.wv.key_to_index:
            table[i] = model.wv[key]
            present += 1
    assert table[PAD].any() == False, "<pad> must stay zero"  # noqa: E712
    (ROOT / cfg["out_table"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(ROOT / cfg["out_table"], table)
    nonpad = table[1:]
    info = {
        "method": "gensim Word2Vec skip-gram with negative sampling",
        "params": {k: cfg[k] for k in ("vector_size", "window", "negative", "sg", "epochs", "min_count", "seed", "workers")},
        "gensim_version": gensim.__version__,
        "training_seconds": train_s, "training_time_hms": time.strftime("%H:%M:%S", time.gmtime(train_s)),
        "cpu_model": cpu_model(), "loadavg_after": list(__import__("os").getloadavg()),
        "n_sentences": len(sents), "n_tokens": n_tokens, "stream_fraction": args.limit or 1.0,
        "vocab_ids_with_vector": present, "coverage_types_pct": 100.0 * present / (V - 1),
        "coverage_note": "all ids except <pad> occur in the LM-train stream, so coverage of the vocabulary (incl. <unk>, <eos>) is 100 % by construction; <pad> = zeros",
        "per_dim_std_mean": float(nonpad.std(axis=0).mean()), "mean_vector_norm": float(np.linalg.norm(nonpad, axis=1).mean()),
        "special_tokens": {"<pad>": "zeros (never observed)", "<unk>": "trained vector (occurs in the mapped stream)", "<eos>": "trained vector (ends every review)"},
        "out_table": cfg["out_table"], "env": env_info(with_gpu=False),
    }
    write_json(info, cfg["out_json"])
    print({k: v for k, v in info.items() if k not in ("env", "params")})


if __name__ == "__main__":
    main()
