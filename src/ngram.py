"""Interpolated (modified) Kneser–Ney n-gram language model on numpy sorted-key tables (TASK §4.1).

Tables. Level k (k = 1..order) stores every distinct k-gram of the training stream. A k-gram is
keyed as `id_{k-1}(history) * |V| + w`, where `id_{k-1}` is the row index of its (k-1)-token
history in level k-1 (level 1 ids are the token ids themselves). Keys are sorted int64 arrays,
so any order fits in int64 and lookups are `np.searchsorted`; all k-grams sharing a history form a
contiguous key range. Each level stores raw counts and continuation counts N1+(• h w) (number of
distinct left extensions), plus per-history aggregates.

Model. The top level uses raw counts, lower levels continuation counts (interpolated KN). With
discounts D(c) ∈ {D1, D2, D3+} chosen by the count c (a single D for the plain variant):
    P_k(w | h) = max(c_k(h w) − D(c_k(h w)), 0) / c_k(h) + γ_k(h) · P_{k−1}(w | h'),
    γ_k(h) = (D1·N1(h•) + D2·N2(h•) + D3+·N3+(h•)) / c_k(h),
    P_1(w)  = max(c_1(w) − D(c_1(w)), 0) / c_1(•) + γ_1 / |V|,
where h' drops the oldest token of h; an unseen history (c_k(h) = 0) backs off with weight 1.
Every level is therefore exactly normalised and no probability is zero.

Block protocol (TASK §3). For the target at block position j the history is the last
min(order−1, j+1) *input* tokens of the same block; with fewer than order−1 tokens the recursion
simply starts at the order that the available history supports (its lower-order estimate).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from src.common import PAD, UNK, ROOT, apply_limit, load_config, read_json, write_json
from src.evaluate import evaluate_stream, make_blocks, score_blocks, summarize


# ----------------------------------------------------------------------------- count tables
def _lookup(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row index of each query key in the sorted `keys`, or -1 if absent."""
    idx = np.searchsorted(keys, q)
    idx_c = np.minimum(idx, len(keys) - 1)
    found = (idx < len(keys)) & (keys[idx_c] == q)
    return np.where(found, idx_c, -1)


def build_tables(stream: np.ndarray, V: int, max_order: int) -> dict:
    """Raw and continuation counts for all orders 1..max_order (continuation for 1..max_order-1)."""
    s = np.asarray(stream, dtype=np.int64)
    n = len(s)
    levels: dict[int, dict] = {1: {"keys": np.arange(V, dtype=np.int64),
                                   "raw": np.bincount(s, minlength=V).astype(np.int64)}}
    ids = {1: s}  # ids[k][i] = row of the k-gram starting at position i (i = 0..n-k)
    for k in range(2, max_order + 1):
        key = ids[k - 1][: n - k + 1] * V + s[k - 1:]
        keys, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        levels[k] = {"keys": keys, "raw": cnt.astype(np.int64)}
        ids[k] = inv.astype(np.int64)
        # continuation counts of level k-1: distinct k-grams whose suffix is that (k-1)-gram
        suffix_rows = ids[k - 1][first + 1]
        levels[k - 1]["cont"] = np.bincount(suffix_rows, minlength=len(levels[k - 1]["keys"])).astype(np.int64)
    del ids
    for k in range(2, max_order + 1):
        levels[k]["hist"] = levels[k]["keys"] // V
        levels[k]["word"] = levels[k]["keys"] % V
    return {"V": V, "max_order": max_order, "levels": levels, "n_tokens": int(n)}


def _hist_aggregates(hist: np.ndarray, cnt: np.ndarray, n_hist: int) -> dict:
    nz = cnt > 0
    return {
        "c_h": np.bincount(hist[nz], weights=cnt[nz], minlength=n_hist),
        "n1": np.bincount(hist[cnt == 1], minlength=n_hist).astype(np.float64),
        "n2": np.bincount(hist[cnt == 2], minlength=n_hist).astype(np.float64),
        "n3p": np.bincount(hist[cnt >= 3], minlength=n_hist).astype(np.float64),
    }


def count_of_counts(cnt: np.ndarray) -> dict:
    return {f"n{k}": int((cnt == k).sum()) for k in (1, 2, 3, 4)}


def modified_discounts(cnt: np.ndarray, fallback: tuple[float, float, float]) -> tuple[tuple[float, float, float], bool, dict]:
    """Chen–Goodman (1998) estimates from count-of-counts; `fallback` if any n_k = 0 or a D <= 0."""
    coc = count_of_counts(cnt)
    n1, n2, n3, n4 = coc["n1"], coc["n2"], coc["n3"], coc["n4"]
    if min(n1, n2, n3, n4) == 0:
        return tuple(fallback), True, coc
    y = n1 / (n1 + 2 * n2)
    d = (1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3)
    if min(d) <= 0:
        return tuple(fallback), True, coc
    return d, False, coc


# ----------------------------------------------------------------------------- the model
class KneserNeyLM:
    """Interpolated KN of a given order built from `build_tables` output and a discount spec.

    discount_spec: {"type": "single", "D": 0.75} or {"type": "modified", "fallback": [0.5, 1.0, 1.5]}
    """

    def __init__(self, tables: dict, order: int, discount_spec: dict):
        assert 1 <= order <= tables["max_order"]
        self.V = tables["V"]
        self.order = order
        self.spec = discount_spec
        self.levels: dict[int, dict] = {}
        self.discounts: dict[int, dict] = {}
        for k in range(1, order + 1):
            src = tables["levels"][k]
            cnt = src["raw"] if k == order else src["cont"]  # top level raw, lower levels continuation
            if discount_spec["type"] == "single":
                D = float(discount_spec["D"])
                d, fb, coc = (D, D, D), False, count_of_counts(cnt)
            else:
                d, fb, coc = modified_discounts(cnt, tuple(discount_spec.get("fallback", (0.5, 1.0, 1.5))))
            self.discounts[k] = {"D1": d[0], "D2": d[1], "D3p": d[2], "fallback_used": fb, **coc,
                                 "count_type": "raw" if k == order else "continuation"}
            lvl = {"keys": src["keys"], "cnt": cnt}
            if k == 1:
                total = float(cnt.sum())
                n1, n2, n3p = float((cnt == 1).sum()), float((cnt == 2).sum()), float((cnt >= 3).sum())
                gamma1 = (d[0] * n1 + d[1] * n2 + d[2] * n3p) / total
                disc = np.maximum(cnt - self._d_of(cnt, d), 0.0) / total
                lvl["p1"] = disc + gamma1 / self.V          # dense unigram distribution
                lvl["total"] = total
            else:
                n_hist = len(tables["levels"][k - 1]["keys"])
                agg = _hist_aggregates(src["hist"], cnt, n_hist)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gamma = np.where(agg["c_h"] > 0,
                                     (d[0] * agg["n1"] + d[1] * agg["n2"] + d[2] * agg["n3p"]) / agg["c_h"], 1.0)
                lvl["c_h"] = agg["c_h"]
                lvl["gamma"] = gamma
                lvl["word"] = src["word"]
                lvl["disc"] = np.maximum(cnt - self._d_of(cnt, d), 0.0)   # discounted numerators
            self.levels[k] = lvl

    @staticmethod
    def _d_of(cnt: np.ndarray, d: tuple[float, float, float]) -> np.ndarray:
        return np.where(cnt >= 3, d[2], np.where(cnt == 2, d[1], np.where(cnt == 1, d[0], 0.0)))

    # --- sizes
    def n_stored(self) -> dict[int, int]:
        return {k: int(len(self.levels[k]["keys"])) for k in range(1, self.order + 1)}

    def memory_bytes(self) -> int:
        total = 0
        for k, lvl in self.levels.items():
            for name in ("keys", "cnt", "p1", "c_h", "gamma", "word", "disc"):
                if name in lvl:
                    total += lvl[name].nbytes
        return int(total)

    # --- scoring under the block protocol
    def logprob_blocks(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """log P(targets[b, j] | inputs[b, max(0, j-order+2) .. j]) for every position, shape [B, L]."""
        x = np.asarray(inputs, dtype=np.int64)
        y = np.asarray(targets, dtype=np.int64)
        B, L = x.shape
        y_safe = np.where((y >= 0) & (y < self.V), y, 0)
        P = self.levels[1]["p1"][y_safe]
        hist = x.copy()  # id_{k-1} of the (k-1)-token history ending at j; starts with 1-token histories
        for k in range(2, self.order + 1):
            lvl = self.levels[k]
            valid = hist >= 0
            h = np.where(valid, hist, 0)
            c_h = lvl["c_h"][h]
            use = valid & (c_h > 0)
            row = _lookup(lvl["keys"], h * self.V + y_safe)
            num = np.where(row >= 0, lvl["disc"][np.maximum(row, 0)], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                P_k = num / np.where(use, c_h, 1.0) + lvl["gamma"][h] * P
            P = np.where(use, P_k, P)
            if k < self.order:  # history ids for the next level: k tokens ending at j
                nxt = np.full_like(hist, -1)
                prev = hist[:, :-1]
                q = np.where(prev >= 0, prev, 0) * self.V + x[:, 1:]
                r = _lookup(lvl["keys"], q)
                nxt[:, 1:] = np.where(prev >= 0, r, -1)
                hist = nxt
        return np.log(P)

    # --- dense next-token distribution for sampling / normalisation tests
    def history_ids(self, history: list[int]) -> list[int]:
        """id_k of the last k tokens of `history` for k = 1..min(order-1, len(history)); -1 if unseen."""
        ids = []
        for k in range(1, min(self.order - 1, len(history)) + 1):
            toks = history[-k:]
            hid = toks[0]
            for m in range(2, k + 1):
                hid = int(_lookup(self.levels[m]["keys"], np.array([hid * self.V + toks[m - 1]]))[0]) if hid >= 0 else -1
            ids.append(hid)
        return ids

    def next_distribution(self, history: list[int]) -> np.ndarray:
        """P(· | history) over the vocabulary using the last order-1 tokens (dense, sums to 1)."""
        P = self.levels[1]["p1"].copy()
        ids = self.history_ids([int(t) for t in history])
        for k in range(2, min(self.order, len(history) + 1) + 1):
            hid = ids[k - 2]
            if hid < 0:
                continue
            lvl = self.levels[k]
            c_h = lvl["c_h"][hid]
            if c_h <= 0:
                continue
            lo = np.searchsorted(lvl["keys"], hid * self.V)
            hi = np.searchsorted(lvl["keys"], (hid + 1) * self.V)
            P = lvl["gamma"][hid] * P
            P[lvl["word"][lo:hi]] += lvl["disc"][lo:hi] / c_h
        return P

    # --- persistence
    def save(self, path: str | Path) -> None:
        path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for k, lvl in self.levels.items():
            for name, arr in lvl.items():
                if isinstance(arr, np.ndarray):
                    arrays[f"L{k}_{name}"] = arr
        meta = {"V": self.V, "order": self.order, "spec": self.spec, "discounts": self.discounts,
                "total": self.levels[1]["total"]}
        np.savez(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "KneserNeyLM":
        z = np.load(ROOT / path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        obj = cls.__new__(cls)
        obj.V, obj.order, obj.spec = meta["V"], meta["order"], meta["spec"]
        obj.discounts = {int(k): v for k, v in meta["discounts"].items()}
        obj.levels = {}
        for name in z.files:
            if name == "meta":
                continue
            k, arr_name = name[1:].split("_", 1)
            obj.levels.setdefault(int(k), {})[arr_name] = z[name]
        obj.levels[1]["total"] = meta["total"]
        return obj


# ----------------------------------------------------------------------------- selection CLI
def spec_label(spec: dict) -> str:
    return f"D={spec['D']}" if spec["type"] == "single" else "modifiedKN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part1/ngram.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke mode: use data/smoke streams")
    args = ap.parse_args()
    cfg = apply_limit(load_config(args.config, {"limit_reviews": args.limit}))
    if cfg.get("limit_reviews"):
        cfg["derived_dir"] = "data/smoke/derived"
        cfg["model_out"] = "checkpoints/smoke/kn_selected.npz"
    streams = ROOT / cfg["streams_dir"]
    results = ROOT / cfg["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    L, bs = cfg["block_length"], cfg["batch_size"]

    vocab = read_json(streams / "vocab.json")
    V = len(vocab["itos"])
    train = np.load(streams / "lm_train.npy")
    val = np.load(streams / "lm_val.npy")
    test = np.load(streams / "lm_test.npy")
    val_inputs, val_targets = make_blocks(val, L)

    orders = list(cfg["candidate_orders"])
    ref_orders = list(cfg.get("reference_orders_modified_kn", []))
    max_order = max(orders + ref_orders)
    fallback = tuple(cfg["fallback_discounts"])

    print(f"building count tables up to order {max_order} on {len(train):,} tokens (|V|={V}) ...", flush=True)
    t0 = time.perf_counter()
    tables = build_tables(train, V, max_order)
    print(f"  done in {time.perf_counter() - t0:.1f}s; distinct n-grams per order:",
          {k: len(v["keys"]) for k, v in tables["levels"].items()}, flush=True)

    candidates = []
    for order in orders:
        for D in cfg["discounts"]:
            candidates.append({"order": order, "spec": {"type": "single", "D": float(D)}, "role": "candidate"})
        if cfg.get("include_modified_kn", True):
            candidates.append({"order": order, "spec": {"type": "modified", "fallback": list(fallback)}, "role": "candidate"})
    for order in ref_orders:
        candidates.append({"order": order, "spec": {"type": "modified", "fallback": list(fallback)}, "role": "reference"})

    rows = []
    for c in candidates:
        t1 = time.perf_counter()
        lm = KneserNeyLM(tables, c["order"], c["spec"])
        ce = score_blocks(lm.logprob_blocks, val_inputs, val_targets, bs)
        s = summarize(ce, val_targets)
        row = {"role": c["role"], "order": c["order"], "discount": spec_label(c["spec"]),
               "val_ce": s["ce"], "val_ppl": s["ppl"], "val_n_targets": s["n_targets"],
               "n_stored_total": sum(lm.n_stored().values()), "memory_mb": lm.memory_bytes() / 2**20,
               "seconds": time.perf_counter() - t1}
        for k in range(1, c["order"] + 1):
            d = lm.discounts[k]
            row[f"L{k}_D1_D2_D3p"] = f"{d['D1']:.4f}/{d['D2']:.4f}/{d['D3p']:.4f}" + ("*" if d["fallback_used"] else "")
            row[f"L{k}_n1_n2_n3_n4"] = f"{d['n1']}/{d['n2']}/{d['n3']}/{d['n4']}"
        rows.append(row)
        print(f"  {row['role']:9s} order={row['order']} {row['discount']:12s} val CE={row['val_ce']:.4f} PPL={row['val_ppl']:.2f} ({row['seconds']:.1f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(results / "ngram_selection.csv", index=False)

    cand = df[df.role == "candidate"]
    best = cand.loc[cand.val_ce.idxmin()]
    best_spec = next(c for c in candidates if c["role"] == "candidate" and c["order"] == best.order and spec_label(c["spec"]) == best.discount)
    ref5 = df[(df.role == "reference") & (df.order == 5)]
    print(f"\nselected: order {best.order}, {best.discount}, val CE {best.val_ce:.4f}")
    if len(ref5) and float(ref5.val_ce.iloc[0]) < float(best.val_ce):
        print(f"NOTE: reference order-5 modified KN has lower val CE ({float(ref5.val_ce.iloc[0]):.4f}) than the selected candidate — reported, not selected.")

    # --- rebuild the selected order on its own, timed (count-build time = the n-gram "training time")
    del tables
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t2 = time.perf_counter()
    tables_sel = build_tables(train, V, int(best.order))
    lm = KneserNeyLM(tables_sel, int(best.order), best_spec["spec"])
    build_seconds = time.perf_counter() - t2
    lm.save(cfg["model_out"])

    # --- single LM-test evaluation of the selected model
    summary, ce_test, targets_test = evaluate_stream(lm.logprob_blocks, test, L, bs, return_ce=True)
    derived = ROOT / cfg["derived_dir"]
    derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "ngram_test_ce.npy", ce_test.astype(np.float32))
    val_summary = evaluate_stream(lm.logprob_blocks, val, L, bs)

    from src.common import cpu_model, env_info
    model_json = {
        "selected": {"order": int(best.order), "discount": best.discount, "spec": best_spec["spec"],
                     "discounts_per_level": lm.discounts},
        "selection_rule": "lowest LM-val CE among the 14 pre-specified candidates (orders 3,4 x 6 single discounts + modified KN); reference rows never selected",
        "val": {k: v for k, v in val_summary.items() if k != "ce_by_position"},
        "test": {k: v for k, v in summary.items() if k != "ce_by_position"},
        "test_ce_by_position": summary["ce_by_position"],
        "n_stored_per_order": lm.n_stored(),
        "n_stored_total": sum(lm.n_stored().values()),
        "memory_bytes": lm.memory_bytes(),
        "memory_mb": lm.memory_bytes() / 2**20,
        "count_build_seconds": build_seconds,
        "count_build_note": "wall-clock of build_tables + model construction for the selected order, CPU single process, OMP_NUM_THREADS=" + os.environ.get("OMP_NUM_THREADS", "unset"),
        "lm_train_tokens": int(len(train)),
        "cpu_model": cpu_model(),
        "env": env_info(with_gpu=False),
        "model_path": cfg["model_out"],
        "test_ce_path": str(derived / "ngram_test_ce.npy"),
    }
    write_json(model_json, results / "ngram_model.json")
    print(json.dumps({k: v for k, v in model_json.items() if k not in ("test_ce_by_position", "env")}, indent=1, default=str))


if __name__ == "__main__":
    main()
