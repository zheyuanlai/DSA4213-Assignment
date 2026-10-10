"""Generate one YAML per run (TASK §7) from the base settings below. Re-run after LR selection.

Usage: python scripts/make_configs.py [--lr-file results/part1/lr_selected.json]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

BASE = {
    "seed": 0,
    "data": {"streams_dir": "data/streams", "block_length": 256, "train_fraction": 1.0},
    "embedding": {"init": "random", "path": None, "freeze": False},
    "optim": {"lr": None, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1.0e-8, "warmup_steps": 300,
              "final_lr_fraction": 0.1, "grad_clip": 1.0, "batch_size": 64},
    "train": {"max_epochs": 12, "schedule_epochs": 12, "patience": 3, "evaluate_test": True, "log_every": 50},
}
MODELS = {
    "rnn": {"type": "rnn", "d_model": 300, "hidden": 1072, "n_layers": 2, "dropout": 0.3},
    "lstm": {"type": "lstm", "d_model": 300, "hidden": 512, "n_layers": 2, "dropout": 0.3},
    "transformer": {"type": "transformer", "d_model": 300, "n_layers": 4, "n_heads": 6, "ffn": 1200, "dropout": 0.1, "max_len": 256},
}
PROBE_LRS = [3e-4, 1e-3, 2e-3]
SEEDS = [0, 1, 2]


def dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-file", default="results/part1/lr_selected.json")
    args = ap.parse_args()
    lr_file = ROOT / args.lr_file
    selected = json.load(open(lr_file)) if lr_file.exists() else {}
    n = 0
    for m, mcfg in MODELS.items():
        # LR probes: 1 epoch of the real 12-epoch schedule, seed 0, no test evaluation
        for lr in PROBE_LRS:
            c = copy.deepcopy(BASE)
            c.update({"run_name": f"probes/{m}_lr{lr:g}", "model": mcfg})
            c["optim"]["lr"] = lr
            c["train"].update({"max_epochs": 1, "schedule_epochs": 12, "evaluate_test": False})
            dump(c, ROOT / f"configs/probes/{m}_lr{lr:g}.yaml"); n += 1
        # smoke: 1 % of LM-train, 2 epochs, short warm-up, no test evaluation
        c = copy.deepcopy(BASE)
        c.update({"run_name": f"smoke/{m}", "model": mcfg})
        c["optim"].update({"lr": 1e-3, "warmup_steps": 5})
        c["data"]["train_fraction"] = 0.01
        c["train"].update({"max_epochs": 2, "schedule_epochs": 2, "evaluate_test": False})
        dump(c, ROOT / f"configs/smoke/{m}.yaml"); n += 1
        # full Part I runs (LR filled from the probe selection once available)
        for s in SEEDS:
            c = copy.deepcopy(BASE)
            c.update({"run_name": f"part1/{m}_s{s}", "seed": s, "model": mcfg})
            c["optim"]["lr"] = selected.get(m, {}).get("lr")
            dump(c, ROOT / f"configs/part1/{m}_seed{s}.yaml"); n += 1
    # Part II frozen-embedding LSTM runs (same LR as the Part I LSTM) + their smoke configs
    # "glovescaled" = extra control (user D95): GloVe table rescaled to the word2vec per-dim std, kept out of the main table
    for setting, path in (("ownfrozen", "data/embeddings/word2vec_300.npy"), ("glovefrozen", "data/embeddings/glove_300.npy"),
                          ("glovescaled", "data/embeddings/glove_300_scaled.npy")):
        for s in SEEDS:
            c = copy.deepcopy(BASE)
            c.update({"run_name": f"part2/lstm_{setting}_s{s}", "seed": s, "model": MODELS["lstm"]})
            c["embedding"] = {"init": "file", "path": path, "freeze": True}
            c["optim"]["lr"] = selected.get("lstm", {}).get("lr")
            dump(c, ROOT / f"configs/part2/lstm_{setting}_seed{s}.yaml"); n += 1
        c = copy.deepcopy(BASE)
        c.update({"run_name": f"smoke/lstm_{setting}", "model": MODELS["lstm"]})
        c["embedding"] = {"init": "file", "path": path, "freeze": True}
        c["optim"].update({"lr": 1e-3, "warmup_steps": 5})
        c["data"]["train_fraction"] = 0.01
        c["train"].update({"max_epochs": 2, "schedule_epochs": 2, "evaluate_test": False})
        dump(c, ROOT / f"configs/smoke/lstm_{setting}.yaml"); n += 1
    print(f"wrote {n} configs; selected LRs: {selected or 'none yet'}")


if __name__ == "__main__":
    main()
