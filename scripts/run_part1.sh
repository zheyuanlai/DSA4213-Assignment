#!/usr/bin/env bash
# Part I. Steps are appended stage by stage; every step is config-driven and idempotent.
# Usage: bash scripts/run_part1.sh [--limit N]   (--limit N = smoke mode on the small subset)
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # exactly one GPU (shared node)

# --- Stage 2: Kneser–Ney n-gram — selection on LM-val, timed rebuild, single LM-test evaluation (CPU)
OMP_NUM_THREADS=1 python -m src.ngram --config configs/part1/ngram.yaml "$@"

# --- Stage 3: parameter breakdown, tests (a)/(d), smoke runs, LR probes and LR selection
python -m src.models --out results/part1/param_breakdown.csv
python -m pytest tests/ -q
for m in rnn lstm transformer; do python -m src.train_lm --config configs/smoke/$m.yaml; done
for m in rnn lstm transformer; do for lr in 0.0003 0.001 0.002; do
  python -m src.train_lm --config configs/probes/${m}_lr${lr}.yaml
done; done
python -m src.analysis.part1_tables probes          # -> results/part1/lr_probes.csv, lr_selected.json
python scripts/make_configs.py                      # fills the selected LR into configs/part1 and configs/part2

# --- Stage 4: full Part I training (seed 0 first, then seeds 1-2), strictly sequential on one GPU
for s in 0 1 2; do for m in rnn lstm transformer; do
  python -m src.train_lm --config configs/part1/${m}_seed${s}.yaml || echo "RUN FAILED: ${m} seed ${s} (rule D31)"
done; done

# --- Stage 4: timing (env.json, workloads A/B with GPU-idle checks), generation, tables and figures
python -m src.timing --runs part1/rnn_s0 part1/lstm_s0 part1/transformer_s0 --ngram checkpoints/ngram/kn_selected.npz --out results/part1/timing.json
python -m src.generate --runs part1/rnn_s0 part1/lstm_s0 part1/transformer_s0 --ngram checkpoints/ngram/kn_selected.npz --out results/part1
python -m src.analysis.part1_tables main
