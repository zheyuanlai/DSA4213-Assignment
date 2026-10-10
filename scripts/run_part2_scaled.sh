#!/usr/bin/env bash
# Extra control for Part II (user D95): C-scaled = GloVe table rescaled to the word2vec per-dimension std, frozen.
# Smoke first, then seeds 0-2 on one idle GPU. Kept out of the main Part II table (reported in ablation_glove_scaled.csv).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/OVERNIGHT_LOG.md
note() { echo "| $(date -u +%m-%d\ %H:%M) | $1 | $2 |" >> "$LOG"; }
GPU=$(python scripts/wait_gpu.py); export CUDA_VISIBLE_DEVICES=$GPU
python -m src.train_lm --config configs/smoke/lstm_glovescaled.yaml || { note FAILED "smoke/lstm_glovescaled"; exit 1; }
note done "C-scaled smoke passed (GPU $GPU)"
for s in 0 1 2; do
  GPU=$(python scripts/wait_gpu.py); export CUDA_VISIBLE_DEVICES=$GPU
  note start "part2/lstm_glovescaled_s${s} on GPU ${GPU}"
  python -m src.train_lm --config configs/part2/lstm_glovescaled_seed${s}.yaml && note done "part2/lstm_glovescaled_s${s}" || note FAILED "part2/lstm_glovescaled_s${s} (rule D31 PENDING USER REVIEW)"
done
echo "SCALED PIPELINE FINISHED $(date -u)"
