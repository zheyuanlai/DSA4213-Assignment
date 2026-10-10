#!/usr/bin/env bash
# Part II (TASK §5): own word2vec, GloVe table + coverage, frozen-embedding LSTM runs (smoke first,
# then seeds 0-2 for settings B and C, one idle GPU at a time), generation and analyses.
# Usage: bash scripts/run_part2.sh            (all steps are idempotent: finished embeddings are reused)
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/OVERNIGHT_LOG.md
note() { echo "| $(date -u +%H:%M) | $1 | $2 |" >> "$LOG"; }

# 1. embeddings (reused if their JSON reports already exist)
[ -f results/part2/word2vec_training.json ] || python -m src.word2vec --config configs/part2/word2vec.yaml
[ -f results/part2/glove_coverage.json ] || python -m src.glove --config configs/part2/glove.yaml
note done "embeddings ready: data/embeddings/word2vec_300.npy, data/embeddings/glove_300.npy"

# 2. smoke runs of both frozen settings (1 % of LM-train, 2 epochs)
GPU=$(python scripts/wait_gpu.py); export CUDA_VISIBLE_DEVICES=$GPU
for e in ownfrozen glovefrozen; do
  python -m src.train_lm --config configs/smoke/lstm_${e}.yaml || { note FAILED "smoke/lstm_${e}"; exit 1; }
done
note done "Part II smoke runs passed (GPU $GPU)"

# 3. full runs: seed 0 first, then seeds 1 and 2; one idle GPU, re-checked before every run
for s in 0 1 2; do for e in ownfrozen glovefrozen; do
  GPU=$(python scripts/wait_gpu.py); export CUDA_VISIBLE_DEVICES=$GPU
  note start "part2/lstm_${e}_s${s} on GPU ${GPU}"
  if python -m src.train_lm --config configs/part2/lstm_${e}_seed${s}.yaml; then
    note done "part2/lstm_${e}_s${s} -> results/runs/part2/lstm_${e}_s${s}/"
  else
    note FAILED "part2/lstm_${e}_s${s} (see results/runs/part2/lstm_${e}_s${s}/log.jsonl; rule D31 PENDING USER REVIEW)"
  fi
done; done

# 4. generation (Section 4.5 protocol) and Part II tables/figures
GPU=$(python scripts/wait_gpu.py); export CUDA_VISIBLE_DEVICES=$GPU
python -m src.generate --runs part1/lstm_s0 part2/lstm_ownfrozen_s0 part2/lstm_glovefrozen_s0 --labels A-random B-own C-glove --ngram none --out results/part2 \
  && note done "Part II generations -> results/part2/generations.md" || note FAILED "Part II generation"
python -m src.analysis.part2_tables && note done "Part II tables/figures -> results/part2/, results/figures/part2_*" || note FAILED "Part II analysis"
# 5. extra control (user D95): C-scaled = rescaled frozen GloVe; smoke + 3 seeds; reported outside the main table
python - <<'PY'
import numpy as np, json
g = np.load("data/embeddings/glove_300.npy"); w = np.load("data/embeddings/word2vec_300.npy")
pds = lambda t: float(t.std(axis=0).mean()); k = pds(w) / pds(g); t = (g * k).astype(np.float32); t[0] = 0
np.save("data/embeddings/glove_300_scaled.npy", t); json.dump({"scalar": k, "per_dim_std_glove": pds(g), "per_dim_std_word2vec": pds(w), "per_dim_std_scaled": pds(t)}, open("results/part2/glove_scaled.json", "w"), indent=1)
PY
bash scripts/run_part2_scaled.sh
python -m src.analysis.part2_tables      # adds the *_with_scaled / ablation_glove_scaled tables
echo "PART2 PIPELINE FINISHED $(date -u)"
