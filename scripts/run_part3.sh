#!/usr/bin/env bash
# Part III (TASK §6): frozen-LM features (seed-0 checkpoints, <eos> + first 255 tokens), classifiers for
# 10 LM feature sets + TF-IDF and GloVe-mean baselines (6-value C grid), bootstrap CIs, paired bootstrap,
# D46 robustness subset, length buckets, error exports, the TASK §6 optional ablation (last 255 tokens,
# best LM by classifier-val accuracy), the w2v_mean / lstmA_emb_mean baselines, the follow-up checks, the
# Part III figure and the figure captions.
# Usage: bash scripts/run_part3.sh [--limit N]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(python scripts/wait_gpu.py)}"
python -m src.features --config configs/part3/classify.yaml "$@"
python -m src.classify --config configs/part3/classify.yaml --mode main "$@"
# optional ablation (requires best_pair.json from the main run)
BEST_LM=$(python -c "import json,sys; p=json.load(open('results/part3/best_pair.json')); print(p['A'].rsplit('_',1)[0])")
python -m src.features --config configs/part3/classify.yaml --truncate last --only-lm "$BEST_LM" "$@"
python -m src.classify --config configs/part3/classify.yaml --mode ablation "$@"
# embedding-mean baselines w2v_mean and lstmA_emb_mean (fits only these sets, recomputes the paired bootstrap)
python -m src.classify --config configs/part3/classify.yaml --mode extra_baselines "$@"
# follow-up checks, Part III figure and captions read existing full-mode outputs only (skipped in smoke mode)
if [ $# -eq 0 ]; then
  python -m src.analysis.followup_checks all
  python -m src.analysis.part3_tables
  python -m src.analysis.captions
fi
