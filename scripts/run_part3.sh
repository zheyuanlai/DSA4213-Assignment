#!/usr/bin/env bash
# Part III (TASK §6): frozen-LM features (seed-0 checkpoints, <eos> + first 255 tokens), classifiers for
# 10 LM feature sets + 2 baselines, bootstrap CIs, D46 robustness subset, length buckets, error exports,
# then the TASK §6 optional ablation (last 255 tokens, best LM by classifier-val accuracy).
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
