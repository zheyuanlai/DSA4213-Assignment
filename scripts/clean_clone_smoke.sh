#!/usr/bin/env bash
# Clean-clone reproduction test in SMOKE mode (TASK §8 last item; user instruction D77): clone this repo
# into a temporary directory and run every pipeline on tiny subsets. The full experiments are NOT rerun.
# The GloVe zip is copied from the source checkout instead of being downloaded again (862 MB).
# Usage: bash scripts/clean_clone_smoke.sh [/tmp/target_dir]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$(mktemp -d /tmp/dsa4213_clone_XXXX)}"
echo "cloning $SRC -> $DST"; git clone -q "$SRC" "$DST"; cd "$DST"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(python "$SRC/scripts/wait_gpu.py")}"
t0=$(date +%s)
bash scripts/prepare_data.sh                                   # full data prep (download + 45 s); needed by every smoke config
bash scripts/prepare_data.sh --limit 500                       # smoke streams for the n-gram smoke
python -m pytest tests/ -q
python -m src.ngram --config configs/part1/ngram.yaml --limit 500
python -m src.models --out results/smoke/param_breakdown.csv
for m in rnn lstm transformer; do python -m src.train_lm --config configs/smoke/$m.yaml; done
python -m src.word2vec --limit 0.01
mkdir -p data/embeddings data/glove && cp data/smoke/embeddings/word2vec_300.npy data/embeddings/word2vec_300.npy   # smoke table stands in for the full one
cp "$SRC/data/glove/glove.6B.zip" data/glove/ && python -m src.glove --config configs/part2/glove.yaml
for e in ownfrozen glovefrozen; do python -m src.train_lm --config configs/smoke/lstm_$e.yaml; done
python -m src.generate --runs smoke/rnn smoke/lstm smoke/transformer --ngram none --out results/smoke/gen
python -m src.timing --runs smoke/lstm --ngram none --out results/smoke/timing.json --env-out results/smoke/env.json
python -m src.features --config configs/smoke/part3.yaml --limit 500
python -m src.classify --config configs/smoke/part3.yaml --limit 500
echo "CLEAN CLONE SMOKE OK in $(( $(date +%s) - t0 )) s at $DST"
