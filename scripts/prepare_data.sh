#!/usr/bin/env bash
# Stage 1: download IMDb, clean/tokenise, de-duplicate, split, build vocabulary and streams.
# Usage: bash scripts/prepare_data.sh            (full data)
#        bash scripts/prepare_data.sh --limit 500 (smoke test on the first 500 reviews per split)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1
python -m src.data --config configs/data.yaml "$@"
python -m src.vocab --config configs/data.yaml "$@"
