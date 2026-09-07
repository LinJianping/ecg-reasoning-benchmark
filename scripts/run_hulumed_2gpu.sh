#!/usr/bin/env bash
set -euo pipefail
export HULUMED_VARIANT=32B
source "$(dirname -- "${BASH_SOURCE[0]}")/hulumed_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$ECG_REPRO_ASSETS/logs"
if [[ $# -eq 0 ]]; then set -- ptbxl; fi
python -u "$ECG_REPRO_REPO/scripts/run_hulumed_batch.py" "$@" 2>&1 | tee -a "$ECG_REPRO_ASSETS/logs/inference-$(date -u +%Y%m%dT%H%M%SZ).log"
