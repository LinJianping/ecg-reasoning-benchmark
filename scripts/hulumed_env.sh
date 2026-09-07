#!/usr/bin/env bash
# Source this file from any shell. It does not edit your global shell settings.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export ECG_REPRO_ASSETS="${ECG_REPRO_ASSETS:-/root/autodl-tmp/ecg-repro-assets}"
export ECG_REPRO_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$ECG_REPRO_ASSETS/cache/huggingface"
export ERB_CACHE="$ECG_REPRO_ASSETS/cache/evaluation"
export MPLCONFIGDIR="$ECG_REPRO_ASSETS/cache/matplotlib"
export MPLBACKEND=Agg
export TMPDIR="$ECG_REPRO_ASSETS/tmp"
export HULUMED_VARIANT="${HULUMED_VARIANT:-32B}"
export HULUMED_MODEL_PATH="${HULUMED_MODEL_PATH:-$ECG_REPRO_ASSETS/models/Hulu-Med-$HULUMED_VARIANT}"
export HULUMED_ATTN_IMPLEMENTATION="${HULUMED_ATTN_IMPLEMENTATION:-flash_attention_2}"
export HULUMED_MAX_MEMORY_GIB="${HULUMED_MAX_MEMORY_GIB:-72}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-$ECG_REPRO_ASSETS/data/benchmark-paper-v1}"
export PTBXL_DIR="${PTBXL_DIR:-$ECG_REPRO_ASSETS/data/ptb-xl}"
export MIMIC_ECG_DIR="${MIMIC_ECG_DIR:-$ECG_REPRO_ASSETS/data/mimic-iv-ecg}"
export RESULTS_DIR="${RESULTS_DIR:-$ECG_REPRO_ASSETS/results/paper-v1/$HULUMED_VARIANT}"
export EVAL_DIR="${EVAL_DIR:-$ECG_REPRO_ASSETS/eval-results/paper-v1/$HULUMED_VARIANT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$ERB_CACHE" "$MPLCONFIGDIR" "$TMPDIR"
source "$ECG_REPRO_ASSETS/envs/hulumed/bin/activate"
