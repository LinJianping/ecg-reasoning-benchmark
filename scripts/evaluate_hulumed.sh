#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/hulumed_env.sh"
evaluator="${1:-heuristic}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -eq 0 ]]; then set -- ptbxl; fi
extra=()
if [[ "$evaluator" == gemini ]]; then
    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo 'Set GOOGLE_API_KEY locally to use the paper judge; no API key is stored in this repository.' >&2
        exit 2
    fi
    extra=(--gemini-model "${GEMINI_MODEL:-gemini-3-flash-preview}" --use-cache --save-cache --load-cache --save-cache-interval 1)
elif [[ "$evaluator" != heuristic ]]; then
    echo 'Usage: bash scripts/evaluate_hulumed.sh [heuristic|gemini] [ptbxl mimic_iv_ecg]' >&2
    exit 2
fi
ecg-reasoning-benchmark-evaluate "$RESULTS_DIR" --dataset "$@" --model "hulumed-hf_$HULUMED_VARIANT" --evaluator "$evaluator" --save-dir "$EVAL_DIR" "${extra[@]}"
