#!/usr/bin/env bash
# Run the entire PTB-XL benchmark and evaluate it after inference completes.
set -euo pipefail
export HULUMED_VARIANT=7B
export HULUMED_ATTN_IMPLEMENTATION="${HULUMED_ATTN_IMPLEMENTATION:-flash_attention_2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/hulumed_env.sh"
mkdir -p "$ECG_REPRO_ASSETS/logs"
echo "$$" > "$ECG_REPRO_ASSETS/logs/7b-full-pipeline.pid"
mkdir -p "$RESULTS_DIR"
status_file="$RESULTS_DIR/pipeline-status.json"
write_status() {
    python - "$status_file" "$1" <<'PY'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps({'state': sys.argv[2], 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getppid()}, indent=2))
tmp.replace(path)
PY
}
trap 'write_status failed' ERR
write_status verifying_data
python "$ECG_REPRO_REPO/scripts/download_ecg_data.py" --dataset ptbxl --benchmark-dir "$BENCHMARK_DIR" --additional-benchmark-dir "$ECG_REPRO_REPO/data" --verify-only
write_status inference
bash "$ECG_REPRO_REPO/scripts/run_hulumed_7b.sh" ptbxl
write_status evaluation
bash "$ECG_REPRO_REPO/scripts/evaluate_hulumed.sh" heuristic ptbxl
python "$ECG_REPRO_REPO/scripts/report_hulumed.py" --dataset ptbxl
write_status completed
