#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/hulumed_env.sh"
mkdir -p "$ECG_REPRO_ASSETS/logs"
echo "$$" > "$ECG_REPRO_ASSETS/logs/download-32b.pid"
exec 9>"$ECG_REPRO_ASSETS/logs/download-32b.lock"
flock -n 9 || { echo 'A managed 32B download is already running.' >&2; exit 2; }
for attempt in 1 2 3 4 5; do
    echo "32B download attempt $attempt at $(date -u +%FT%TZ)"
    if python -u "$ECG_REPRO_REPO/scripts/download_hulumed.py" --variant 32B --weight-source modelscope --workers 4; then
        echo "32B download and SHA256 verification complete at $(date -u +%FT%TZ)"
        exit 0
    fi
    echo 'Download interrupted; retrying missing ranges in 10 seconds.'
    sleep 10
done
echo '32B download still incomplete after retries. Re-run this script to resume.' >&2
exit 1
