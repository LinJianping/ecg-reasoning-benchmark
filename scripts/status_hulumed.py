#!/usr/bin/env python3
"""Show validated weight download progress and saved prediction counts."""
import json
from pathlib import Path

root = Path("/root/autodl-tmp/ecg-repro-assets")
status = {}
for variant in ("7B", "32B"):
    model = root / "models" / f"Hulu-Med-{variant}"
    manifest = model / "download-manifest.json"
    if not manifest.is_file():
        status[variant] = {"model_download": "not started"}
        continue
    data = json.loads(manifest.read_text())
    total, downloaded, files = 0, 0, 0
    for entry in data["files"]:
        if not entry["path"].endswith(".safetensors"):
            continue
        size = entry["size"]
        total += size
        path = model / entry["path"]
        if path.is_file() and path.stat().st_size == size:
            downloaded += size
            files += 1
        elif path.with_suffix(".safetensors.chunks.json").exists():
            progress = json.loads(path.with_suffix(".safetensors.chunks.json").read_text())
            for index in progress["done"]:
                downloaded += min(progress["chunk_size"], size - index * progress["chunk_size"])
    status[variant] = {"weight_gb_downloaded": round(downloaded / 1e9, 2), "weight_gb_total": round(total / 1e9, 2), "percent": round(100 * downloaded / total, 2), "finalized_shards": files}
    runs = root / "results" / "paper-v1" / variant
    counts = {}
    for dataset in ("ptbxl", "mimic_iv_ecg"):
        counts[dataset] = len(list((runs / f"hulumed-hf_{variant}" / dataset).glob("*/*.json")))
    status[variant]["full_run_saved_samples"] = counts
    pipeline = runs / "pipeline-status.json"
    if pipeline.exists():
        status[variant]["pipeline"] = json.loads(pipeline.read_text())
    status[variant]["smoke_saved_samples"] = len(list(runs.glob(f"smoke-*/hulumed-hf_{variant}/*/*/*.json")))
print(json.dumps(status, indent=2, ensure_ascii=False))
