#!/usr/bin/env python3
"""Run the official full teacher-forced protocol, with atomic, resumable output."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path

from check_hulumed_setup import check_assets, check_runtime, select_samples


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def complete_result(path, sample):
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        for key in ("id", "ecg_id", "target_dx", "dx_label"):
            if data["metadata"][key] != sample["metadata"][key]:
                return False
        steps = [data["data"]["initial_diagnostic_question"]]
        for loop in data["data"]["reasoning"]:
            for step in loop.values():
                steps.extend(step if isinstance(step, list) else [step])
        return all(isinstance(s.get("model_response"), str) and s["model_response"].strip() for s in steps)
    except (ValueError, KeyError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", nargs="+", choices=["ptbxl", "mimic_iv_ecg"])
    p.add_argument("--limit", type=int, help="Smoke test N samples per dataset in a separate output directory")
    p.add_argument("--per-diagnosis", type=int, help="Smoke test N samples from each of the 17 diagnoses")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.per_diagnosis is not None and (args.per_diagnosis < 1 or args.limit):
        p.error("--per-diagnosis must be positive and cannot be combined with --limit")
    variant = os.environ["HULUMED_VARIANT"]
    model_name = f"hulumed-hf_{variant}"
    check_assets(args.dataset, args.limit, args.per_diagnosis)
    check_runtime(require_two=variant == "32B")
    root = Path(os.environ["RESULTS_DIR"])
    if args.limit:
        root = root / f"smoke-{args.limit}"
    if args.per_diagnosis:
        root = root / f"smoke-per-diagnosis-{args.per_diagnosis}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("w") as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another inference process is writing to {root}")
        benchmark = Path(os.environ["BENCHMARK_DIR"])
        model_dir = Path(os.environ["HULUMED_MODEL_PATH"])
        repo = Path(os.environ["ECG_REPRO_REPO"])
        fingerprint = {
            "protocol": "official-default-full-chain",
            "condensed_chat": True,
            "model": f"ZJU-AI4H/Hulu-Med-{variant}",
            "model_revision": json.loads((model_dir / "download-manifest.json").read_text())["revision"],
            "dataset_sha256": {ds: sha(benchmark / f"{ds}.jsonl") for ds in ("ptbxl", "mimic_iv_ecg")},
            "code_sha256": {f: sha(repo / f) for f in ("ecg_reasoning_benchmark/inference.py", "ecg_reasoning_benchmark/utils.py", "ecg_reasoning_benchmark/models/hulumed/hulumed_hf.py")},
            "attention": os.environ["HULUMED_ATTN_IMPLEMENTATION"],
            "generation": {"max_new_tokens": 1024, "do_sample": False, "temperature": 0.0, "num_beams": 1},
        }
        manifest = root / "run-manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
            raise SystemExit("Run configuration changed; choose a new RESULTS_DIR to avoid mixing results.")
        atomic_json(manifest, fingerprint)
        from ecg_reasoning_benchmark import Inferencer
        from ecg_reasoning_benchmark.models import build_model
        from tqdm import tqdm
        import torch
        torch.manual_seed(42)
        model = build_model("hulumed-hf", model_variant=variant)
        device_map = getattr(model.model, "hf_device_map", {})
        print("Model device map:", device_map, flush=True)
        if any(str(d) in ("cpu", "disk") for d in device_map.values()):
            raise RuntimeError("Model spilled to CPU/disk; inspect GPU memory before full inference")
        inferencer = Inferencer(model)
        for ds in args.dataset:
            samples = [json.loads(line) for line in (benchmark / f"{ds}.jsonl").read_text().splitlines()]
            samples = select_samples(samples, args.limit, args.per_diagnosis)
            ecg_dir = os.environ["PTBXL_DIR" if ds == "ptbxl" else "MIMIC_ECG_DIR"]
            for sample in tqdm(samples, desc=ds, ncols=110):
                meta = sample["metadata"]
                path = root / model_name / ds / meta["target_dx"] / f"{meta['id']}.json"
                if complete_result(path, sample):
                    continue
                result = inferencer.inference(copy.deepcopy(sample), ecg_dir, enable_condensed_chat=True)
                atomic_json(path, result)
            invalid = [s["metadata"]["id"] for s in samples if not complete_result(root / model_name / ds / s["metadata"]["target_dx"] / f"{s['metadata']['id']}.json", s)]
            if invalid:
                raise RuntimeError(f"{ds}: {len(invalid)} incomplete/empty responses, first IDs: {invalid[:10]}; rerun to retry")
            print(f"Complete: {ds}, {len(samples)} samples in {root}", flush=True)


if __name__ == "__main__":
    main()
