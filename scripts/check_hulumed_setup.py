#!/usr/bin/env python3
"""Check local assets and CUDA without loading the 32B checkpoint."""
import argparse
import json
import os
from pathlib import Path


def record_path(meta, root):
    ecg = str(meta["ecg_id"])
    if meta["data_source"] == "ptbxl":
        return root / "records500" / f"{int(ecg) // 1000 * 1000:05d}" / f"{int(ecg):05d}_hr"
    subject = str(meta["subject_id"])
    return root / "files" / f"p{subject[:4]}" / f"p{subject}" / f"s{ecg}" / ecg


def select_samples(samples, limit=None, per_diagnosis=None):
    if limit:
        return samples[:limit]
    if per_diagnosis:
        from collections import Counter
        counts, selected = Counter(), []
        for sample in samples:
            dx = sample["metadata"]["target_dx"]
            if counts[dx] < per_diagnosis:
                selected.append(sample)
                counts[dx] += 1
        return selected
    return samples


def check_assets(datasets, limit=None, per_diagnosis=None):
    model = Path(os.environ["HULUMED_MODEL_PATH"])
    manifest = json.loads((model / "download-manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["type"] == "file":
            path = model / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Model file missing/incomplete: {path}")
    print(f"Model files complete: {manifest['revision']}")
    for dataset in datasets:
        path = Path(os.environ["BENCHMARK_DIR"]) / f"{dataset}.jsonl"
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        total_samples = len(samples)
        samples = select_samples(samples, limit, per_diagnosis)
        root = Path(os.environ["PTBXL_DIR" if dataset == "ptbxl" else "MIMIC_ECG_DIR"])
        records = {record_path(s["metadata"], root) for s in samples}
        missing = [str(p) + ext for p in sorted(records) for ext in (".hea", ".dat")
                   if not Path(str(p) + ext).is_file() or Path(str(p) + ext).stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} waveform files missing. First: {missing[0]}. "
                               "For MIMIC, set MIMIC_ECG_DIR to a legally obtained local files/ tree.")
        print(f"{dataset}: {len(samples)}/{total_samples} samples selected, {len(records)} ECGs, {len({s['metadata']['target_dx'] for s in samples})} diagnoses")


def check_runtime(require_two=True):
    import torch
    import transformers
    import accelerate
    from transformers import AutoConfig, AutoProcessor
    from transformers.image_utils import VideoInput  # required by custom processor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    assert transformers.__version__ == "4.51.2", transformers.__version__
    count = torch.cuda.device_count()
    if count < (2 if require_two else 1):
        raise RuntimeError(f"Visible CUDA GPUs: {count}; restart with two H800-80GB GPUs before inference.")
    for i in range(count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GiB")
        q = torch.randn(1, 16, 4, 64, dtype=torch.bfloat16, device=f"cuda:{i}")
        if os.environ.get("HULUMED_ATTN_IMPLEMENTATION", "flash_attention_2") == "flash_attention_2":
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, q, q, causal=True)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        assert out.shape == q.shape and torch.isfinite(out).all()
    model = os.environ["HULUMED_MODEL_PATH"]
    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], model, local_files_only=True)
    print(f"Runtime OK: torch={torch.__version__}, transformers={transformers.__version__}, accelerate={accelerate.__version__}, class={cls.__name__}, processor={type(processor).__name__}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", nargs="*", choices=["ptbxl", "mimic_iv_ecg"], default=["ptbxl"])
    p.add_argument("--allow-single-gpu", action="store_true")
    p.add_argument("--runtime-only", action="store_true")
    args = p.parse_args()
    if not args.runtime_only:
        check_assets(args.dataset)
    check_runtime(require_two=not args.allow_single_gpu)


if __name__ == "__main__":
    main()
