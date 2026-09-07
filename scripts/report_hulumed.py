#!/usr/bin/env python3
"""Write a concise, explicitly scoped summary of saved heuristic evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ptbxl", "mimic_iv_ecg"], default="ptbxl")
    args = p.parse_args()
    variant = os.environ["HULUMED_VARIANT"]
    path = Path(os.environ["EVAL_DIR"]) / "heuristic" / args.dataset / "total.csv"
    rows = list(csv.DictReader(path.open()))
    row = next(r for r in rows if r["model"] == f"hulumed-hf_{variant}")
    benchmark = Path(os.environ["BENCHMARK_DIR"]) / f"{args.dataset}.jsonl"
    expected = sum(1 for line in benchmark.open() if line.strip())
    n = int(row["idq_total"])
    results = Path(os.environ["RESULTS_DIR"]) / f"hulumed-hf_{variant}" / args.dataset
    predictions = [json.loads(p.read_text()) for p in results.glob("*/*.json")]
    positive = sum(bool(d["metadata"]["dx_label"]) for d in predictions)
    summary = {
        "model": f"Hulu-Med-{variant}", "dataset": args.dataset,
        "evaluated_samples": n, "expected_full_dataset_samples": expected,
        "scope": "full_dataset" if n == expected else "subset_only",
        "positive_samples": positive,
        "negative_samples": len(predictions) - positive,
        "diagnoses_covered": len({d["metadata"]["target_dx"] for d in predictions}),
        "evaluator": "heuristic (paper uses Gemini)",
        "idq_correct": int(row["idq_correct"]),
        "gt_rda_correct": int(row["gt_reasoning_based_diagnosis_correct"]),
        "gt_rda_total": int(row["gt_reasoning_based_diagnosis_total"]),
        "idq_percent": 100 * int(row["idq_correct"]) / n,
        "gt_rda_percent": 100 * int(row["gt_reasoning_based_diagnosis_correct"]) / int(row["gt_reasoning_based_diagnosis_total"]),
        "csv": str(path),
        "limitation": "Subset scores are pipeline checks, not paper metric reproduction. The first sample of each diagnosis is positive; it is not a balanced accuracy estimate. Dataset/code release and judge must be matched for a paper comparison.",
    }
    target = path.parent / "reproduction-summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
