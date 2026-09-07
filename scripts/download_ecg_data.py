#!/usr/bin/env python3
"""Download and verify only the WFDB records referenced by this benchmark.

Network requests explicitly ignore proxy environment variables. PTB-XL files
are checked against the official SHA256SUMS as well as their WFDB headers.
HTTP 403 can indicate regional legal/policy restrictions. Consult the official
project page; this script does not bypass restrictions. Existing lawfully
obtained local records can be checked with --verify-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import requests
import wfdb


DATASETS = {
    "ptbxl": {
        "directory": "ptb-xl",
        "base_url": "https://physionet.org/files/ptb-xl/1.0.3/",
        "metadata": ["LICENSE.txt", "SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv"],
    },
    "mimic_iv_ecg": {
        "directory": "mimic-iv-ecg",
        "base_url": "https://physionet.org/files/mimic-iv-ecg/1.0/",
        "metadata": ["LICENSE.txt"],
    },
}
THREAD_STATE = threading.local()


class AccessDenied(RuntimeError):
    pass


def session() -> requests.Session:
    if not hasattr(THREAD_STATE, "session"):
        client = requests.Session()
        client.trust_env = False
        client.headers["User-Agent"] = "ecg-reasoning-benchmark-reproduction/1.0"
        username = os.environ.get("PHYSIONET_USERNAME")
        password = os.environ.get("PHYSIONET_PASSWORD")
        if username and password:
            client.auth = (username, password)
        THREAD_STATE.session = client
    return THREAD_STATE.session


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, *, attempts: int, expected_hash: str | None = None,
             expected_size: int | None = None, header_name: str | None = None) -> bool:
    def valid(candidate: Path) -> bool:
        if not candidate.is_file() or not candidate.stat().st_size:
            return False
        if expected_size is not None and candidate.stat().st_size != expected_size:
            return False
        if expected_hash is not None and sha256(candidate) != expected_hash:
            return False
        with candidate.open("rb") as handle:
            first = handle.read(256)
        if first.lstrip().lower().startswith((b"<!doctype", b"<html")):
            return False
        if header_name is not None and not first.startswith(f"{header_name} 12 500 ".encode()):
            return False
        return True

    if valid(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            with session().get(url, stream=True, timeout=(20, 90)) as response:
                if response.status_code in (401, 403):
                    raise AccessDenied(f"HTTP {response.status_code}: {url}; access refused. Check the official project page for regional legal/policy restrictions")
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 256):
                        handle.write(block)
            if not valid(partial):
                raise ValueError(f"Downloaded file failed validation: {path}")
            partial.replace(path)
            return True
        except AccessDenied:
            raise
        except (requests.RequestException, OSError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise AssertionError("unreachable")


def records_from_manifest(manifest: Path, dataset: str) -> tuple[list[str], int]:
    paths = set()
    entries = 0
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            metadata = json.loads(line)["metadata"]
            ecg_id = str(metadata["ecg_id"])
            if dataset == "ptbxl":
                number = int(ecg_id)
                path = f"records500/{number // 1000 * 1000:05d}/{number:05d}_hr"
            else:
                subject = str(metadata["subject_id"])
                path = f"files/p{subject[:4]}/p{subject}/s{ecg_id}/{ecg_id}"
            paths.add(path)
            entries += 1
    return sorted(paths), entries


def validate_record(path: Path) -> dict:
    record = wfdb.rdrecord(str(path), physical=False)
    if record.fs != 500 or record.n_sig != 12 or record.d_signal.shape != (5000, 12):
        raise ValueError(f"Unexpected waveform dimensions/fs: {path}: {record.d_signal.shape}, {record.fs}")
    if record.checksum is not None:
        actual = record.d_signal.astype(np.int64).sum(axis=0) % 65536
        expected = np.asarray(record.checksum, dtype=np.int64) % 65536
        if not np.array_equal(actual, expected):
            raise ValueError(f"WFDB signal checksum mismatch: {path}")
    return {"shape": list(record.d_signal.shape), "fs": record.fs}


def process_record(root: Path, relative: str, base_url: str, hashes: dict[str, str],
                   attempts: int, verify_only: bool) -> dict:
    path = root / relative
    changed = 0
    if verify_only:
        for extension in (".hea", ".dat"):
            expected = hashes.get(relative + extension)
            if expected is not None and sha256(path.with_suffix(extension)) != expected:
                raise ValueError(f"Official SHA256 mismatch: {relative}{extension}")
    if not verify_only:
        changed += download(base_url + relative + ".hea", path.with_suffix(".hea"),
                            attempts=attempts, expected_hash=hashes.get(relative + ".hea"),
                            header_name=path.name)
        header = wfdb.rdheader(str(path))
        if header.n_sig != 12 or header.fs != 500 or header.sig_len != 5000:
            raise ValueError(f"Unexpected WFDB header: {relative}")
        if set(header.fmt) != {"16"} or set(header.file_name) != {path.name + ".dat"}:
            raise ValueError(f"Unsupported WFDB storage layout: {relative}")
        expected_size = header.n_sig * header.sig_len * 2
        changed += download(base_url + relative + ".dat", path.with_suffix(".dat"),
                            attempts=attempts, expected_hash=hashes.get(relative + ".dat"),
                            expected_size=expected_size)
    try:
        validate_record(path)
    except ValueError:
        if verify_only:
            raise
        # A same-size partial/corrupt dat file must not pass a resumed run.
        dat = path.with_suffix(".dat")
        if dat.exists():
            dat.unlink()
        changed += download(base_url + relative + ".dat", dat, attempts=attempts,
                            expected_hash=hashes.get(relative + ".dat"), expected_size=120000)
        validate_record(path)
    return {"downloaded_files": changed,
            "waveform_bytes": sum(path.with_suffix(ext).stat().st_size for ext in (".hea", ".dat"))}


def run_dataset(args: argparse.Namespace, name: str) -> dict:
    config = DATASETS[name]
    root = args.data_root / config["directory"]
    root.mkdir(parents=True, exist_ok=True)
    records, entries = records_from_manifest(args.benchmark_dir / f"{name}.jsonl", name)
    manifests = [{"directory": str(args.benchmark_dir), "entries": entries, "records": len(records)}]
    union = set(records)
    for directory in args.additional_benchmark_dir:
        additional, count = records_from_manifest(directory / f"{name}.jsonl", name)
        manifests.append({"directory": str(directory), "entries": count, "records": len(additional)})
        union.update(additional)
    records = sorted(union)
    summary = {"dataset": name, "base_url": config["base_url"], "data_root": str(root),
               "benchmark_entries": entries, "manifests": manifests, "expected_records": len(records),
               "verified_records": 0, "downloaded_files": 0, "waveform_files": 0,
               "waveform_bytes": 0, "metadata_errors": [], "failures": [],
               "shape": [5000, 12], "fs": 500, "started_at": datetime.now(timezone.utc).isoformat()}
    verified = set()
    try:
        # Check actual file access before scheduling thousands of denied requests.
        if not args.verify_only:
            relative = records[0] + ".hea"
            download(config["base_url"] + relative, root / relative, attempts=args.retries,
                     header_name=Path(records[0]).name)
        for metadata in config["metadata"]:
            if args.verify_only:
                continue
            try:
                log(f"{name}: preparing metadata {metadata}")
                download(config["base_url"] + metadata, root / metadata, attempts=args.retries)
            except Exception as exc:
                summary["metadata_errors"].append({"file": metadata, "error": str(exc)})
        hashes = {}
        checksums = root / "SHA256SUMS.txt"
        if checksums.exists():
            for line in checksums.read_text().splitlines():
                digest, relative = line.split(maxsplit=1)
                hashes[relative.lstrip("*").removeprefix("./")] = digest
        if name == "ptbxl" and not hashes:
            raise ValueError("Official SHA256SUMS.txt is required to verify PTB-XL downloads")
        log(f"{name}: {entries} benchmark entries, {len(records)} distinct records, {args.workers} workers")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_record, root, relative, config["base_url"], hashes,
                                   args.retries, args.verify_only): relative for relative in records}
            for completed, future in enumerate(as_completed(futures), 1):
                relative = futures[future]
                try:
                    result = future.result()
                    summary["verified_records"] += 1
                    verified.add(relative)
                    summary["waveform_files"] += 2
                    summary["waveform_bytes"] += result["waveform_bytes"]
                    summary["downloaded_files"] += result["downloaded_files"]
                except Exception as exc:
                    summary["failures"].append({"record": relative, "error": str(exc)})
                    log(f"{name}: FAILED {relative}: {exc}")
                if completed % 100 == 0 or completed == len(records):
                    log(f"{name}: {completed}/{len(records)} completed; verified={summary['verified_records']}; "
                        f"failed={len(summary['failures'])}; elapsed={time.monotonic() - started:.0f}s")
    except Exception as exc:
        summary["failures"].append({"record": "dataset_preflight", "error": str(exc)})
        log(f"{name}: {exc}")
    summary["complete"] = summary["verified_records"] == len(records) and not summary["failures"]
    missing = sorted(set(records) - verified)
    summary["missing_records"] = len(missing)
    (root / "required_records.txt").write_text("".join(record + "\n" for record in records))
    (root / "missing_records.txt").write_text("".join(record + "\n" for record in missing))
    for manifest in manifests:
        subset, _ = records_from_manifest(Path(manifest["directory"]) / f"{name}.jsonl", name)
        manifest["verified_records"] = len(set(subset) & verified)
        manifest["complete"] = set(subset) <= verified
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["metadata_bytes"] = sum((root / name).stat().st_size for name in config["metadata"] if (root / name).is_file())
    (root / "download_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/ecg-repro-assets/data"))
    default_benchmark = Path("/root/autodl-tmp/ecg-repro-assets/data/benchmark-paper-v1")
    parser.add_argument("--benchmark-dir", "--manifest-dir", dest="benchmark_dir", type=Path,
                        default=default_benchmark if default_benchmark.exists() else Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--additional-benchmark-dir", action="append", type=Path, default=[],
                        help="Also fetch the union with this benchmark manifest directory; repeatable")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")
    args.data_root = args.data_root.resolve()
    args.benchmark_dir = args.benchmark_dir.resolve()
    args.additional_benchmark_dir = [path.resolve() for path in args.additional_benchmark_dir]
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(args, name) for name in names]
    args.data_root.mkdir(parents=True, exist_ok=True)
    combined = []
    for config in DATASETS.values():
        summary_path = args.data_root / config["directory"] / "download_summary.json"
        if summary_path.exists():
            combined.append(json.loads(summary_path.read_text()))
    (args.data_root / "download_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    log(json.dumps(summaries, indent=2))
    return 0 if all(item["complete"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
