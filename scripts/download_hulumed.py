#!/usr/bin/env python3
"""Download pinned Hulu-Med weights via HF mirror; resume ranges and verify SHA256."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests

REVISIONS = {"32B": "dfac09d0653d00ac974c02a09caacaaab22fa819", "7B": "258594714a0d3835eb2c9e4cc165a4242e606d71"}
CHUNK = 16 * 1024 * 1024
local = threading.local()


def session():
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.trust_env = False
    return local.session


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list(REVISIONS), default="32B")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--weight-source", choices=["hf", "modelscope"], default="hf")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    repo = f"ZJU-AI4H/Hulu-Med-{args.variant}"
    revision = REVISIONS[args.variant]
    if args.model_dir is None:
        args.model_dir = Path(f"/root/autodl-tmp/ecg-repro-assets/models/Hulu-Med-{args.variant}")
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    response = session().get(f"{endpoint}/api/models/{repo}/tree/{revision}?recursive=true&expand=false", timeout=60)
    response.raise_for_status()
    files = response.json()
    msfiles = {}
    if args.weight_source == "modelscope":
        response = session().get("https://modelscope.cn/api/v1/models/Med-Team/Hulu-Med/repo/files?Revision=master&Recursive=true", timeout=60)
        response.raise_for_status()
        msfiles = {f["Path"]: f for f in response.json()["Data"]["Files"]}
    (args.model_dir / "download-manifest.json").write_text(json.dumps({"repository": repo, "revision": revision, "endpoint": endpoint, "weight_source": args.weight_source, "files": files}, indent=2))
    lock = threading.Lock()
    jobs, states = [], {}
    started = time.monotonic()
    transferred = 0
    for entry in files:
        if entry["type"] != "file":
            continue
        name, size = entry["path"], entry["size"]
        target = args.model_dir / name
        url = f"{endpoint}/{repo}/resolve/{revision}/{name}"
        if not name.endswith(".safetensors"):
            if not target.exists():
                r = session().get(url, timeout=60)
                r.raise_for_status()
                target.write_bytes(r.content)
            continue
        sha = entry["lfs"]["oid"]
        if args.weight_source == "modelscope":
            mspath = f"Hulu-Med-{args.variant}-HF/{name}"
            ms = msfiles[mspath]
            if ms["Size"] != size or ms["Sha256"] != sha:
                raise RuntimeError(f"ModelScope weights do not match pinned HF bytes: {name}")
            url = f"https://modelscope.cn/models/Med-Team/Hulu-Med/resolve/{ms['Revision']}/{mspath}"
        if target.exists():
            if target.stat().st_size == size and digest(target) == sha:
                print(f"Verified existing {name}", flush=True)
                continue
            raise RuntimeError(f"Existing file has wrong size/hash: {target}; inspect before replacing")
        part = target.with_suffix(target.suffix + ".part")
        progress = target.with_suffix(target.suffix + ".chunks.json")
        done = set()
        if progress.exists() and part.exists():
            saved = json.loads(progress.read_text())
            if saved["sha256"] == sha and saved["chunk_size"] == CHUNK and part.stat().st_size == size:
                done = set(saved["done"])
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        states[name] = dict(fd=fd, done=done, progress=progress, part=part, target=target, sha=sha, size=size)
        for i, start in enumerate(range(0, size, CHUNK)):
            if i not in done:
                jobs.append((name, url, i, start, min(start + CHUNK, size) - 1))

    def fetch(job):
        nonlocal transferred
        name, url, i, start, end = job
        state = states[name]
        for attempt in range(6):
            try:
                # Query distinguishes independently cached range requests at the mirror.
                with session().get(url + f"?download=true&chunk={i}", headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 90)) as r:
                    r.raise_for_status()
                    expected = f"bytes {start}-{end}/{state['size']}"
                    if r.status_code != 206 or r.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Range mismatch: {r.status_code} {r.headers.get('Content-Range')} != {expected}")
                    pos = start
                    for block in r.iter_content(1024 * 1024):
                        if pos + len(block) > end + 1:
                            raise RuntimeError("Range response exceeded requested size")
                        offset = 0
                        while offset < len(block):
                            offset += os.pwrite(state["fd"], block[offset:], pos + offset)
                        pos += len(block)
                    if pos != end + 1:
                        raise RuntimeError(f"Truncated range: {pos} != {end + 1}")
                with lock:
                    state["done"].add(i)
                    tmp = state["progress"].with_suffix(".tmp")
                    tmp.write_text(json.dumps({"sha256": state["sha"], "chunk_size": CHUNK, "done": sorted(state["done"])}))
                    tmp.replace(state["progress"])
                    transferred += end - start + 1
                return
            except Exception as exc:
                if attempt == 5:
                    raise RuntimeError(f"{name} chunk {i}: {exc}") from exc
                time.sleep(min(2 ** attempt, 16))

    print(f"Downloading {len(jobs)} missing ranges, workers={args.workers}, revision={revision}, source={args.weight_source}", flush=True)
    failures = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    f.result()
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"ERROR {exc}", flush=True)
                if n % 16 == 0 or n == len(jobs):
                    elapsed = time.monotonic() - started
                    print(f"ranges {n}/{len(jobs)}, transferred {transferred / 1e9:.2f} GB, {transferred / elapsed / 1e6:.2f} MB/s", flush=True)
    finally:
        for state in states.values():
            os.fsync(state["fd"])
            os.close(state["fd"])
    if failures:
        raise RuntimeError(f"{len(failures)} ranges failed. Re-run to resume. First: {failures[0]}")
    verified = []
    for name, state in states.items():
        actual = digest(state["part"])
        if actual != state["sha"]:
            raise RuntimeError(f"SHA256 mismatch: {name}; keep .part for inspection; remove its .chunks.json to redownload")
        state["part"].replace(state["target"])
        state["progress"].unlink()
        verified.append({"file": name, "size": state["size"], "sha256": actual})
        print(f"SHA256 OK {name}", flush=True)
    print("All pinned files downloaded and weights verified.", flush=True)


if __name__ == "__main__":
    main()
