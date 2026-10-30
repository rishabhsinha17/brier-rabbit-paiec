"""Fetch the measurement-db core tables at the pinned revision bc8204d8, or verify a local copy against its checksums.

The dataset aims-foundations/measurement-db is gated on Hugging Face: accept its terms on the dataset page while logged
in, then authenticate with `huggingface-cli login` (or set HF_TOKEN in the environment). This script never reads,
prints or stores a token itself; huggingface_hub picks up the login.

usage (from the repository root):
  python scripts/fetch_measurement_db.py              # download the 47 core files (~83 MB) to ../data/measurement-db
  python scripts/fetch_measurement_db.py --verify     # check sizes and sha256 against scripts/data_manifest_bc8204d8.json
  python scripts/fetch_measurement_db.py --out DIR    # another location (paiec/data.py expects ../data/measurement-db)

Core = every file of the revision except traces, raw archives and reproduction checks (same filter as the copy used
for every number in the report). Requires huggingface_hub for downloading; --verify needs only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ID = "aims-foundations/measurement-db"
REVISION = "bc8204d811823da849c6686bf124d4ca9f82e4de"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "data" / "measurement-db"
MANIFEST = Path(__file__).resolve().parent / "data_manifest_bc8204d8.json"
MAX_BYTES = 100e6


def is_core(path: str) -> bool:
    return not ("/raw/" in path or path.startswith("raw/") or "reproduction" in path or "trace" in path.split("/")[-1])


def verify(out: Path) -> int:
    manifest = json.loads(MANIFEST.read_text())
    bad = 0
    for f in manifest["files"]:
        p = out / f["path"]
        if not p.is_file():
            print(f"MISSING  {f['path']}")
            bad += 1
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stat().st_size != f["bytes"] or h != f["sha256"]:
            print(f"MISMATCH {f['path']}")
            bad += 1
    print(f"{'ok' if not bad else 'FAIL'}: {manifest['n_files'] - bad}/{manifest['n_files']} files match revision "
          f"{manifest['revision'][:8]} ({out})")
    return 1 if bad else 0


def fetch(out: Path) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("pip install huggingface_hub (any recent version) to download; --verify works without it")
    api = HfApi()
    core = []
    for f in api.list_repo_tree(REPO_ID, repo_type="dataset", revision=REVISION, recursive=True):
        size = getattr(f, "size", None)
        if size is not None and is_core(f.path):
            core.append((f.path, size))
    total = sum(s for _, s in core)
    print(f"{len(core)} core files, {total / 1e6:.1f} MB at revision {REVISION[:8]} -> {out}", flush=True)
    if total > MAX_BYTES:
        sys.exit(f"refusing: core set is {total / 1e6:.0f} MB, expected about 83 MB")
    out.mkdir(parents=True, exist_ok=True)
    for i, (path, size) in enumerate(sorted(core), 1):
        hf_hub_download(REPO_ID, path, repo_type="dataset", revision=REVISION, local_dir=out)
        print(f"[{i}/{len(core)}] {size / 1e6:8.3f} MB  {path}", flush=True)
    (out / "PROVENANCE.json").write_text(json.dumps({
        "repo": REPO_ID, "revision": REVISION, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": sorted(p for p, _ in core), "bytes": total}, indent=2))
    return verify(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    return verify(out) if args.verify else fetch(out)


if __name__ == "__main__":
    raise SystemExit(main())
