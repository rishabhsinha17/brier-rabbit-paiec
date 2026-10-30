"""Package a submission directory, run the official kit checks, and record it in the ledger.

usage: python tools/build_submission.py submission_src/<name> [--note "..."]
Writes submissions/<name>.zip and appends a row to submissions/LEDGER.md. Never uploads anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KIT = REPO.parent
OUT = REPO / "submissions"
SECRET_PATTERNS = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|api_key\"?\s*[:=]\s*\"[^\"]+\")")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    src = args.src.resolve()
    name = src.name
    OUT.mkdir(exist_ok=True)
    zpath = OUT / f"{name}.zip"

    files = sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store")
    for p in files:
        if p.suffix in {".py", ".json", ".txt", ".yaml", ".yml"} and SECRET_PATTERNS.search(p.read_text(errors="ignore")):
            print(f"refusing: possible credential in {p.relative_to(src)}", file=sys.stderr)
            return 1
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            info = zipfile.ZipInfo(str(p.relative_to(src)), date_time=(2026, 1, 1, 0, 0, 0))  # reproducible bytes
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())

    py = sys.executable
    checks = [
        [py, str(KIT / "tools" / "check_submission_zip.py"), str(zpath)],
        [py, str(KIT / "tools" / "run_smoke_test.py"), str(src)],
        [py, str(KIT / "tools" / "run_smoke_test.py"), str(src), "--concurrency", "4"],
    ]
    for cmd in checks:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=KIT)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {' '.join(Path(c).name for c in cmd[1:])}: {tail[0]}")
        if r.returncode:
            return r.returncode

    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    builds = OUT / "builds.csv"  # build log; uploads and scores live in ledger.csv (tools/ledger.py)
    if not builds.exists():
        builds.write_text("built_utc,zip,sha256,bytes,note\n")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    with builds.open("a") as fh:
        note = args.note.replace('"', "'")
        fh.write(f'{stamp},{zpath.name},{sha},{zpath.stat().st_size},"{note}"\n')
    print(f"built {zpath} sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
