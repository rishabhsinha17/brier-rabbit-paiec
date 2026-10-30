"""Snapshot the pre-registered upload plan before an upload batch (evidence for the report's pre-registration claims).

usage: python tools/prereg_snapshot.py "<what is about to be uploaded>"
Copies submissions/QUEUE.md to submissions/prereg/QUEUE_<UTC stamp>.md and appends the UTC time, sha256 and note to
submissions/prereg/LOG.csv. The repo is not under version control, so these hashed copies are the audit trail.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "submissions" / "QUEUE.md"
OUT = REPO / "submissions" / "prereg"


def main() -> None:
    note = " ".join(sys.argv[1:]).replace('"', "'")
    OUT.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = OUT / f"QUEUE_{stamp}.md"
    shutil.copy2(SRC, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    log = OUT / "LOG.csv"
    if not log.exists():
        log.write_text("utc,file,sha256,note\n")
    with log.open("a") as fh:
        fh.write(f'{stamp},{dst.name},{sha},"{note}"\n')
    print(f"snapshot {dst.name} sha256={sha[:16]}")
    if (REPO / ".git").exists():  # local-only repo (user-approved 2026-10-07): give the snapshot a commit hash
        import subprocess
        subprocess.run(["git", "add", "submissions/QUEUE.md", "submissions/prereg"], cwd=REPO, check=False)
        msg = (f"prereg {stamp}: {note}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        r = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO, capture_output=True, text=True)
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        print(f"git commit {h}" if r.returncode == 0 else f"git commit skipped ({(r.stdout + r.stderr).strip()[:80]})")


if __name__ == "__main__":
    main()
