"""Check that a built submission reproduces the research model exactly on protocol replays.

usage: python tools/verify_shipped.py submission_src/<name> [--benches real_webagents,matharena] [--runs 1]
Imports submission_src/<name>/model.py platform-like (repo off sys.path, paiec imports blocked), builds the research model from the same params.json and
subject_prior.json (paiec.hier.HierModel, or the plug-in class named by "class" in params.json), replays
streaming_alc_v1 on public draws (k=1 and k=2) and fails unless max |p_shipped - p_research| < 1e-12.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paiec.data import load_responses  # noqa: E402
from paiec.evaluate import FoldData  # noqa: E402
from paiec.hier import HierModel  # noqa: E402
from paiec.protocol import run_protocol  # noqa: E402


class _BlockPaiec(importlib.abc.MetaPathFinder):
    """The platform has no paiec package: any import of it from the shipped code must fail loudly."""

    def find_spec(self, name, path=None, target=None):
        if name == "paiec" or name.startswith("paiec."):
            raise ImportError(f"shipped model imports {name}")
        return None


@contextlib.contextmanager
def platform_like():
    """Run shipped code as the platform would: repo off sys.path, paiec modules hidden and blocked."""
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "paiec" or k.startswith("paiec.")}
    repo_dirs = {REPO, REPO / "tools"}
    sys.path[:] = [d for d in sys.path if Path(d or ".").resolve() not in repo_dirs]
    for k in saved_mods:
        del sys.modules[k]
    blocker = _BlockPaiec()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.path[:] = saved_path
        sys.modules.update(saved_mods)


def research_model(params: dict, prior: dict):
    params = dict(params)
    cls_path = params.pop("class", None)
    cls = HierModel
    if cls_path:
        mod_name, cls_name = cls_path.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(prior, params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--benches", default="real_webagents,matharena")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    src = args.src.resolve()
    spec = importlib.util.spec_from_file_location("shipped_model", src / "model.py")
    shipped = importlib.util.module_from_spec(spec)
    with platform_like():
        spec.loader.exec_module(shipped)
    research = research_model(json.loads((src / "params.json").read_text()),
                              json.loads((src / "subject_prior.json").read_text()))

    benches = args.benches.split(",")
    fd = FoldData(load_responses(), benches)
    worst, calls = 0.0, 0

    def both(inp, labeled):
        nonlocal worst, calls
        with platform_like():
            a = shipped.predict(inp, labeled)
        b = research.predict(inp, labeled)
        worst = max(worst, abs(a - b))
        calls += 1
        return a

    for bench in benches:
        for k in (1, 2):
            for _, chosen in fd.draws(bench, k, args.runs):
                run_protocol(chosen, both)
    ok = worst < 1e-12 and calls > 0
    print(f"[{'ok' if ok else 'FAIL'}] {src.name}: {calls} predictions, max |shipped - research| = {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
