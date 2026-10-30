"""Shared helpers for the round-5 runtime/robustness audit of the shipped composite zips.

- shipped(name): extract submissions/<name>.zip into audit/shipped/<name>/ after checking its sha256 against
  submissions/builds.csv, then import model.py the way a platform worker does (fresh module from its file path, repo
  and paiec off sys.path, paiec imports blocked).
- draw(...): synthetic draw shaped like a formative draw (pairs over benchmarks, persistent 50/50 split per pair, random
  acquisition order, partners on a benchmark answer the same item texts), returning the platform's `labeled` and the
  unique evaluation inputs at a budget. Uses only public-prior subject names and made-up texts; no live data.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.abc
import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
REPO = next(d for d in AUDIT.parents if (d / "tools" / "make_submission.py").is_file())  # works from any depth
KIT = REPO.parent


def builds_sha(zip_name: str) -> str:
    rows = [r for r in csv.DictReader((REPO / "submissions" / "builds.csv").open()) if r["zip"] == zip_name]
    return rows[-1]["sha256"]


_EXTRACTED: dict[str, Path] = {}


def extract(name: str) -> Path:
    if name in _EXTRACTED:
        return _EXTRACTED[name]
    z = REPO / "submissions" / f"{name}.zip"
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    if sha != builds_sha(z.name):
        raise SystemExit(f"{z.name}: sha256 {sha[:12]} does not match builds.csv")
    out = AUDIT / "shipped" / name
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)
    _EXTRACTED[name] = out
    return out


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "paiec" or name.startswith("paiec."):
            raise ImportError(f"shipped model imports {name}")
        return None


@contextlib.contextmanager
def platform_like():
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in {REPO, REPO / "tools", AUDIT}]
    blocker = _Block()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.path[:] = saved


def shipped(name: str, module_name: str | None = None):
    d = extract(name)
    spec = importlib.util.spec_from_file_location(module_name or f"participant_model_{name}", d / "model.py")
    mod = importlib.util.module_from_spec(spec)
    with platform_like():
        spec.loader.exec_module(mod)
    return mod, d


def prior_names(d: Path) -> list[str]:
    prior = json.loads((d / "subject_prior.json").read_text())
    return sorted(prior["names"])


def draw(known_names, *, n_pairs=9, n_bench=7, layout=None, n_items=120, budget=31, seed=0, big_bench=1,
         big_kb=100):
    """Return (labeled at `budget`, unique eval inputs). layout[i] = benchmark index of pair i (default: the formative
    shape, one pair per benchmark plus extras on the first benchmarks). Partners on a benchmark share item texts.
    big_bench benchmarks get ~big_kb KB item texts (ResearchCodeBench-like)."""
    rng = random.Random(seed)
    if layout is None:
        layout = [j % n_bench for j in range(n_pairs)]
    benches = [f"benchmark_{100000 + 7919 * j}" for j in range(n_bench)]  # synthetic, not platform IDs
    items = {}
    for j, b in enumerate(benches):
        pad = "x" * (big_kb * 1000 if j < big_bench else rng.randrange(200, 5000))
        items[b] = [{"item_content": f"{b} problem {i}: " + pad + f" end {i}",
                     "item_features": f"lang=l{i % 6};difficulty=d{i % 3}", "interactors": "",
                     "benchmark_id": b} for i in range(n_items)]
    subjects = []
    for i in range(len(layout)):
        if i % 2 == 0 and known_names:
            subjects.append({"normalized_name": known_names[(7 * i) % len(known_names)], "provider": "",
                             "release_date": "2025-06-01", "access_date": "", "harness": "", "reasoning_effort": "",
                             "harness_version": "", "subject_features_extra": ""})
        else:
            subjects.append({"normalized_name": f"Unseen Model {i}", "provider": ["OpenAI", "Google", "Anthropic"][i % 3],
                             "release_date": f"2026-0{1 + i % 9}-15", "access_date": "", "harness": f"agent-{i % 2}",
                             "reasoning_effort": ["low", "medium", "high"][i % 3], "harness_version": "1.0",
                             "subject_features_extra": f"source_model_name=Unseen Model {i}"})
    labeled, evals = [], []
    for i, s in enumerate(subjects):
        b = benches[layout[i]]
        idx = list(range(n_items))
        rng.shuffle(idx)                       # persistent 50/50 split, then random acquisition order
        acq, ev = idx[: (n_items + 1) // 2], idx[(n_items + 1) // 2:]
        level = rng.choice([-3.0, -1.0, 0.0, 1.0])
        for it in acq[:budget]:
            p = 1.0 / (1.0 + 2.718281828 ** (-(level + rng.gauss(0, 1.5))))
            labeled.append([[s, items[b][it]], int(rng.random() < p)])
        evals += [[s, items[b][it]] for it in ev]
    seen, unique = set(), []
    for inp in evals:
        k = json.dumps(inp, sort_keys=True)
        if k not in seen:
            seen.add(k)
            unique.append(inp)
    return labeled, unique
