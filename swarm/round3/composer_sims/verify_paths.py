"""Non-vacuous shipped-vs-research check for a ComboModel build: every component path must be exercised.

tools/verify_shipped.py replays REAL and MathArena with known subject names, so the attribute (subjcal) path never runs
and name/harness coincidences (pool_sim) are rare. This replays, with the shipped model.py imported platform-like
(tools/verify_shipped.platform_like: repo off sys.path, paiec imports blocked):
  (a) benchmark-keyed k=2 draws on all four public benchmarks (cavity path: partner-labelled evaluation items);
  (b) REAL x ResearchCodeBench draws with the SAME canonical model name on both benchmarks and decoupled subject dicts
      (pool_sim name path, as in the hidden pool where no subject dict spans benchmarks);
  (c) the draws of (a) with every subject renamed to an unseen name (subjcal attribute path).
Reports max |shipped - research| (must be 0) and, per part, how many calls differ from plain HierModel (must be > 0).
usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round3/composer_sims/verify_paths.py \
           submission_src/test_combo_all
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "swarm/round2/skeptic_plugins_sims"))
from common import draws, load_cache  # noqa: E402
from verify_shipped import platform_like, research_model  # noqa: E402

from paiec.hier import HierModel  # noqa: E402
from paiec.protocol import materialize, run_protocol, unit  # noqa: E402
from paiec.subjects import name_key  # noqa: E402


def main():
    src = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location("shipped_model", src / "model.py")
    shipped = importlib.util.module_from_spec(spec)
    with platform_like():
        spec.loader.exec_module(shipped)
    params = json.loads((src / "params.json").read_text())
    prior = json.loads((src / "subject_prior.json").read_text())
    research = research_model(params, prior)
    plain = HierModel(prior, {k: v for k, v in params.items() if k != "class"})
    print(f"shipped m0 {shipped._MODEL.p['m0']!r} research m0 {research.p['m0']!r}")
    raw, _ = load_cache()
    stats = {}

    def runner(part):
        st = stats.setdefault(part, {"calls": 0, "worst": 0.0, "differs": 0})

        def both(inp, labeled):
            with platform_like():
                a = shipped.predict(inp, labeled)
            b = research.predict(inp, labeled)
            c = plain.predict(inp, labeled)
            st["calls"] += 1
            st["worst"] = max(st["worst"], abs(a - b))
            st["differs"] += int(abs(b - c) > 1e-12)
            return a
        return both

    # (a) benchmark-keyed k=2 draws, and (c) the same with unseen names
    for bench in ("real_webagents", "researchcodebench", "matharena", "multi_swebench"):
        for r, pairs in draws(bench, 2, 1, seed0=3, subsample="bench"):
            run_protocol(pairs, runner("a bench-keyed k=2"), seed=r)
            for j, pr in enumerate(pairs):
                pr.subject = dict(pr.subject, normalized_name=f"Unseen Model {bench} {j}", subject_features_extra="")
            run_protocol(pairs, runner("c unseen names (attr path)"), seed=r)
    # (b) same canonical name on REAL and RCB, decoupled dicts
    held = ("real_webagents", "researchcodebench")
    names = {b: {} for b in held}
    for b in held:
        for rp in raw[b]:
            nk = name_key(rp["subject"])
            if nk:
                names[b].setdefault(nk, []).append(rp)
    common = sorted(set(names[held[0]]) & set(names[held[1]]))[:3]
    for nm in common:
        pairs = []
        for b in held:
            rp = sorted(names[b][nm], key=lambda x: unit("xn", 0, x["key"]))[0]
            pairs += materialize([rp], size_range=(88, 226), seed=0)
        for pr in pairs:
            pr.subject = dict(pr.subject, access_date="decoupled-" + pr.bench)
        run_protocol(pairs, runner("b same name x-bench (sim path)"), seed=0)
    ok = True
    for part, st in stats.items():
        good = st["worst"] < 1e-12 and st["differs"] > 0
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {part:32s} calls {st['calls']:5d}  max|shipped-research| {st['worst']:.2e}  "
              f"calls where research != plain HierModel: {st['differs']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
