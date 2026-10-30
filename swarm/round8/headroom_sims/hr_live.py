"""Refresh the round-4 design's live evidence into a PRIVATE npz (round 8, headroom designer).

Same code path as `tools/swarm_live_design.py --refresh-only` (ld_data.build on the selection set; the QUEUE.md hash
holdout is never read; competitor logs are connectors only), but written to swarm/round8/headroom_sims/cache/ so the
PI's daily cache in swarm/round4/design_sims/cache/ is untouched.

usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round8/headroom_sims/hr_live.py [reps]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "swarm" / "round4" / "design_sims"))
import ld_calc as lc  # noqa: E402
import ld_data as ldt  # noqa: E402
import swarm_live_design as sld  # noqa: E402  (library use only)

CACHE = HERE / "cache"
OUT = CACHE / "live_hr.npz"


def register_family():
    """Ledger family arms -> lc.LIVE_FAMILY (exactly as swarm_live_design.main does); returns (arms, our_arms)."""
    arms, have = sld.ledger_state()
    fam_live = {n: d for n, d in arms.items() if d["spec"] is not None}
    for n, d in fam_live.items():
        s = d["spec"]
        lc.LIVE_FAMILY[n] = (s["m0"], s["tau_mu"], s["tau_delta"], s["tau_phi"])
    our_arms = sorted(n for n in fam_live if n != lc.REF and fam_live[n]["done"] > 0) + list(lc.OTHER_OURS)
    return arms, our_arms, have


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    arms, our_arms, have = register_family()
    res, kt, runs, stamp = ldt.build(reps, OUT, our_arms=our_arms)
    (CACHE / "live_hr.key.json").write_text(json.dumps({"feedback": sorted(have), "our_arms": our_arms,
                                                        "family": {k: list(v) for k, v in lc.LIVE_FAMILY.items()}}))
    print(stamp)
    print("our arms:", our_arms)
    print(f"done in {time.time() - t0:.1f}s -> {OUT}")
