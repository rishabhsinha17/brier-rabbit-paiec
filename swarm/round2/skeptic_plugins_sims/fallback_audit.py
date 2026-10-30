"""Silent-fallback audit: every plug-in wraps its body in try/except and falls back to HierModel or to sigmoid(m0).
A fuzz cannot see a fallback on WELL-FORMED inputs, so this traces exceptions raised (even if caught) inside
paiec/ext/*.py and paiec/hier.py while the recommended configurations predict on real public replay calls.

usage: PAIEC_WORKERS=1 nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round2/skeptic_plugins_sims/fallback_audit.py
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PLUGINS, load_cache, make  # noqa: E402
from replay_checks import RECOMMENDED, all_records, calls  # noqa: E402

EVENTS = collections.Counter()


def _local(frame, event, arg):
    if event == "exception":
        exc_type = arg[0].__name__
        EVENTS[(os.path.basename(frame.f_code.co_filename), frame.f_lineno, exc_type)] += 1
    return _local


def _global(frame, event, arg):
    fn = frame.f_code.co_filename
    if "paiec/ext/" in fn or fn.endswith("paiec/hier.py"):
        return _local
    return None


def main():
    _, priors = load_cache()
    recs = all_records()
    for name, (key, params) in RECOMMENDED.items():
        EVENTS.clear()
        n = 0
        for bench, recs_b in recs.items():
            cl = calls(recs_b)
            cl = [c for j, c in enumerate(cl) if j % 4 == 0]  # every 4th call, all checkpoints
            m = make(PLUGINS[key], priors[bench], params)
            sys.settrace(_global)
            try:
                for _, _, _, lab, inp in cl:
                    m.predict(inp, lab)
                    n += 1
            finally:
                sys.settrace(None)
        # an exception propagating through k frames is reported once per frame: keep the innermost (first) line
        print(f"{name:22s} calls {n:5d} | exceptions raised inside plug-in/hier code: "
              + (", ".join(f"{f}:{ln} {t} x{c}" for (f, ln, t), c in EVENTS.most_common(6)) or "none"), flush=True)


if __name__ == "__main__":
    main()
