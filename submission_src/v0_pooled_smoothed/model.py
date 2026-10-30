"""Brier Rabbit v0: two-level shrinkage of revealed labels (benchmark rate, then pair rate).

Pure standard library. predict() is a deterministic function of (input, labeled) and never raises.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARAMS = json.loads(Path(__file__).with_name("params.json").read_text())
P0 = float(PARAMS["p0"])
KAPPA_BENCH = float(PARAMS["kappa_bench"])
KAPPA_PAIR = float(PARAMS["kappa_pair"])
CLIP = float(PARAMS.get("clip", 0.01))

_cache = {"key": None, "pair": {}, "bench": {}}


def _subject_key(subject) -> str:
    return json.dumps(subject, sort_keys=True, separators=(",", ":"), default=str)


def _counts(labeled):
    """Label sums by (subject, benchmark) and by benchmark, cached per evidence set."""
    h = hashlib.sha1()
    for entry in labeled or []:
        try:
            (subject, item), label = entry
            h.update(_subject_key(subject).encode())
            h.update(str(item.get("benchmark_id", "")).encode())
            h.update(b"1" if int(label) else b"0")
        except Exception:
            continue
    key = h.hexdigest()
    if key != _cache["key"]:
        pair, bench = {}, {}
        for entry in labeled or []:
            try:
                (subject, item), label = entry
                y = 1 if int(label) else 0
            except Exception:
                continue
            b = str(item.get("benchmark_id", ""))
            pk = (_subject_key(subject), b)
            s, n = pair.get(pk, (0, 0))
            pair[pk] = (s + y, n + 1)
            s, n = bench.get(b, (0, 0))
            bench[b] = (s + y, n + 1)
        _cache.update(key=key, pair=pair, bench=bench)
    return _cache["pair"], _cache["bench"]


def predict(input, labeled=None) -> float:
    try:
        subject, item = input
        b = str(item.get("benchmark_id", ""))
        pair, bench = _counts(labeled)
        s, n = pair.get((_subject_key(subject), b), (0, 0))
        sb, nb = bench.get(b, (0, 0))
        p_bench = (KAPPA_BENCH * P0 + (sb - s)) / (KAPPA_BENCH + (nb - n))
        p = (KAPPA_PAIR * p_bench + s) / (KAPPA_PAIR + n)
    except Exception:
        p = P0
    return float(min(1.0 - CLIP, max(CLIP, p)))
