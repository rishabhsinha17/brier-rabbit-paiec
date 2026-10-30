"""Reference predictors with the platform signature predict(input, labeled) -> float."""
from __future__ import annotations

import hashlib
import json


def subject_key(subject: dict) -> str:
    return json.dumps(subject, sort_keys=True, separators=(",", ":"))


def evidence_key(labeled) -> str:
    h = hashlib.sha1()
    for (subject, item), label in labeled or []:
        h.update(subject_key(subject).encode())
        h.update(str(item.get("benchmark_id", "")).encode())
        h.update(b"1" if label else b"0")
    return h.hexdigest()


class _EvidenceCounts:
    """Per-call cache of label counts by (subject, benchmark) and by benchmark."""

    def __init__(self):
        self._key = None
        self.pair = {}
        self.bench = {}

    def update(self, labeled):
        key = evidence_key(labeled)
        if key == self._key:
            return
        self._key, self.pair, self.bench = key, {}, {}
        for (subject, item), label in labeled or []:
            b = item.get("benchmark_id", "")
            pk = (subject_key(subject), b)
            s, n = self.pair.get(pk, (0, 0))
            self.pair[pk] = (s + int(label), n + 1)
            s, n = self.bench.get(b, (0, 0))
            self.bench[b] = (s + int(label), n + 1)


class Constant:
    def __init__(self, p: float = 0.5):
        self.p = p

    def predict(self, input, labeled=None):
        return self.p


class EmpiricalMean:
    """Organizer baseline: mean revealed label for the target pair, 0.5 without matching labels."""

    def __init__(self):
        self.ev = _EvidenceCounts()

    def predict(self, input, labeled=None):
        self.ev.update(labeled)
        subject, item = input
        s, n = self.ev.pair.get((subject_key(subject), item.get("benchmark_id", "")), (0, 0))
        return s / n if n else 0.5


class SmoothedMean:
    """Beta(kappa*p0, kappa*(1-p0)) prior on the pair rate, updated with the pair's own labels."""

    def __init__(self, p0: float, kappa: float):
        self.p0, self.kappa = p0, kappa
        self.ev = _EvidenceCounts()

    def predict(self, input, labeled=None):
        self.ev.update(labeled)
        subject, item = input
        s, n = self.ev.pair.get((subject_key(subject), item.get("benchmark_id", "")), (0, 0))
        return (self.kappa * self.p0 + s) / (self.kappa + n)


class PooledSmoothed:
    """Two-level shrinkage: benchmark rate from every subject's labels, then the pair rate around it."""

    def __init__(self, p0: float, kappa_bench: float, kappa_pair: float):
        self.p0, self.kb, self.kp = p0, kappa_bench, kappa_pair
        self.ev = _EvidenceCounts()

    def predict(self, input, labeled=None):
        self.ev.update(labeled)
        subject, item = input
        b = item.get("benchmark_id", "")
        sb, nb = self.ev.bench.get(b, (0, 0))
        s, n = self.ev.pair.get((subject_key(subject), b), (0, 0))
        # Benchmark level uses other subjects' labels only, so the pair's own labels are not counted twice.
        p_bench = (self.kb * self.p0 + (sb - s)) / (self.kb + (nb - n))
        return (self.kp * p_bench + s) / (self.kp + n)
