"""Local replica of the platform's streaming_alc_v1 evaluation (starting kit README + tools/streaming_ingestion.py).

Mirrors: >=80-item eligibility, a persistent 50/50 split per (subject, benchmark) pair (odd count -> extra item to
acquisition), a fixed random candidate order, nested caps 0/1/3/7/15/31 where every pair finishes a checkpoint before
any pair gets more labels, the official default random acquisition rule, evidence shared across all sampled pairs,
deduplicated predictions per checkpoint, all recorded runs of an evaluation item scored, Brier ALC per pair with
weights 0.1/0.2/0.2/0.2/0.2/0.1, and an unweighted mean over pairs.
Not replicated (unknown): which recorded run a reveal returns (seeded random here), the exact pair processing order.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KIT_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(KIT_TOOLS) not in sys.path:
    sys.path.insert(0, str(KIT_TOOLS))
from prediction_workers import prediction_input_key  # noqa: E402  official dedupe key
from streaming_ingestion import AcquisitionHook  # noqa: E402  official hook call-shape logic

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
MAX_LABELS = 31
MIN_ITEMS = 80
DRAW_CAP = 1000


def unit(*parts) -> float:
    """Deterministic uniform in [0, 1) from any JSON-serialisable key (never Python hash())."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


@dataclass
class Pair:
    key: str
    bench: str
    benchmark_id: str
    subject_id: str
    subject: dict
    items: dict  # item_id -> {"item": dict, "labels": [0/1 per recorded run]}
    acq: list = field(default_factory=list)
    eval: list = field(default_factory=list)

    @property
    def n_items(self) -> int:
        return len(self.items)


def index_pairs(df, min_items: int = MIN_ITEMS) -> list[dict]:
    """Group response rows into eligible pairs once; seeds only reshuffle these structures."""
    from .data import SUBJECT_FIELDS

    item_cols = ["item_content", "item_features", "interactors", "benchmark_id"]
    raw = []
    for key, g in df.groupby("pair", sort=True):
        if g["item_id"].nunique() < min_items:
            continue
        first = g.iloc[0]
        labels = g.groupby("item_id", sort=False)["label"].agg(list)
        meta = g.drop_duplicates("item_id").set_index("item_id")[item_cols]
        items = {i: {"item": {c: meta.at[i, c] for c in item_cols}, "labels": [int(v) for v in labels[i]]}
                 for i in labels.index}
        raw.append({"key": key, "bench": first["bench"], "benchmark_id": first["benchmark_id"],
                    "subject": {f: first[f] for f in SUBJECT_FIELDS}, "items": items})
    return raw


SUBSAMPLE_MODES = ("pair", "bench")


def check_subsample(subsample: str) -> str:
    if subsample not in SUBSAMPLE_MODES:
        raise ValueError(f"subsample must be one of {SUBSAMPLE_MODES}, got {subsample!r}")
    return subsample


def materialize(raw_pairs: list[dict], size_range=None, seed: int = 0, min_items: int = MIN_ITEMS,
                subsample: str = "pair") -> list[Pair]:
    """Optionally subsample each pair to a platform-like size, then make the persistent 50/50 split.

    subsample="pair" (default, the original replica): the item ranking and the size target are keyed on the pair, so
    two pairs of one benchmark keep independent random subsets and share fewer items than complete platform groups.
    subsample="bench": both are keyed on the benchmark, so pairs of one benchmark keep the same items wherever their
    item sets overlap (their kept common items are nested prefixes of one benchmark-level ranking). The 50/50 split and
    the acquisition order stay keyed on the pair in both modes, as on the platform."""
    check_subsample(subsample)
    pairs = []
    for rp in raw_pairs:
        key, items = rp["key"], rp["items"]
        skey = key if subsample == "pair" else rp["bench"]
        ids = sorted(items, key=lambda i: unit(seed, "subsample", skey, i))
        if size_range is not None:
            lo, hi = size_range
            target = lo + int(unit(seed, "size", skey) * (hi - lo + 1))
            ids = ids[: max(min_items, min(target, len(ids)))]
        order = sorted(ids, key=lambda i: unit(seed, "split", key, i))
        n_acq = math.ceil(len(order) / 2)
        acq = sorted(order[:n_acq], key=lambda i: unit(seed, "order", key, i))
        pairs.append(Pair(
            key=key, bench=rp["bench"], benchmark_id=rp["benchmark_id"],
            subject_id=f"subject_{100000 + int(unit('sid', key) * 900000)}",
            subject=dict(rp["subject"]), items={i: items[i] for i in ids}, acq=acq, eval=order[n_acq:],
        ))
    return pairs


def build_pairs(df, min_items: int = MIN_ITEMS, size_range=None, seed: int = 0, subsample: str = "pair") -> list[Pair]:
    """Group response rows into eligible pairs, optionally subsample to platform-like sizes, then split."""
    return materialize(index_pairs(df, min_items), size_range=size_range, seed=seed, min_items=min_items,
                       subsample=subsample)


def ece(probs, ys, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(probs[mask].mean() - ys[mask].mean())
    return float(total / len(probs))


def _validate(value) -> float:
    if isinstance(value, bool):
        raise ValueError("predict() returned a bool")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"predict() returned {value!r}")
    return out


def run_protocol(pairs: list[Pair], predict, acquisition_function=None, seed: int = 0, strict_copies: bool = False,
                 record: list | None = None):
    """Run one evaluation over `pairs` (one formative draw). Returns per-pair, per-budget results."""
    hook = AcquisitionHook(acquisition_function)
    if hook.mode == "legacy_rank":
        raise NotImplementedError("legacy one-argument ranking hooks are not replicated")
    labeled: list = []
    acquired = {p.key: 0 for p in pairs}
    cursor = {p.key: 0 for p in pairs}
    results = {p.key: {} for p in pairs}
    clone = copy.deepcopy if strict_copies else (lambda x: x)

    def evaluate(budget: int) -> None:
        frozen = list(labeled)
        for p in pairs:
            cache, probs, ys = {}, [], []
            for item_id in p.eval:
                inp = [dict(p.subject), dict(p.items[item_id]["item"])]
                k = prediction_input_key(inp)
                if k not in cache:
                    cache[k] = _validate(predict(clone(inp), clone(frozen)))
                for y in p.items[item_id]["labels"]:
                    probs.append(cache[k])
                    ys.append(y)
                    if record is not None:
                        record.append((p.key, budget, cache[k], y))
            arr_p, arr_y = np.asarray(probs), np.asarray(ys, dtype=float)
            results[p.key][budget] = {
                "brier": float(np.mean((arr_p - arr_y) ** 2)), "ece": ece(arr_p, arr_y),
                "rows": len(ys), "unique": len(p.eval), "labels_used": acquired[p.key],
            }

    evaluate(0)
    for cap in BUDGETS[1:]:
        for p in pairs:
            while acquired[p.key] < cap and cursor[p.key] < len(p.acq):
                item_id = p.acq[cursor[p.key]]
                inp = [dict(p.subject), dict(p.items[item_id]["item"])]
                context = {
                    "subject_id": p.subject_id, "benchmark_id": p.benchmark_id,
                    "labels_remaining": MAX_LABELS - acquired[p.key], "labels_acquired": acquired[p.key],
                    "max_labels": MAX_LABELS, "items_remaining": len(p.acq) - cursor[p.key],
                }
                if hook.mode == "random":  # verbatim rule from streaming_ingestion.run_streaming
                    key = json.dumps([context, inp], sort_keys=True, separators=(",", ":"))
                    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
                    query = uniform < min(1, context["labels_remaining"] / context["items_remaining"])
                else:
                    prob = _validate(predict(clone(inp), clone(list(labeled)))) if hook.needs_prediction else None
                    query = hook(inp, prob, labeled, context)
                cursor[p.key] += 1
                if query:
                    runs = p.items[item_id]["labels"]
                    label = runs[int(unit(seed, "reveal", p.key, item_id) * len(runs))]
                    labeled.append([inp, int(label)])
                    acquired[p.key] += 1
        evaluate(cap)
    return results


def pair_alc(budget_results: dict) -> float:
    return sum(WEIGHTS[b] * budget_results[b]["brier"] for b in BUDGETS)


def summarize(results: dict) -> dict:
    """Platform aggregation: Brier ALC per pair, then the unweighted mean over pairs."""
    alcs = [pair_alc(r) for r in results.values()]
    return {
        "alc": float(np.mean(alcs)),
        "per_budget": {b: float(np.mean([r[b]["brier"] for r in results.values()])) for b in BUDGETS},
        "pairs": len(alcs),
    }


def platform_draw(pairs: list[Pair], seed: int, cap_items: int = DRAW_CAP) -> list[Pair]:
    """Greedy draw of whole pairs under the item cap, preferring benchmarks and subjects least represented so far."""
    remaining, chosen = cap_items, []
    bench_count: dict = {}
    subj_count: dict = {}
    pool = list(pairs)
    while True:
        feasible = [p for p in pool if p.n_items <= remaining]
        if not feasible:
            return chosen
        pick = min(feasible, key=lambda p: (bench_count.get(p.bench, 0),
                                            subj_count.get(p.subject["normalized_name"] or p.key, 0),
                                            unit(seed, "draw", p.key)))
        chosen.append(pick)
        pool.remove(pick)
        remaining -= pick.n_items
        bench_count[pick.bench] = bench_count.get(pick.bench, 0) + 1
        name = pick.subject["normalized_name"] or pick.key
        subj_count[name] = subj_count.get(name, 0) + 1
