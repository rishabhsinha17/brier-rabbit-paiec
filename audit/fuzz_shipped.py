"""Malformed-input fuzz and interpreter/numpy compatibility of the shipped composite zips. Standard library + numpy
(through the shipped model only), so it also runs under the system python3.11 with numpy 1.26.

usage (from brier_rabbit/):
  nice -n 19 taskpolicy -b ../.venv/bin/python swarm/round5/release/audit/fuzz_shipped.py [--json OUT]
  nice -n 19 taskpolicy -b python3.11 swarm/round5/release/audit/fuzz_shipped.py --json OUT2   # numpy 1.26
  python3 swarm/round5/release/audit/fuzz_shipped.py --compare OUT OUT2

For every case: predict() must return a Python float, finite, in [0, 1], without raising. "fallback" marks a result
equal to the constant last-resort value clip(sigmoid(m0)), i.e. the model gave up on the evidence or the input.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

# A platform worker imports model.py before numpy, so the model's own single-thread BLAS pinning takes effect. Keep
# that order here: never import numpy before the shipped module (OpenBLAS builds slow down badly otherwise).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAMES = ("v6_combo_ref", "v6_combo_w20", "v6_combo_l15")


def cases(names_known):
    labeled, inputs = common.draw(names_known, budget=7, seed=3)
    s, it = copy.deepcopy(inputs[0][0]), copy.deepcopy(inputs[0][1])
    valid = [s, it]
    big = dict(it, item_content="y" * 1_000_000)
    sur = dict(it, item_content="abc\ud800def")
    odd = dict(it, item_content="nul\x00 emoji \U0001F600 rtl ‮")
    yield "valid input", valid, labeled
    # ---- malformed input, valid evidence
    for name, inp in [
        ("input None", None), ("input []", []), ("input [None, None]", [None, None]), ("input [{}, {}]", [{}, {}]),
        ("input str", "subject|item"), ("input int", 42), ("input 3 elements", [s, it, {}]), ("input tuple", (s, it)),
        ("item without benchmark_id/features", [s, {"item_content": "x"}]),
        ("benchmark_id ''", [s, dict(it, benchmark_id="")]), ("benchmark_id None", [s, dict(it, benchmark_id=None)]),
        ("subject all None", [{k: None for k in s}, it]),
        ("subject ints", [dict(s, normalized_name=123, release_date=20260101), it]),
        ("release_date 2026-13-45", [dict(s, normalized_name="Brand New 9", release_date="2026-13-45"), it]),
        ("release_date garbage", [dict(s, normalized_name="Brand New 9", release_date="abcd-ef"), it]),
        ("features_extra garbage", [dict(s, subject_features_extra="a=b=c;;;=;source_model_name=;model_config=/"), it]),
        ("reasoning_effort unknown", [dict(s, reasoning_effort="ultra-max"), it]),
        ("item_content 1 MB", [s, big]), ("item_content lone surrogate", [s, sur]), ("item_content NUL/emoji/RTL", [s, odd]),
        ("item_features weird", [s, dict(it, item_features=";;;==;lang")]),
        ("unseen benchmark", [s, dict(it, benchmark_id="benchmark_999999")]),
        ("subject extra keys", [dict(s, extra_key="z", another=[1, 2]), it]),
        ("item_content is a dict", [s, dict(it, item_content={"a": 1})]),
        ("subject is a list", [[1, 2], it]),
    ]:
        yield name, inp, labeled
    # ---- valid input, malformed evidence
    e0 = labeled[0]
    for name, lab in [
        ("labeled None", None), ("labeled []", []), ("labeled str", "not a list"), ("labeled [None]", [None]),
        ("entry [None, 1]", labeled + [[None, 1]]), ("entry without label", labeled + [[e0[0]]]),
        ("label '1'", labeled + [[e0[0], "1"]]), ("label 2", labeled + [[e0[0], 2]]), ("label 1.5", labeled + [[e0[0], 1.5]]),
        ("label NaN", labeled + [[e0[0], float("nan")]]), ("label inf", labeled + [[e0[0], float("inf")]]),
        ("label None", labeled + [[e0[0], None]]), ("label True", labeled + [[e0[0], True]]),
        ("entry item None", labeled + [[[e0[0][0], None], 1]]),
        ("evidence item lone surrogate", labeled + [[[e0[0][0], sur], 1]]),
        ("labeled tuple", tuple(labeled)),
        ("2000 duplicate entries", labeled + [e0] * 2000),
        ("31 ones on the target pair", labeled + [[[s, dict(it, item_content=f"q{j}")], 1] for j in range(31)]),
        ("31 zeros on the target pair", labeled + [[[s, dict(it, item_content=f"q{j}")], 0] for j in range(31)]),
        ("contradictory labels same item", labeled + [[[s, it], 1], [[s, it], 0]] * 5),
        ("labels on 60 benchmarks", labeled + [[[s, dict(it, benchmark_id=f"benchmark_{100000 + 13 * j}")], j % 2]
                                               for j in range(60)]),
    ]:
        yield name, valid, lab


def run(name, records, mod):
    p = mod._PARAMS
    fallback = float(min(1 - p.get("clip", 0.005), max(p.get("clip", 0.005), 1.0 / (1.0 + math.exp(-p["m0"])))))
    names_known = common.prior_names(common.extract(name))
    for case, inp, lab in cases(names_known):
        t0 = time.perf_counter()
        err = None
        try:
            out = mod.predict(copy.deepcopy(inp) if not isinstance(inp, str) else inp, lab)
        except BaseException as exc:  # noqa: BLE001  the point is to catch anything
            out, err = None, f"{type(exc).__name__}: {exc}"
        ms = 1e3 * (time.perf_counter() - t0)
        ok = err is None and type(out) is float and math.isfinite(out) and 0.0 <= out <= 1.0
        records.append({"zip": name, "case": case, "ok": ok, "value": out, "fallback": ok and out == fallback,
                        "ms": round(ms, 2), "error": err})


def compat_report():
    src = (common.extract("v6_combo_w20") / "model.py").read_text()
    lines = []
    for v in ((3, 7), (3, 8), (3, 9), (3, 10)):
        try:
            ast.parse(src, feature_version=v)
            lines.append(f"syntax ok for Python {v[0]}.{v[1]} grammar")
        except SyntaxError as exc:
            lines.append(f"SYNTAX FAILS for Python {v[0]}.{v[1]}: line {exc.lineno}: {exc.msg}")
    tree = ast.parse(src)
    mods = sorted({a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
                  | {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level})
    lines.append(f"absolute imports: {', '.join(m for m in mods if m)}")
    lines.append("relative import inside try/except ImportError: " +
                 str(any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))))
    newer = [w for w in ("removeprefix", "removesuffix", "math.lcm", "strict=", "functools.cache", "match ",
                         ".bit_count(", "zoneinfo", "tomllib", "typing.Self", "ExceptionGroup", "np.trapezoid",
                         "np.bool8", "np.float_", "np.int_(", "np.product", "np.cumproduct", "np.in1d", "row_stack")
             if w in src]
    lines.append(f"post-3.8 stdlib / numpy-version-sensitive names used: {newer or 'none'}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.loads(Path(f).read_text()) for f in args.compare)
        va = {(r["zip"], r["case"]): r["value"] for r in a["records"]}
        vb = {(r["zip"], r["case"]): r["value"] for r in b["records"]}
        diffs = [abs(va[k] - vb[k]) for k in va if k in vb and va[k] is not None and vb[k] is not None]
        print(f"{a['env']}  vs  {b['env']}: {len(diffs)} cases, max |diff| = {max(diffs):.2e}")
        return
    records, mods = [], {}
    for name in NAMES:
        mods[name], _ = common.shipped(name, module_name=f"fz_{name}")
    env = f"python {sys.version.split()[0]} numpy {sys.modules['numpy'].__version__}"
    print(env, flush=True)
    for line in compat_report():
        print("  " + line)
    for name in NAMES:
        run(name, records, mods[name])
    bad = [r for r in records if not r["ok"]]
    fb = sorted({r["case"] for r in records if r["fallback"]})
    slow = sorted({(r["case"], r["ms"]) for r in records if r["ms"] > 500}, key=lambda x: -x[1])
    print(f"{len(records)} calls over {len(NAMES)} zips: {len(records) - len(bad)} ok, {len(bad)} raised or invalid")
    for r in bad:
        print(f"  BAD {r['zip']} {r['case']}: {r['error'] or r['value']}")
    print(f"constant-fallback cases ({len(fb)}): {', '.join(fb)}")
    print(f"slowest cases (> 500 ms): {slow or 'none'}")
    w20 = {r["case"]: r for r in records if r["zip"] == "v6_combo_w20"}
    print("v6_combo_w20 values: " + "; ".join(f"{c} {r['value']:.4f}{' (fallback)' if r['fallback'] else ''}"
                                              for c, r in w20.items() if r["value"] is not None))
    if args.json:
        Path(args.json).write_text(json.dumps({"env": env, "records": records}, indent=1, default=str))


if __name__ == "__main__":
    main()
