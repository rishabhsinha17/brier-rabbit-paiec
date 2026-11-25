"""Targeted checks for paper 51. Run from the repository with ../.venv/bin/python."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import sys
import json
import time
import hashlib
import importlib.util
from pathlib import Path
from itertools import combinations
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "camera_ready"))
import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom
from build_corrected import build
from paiec.data import load_responses, BINARY_BENCHMARKS, DATA_ROOT
from paiec.hier import fit_subject_prior
from paiec.evaluate import FoldData
from paiec.protocol import BUDGETS, pair_alc, run_protocol

OUT = ROOT / "camera_ready/results"
OUT.mkdir(exist_ok=True)


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_checks(old, new):
    subject = {"normalized_name": "audit-model", "harness": "audit"}
    item = {"benchmark_id": "audit-benchmark", "item_content": "evidence", "item_features": "group=a"}
    target = dict(item, item_content="target")
    first = [([subject, item], 1)]
    second = [([subject, dict(item, item_features="group=b")], 1)]
    rows = {}
    for name, mod in (("original", old), ("corrected", new)):
        model = mod.ComboModel({}, mod._PARAMS)
        model.predict([subject, target], first)
        cached = model.predict([subject, target], second)
        fresh = mod.ComboModel({}, mod._PARAMS).predict([subject, target], second)
        rows[name] = {"cached": cached, "fresh": fresh, "absolute_error": abs(cached-fresh)}
    assert rows["original"]["absolute_error"] > 1e-5
    assert rows["corrected"]["absolute_error"] < 1e-12
    prefix, suffix = "a" * 2048, "z" * 2048
    a = dict(item, item_content=prefix + "middle A" + suffix)
    b = dict(item, item_content=prefix + "middle B" + suffix)
    assert old._item_key(a) == old._item_key(b)
    assert new._item_key(a) != new._item_key(b)
    evidence = [([subject, a], 1), ([dict(subject, normalized_name="partner"), b], 0)]
    model = new.ComboModel({}, new._PARAMS)
    # Reuse the same instance across list copies, permutations, replacements and in-place feature edits.
    streams = [[], evidence[:1], evidence, list(reversed(evidence)), json.loads(json.dumps(evidence)), second]
    errors = []
    for stream in streams:
        for query in (a, b, target):
            p = model.predict([subject, query], stream)
            q = new.ComboModel({}, new._PARAMS).predict([subject, query], stream)
            errors.append(abs(p-q))
    model.predict([subject, target], evidence)
    evidence[0][0][1]["item_features"] = "group=mutated"
    errors.append(abs(model.predict([subject, target], evidence) -
                      new.ComboModel({}, new._PARAMS).predict([subject, target], evidence)))
    assert max(errors) < 1e-12
    rows["corrected_fresh_instance_checks"] = len(errors)
    rows["corrected_max_error"] = max(errors)
    return rows


def collisions(old):
    rows = []
    for bench in BINARY_BENCHMARKS:
        items = pd.read_parquet(DATA_ROOT / bench / "items.parquet", columns=["item_id", "content"])
        groups = defaultdict(set)
        for text in items.content.fillna("").map(str):
            groups[old._item_key({"item_content": text})].add(hashlib.sha256(text.encode()).hexdigest())
        bad = {key: vals for key, vals in groups.items() if len(vals) > 1}
        rows.append({"bench": bench, "item_rows": len(items), "distinct_full_texts": len(set().union(*groups.values())),
                     "collision_groups": len(bad), "distinct_texts_in_collision_groups": sum(map(len, bad.values())),
                     "excess_texts_merged": sum(len(s)-1 for s in bad.values())})
    pd.DataFrame(rows).to_csv(OUT / "collisions.csv", index=False)
    return rows


def exact_checks(old):
    # Independent enumeration: average squared error over acquisition subsets and all remaining targets.
    population = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    errors = []
    for n in range(5):
        values = []
        for indices in combinations(range(8), n):
            k = int(population[list(indices)].sum())
            q = (k + 0.8) / (n + 2)
            remaining = np.delete(population, indices)
            values.append(np.mean((q-remaining)**2))
        ks = np.arange(n+1)
        q = (ks + .8) / (n+2)
        rest = (3-ks) / (8-n)
        risk = np.sum(hypergeom.pmf(ks, 8, 3, n) * ((q-rest)**2+rest*(1-rest)))
        errors.append(abs(float(risk)-np.mean(values)))
    assert max(errors) < 1e-14
    subject = {"normalized_name": ""}
    qtables = {}
    for name, tau in (("REF", 1.2), ("W20", 2.0)):
        model = old.ComboModel({}, dict(old._PARAMS, tau_mu=tau))
        table = {}
        for n in BUDGETS:
            vals = []
            for k in range(n+1):
                evidence = [([subject, {"benchmark_id": "finite", "item_content": f"item {i}"}], int(i < k)) for i in range(n)]
                vals.append(model.predict([subject, {"benchmark_id": "finite", "item_content": "target"}], evidence))
            table[n] = np.array(vals)
        qtables[name] = table
    records = []
    for N in (80, 88, 160, 226):
        for S in range(N+1):
            for n in BUDGETS:
                ks = np.arange(n+1)
                r = S/N
                rest = (S-ks)/(N-n)
                record = {"N": N, "S": S, "n": n}
                for name in ("REF", "W20"):
                    q = qtables[name][n]
                    record[name+"_iid"] = float(np.sum(binom.pmf(ks, n, r)*((q-r)**2+r*(1-r))))
                    record[name+"_finite"] = float(np.sum(hypergeom.pmf(ks, N, S, n)*((q-rest)**2+rest*(1-rest))))
                records.append(record)
    df = pd.DataFrame(records)
    df["iid_contrast"] = df.W20_iid-df.REF_iid
    df["finite_contrast"] = df.W20_finite-df.REF_finite
    df.to_csv(OUT / "finite_population_grid.csv", index=False)
    delta = (df.finite_contrast-df.iid_contrast).abs()
    different = (df.iid_contrast*df.finite_contrast < 0) & (df.iid_contrast.abs()>1e-12) & (df.finite_contrast.abs()>1e-12)
    result = {"enumerated_populations":5, "enumeration_max_error":max(errors),
              "grid_cells":len(df), "max_arm_risk_difference":float(max((df.REF_iid-df.REF_finite).abs().max(),(df.W20_iid-df.W20_finite).abs().max())),
              "max_contrast_difference":float(delta.max()), "max_contrast_case":df.loc[delta.idxmax()].to_dict(),
              "contrast_sign_changes":int(different.sum())}
    return result


def replays(old, new):
    df = load_responses()
    benches = BINARY_BENCHMARKS[:4]
    folds = FoldData(df, benches, subsample="bench")
    rows, traces = [], []
    start = time.monotonic()
    for bench in benches:
        prior = fit_subject_prior(df[df.bench != bench])
        for k in (1, 2):
            for run, pairs in folds.draws(bench, k, 5, 0):
                results, predictions = {}, {}
                for name, mod in (("original",old),("corrected",new)):
                    model = mod.ComboModel(prior, mod._PARAMS)
                    preds = []
                    def predict(inp, labeled=None):
                        p = model.predict(inp,labeled)
                        preds.append(p)
                        return p
                    results[name] = run_protocol(pairs, predict, seed=run)
                    predictions[name] = np.array(preds)
                differences = np.abs(predictions["original"] - predictions["corrected"])
                traces.append({"bench":bench,"k":k,"run":run,"predictions":len(differences),
                               "changed_predictions":int((differences>1e-12).sum()),"max_absolute_change":float(differences.max())})
                for key in results["original"]:
                    a,b = results["original"][key],results["corrected"][key]
                    rows.append({"bench":bench,"k":k,"run":run,"pair":key,"original_alc":pair_alc(a),
                                 "corrected_alc":pair_alc(b),"delta_alc":pair_alc(b)-pair_alc(a)})
            pd.DataFrame(rows).to_csv(OUT / "replay_pairs.csv",index=False)
            pd.DataFrame(traces).to_csv(OUT / "replay_predictions.csv",index=False)
            print(f"completed {bench} k={k}; elapsed {time.monotonic()-start:.1f}s",flush=True)
    return {"draws":len(traces),"pair_scores":len(rows),"wall_seconds":time.monotonic()-start}


if __name__ == "__main__":
    hashes=build()
    old=load("original_w20",ROOT/"submission_src/v6_combo_w20_c1")
    new=load("corrected_w20",ROOT/"camera_ready/corrected_w20")
    summary={"hashes":hashes,"cache":cache_checks(old,new),"collisions":collisions(old),"finite_population":exact_checks(old)}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    summary["replays"]=replays(old,new)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
