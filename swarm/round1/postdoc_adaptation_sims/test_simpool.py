"""Sanity checks for paiec/ext/pdadapt_simpool.py: identity without coincidences, pooling with them."""
import sys, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.hier import HierModel
from paiec.ext.pdadapt_simpool import SimilarityPoolModel

def subj(name, harness="", acc=""):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": acc, "harness": harness,
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
def item(i, b):
    return {"item_content": f"task {b} {i}", "item_features": "", "interactors": "", "benchmark_id": b}
def lab(s, k, n, b, off=0):
    return [[[s, item(off + i, b)], int(i < k)] for i in range(n)]
prm = {"tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5, "tau_theta": 0.5,
       "damped_newton": True, "m0": -2.0}
H, S = HierModel({}, prm), SimilarityPoolModel({}, prm)
A1 = subj("Zeta Model 9", acc="2026-01-01"); A2 = subj("Zeta Model 9", acc="2026-02-02")   # same name, different dicts
B1 = subj("Other Model 3", acc="2026-01-01")
cases = {
    "alone 3/7": lab(A1, 3, 7, "bench_1"),
    "two benchmarks, different names": lab(A1, 3, 7, "bench_1") + lab(B1, 6, 7, "bench_2"),
    "same benchmark, different names": lab(A1, 3, 7, "bench_1") + lab(B1, 6, 7, "bench_1", 100),
    "two benchmarks, SAME name (partner 6/7)": lab(A1, 3, 7, "bench_1") + lab(A2, 6, 7, "bench_2"),
    "two benchmarks, SAME name (partner 0/7)": lab(A1, 3, 7, "bench_1") + lab(A2, 0, 7, "bench_2"),
    "two benchmarks, SAME name, target 0/1, partner 1/1": lab(A1, 0, 1, "bench_1") + lab(A2, 1, 1, "bench_2"),
    "two benchmarks, SAME name, target 1/3, partner 3/3": lab(A1, 1, 3, "bench_1") + lab(A2, 3, 3, "bench_2"),
}
t0 = time.time()
for name, L in cases.items():
    print(f"{name:55s} HierModel {H.predict([A1, item(999, 'bench_1')], L):.4f}  SimPool {S.predict([A1, item(999, 'bench_1')], L):.4f}")
# harness within benchmark
X1 = subj("Model A", harness="AgentX"); X2 = subj("Model B", harness="AgentX"); Y2 = subj("Model B", harness="AgentY")
for name, L in {"same bench, same harness (partner 6/7)": lab(X1, 1, 7, "b") + lab(X2, 6, 7, "b", 100),
                "same bench, diff harness (partner 6/7)": lab(X1, 1, 7, "b") + lab(Y2, 6, 7, "b", 100),
                "alone with harness 1/7": lab(X1, 1, 7, "b")}.items():
    print(f"{name:55s} HierModel {H.predict([X1, item(999, 'b')], L):.4f}  SimPool {S.predict([X1, item(999, 'b')], L):.4f}")
print("B0:", H.predict([A1, item(1, 'bench_1')], []), S.predict([A1, item(1, 'bench_1')], []), f"{time.time() - t0:.1f}s")
