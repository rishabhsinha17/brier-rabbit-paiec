"""Sanity table for paiec/ext/pdadapt_bcm.py: target predictions with a partner on the same benchmark."""
import sys, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.hier import HierModel
from paiec.ext.pdadapt_bcm import MixturePoolModel

def subj(name):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
def item(i, b="benchmark_1"):
    return {"item_content": f"task {i}", "item_features": "", "interactors": "", "benchmark_id": b}
T, P = subj("unknown-target-x"), subj("unknown-partner-y")
def lab(s, k, n, off):
    return [[[s, item(off + i)], int(i < k)] for i in range(n)]
base = {"tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5,
        "tau_theta": 0.5, "damped_newton": True, "m0": -1.0, "mix_pi": 0.35, "mix_m": -4.0, "mix_s": 0.7}
models = {"HierModel mix": HierModel({}, base), "BCM rho0": MixturePoolModel({}, base),
          "BCM rho.5": MixturePoolModel({}, dict(base, hard_rho=0.5)), "BCM rho.9": MixturePoolModel({}, dict(base, hard_rho=0.9))}
partners = [None, (0, 1), (1, 1), (0, 3), (2, 3), (0, 7), (4, 7), (0, 15), (0, 31), (25, 31)]
cases = [(1, 1), (0, 1), (2, 3), (0, 3), (5, 7), (0, 7), (12, 15), (20, 31)]
t0 = time.time()
for name, m in models.items():
    print(f"\n{name}: P(T solves new item)  [rows: T labels; cols: partner labels]")
    print("  T \\ P   " + " ".join(f"{('none' if q is None else f'{q[0]}/{q[1]}'):>7}" for q in partners))
    for kT, nT in cases:
        row = []
        for q in partners:
            L = lab(T, kT, nT, 0) + ([] if q is None else lab(P, q[0], q[1], 1000))
            row.append(m.predict([T, item(999999)], L))
        print(f"  {kT:>2}/{nT:<2}   " + " ".join(f"{v:7.3f}" for v in row))
# identity checks: alone pair and other-benchmark partner must equal HierModel
h, bc = models["HierModel mix"], models["BCM rho0"]
L = lab(T, 2, 7, 0) + [[[P, item(5000 + i, "benchmark_2")], i % 2] for i in range(7)]
print("\nidentity (partner on another benchmark):", h.predict([T, item(1)], L), bc.predict([T, item(1)], L))
print("B0:", h.predict([T, item(1)], []), bc.predict([T, item(1)], []))
print(f"elapsed {time.time() - t0:.1f}s")
