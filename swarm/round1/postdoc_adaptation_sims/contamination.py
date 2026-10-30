"""How much does a partner pair on the same benchmark move the target's prediction in HierModel?
Synthetic: unknown subjects (theta=0), distinct items, no item_features. Target T has kT/nT; partner P has kP/nP."""
import sys
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.hier import HierModel

def subj(name):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
def item(i):
    return {"item_content": f"task {i}", "item_features": "", "interactors": "", "benchmark_id": "benchmark_1"}
T, P = subj("unknown-target-x"), subj("unknown-partner-y")
def lab(s, k, n, off):
    return [[[s, item(off + i)], int(i < k)] for i in range(n)]

base = {"tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5,
        "tau_theta": 0.5, "damped_newton": True}
arms = {"cur m0=-2": dict(base, m0=-2.0), "mix.35 m0=-1": dict(base, m0=-1.0, mix_pi=0.35, mix_m=-4.0, mix_s=0.7)}
for name, prm in arms.items():
    m = HierModel({}, prm)
    print(f"\n{name}: P(T solves new item)  [rows: target labels; cols: partner labels]")
    partners = [None, (0, 1), (0, 3), (0, 7), (0, 15), (0, 31), (3, 7), (15, 31), (25, 31)]
    print("  T \\ P   " + " ".join(f"{('none' if p is None else f'{p[0]}/{p[1]}'):>7}" for p in partners))
    for kT, nT in [(0, 0), (1, 1), (0, 1), (1, 3), (2, 3), (3, 7), (5, 7), (8, 15), (12, 15), (20, 31)]:
        row = []
        for pp in partners:
            L = lab(T, kT, nT, 0) + ([] if pp is None else lab(P, pp[0], pp[1], 1000))
            row.append(m.predict([T, item(999999)], L))
        print(f"  {kT:>2}/{nT:<2}   " + " ".join(f"{v:7.3f}" for v in row))
