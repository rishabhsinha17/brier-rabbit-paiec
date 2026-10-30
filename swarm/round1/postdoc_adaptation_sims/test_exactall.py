"""Sanity checks for paiec/ext/pdadapt_exactall.py: alone pairs equal hrB ExactAloneHier; shared pairs vs the 2-D exact
two-pair computation (shared_laplace_excess.py, cur_final normal model, unknown subjects, distinct items)."""
import sys, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + ""))
from paiec.hier import HierModel
from paiec.ext.hrB_exact import ExactAloneHier
from paiec.ext.pdadapt_exactall import ExactAllHier

def subj(name):
    return {"normalized_name": name, "provider": "", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
def item(i, feats=""):
    return {"item_content": f"task {i}", "item_features": feats, "interactors": "", "benchmark_id": "benchmark_1"}
T, P = subj("unknown-target-x"), subj("unknown-partner-y")
def lab(s, k, n, off, feats=lambda i: ""):
    return [[[s, item(off + i, feats(i))], int(i < k)] for i in range(n)]
prm = {"tau_mu": 1.2, "lam": 1.0, "tau_delta": 1.0, "sigma_i": 0.6, "sigma_g": 1.0, "tau_phi": 0.5, "tau_theta": 0.5,
       "damped_newton": True, "m0": -2.0}
H, B, E = HierModel({}, prm), ExactAloneHier({}, prm), ExactAllHier({}, prm)
t0 = time.time()
print("alone pairs (HierModel / hrB exact / exact-all):")
for k, n in [(0, 1), (1, 1), (0, 7), (7, 7), (3, 7), (0, 31)]:
    L = lab(T, k, n, 0)
    print(f"  {k}/{n}: {H.predict([T, item(999)], L):.4f} {B.predict([T, item(999)], L):.4f} {E.predict([T, item(999)], L):.4f}")
grp = lambda i: f"grp=g{i % 3}"
L = lab(T, 2, 7, 0, grp)
print("  with groups 2/7, eval in g0:", round(H.predict([T, item(999, 'grp=g0')], L), 4), round(B.predict([T, item(999, 'grp=g0')], L), 4), round(E.predict([T, item(999, 'grp=g0')], L), 4))
print("shared pairs (HierModel Laplace / exact-all cavity / 2-D exact from shared_laplace_excess):")
ref = {(7, 0, 0): 0.055, (7, 7, 7): 0.855, (7, 0, 7): 0.134, (3, 0, 0): 0.089, (3, 3, 3): 0.731, (3, 0, 3): 0.186,
       (1, 0, 0): 0.140, (1, 1, 1): 0.505, (1, 0, 1): 0.221, (31, 0, 0): 0.020, (31, 31, 31): 0.956, (31, 0, 31): 0.055}
for (n, kT, kP), ex in ref.items():
    L = lab(T, kT, n, 0) + lab(P, kP, n, 1000)
    print(f"  n={n:2d} T {kT}/{n} P {kP}/{n}: {H.predict([T, item(999)], L):.3f}  {E.predict([T, item(999)], L):.3f}  {ex:.3f}")
# item pooling: partner labelled the eval item
L = lab(T, 2, 7, 0) + lab(P, 4, 7, 1000) + [[[P, item(999)], 1]]
L0 = lab(T, 2, 7, 0) + lab(P, 4, 7, 1000) + [[[P, item(999)], 0]]
print("partner labelled the eval item 1 vs 0:", round(H.predict([T, item(999)], L), 4), round(H.predict([T, item(999)], L0), 4), "| exact-all", round(E.predict([T, item(999)], L), 4), round(E.predict([T, item(999)], L0), 4))
print(f"{time.time() - t0:.1f}s")
