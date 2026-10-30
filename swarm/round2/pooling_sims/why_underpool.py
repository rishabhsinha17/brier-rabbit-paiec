"""Why HierModel under-pools: one sigma_i serves own-label absorption, unseen-item moderation and partner pooling.

(a) Own-level learning (no partner): A has k of n labels; prediction for an unlabelled item, HierModel at sigma_i
    0.6 / 1.5 / 2.0 vs CavityPoolModel (identical to base here) and the empirical rate.
(b) Pooling strength: partner B (n labels, 30%) labelled the target item; logit shift of A's prediction on that item
    relative to A's unlabelled-item prediction, HierModel sigma_i 0.6 / 1.5 / 2.0 vs CavityPoolModel sigma_x 2.0/2.5.
Live hyperparameters otherwise (m0 -2, tau_mu 1.2, tau_delta 1.0, tau_phi 0.5, damped Newton), no subject prior."""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from unit_checks import LIVE, evidence, item, subj  # noqa: E402
from paiec.ext.pool_cavity import CavityPoolModel  # noqa: E402
from paiec.hier import HierModel  # noqa: E402


def lg(p):
    return math.log(p / (1 - p))


print("(a) A alone with k/n own labels -> P(unlabelled item):")
print("   k/n   | si0.6  si1.0  si1.5  si2.0 | empirical")
for n, k in ((3, 2), (7, 5), (15, 12), (31, 25), (31, 9)):
    ev = evidence([("A", 1000 + i, int(i < k)) for i in range(n)])
    vals = [HierModel({}, dict(LIVE, sigma_i=s)).predict([subj("A"), item(99)], ev) for s in (0.6, 1.0, 1.5, 2.0)]
    print(f"  {k:2d}/{n:2d}  | " + "  ".join(f"{v:.3f}" for v in vals) + f" | {k / n:.3f}")
print("(b) partner B labelled the target item; logit shift of A's prediction vs A's unlabelled item (A, B 30%, n each):")
print("   n  y | si0.6   si1.0   si1.5   si2.0  | cav_x2.0 cav_x2.5")
for n in (3, 7, 15, 31):
    for y in (1, 0):
        k = int(round(0.3 * (n - 1)))
        ka = int(round(0.3 * n))
        ev = evidence([("A", 1000 + i, int(i < ka)) for i in range(n)] + [("B", 2000 + i, int(i < k)) for i in range(n - 1)]
                      + [("B", 99, y)])
        out = []
        for s in (0.6, 1.0, 1.5, 2.0):
            m = HierModel({}, dict(LIVE, sigma_i=s))
            out.append(lg(m.predict([subj("A"), item(99)], ev)) - lg(m.predict([subj("A"), item(98)], ev)))
        for sx in (2.0, 2.5):
            m = CavityPoolModel({}, dict(LIVE, sigma_x=sx))
            out.append(lg(m.predict([subj("A"), item(99)], ev)) - lg(m.predict([subj("A"), item(98)], ev)))
        print(f"  {n:2d}  {y} | " + "  ".join(f"{v:+.3f}" for v in out))
