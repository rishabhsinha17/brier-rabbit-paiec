"""Full-grid expected Brier (every pool x every true d) for named candidates, cached, so that any posterior variant
(no B31 constraint, no competitor arms, alternative low modes...) is only a re-weighting.

E_full[name] : (nPool=140, nD=13, 6) bracket-mean expected Brier, cache/Efull_<name>.npz
"""
from __future__ import annotations

import json

import numpy as np

import hr_common as hc
import hr_fast as hf

SPECS = {
    "REF": {"m0": -2.0, "V": 2.69},
    "W20": {"m0": -2.0, "V": 5.25},
    "L15": {"m0": -1.5, "V": 2.69},
    "M150": {"m0": -1.5, "V": 5.25},
    "NLs": {"m0": -1.5, "V": 4.5},
    "NEs": {"m0": -1.25, "V": 2.69, "exact": True},
    "MXs": {"m0": -0.5, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55},
    "MXh": {"m0": -0.75, "V": 2.0, "mix_pi": 0.25, "mix_m": -2.9, "mix_s": 0.55},
    "MXd": {"m0": -1.0, "V": 1.5, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55},
    "MEs": {"m0": -0.5, "V": 1.0, "mix_pi": 0.3, "mix_m": -2.9, "mix_s": 0.55, "exact": True},
}


class FullEval(hf.CellEval):
    def __init__(self, th=None, tv=None):
        super().__init__(mass=1.0, th=th, tv=tv)
        nP, nD = len(hc.setup()["pools"]), len(hc.DS)
        self.cells = np.array([[i, j] for i in range(nP) for j in range(nD)])
        self.wc = np.full(len(self.cells), 1.0 / len(self.cells))
        self.ds = list(range(nD))


_FE = {}


def full(name, spec=None, tag=""):
    f = hc.CACHE / f"Efull_{name}{tag}.npz"
    spec = spec or SPECS[name]
    if f.exists():
        z = np.load(f)
        if json.loads(str(z["spec"])) == spec:
            return z["E"]
    key = tag
    if key not in _FE:
        if tag == "_S":
            th, tv = hc.thetas_S()
            _FE[key] = FullEval(th, tv)
        else:
            _FE[key] = FullEval()
    FE = _FE[key]
    E = FE.eval_spec(spec).reshape(len(hc.setup()["pools"]), len(hc.DS), 6)
    np.savez_compressed(f, E=E, spec=json.dumps(spec))
    return E


def full_tables(name, tabfun):
    """Full-grid E for a predictor given as tabfun(d, br) (e.g. the Bayes tables)."""
    f = hc.CACHE / f"Efull_{name}.npz"
    if f.exists():
        return np.load(f)["E"]
    if "" not in _FE:
        _FE[""] = FullEval()
    E = _FE[""].eval_tables(tabfun).reshape(len(hc.setup()["pools"]), len(hc.DS), 6)
    np.savez_compressed(f, E=E, spec=json.dumps({"tables": name}))
    return E


def oracle_full(mode="model"):
    """Oracle-cell Bayes: per pool table (d-free), E (nPool, nD, 6) (same for every d). NaN where no table."""
    f = hc.CACHE / f"Efull_Oracle_{mode}.npz"
    if f.exists():
        return np.load(f)["E"]
    s = hc.setup()
    z = np.load(hc.CACHE / f"bayes_{mode}.npz")
    QO = z["QO"]
    E = np.full((len(s["pools"]), len(hc.DS), 6), np.nan)
    const = (s["wts"][None, :] * s["R"]).sum(1)
    for i in range(len(QO)):
        if np.isnan(QO[i, 0, 0]):
            continue
        A1 = s["wts"][:, None] * s["PMF"][i]
        g = (QO[i] ** 2 * A1).sum(0) - 2.0 * (QO[i] * A1 * s["R"][i][:, None]).sum(0)
        for j, n in enumerate(hc.BUD):
            E[i, :, j] = g[hf.fp_idx(n)].sum() + const[i]
    np.savez_compressed(f, E=E, spec=json.dumps({"oracle": mode}))
    return E


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SPECS)
    for n in names:
        full(n)
        print(f"{n} {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > 90:
            print("time budget; rerun for the rest")
            break
    hf.save_disk()
