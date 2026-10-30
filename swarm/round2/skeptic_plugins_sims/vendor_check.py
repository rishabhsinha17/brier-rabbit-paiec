"""Vendoring hazard check. tools/make_submission.py ships ONE model.py = subjects.py + hier.py (+ footer instantiating
HierModel). Shipping a plug-in means appending its source with the `from paiec...` lines removed. Unguarded imports
fail loudly (ImportError at import time; the kit smoke test catches that), but a GUARDED fallback such as
pdadapt_simpool's
    try: from paiec.subjects import name_key
    except ImportError: name_key = None
silently rebinds the vendored name_key to None on the platform, while tools/verify_shipped.py (which puts the repo on
sys.path before importing the shipped file) still sees the real function and passes.
This script builds such a naive vendored file for each plug-in in a temp dir and imports it twice: in a clean
interpreter (python -I, cwd = temp dir: the platform) and with the repo on sys.path (verify_shipped.py).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "")
PY = (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../.venv/bin/python")


def _code(path):
    body = open(path).read().split('"""', 2)[2]
    return body.replace("from __future__ import annotations\n", "")


def strip_unguarded(src):
    """Remove top-level (unindented) `from paiec... import ...` statements, including parenthesised continuations."""
    return re.sub(r"^from paiec[\w.]* import [^\n(]*(\([^)]*\))?[^\n]*\n", "", src, flags=re.M)


PLUG = {"simpool": ("pdadapt_simpool.py", "SimilarityPoolModel", []),
        "subjcal": ("pd1_subjcal.py", "SubjCalHier", []),
        "prior": ("pd1_prior.py", "PriorCalHier", ["pd1_subjcal.py"])}

PROBE = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("model", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S1 = {"normalized_name": "Zeta Model 9", "access_date": "a"}; S2 = {"normalized_name": "Zeta Model 9", "access_date": "b"}
it = lambda i, b: {"item_content": f"t{b}{i}", "item_features": "", "interactors": "", "benchmark_id": b}
lab = [[[S1, it(i, "b1")], int(i < 3)] for i in range(7)] + [[[S2, it(i, "b2")], int(i < 6)] for i in range(7)]
print(json.dumps({"name_key": repr(m.name_key)[:40], "pred_same_name": m.predict([S1, it(99, "b1")], lab),
                  "B0": m.predict([S1, it(99, "b1")], [])}))
'''


def main():
    tmp = tempfile.mkdtemp(prefix="vendor_check_")
    params = json.load(open(os.path.join(ROOT, "submission_src/v3b_noM_n_m0n20/params.json")))
    open(os.path.join(tmp, "params.json"), "w").write(json.dumps(params))
    open(os.path.join(tmp, "subject_prior.json"), "w").write(open(os.path.join(ROOT, "submission_src/v3b_noM_n_m0n20/subject_prior.json")).read())
    open(os.path.join(tmp, "probe.py"), "w").write(PROBE)
    for key, (fname, cls, deps) in PLUG.items():
        code = ("from __future__ import annotations\nimport json\nfrom pathlib import Path\n"
                + _code(os.path.join(ROOT, "paiec/subjects.py")) + _code(os.path.join(ROOT, "paiec/hier.py")))
        for dep in deps + [fname]:
            code += strip_unguarded(_code(os.path.join(ROOT, "paiec/ext", dep)))
        code += (f"\n_HERE = Path(__file__).resolve().parent\n_MODEL = {cls}(json.loads((_HERE / 'subject_prior.json').read_text()),"
                 f" json.loads((_HERE / 'params.json').read_text()))\n\ndef predict(input, labeled=None):\n    return _MODEL.predict(input, labeled)\n")
        path = os.path.join(tmp, f"model_{key}.py")
        open(path, "w").write(code)
        clean = subprocess.run([PY, "-I", "probe.py", path], cwd=tmp, capture_output=True, text=True)
        repo = subprocess.run([PY, "probe.py", path], cwd=tmp, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=ROOT))
        print(f"{key:8s} platform-like import : {(clean.stdout or clean.stderr).strip()[-300:]}")
        print(f"{'':8s} verify_shipped-like : {(repo.stdout or repo.stderr).strip()[-300:]}")
    print(f"(files in {tmp})")


if __name__ == "__main__":
    main()
