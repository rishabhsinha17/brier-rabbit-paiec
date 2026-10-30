"""tools/make_submission.py: the default build is unchanged; --plugin vendors a paiec.ext class into one model.py."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paiec.data import SUBJECT_FIELDS  # noqa: E402

# sha256 of HEADER + THREADS + FOOTER in the tool before --plugin existed (2026-10-06): the default template is pinned.
DEFAULT_TEMPLATE_SHA = "a5c23177af27c559106a1f8ac9154b9ed1ce0e5b6d8d35a89ed91e7ee51d6f1f"
STUB_PRIOR = {"attr": None, "names": {}}


def load_tool():
    spec = importlib.util.spec_from_file_location("make_submission_under_test", REPO / "tools" / "make_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(tmp_path, monkeypatch, name, params, *extra):
    """Run main() into tmp_path with a stub subject prior (the real fit needs the public data and is slow)."""
    ms = load_tool()
    monkeypatch.setattr(ms, "load_responses", lambda *a, **k: None)
    monkeypatch.setattr(ms, "fit_subject_prior", lambda df: STUB_PRIOR)
    monkeypatch.setattr(sys, "argv", ["make_submission.py", name, "--params", json.dumps(params),
                                      "--outdir", str(tmp_path), *extra])
    ms.main()
    return ms


def import_shipped(path: Path):
    spec = importlib.util.spec_from_file_location(f"shipped_{path.name}", path / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample():
    subject = {f: "" for f in SUBJECT_FIELDS}
    subject.update(normalized_name="unknown model", release_date="2026-09-01")
    item = lambda i: {"item_content": f"question {i}", "item_features": "", "interactors": "",  # noqa: E731
                      "benchmark_id": "benchmark_1"}
    labeled = [[[dict(subject), item(1)], 1], [[dict(subject), item(2)], 0], [[dict(subject), item(3)], 0]]
    return [subject, item(9)], labeled


def test_default_build_is_unchanged(tmp_path, monkeypatch):
    params = {"m0": -2.0, "tau_mu": 1.2, "damped_newton": True}
    ms = build(tmp_path, monkeypatch, "v_test", params)
    assert hashlib.sha256((ms.HEADER + ms.THREADS + ms.FOOTER).encode()).hexdigest() == DEFAULT_TEMPLATE_SHA
    expected = ("from __future__ import annotations\n" + ms.HEADER.format(name="v_test") + ms.THREADS
                + "\nimport json\nfrom pathlib import Path\n" + ms._code(REPO / "paiec" / "subjects.py")
                + ms._code(REPO / "paiec" / "hier.py") + ms.FOOTER)
    assert (tmp_path / "model.py").read_text() == expected
    assert (tmp_path / "params.json").read_text() == json.dumps(params, indent=1)
    assert json.loads((tmp_path / "subject_prior.json").read_text()) == STUB_PRIOR


def test_plugin_build_ships_the_class_and_ignores_the_class_key(tmp_path, monkeypatch):
    params = {"m0": -2.0, "damped_newton": True, "attr_scale": 1.56, "attr_year_max": 2026.4}
    build(tmp_path, monkeypatch, "v_plugin", params, "--plugin", "paiec.ext.pd1_subjcal:SubjCalHier")
    code = (tmp_path / "model.py").read_text()
    imports = [ln for ln in code.splitlines() if ln.lstrip().startswith(("import ", "from "))]
    assert not [ln for ln in imports if "paiec" in ln]
    shipped = json.loads((tmp_path / "params.json").read_text())
    assert shipped == {"class": "paiec.ext.pd1_subjcal:SubjCalHier", **params}
    mod = import_shipped(tmp_path)
    assert type(mod._MODEL).__name__ == "SubjCalHier" and "class" not in mod._MODEL.p
    assert mod._MODEL.p["attr_scale"] == 1.56
    from paiec.ext.pd1_subjcal import SubjCalHier

    research = SubjCalHier(STUB_PRIOR, params)  # what tools/verify_shipped.py builds from params.json
    inp, labeled = sample()
    for lab in ([], labeled):
        assert mod.predict(inp, lab) == research.predict(inp, lab)


def test_class_key_in_params_selects_the_plugin(tmp_path, monkeypatch):
    params = {"class": "paiec.ext.pd1_subjcal:SubjCalHier", "m0": -2.0}
    build(tmp_path, monkeypatch, "v_cls", params)
    assert type(import_shipped(tmp_path)._MODEL).__name__ == "SubjCalHier"
    with pytest.raises(SystemExit, match="disagrees"):
        build(tmp_path, monkeypatch, "v_cls", params, "--plugin", "paiec.ext.hrB_exact:ExactAloneHier")


def test_plugin_dependencies_and_clashing_helpers(tmp_path, monkeypatch):
    """pd1_prior imports SubjCalHier from pd1_subjcal and defines its own clamped _sigmoid, while subjects.py has an
    unclamped _sigmoid: the dependency is vendored first and the plug-in's helper is renamed, not merged."""
    params = {"q0": 0.25, "damped_newton": True, "tau_delta": 0.6, "tau_phi": 0.3}
    build(tmp_path, monkeypatch, "v_prior", params, "--plugin", "paiec.ext.pd1_prior:PriorCalHier")
    mod = import_shipped(tmp_path)
    assert type(mod._MODEL).__name__ == "PriorCalHier"
    with pytest.raises(OverflowError):
        mod._sigmoid(-1000.0)  # still the subjects.py helper
    assert 0.0 < mod._sigmoid__pd1_prior(-1000.0) < 1e-12
    from paiec.ext.pd1_prior import PriorCalHier

    research = PriorCalHier(STUB_PRIOR, params)
    inp, labeled = sample()
    for lab in ([], labeled):
        assert mod.predict(inp, lab) == research.predict(inp, lab)


def test_plugin_vendoring_refuses_what_it_cannot_ship(tmp_path, monkeypatch):
    ms = load_tool()
    ext = tmp_path / "paiec" / "ext"
    ext.mkdir(parents=True)
    (ext / "uses_data.py").write_text('"""doc"""\nfrom paiec.data import SUBJECT_FIELDS\n\n\nclass A:\n    pass\n')
    (ext / "nested.py").write_text("def f():\n    from paiec.hier import HierModel\n    return HierModel\n\n\n"
                                   "class B:\n    pass\n")
    (ext / "missing.py").write_text("from paiec.hier import NotThere\n\n\nclass C:\n    pass\n")
    (ext / "noclass.py").write_text("X = 1\n")
    monkeypatch.setattr(ms, "REPO", tmp_path)
    vendored = "class HierModel:\n    pass\n"
    for path, pattern in (("uses_data:A", "not vendored"), ("nested:B", "remains"), ("missing:C", "no vendored code"),
                          ("noclass:Z", "defines no top-level Z"), ("absent:D", "not found")):
        with pytest.raises(SystemExit, match=pattern):
            ms.plugin_code(f"paiec.ext.{path}", vendored)
    with pytest.raises(SystemExit, match="must look like"):
        ms.plugin_code("paiec.hier:HierModel", vendored)
