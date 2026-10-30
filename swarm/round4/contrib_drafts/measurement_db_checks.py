"""Data-quality checks on the public AI Measurement Data Bank (aims-foundations/measurement-db).

Reproduces every number in the draft GitHub issue (contrib_drafts/A_github_issue.md). Needs only pandas + pyarrow and a
local copy of the core tables (subjects/items/response/benchmarks parquet files) at revision bc8204d8. Optional second
argument: a checkout of github.com/aims-foundations/measurement-db (for the registry check).

usage: python measurement_db_checks.py /path/to/measurement-db [/path/to/measurement-db-github-checkout]
Read-only. Public training data only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "measurement-db")
GH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SYS_PROMPT = "You are ChatGPT, a large language model trained by OpenAI."
HEADER = re.compile(r"^Date: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\nID: [0-9a-f]+\n$")


def feats(value) -> dict:
    out = {}
    for part in str(value or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def matharena():
    s = pd.read_parquet(ROOT / "matharena/subjects.parquet")
    it = pd.read_parquet(ROOT / "matharena/items.parquet")
    r = pd.read_parquet(ROOT / "matharena/response.parquet")
    f = it.item_features.map(feats)
    it["competition"] = f.map(lambda d: d.get("competition"))
    it["problem_idx"] = f.map(lambda d: int(d.get("problem_idx", -1)))
    it["scale"] = it.grading_criterion.map(lambda g: json.loads(g)["response_scale"]["kind"])
    it["gold"] = it.grading_criterion.map(lambda g: json.loads(g).get("reference_answer"))
    return s, it, r


def section(title):
    print(f"\n== {title}")


def check_subjects(s, r):
    section("1. MathArena subjects without registry fields")
    empty = s[s.normalized_name.isna()]
    print("subjects:", len(s), "| normalized_name null:", len(empty),
          "| provider null:", int(s.provider.isna().sum()), "| release_date null:", int(s.release_date.isna().sum()),
          "| all three null on the same rows:", int((s.normalized_name.isna() & s.provider.isna() & s.release_date.isna()).sum()))
    print("display_name equals source_model_name in subject_features_extra:",
          bool((s.display_name == s.subject_features_extra.map(lambda v: feats(v).get("source_model_name"))).all()))
    share = r.subject_id.isin(empty.subject_id)
    print(f"responses from these subjects: {int(share.sum())} of {len(r)} ({share.mean():.1%})")
    n_items = r[r.subject_id.isin(empty.subject_id)].groupby("subject_id").item_id.nunique()
    print("of these subjects, with >= 80 distinct items:", int((n_items >= 80).sum()))
    cfg = s.subject_features_extra.map(lambda v: feats(v).get("model_config"))
    dup = s[cfg.duplicated(keep=False)].assign(model_config=cfg)
    print("subjects sharing a model_config:", dup[["display_name", "model_config"]].values.tolist())
    if GH is not None:
        reg = json.load(open(GH / "scripts/build_measurement_tables/map_model_registry.json"))
        missing = sorted(set(s.display_name) - set(reg))
        print("curation-guide registry check, labels not in map_model_registry.json:", len(missing))
        canon = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
        keys = {}
        for k in reg:
            keys.setdefault(canon(k), []).append(k)
        near = [m for m in missing if canon(m) in keys or canon(re.sub(r"\(.*?\)", "", m)) in keys]
        print("  of these, a registry key exists after lowercasing, dropping punctuation or a '(...)' qualifier:", len(near))
        print("  missing labels:", missing)


def check_contents(s, it, r):
    section("2. MathArena item contents that are a system prompt or a log header")
    bad = it[it.content.str.startswith(SYS_PROMPT) | it.content.str.match(HEADER)]
    print("item IDs:", len(bad), "| system prompt:", int(bad.content.str.startswith(SYS_PROMPT).sum()),
          "| 'Date: ... ID: ...' header:", int(bad.content.str.match(HEADER).sum()))
    rb = r[r.item_id.isin(bad.item_id)].merge(s[["subject_id", "display_name"]], on="subject_id")
    print("responses:", len(rb), "| by subject:", rb.display_name.value_counts().to_dict())
    print("competitions:", bad.competition.value_counts().to_dict())
    # labels still look like grades of the real problems: compare with all subjects on the same (competition, idx)
    rr = r.merge(it[["item_id", "competition", "problem_idx"]], on="item_id")
    good = rr[~rr.item_id.isin(bad.item_id)].groupby(["competition", "problem_idx"]).response.mean()
    cell = rr[rr.item_id.isin(bad.item_id)].groupby(["competition", "problem_idx"]).response.mean()
    print(f"per-problem correlation with other subjects' success on the same problems: {cell.corr(good.reindex(cell.index)):.2f}")


def check_alignment(s, it, r):
    section("3. MathArena prompts that state a different problem than the graded one")
    fa = it[(it.scale == "discrete") & ~it.content.str.startswith(SYS_PROMPT) & ~it.content.str.match(HEADER)
            & ~it.content.str.startswith("You are given a Math Kangaroo problem")].copy()
    fa["suffix"] = fa.content.str.strip().str[-120:]
    rr = r.merge(fa[["item_id", "competition", "problem_idx", "suffix"]], on="item_id").merge(
        s[["subject_id", "display_name"]], on="subject_id")
    majority = (rr.groupby(["competition", "problem_idx", "suffix"]).subject_id.nunique().reset_index()
                .sort_values("subject_id", ascending=False).drop_duplicates(["competition", "problem_idx"]))
    own = {(c, i): x for c, i, x in zip(majority.competition, majority.problem_idx, majority.suffix)}
    where = {(c, x): i for (c, i), x in own.items()}
    cells = rr.groupby(["display_name", "competition", "problem_idx", "suffix"]).response.agg(["size", "mean"]).reset_index()
    cells["shown_idx"] = [where.get((c, x)) if own.get((c, i)) != x else i
                          for c, i, x in zip(cells.competition, cells.problem_idx, cells.suffix)]
    mis = cells[cells.shown_idx.notna() & (cells.shown_idx != cells.problem_idx)]
    out = mis.groupby(["display_name", "competition"]).agg(items=("problem_idx", "size"), idx_from=("problem_idx", "min"),
                                                           idx_to=("problem_idx", "max"), responses=("size", "sum"),
                                                           success=("mean", "mean"))
    print(out.to_string())
    ref = cells[cells.shown_idx == cells.problem_idx].groupby(["competition", "suffix"])["mean"].mean()
    others = [ref.get((c, x)) for c, x in zip(mis.competition, mis.suffix)]
    print(f"other subjects on the same statements under the matching index: mean success {pd.Series(others).mean():.3f}")
    glm = rr[(rr.display_name == "GLM 4.6") & rr.competition.isin(["aime_2025", "brumo_2025", "cmimc_2025", "hmmt_feb_2025"])]
    print("GLM 4.6 success elsewhere:", glm.groupby("competition").response.mean().round(3).to_dict())


def check_duplicate_runs(s, it, r):
    section("4. The same attempts released under two competition labels")
    fa = it[(it.scale == "discrete") & ~it.content.str.startswith(SYS_PROMPT) & ~it.content.str.match(HEADER)
            & ~it.content.str.startswith("You are given a Math Kangaroo problem")]
    rr = r.merge(fa[["item_id", "content", "competition"]], on="item_id")
    vec = (rr.sort_values("trial").groupby(["subject_id", "content", "competition"]).response
           .agg(lambda v: tuple(int(x) for x in v)).reset_index())
    n_comp = vec.groupby(["subject_id", "content"]).competition.nunique()
    v = vec.set_index(["subject_id", "content"]).loc[n_comp[n_comp > 1].index].reset_index()
    g = v.groupby(["subject_id", "content"])
    out = pd.DataFrame({"combo": g.competition.agg(lambda x: "+".join(sorted(x))),
                        "same": g.response.agg(lambda x: len(set(x)) == 1),
                        "mixed": g.response.agg(lambda x: any(0 < sum(t) < len(t) for t in x))})
    tab = out.groupby("combo").agg(cells=("same", "size"), identical=("same", "sum"),
                                   mixed=("mixed", "sum"))
    tab["identical_among_mixed"] = out[out.mixed].groupby("combo").same.sum()
    print(tab.fillna(0).astype(int).sort_values("cells", ascending=False).to_string())
    aime_only = out.combo.map(lambda c: all(x.startswith("aime") for x in c.split("+")))
    aime = out[aime_only & out.same].reset_index()
    rows = rr.merge(aime[["subject_id", "content"]], on=["subject_id", "content"])
    # subjects with >= 80 distinct binary-graded items, before and after dropping the combined-label copies
    binary = r[r.response.isin([0.0, 1.0])]
    dup_cells = out[aime_only].reset_index()[["subject_id", "content"]]
    combined = rr[rr.competition.isin(["aime_2025", "aime_2026"])].merge(dup_cells, on=["subject_id", "content"])
    kept = binary[~binary.set_index(["subject_id", "item_id"]).index.isin(
        combined.set_index(["subject_id", "item_id"]).index)]
    n80 = lambda df: int((df.groupby("subject_id").item_id.nunique() >= 80).sum())
    print(f"subjects with >= 80 binary-graded items: {n80(binary)} -> {n80(kept)} without the combined-label copies "
          f"({len(binary) - len(kept)} rows dropped)")
    print(f"AIME cells with identical attempts: {len(aime)} for {aime.subject_id.nunique()} subjects; "
          f"second-copy rows: {len(rows) // 2} of {int((r.merge(it[['item_id', 'scale']], on='item_id').scale == 'discrete').sum())} "
          "final-answer rows")


def check_identity(it, r):
    section("5. MathArena problems, item IDs and texts (documentation)")
    print("problems (raw_item_id):", it.raw_item_id.nunique(), "| item IDs:", len(it), "| distinct contents:", it.content.nunique())
    print("items by response_scale kind:", it.scale.value_counts().to_dict())
    rs = r.merge(it[["item_id", "scale"]], on="item_id")
    crit = rs[rs.scale == "interval"]
    print("rubric-criterion grades:", len(crit), "| exactly 0 or 1:", int(crit.response.isin([0.0, 1.0]).sum()))
    kang = it[it.content.str.startswith("You are given a Math Kangaroo problem")]
    print("Kangaroo items:", len(kang), "| distinct contents:", kang.content.nunique(), "| distinct asset manifests:",
          kang.asset_manifest.nunique(), "| image_detail values:", kang.item_features.map(lambda v: feats(v).get("image_detail")).value_counts().to_dict())
    fa = it[it.scale == "discrete"]
    variants = fa[~fa.content.str.startswith(SYS_PROMPT) & ~fa.content.str.match(HEADER)].groupby("raw_item_id").content.nunique()
    print("final-answer problems:", fa.raw_item_id.nunique(), "| with 2-6 prompt-text variants:", int((variants > 1).sum()))
    pr = it[it.scale == "interval"].groupby("raw_item_id").item_id.size()
    print("proof problems:", len(pr), "| item IDs:", int(pr.sum()), "| per problem median/max:", int(pr.median()), int(pr.max()))


def check_small():
    section("6. Smaller items")
    m = pd.read_parquet(ROOT / "multi_swebench/subjects.parquet")
    print("multi_swebench harness values differing only by case:",
          [h for h, n in Counter(x.lower() for x in m.harness.dropna().unique()).items() if n > 1],
          "| 'MSWE-Agent' subjects:", m[m.harness == "MSWE-Agent"].display_name.tolist(),
          "| 'MSWE-agent' subjects:", int((m.harness == "MSWE-agent").sum()))
    rs = pd.read_parquet(ROOT / "real_webagents/subjects.parquet")
    rr = pd.read_parquet(ROOT / "real_webagents/response.parquet")
    for name in ["Test", "Update Eval"]:
        sid = rs[rs.display_name == name].subject_id
        x = rr[rr.subject_id.isin(sid)]
        print(f"real_webagents subject {name!r}: items {x.item_id.nunique()}, success {x.response.mean():.3f}, "
              f"provider {rs[rs.display_name == name].provider.iloc[0]!r}")
    print("real_webagents 'x-ai/grok-4-fast' ->", rs[rs.display_name == "x-ai/grok-4-fast"][["normalized_name", "release_date"]].values.tolist())
    if GH is not None:
        reg = json.load(open(GH / "scripts/build_measurement_tables/map_model_registry.json"))
        for k in ["x-ai/grok-4-fast", "grok-4-fast", "DeepSeek-V3-0324", "DeepSeek V3 0324", "mistral-7b", "mistral_7b"]:
            e = reg.get(k, {})
            print(f"  registry {k!r} -> {e.get('model')!r} ({e.get('company')}, {e.get('release_date')})")


def check_passed():
    section("Checks that passed")
    bad = []
    for b in ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]:
        txt = (ROOT / b / "characterization.yaml").read_text().split("source_claims:")[0]
        declared = {m.group(1): int(m.group(2)) for m in re.finditer(r"^  (\w+):\n    rows: (\d+)", txt, re.M)}
        for t, f in [("items", "items"), ("responses", "response"), ("subjects", "subjects")]:
            n = len(pd.read_parquet(ROOT / b / f"{f}.parquet"))
            if declared.get(t) != n:
                bad.append((b, t, declared.get(t), n))
        r = pd.read_parquet(ROOT / b / "response.parquet")
        if r.duplicated(["subject_id", "item_id", "trial", "test_condition", "interactors"]).any():
            bad.append((b, "duplicate response keys"))
    print("characterization row counts match and no duplicate response keys:", "yes" if not bad else bad)


if __name__ == "__main__":
    s, it, r = matharena()
    check_subjects(s, r)
    check_contents(s, it, r)
    check_alignment(s, it, r)
    check_duplicate_runs(s, it, r)
    check_identity(it, r)
    check_small()
    check_passed()
