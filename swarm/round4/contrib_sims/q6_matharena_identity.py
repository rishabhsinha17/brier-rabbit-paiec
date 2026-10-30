"""Q6: MathArena item identity: problems vs item IDs vs texts; where the extra IDs come from (bc8204d8)."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT = Path((__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/../data/measurement-db"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 100)
def feats(v):
    d = {}
    for part in str(v or "").split(";"):
        if "=" in part:
            k, x = part.split("=", 1); d[k] = x
    return d
it = pd.read_parquet(ROOT / "matharena" / "items.parquet")
F = it.item_features.map(feats)
it["competition"] = F.map(lambda d: d.get("competition", "")); it["problem_idx"] = F.map(lambda d: d.get("problem_idx", ""))
it["image_detail"] = F.map(lambda d: d.get("image_detail", ""))
it["vclass"] = it.verifier.map(lambda v: json.loads(v).get("class", "?") if str(v).startswith("{") else str(v)[:30])
print("raw_item_id examples:", it.raw_item_id.head(3).tolist())
print("raw_item_id distinct", it.raw_item_id.nunique(), "| (competition, problem_idx) distinct",
      it.groupby(["competition", "problem_idx"]).ngroups,
      "| raw_item_id is a function of (competition, problem_idx):",
      bool((it.groupby(["competition", "problem_idx"]).raw_item_id.nunique() == 1).all()))
print("verifier classes:", it.vclass.value_counts().to_dict())
PROOF = {"imc_2025", "imo_2025", "miklos_2025", "putnam_2025", "usamo_2025"}
print("verifier class x proof competition:\n", pd.crosstab(it.vclass, it.competition.isin(PROOF)).to_string())
SYS = it.content.str.startswith("You are ChatGPT, a large language model trained by OpenAI.") | it.content.str.match(r"^\s*Date: \d{4}-")
it["sysprompt"] = SYS
# classify each extra item ID within a (competition, problem_idx) problem
rows = []
for (c, i), g in it.groupby(["competition", "problem_idx"]):
    n = len(g)
    n_sys = int(g.sysprompt.sum())
    ok = g[~g.sysprompt]
    n_text = ok.content.nunique()
    n_img = ok.image_detail.nunique()
    rows.append(dict(comp=c, idx=i, n_ids=n, n_sys=n_sys, n_texts_ok=n_text, n_img=n_img,
                     n_gc=ok.grading_criterion.nunique(), n_ver=ok.verifier.nunique(), proof=c in PROOF))
P = pd.DataFrame(rows)
print("\nproblems", len(P), "| item IDs", int(P.n_ids.sum()))
print("item IDs whose content is a system prompt / log header:", int(P.n_sys.sum()), "on", int((P.n_sys > 0).sum()), "problems")
fa = P[~P.proof]
print("final-answer problems", len(fa), "| item IDs", int(fa.n_ids.sum()), "| problems with >1 prompt text (excl. sys):",
      int((fa.n_texts_ok > 1).sum()), "| extra IDs from text variants:", int((fa.n_texts_ok - 1).clip(lower=0).sum()))
print("  Kangaroo image_detail variants: problems with 2 image_detail values:", int((fa.n_img > 1).sum()))
pr = P[P.proof]
print("proof problems", len(pr), "| item IDs", int(pr.n_ids.sum()), "| median IDs per proof problem", pr.n_ids.median(),
      "| max", int(pr.n_ids.max()))
print("prompt-text variants per final-answer problem:", Counter(fa.n_texts_ok).most_common())
# cross-competition identical texts (excluding system-prompt and generic Kangaroo image prompt)
KANG = it.content.str.startswith("You are given a Math Kangaroo problem in the form of an image")
ok = it[~it.sysprompt & ~KANG]
tg = ok.groupby("content").competition.nunique()
xc = tg[tg > 1]
print("\nidentical problem texts appearing in >1 competition:", len(xc), "texts;",
      int(ok[ok.content.isin(xc.index)].item_id.nunique()), "item IDs")
print("Kangaroo items sharing one generic text:", int(KANG.sum()), "items,", it[KANG].content.nunique(), "distinct text(s),",
      it[KANG].asset_manifest.nunique(), "distinct asset manifests")
# what does a template variant look like
g = it[(it.competition == "aime_2025") & (it.problem_idx == "15") & ~it.sysprompt]
for t in g.content.unique():
    print("  aime_2025 #15 variant:", repr(t[:110]))
# responses affected
r = pd.read_parquet(ROOT / "matharena" / "response.parquet")
print("\nresponses on system-prompt/header items:", int(r.item_id.isin(it[it.sysprompt].item_id).sum()), "of", len(r))
