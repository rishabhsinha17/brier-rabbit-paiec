import sys, time
sys.path.insert(0, (__import__("os").environ.get("BRIER_RABBIT_ROOT", ".") + "/swarm/round1/postdoc_priors_sims"))
t0 = time.time()
from light import *
from paiec.subjects import name_key
df = load_light()
print("loaded", len(df), "rows in %.1fs" % (time.time() - t0))
# eligible pairs (>=80 unique items)
g = df.groupby(["bench", "pair"]).agg(n=("item_id", "nunique"), k=("label", "sum"), m=("label", "size")).reset_index()
g["rate"] = g.k / g.m
el = g[g.n >= 80].copy()
print("\n== eligible pairs and pair success-rate distribution per benchmark ==")
for b, h in el.groupby("bench"):
    r = h.rate.values
    print(f"{b:18s} n={len(h):3d} mean={r.mean():.3f} sd={r.std():.3f} q10={np.quantile(r,.1):.3f} q25={np.quantile(r,.25):.3f} med={np.median(r):.3f} q75={np.quantile(r,.75):.3f} q90={np.quantile(r,.9):.3f} "
          f"<2%={np.mean(r<.02):.3f} <5%={np.mean(r<.05):.3f} <10%={np.mean(r<.10):.3f} >90%={np.mean(r>.9):.3f}")
# subject id overlap across benchmarks; name overlap
subj = df.drop_duplicates(["bench", "subject_id"])[["bench", "subject_id", *SUBJECT_FIELDS]].copy()
subj["key"] = [name_key({f: r[f] for f in SUBJECT_FIELDS}) for _, r in subj.iterrows()]
sid_b = subj.groupby("subject_id").bench.nunique()
print("\nsubject_ids:", len(sid_b), "on >1 benchmark:", int((sid_b > 1).sum()))
el_keys = subj.merge(el[["bench", "pair"]].assign(subject_id=lambda d: d.pair.str.split("|").str[1]), on=["bench", "subject_id"])
kb = el_keys[el_keys.key != ""].groupby("key").bench.nunique()
print("eligible-pair canonical names:", len(kb), "on >1 benchmark:", int((kb > 1).sum()), " on >=3:", int((kb >= 3).sum()))
print("eligible pairs with empty name key:", int((el_keys.key == "").sum()), "of", len(el_keys))
for b in sorted(el.bench.unique()):
    ks = set(el_keys[(el_keys.bench == b) & (el_keys.key != "")].key)
    other = set(el_keys[(el_keys.bench != b) & (el_keys.key != "")].key)
    print(f"  {b:18s} names={len(ks):3d} also-elsewhere={len(ks & other):3d}")
# full-dict duplicates across benchmarks (what phi_s keys on)
subj["sk"] = subj[list(SUBJECT_FIELDS)].astype(str).agg("|".join, axis=1)
skb = subj.groupby("sk").bench.nunique()
print("identical full subject dicts on >1 benchmark:", int((skb > 1).sum()))
print("\nharness values per bench:")
for b, h in subj.groupby("bench"):
    print(f"  {b:18s}", dict(h.harness.value_counts().head(8)))
print("release_date range per bench:")
for b, h in subj.groupby("bench"):
    yrs = pd.to_datetime(h.release_date, errors="coerce")
    print(f"  {b:18s} missing={yrs.isna().mean():.2f} min={yrs.min()} max={yrs.max()}")
print("item_features first token examples:")
for b in BINARY_BENCHMARKS:
    v = df[df.bench == b].drop_duplicates("item_id").item_features
    first = v.map(lambda s: s.split(";")[0].split("=")[0] if s else "")
    print(f"  {b:18s} has-features={np.mean(v != ''):.2f} keys={dict(first.value_counts().head(4))} ngroups={v.map(lambda s: s.split(';')[0] if s else '').nunique()}")
print("done %.1fs" % (time.time() - t0))
