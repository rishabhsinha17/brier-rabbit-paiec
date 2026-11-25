# Camera-ready verification, 24 November 2026

This plan was saved before running the new checks. No model settings will be selected using these results.
The selected Codabench artifact remains run 1020301. All new work is local CPU work; paid API budget is zero.

1. Reproduce the omitted-feature cache defect with a synthetic evidence change. Check repeated, reordered,
   copied and mutated evidence against fresh model instances. Separate this from immutable protocol streams.
2. Count truncated-text collisions on all five original training benchmarks. Compare original and corrected
   W20 on identical leave-one-benchmark-out draws from the four original evaluation benchmarks, with seeds
   0 through 4 and one or two pairs per draw (benchmark-keyed subsampling). Fit each subject prior without
   the evaluation benchmark. Report per-benchmark paired ALC changes and prediction counts. This small
   diagnostic estimates sensitivity to the correction; it is not a new model-selection experiment.
3. Derive the finite-population expected Brier expression and check it against complete enumeration on a
   small binary population. Compare IID and without-replacement risks for REF and W20 at population sizes
   80, 88, 160 and 226, every possible success count, and the six official budgets. These are constructed
   populations with one binary outcome per item, not the hidden competition pool.
4. Reconcile the saved late-study benchmark table with the current official inventory. Keep an explicit
   record of missing names. Do not use the expanded dataset for new fitting or replays. Withdraw any
   eligibility claim that the source check cannot support.

Record commands, results and hashes in this directory. Keep the original code and experiment tables intact.
