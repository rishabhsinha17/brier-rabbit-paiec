# Paper 51: response to the reviews

This response accompanies the camera-ready revision of “Predicting AI Evaluation Outcomes on Unseen Benchmarks:
A Shrinkage-First Hierarchical Model and Matched-Pair Live Evaluation.” Section references below refer to the revised
paper. The selected Codabench submission remains 1020301.

## Expected Brier loss and the finite-population protocol

Both reviewers correctly identified an insufficient assumption in Section 7.2 and Appendix E. Exchangeability alone
does not imply binomial acquisition counts or independence between acquired and evaluation outcomes. We now state
the conditional-IID Bernoulli assumptions, derive the hypergeometric expression for a fixed binary population, and
explain the distinction between averaging over random splits and conditioning on the platform's fixed pools. We
also state why repeated responses and item-dependent predictions require more than a count-only model.

The new Appendix K verifies the finite-population formula by complete enumeration, with maximum discrepancy
5.6e-17. A constructed 3,348-cell grid shows that the sampling assumption can change the REF/W20 expected-score
contrast by up to 0.000656 and reverse its sign in 11 cells. These calculations describe simplified populations.
We have not refitted the original hidden-pool analysis under the finite-population model. The abstract, Section 7.2
and Appendix E now frame those fit statistics as conditional scenario evidence and remove the claim that they
establish the hidden pool's distribution.

## Truncated item keys and the incomplete evidence-cache key

We reproduced both defects. The truncated key creates three collision groups spanning six of 212 distinct
ResearchCodeBench texts. The other four original benchmark item tables have no collisions between distinct full
texts. As reviewer 6Ps7 noted, full-text matching in the cavity lookup does not repair the base posterior.

We provide a separate corrected predictor using a full-text SHA-256 item key and a framed evidence hash that includes
item features. In a synthetic feature-change example, the submitted code returns a cached probability of 0.559657
and a fresh instance returns 0.497601. The corrected code returns 0.497601 in both cases and agrees with fresh
instances across 19 checks involving copied, reordered, replaced and mutated evidence. This reproduces the failure
mode reported by reviewer oDoq using a separate constructed input.

We compared the original and corrected W20 composites in 40 matched LOBO draws across the four original public
benchmarks. We refit the prior within each training fold and keep the selected parameters fixed. Of 25,134 prediction
calls, 148 change, all on ResearchCodeBench. The largest probability change is 0.035523. That benchmark's mean
corrected-minus-original ALC is +0.0112 ± 0.0076 in units of 1e-3. The other folds are identical. The repair did not improve the average score in this small audit, and its effect on hidden competition runs remains
unknown. Sections 4.3, 4.5 and Appendices A, F and K now distinguish tested fixed-attribute behavior from
general cache correctness. The original selected ZIP and historical results remain unchanged.

## Inventory eligibility of the late expanded-data study

Reviewer oDoq correctly identified a gap between the study's public-release filter and the competition's inventory
rule. The saved retained-source table includes FinanceBench and LegalBench. We checked every cell in both worksheets
of the official inventory downloaded on November 24, 2026, and found neither source name. Our original five sources
match the inventory, including the REAL alias. The historical pair-rate artifact also confirms processing of 16 FinanceBench pair records (2,400 responses) and
31 LegalBench records (63,457 responses). A current snapshot cannot establish the historical inventory or resolve
every naming alias.

The author's October 11/18 inquiry has not received a reply or clarification. We have withdrawn the late-study
performance table and conclusions from the paper's supporting evidence. Appendix J retains a disclosure of what
happened and points to the original archived report. Sections 3, 10, 11, 13 and Appendix I now state the unresolved
scope issue. Keeping the expanded-data priors out of the selected submission does not erase that development activity
or settle compliance. The organizers must determine its status. We conducted no new expanded-data fitting or replays.

## Selection claims and reproducibility

The abstract and Section 7 now distinguish passing the implemented selection checklist from independently showing
W20's superiority. The fresh-pair estimate remains -6.0 ± 4.0 in units of 1e-3, z=-1.49. We retain the disclosures
of amendments after interim results, the early final look, the composite robustness-rule override and simulation
omissions. Zero worse-arm switches in the reported simulated campaigns is a scenario-specific observation.

Section 12 and Appendix H now state the public release's limits. Prediction, public-data replays and the new
correction checks are reproducible with the pinned data and official kit. Our formative logs, derived per-pair files
and pre-registration snapshots are not public, so the live estimates and historical selection cannot be independently
rerun from the release. The relevant checklist answers are now No, with that explanation. Original tests pass
(25 passed, 1 skipped because external logs are absent), and the correction checks and raw public results accompany
the revision.

Reviews addressed: [6Ps7](https://openreview.net/forum?id=5P6t3ancaQ&noteId=B6NIbUzlAx) and
[oDoq](https://openreview.net/forum?id=5P6t3ancaQ&noteId=MT7MEZsA3i).

## Scope and formatting of the revision

The original method, parameters, selected artifact and retained numerical tables are unchanged. We restored the
submitted wording wherever the review findings did not require a substantive correction. Detailed table columns and
several descriptive blocks are preserved in Appendices I and L to fit eight main-text pages with normal-size tables.
The acknowledgments use the official unnumbered environment, and the complete checklist is last. Checklist items 8
and 9 now answer No: historical resource details are incomplete, and the unresolved data scope prevents us from
affirming full conformity. These answers expose the remaining limitations rather than claiming that disclosure
settles them.
