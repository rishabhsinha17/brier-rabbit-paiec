# Predicting AI Evaluation Outcomes on Unseen Benchmarks: A Shrinkage-First Hierarchical Model and Matched-Pair Live Evaluation

*Rishabh Sinha (Team Brier Rabbit), independent researcher participating in a personal capacity;
rsinha17@terpmail.umd.edu. Code: https://github.com/rishabhsinha17/brier-rabbit-paiec (MIT license; code at commit 3aeaa02c).*

*Keywords: predictive evaluation; item response theory; hierarchical Bayesian models; Laplace approximation; Brier
score; cold start; matched-pair experiments; pre-registration.*

*TL;DR: A shrinkage-first hierarchical logistic model predicts AI systems' outcomes on unseen benchmarks from 0 to 31
labels; per-pair determinism turns formative feedback into a matched-pair experiment with calibrated errors and hashed
pre-registration.*

*PAIEC technical report, final version, 2026-10-30. The final look, the composite guard and the selected run date from
04:15 to 04:21 UTC on 2026-10-30 (Sections 7.4 and 10, Table 6); other live numbers are as of 03:24 UTC on 2026-10-20
unless dated otherwise; local replays and simulations finished by about 16:00 UTC on 2026-10-06, plus the headroom,
guard and additivity analyses of 2026-10-07 to 10-15 (Section 7.4, Appendices C and D); hashed pre-registration
snapshots run up to 04:21 UTC on 2026-10-30. Sections 1 to 11 are the main text; Appendix H is the NeurIPS paper
checklist. Effects are in $10^{-3}$ Brier ALC, arm minus reference (negative favours the arm), unless a unit is given.*

# Abstract

We predict whether an AI system solves an item of a benchmark absent from training, after 0 to 31 revealed labels of the
same system on that benchmark, scored by the Brier area under the learning curve (ALC). Our predictor is a small
hierarchical logistic model with transferred subject ability, refit at every checkpoint by a damped-Newton Laplace
approximation with probit-moderated predictions, plus three components decided on public data: rescaled
attribute-predicted abilities, cavity-corrected pooling of other systems' labels on the same item, and pooling of pair
effects by model name and harness. On leave-one-benchmark-out replays it lowers Brier ALC by $10.6 \pm 1.7 \times
10^{-3}$ against a smoothed mean, $1.24 \pm 0.22 \times 10^{-3}$ of it from the three components. The platform is
deterministic per (system, benchmark) pair under the default acquisition rule, so we analyse formative feedback as a
matched-pair experiment with null-calibrated standard errors, hashed pre-registration with disclosed amendments, a pair
holdout and other teams' public logs as connector arms. Effects below about 0.002 ALC are hard to resolve live, so
public data decide most settings. Predicted against observed arm contrasts are inconsistent with a narrow unimodal
hidden pool and favour faster level adaptation but do not identify the level, which an adaptive allocation explores
under a pre-registered frequentist switch rule that never adopted a worse arm in simulation. We also correct a replay
bias that understated item pooling by about a third ($-0.21 \pm 0.11 \times 10^{-3}$) and report twenty negative
results. At the final look the wider level prior (W20) passed every switch item against our reference (all pairs
$-1.7 \pm 1.0 \times 10^{-3}$; confirmatory fresh pairs $-6.0 \pm 4.0 \times 10^{-3}$, $z = -1.49$), so the final
submission is W20's level block with the three components. Its selected run scored 0.190 formative Brier ALC on one
draw of 9 pairs.

# 1 Introduction

Predictive evaluation forecasts a system's outcome on items it has not yet been run on. The organizers' measurement
roadmap [2] calls prediction for never-seen items an emerging problem and simulation of never-observed systems the
frontier, and on a new benchmark every item is never-seen. PAIEC tests this regime: test benchmarks are disjoint from the
training pool and hidden behind anonymous IDs, and each (system, benchmark) pair reveals at most 31 labels in a fixed
random order. At budget 0 the prediction rests on a prior over benchmark difficulty learned from four public benchmarks,
and since 80% of the ALC weight is on budgets 1 to 15, the speed at which a model leaves that prior decides most of the
score. The only signal about the hidden pool is formative feedback on one draw of about nine pairs per submission, and
it varies widely: our identical incumbent submission, run 103 times by 2026-10-20, scored with an SD of 0.021.

We took a shrinkage-first approach. The predictor is a Bayesian hierarchical model cheap enough to refit at every
checkpoint and small enough for exact calculations, paired public replays decide whatever public data can decide, and
live tests are reserved for the level prior, which public data cannot identify. We contribute the model and the
submitted composite with its ablation and decision records (Section 4); a protocol replica and a subsampling bias in it
that understated pooling (Section 5); a matched-pair live methodology built on per-pair determinism, with calibrated
standard errors, hashed pre-registration, a holdout, connector arms, a regime test with exact predictions and an
adaptive allocation of uploads, whose operating characteristics we simulate (Sections 6 and 7); an analysis of why one
item variance under-pools, with a cavity correction (Section 8); and twenty graded negative results (Section 9).

**Related work.** Item response theory is the standard tool for AI evaluation with few labels [1, 20, 21]. Our model is
an explanatory, GLMM-style IRT model [27] without per-item discrimination, with shrinkage in the empirical-Bayes
tradition [8, 35]. Instance-level prediction for a new system [17] and task-level prediction in agentic coding [16]
find weak transfer to new benchmarks, benchmark prediction from fewer data fails to extrapolate to new systems that are
more accurate than those seen [18], and Laplace approximations for binary latent-Gaussian models under-react [5, 6,
7]. For evaluation statistics we follow [24] and cluster-robust inference [36].

# 2 Task and protocol

A submission provides `predict([subject, item], labeled) -> float`, where `labeled` contains every label revealed so far
across all pairs of the draw. A (subject, benchmark) pair with at least 80 distinct items has a persistent 50/50 split
into acquisition and evaluation pools. A draw selects complete pairs up to about 1,000 source items (our draws held 8 or
9 pairs), candidates arrive in a fixed random order, and predictions are committed at caps $n \in \{0,1,3,7,15,31\}$,
every pair finishing a checkpoint before any pair gets more labels. $B_n(p)$ is the Brier score [25] of pair $p$'s
evaluation pool at budget $n$, counting every recorded run, and $\mathrm{ALC}(p) = \sum_n w_n B_n(p)$ with $w = (0.1,
0.2, 0.2, 0.2, 0.2, 0.1)$, the normalized trapezoid rule on a $\log_2(1+n)$ axis, averaged over pairs. The Brier score
is strictly proper [26], so calibration and sharpness both count.

**Determinism.** Without an acquisition hook the platform reveals candidate $x$ of pair $p$ if and only if $u(x) <
\min(1, L_{\mathrm{rem}} / I_{\mathrm{rem}})$, where $u$ is a SHA-256 hash of the candidate's input and a context of
pair IDs and label counters, and $L_{\mathrm{rem}}$ and $I_{\mathrm{rem}}$ count the labels and candidates left. With
the persistent split, a pair's revealed labels and evaluation items are then the same in every submission, which differ
on a pair only through their predictors and the labels of other pairs in the draw. In 455 formative logs (401 ours, 54
public; 2026-10-20) no pair's evaluation-pool size changed and the budget-0 scores of all 768 arm-pair combinations seen
more than once were identical; on 2026-10-06 all 50 repeated alone-pair scores were identical at all six budgets.

# 3 Data

We trained and tuned only on the public measurement-db release (`aims-foundations/measurement-db`, revision `bc8204d8`;
curation under CC BY-SA 4.0, each benchmark under its upstream licence) [3]. On 2026-10-06 the organizers published
formatted tables for about 280 benchmarks on the dataset's `main` branch (revision `54074453`), against six at
`bc8204d8`, whose tables are unchanged except one added Multi-SWE-bench item and its rows. The competition page does not
say whether the
new benchmarks belong to the public training pool (the dataset card still lists the six), so nothing in the submission
or in this report uses them: every number here comes from revision `bc8204d8`, which is still served.

**Table 1.** Public training data, binary-valued responses only. Script:
`swarm/round3/report_writer_sims/data_facts.py`.

| Benchmark | Binary responses | Item IDs (distinct texts) | Subjects | Eligible pairs | Median items per pair | Mean success |
|----------------------------|---------|-------------|--------|--------|--------|--------|
| MathArena [28] | 92,861 | 1,751 (894) | 104 | 81 | 213 | 0.654 |
| Multi-SWE-bench [29] | 57,975 | 2,126 (2,078) | 82 | 82 | 495 | 0.148 |
| REAL [30] | 4,449 | 233 (233) | 33 | 26 | 112 | 0.358 |
| ResearchCodeBench [31] | 6,784 | 212 (212) | 31 | 31 | 212 | 0.361 |
| SWE-rebench [32] (subject prior only) | 67,074 | 6,306 (6,271) | 1 | 1 | 6,306 | 0.479 |

Our loader keeps every row graded exactly 0 or 1, so MathArena's rows include 3,379 rubric-criterion grades besides
89,482 final-answer grades. MMDocRAG [33] is graded fractionally, so we exclude it, as the official baseline does. Local
evaluation uses the four multi-pair benchmarks (220 eligible pairs). For 52 MathArena subjects whose `normalized_name`,
`provider` and `release_date` are empty we recover the canonical name from `subject_features_extra`, giving 122 named
models, 96 on one benchmark only. Item difficulty is largely shared across systems (crossed Rasch item SDs of 2.5 to 3.1
logits outside MathArena), and the first `item_features` token explains 1% to 39% of item easiness variance. We use no
pretrained models, embeddings, runtime APIs or inventory-wide catalogs. The MathArena tables also hold system prompts
stored as item content, prompts that differ from the graded problem and repeated attempts (202 items, 19 items and
5,848 rows; Appendix C). Local numbers include these rows; removing them with the rubric grades moves the composite's
MathArena gain from $-2.55 \pm 0.74$ to $-3.30 \pm 0.69$ (Appendix C).

# 4 Model

## 4.1 Structure

For subject $s$, benchmark $b$ and item $i$,
$$\operatorname{logit} P(y_{sbi} = 1) = \mu_b + \lambda\theta_s + \phi_s + \delta_{sb} + \sum_{g \in G(i)} \gamma_{bg} + \epsilon_{bi}.$$
The level $\mu_b \sim N(m_0, \tau_\mu^2)$ starts every test benchmark at the same prior, and the transferred ability
$\theta_s$ (Section 4.2) enters with weight $\lambda$ as a known offset whose variance is added at prediction time. The
subject offset $\phi_s \sim N(0, \tau_\phi^2)$ and pair deviation $\delta_{sb} \sim N(0, \tau_\delta^2)$ are told apart
only by a subject's labels on another benchmark of the same draw; without such labels they act as one pair effect with
variance $\tau_\phi^2 + \tau_\delta^2$. The group effect $\gamma_{bg} \sim N(0,
\sigma_g^2)$ belongs to the item's first `item_features` token, and the item effect $\epsilon_{bi} \sim N(0,
\sigma_i^2)$ is keyed by item text. Because a pair's two pools are disjoint, mostly other systems' labels inform it (on
MathArena, with its shared texts, a pair's own labels can too). REF, the incumbent, uses $m_0 = -2.0$, $\tau_\mu = 1.2$,
$\lambda = 1.0$, $\tau_\phi = 0.5$, $\tau_\delta = 1.0$, $\sigma_g = 1.0$ and $\sigma_i = 0.6$ (Appendix A gives every
arm and the source of each value).

## 4.2 Subject prior

For each canonical name, $\theta_s$ averages the benchmark-centred smoothed logits $\log\frac{k + 0.5}{n - k + 0.5}$ of
its public pairs over the $K$ benchmarks that contain it, shrunk by $K/(K+1)$, with variance $1/(K+1)$. Unseen names get
$\theta_s$ from a weighted ridge regression on 17 attributes and an intercept (provider, release year, reasoning effort,
a reasoning-mode flag, size words). In leave-one-benchmark-out fits $\theta_s$ correlates 0.30 to 0.47 with the
held-out pair logit (one correlation per public benchmark, each over its 26 to 82 eligible pairs).
Because the ridge is fit on shrunk targets, its predictions are compressed by about 1.56 (Section 4.6).

## 4.3 Inference and prediction

At every new evidence set the model finds the posterior mode of all effects present in `labeled` and the Laplace
covariance $\Sigma = H^{-1}$, cached under a hash of the evidence. Undamped Newton steps diverged under prior-data
conflict (at $m_0 = -2.0$, 31 labels at 74% success drove $\hat\mu_b$ to $-13.5$). The solver now halves each step until
the log posterior does not decrease, which gains $-0.43 \pm 0.22$, all at B15 and B31 (Table C1). The predictive mean
$m$ and variance $v$ add the posterior moments of the target's effects in the fit and the prior moments of absent
effects, and the output is the probit approximation [4] $\hat p = \sigma(m / \sqrt{1 + \pi v/8})$, clipped to $[0.005,
0.995]$. In a synthetic study the plug-in $\sigma(m)$ cost $+3.95 \pm 0.14$ and was worse at every budget. For the
reference subject used throughout ($\theta = 0$ with variance 0.25, no item group), REF predicts 0.211 at budget 0.

## 4.4 Adaptation speed

Laplace with probit moderation under-reacts (Figure 1): after 0 of 7 REF predicts 0.101 against the exact 0.067, and
after 7 of 7 it predicts 0.761 against 0.806, the known Laplace bias for binary models [5, 6, 7]. We keep it because exact
inference did not win on public replays ($+1.3$ to $+2.5$ per alone pair on each benchmark, each about one SE) and
ranged from $-0.6$ to $+0.7$ under the hypotheses that Section 7.2 does not reject. Switching would also need a joint
re-tuning of the level block.

![Figure 1. Prediction for an unseen item of the reference subject on a new benchmark after 0/n (left) and n/n (right) own labels on distinct items, with the real model class. Dashed: exact Bayes under REF's prior. Script: `swarm/round3/report_writer_sims/paper_figures.py`.](../swarm/round3/report_writer_sims/fig2_adaptation.png){width=46%}

## 4.5 Engineering

The predictor uses numpy only, is deterministic and never raises, and before every upload the generated single-file
model is checked to reproduce the research model bit for bit (maximum difference 0.0 over 2,166 replay predictions;
Appendix A). The organizers allow external packages on request, installed under their control (their Discord reply of
2026-10-06; the website still says packages are disabled by default); the submission needs only numpy and ships no
`requirements.txt`, so the policy does not bind it.

## 4.6 The submitted composite

The submitted predictor adds three components to the level block chosen live (Section 7). Each passed a local adoption
rule written before its decisive replay finished, none was tuned on formative feedback, and each leaves the predictions
it does not target unchanged (P up to floating-point rounding). A later robustness rule for the whole composite failed
at B0 on one benchmark (dense-draw record below). **S** multiplies attribute-path abilities by 1.56 (refitting the
ridge on unshrunk targets scales its predictions by 1.56 to 1.60 in each fold) and clamps the release-year feature at
2026.4, the last public release. Name-path abilities are unchanged. **C** is cavity item pooling with $\sigma_x = 3.0$
(Section 8), which acts only on items another system of the draw labelled. **P** splits the pair effect into a component
shared by subjects with the same canonical model name (on any benchmark), one shared by subjects with the same harness
on the same benchmark, and a remainder, with variance shares 0.45 and 0.288 of $\tau_\phi^2 + \tau_\delta^2$, so every
pair's prior variance is unchanged. On public data the pair residuals of one model on two benchmarks correlate at
$+0.49$ (72 names, pooled over benchmark combinations by Fisher $z$, SE 0.14 on the $z$ scale), and those of models
sharing a harness on Multi-SWE-bench at $+0.82$ (401 pairs of pairs).

**Table 2.** Composite ablation on one benchmark-keyed replay (18 draws per benchmark and $k \in \{1, 2\}$, 216
pair-draws per arm; draw-clustered SE). Benchmark columns are ALC (MSWE = Multi-SWE-bench). Script:
`swarm/round5/writer_sims/paper_tables_v3.py`.

| Comparison | $\Delta$ALC (SE) | MathArena | MSWE | REAL | RCB | B0 | B1 | B3 | B7 | B15 | B31 |
|------------------|-------------|-----------|-------|------|------|-----|-----|-----|-----|-----|-----|
| S vs REF | -0.88 (0.21) | -2.19 | -0.25 | -1.14 | +0.06 | -2.6 | -1.6 | -0.9 | -0.5 | -0.2 | -0.1 |
| C vs REF | -0.35 (0.10) | -0.35 | -0.29 | -0.25 | -0.51 | 0.0 | -0.1 | -0.2 | -0.3 | -0.6 | -1.1 |
| P vs REF | 0.00 (0.01) | 0.00 | -0.01 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S + C + P vs REF | -1.24 (0.22) | -2.55 | -0.54 | -1.41 | -0.46 | -2.6 | -1.6 | -1.1 | -0.8 | -0.8 | -1.2 |
| S + C + P on W20 vs W20 | -1.07 (0.17) | -1.74 | -0.57 | -1.28 | -0.69 | -2.5 | -1.2 | -0.9 | -0.7 | -0.7 | -1.2 |
| S + C + P on L15 vs L15 | -1.31 (0.22) | -2.27 | -0.66 | -1.52 | -0.79 | -2.9 | -1.7 | -1.2 | -0.8 | -0.8 | -1.2 |

Together the components gain $-1.24 \pm 0.22$, negative in every benchmark and budget column: S at B0 to B3, mainly on
MathArena and REAL, and C mostly from B3 on and only when a benchmark has two pairs ($k = 2$: $-0.70 \pm 0.19$). P
changed 6 of 216 pair-draws by more than $10^{-6}$ (all Multi-SWE-bench at $k = 2$; the rest agree with REF to within
$3 \times 10^{-16}$), and its evidence is a replay with the same model on two held-out benchmarks ($-0.58 \pm 0.22$;
different-name controls exactly 0). The components are additive (residual $-0.004 \pm 0.002$), and one plug-in class
carries all three, bit-identical to each stand-alone class on 4,158 recorded calls at five level blocks.

**Decision records** (Appendix C). Four variants were not adopted under their pre-registered rules: SY1.0, a version
of S that leaves the release-year coefficient unscaled, is worse than S on public replays ($+0.53 \pm 0.14$) and on a
strict temporal hold-out that S wins ($-3.98 \pm 0.90$ and $-3.82 \pm 1.06$ at cut-offs 2025.5 and 2026.0; newer
systems alone on their benchmark, the closest public analogue to the eligible test subjects [37]; Table C6); a
provider-free ridge (NP) broke its per-benchmark cap; a larger item-group variance failed its rule; and $\lambda = 1.6$
gains mainly where we read it as compensating a mis-centred level. One rule failed and was overridden: the composite's
dense replay (four and eight pairs per benchmark; Section 8) carried a rule written in its job file before it ran, and
S + C + P failed it at B0 on ResearchCodeBench through S ($+2.81 \pm 0.81$ at $k = 4$). We kept S + C + P, a decision
taken after the result and hashed at 13:51 UTC (Table D3), because the rule targeted early-budget costs of pooling,
which did not occur, the composite's ResearchCodeBench ALC is negative at every density and S wins the hold-out; the
literal rule is reported as failed, and Appendix C gives every rule, result and reason.

# 5 Local evaluation

**Replica and replays.** `paiec/protocol.py` reproduces the streaming evaluation (split, candidate order, nested caps,
the default acquisition rule verbatim, shared evidence, de-duplication, scoring of every run) and imports the kit's own
`AcquisitionHook` and `prediction_input_key`. All 26 unit tests pass. In leave-one-benchmark-out (LOBO) replays the
subject prior is fit on the other public benchmarks, and a draw holds $k \in \{1, 2\}$ pairs of the held-out benchmark,
subsampled so that evaluation pools match the platform's 44 to 113 items. Every local number is a paired difference on
identical draws, benchmark-equal and $k$-averaged, with a draw-clustered SE. We report per-benchmark values because
averages hide failures [34].

**Use of AI assistance.** An LLM coding assistant (Claude Code) wrote most of the model, replica and analysis code and
drafted this text under the author's direction; Section 13 gives the disclosure and the checks behind every number.

**Table 3.** Local paired replays on one draw set (benchmark-keyed, 18 draws per benchmark and $k$; a constant 0.5
scores 0.2500 and the organizers' empirical mean 0.2250). Each $\Delta$ compares a row with the row named in "vs".
Script: `swarm/round5/writer_sims/paper_tables_v3.py`; Table C1 holds the pair-keyed ladder, Figure C1 plots this one.

| Model | Macro ALC | $\Delta$ALC (SE) | vs | $\Delta$B0 | $\Delta$B1 | $\Delta$B3 | $\Delta$B7 | $\Delta$B15 | $\Delta$B31 |
|--------------------------------|-------|-----------|-------------|------|------|------|------|------|------|
| Smoothed mean ($\kappa = 2$) | 0.2016 | -48.4 (5.2) | constant | +16.1 | -25.3 | -51.3 | -63.9 | -71.6 | -75.3 |
| Hierarchical, untuned ($m_0 = -0.5$) | 0.1947 | -6.9 (1.1) | smoothed | -34.2 | -11.6 | -2.2 | +0.2 | -1.8 | -4.3 |
| Hierarchical, tuned | 0.1925 | -2.2 (0.5) | untuned | -0.9 | -0.9 | -2.5 | -2.9 | -3.0 | -3.1 |
| + subject prior ($m_0 = -0.5$) | 0.1891 | -3.4 (0.5) | tuned | -11.5 | -6.0 | -3.1 | -1.4 | -0.5 | -0.2 |
| REF: damped Newton, $m_0 = -2.0$ | 0.1923 | +3.2 (1.7) | + subject prior | +22.9 | +6.3 | +0.1 | -0.9 | -0.7 | -0.5 |
| REF + S + C + P (Section 4.6) | 0.1910 | -1.24 (0.22) | REF | -2.6 | -1.6 | -1.1 | -0.8 | -0.8 | -1.2 |
| W20 + S + C + P | 0.1888 | -2.20 (0.66) | REF + S+C+P | -7.8 | -5.0 | -2.1 | -0.1 | 0.0 | 0.0 |
| L15 + S + C + P | 0.1880 | -3.06 (0.54) | REF + S+C+P | -14.9 | -6.0 | -1.7 | -0.3 | +0.1 | +0.1 |
| REF + S + C + P vs smoothed mean | | -10.6 (1.7) | smoothed | -26.3 | -13.8 | -8.9 | -5.8 | -6.8 | -9.3 |

The hierarchical model gains most at budgets 0 and 1, through the subject prior and the moderated predictive. Moving to
the live-chosen $m_0 = -2.0$ costs $+3.2 \pm 1.7$, all at B0 and B1, and the best level flips across public benchmarks
(MathArena $+25.7$, Multi-SWE-bench $-9.9$ for that move): with four public benchmarks the level of hidden benchmarks is
not identified, so it is the one block we tune live.

**A replay-subsampling bias.** The original replica subsampled items by a hash keyed by the pair, so co-drawn pairs kept
independent subsets and a partner covered as few of a pair's items as in the larger public groups. Keying the subsample
on the benchmark fixes this (B31 partner coverage on Multi-SWE-bench 0.015 to 0.051; Table C3). On identical picks
cavity pooling gains $-0.39 \pm 0.16$ under the old subsample and $-0.60 \pm 0.18$ under the new one: the old replica
understated the value of item pooling by about a third (paired difference $-0.21 \pm 0.11$, Table C4).

# 6 Live evaluation methodology

**Data and use of IDs.** We use our own logs (401 scored runs by 2026-10-20; 496 at the final look) and, as
connector arms in
arm-level contrasts, the public MIT-licensed logs of two other teams (52 from `aghasalim/predictive-eval` at `e40aa6d0`,
3 from `doctawho42/PAIEC` at `d64ff8b9`), cited at those commits and used as their MIT licenses allow. Anonymous IDs are used only as opaque keys: to join a pair's results across submissions,
as the organizers intend, to assign the holdout by a fixed hash, and to mark whether a pair shared its benchmark within
a draw (Appendix D lists every use). No per-benchmark or per-pair value is used otherwise or reported, we never tried to
identify a benchmark or subject, feedback chose only global settings that act identically on every pair, and feedback
numbers are never model inputs. On 2026-10-17 the platform created 529 submissions of our own zips in our account that
we had not uploaded; none was scored and all were removed by 2026-10-19, so none entered the evidence. Uploads paused
while they might count toward the per-person cap, and the 10-17 and 10-18 windows were missed while our upload session
was down (Appendix D).

**Estimators and errors.** Two-way fixed effects $y_{ua} = \alpha_u + \beta_a + e_{ua}$ with $\beta_{\mathrm{REF}} = 0$
identify arm effects directly or through chains. A co-drawn pair of the same benchmark improves a pair's score through
pooling ($-5.0 \pm 2.7$ for our arms), so the condition, alone or sharing, must enter. The pre-registered primary
estimator is a pair fixed effect with a shared-condition covariate on our logs alone ("ours only"). The secondary one
adds the connector arms and uses alone cells for B1 to B31 and every cell for B0 ("with connectors"). Every live number
here gives both. A stratified pair bootstrap was anti-conservative in a null simulation on the real design, and our
first tool was too small by a median factor of 2.05, so we use $\mathrm{SE}_h = \max(1.2\,\mathrm{SE}_{\mathrm{boot}},
\mathrm{SE}_{\mathrm{CR1}}, \mathrm{SD}_{\mathrm{arm}}/\sqrt{K})$, with a pair-clustered CR1 sandwich [36], $K$ the
pairs matched directly with REF and a floor on $\mathrm{SD}_{\mathrm{arm}}$ (ALC 0.0147; Appendix D). Then 20 REF and 10
challenger runs match about 40 pairs for an ALC SE of 0.0023; by 2026-10-20, W20's 87 runs matched 141 selection-set
pairs with REF for an ALC SE$_h$ of 0.0012, at the floor, and local ladder steps of 0.001 to 0.002 are decided on
public data.

**Pre-registration and its timeline.** Rules and predictions were written before the uploads or replays they govern, in
the round reports and a queue file, and from 10:14 UTC on 2026-10-06 also as SHA-256-hashed snapshots of the queue file
(`submissions/prereg/LOG.csv`); from 2026-10-07 each snapshot is also a commit in the author's local git history. A new arm is dropped after about six runs if its ALC effect exceeds
$+3.5$ or B0, B1 or B3 is two SEs worse than REF, and the incumbent changes only if a six-item checklist passes,
including a holdout effect of at most zero and $z \le -1.64$ on all pairs ($-1.8$ with two or more finalists) [22, 23].
The holdout (a fixed hash of the pair IDs; 21.8% of the 303 distinct pairs seen by 2026-10-20) was implemented at 09:14
UTC, after the decisions of Section 7.1 and before any decision under the screen rule, so it gives no clean check of
the earlier choice of REF.
Each revision of the live rules was recorded before the uploads it governs (Tables D3 and D4; Appendix D states every
rule and revision). Of the five on 2026-10-06, one (10:14 UTC) let disagreement between the estimators keep $m_0 =
-2.0$, a reading written after interim data that favours the incumbent, so Table 4 reports both readings, and the last
(16:03 UTC), after all feedback of that day, dropped the SD floor from the switch SE at the final look for arms with at
least 60 direct matches with REF, after simulations showed that it blocked effects weaker than about $-1.7$. Later
amendments added the exploration arm MXs (2026-10-07), additivity guards for composites on new level blocks (2026-10-07
and 10-15), a composite readiness amendment (2026-10-16), a treatment rule for the platform-created runs above and
insurance builds (2026-10-19; Section 10).

# 7 Evidence about the hidden pool

This section reports arm-level contrasts and global summaries only. Anonymous IDs stay fixed across submissions, and
every draw is selected by the same documented rule from the formative evaluation data (kit README), so draw-level
hyperparameters are constants of that data, better estimated across draws than inside one draw [11]. A pair can share
its benchmark with another pair of its draw, and then pooling improves its score (Section 6).


## 7.1 Early decisions, re-estimated

Our first level moves predate the matched-pair estimator ($m_0$ from $-0.5$ to $-1.5$ after two single draws, then to
$-2.0$, REF, on five matched pairs with an SE tool later found about half the true SE). Re-estimated with
calibrated SEs (Table D1), the early ALC contrasts with connectors stay within 1.4 SE of zero (1.8 SE without the one
public log that the pre-registered connector set excludes), except one shared-level variant ($+24.8 \pm 7.4$); on our
logs alone none exceeds 1.3 SE and six of 13 change sign, so the early drops rest on the connector
estimator plus local evidence (Section 9). Live, REF and $m_0 = -1.5$ still differ by at most about one SE after 43 L15 runs
($+0.4 \pm 1.7$ with connectors, $-1.5 \pm 1.4$ ours only; Table 4), while public replays favour $-1.5$ by
$2.4 \pm 0.5$: REF remains the default of our rules only as the incumbent; the evidence does not support it over
$m_0 = -1.5$.

## 7.2 A regime test with predicted and observed arm contrasts

For a pair alone on its benchmark with success rate $r$, random acquisition makes revealed labels exchangeable with
evaluation items, so
$$\mathbb{E}[B_n \mid r] = \sum_{k=0}^{n} \binom{n}{k} r^k (1 - r)^{n-k}\left[(q(k, n) - r)^2 + r(1 - r)\right],$$
where $q(k, n)$ is an arm's prediction after $k$ successes in $n$ own labels, tabulated with the real model classes and
validated replicas of the connector teams' predictors. A hypothesis about the hidden pool transforms the 220 public pair
rates (logit shift, spread factor, share moved to a low-success component), and averaging over pairs gives its predicted
contrasts, which we compare with 13 observed arm contrasts at B0, B1, B3 and B7 (52 numbers, with connectors, joint
bootstrap covariance) plus one global constraint, the benchmark-balanced mean of B31 (Appendix E).

Against the best point of a 315-point grid of transformations, $\Delta\chi^2$ is 63 for a narrow core (R1: spread
$\times 0.8$, shift $-0.5$), 78 for a zero-inflated pool (R2: 35% of pairs near 1.5%), 13 for a wide pool (R3: spread
$\times 1.3$) and 34 for the unchanged public pool (Table E1 adds six variants and the pre-registered-set rerun: 73,
90, 9.9 and 30). A parametric bootstrap that refits the
whole grid in every replicate (Table E2) puts the 99% point of $\Delta\chi^2$ at 9 to 13: R1, R2 and the unchanged
public pool are rejected ($p < 0.001$; $p \le 0.003$ after scaling the covariance by the best point's misfit factor of
1.68), while R3 is inconsistent with the data at $p = 0.005$ but not after that scaling ($p = 0.07$). The best grid
point itself misfits ($p = 0.0035$) unless the subject-calibration nuisance of Section 7.3 is added, which lowers its
$\chi^2$ from 83.8 to 55.1 ($p = 0.35$; Appendix E). The cleanest experiment is another team's trial of three arms that
differ only in how fast the level moves: the faster arm did better than the base arm at B1 ($-7.5 \pm 5.0$, SE$_h$)
and the slower arm lost at B7 ($+2.5 \pm 1.3$), against R1 predictions of $+16.1$ and $-1.0$. R1 is inconsistent with
the data in every variant except one restricted to our sparse arms, and R2 fits the spread trial but not budget 0 or
the B31 constraint (Appendix E). The best transformations are as dispersed as the B31 constraint allows and include a
low-success component, which favours faster level adaptation over narrow pair priors and zero-inflated mixtures.

## 7.3 The level is not identified globally

The connector arms are not penalized at B0 although their budget-0 predictions are well above ours, while our own $m_0$
sweep prefers $-2.0$ to $-1.0$. A joint fit reconciles the two only through a global offset of our subject prior on
hidden subjects ($\Delta\chi^2 = 29$ for 1.5 to 2 logits). Our prior does not explain such an offset: on a strict public
temporal hold-out it under-predicts newer systems by 0.6 to 1.2 logits and places a realistic 2025 to 2026 cohort only
0.2 to 0.7 logits above the public reference, so most of the offset would belong to the benchmark level, which $m_0$
absorbs. We keep two hypotheses, H_pub (the prior is calibrated on hidden subjects) and H_cal (the offset exists, so
REF's level is near our budget-0 optimum), and pre-registered two probes (Table 4).

**Table 4.** Predictions and interim observations (B0/B1/B3/B7 in brackets). Predictions computed at 08:37 UTC and
recorded with the rules at 08:50 UTC, before the first probe upload at 09:00 UTC (file times); they are pre-registered
for every row except $m_0 = -2.5$, whose prediction comes from a regime fit that already included two of its three
runs. H_cal is the mean over its joint-fit cells, range in brackets. Observed: selection set at 03:24 UTC on
2026-10-20, SE$_h$, ours only / with connectors; at the final look (04:15 UTC on 2026-10-30) L15 $-1.2 \pm 1.3$ /
$-0.4 \pm 1.5$ (49 runs), W20 $-1.8 \pm 1.2$ / $-0.0 \pm 1.4$ with B1 $-3.0 \pm 3.8$ / $+1.1 \pm 4.5$ (101 runs), and W20
at $m_0 = -1.5$ $-1.2 \pm 1.7$ / $-0.7 \pm 1.6$ (35 runs). Scripts: `swarm/round2/adjudicator_sims/recommend2.py`,
`tools/daily_report.py`.

| Arm | Change from REF | H_pub | H_cal | R3 | Observed (interim) |
|------------|--------------|----------------------|-------------------|-----|--------------------------------|
| L15 | $m_0 = -1.5$ | -3.1 (-13.1/-6.0/-2.2/-0.6) | +1.6 (-0.3 to +3.6); B0 +6.7 | -4.5 | ALC -1.5 ± 1.4 / +0.4 ± 1.7; B0 -7.9 ± 5.4 / -7.4 ± 5.4 (43 runs) |
| W20 | $\tau_\mu = 2.0$ | -3.1 (-8.5/-6.4/-3.1/-1.25) | -0.25 (-1.9 to +0.6); B1 +0.4 | -3.7 | ALC -2.0 ± 1.2 / +0.6 ± 1.5; B1 -3.7 ± 4.1 / +2.3 ± 4.9 (87 runs) |
| W20 at $m_0 = -1.5$ | both | -4.3 | +0.9 (-1.9 to +2.4) | -5.9 | ALC -1.2 ± 1.8 / +0.0 ± 1.7 (35 runs) |
| d06 | $\tau_\delta$ 0.6, $\tau_\phi$ 0.3, $m_0$ -1.69 | +0.03 | +1.5 | -0.7 | no feedback (never left the platform queue) |
| $m_0 = -2.5$ | $m_0 = -2.5$ | +4.4 | +1.0 | +5.8 | ALC +3.9 ± 5.1 / +9.5 ± 6.4 (3 runs; dropped) |

L15's budget-0 contrast separates the two predictions ($-13.1$ against $+6.7$, per-pair SD about 35). After 43 runs the
original reading (Appendix D) is undecided ($z = -0.77$ against $-3.2$), and under the amended one both estimators
lie below $-3.2$, selecting H_pub, a reading we report only for the record. In simulation the rule is a weak test of
the offset $d$ (it calls H_pub in only 50% of design-posterior worlds with $d \le 0.75$; Appendix D), the design
posterior replaced it for arm choice on 2026-10-06 (P(H_pub) 0.24 then, 0.98 on 2026-10-20), and we do not cite the
test's outcome as evidence on $d$. W20 widens the level prior (after one success it predicts 0.53 instead of 0.42, with budget 0 almost unchanged); its B1 effect is negative ours only, as H_pub predicts, but positive with
connectors (Table 4); local and live evidence agree poorly on the level, since raising $m_0$ helped B0 locally but hurt
it live in round 1 and public replays favour both probes almost entirely through MathArena (Appendix C).

## 7.4 Adaptive allocation of the remaining uploads

From 2026-10-07 an adaptive allocation replaces the fixed rotation of REF, L15 and W20 (Appendix D). A grid posterior
over pool regimes, our prior's offset $d$ (H_pub is $d \le 0.85$) and a discrepancy scale is fitted to the selection-set
contrasts and the B31 constraint without the holdout pairs; arm predictions are the exact calculation of Section 7.2
plus a Gaussian-process discrepancy over $(m_0, \tau_\mu, \tau_\delta)$. Each day the final run counts of at most five
non-REF grid arms are chosen to minimize the expected true effect of whatever the switch checklist would select, REF
keeps at least 20% of the uploads, and the screen still drops arms permanently. In 225 simulated live streams from the
policy's own posterior (which favours it), the arm the plan selected beat the rotation's choice by $-1.18 \pm 0.10$ in
true ALC and was never worse than REF; without arm discrepancies the gain is $-0.40 \pm 0.19$.

**Operating characteristics.** In 1,780 simulated campaigns from the state of 2026-10-06 to the final look, on pools
calibrated to our logs (Appendix D), the amended procedure never switched when all candidates equal REF on the
formative pool (0 of 77), switched in 3% to 5% when they equal REF only in a larger population, never switched to a
worse arm (0 of 424), and selected an arm better by 2 in 97% of campaigns and one better by 1 in 29% (88% and 1%
before the amendment).

**Headroom (2026-10-07).** An exact calculus over the design posterior refit with that day's runs (which put
P(H_pub) at 0.87, from 0.24 the day before) bounds what any level rule for pairs alone on their benchmark
can still gain over W20 at $-1.37$; the best normal level prior captures 28% to 35% of that, so this analysis added no
normal-family arm, and a two-component level prior matched to the posterior (MXs; Table A1), which captures 92% in
the calculus but is not robust across views, was pre-registered at 05:10 UTC as an exploration arm outside the grid.
Appendix D gives the calculus, the guards and a red-team pass that found no lever worth 2 (Appendix G, item 20).

**State on 2026-10-20.** With 401 scored runs (103 of REF), four arms are provisional finalists (Tables 4 and D6): W20
($-2.0 \pm 1.2$, $z = -1.60$, 87 runs), W15m16 ($m_0 = -1.5$, $\tau_\mu = 1.6$, $\tau_\delta = 1.4$; $-2.1 \pm
1.6$, 39 runs), MXs ($-1.6 \pm 1.5$, 30 runs) and L15 ($-1.5 \pm 1.4$, 43 runs), all within 0.6 of zero with
connectors. Read daily with the SD floor, none passes the switch checklist (W20 and W15m16 fail only (b), $z = -1.57$
and $-1.22$ against $-1.8$); a final-look rehearsal on the data of 2026-10-16, without that floor, had W20 pass every
item ($z = -2.17$). The design posterior gives P(H_pub) 0.98 and W15m16 the lead (best among the shortlisted arms and
REF with probability 0.38; an interim commit needs 0.80), with probability 0.13 that no arm passes. The final look
below made the decision.

**Final look (2026-10-30).** It ran at 04:15 UTC on 2026-10-30, before the pre-registered window of 12:00 to 20:00 UTC,
because the author asked for the release then and earlier sessions of the upload agent had ended without warning.
Availability set the timing, which a hashed snapshot recorded before the run. Rules and thresholds stayed as
registered, and the guard tool gained only an option to read rule (b) on an insurance build and its parent together, as
that day's plan specified. Of that day's runs, five were still in flight after two hours and counted as dead before
scoring. Since the
state above, uploads ran on 2026-10-21, 10-28 and 10-30, and we missed the windows of 2026-10-22 to 10-27 and of 10-29
(Appendix D). With 496 scored runs (150 of REF), the finalists were the five non-dropped arms with at least 30 runs
(W20, W15m16, L15, MXs and W20 at $m_0 = -1.5$), so item (b) needed $z \le -1.8$. W20 passed every item: holdout
$-1.5 \pm 2.4$ ($K = 38$), all pairs $-1.7 \pm 1.0$ with $z = -2.06$ without the floor ($K = 200$), benchmark-balanced
$-2.0$, largest leave-one-benchmark-out estimate $-0.4$, no budget worse than REF by two SE$_h$, and the local guard.
W15m16 failed only (b) ($z = -1.26$), L15 and W20 at $m_0 = -1.5$ failed (a), (b) and (d), and MXs failed (a) and (b).
W20's level block replaces REF's in the final model. Including connectors, W20's all-pairs effect is $+0.4 \pm 1.2$, so the switch
rests on our own matched runs. On the fresh pairs, the only confirmatory estimate, W20 is $-6.0 \pm 4.0$ against REF
($z = -1.49$, $K = 33$). Every finalist's fresh-pairs contrast is negative, W15m16's the largest ($-10.0 \pm 4.3$,
$z = -2.34$, $K = 23$), and by the pre-registered rules none of them selects an arm. Read on all four scored runs of
the W20 composite and its insurance build, the composite guard of rule (b) gave $-0.7 \pm 2.9$ against plain W20 on
the selection set ($K = 25$) with no composite run ending without scores, which keeps the composite in the final build
(Section 10).

# 8 Cross-system item pooling

On shared pairs the best-covered public arm scored better than each of our arms at B3 to B31 by about 3 to 9 (each about
one SE). Both connector teams give each item its own effect with SD 2 to 3 logits, against our $\sigma_i = 0.6$, but a
larger $\sigma_i$ loses on every pair, because in the joint Laplace fit the same variance also lets a pair's own labels
explain away its evidence and moderates every unlabelled item, while it pays only on rows a partner labelled (Appendix
F). So pooling runs at a fifth of item-level strength: after a partner's success our prediction moves by $+0.20$
logit at $n = 31$, against $+1.03$ for an exact item-level model (Table F1), while $\sigma_i = 2$ loses $+2.0$ to
$+3.5$ per budget from B3.

`CavityPoolModel` changes a prediction only if another system labelled the target's item text: it removes those labels
from the Laplace posterior by a Woodbury downdate (which also removes the double counting of a partner's label in its
level), integrates a shared item effect $u \sim N(0, \sigma_x^2)$ over the labellers' cavity levels and adds the
resulting logit shift to the base prediction (Appendix F). In its pre-registered benchmark-keyed replay ($k = 2$, 20
draws per benchmark; Table C4) it gains $-0.52 \pm 0.13$, $-0.58 \pm 0.16$ and $-0.60 \pm 0.18$ at $\sigma_x = 2.0$,
2.5 and 3.0, with every benchmark negative, and its rule selects 3.0. Simpler designs tie in ALC (two-scale pooling
$-0.47 \pm 0.12$, plug-in MAP offsets $-0.54 \pm 0.24$), so the case for the cavity version is its calibration after a
partner's label (Table F1). The projected live gain is at most about $-0.5$ at formative density and grows with pairs
per benchmark: in denser benchmark-keyed draws (`results/combo_dense.csv`) C alone gains $-2.64 \pm 0.48$ and
$-5.44 \pm 0.54$ on ResearchCodeBench at $k = 4$ and 8, with no cell from B1 to B31 worse than $+0.07$ (Appendix C
gives the composite's cells and the B0 failure of that replay's rule, which belongs to S, since C is exactly zero at
B0).

# 9 Negative results

Table 5 lists the ideas with replay or live evidence that are not in the final model. Section 4.6 covers $\sigma_g$,
$\lambda$, SY1.0 and NP, and Appendix G gives all twenty, graded A (paired replay or live contrast with an SE), B
(exact calculation or simulation) or C (reasoning only).

**Table 5.** Negative and neutral results with replay or live evidence (live values: Table D1).

| Idea | Grade | Key result | Decision |
|-------------------------|------|------------------------------------------------------------|-------------------|
| Shared level ($\tau_M$) | A | replay at most +0.6 (no SE); live +3.1 to +24.8 (connectors), -5.0 to +11.8 (ours) | dropped |
| Student-t(3) level prior | A, B | replay -0.41 ± 0.07, B0 only; live +5.9 ± 5.6 (connectors), +1.0 ± 7.1 (ours) | dropped |
| Zero-inflated mixture | A, B | replay -1.79 ± 0.60, MathArena; live +5.6 ± 4.4 (connectors), -4.6 ± 5.6 (ours) | dropped |
| Narrow pair priors | A, B | d06 -0.33 ± 0.30, taus × 0.8 -0.28 ± 0.26; worse from B3 | dropped |
| Per-budget recalibration | A | cross-fitted Brier B0 +16.3, B1 +17.1, B3 +8.6 (no SE) | not used |
| Exact inference | A, B | +1.3 to +2.5 per alone pair per benchmark; -0.6 to +0.7 by calculation | not used |
| $\sigma_i = 2$; plug-in offsets | A | +1.15 ± 0.70; offsets tie the cavity in ALC but overshoot | cavity pooling |
| Likelihood tempering [15] | B, A | synthetic regret lower at $\alpha$ 1.25 to 1.5; replay at 1.3 +0.47 (n.s.) | superseded by W20 |
| Acquisition policies [19] | B | synthetic -0.65 ± 0.16 with informative item features, +0.03 without; about -0.1 expected here | default kept |
| Late-budget de-moderation | A | bias confirmed (+0.05 at B31 on near-zero pairs); plug-in -0.09 ± 0.04 against a pre-registered -0.15 | not adopted |

Several ideas gain locally only on MathArena at B0 and B1, where REF's level is centred too low for an easy benchmark.
The two of them tested live, the mixture and the Student-t prior, showed no resolvable live gain.

# 10 Final submission

**Table 6.** Final submission (local values on the draws of Table 3).

| Item | Value |
|----------------------|--------------------------------------------------------------|
| Configuration | level block W20 ($m_0 = -2.0$, $\tau_\mu = 2.0$, $\tau_\delta = 1.0$) + S + C ($\sigma_x = 3.0$) + P, shipped as the insurance build `v6_combo_w20_c1.zip` (SHA-256 prefix e15a4e3f); selected Codabench submission 1020301 |
| Local macro ALC | 0.1910, 0.1888, 0.1880 with the REF, W20, L15 blocks (builds verified bit-exact) |
| Live effect vs REF: all, holdout, fresh pairs | level block: $-1.7 \pm 1.0$ ($K = 200$), $-1.5 \pm 2.4$ ($K = 38$), $-6.0 \pm 4.0$ ($K = 33$) (fresh pairs: first seen after the snapshot of 2026-10-06 13:51 UTC; the only confirmatory estimate, Appendix D); composite against plain W20 on the selection set: $-0.7 \pm 2.9$ ($K = 25$) |
| Formative mean (draw CI); platform ECE | 0.186 over the four Finished runs of the build and its parent (95% $t$ interval over draws 0.165 to 0.208; the selected run 0.190 on 9 pairs; the mean is optimistic: selected among arms); platform ECE 0.117 to 0.142 on those runs (0.121 for the selected run); locally the platform's ECE is 0.135 to 0.137 with the W20 or REF block, against a noise floor of 0.048 (Appendix C) |
| Platform; runtime; uploads | composites Finished on the W20 (2026-10-07), REF and W20 at $m_0 = -1.5$ blocks (2026-10-19); the W15m16 composite's check and its rerun died before scoring and its insurance build Finished three times; the shipped build Finished three times (1006743 on 2026-10-21, 1020286 and 1020301 on 2026-10-30; a fourth run was still in flight at the final look); locally at most about 30 s per draw and worker (REF 18 s; Appendix A); 575 uploads (467 by 2026-10-20) |

**Composite guard and insurance builds.** A readiness amendment (2026-10-16) made precise when composite runs trip
final-selection rule (b), which then ships the plain level-block build, because platform failures alone would trip its
literal reading with probability about 0.5; it moved composite platform checks forward and added
`tools/composite_guard.py` to read the rule (Appendix D). As the summative draw size is unknown, the W20 and W15m16
composites also have insurance builds that only add `prediction_config.json` with `{"concurrency": 1}` (one worker
instead of up to four) and predict identically; the winner's ships in its place if it has a Finished run. W20's insurance
build `v6_combo_w20_c1.zip` has three and is the final build. Rule (a) names the most recent of them, 1020301, in the
selection e-mail. The script `swarm/round5/release/final_check.py` confirmed its status, upload time, SHA-256 and bit-exact
predictions.

# 11 Limitations

The level of hidden benchmarks is not identified by four public folds, and local and live evidence disagree on B0, so
the level block comes from formative feedback, which may differ from the summative subset. We do not know how many pairs
per benchmark that subset holds, and within the composite W20's local gain over REF shrinks from
$-3.31 \pm 1.13$ with one pair per benchmark to $-1.08 \pm 0.69$ and $-0.76 \pm 0.45$ with two and four (Appendix C).
Live power is weak, 31 configurations were uploaded (28 reported), and deterministic pairs keep favouring
the decisions they informed. The holdout, the thresholds and the confirmatory estimate on fresh pairs, defined before
the 2026-10-07 uploads (Appendix D), are the guard; every other live number in this report is exploratory. We missed
nine upload windows while the upload session was down, and the final look ran before its pre-registered window, so the
finalists have fewer runs than planned. The regime
test models alone pairs only. Tuned settings compensate for the Laplace under-reaction. The composite is decided
locally. Its largest component, S, acts mainly on MathArena, whose tables have the defects of Section 3 (the composite's
gain there survives their removal), and the composite failed, through S, a dense-draw robustness rule at B0 on
ResearchCodeBench, which we report and overrode (Section 4.6). An LLM coding assistant produced most of the code and
analysis (Section 13). Tests, bit-exact checks and script-generated tables reduce the risk of errors but cannot remove
it.

# 12 Reproducibility

The repository (https://github.com/rishabhsinha17/brier-rabbit-paiec, MIT license) contains the predictor, the protocol replica, the replay and analysis
scripts with their saved outputs, the build and verification tools, and 26 unit tests (`../.venv/bin/python -m pytest -q
tests`). The environment is Python 3.11 with numpy, scipy, pandas, pyarrow, scikit-learn and matplotlib; the submission
needs only numpy. The data fetcher pins revision `bc8204d8` (Hugging Face access to the gated dataset is required); it
and the synthetic-study scripts live outside `brier_rabbit/` in our tree and move into the repository at release. Every
queued replay used `experiments/paired.py` with the arguments recorded in its log, and quick checks used the scripts in
Appendix B. The release is built from an allow-list: `paiec/`, `experiments/`, `tools/` without feedback dumps,
`tests/`, `report/`, analysis scripts whose outputs use public data, and the arm-level live tables of this report. It
excludes our per-pair formative logs and every file derived from them. Other teams' logs are referenced by repository
and commit and are not redistributed. We do not release our own formative logs, which list
the hidden pool's anonymous IDs.

# 13 Disclosures and acknowledgments

**LLM assistance.** The author used Claude Code (Anthropic), an LLM coding
assistant, throughout the project: to write the model, the protocol replica and the analysis tools, to run and summarize
experiments, to search the literature, and to draft this report. Several assistant sessions worked in parallel as
specialized reviewers, including one that re-derived, fuzz-tested and checked the purity of every model variant. Every
number in the main text is traceable to a script or saved output listed in Appendix B, each citation was checked against
arXiv, Semantic Scholar, Crossref or the publisher's page, and the submitted model is verified to reproduce the research
code bit for bit. The author reviewed the code and the text and takes full responsibility for both.

**Formative feedback.** Used as described in Section 6: global settings only, through arm-level comparisons and global
summaries, with anonymous IDs used only as opaque keys. This report publishes neither per-pair results nor per-benchmark
summaries, and we did not attempt to identify hidden benchmarks or subjects. Number of uploads: 575 of the
1,000 allowed per person (467 by 2026-10-20), not counting the 529 submissions the platform created in our account on
2026-10-17 (Section 6).

**Training data and models.** Public measurement-db revision `bc8204d8` only (the six-benchmark release of
2026-09-18); the expanded collection published on 2026-10-06 (Section 3) was not used for any shipped or reported
result. The submission is self-contained numpy
code without pretrained models, embeddings, external data or inference APIs. Compute was one laptop CPU, without a GPU.

**Other teams' public material.** We thank the GitHub users aghasalim (`github.com/aghasalim/predictive-eval`) and
doctawho42 (`github.com/doctawho42/PAIEC`) for publishing their code and formative logs under the MIT license. We used
their logs only as connector arms in arm-level contrasts, and their code to build validated replicas of their predictors
for the regime test and to check their acquisition setting. We also thank lucaluo925 for MIT-licensed public results we
cite in Appendices D and G.

**Community and data contributions.** We release the code, the protocol replica with its unit tests, the replay
outputs on public data and the data-quality checks of Appendix C under the MIT license (https://github.com/rishabhsinha17/brier-rabbit-paiec).

# References

1. Truong, Tu, Liang, Li, Koyejo. Reliable and Efficient Amortized Model-based Evaluation. ICML 2025. https://proceedings.mlr.press/v267/truong25c.html
2. Truong, Goodman, Brunskill, Domingue, Haber, Koyejo. A Measurement Science Roadmap: From Human Assessment to AI Evaluation. Preprint, 2026. https://ai.stanford.edu/~sttruong/assets/pdf/roadmap.pdf
3. AIMS Foundations. Measurement Data Bank (measurement-db). https://huggingface.co/datasets/aims-foundations/measurement-db and https://github.com/aims-foundations/measurement-db
4. MacKay. The Evidence Framework Applied to Classification Networks. Neural Computation, 1992. doi:10.1162/neco.1992.4.5.720
5. Kuss, Rasmussen. Assessing Approximate Inference for Binary Gaussian Process Classification. JMLR 6, 2005. https://www.jmlr.org/papers/v6/kuss05a.html
6. Nickisch, Rasmussen. Approximations for Binary Gaussian Process Classification. JMLR 9, 2008. https://www.jmlr.org/papers/v9/nickisch08a.html
7. Breslow, Lin. Bias Correction in Generalised Linear Mixed Models with a Single Component of Dispersion. Biometrika, 1995. doi:10.1093/biomet/82.1.81
8. Efron, Morris. Data Analysis Using Stein's Estimator and Its Generalizations. JASA, 1975. doi:10.1080/01621459.1975.10479864
9. Hardy, Azam, Reuel, Kochenderfer, Koyejo. Agent Evaluation Reliability: More Tasks Won't (Always) Fix An Agent Leaderboard. arXiv:2610.00651, 2026. https://arxiv.org/abs/2610.00651
10. Shi et al. Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation. arXiv:2609.04298, 2026. https://arxiv.org/abs/2609.04298
11. Gelman. Prior Distributions for Variance Parameters in Hierarchical Models. Bayesian Analysis, 2006. doi:10.1214/06-ba117a
12. Müller, Hollmann, Pineda Arango, Grabocka, Hutter. Transformers Can Do Bayesian Inference. ICLR 2022. https://arxiv.org/abs/2112.10510
13. Raventós, Paul, Chen, Ganguli. Pretraining Task Diversity and the Emergence of Non-Bayesian In-Context Learning for Regression. NeurIPS 2023. https://arxiv.org/abs/2306.15063
14. Yao, Vehtari, Simpson, Gelman. Using Stacking to Average Bayesian Predictive Distributions. Bayesian Analysis, 2018. https://arxiv.org/abs/1704.02030
15. Grünwald, van Ommen. Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It. Bayesian Analysis, 2017. https://arxiv.org/abs/1412.3730
16. Ge, Kryvosheieva, Fried, Girit, Hariharan. Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks. arXiv:2604.00594, 2026. https://arxiv.org/abs/2604.00594
17. Pacchiardi, Cheke, Hernández-Orallo. 100 Instances Is All You Need: Predicting the Success of a New LLM on Unseen Data by Testing on a Few Instances. arXiv:2409.03563, 2024. https://arxiv.org/abs/2409.03563
18. Zhang, Dorner, Hardt. How Benchmark Prediction from Fewer Data Misses the Mark. arXiv:2506.07673, 2025. https://arxiv.org/abs/2506.07673
19. Kossen, Farquhar, Gal, Rainforth. Active Testing: Sample-Efficient Model Evaluation. ICML 2021. https://arxiv.org/abs/2103.05331
20. Polo, Weber, Choshen, Sun, Xu, Yurochkin. tinyBenchmarks: Evaluating LLMs with Fewer Examples. ICML 2024. https://arxiv.org/abs/2402.14992
21. Jiang, Kwon, Luo, Xiao, Zhang. Can We Trust Item Response Theory for AI Evaluation? arXiv:2607.15190, 2026. https://arxiv.org/abs/2607.15190
22. Jamieson, Talwalkar. Non-stochastic Best Arm Identification and Hyperparameter Optimization. AISTATS 2016. https://arxiv.org/abs/1502.07943
23. Dunnett. A Multiple Comparison Procedure for Comparing Several Treatments with a Control. JASA, 1955. doi:10.1080/01621459.1955.10501294
24. Miller. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640
25. Brier. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 1950. doi:10.1175/1520-0493(1950)078<0001:vofeit>2.0.co;2
26. Gneiting, Raftery. Strictly Proper Scoring Rules, Prediction, and Estimation. JASA, 2007. doi:10.1198/016214506000001437
27. De Boeck, Wilson (eds.). Explanatory Item Response Models. Springer, 2004. doi:10.1007/978-1-4757-3990-9
28. Balunović et al. MathArena: Evaluating LLMs on Uncontaminated Math Competitions. arXiv:2505.23281, 2025. https://arxiv.org/abs/2505.23281
29. Zan et al. Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving. arXiv:2504.02605, 2025. https://arxiv.org/abs/2504.02605
30. Garg et al. REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites. arXiv:2504.11543, 2025. https://arxiv.org/abs/2504.11543
31. Hua et al. ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code. arXiv:2506.02314, 2025. https://arxiv.org/abs/2506.02314
32. Badertdinov et al. SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. arXiv:2505.20411, 2025. https://arxiv.org/abs/2505.20411
33. Dong et al. Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering (MMDocRAG). arXiv:2505.16470, 2025. https://arxiv.org/abs/2505.16470
34. Salaudeen, Zhang, Alhamoud, Beery, Ghassemi. Aggregation Hides Out-of-Distribution Generalization Failures from Spurious Correlations. arXiv:2510.24884, 2025. https://arxiv.org/abs/2510.24884
35. Brown. In-Season Prediction of Batting Averages: A Field Test of Empirical Bayes and Bayes Methodologies. Annals of Applied Statistics, 2008. https://arxiv.org/abs/0803.3697
36. Cameron, Miller. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources 50(2):317-372, 2015. doi:10.3368/jhr.50.2.317
37. AIMS Foundations. Instructions for Item-level Measurement Data Curation (competition eligibility of base models). Organizers' guide, 2026, accessed 2026-10-04. Linked from the data bank page, <https://aimslab.stanford.edu/measurement-db> (document: <https://docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU>)
38. Murphy. A New Vector Partition of the Probability Score. Journal of Applied Meteorology 12(4):595-600, 1973. doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
39. Stephenson, Coelho, Jolliffe. Two Extra Components in the Brier Score Decomposition. Weather and Forecasting 23(4):752-757, 2008. doi:10.1175/2007WAF2006116.1

# Appendix A. Configurations and engineering

**Table A1.** Hyperparameters of REF and of the arms discussed. Empty cells equal REF. All arms share the subject prior
of Section 4.2, the default acquisition rule and $\tau_\theta = 0.5$, the ability SD used only when neither a name nor
an attribute prediction exists.

| Arm | $m_0$ | $\tau_\mu$ | $\tau_\delta$ | $\tau_\phi$ | $\lambda$ | $\sigma_i$ | $\sigma_g$ | Other | Solver |
|----------------|-------|-------|------|------|-----|-----|-----|-----------------------|----------|
| REF (incumbent) | -2.0 | 1.2 | 1.0 | 0.5 | 1.0 | 0.6 | 1.0 | normal level prior, shared level and mixture off | damped |
| L15 | -1.5 | | | | | | | | damped |
| W20 | | 2.0 | | | | | | | damped |
| Round-4 grid arms | -2.5 to -1.0 | 1.2 to 2.5 | 0.6 to 1.4 | | | | | 84 arms; the five run live follow (Appendix D) | damped |
| W20 at $m_0 = -1.5$ | -1.5 | 2.0 | | | | | | grid arm from 2026-10-07 | damped |
| WLA | -2.5 | 2.5 | 0.6 | | | | | grid arm from 2026-10-07 | damped |
| WLB | -2.25 | 2.5 | 1.4 | | | | | grid arm from 2026-10-07 | damped |
| W15d6 | -1.5 | 2.0 | 0.6 | | | | | grid arm from 2026-10-08 | damped |
| W15m16 | -1.5 | 1.6 | 1.4 | | | | | grid arm from 2026-10-10 | damped |
| d06 | -1.693 | | 0.6 | 0.3 | | | | $m_0$ solved for a budget-0 prediction of 0.25 for the reference subject with one item group | damped |
| Student-t | -1.5 | | | | | | | $\nu = 3$, variance-matched | damped |
| Shared level (Min / Mtop) | -1.0 to -2.0 | 0.894 / 1.2 | | | | | | $\tau_M = 0.8$ | undamped |
| Mixtures | -1.0 | | | | | | | $\pi$ 0.25 or 0.35, hard component $N(-4, 0.7^2)$ | damped |
| MXs (exploration arm, 2026-10-07) | -0.5 | 1.07 | 0.33 | | | | | $\pi$ 0.3, hard component $N(-2.9, 0.55^2)$ | damped |
| Composite (any level block) | | | | | | | | S: attribute scale 1.56, year clamp 2026.4; C: $\sigma_x = 3.0$; P: variance shares 0.45 (name), 0.288 (harness) | damped |

**Sources of the values.** $\tau_\mu$, $\lambda$, $\tau_\delta$, $\sigma_i$ and $\sigma_g$ come from the v1 coordinate
search (undamped solver, $m_0 = -0.5$, no subject prior; E2), which also swept $\tau_\phi$ and $\tau_\theta$ and kept
their defaults of 0.5. $m_0 = -2.0$ was chosen live (Section 7.1). Two later coordinate searches (around $m_0 = -0.5$
with $\tau_\mu = 0.8$, and around $m_0 = -1.2$, $\lambda = 1.6$ with a Student-t level prior) were not adopted. The
composite values come from their own rules (Section 4.6). With one item group, REF's budget-0 prediction for the
reference subject is 0.228 instead of 0.211.

**Engineering details.** For REF a call takes 0.42 ms of CPU amortized over a replay and a refit with 300 labels 66 to
75 ms; a formative draw (at most about 370 labels, about 4,000 calls) takes roughly 10 s per worker at an average of
about 2.5 ms per call. At a synthetic final checkpoint (300 labels, ten pairs, 100 KB item texts) the composite takes 98
ms for the refit and 7.5 ms per call amortized, against 70 ms and 4.4 ms for REF. Charging every call of a draw at this
final-checkpoint rate bounds a formative draw at about 30 s per worker for the composite and 18 s for REF, against the
8-hour limit.
On an internal error a prediction falls back to the base model, then to $\sigma(m_0)$. A build tool generates `model.py`
from the research code and vendors the plug-in classes; the verification runs the shipped file with the repository off
the import path and `paiec` imports blocked, and the composite builds also pass a check of every component path (maximum
difference 0.0 over 9,468 calls). A fuzz test of 7,650 malformed calls over 15 configurations, and 3,060 more for the
composite, returned a finite probability inside (0, 1) every time, without exceptions or mutated arguments, and the
composite's predictions are identical under fresh instances per checkpoint, reversed call order and random interleaving.

# Appendix B. Sources of each number

Paths are relative to the repository root. Each replay's exact arguments are in the log next to its output. Rows marked "not released" read our per-pair formative logs or files derived from them, which the release excludes (Section 12); the scripts of those rows are released where they read only public data.

| Item | Script | Inputs |
|----------------|------------------------------------------------------------|--------------------|
| Table 1, Section 3 | `swarm/round3/report_writer_sims/data_facts.py`; data-quality counts `swarm/round4/contrib_drafts/measurement_db_checks.py` | measurement-db `bc8204d8` |
| Tables 2, 3, C2 ($\sigma_g$ rows), C4, C5; local numbers of Sections 4.6, 5, 7.3, 8, 9 and Appendix C | `swarm/round5/writer_sims/paper_numbers_v3.py`, `paper_tables_v3.py` (outputs `.out`) | `results/combo_v1.csv`, `combo_sy.csv`, `combo_np.csv`, `pdpool_bench_k2.csv`, `builder_pair_twin_k2.csv`, `adj_wide.csv`, `pdadapt_sigma_g.csv`, `pdadapt_simpool.csv` |
| Table C6 | `swarm/round3/shift_sims/temporal_replay.py` (log `results_shift_temporal_replay.log`) | public data |
| Dense draws (Section 8), dense-draw record (Section 4.6, Appendix C), P's changed pair-draws (Section 4.6), guards of the 2026-10-07 grid arms (Appendix D), MathArena without the data defects (Section 3, Appendix C), composition stress test (Section 11, Appendix C) | `swarm/round7/writer_sims/paper_numbers_v3_1.py` (output `.out`); the dense rule's text is in `compute_queue/done/combo_dense.json`, the cleaning job's reading in `compute_queue/done/contrib_ma_clean.json` (their `why` fields, which the release's job listing omits; the dense rule is quoted in Section 4.6 and Appendix C, the cleaning job's reading in Appendix C), the stress test's rules in `swarm/round4/frontier_watch.md` (section 5) | `results/combo_dense.csv`, `combo_v1.csv`, `ld_guard_1007.csv`, `adj_wide.csv`, `contrib_ma_clean.csv`, `fw_kstress.csv` |
| Reweighted guards of the 2026-10-07 grid arms (Appendix D) | `swarm/round7/writer_sims/writer_B_guard_rw.py` (calls `tools/daily_report.py`; output `.out`) | `results/adj_wide.csv`, `ld_guard_1007.csv`; global summaries of our logs (not released: derived from our formative logs) |
| Calibration (Table C7, Figure C2, Table 6) | `swarm/round6/cal_sims/replay_preds.py`, `analyze.py` (outputs `out_bench/`, `out_pair/`); table rounding `swarm/round7/writer_sims/writer_B_cal_table.py` | `results/cal_preds_bench.parquet`, `cal_preds_pair.parquet` (public data) |
| Figure C1 | `swarm/round5/writer_sims/fig_ladder_v3.py` | `results/combo_v1.csv` |
| Tables C1, C2 | `swarm/round3/report_writer_sims/paper_tables.py`, `report/make_tables.py` | `results/paired*.csv`, `results/pd1_priors.csv`, `results/headA_itempool.csv` |
| Figure 1, Section 4.4 | `swarm/round3/report_writer_sims/paper_figures.py`, `swarm/round1/head_researcher_B_sims/check_laplace.py`, `effective_alpha.py` | real model class, synthetic inputs |
| Sections 3 and 4.2 (Rasch fit, group variance, $\theta$ correlation) | `swarm/round1/head_researcher_A.md` and its sims; `swarm/round1/postdoc_adaptation_sims/item_signal.py` | public data |
| Section 4.2 attribute list | `paiec/subjects.py` (feature vector of the attribute ridge) | public data |
| Section 4.3 plug-in cost | synthetic study, `research/a6_literature/NOTES.md` (C1) and `research/a6_literature/sim/` (outside `brier_rabbit/`; moves into the repository at release) | synthetic |
| Section 4.6: S scale 1.56 to 1.60, year clamp 2026.4, name-path slope 0.65; residual correlations +0.49 and +0.82 | `swarm/round1/postdoc_priors.md` (item 5 and its sims); `swarm/round1/postdoc_adaptation.md` (item 4 and its sims) | public data |
| Section 7.1, five matched pairs ($-3.7$) | `submissions/cur_final.txt` (decision note of submission 963675) | our formative logs (not released: our upload records) |
| Section 7.3, predictions after one success (0.53, 0.42) and at budget 0 (0.21, 0.25) | `swarm/round2/adjudicator_level.md` section 1.2; `swarm/round3/report_writer_sims/paper_figures.out` | real model class |
| Composite identities, timing, fuzz | `swarm/round3/composer_sims/identity.py`, `verify_paths.py`, `robust.py`, `timing.py` | recorded calls |
| Determinism counts | `swarm/round3/report_writer_sims/determinism_check.py` (12:10 UTC output: `swarm/round5/writer_sims/determinism_v3.out`); counts of 2026-10-20 from the drift line of `tools/daily_report.py` (`swarm/round11/daily_report_1020.txt`) | formative logs (not released: derived from our formative logs) |
| REF SD across draws (Section 1, Appendix D); upload and configuration counts (Sections 6, 10, 11, 13, Table D2) | `submissions/ledger.csv` (arm `v3b_noM_n_m20`, column `alc`, scored runs; one row per upload) | our own draw-level scores (not released: our upload records) |
| Section 6 and Appendix D | `swarm/round2/adjudicator_sims/audit_evalsci.py`, `tools/swarm_evalsci.py power`, `swarm/round2/builder_infra_sims/public_arms_se.py`, `public_sensitivity.py`, `public_split_check.py`, `k_eff.py`, `swarm/round2/adjudicator_sims/validate_calc.py` | formative logs, public code (not released: derived from our formative logs) |
| Table D1 | `swarm/round2/adjudicator_sims/live_evidence.py` (with connectors); `tools/daily_report.py --upto 963848 --public all --all-arms --reps 200 --csv` (ours only; outputs `swarm/round5/writer_sims/daily_report_upto963848.*`) | formative logs (not released: derived from our formative logs) |
| Table 4 observed, Table D6, live numbers of 2026-10-20 (Sections 2, 6, 7), local guard (Appendix C) | `tools/daily_report.py --reps 500` at 03:24 UTC on 2026-10-20 (output `swarm/round11/daily_report_1020.txt`); the 12:07 UTC values of 2026-10-06 from `--reps 300` (outputs `swarm/round5/writer_sims/daily_report_1207utc.*`) | formative logs, `results/adj_wide.csv`, `ld_guard_1010.csv`, `hr_mix_guard.csv` (not released: derived from our formative logs) |
| Section 7.2, Appendix E | `swarm/round2/adjudicator_sims/regime_test3.py`, `regime_variants.py`, `spread_test.py`, `regime_dtheta.py`, `regime_subject.py`; rerun on the pre-registered connector set `swarm/round10/regime_rerun/rerun_prereg.py` (outputs beside it: `summary_prereg.json`, `loao_prereg.csv`, `pred_vs_obs_best_prereg.csv`) | formative logs, public data (not released: derived from our formative logs) |
| Table 4 predictions | `swarm/round2/adjudicator_sims/recommend.py`, `recommend2.py` | exact calculation |
| L15 test operating characteristics (Section 7.3, Appendix D) | `swarm/round6/oc_sims/oc_calib.py` (caches `cache/calib_stylised.jsonl`, `calib_posterior.jsonl`; summary in `swarm/round6/operating_characteristics.md` section 2.4) | arm-level summaries of our logs, design posterior (not released: derived from our formative logs) |
| Regime-test bootstrap (Section 7.2, Table E2) | `swarm/round6/oc_sims/regime_boot.py` (`boot 2000` output in `cache/plan.log`; `scaled 300` output `swarm/round7/writer_sims/regime_boot_scaled300.out`) | arm-level contrasts of Section 7.2 (`swarm/round2/adjudicator_sims/draws_all.npz`), public data |
| Operating characteristics (Section 7.4, Appendix D, Table D5) | `swarm/round6/oc_sims/` (`run_oc.py`, `oc_pipeline.py`, `oc_core.py`; tables `oc_tables.py`, `oc_diag.py`, `oc_extra.py`, outputs `cache/final_*.txt`; report `swarm/round6/operating_characteristics.md`) | arm-level and global summaries of our logs (`live_targets.json`), design posterior, public data (not released: derived from our formative logs) |
| Section 7.4, Appendix D | `tools/swarm_live_design.py`, `swarm/round4/design_sims/` (`run_sim.py`, `agg_sim.py`; plan `cache/plan_2026-10-07.txt`); daily refits `swarm/round11/design_plan_2026-10-09.txt` to `design_plan_2026-10-21.txt` (the last at 03:26 UTC on 2026-10-20) | formative logs, public data (not released: derived from our formative logs) |
| Final-look rehearsal (Section 7.4, Appendix D) | `tools/daily_report.py --final-look --reps 300` on the data of 2026-10-16 (output `swarm/round11/finallook_rehearsal_1016.txt`) | formative logs (not released: derived from our formative logs) |
| Composite additivity guards (Appendix D) | `swarm/round11/scp_add_read.py` (outputs `swarm/round11/scp_add_1008_reading.txt`, `scp_add_1015_reading.txt`) | `results/scp_add_1008.csv`, `scp_add_1015.csv` (public data) |
| Composite readiness, guard tool, insurance builds, platform-created runs (Sections 6 and 10, Appendix D) | `submissions/QUEUE.md` (notes of 2026-10-16 to 10-20), `tools/composite_guard.py`, `tools/verify_shipped.py`, `submissions/builds.csv`, `submissions/platform_runs_2026-10-17.md` | our upload records (not released) |
| Headroom calculus, MXs and wide-arm guards, late-budget de-moderation (Section 7.4, Table 5, Table A1, Appendix D, Appendix G item 20) | `swarm/round8/headroom_sims/` (scripts, `params_mxs.json`; summary `swarm/round8/headroom.md`), `paiec/ext/rt8_unseen.py`; job files `compute_queue/done/hr_mix_guard.json`, `r8_dlike_guard.json`, `rt8_unseen.json` | `results/hr_mix_guard.csv`, `r8_dlike_guard.csv`, `rt8_unseen.csv` (public data); the design posterior refit (not released: derived from our formative logs) |
| Table F1, Section 8, Appendix F | `swarm/round2/pooling_sims/why_underpool.py`, `unit_checks.py`, `shift_check.py`, `check_cavity.py`, `replay.py`, `pool_tables.py`, `diag_covered.py`, `scale_to_platform.py` | public data |
| Table 5, Appendix G | `experiments/l2o_shared_level.py`, `experiments/calibration_check.py`, `swarm/round2/skeptic_plugins_sims/quick_alc.py`, `two_pair_narrow.py`, `swarm/round1/postdoc_adaptation_sims/two_pair_exact.py`, `stacking_headroom.py`, `item_signal.py`, `test_bcm.py`, `swarm/round1/head_researcher_B_sims/eb_sims.py`, `time_forward.py`, synthetic study (moves into the repository at release) | public data, synthetic |
| Section 4.5, Appendix A | `tools/verify_shipped.py`, `tools/build_submission.py`, `swarm/round2/skeptic_plugins_sims/fuzz.py`, `replay_checks.py`, `timing_refit.py` | built submissions |
| Pre-registered rules | `submissions/QUEUE.md`; hashed snapshots in `submissions/prereg/` (`LOG.csv`) | |
| Final look, composite guard, selected run (abstract, Sections 7.4 and 10, Table 6, Appendix D) | `tools/daily_report.py --final-look --finalists 5 --reps 500` at 04:15 UTC on 2026-10-30 (output `swarm/round11/FINAL_LOOK_2026-10-30.txt`); `tools/composite_guard.py v6_combo_w20_c1 v5_w20_m20 --pool v6_combo_w20` (outputs `swarm/round11/COMPOSITE_GUARD_2026-10-30_pooled.txt`, `_c1.txt`, `_parent.txt`); `swarm/round5/release/final_check.py 1020301 --verify` (output `swarm/round11/FINAL_CHECK_1020301.txt`); `submissions/ledger.csv` (formative mean, platform ECE, upload counts) | formative logs, our upload records (not released: derived from our formative logs) |

# Appendix C. Additional local comparisons

**Table C1.** The pair-keyed ladder of earlier drafts (benchmark-equal macro, draw-clustered paired SE). Draw sets: A
has 30 draws per benchmark and $k$, B has 25. It holds the solver and level comparisons that Table 3 does not.

| Model | Set | Macro ALC | $\Delta$ALC (SE) | vs | $\Delta$B0 | $\Delta$B1 | $\Delta$B3 | $\Delta$B7 | $\Delta$B15 | $\Delta$B31 |
|------------------------------|----|-------|---------|------------|------|------|------|------|------|------|
| Smoothed mean ($\kappa = 2$) | A | 0.2067 | -43.3 (4.1) | constant 0.5 | +13.0 | -18.7 | -43.2 | -58.9 | -67.1 | -70.0 |
| Hierarchical, untuned ($m_0 = -0.5$) | A | 0.1992 | -7.6 (1.0) | smoothed | -31.0 | -13.8 | -4.3 | -0.1 | -1.5 | -5.0 |
| Hierarchical, tuned | A | 0.1975 | -1.7 (0.4) | untuned | -0.4 | +0.0 | -1.1 | -2.3 | -3.1 | -3.6 |
| + subject prior | A | 0.1947 | -2.8 (0.4) | tuned | -9.9 | -5.3 | -2.2 | -1.0 | -0.4 | -0.1 |
| Smoothed mean ($\kappa = 2$) | B | 0.2047 | | | | | | | | |
| + subject prior | B | 0.1928 | | | | | | | | |
| $m_0 = -1.5$ (undamped) | B | 0.1922 | -0.6 (1.0) | + subject prior | +4.7 | -1.4 | -2.6 | -2.0 | -0.5 | +2.6 |
| + damped Newton, $m_0 = -1.5$ | B | 0.1918 | -0.4 (0.2) | undamped | 0.0 | 0.0 | 0.0 | 0.0 | -0.6 | -3.2 |
| same, against the smoothed mean | B | | -12.9 (1.2) | smoothed | -37.5 | -21.6 | -9.8 | -4.7 | -5.4 | -8.8 |
| REF: damped, $m_0 = -2.0$ | B | 0.1942 | +2.4 (0.5) | $m_0 = -1.5$ | +13.2 | +5.0 | +1.0 | -0.2 | -0.3 | -0.2 |
| REF, against the smoothed mean | B | | -10.5 (1.5) | smoothed | -24.3 | -16.6 | -8.7 | -4.9 | -5.7 | -9.0 |
| damped, $m_0 = -1.0$ | B | 0.1911 | -0.6 (0.5) | $m_0 = -1.5$ | -6.9 | -1.4 | +0.5 | +0.7 | +0.4 | +0.3 |

On set B, moving $m_0$ from $-0.5$ to $-1.5$ changes MathArena by $+12.7$ and Multi-SWE-bench by $-10.6$.

![Figure C1. Paired Brier difference against the smoothed mean by budget, with 95% confidence intervals (draw-clustered), on the draws of Table 3. Script: `swarm/round5/writer_sims/fig_ladder_v3.py`.](../swarm/round5/writer_sims/fig_c1_ladder.png){width=80%}

**Table C2.** Pair-keyed paired replay differences (draw-clustered SE) with per-benchmark means. Draw set C has 20 draws
per benchmark and $k$, and "base" is REF on those draws (macro 0.1921). On set C the subject-prior model at $m_0 = -0.5$
scores 0.1908 ($-1.3 \pm 1.6$ against REF, with MathArena $-21.7$ and Multi-SWE-bench $+12.0$), another view of the
level problem. MSWE = Multi-SWE-bench.

| Variant (reference) | $\Delta$ALC (SE) | $\Delta$B0 | $\Delta$B1 | $\Delta$B3 | $\Delta$B31 | MathArena | MSWE | REAL | RCB |
|-------------------------------|----------|------|------|------|------|----------|-------|------|------|
| Student-t(3), $m_0 = -2.0$ (normal, set B) | -0.41 (0.07) | -7.8 | +1.2 | +0.5 | 0.0 | -0.9 | -0.2 | 0.0 | -0.5 |
| Student-t(3), $m_0 = -1.5$ (normal, set B) | -0.15 (0.07) | -3.1 | +0.4 | +0.3 | 0.0 | -0.4 | 0.0 | +0.2 | -0.4 |
| Mixture $\pi = 0.25$, $m_0 = -1.0$ (base) | -1.79 (0.60) | -9.3 | -2.6 | -1.6 | +0.1 | -9.5 | +2.4 | -1.9 | +1.7 |
| Mixture $\pi = 0.35$, $m_0 = -1.0$ (base) | -0.35 (0.56) | -1.1 | +0.2 | -1.1 | +0.1 | -5.7 | +2.2 | -0.8 | +2.9 |
| Ability-tilted mixture (untilted $\pi = 0.25$) | -0.72 (0.28) | -4.2 | -1.5 | -0.2 | 0.0 | -2.1 | +0.3 | -2.0 | +0.9 |
| d06 (base) | -0.33 (0.30) | -6.4 | -0.9 | +0.8 | +0.3 | -0.5 | -0.1 | +0.4 | -1.1 |
| All taus × 0.8 (base) | -0.28 (0.26) | -6.5 | -0.1 | +1.1 | +0.1 | +0.3 | -0.2 | -0.1 | -1.2 |
| Ability rescale × 1.56, component S (base) | -0.86 (0.18) | -2.7 | -1.5 | -0.8 | -0.1 | -2.0 | -0.2 | -1.1 | -0.2 |
| $\lambda = 1.3$ (base) | -0.76 (0.16) | -3.0 | -1.6 | -0.6 | +0.1 | -2.1 | +0.1 | -1.1 | +0.1 |
| $\lambda = 1.6$ (base) | -1.40 (0.31) | -5.8 | -2.9 | -1.1 | +0.3 | -4.0 | +0.3 | -2.0 | +0.2 |
| All feature tokens as groups (base) | -1.04 (0.16) | -4.0 | -2.4 | -0.8 | +0.1 | -4.2 | 0.0 | 0.0 | 0.0 |
| Ability as a fitted effect, $m_0 = -1.5$ (fixed offset, set B) | -0.13 (0.11) | 0.0 | 0.0 | -0.1 | -0.3 | -0.9 | +0.3 | -0.3 | +0.4 |
| Two-scale item pooling, pair-keyed (base, $k = 2$ only) | -0.31 (0.10) | 0.0 | 0.0 | -0.1 | -1.3 | -0.4 | 0.0 | -0.5 | -0.4 |
| $\sigma_g = 1.4$ (base) | +0.48 (0.24) | -3.3 | +0.6 | +1.4 | 0.0 | +0.01 | +1.67 | +0.81 | -0.56 |
| $\sigma_g = 2.0$ (base) | +2.31 (0.57) | -8.0 | +2.5 | +4.6 | +1.4 | +1.05 | +5.04 | +3.17 | 0.00 |

The gains of the mixture and of using every feature token come almost entirely from MathArena at B0 and B1, where REF's
level prior is centred too low for an easy benchmark; those of a larger ability weight come mainly from MathArena and
REAL at B0 to B3. Using every feature token adds
moderation through extra group-variance terms rather than information. The rescaled attribute regression passed its
pre-registered rule (macro at most $-0.3$, no benchmark worse than $+0.5$) and is component S. The $\sigma_g$ rows
failed theirs (adopt only if the macro is at most $-0.5$).

**Table C3.** Partner coverage at B31, $k = 2$, 30 draws per benchmark. "Complete public groups" uses the same two pairs
without subsampling; the old pair-keyed subsample reproduces it, and the benchmark-keyed one reproduces platform-sized
groups. Script: `swarm/round2/builder_infra_sims/partner_coverage.py`.

| Benchmark | Complete public groups | Pair-keyed (old) | Benchmark-keyed (new) | Change (SE) |
|--------------------|------------|------------|--------------|------------|
| MathArena | 0.063 | 0.063 | 0.074 | +0.011 (0.006) |
| Multi-SWE-bench | 0.015 | 0.015 | 0.051 | +0.036 (0.008) |
| REAL | 0.221 | 0.221 | 0.229 | +0.008 (0.006) |
| ResearchCodeBench | 0.149 | 0.141 | 0.207 | +0.066 (0.008) |

**Table C4.** Decisive item-pooling replay (benchmark-keyed, $k = 2$, 20 draws per benchmark, random partners; against
REF, draw-clustered SE). The last four rows use the old pair-keyed subsample on identical picks. The digests of 730
draws and a small replay output are identical in the default mode before and after the change, and golden digests are
pinned in the tests. MSWE = Multi-SWE-bench.

| Arm | $\Delta$ALC (SE) | MathArena | MSWE | REAL | RCB | B0 | B1 | B3 | B7 | B15 | B31 |
|------------------|-------------|-----------|-------|------|------|-----|-----|-----|-----|-----|-----|
| Cavity, $\sigma_x = 1.5$ | -0.41 (0.10) | -0.40 | -0.31 | -0.33 | -0.59 | 0.0 | -0.1 | -0.2 | -0.4 | -0.7 | -1.4 |
| Cavity, $\sigma_x = 2.0$ | -0.52 (0.13) | -0.51 | -0.41 | -0.41 | -0.75 | 0.0 | -0.1 | -0.3 | -0.5 | -0.9 | -1.8 |
| Cavity, $\sigma_x = 2.5$ | -0.58 (0.16) | -0.57 | -0.47 | -0.43 | -0.83 | 0.0 | -0.1 | -0.3 | -0.5 | -1.0 | -1.9 |
| Cavity, $\sigma_x = 3.0$ (selected) | -0.60 (0.18) | -0.60 | -0.51 | -0.44 | -0.87 | 0.0 | -0.1 | -0.3 | -0.6 | -1.0 | -2.0 |
| Cavity, $\sigma_x = 4.0$ | -0.62 (0.20) | -0.62 | -0.55 | -0.43 | -0.89 | 0.0 | -0.1 | -0.3 | -0.6 | -1.1 | -2.0 |
| Two-scale item pooling, $\sigma_x = 2.0$ | -0.47 (0.12) | -0.41 | -0.38 | -0.39 | -0.72 | 0.0 | -0.1 | -0.2 | -0.4 | -0.9 | -1.6 |
| Plug-in MAP item offsets, SD 2.0 | -0.54 (0.24) | -0.54 | -0.55 | -0.19 | -0.86 | 0.0 | -0.1 | -0.4 | -0.5 | -0.8 | -1.8 |
| $\sigma_i = 1.0$ | -0.23 (0.17) | -0.63 | -0.32 | +0.71 | -0.68 | -2.2 | -0.4 | +0.3 | +0.3 | 0.0 | -0.3 |
| $\sigma_i = 1.5$ | +0.08 (0.43) | -0.85 | +0.02 | +2.46 | -1.29 | -5.6 | -0.7 | +1.5 | +1.4 | +0.9 | +0.3 |
| $\sigma_i = 2.0$ | +1.15 (0.70) | -0.46 | +1.39 | +5.10 | -1.43 | -8.9 | -0.5 | +3.5 | +3.4 | +2.7 | +2.0 |
| Cavity $\sigma_x = 2.0$, pair-keyed | -0.35 (0.12) | -0.46 | +0.01 | -0.49 | -0.43 | 0.0 | 0.0 | -0.1 | -0.2 | -0.6 | -1.5 |
| bench-keyed minus pair-keyed | -0.17 (0.09) | -0.05 | -0.42 | +0.09 | -0.32 | 0.0 | -0.1 | -0.2 | -0.2 | -0.3 | -0.3 |
| Cavity $\sigma_x = 3.0$, pair-keyed | -0.39 (0.16) | -0.55 | +0.02 | -0.56 | -0.49 | 0.0 | 0.0 | -0.1 | -0.3 | -0.7 | -1.7 |
| bench-keyed minus pair-keyed | -0.21 (0.11) | -0.05 | -0.53 | +0.12 | -0.37 | 0.0 | -0.1 | -0.2 | -0.3 | -0.3 | -0.3 |

**Table C5.** Subject-prior variants inside the composite, on the draws of Table 3 (against S + C + P on REF's level).
Scripts as Table 2, on `results/combo_sy.csv`, which reproduces the base, S and S + C + P rows of `combo_v1.csv`
exactly, and `results/combo_np.csv`, which reproduces its base and S + C + P rows exactly. MSWE = Multi-SWE-bench.

| Comparison | $\Delta$ALC (SE) | MathArena | MSWE | REAL | RCB | B0 | B1 | B3 | B7 | B15 | B31 |
|------------------|-------------|-----------|-------|------|------|-----|-----|-----|-----|-----|-----|
| SY1.0 in place of S | +0.53 (0.14) | +1.88 | +0.06 | +0.06 | +0.11 | +1.7 | +1.0 | +0.5 | +0.2 | +0.1 | 0.0 |
| NP ridge added | -0.12 (0.40) | -1.99 | +0.19 | +1.26 | +0.05 | -0.4 | -0.3 | 0.0 | 0.0 | -0.1 | 0.0 |

**Table C6.** Strict temporal hold-out: the subject prior is fit on systems released before $T$ and scored on newer
systems, each alone on its benchmark, against REF fit on the same older systems with $m_0$ at the older cohort's optimum
(SE clustered by pair; 34 and 21 new pairs, two seeds). Brackets: predictions pre-registered from an exact calculation.
The pre-registered rule required S at most $-2.0$ at both cut-offs and NP $\le$ S $\le$ SY1.0 $\le 0$ within one SE.
Script: `swarm/round3/shift_sims/temporal_replay.py` (log `results_shift_temporal_replay.log`).

| Subject prior | $T = 2025.5$ | $T = 2026.0$ | B0 at $T = 2025.5$ / 2026.0 |
|------------------------|------------------------|------------------------|------------------------|
| S (submitted) | -3.98 ± 0.90 [-4.19] | -3.82 ± 1.06 [-4.55] | -16.2 / -17.5 |
| S with SY1.0 | +2.66 ± 0.66 [-1.00] | +4.45 ± 0.90 [-0.42] | +10.3 / +19.4 |
| S with NP | -6.03 ± 1.01 [-5.95] | -6.10 ± 1.21 [-7.33] | -26.0 / -29.3 |

**Decision records, summary.** SY1.0 passed its job rule, which compared it with REF ($-0.71 \pm 0.19$ inside the
composite); the decision, however, is between SY1.0 and S, and SY1.0 is worse on public replays ($+0.53 \pm 0.14$) and
failed a second pre-registered rule, at most zero within one SE on the strict temporal hold-out. We kept S (hashed at
10:50 UTC) and recorded the rule conflict in the snapshot of 11:56 UTC (Table D3). A provider-free ridge (NP) cost REAL
$+1.26 \pm 0.44$, above its pre-registered cap of $+0.5$. A larger item-group variance failed its rule ($+0.48 \pm
0.24$ at $\sigma_g = 1.4$). We did not adopt $\lambda = 1.6$ ($-1.40 \pm 0.31$) because it also scales name-path
abilities, whose public slope is already 0.65, and gains mainly on MathArena and REAL at B0 to B3 (MathArena $-4.0$,
REAL $-2.0$; Table C2); we read the MathArena part as compensation for a mis-centred level. On the hold-out (prior fit
on systems released before a cut-off, scored on newer systems alone on their benchmark, against REF fit on the same
older systems; Table C6), S gains $-3.98 \pm 0.90$ and $-3.82 \pm 1.06$ at cut-offs 2025.5 and 2026.0, while SY1.0
scores $+2.66 \pm 0.66$ and $+4.45 \pm 0.90$ against pre-registered predictions of $-1.00$ and $-0.42$. Eligible test
subjects are base models released in 2025 and 2026 [37], so this is the closest public analogue to them.

**Decision records in full.** (i) *SY1.0.* SY1.0 multiplies the attribute ridge by 1.56 except its release-year term. It
was proposed as the hedge for an H_cal world in which the hidden-subject offset is carried by the attribute ridge, a
world the subject-shift study found implausible (Section 7.3). Two rules applied. The round-3 amendment (hashed at 10:14
UTC) said to adopt SY1.0 if its job passed, and the job's rule compared S + C + P with SY1.0 against REF: at most $-0.5$
with no benchmark above $+0.3$. It passed at 10:29 UTC ($-0.71 \pm 0.19$; MathArena $-0.67$, Multi-SWE-bench $-0.49$,
REAL $-1.35$, ResearchCodeBench $-0.34$). The temporal-hold-out job, pre-registered by the subject-shift study before it
ran, required S at most $-2.0$ at both cut-offs and the order NP $\le$ S $\le$ SY1.0 $\le 0$ within one SE. At 10:33 UTC
every part held except SY1.0 $\le 0$ (Table C6). The first rule compared SY1.0 with the wrong reference: the decision is
between SY1.0 and S, and SY1.0 is worse than S on public replays (Table C5) and on the hold-out ($+6.64 \pm 0.91$ and
$+8.27 \pm 1.45$ against S). We kept S; the decision is in the hashed snapshot of 10:50 UTC and the conflict is recorded
in the queue file from the snapshot of 11:56 UTC on. (ii) *NP.* The provider-free ridge was pre-registered at 10:50 UTC,
before its replay: adopt if the composite with NP is at most $+0.2$ worse than the composite without it, with no
benchmark above $+0.5$. The macro passed ($-0.12 \pm 0.40$), but REAL was $+1.26 \pm 0.44$ ($+1.13$ in the pair-weighted
convention of the decision record), so NP was not adopted at 11:30 UTC. (iii) *$\sigma_g$.* The rule (adopt 1.4 if the
macro is at most $-0.5$, B1 to B31 improve on two of the three benchmarks with informative features, and Multi-SWE-bench
is not worse than $+0.5$) failed on its first item (Table C2). (iv) *$\lambda$.* No rule was pre-registered; Section 4.6
gives the reason it was not adopted. (v) *Dense-draw robustness rule.* Another team reported a B1 cost of pooling on
dense draws of one public benchmark, so before the composite's dense replay ran (REAL and ResearchCodeBench, $k = 4$ and
8, 10 draws per cell; job file by 11:35 UTC, result 12:23 UTC) its job file stated the rule: pass if S + C + P and C are
not worse than REF at any budget by more than two SEs and B1 is not worse than $+0.5$. C passes, with every cell from B1
to B31 negative and B0 exactly zero. S + C + P has no cell from B1 to B31 above $+0.07$ ($z = 0.1$), but
ResearchCodeBench B0 is $+2.81 \pm 0.81$ ($z = 3.5$) at $k = 4$ and $+1.10 \pm 0.33$ ($z = 3.3$) at $k = 8$, so the rule
fails. Our experiment log first recorded a pass, having checked only B1 to B31; an independent fact-check found the B0
failure before the decision. The cost is S's (C is zero at B0; S alone gives ResearchCodeBench B0 $+0.82 \pm 0.72$ at
$k = 1$ and $+1.43 \pm 0.52$ at $k = 2$ in the replay of Table 2), and B0 carries a tenth of the ALC weight, so it is
at most about $+0.3$ of ALC on that benchmark. The decision to keep S + C + P was taken after the result, for three
reasons (the rule was written against early-budget costs of pooling, which did not occur; the composite's
ResearchCodeBench ALC is negative at every density tried, $-0.12 \pm 0.31$, $-0.79 \pm 0.46$, $-2.56 \pm 0.63$ and
$-5.92 \pm 0.61$ at $k = 1$, 2, 4 and 8, and its REAL ALC $-1.01 \pm 0.51$ and $-3.28 \pm 0.83$ at $k = 4$ and 8; and
S wins the temporal hold-out of Table C6), and hashed at 13:51 UTC (Table D3).

**MathArena without the data defects.** In the MathArena tables 37.6% of a pair's items share their text with another
item of the pair (Section 3). The replay of Table 2 on MathArena with the system prompts, mismatched prompts,
repeated AIME attempts and rubric grades removed (`results/contrib_ma_clean.csv`, 18 draws per $k$) gives the composite
$-3.09 \pm 1.17$ at $k = 1$ and $-3.50 \pm 0.75$ at $k = 2$ ($-3.30 \pm 0.69$ averaged, against $-2.72$, $-2.38$ and
$-2.55 \pm 0.74$ with the defects), and C alone $-0.74 \pm 0.34$ at $k = 2$ ($-0.71 \pm 0.29$ with them). The job's
reading, written before it ran, was "no change" within $\pm 0.5$ of the uncleaned value and a down-weighting of
MathArena if the gain shrank by more than half; it grew by 0.75 (within noise), so the MathArena gain is not an
artifact of the defects. Without them MathArena is even easier relative to REF's level prior (B0 Brier of REF 0.389
against 0.354).

**Composition.** Within the composite, W20's gain over REF is $-3.31 \pm 1.13$ at $k = 1$ and $-1.08 \pm 0.69$ at $k =
2$ (B1 $-8.4$ against $-1.5$), and L15's is $-3.80 \pm 0.92$ and $-2.32 \pm 0.57$; at B1 both advantages shrink in every
benchmark when the benchmark has two pairs. The composite itself gains $-1.13 \pm 0.34$ at $k = 1$ (S only) and $-1.35
\pm 0.30$ at $k = 2$.

**Composition stress test.** The final subset's number of pairs per benchmark is not documented, so a replay added
draws with four pairs per benchmark (`results/fw_kstress.csv`, 18 draws per benchmark and $k$; its $k = 1$ and 2 rows
reproduce `combo_v1.csv` exactly). Within the composite, W20 against REF gives $-0.76 \pm 0.45$ at $k = 4$, with B0
near $-8$ at every $k$ ($-8.3$, $-7.3$ and $-8.1$): the gain shrinks with pairs per benchmark but stays negative, so the
flag written for this replay before it ran (W20 minus REF above $+0.5$ at $k = 4$ with $z \ge 1.5$) did not fire. L15
is ahead of W20 locally at every density, clearly so at $k = 2$ and 4 ($-0.48 \pm 0.57$, $-1.23 \pm 0.39$ and
$-1.36 \pm 0.25$ at $k = 1$, 2 and 4); the level stays a live decision (Section 7.3). Two splits of W20's prior variance between benchmark level and pair
($\tau_\mu = 1.676$ with $\tau_\delta = 1.48$, and $\tau_\mu = 2.112$ with $\tau_\delta = 0.735$), which leave pairs
alone on their benchmark and B0 unchanged (to within $2 \times 10^{-5}$ in Brier; the control's tolerance of $10^{-5}$
was set from smaller solver noise in smoke runs, so that invariance control, item (a) of the rule, also failed narrowly,
from solver noise), changed W20 by $+0.01 \pm 0.30$ and $+0.27 \pm 0.20$ at $k = 2$ and by
$-0.02 \pm 0.30$ and $+0.50 \pm 0.20$ at $k = 4$. Both failed their adoption rule, written with the flag (at $k = 2$ at
most $-0.5$ with $z \le -2$, at $k = 4$ at most 0, and no benchmark worse than $+0.3$ at $k = 2$ or 4), so the composite
keeps $\tau_\mu = 2.0$ and $\tau_\delta = 1.0$ for W20.

**Local guard of the live arms.** On pair-keyed replays (20 draws per benchmark and $k$; `results/adj_wide.csv`), L15
gains $-2.38 \pm 0.52$ (MathArena $-9.2$, Multi-SWE-bench $+2.1$, REAL $-1.9$, ResearchCodeBench $-0.5$), W20 $-1.22 \pm
0.64$ ($-9.0$, $+3.0$, $-0.8$, $+2.0$) and W20 at $m_0 = -1.5$ $-1.98 \pm 0.97$ ($-14.5$, $+5.2$, $-1.4$, $+2.7$).
Reweighted to the live joint distribution of B0 and B31 at 12:07 UTC on 2026-10-06, L15 and W20 give $-1.69 \pm 0.82$
and $-0.47 \pm 0.72$ (W20 at $m_0 = -1.5$: $-0.76 \pm 1.22$), and on 2026-10-20 $-2.01 \pm 0.73$ and $-0.38 \pm 0.71$
($-0.90 \pm 1.16$). On the same draws W15m16 gains $-2.21 \pm 0.89$ (`results/ld_guard_1010.csv`; MathArena $-14.0$,
Multi-SWE-bench $+4.9$, REAL $-2.0$, ResearchCodeBench $+2.3$; reweighted on 2026-10-20 $-1.15 \pm 1.09$) and MXs
$-2.98 \pm 0.71$ (`results/hr_mix_guard.csv`; reweighted $-2.54 \pm 0.85$). All pass the pre-registered guard (at most
2.0 worse than REF), as do WLA, WLB and W15d6 (Table D6). In round 1, local and live ALC effects agreed in sign for
only 1 to 3 of 7 arms ($\chi^2$ 1.8 on 7 degrees of freedom, which reflects no power rather than agreement), and 3 to 5 of 7 after this reweighting ($\chi^2$ 0.9). The one
consistent disagreement was B0 on the level: raising $m_0$ above $-2.0$ helped B0 locally by 13 to 21 and hurt it live
by 9 to 56 (pair fixed effect with a shared covariate, round-1 arms).

**Calibration.** A replay of the draws of Table 3 that saves every prediction (`results/cal_preds_bench.parquet`; all
five models reproduce the pair scores of `combo_v1.csv` bit for bit) gives, per budget and over all pair-draws pooled
with pair-equal weights, the expected calibration error (ECE) with ten equal-mass bins, a binned Murphy decomposition
Brier = REL $-$ RES + UNC + R [38], with R the within-bin term [39], and draw-clustered bootstrap SEs (200 replicates,
bins re-formed in each; Table C7, Figure C2). With equal-mass bins RES depends only on how predictions rank, so a
change of level acts mainly through REL; and at B0 the model predicts almost the same value for every item of a pair,
so B0 calibration is that of the level prior across pairs.

**Table C7.** Calibration on the draws of Table 3 (216 pair-draws per model, pooled with pair-equal weights, so $k = 2$
draws count twice and the ALC column differs from the benchmark-equal macro of Table 3; ten equal-mass bins;
$\pm$ draw-clustered bootstrap SE). A perfectly calibrated forecaster with the same predictions (labels drawn from them,
20 times) has an ECE of 0.007 to 0.009 in every cell. REL, RES and the differences against REF (paired SEs in the last
column; a RES increase lowers Brier) are in $10^{-3}$ Brier, and the bias (mean prediction minus success rate) in
$10^{-3}$ of probability; UNC is 232.0 in every column. Script: `swarm/round6/cal_sims/analyze.py`.

| Row | B0 | B1 | B3 | B7 | B15 | B31 | ALC-weighted |
|------------------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| ECE, REF | 0.149 ± 0.013 | 0.088 ± 0.012 | 0.053 ± 0.007 | 0.025 ± 0.006 | 0.031 ± 0.006 | 0.033 ± 0.005 | 0.058 ± 0.005 |
| ECE, REF + S + C + P | 0.146 ± 0.013 | 0.091 ± 0.012 | 0.049 ± 0.007 | 0.025 ± 0.006 | 0.030 ± 0.006 | 0.031 ± 0.005 | 0.057 ± 0.005 |
| ECE, W20 + S + C + P | 0.118 ± 0.013 | 0.048 ± 0.010 | 0.028 ± 0.007 | 0.021 ± 0.006 | 0.026 ± 0.006 | 0.029 ± 0.005 | 0.039 ± 0.003 |
| REL / RES / bias, REF | 31.3 / 21.8 / -144 | 10.0 / 28.7 / -78 | 3.7 / 42.8 / -33 | 1.1 / 50.1 / -7 | 1.1 / 58.6 / +1 | 1.2 / 65.4 / +7 | 6.4 / 44.8 / -37 |
| $\Delta$REL / $\Delta$RES, REF + S + C + P | 0.0 / +2.6 | -0.3 / +1.3 | -0.5 / +0.2 | 0.0 / +1.3 | 0.0 / +1.1 | -0.1 / +1.3 | -0.2 ± 0.3 / +1.2 ± 0.4 |
| $\Delta$REL / $\Delta$RES, W20 + S + C + P | -7.6 / +2.6 | -6.7 / -1.7 | -2.2 / +1.2 | -0.5 / +0.9 | -0.2 / +0.8 | -0.2 / +1.3 | -2.7 ± 0.8 / +0.6 ± 0.5 |
| $\Delta$Brier, W20 + S + C + P | -10.1 ± 1.1 | -5.3 ± 1.8 | -2.8 ± 0.8 | -0.7 ± 0.5 | -0.9 ± 0.3 | -1.5 ± 0.4 | -3.1 ± 0.7 |

![Figure C2. Reliability by budget on the draws of Table 3 for REF, REF + S + C + P and W20 + S + C + P (ten equal-mass bins, pair-equal weights; each legend gives the ECE ± draw-clustered SE). Script: `swarm/round6/cal_sims/analyze.py`.](../swarm/round6/cal_sims/out_bench/reliability_grid.png){width=80%}

Locally the level block acts through reliability and the composite through resolution: W20 + S + C + P lowers REL by
2.7 in ALC (7.6 at B0, 6.7 at B1) with RES almost unchanged, while REF + S + C + P raises RES by 1.2 with REL unchanged.
REF predicts too low at B0 on all four benchmarks (bias $-350$ on MathArena to $-22$ on Multi-SWE-bench), as expected of
the level chosen live (Section 5), and from B1 on its bias has opposite signs on MathArena ($-228$ to $-20$) and
Multi-SWE-bench ($+24$ to $+38$), so no single global map corrects both, consistent with the failed recalibration of
Appendix G (item 3). The platform's ECE computed locally (ten fixed bins within each pair, ALC-weighted mean over
pairs) is 0.138, 0.137 and 0.135 for the three models, against 0.046 to 0.048 for a perfectly calibrated forecaster with
the same predictions: with 44 to 113 evaluation items per pair it mostly measures binning noise and does not separate
the models. On the pair-keyed draws of the local guard (`results/cal_preds_pair.parquet`; REF reproduces
`adj_wide.csv` bit for bit) the ALC-weighted ECE is 0.047 for REF and 0.036 for W20 + S + C + P, whose Brier
difference of $-2.19 \pm 0.63$ splits into REL $-1.91 \pm 0.64$ and RES $+0.19 \pm 0.42$.
Local calibration describes the method on public data only; local and live evidence disagree on the level at B0
(Section 7.3), and the level block is chosen live. The platform ECE of the final build's four Finished runs is 0.117 to 0.142 (Table 6).

# Appendix D. Live methodology details

**Table D1.** Early live decisions re-estimated (arm minus REF; negative favours the arm) on the data of 2026-10-06
08:00 UTC (22 of our logs and 55 public). With connectors: the secondary estimator of Section 6, SE = max(1.2 ×
bootstrap, CR1), $K$ = alone pairs matched directly with REF (`live_evidence.py`, 300 replicates). Pair FE with
connectors: the primary estimator's model fitted on our logs and the connector arms (the column the v2 draft mislabelled
"our arms only"). Both connector columns come from the round-2 fit, which also includes the one public log that the
pre-registered connector set excludes (an earlier, much weaker model of one team; see *Public logs, connectors and
selection effects* below); without it, the ALC effects with connectors move by up to 7.7 (for $m_0 = -1.5$, $+0.7 \pm
6.8$ becomes $+8.5 \pm 4.8$) and those of the pair FE by up to 7.6 (Min at $m_0 = -1.5$, $+6.9$ becomes $-0.7$), and
five more arms exceed 1.4 SE with connectors, all of them worse than REF. Ours only: the primary estimator,
same SE rule, $K$ = pairs matched directly in any condition
(`tools/daily_report.py`, 200 replicates). The pre-registered decision SE adds the floor $\mathrm{SD}/\sqrt{K}$, which
binds for 9 of the 13 arms in the ours-only column ($K \le 4$); it only lowers $|z|$. Runs are uploads with feedback.
Headers: Conn. = with connectors, Pair FE = pair FE with connectors, Ours = ours only.

| Arm (change from REF) | Runs | Conn. ALC | Conn. B0 | Conn. B1 | Conn. B3 | $K$ | Pair FE ALC | Ours ALC | Ours B0 | $K$ |
|----------------------------------|------|----------|----------|----------|----------|---|----------|----------|----------|---|
| $m_0 = -2.5$ | 2 | +10.5 ± 12.3 | -4.8 ± 34.1 | +22.9 ± 26.7 | +11.4 ± 9.6 | 1 | -0.2 ± 8.8 | -5.9 ± 12.0 | -25.0 ± 37.0 | 3 |
| $m_0 = -1.5$ | 1 | +0.7 ± 6.8 | -14.6 ± 15.0 | -1.5 ± 15.9 | +8.5 ± 6.5 | 1 | +1.2 ± 5.1 | -1.0 ± 5.3 | -4.9 ± 21.4 | 3 |
| $m_0 = -1.0$ | 1 | +12.9 ± 9.5 | +3.1 ± 17.2 | +23.4 ± 18.9 | +15.7 ± 11.3 | 0 | +0.8 ± 6.9 | -1.7 ± 11.9 | +33.6 ± 24.2 | 0 |
| undamped, $m_0 = -1.5$ | 2 | +3.6 ± 5.6 | -10.7 ± 13.9 | +6.4 ± 12.2 | +8.3 ± 6.1 | 2 | -4.6 ± 4.8 | -4.3 ± 6.8 | -6.3 ± 19.5 | 4 |
| undamped, $m_0 = -1.0$ | 1 | +14.0 ± 10.7 | +9.1 ± 24.0 | +18.3 ± 24.9 | +18.1 ± 10.1 | 1 | +6.6 ± 7.2 | +3.2 ± 19.1 | +30.4 ± 62.6 | 3 |
| Student-t(3) level prior, $m_0 = -1.5$ | 1 | +5.9 ± 5.6 | +8.5 ± 17.4 | +9.0 ± 13.3 | +9.2 ± 6.4 | 1 | -0.2 ± 4.8 | +1.0 ± 7.1 | +2.2 ± 20.0 | 1 |
| shared level $\tau_M$ (5 variants, undamped) | 5 | +3.1 to +24.8 (each ±6 to 9) | | | | 0 or 1 | -0.3 to +11.4 | -5.0 to +11.8 (each ±3.9 to 21.3) | | 2 or 3 |
| mixture $\pi = 0.25$, $m_0 = -1.0$ | 1 | +5.6 ± 4.4 | -6.1 ± 18.5 | +9.0 ± 9.6 | +6.5 ± 5.3 | 1 | -3.4 ± 5.0 | -4.6 ± 5.6 | -16.5 ± 14.8 | 3 |
| mixture $\pi = 0.35$, $m_0 = -1.0$ | 1 | +7.4 ± 12.7 | -26.8 ± 19.4 | +15.1 ± 26.7 | +21.1 ± 22.8 | 2 | -5.9 ± 11.4 | +5.7 ± 12.4 | +8.8 ± 23.0 | 5 |
| narrow pair prior (d06) | 0 | no feedback | | | | | | | | |
| REF minus the best-covered public arm (context) | | -1.7 ± 4.8 | +4.9 ± 12.5 | -1.6 ± 9.9 | -2.6 ± 5.1 | | | | | |

The positive B3 column with connectors partly reflects noise shared through REF's own chain (bootstrap correlations 0.5
to 0.9); the regime test (Appendix E) re-references its contrasts to the best-covered public arm to remove it. With
connectors, the five shared-level variants have positive point estimates (z 0.4 to 3.4), and only the variant at
$m_0 = -1.0$ that splits $\tau_\mu = 1.2$ into a shared 0.8 and a benchmark-specific 0.894 (Min) exceeds two SEs; the
other variant at $m_0 = -1.0$, on top of
$\tau_\mu = 1.2$, is $+3.1 \pm 7.5$. The Student-t arm was not better, and half of its budget-0 shift came from our
variance-matched moment approximation (0.308 against an exact 0.288 and a normal 0.271) rather than the heavy tail.
$m_0 = -1.0$ is worse by point estimate in both model families with connectors (each about 1.3 SE) and not ours only.
Published decompositions put most of a new pair's level variance at the benchmark (SD 2.45 logits across HAL benchmarks
[9], intra-class correlation 0.75 on Harbor-Index [10]), consistent with dropping the shared level.

**Table D2.** Our live arms at the final look (04:15 UTC on 2026-10-30; 31 configurations; 575 uploads, 496 with
feedback; $m_0$ at six values plus d06's $-1.693$; by 2026-10-20 the counts were 29 configurations and 467 uploads, 401
with feedback, and on 2026-10-06 19 configurations and 40 uploads, 37 with feedback). Global counts only; Table D6 gives
the contrasts of 2026-10-20 and Section 7.4 those of the final look. UTC date: first upload.

| Ledger arm | Change from REF | Uploads (with feedback) | UTC date | Status, reason |
|------------------------|---------------------------|--------|------|--------------------------|
| v1_m05, v1_m15 | first model (no subject prior, undamped), $m_0$ -0.5 / -1.5 | 1 (1), 1 (1) | 10-05 | superseded by the subject prior (E3, E4) |
| v2_noM_m15, v2_noM_m10 | subject prior, undamped, $m_0$ -1.5 / -1.0 | 2 (2), 1 (1) | 10-05 | superseded by the damped solver (E9) |
| v2_Min_m10, _m15, _m20 | shared level $\tau_M$ 0.8 with $\tau_\mu$ 0.894 | 1 (1) each | 10-05 | dropped: shared level (E11) |
| v2_Mtop_m10, _m15 | shared level on top of $\tau_\mu$ 1.2 | 1 (1) each | 10-05 | dropped: shared level (E11) |
| v3_noM_t3_m15, _m10 | damped, Student-t(3) level, $m_0$ -1.5 / -1.0 | 1 (1), 1 (0) | 10-06 | dropped (E12) |
| v3b_noM_n_m20 (REF) | damped, normal level, $m_0$ -2.0 | 172 (150) | 10-06 | incumbent until the final look, submission 963675 (E13, E14) |
| v3b_noM_n_m15 (L15) | $m_0$ -1.5 | 60 (49) | 10-06 | probe (Section 7.3); finalist, fails (a), (b) and (d) |
| v3b_noM_n_m10 | $m_0$ -1.0 | 1 (1) | 10-06 | dropped (E14) |
| v3b_noM_n_m25 | $m_0$ -2.5 | 3 (3) | 10-06 | dropped (round-2 regime analysis) |
| v4_mix25_m10, v4_mix35_m10 | mixture $\pi$ 0.25 / 0.35, $m_0$ -1.0 | 2 (2), 1 (1) | 10-06 | dropped (E17, regime test) |
| v5_d06_m169 | $\tau_\delta$ 0.6, $\tau_\phi$ 0.3, $m_0$ -1.693 | 1 (0) | 10-06 | never reported; no further uploads |
| v5_w20_m20 (W20) | $\tau_\mu$ 2.0 | 118 (101) | 10-06 | probe; finalist, passes every switch item: the final level block |
| v6_combo_w20 | composite S + C + P on W20 | 1 (1) | 10-07 | platform check, Finished; parent of the shipped insurance build |
| v6_m150_mu20_d10 (W20 at $m_0 = -1.5$) | $m_0$ -1.5, $\tau_\mu$ 2.0 | 41 (35) | 10-07 | grid arm; finalist, fails (a), (b) and (d) |
| v6_m250_mu25_d06 (WLA) | $m_0$ -2.5, $\tau_\mu$ 2.5, $\tau_\delta$ 0.6 | 23 (19) | 10-07 | grid arm; under 30 scored runs, not a finalist |
| v6_m225_mu25_d14 (WLB) | $m_0$ -2.25, $\tau_\mu$ 2.5, $\tau_\delta$ 1.4 | 14 (14) | 10-07 | grid arm; not a finalist |
| v8_mxs (MXs) | two-component level prior (Table A1) | 42 (36) | 10-07 | exploration arm; finalist, fails (a) and (b) |
| v6_m150_mu20_d06 (W15d6) | $m_0$ -1.5, $\tau_\mu$ 2.0, $\tau_\delta$ 0.6 | 15 (13) | 10-08 | grid arm; not a finalist |
| v6_m150_mu16_d14 (W15m16) | $m_0$ -1.5, $\tau_\mu$ 1.6, $\tau_\delta$ 1.4 | 57 (52) | 10-10 | grid arm; finalist, fails only (b) |
| v6_combo_ref, v6_combo_m150_mu20_d10 | composites on REF and on W20 at $m_0 = -1.5$ | 1 (1) each | 10-19 | early platform checks, Finished |
| v6_combo_m150_mu16_d14 | composite on W15m16 | 2 (0) | 10-19 | early platform check and its rerun of 10-21, both died before scoring |
| v6_combo_w20_c1, _m150_mu16_d14_c1 | insurance builds of the W20 and W15m16 composites | 4 (3) each | 10-21 | 3 Finished each; the W20 build ships, selected run 1020301 |

**Table D3.** Hashed pre-registration snapshots (`submissions/prereg/LOG.csv`; SHA-256 of a copy of the queue file).

| UTC | SHA-256 prefix | Content |
|------------|-----------|------------------------------------------------------------|
| 10-06 10:14:58 | 4e43b4e0 | round-3 amendments: decision tool, raised floors, amended L15 reading, final composite |
| 10-06 10:23:06 | 4e43b4e0 | unchanged; recorded before the last three uploads of the day |
| 10-06 10:50:11 | f9e27d38 | S kept and SY1.0 rejected; NP adoption rule recorded before its replay |
| 10-06 11:34:23 | d99d00cd | NP not adopted; composite fixed; platform check planned |
| 10-06 11:56:49 | 70f464a2 | round-4 adaptive allocation, before the 2026-10-07 uploads; SY1.0 decision record |
| 10-06 13:51:27 | 633ed4eb | round-5 additions: dense-draw rule reported as failed and the composite kept; confirmatory fresh-pairs estimate defined; final-selection rules |
| 10-06 16:03:12 | 7dad54d4 | round-6 amendment: no SD floor in the switch SE at the final look for arms with at least 60 direct matches; finalist count pinned |
| 10-07 00:00:44 | 7dad54d4 | unchanged; start of the 2026-10-07 uploads |
| 10-07 03:30:07 | 8da4ce15 | daily upload cap raised from 30 to 36 (author's decision); the day's remaining uploads listed |
| 10-07 05:10:10 | f7f52d32 | round-8 amendment: the two-component level arm MXs as an exploration arm outside the grid, after its local guard passed (Section 7.4, Appendix D) |
| 10-07 09:12:48 | 0769e88e | plan for 2026-10-08 (design refresh); new grid arm W15d6 |
| 10-07 14:32:34 | a7ec6302 | composite additivity amendment, before its job ran (Pre-registered rules above) |
| 10-09 05:00:01 | 72684c81 | plan for 2026-10-10; new grid arm W15m16, conditional on its local guard |
| 10-13 04:12:41 | ebb4aeac | plan for 2026-10-14; checkpoint A (interim look), no interim commit |
| 10-15 11:51:22 | 94f4fae8 | additivity guard of the W15m16 composite, before its job ran |
| 10-08 01:40 to 10-16 01:24 | (16 snapshots) | the other daily plans and starts of upload days; no rule change |
| 10-16 03:20:03 | bf1d08a8 | composite readiness amendment: failure clause of final-selection rule (b), early composite platform checks |
| 10-16 03:22:42 | 2ae0bdbd | guard tool: decisive criterion of rule (b), `tools/composite_guard.py` |
| 10-19 05:24:40 | 4c40ab5a | plan for 2026-10-19; the 10-17 and 10-18 upload windows missed |
| 10-19 05:41:33 | b1560772 | treatment rule for the platform-created runs, before any was scored; uploads paused |
| 10-19 16:29:34 | 937b84bc | pause lifted; the composite platform checks go first |
| 10-19 19:45:52 | 909148a0 | plan for 2026-10-20; checkpoint B, no interim commit; insurance builds (D-3) |
| 10-20 03:26:30 | 45b40096 | plan for 2026-10-21 with the insurance runs and the rerun of the W15m16 composite check |
| 10-21 02:40:39 | 128b677d | plan for 2026-10-22 (never used: the windows of 10-22 to 10-27 were missed) |
| 10-28 05:12:59 | 40b72c4b | plan for 2026-10-28; missed windows recorded; schedule of the last days restated |
| 10-30 02:11:34 | d97004d2 | plan for 2026-10-30, the last upload day; rule (b) to be read on an insurance build and its parent together |
| 10-30 04:15:20 | 17894320 | final look moved earlier by availability; runs still in flight count as dead |
| 10-30 04:21:26 | 7c0d7f31 | final decision: switch to W20, rule (b) holds, the insurance build ships, selected run 1020301 |

**Table D4.** Timeline of 2026-10-06 (UTC; file modification times and queue-file entries).

| Time | Event |
|------|----------------------------------------------------------------|
| 08:37 | predictions for L15, W20 and W20 at $m_0 = -1.5$ written (`recommend2_out.txt`) |
| 08:50 | first revision: rules and predictions in the adjudication report; a B3-based regime gate whose sign depended on the estimator retired |
| 09:00 | probe uploads (REF, W20, L15); queue file revised at about 09:10 |
| 09:14 | holdout implemented, before any decision under the screen rule (an earlier mixture drop under the round-1 regime gate used all pairs) |
| 09:30, 09:56 | first and second probe feedback (six results) |
| 09:36 | decisive item-pooling replay finished (its rule was written in the pooling report at 08:51) |
| about 09:45 | internal analysis: the two estimators point to opposite hypotheses on L15's budget-0 test |
| 10:10 | composite replay finished |
| 10:14 | second revision, first hashed snapshot |
| 10:23, 10:55 | third and fourth probe feedback |
| 10:29, 10:33 | SY1.0 passes its job rule; the temporal hold-out fails it |
| 10:50 | snapshot: S kept; NP rule recorded |
| 11:30 | NP replay fails its benchmark cap |
| 11:56 | third revision: adaptive allocation (Section 7.4) |
| 12:23 | dense-draw replay finished; its rule fails at B0 on ResearchCodeBench (first logged as a pass) |
| 12:41 | local guards of two new grid arms finished |
| 13:38 | MathArena replay without the data defects finished (Appendix C) |
| 13:51 | fourth revision: dense-draw decision record, confirmatory fresh-pairs estimate, final-selection rules |
| 14:38 | composition stress test finished; the variance splits fail their rule and the composition flag does not fire (Appendix C) |
| 15:05 | calibration replays finished (reporting only; Appendix C) |
| 15:41 | compute job of the operating-characteristics simulation finished |
| 15:57 | last simulation chunks and final tables of the operating characteristics (Table D5) |
| 16:03 | fifth revision: final-look SE without the SD floor at 60 or more direct matches |

**Use of IDs in full (Section 6).** Anonymous IDs are used only as opaque keys: to join a pair's results across
submissions, as the organizers intend; as the input of a fixed hash that assigns the holdout; and to group pairs by
benchmark key within a draw, to mark whether a pair shared its benchmark, stratify the bootstrap and compute two
further gates of the switch rule. Planning simulations model draw composition with per-key pair counts from our logs,
used only to simulate how matched pairs accumulate, never as a model input.

**Holdout and the revisions of the live rules (Section 6).** The holdout is the set of pairs with
`sha256("brier-rabbit-holdout|subject_<id>|benchmark_<id>")[:2] < "40"` (22.3% of distinct pairs on 2026-10-06, 21.8%
of the 303 seen by 2026-10-20); it was specified
after the decisions of Section 7.1 and implemented at 09:14 UTC, before any decision under the screen rule (an earlier
mixture drop under the round-1 regime gate used all pairs). The revisions: at 08:50 UTC, before the probe uploads, to
add L15 and W20 with their predictions; at 10:14 UTC, after six probe results and an internal finding that the two
estimators pointed to opposite hypotheses on L15's budget-0 test, to raise SE floors, fix the decision tool and let
disagreement between the estimators keep $m_0 = -2.0$, a reading written after interim data that favours the incumbent,
so Table 4 reports both readings; at 11:56 UTC, before the 2026-10-07 uploads, to adopt the adaptive allocation of
Section 7.4, which leaves the drop rules and the switch checklist unchanged; at 13:51 UTC, to record the failed
dense-draw rule and the decision to keep the composite (Section 4.6), define the confirmatory fresh-pairs estimate
(Section 10) and fix the rules for naming the final run; and at 16:03 UTC, after simulations of the whole decision
pipeline showed that the SD floor blocked effects weaker than about $-1.7$, to drop that floor from the switch SE at
the final look for arms with at least 60 direct matches with REF. A sixth amendment, on 2026-10-07 at 05:10 UTC, added
the exploration arm MXs with its own allocation and stop rules (Section 7.4, the round-8 paragraph below, Table D3).
Later amendments, each hashed before the uploads, jobs or runs it governs, are listed in Table D3 and stated in full at
the end of the next paragraph.

**Pre-registered rules.** Screen, at six runs and after every further ten: drop an arm permanently if its ALC effect
exceeds $+3.5$ or any of B0, B1 and B3 is worse than REF by more than two SE$_h$; an arm with a negative ALC effect and
$z \le -1$ is a provisional finalist. Switch, on 2026-10-30 for every non-dropped arm with at least 30 runs, with the
primary estimator: (a) the holdout effect is at most zero; (b) $z \le -1.64$ on all pairs ($-1.8$ with two or more
finalists); (c) the benchmark-balanced effect is at most zero; (d) every leave-one-benchmark-out estimate is at most
$+1.0$; (e) no budget is worse than REF by more than two SE$_h$; (f) the local replay is at most 2.0 worse, unweighted
and reweighted to the live distribution. Ties within 1.0 go to the simpler model. New arms freeze on 2026-10-28 at 23:59
UTC, and 2026-10-29 and 10-30 carry only REF, at most two finalists and the composite's guard runs. Added at 13:51 UTC:
for every finalist we also report its contrast with REF on fresh pairs, those whose first appearance in any of our runs
or in the connector logs is in a run uploaded after that snapshot, and only this estimate is labelled confirmatory; the
selection email names the most recent Finished run of the final build, whose hash matches the build record
(final-selection rule (a)); if the composite's guard runs trip the screen (ALC above $+3.5$ against the plain
level-block build on matched pairs, or a failed or unfinished run), the plain level-block build is shipped instead
(rule (b)); and if the chosen build has no Finished run by 2026-10-30 18:00 UTC, the most recent Finished run of the
plain level-block build, else of REF, is named (rule (c)). Added at 16:03 UTC: at the final look, items (b) and (e) use
SE$_h$ without the $\mathrm{SD}/\sqrt{K}$ floor for an arm with at least 60 direct matches with REF, and the finalists
counted in (b) are all non-dropped arms with at least 30 runs. Added on 2026-10-07 at 14:32 UTC and on 2026-10-15 at
11:51 UTC, before their jobs ran: the composite on a normal-family level winner other than REF, W20 and L15 (whose
additivity evidence is the replay of Table 2) ships only if its local gain from S + C + P is within $\pm 0.3$ of W20's
on the same draws, else the plain level-block build ships; the guards passed for W20 at $m_0 = -1.5$ ($-1.00$), W15d6
($-1.03$) and W15m16 ($-1.01$), against $-0.94$ for W20, with every benchmark and budget negative. Amended on
2026-10-16 at 03:20 UTC (composite readiness amendment), because the platform had ended 77 of our 355 uploads since 2026-10-07 without a Finished status, in every arm including REF, so that the
literal failure clause would trip rule (b) with probability about 0.5 over three guard runs: a composite run that ends
Failed-with-scores (complete scores) is scored and enters the guard contrast; one that dies before scoring (stuck in
Submitting, Preparing or processing) is ignored and replaced, as for every arm; and rule (b) trips on a composite run
that ends without scores while plain runs uploaded within an hour of it finish, on any composite run whose logs show a
time or memory limit or an exception raised in the submission code, and on two composite runs ending without scores
whatever the cause. The $+3.5$ part of rule (b) and rules (a) and (c) are unchanged. The same amendment scheduled one
early platform check each of the composites on W15m16, W20 at $m_0 = -1.5$ and REF (passed if the run reaches Finished,
or Failed-with-scores with complete scores; a run that dies before scoring is replaced once; a run without scores or
with an exception from the submission code is investigated before 2026-10-24); their contrasts with the plain builds
decide nothing. Its reader, `tools/composite_guard.py` (2026-10-16 03:22 UTC), classifies every run of the composite,
fits the primary estimator on our runs of the two builds only (plain build as reference; selection set decisive, all
pairs shown; SE$_h$ as in the daily report) and prints the verdict; its decisive criterion is that of rule (b), and the
screen's B0, B1 and B3 clauses are printed for information only. Added on 2026-10-19 at 05:41 UTC, before any could be
scored: a run the platform created in our account would enter the evidence only once scored, only if its file matches
our build record and not if it repeats the pair set of an earlier run of the same build, and would then be an ordinary
run of its arm under every rule. Added on 2026-10-19 at 19:45 UTC (decision D-3, taken because our question to the
organizers on the summative subset was unanswered and they had said nothing about the size of the summative draws):
insurance builds of the composites of the two leading finalists (W20
and W15m16) that add only `prediction_config.json` with `{"concurrency": 1}` must reproduce their parents' predictions
exactly (`tools/verify_shipped.py`) and get one platform run by 2026-10-24, taken from that day's design uploads; at
the final look the insurance build of the winning composite ships in its parent's place if it has a Finished run and
identical predictions, otherwise the parent ships, with rules (a) to (c) unchanged.

**Pre-registered readings of the L15 test.** Original (08:50 UTC): with connectors, B0 at most $-3.2$ with $z \le -1$
relative to $-3.2$ selects H_pub, B0 at least $-3.2$ with $z \ge +1$ selects H_cal, and otherwise two more L15 runs
follow and the point estimate decides. Amended (10:14 UTC): both estimators must agree in sign relative to $-3.2$; after
six L15 runs the point estimates are used if they agree, and H_cal (keep $m_0 = -2.0$) is the default if they disagree.
From 11:56 UTC the test is still reported but no longer drives arm choice, which follows the design posterior over $d$.
Simulated at the planned 12 REF and 6 L15 runs (400 worlds per truth model, connector arms from their validated
replicas), the original z-rule decides in 25% (data generated from the H_pub prediction) and 28% (H_cal) of worlds and
is wrong in about 1%; with its fallback it is right in 88% and 94%, and the amended reading in 76% and 95%. In worlds
drawn from the design posterior with $d \le 0.75$ or $d \ge 1.25$, the original rule with its fallback is right in 62%
and 82% and the amended reading in 50% and 86%, and with 60 REF and 30 L15 runs still only in 65% and 87% (amended 61%
and 88%): the sign of L15's budget-0 contrast depends on the pool regime as much as on $d$ (its true value is below
$-3.2$ in 68% of the worlds with $d \le 0.75$ and above it in 90% of those with $d \ge 1.25$).

**Standard errors.** The null simulation used the real 77-log design (60 simulations of 80 draws, independent per-pair
heterogeneity). The bootstrap SE's median ratio to the true sampling SD was 0.95 (10th percentile 0.82 to 0.84, sparse
arms 0.77 to 0.87). Real heterogeneity that depends on each pair's rate, and the dropping of draws in which an arm is
unidentified, push in the same direction. The first feedback tool used $\sigma/\sqrt{K}$ with the wrong degrees of
freedom; the corrected classical SE matches brute-force dummy-variable least squares to $10^{-9}$ (unit tests), and the
median ratio of the bootstrap SE to it is now 0.81 (it was 2.05). The pooled floor of $\mathrm{SD}_{\mathrm{arm}}$ is
ALC 0.0147, B0 0.055, B1 0.0487, B3 0.0132, B7 0.0076, B15 0.0043 and B31 0.0039; the B0, B3 and later floors were
raised on 2026-10-06 to the live matched SDs. The arm-specific SD grows with how far an arm's predictions move from
REF's: exact calculations give B0 SDs of 0.013 for d06, 0.020 for W20, 0.035 for L15 and 0.054 for $m_0 = -1.25$, and
competitor-type arms show direct matched SDs of 0.088 at B0, 0.0186 for ALC, 0.0225 at B3 and 0.0118 at B7 (1.3 times
the pooled floor for ALC and about twice the earlier floors at B0, B3 and B7).

**Switch-rule power (illustration).** With an independent normal approximation and the ALC floor, a true effect of
$-3.1$ passes $z \le -1.64$ with probability 0.21, 0.38 and 0.68 at 16, 40 and 100 matched pairs, and a holdout covering
about 22% of pairs is at most zero with probability 0.65, 0.73 and 0.84. The simulation below replaces this
illustration for the procedure as run.

**Operating characteristics (simulation).** We simulated the complete live procedure: the screens, the adaptive
allocation of Section 7.4 (its in-simulation version, with a fixed shortlist of seven arms) and the switch checklist,
with the decision tool's own estimators (on simulated logs its effects and cluster SEs agree with the tool to
$10^{-16}$; the bootstrap term of SE$_h$ is a calibrated proxy in the daily screens and a 100-replicate cell bootstrap
at the final look). Synthetic formative pools have the structure of our logs: draw sizes and shared-benchmark appearances as
observed, deterministic scores for pairs alone on their benchmark, a simulated holdout of 22.6% of pairs, as in our
logs, and per-pair contrasts
from the exact alone-pair calculation of Section 7.2 at pool regimes drawn from the design posterior, with item and
partner noise calibrated to the matched-pair SDs of W20 and L15 at every budget (the observed SDs fall inside the
simulated 10th to 90th percentiles except W20's B1 SD, which the simulation overstates). From the state of 2026-10-06
each campaign allocates 30 uploads a day and applies the checklist on 2026-10-30 (Table D5). Two null targets bracket
the truth: the candidates equal REF on the formative pool itself, or only in a larger population of which the pool is
a sample; which one applies depends on how the final subset relates to the formative pool, which is not documented.
Under the rule as it stood before 16:03 UTC the finalists' $z$ statistics had SD 0.33 rather than 1 under the pool
null: the SD floors (ALC 0.0147) exceed the realised per-pair SDs of the level arms (0.006 to 0.012) and bind for 70% to 78% of
finalists, and matched pairs cover about 70% of a finite pool, so effects weaker than about $-1.7$ could not pass at any
number of uploads. The round-6 amendment (Section 6) drops the floor at the final look for arms with at least 60 direct
matches (in simulation every finalist reached that): it raises the power at $-1$ and $-2$ to 0.29 and 0.97, keeps the
population false-switch rate at or below 5% within Monte Carlo error, and adds no switch to a worse arm. Before the
amendment all 14 population-null switches went to an arm better than REF on the formative pool itself (by 0.7 to 3.3);
the saved outputs do not record this for the extra switches of the amended rule. The screens dropped a null arm in
1.3% (pool), 4.6% (population) and 10.5% (SDs at the floors) of started arms, a truly better arm in at most 1% and an
arm worse by 2 in 30%; the interim commit never fired under a null and chose the truly better arm in all 202 commits.
Not simulated: the composite's guard runs, item (f), the real tool's daily choice of up to five arms among 84, and new
public logs; simulated effects are uniform shifts with W20's predicted budget profile.

**Table D5.** Simulated operating characteristics (1,780 campaigns; probability that the procedure switches, or for a
better arm that it ships that arm; $\pm$ Monte Carlo SE, zero counts with a one-sided 95% upper bound). Before: the
rule before the round-6 amendment; adopted: as amended at 16:03 UTC. Calibrated: per-pair SDs as observed; floors: every
per-pair SD at the SE floors. Scripts: `swarm/round6/oc_sims/` (Appendix B).

| Truth | Campaigns | Before | Adopted |
|------------------------------------------------|-----------|---------------|---------------|
| All candidates equal REF on the formative pool | 77 | 0 ($\le$ 0.038) | 0 ($\le$ 0.038) |
| Equal to REF in the population, calibrated | 154 | 0.019 ± 0.011 | 0.032 ± 0.014 |
| Equal to REF in the population, floors | 186 | 0.043 ± 0.015 | 0.043 ± 0.015 |
| Same on a smaller pool, calibrated / floors | 40 / 60 | 0.025 / 0.033 | 0.050 / 0.033 |
| All candidates worse by 1 or 2, or W20 worse by 2 | 424 | 0 ($\le$ 0.007) | 0 ($\le$ 0.007) |
| W20 better by 1 | 133 | 0.008 ± 0.007 | 0.286 ± 0.039 |
| W20 better by 2 | 116 | 0.879 ± 0.030 | 0.974 ± 0.015 |
| W20 better by 2, floors | 115 | 0.435 ± 0.046 | 0.435 ± 0.046 |
| W20 better by 3, calibrated / floors | 115 / 115 | 1.000 / 0.730 | 1.000 / 0.730 |
| W20 better by 5 | 130 | 1.000 | 1.000 |
| L15 better by 3 | 115 | 0.983 ± 0.012 | 0.983 ± 0.012 |

**Public logs, connectors and selection effects.** We fetched the other teams' files read-only at the pinned commits,
checked that each file's git blob hash equals the hash reported by the GitHub API, and mapped the logs to model arms
using each owner's own documentation. aghasalim's submission directories contain no acquisition policy; doctawho42's
contains an optional `labeling.py` that its build ships only with `--labeling` and whose code is identical to the kit's
default rule. Replicas of their predictors built from
their public code match the real predictors to $3 \times 10^{-16}$ and $4 \times 10^{-5}$ on synthetic pairs. With
public arms the effective number of matched pairs rises by a median factor of 1.87 and the bootstrap SE falls by a
median factor of 0.69; with and without connectors the ALC effects differ by a median of 0.33 SE (maximum 1.69; up to
3.1 SE for chained B0 contrasts). The only influential public arm is one draw of one team's earlier, much weaker model,
and the pre-registered connector set excludes it. Splitting every public arm into alone and shared sub-arms moves our
effects by at most 0.7 (ALC), 1.0 (B1) and 0.7 (B3), and residual SDs agree with and without connectors (ALC 0.0114 and
0.0115). Raw draw scores are poor evidence: across six arms with repeated runs the within-arm SD of a draw's ALC is
0.017 (this pools nine draws reported by lucaluo925; 0.023 over our first ten REF runs, 0.021 over the 103 of
2026-10-20), and the best of 20 draws of a model
whose mean is 0.180 lands near 0.13 to 0.15 by chance alone, so single-draw scores, leaderboard entries included, carry
a selection effect.

**Placebo and chained B0.** Public arms whose budget-0 predictions are identical show B0 contrasts of $-3.1 \pm 4.0$,
$+2.2 \pm 4.2$, $+0.3 \pm 3.2$ and $-0.2 \pm 3.1$ through the network, so the B0 machinery is unbiased. B0 effects
depend strongly on each pair's base rate, so a chained B0 contrast averages over a different mix of pairs than a direct
one; the largest disagreement between the two was 3.1 SE, and we report both.

**Holdout string.** Hashing the IDs as printed (with the `subject_` and `benchmark_` prefixes) and hashing bare digits
disagree on the holdout status of a third of appearances, so the prefixed form is pinned. The holdout covers 22.4% of
distinct pairs in the data of Table D1, 22.3% at 12:07 UTC and 21.8% (66 of 303) on 2026-10-20.

**Adaptive allocation in detail (round-4 amendment).** *Posterior.* A uniform prior on a grid: logit shift of the
regular component against the public pairs ($-0.5$ to $+1.0$, 7 values), spread (0.8 to 1.4), near-zero share (0 to 0.4,
logit-normal around 6%), subject offset $d$ ($-0.5$ to $+2.5$ in steps of 0.25) and discrepancy scale (1.0 to 3.0). The
data are one joint stratified bootstrap (1,500 replicates, covariance inflated by $1.2^2$) of 56 selection-set contrasts
(connector arms with connectors at B0 to B7; our level arms ours only) plus the B31 constraint. *Predictions.* Every
alone-pair table depends on an arm only through $m_0 + d$ and $V = \tau_\mu^2 + \tau_\phi^2 + \tau_\delta^2$, so one
table per $(m_0 + d, V)$ serves every arm; the grid reproduces the round-2 predictions (L15 $-2.91$ against $-3.1$, W20
$-3.07$ against $-3.1$). A squared-exponential Gaussian process over $(m_0, \log V, \tau_\mu^2/V)$, zero at REF, models
the discrepancy of our level arms (ALC SD about $1.3 \times$ the scale at W20's distance); the fitted scale is 1.5 to 2,
about twice its prior value. *Objective and allocation.* The value of a plan (final run counts per arm) is the expected
true ALC of the arm the 10-30 checklist selects (REF if none passes), computed by preposterior simulation with matched
pairs that saturate as an arm covers the pool. Coordinate descent over run counts (30 to 220, or stop) for at most five
shortlisted arms, chosen by greedy batch expected improvement, gives the plan; new arms get their six screening runs at
once, others pace evenly to 10-28, and the plan is recomputed daily. An optional interim commit changes only the
allocation, not the incumbent ($z \le -2.5$ and posterior probability 0.8 that the leader is best). *Validation.* Policies: the literal pre-registered
schedule (arms stop at 40 runs), the fixed rotation, a static rotation over five arms, a Thompson-style allocation and
the plan; 75 base worlds plus 30 worlds in each of five variants (H_pub, H_cal, half-size discrepancies, twice-smoother
GP, no discrepancy), every policy decided by the same checklist. Paired against the rotation, the plan gains
$-1.43 \pm 0.18$ (base), $-0.94 \pm 0.25$ (H_pub), $-1.63 \pm 0.31$ (H_cal), $-1.27 \pm 0.24$, $-1.02 \pm 0.29$ and
$-0.40 \pm 0.19$ ($-1.18 \pm 0.10$ over all 225 worlds; better in 55% of worlds, worse in 5%); the static rotation gains
$-1.01 \pm 0.18$ (base) and the Thompson-style allocation $-1.38 \pm 0.18$, so most of the gain comes from testing more
arms and daily adaptation adds $-0.18 \pm 0.05$ over all 225 worlds ($-0.42 \pm 0.10$ in base worlds). In base worlds
the plan selects an arm within 0.5 of the best available arm 75% of the time, against 19% for the rotation, and the
literal schedule uses only about 115 of the about 720 uploads. No policy selected an arm whose true effect was worse
than REF's, in any variant. The validation used the earlier, slightly stricter B0 and B3 floors. *Posterior at 11:44
UTC* (37 of our logs, 54 connector logs): W20 $-2.20 \pm 1.39$ (probability 0.95 of beating REF), L15 $-0.23 \pm 1.83$,
REF best with probability below 0.001, the largest probability of being best among 84 grid arms 0.084, and P(H_pub)
0.24, but 0.70 without the B31 constraint and 0.74 without the connector arms, so it left $m_0$ open. *2026-10-07.*
The plan gives 8 uploads to W20, 8 to REF (one used for the pre-registered platform check of the composite on W20), 3 to
L15, and 4, 4 and 3 to the new grid arms $(m_0, \tau_\mu, \tau_\delta) = (-1.5, 2.0, 1.0)$, $(-2.5, 2.5, 0.6)$ and
$(-2.25, 2.5, 1.4)$. The first passes the local guard ($-1.98 \pm 0.97$; reweighted $-0.76 \pm 1.22$). On the same
draws (`results/ld_guard_1007.csv`) the other two give $-0.44 \pm 0.91$ for $(-2.25, 2.5, 1.4)$ and $+0.39 \pm 0.67$
for $(-2.5, 2.5, 0.6)$ (MathArena $-10.8$ and $-5.7$, Multi-SWE-bench $+5.0$ and $+2.9$, REAL $-0.6$ and $+1.1$,
ResearchCodeBench $+4.6$ and $+3.3$), and $+0.63 \pm 1.00$ and $+0.90 \pm 0.69$ reweighted to the live joint
distribution of B0 and B31 as in Appendix C, so all three pass the guard (at most 2.0 worse than REF, unweighted and
reweighted).

*2026-10-07, headroom and guards (round 8).* The design posterior refit with the day's runs gave $E[d] = 0.36$ and
P(H_pub) $= 0.87$ (1.16 and 0.24 on 2026-10-06). On it, an exact calculus of the alone-pair ALC of candidate level rules
gives, against W20: the Bayes rule under the posterior-mixture prior $-1.37$ (SD 0.99 across regime cells), the
oracle-cell Bayes rule $-2.04$, the best normal prior $-0.39$ under Laplace ($m_0 = -1.5$, total level variance 4.5;
28% of the bound) and $-0.49$ under exact inference ($m_0 = -1.25$, variance 2.69; 35%), and the two-component prior
MXs $-1.25 \pm 0.50$ (92%; probability 0.82 of beating W20 by more than 1 on the base posterior, 0.13 without the B31
constraint), so the normal family is nearly exhausted. MXs passed
switch item (f) on its local guard (`results/hr_mix_guard.csv`: macro 0.1892 against 0.1921 for REF; worst held-out
benchmark, Multi-SWE-bench, $+2.6$; the composite's additivity within $\pm 0.3$) and was pre-registered at 05:10 UTC
(Table D3) as an exploration arm: three uploads a day outside the design tool, the usual screen, an allocation-only stop
when two consecutive evening posteriors give P(H_pub) below 0.5, its contrasts exploratory, and the composite
`v8_mxs_combo` if it ships; with at least 30 runs and no drop it is a finalist candidate at the final look, which the
operating-characteristics simulation (Table D5, seven fixed arms) did not include. The three arms toward another
team's wider pair prior, $(m_0, \tau_\mu, \tau_\delta) =
(-2.0, 2.0, 1.4)$, $(-2.5, 2.5, 1.4)$ and $(-2.5, 2.5, 2.4)$, score $-1.2$, 0.0 and $+0.7$ against REF on the local
guard draws (`results/r8_dlike_guard.csv`), all within item (f); the grid was not amended for $\tau_\delta = 2.4$ and
the off-grid arm was not started, because the calculus puts the three at $+0.7$, 0.0 and $+0.9$ against W20 and bounds
any normal-family arm at about $-0.4$ against it. W20's selection-set contrast after its sixth run was $-4.9 \pm 3.4$
($z = -1.46$). A red-team pass over zero-run decay, slips, SE floors, a guess floor, $\lambda$ and recalibration found
no lever worth 2 and one real bias, the late-budget double moderation of near-zero pairs (about $+0.05$ at B31), whose
pre-registered fix scored $-0.09 \pm 0.04$ against a required $-0.15$ and was not adopted (Appendix G, item 20).

*2026-10-08 to 2026-10-20.* The design was refit after every upload day (Table D3). P(H_pub) stayed between 0.88 and
1.00 (0.98 on 2026-10-20 with $E[d] = 0.42$; 1.00 without the B31 constraint and 0.99 without the connector arms), so
MXs kept its three daily uploads. Two grid arms started after passing their local guards, W15d6 on 2026-10-08 and
W15m16 on 2026-10-10. The leading posterior gains fell from about $-5$ after 2026-10-07 to about $-2$ on 2026-10-20
(W15m16 $-2.13 \pm 1.01$, W20 $-2.02 \pm 0.68$, L15 $-1.76 \pm 0.69$, W20 at $m_0 = -1.5$ $-1.45 \pm 0.98$; REF's
probability of being best rounds to 0.000). The leader's probability of being best among the shortlisted arms and REF
was 0.47 at checkpoint A (the interim look for 2026-10-14) and 0.36 at checkpoint B (19:45 UTC on 2026-10-19; 0.38 at
03:26 UTC on 2026-10-20), below the 0.80 an interim commit needs, so no commit was made. The plan stopped allocating to
W15d6 at 13 runs, to WLB at 14 and to W20 at $m_0 = -1.5$ at 35 (stops, not drops, so the last remains a finalist at
the final look), and the screen dropped no arm after 2026-10-06. The design's preposterior probability that no arm
passes on 2026-10-30 was 0.10 on 2026-10-12, 0.03 to 0.07 from 2026-10-13 to 10-19 and 0.13 on 2026-10-20 (the design
does not model MXs). The plan for 2026-10-21 gives REF 11, L15 7, W15m16 5, W20 4 and WLA 3 of its 30 uploads, beside
three MXs uploads, the two insurance runs and the rerun of the W15m16 composite check; its expected true effect of the
shipped arm is $-1.99$, against $-1.91$ for the fixed rotation of REF, W20 and L15.

**Live state on 2026-10-20 in full.** Table D6 lists every level arm of the daily report of 03:24 UTC on 2026-10-20.
Our 401 scored runs (103 of REF) cover 302 distinct pairs; with the 54 connector logs there are 303 distinct pairs, and
38% of the 3,943 appearances share their benchmark with another pair of the draw. W20 at $m_0 = -1.5$ is not a
provisional finalist but has 35 runs and no drop, so five arms (W20, W15m16, MXs, L15 and W20 at $m_0 = -1.5$) would
count as finalists at the final look. In the daily reading, with the SD floor, item (b) fails for every arm, and W20 and
W15m16 fail nothing else. The mechanism readout of W20 is split: its B1 effect supports widening ours only and does not
with connectors. The one-run composite platform checks are not listed; their contrasts decide nothing.

**Table D6.** The live state on 2026-10-20 (03:24 UTC): ALC contrasts against REF on the selection set ($\pm$ SE$_h$
with 500 bootstrap replicates; ours only with $z$, and with connectors), the screen, and the switch checklist as read
daily with the SD floor ((a) on the holdout, (b) to (e) on all pairs, with the primary estimator; four provisional
finalists, so (b) needs $z \le -1.8$). $K$: pairs matched directly with REF on the selection set. Table D1 has the
2026-10-06 values behind the drop decisions. Script: `tools/daily_report.py` (output
`swarm/round11/daily_report_1020.txt`).

| Arm | Runs | $K$ | ALC, ours | $z$ | ALC, connectors | Screen | All pairs, ours | $z$ | Items failed |
|-----------------------|------|-----|------------|-------|-----------------|----------|-----------------|-------|------------|
| W20 | 87 | 141 | -2.0 ± 1.2 | -1.60 | +0.6 ± 1.5 | finalist | -1.7 ± 1.1 | -1.57 | (b) |
| W15m16 | 39 | 110 | -2.1 ± 1.6 | -1.28 | +0.5 ± 1.6 | finalist | -1.7 ± 1.4 | -1.22 | (b) |
| MXs | 30 | 98 | -1.6 ± 1.5 | -1.07 | +0.1 ± 1.8 | finalist | -0.7 ± 1.3 | -0.53 | (a), (b), (d) |
| L15 | 43 | 105 | -1.5 ± 1.4 | -1.04 | +0.4 ± 1.7 | finalist | -0.8 ± 1.3 | -0.67 | (a), (b) |
| W20 at $m_0 = -1.5$ | 35 | 103 | -1.2 ± 1.8 | -0.66 | +0.0 ± 1.7 | continue | -0.2 ± 1.5 | -0.15 | (a), (b), (d) |
| WLB | 14 | 64 | -0.9 ± 1.8 | -0.49 | +0.7 ± 2.2 | continue | -0.6 ± 1.6 | -0.35 | (a), (b) |
| WLA | 16 | 63 | -0.7 ± 1.9 | -0.36 | -0.1 ± 2.3 | continue | -0.8 ± 1.6 | -0.46 | (b) |
| W15d6 | 13 | 63 | +1.4 ± 2.0 | +0.69 | +1.1 ± 2.2 | continue | +1.9 ± 1.7 | +1.15 | (a) to (d) |
| $m_0 = -2.5$ | 3 | 17 | +3.9 ± 5.1 | +0.77 | +9.5 ± 6.4 | dropped | +3.1 ± 4.4 | +0.71 | all but (e) |
| Mixture, $\pi = 0.25$ | 2 | 12 | -3.1 ± 4.2 | -0.73 | +0.2 ± 6.6 | dropped | -2.5 ± 3.7 | -0.69 | (b) |
| Mixture, $\pi = 0.35$ | 1 | 8 | -0.1 ± 7.0 | -0.01 | +5.8 ± 8.9 | dropped | -0.3 ± 6.4 | -0.05 | (b) to (d) |

**Final-look rehearsal.** On the data of 2026-10-16 (347 of our runs, 84 of REF; 300 bootstrap replicates) the
final-look mode of the decision tool ran cleanly. It counts as finalists every non-dropped arm with at least 30 runs
(then four: L15, W20, W15m16 and W20 at $m_0 = -1.5$), drops the $\mathrm{SD}/\sqrt{K}$ floor in items (b) and (e) for
arms with at least 60 direct matches, and adds the confirmatory fresh-pairs estimate. W20 passed every item: holdout
$-0.2 \pm 2.6$, all pairs $-1.8$ with $z = -2.17$ without the floor ($K = 157$), benchmark-balanced $-2.4$, largest
leave-one-benchmark-out estimate $-0.7$, no budget worse than REF by two SE$_h$, and the local guard. No other arm
passed (b) ($z$ from $-1.05$ to $+1.21$). The fresh pairs, first seen in our run 965689 or later and in no public log,
numbered 81 (221 appearances, 56 of them REF's), so each finalist had 11 to 15 direct matches with REF on them; all
four fresh-pairs contrasts were negative and within 1.71 SE of zero. These are rehearsal outputs on data four days old;
the final look of 2026-10-30 reached the same verdict (Section 7.4).

**Composite builds and platform checks.** Composite builds exist for the REF, W20 and L15 blocks (2026-10-06), for W20
at $m_0 = -1.5$, W15d6 and MXs (2026-10-07) and for W15m16 (2026-10-15), with their SHA-256 in
`submissions/builds.csv`; WLA and WLB have none. On the platform the composite on W20 finished on 2026-10-07 (965689).
The early checks of the readiness amendment moved from the 2026-10-18 window to 10-19; those on REF (1003393) and W20
at $m_0 = -1.5$ (1003392) finished, and the one on W15m16 (1003391) was still Running after 11 hours, as 13 plain runs
of that day were still Running, Preparing or Scoring, so it counts as a run that died before scoring and is replaced
once, by item 13 of the 2026-10-21 plan (1006755), which also died before scoring (stuck in Preparing). The composites on L15 and MXs have not run on the platform. The
insurance builds `v6_combo_w20_c1` (SHA-256 prefix e15a4e3f) and `v6_combo_m150_mu16_d14_c1` (f0671ff2), built at 19:46
UTC on 2026-10-19, hold their parents' `model.py`, parameters and subject prior unchanged plus `prediction_config.json`
with `{"concurrency": 1}`, which per the kit's README makes prediction sequential instead of using up to four workers;
`tools/verify_shipped.py` reproduces the parents' predictions exactly and the kit's zip checks pass. Their platform
checks, items 2 and 24 of the 2026-10-21 plan, both finished (1006743 and 1006768), and the plan of 2026-10-30 added
three guard runs of each, of which two finished and one was still in flight at the final look (for W20's build, 1020286
and 1020301 finished and 1020313 counts as dead).

**Platform operations.** From 2026-10-07 to 10-16 the platform ended 77 of our 355 uploads without a Finished status
(27 Failed-with-scores, whose complete scores we use, 7 Failed without scores and 43 dead or stuck in Submitting,
Submitted, Preparing, processing, Running or Scoring), in every arm including REF, and on 2026-10-19 a further 14 runs
hung for more than 11 hours. Server errors during uploads and queue backlogs of up to 90 minutes were frequent, and
some days started late or ended early when the laptop that runs the uploads slept or lost power. On 2026-10-17 between
18:07 and 18:11 UTC, 529 submissions of our own built zips were created in our account without an upload by us. They
stayed in Submitted or Preparing, were never scored, never entered our ledger and were removed in ID order (189 by
08:44 UTC on 2026-10-19, all by 16:29 UTC). The treatment rule above was hashed at 05:41 UTC on 2026-10-19, before any
could be scored, and never applied. Uploads paused from 05:41 to 16:29 UTC on 2026-10-19, while the queue did not move
and these runs might count toward the per-person cap of 1,000 (924 counted at 05:41 UTC); the day's plan then ran with
the three composite checks first. The windows of 2026-10-17 and 10-18 were missed: our upload session was down from
about 15:00 UTC on 2026-10-16 and the platform had signed the browser out, so nothing of the 10-17 plan was sent. By
2026-10-20 we had used 467 of the 1,000 uploads (at most 36 a day by our own cap, under the platform's 50), with at
least 45 reserved for 2026-10-27 to 10-30 (composite guards, a Finished run of the final build, replacements). The
upload session was down again from 18:41 UTC on 2026-10-21 to 05:10 UTC on 10-28 and on 2026-10-29, so we missed the
windows of 2026-10-22 to 10-27 and of 10-29. On 2026-10-28 we uploaded 33 REF and 3 MXs runs, and on 2026-10-30, the
last upload day, 10 runs each of REF, W20 and W15m16 and three guard runs of each insurance build. In all we used 575
uploads.

# Appendix E. Regime test details

Predictions use validated tables $q(k, n)$ for every arm: our arms through the real model classes, and the two other
teams' predictors through replicas of their public code. Hypotheses transform the 220 public pair rates by a logit shift
($-1$ to $+0.75$), a logit spread factor (0.6 to 1.5) and a share of pairs per benchmark (0 to 0.45) moved to a
logit-normal low component centred at 6% or 1.5% (SD 0.5), for 315 grid points. Item groups are bracketed by no group
and by one group confounded with the level. The prediction is the bracket mean and half the gap is added as discrepancy
variance (floor $0.5 \times 10^{-3}$). The observations are the 13 contrasts, re-referenced to the best-covered public
arm, at four budgets from the estimator with connectors, with a 600-draw joint stratified pair bootstrap covariance
shrunk 50% toward its diagonal. These observations come from the round-2 fit, whose connector set includes the one
public log that the pre-registered set excludes (Table D1); dropping that log moves the re-referenced contrasts of the
13 arms by at most 6.2 at B0, 11.4 at B1, 4.9 at B3 and 1.8 at B7, almost all of it on two arms. Rerun on the
pre-registered connector set (the same 22 own logs and the 54 public logs of that set, 666 pair results on 209 pairs,
same bootstrap seed; `swarm/round10/regime_rerun/`), the test gives the same verdicts: $\Delta\chi^2$ 73 for R1, 90 for
R2, 9.9 for R3 and 30 for the unchanged public pool (last column of Table E1). The best grid point moves to a neighbour
(shift $+0.5$, spread $\times 1.2$, 20% of pairs near 6%, which the round-2 fit put at $\Delta\chi^2$ 1.8; the round-2
best point sits at 1.3 in the rerun) with the same pool summaries (mean rate 0.38, mean $r(1 - r)$ 0.148, 13% of pairs
below 5%), its fit is 83.1 against 83.8, and the
benchmark-balanced B31 mean is unchanged (0.1605). The parametric bootstrap of Table E2 was not repeated. The global
B31 constraint compares the calculus B31 (minus a 0.0114 offset for the
31-label estimation error, calibrated on public replays) with the benchmark-balanced live mean B31 over distinct pairs,
with 0.005 of extra SD for the unknown item-information bonus; only the benchmark-balanced mean enters, and no
per-benchmark value is fitted or reported. Without the B31 constraint the fit runs to the most dispersed corner of the
grid; the constraint is what bounds the dispersion.

**Table E1.** Regime test of Section 7.2: the baseline, six variants and the rerun on the pre-registered connector set
($\Delta\chi^2$ from the best grid point of each column). Shape only: B1 to B7 contrasts and the B31 constraint; B0
only: budget-0 contrasts and the B31 constraint; public arms only: contrasts among the connector arms; spread trial
only: the six contrasts of that trial; pre-registered set: the baseline without the one public log outside that set. A
seventh
variant, not shown, keeps only our own arms (with connectors): it fits R1 ($\Delta\chi^2$ 1.4) and rejects R2 (51 to
82), because our sparse arms cannot discriminate shapes; its R3 and public-pool values were not recorded. Also not
shown are three neighbours of the named regimes (baseline $\Delta\chi^2$): R1 with spread $\times 0.7$ (72), R2 with
45% of pairs near 1.5% (114) and R3 with shift $-0.5$ (22).

| Hypothesis | Baseline | SEs × 1.5 | Shape only | B0 only | Public arms only | Spread trial only | Holdout pairs dropped | Pre-registered set |
|-------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| R1 narrow core | 63 | 28 | 71 | 6.0 | 24 | 37 | 47 | 73 |
| R2 zero-inflated | 78 | 61 | 80 | 62 | 39 | 2.7 | 86 | 90 |
| R3 wide | 13 | 4.5 | 7.8 | 19 | 3.3 | 12 | 5.9 | 9.9 |
| Public pool unchanged | 34 | 18 | 35 | 17 | 23 | 31 | 23 | 30 |

In the spread trial (arms of one team that share code, subject prior and budget-0 predictions and differ only in how
fast the level moves; 14, 14 and 15 draws) the observed contrasts against the base arm are, at B1, B3 and B7, $-7.5 \pm
4.1$, $-2.0 \pm 3.0$ and $+0.1 \pm 1.3$ for the faster arm and $+1.3 \pm 4.0$, $+1.9 \pm 2.5$ and $+2.5 \pm 1.1$ for the
slower arm (raw bootstrap SEs). R1 predicts $+16.1$, $+7.4$, $+2.4$ and $-3.5$, $-2.3$, $-1.0$, the unchanged public
pool predicts $+14.8$, $+7.0$, $+2.4$ for the faster arm, and heavy zero-inflation predicts $-2.2$, $-2.0$, $-0.9$. The
subject-calibration nuisance of Section 7.3 scales and shifts the ability of our arms and of the attribute-based
competitor. The best fit has an offset of 1.5 to 2 logits with an unidentified slope, and lowers the total $\chi^2$ from
83.8 to 55.1 (83.1 to 56.8 on the pre-registered connector set, with the same best cell). Under every calibration, R1
stays at $\Delta\chi^2 \ge 40$ and R2 at $\Delta\chi^2 \ge 59$ (43 and 57 on the pre-registered set). Against
$\chi^2$ tails, 83.8 on 53 degrees of freedom gives $p = 0.004$ and 55.1 on about 51 gives $p = 0.32$. These degrees of
freedom subtract the two nuisance parameters but not the three or four fitted grid parameters; with those, $p$ is about
0.001 to 0.002 and 0.20 to 0.22, and the conclusions do not change. The parametric bootstrap below needs no degrees of
freedom, because it refits the grid in every replicate.

**Parametric bootstrap.** For each null hypothesis we generated 2,000 data sets from its predicted contrasts with the
covariance the statistic assumes, refitted each over the whole grid (plus the null itself) and recomputed
$\Delta\chi^2$; a sensitivity run used the unshrunk bootstrap covariance as the generator. Because the best point itself
misfits, a second reference divides every observed $\Delta\chi^2$ by the dispersion $c = 83.8/50.0 = 1.68$ (the
observed minimum over the median minimum of data generated at the best point; 300 replicates, so the smallest attainable
$p$ is 0.0033). $p < 0.001$ means that no replicate reached the observed value.

**Table E2.** Parametric bootstrap of the regime test (Section 7.2). Statistic: $\Delta\chi^2$, except the two fit rows
(minimum $\chi^2$ against data generated at that point) and the last row ($\Delta\chi^2$ between the best cell with the
public calibration of our subject prior and the best cell overall). Script: `swarm/round6/oc_sims/regime_boot.py`.

| Null hypothesis | Observed | Reference median / 95% / 99% | $p$ | $p$, unshrunk generator | $p$ after scaling by $c$ |
|------------------------------|-------------|------------------------------|---------|---------|---------|
| R1 narrow core, spread × 0.8 / × 0.7 | 63.1 / 72.0 | 1.9 / 6.2 / 8.8; 1.4 / 4.8 / 9.3 | < 0.001 | < 0.001 | $\le$ 0.003 |
| R2 zero-inflated, 35% / 45% near 1.5% | 78.3 / 113.9 | 2.6 / 8.9 / 12.3; 2.2 / 8.8 / 13.3 | < 0.001 | < 0.001 | $\le$ 0.003 |
| R3 wide, spread × 1.3 | 13.0 | 1.8 / 7.5 / 11.0 | 0.005 | 0.0045 | 0.066 |
| R3 wide, spread × 1.3, shift $-0.5$ | 21.8 | 1.7 / 7.3 / 10.7 | < 0.001 | < 0.001 | 0.0066 |
| Public pool unchanged | 33.9 | 1.5 / 7.9 / 11.4 | < 0.001 | < 0.001 | $\le$ 0.003 |
| Fit of the best grid point | 83.8 | 50.1 / 67.7 / 77.2 | 0.0035 | < 0.001 | |
| Fit of the best H_cal cell | 55.1 | 51.0 / 69.3 / not recorded | 0.35 | 0.068 | |
| Exact public calibration (offset 0, slope 1), within the offset family | 28.7 | 1.2 / 5.3 / 8.0 | < 0.001 | < 0.001 | |

R1, R2 and the unchanged public pool are rejected under every reference, as is R3's neighbour with shift $-0.5$; R3
itself is inconsistent with the data at $p = 0.005$ but not once the overall misfit is allowed for. Within this model
family the exact public calibration of our subject prior (offset 0, slope 1) is rejected against the offset family;
Section 7.4's H_pub is the wider set $d \le 0.85$ in a model with a discrepancy term, and Section 7.4 shows how much
the rejection rests on the B31 constraint and the connector arms. On the pre-registered connector set, leaving out any
one of the 13 arms keeps R1 at $\Delta\chi^2$ 40 to 84 (lowest without the faster spread-trial arm, then 52 without
$m_0 = -1.5$), R2 at 70 to 99 and R3 at 7.2 to 11.8; the unchanged public pool stays at 26 to 38 except without the
faster spread-trial arm (10), which carries most of the evidence against it
(`swarm/round10/regime_rerun/loao_prereg.csv`). The predicted against observed values of the 52 contrasts at the best
grid point (`pred_vs_obs_best_prereg.csv` and `.png` beside it) have $|z| \le 2.65$, with three arms above 2: the
faster spread-trial arm, REF and the 35% mixture.

# Appendix F. Cavity pooling details

**Why one item variance under-pools (Section 8).** In the joint Laplace fit $\sigma_i$ plays three roles. It is the
prior of an item effect for each of the target's own labels (a large value lets them explain away the pair's
evidence), it moderates every unlabelled evaluation item, and it is the only channel through which a partner's label
on the target's item moves the prediction. The first two roles punish large values on every pair, while the third pays
only on rows a partner labelled.

**Table F1.** Logit shift of the target's prediction on an item a partner (pair rate 30%) labelled, relative to an
unlabelled item, after $n$ own labels. Exact: brute-force grid over the level, pair deviations and item effect. Scripts:
`why_underpool.py`, `unit_checks.py`, `shift_check.py` in `swarm/round2/pooling_sims/`.

| $n$, partner's label | $\sigma_i = 0.6$ (REF) | $\sigma_i = 1.0$ | $\sigma_i = 2.0$ | cavity, $\sigma_x = 2.0$ | cavity, $\sigma_x = 2.5$ | exact, item SD 2 |
|----------------|----------|----------|----------|----------|----------|----------|
| 7, success | +0.17 | +0.39 | +0.85 | +0.78 | +0.92 | +0.89 |
| 7, failure | -0.07 | -0.18 | -0.41 | -0.44 | -0.53 | -0.50 |
| 31, success | +0.20 | +0.44 | +0.93 | +0.91 | +1.09 | +1.03 |
| 31, failure | -0.09 | -0.20 | -0.48 | -0.51 | -0.64 | -0.57 |

**Tuning and double counting.** In a coordinate search around a later base ($m_0 = -1.2$, $\lambda = 1.6$, Student-t
level prior), $\sigma_i = 0.6$ and 1.2 tied (macro 0.1988 and 0.1987) and larger values lost (0.1996 at 1.8, 0.2015 at
2.4). At 0.6 the implied item intra-class correlation is 0.10, against 0.66 to 0.75 in the public data. A partner's
level read from the full fit already contains its label on the target item: after one partner success it is $-0.62$ in
the full fit against $-2.12$ with the label removed and $-2.13$ from an exact refit. Sequential plug-in fits used by
public competitors share this double counting.

**Cavity algebra.** The model removes all labels on the target's item text from the Laplace posterior by a rank-$m$
Woodbury downdate (within 0.02 logit of an exact refit), maps each labeller's cavity level to the item scale,
integrates the shared item effect by Gauss-Hermite quadrature and adds the resulting logit shift to the base
prediction. Let $J$ be the labels on the target's item text, $X_J$ their rows of the design, $p_J$ the fitted
probabilities, $h = p_J(1 - p_J)$, and, for a linear combination $a$ of effects, $t = X_J\Sigma a$. The cavity mean is
$a^\top w - t^\top M^{-1}((y_J - p_J)/h)$ and the cavity variance is $a^\top\Sigma a + t^\top M^{-1} t$, with $M =
\mathrm{diag}(1/h) - X_J\Sigma X_J^\top$. If $M$ is not positive definite, the model falls back to the full fit. Each
labeller's level (benchmark, subject, pair and group terms without the item effect) is read on its own labelled copy of
the text and mapped to the item-level scale with $\kappa = (1 + \pi\sigma_x^2/8)^{-1/2}$. Its likelihood term is
$\sigma\big((m_j/\kappa + u)/\sqrt{1 + \pi v_j/(8\kappa^2)}\big)$ for a success, with $u \sim N(0, \sigma_x^2)$ on a
40-node Gauss-Hermite grid. The prediction is $\sigma(\operatorname{logit}\hat p_{\mathrm{base}} +
\operatorname{logit}\hat p_{\mathrm{item}} - \operatorname{logit}\hat p_{\mathrm{prior}})$, and the zero-inflated
mixture, if used, is applied afterwards. Items are matched on a SHA-1 hash of the full text, because the base model's
head-and-tail key collides on 3 of 212 ResearchCodeBench items, and the cache key includes the target's feature group,
so predictions do not depend on call order.

**Covered rows.** On covered rows the base model under-reacts after a partner's success on every benchmark, and after a
failure on three of four. After a partner's failure the target succeeded on 9.8% of ResearchCodeBench items (base 21.3%,
cavity 18.1% at $\sigma_x = 2$), 26.4% on Multi-SWE-bench (base 40.8%, cavity 31.2%) and 35.4% on MathArena (base 45.3%,
cavity 36.1%), but 21.9% on REAL (base 21.5%, cavity 18.0%), where a partner's failure is weakly informative.

**Favourable partners.** Quick checks that paired each target with a partner sharing at least 80% of its items on
Multi-SWE-bench and MathArena, which overstate per-row gains, gave $-0.80 \pm 0.24$ (ResearchCodeBench), $-0.85 \pm
0.26$ (Multi-SWE-bench), $-1.23 \pm 0.30$ (MathArena) and $+0.03 \pm 0.24$ (REAL) per pair at $\sigma_x = 2.0$, and
$-0.93 \pm 0.32$, $-1.04 \pm 0.35$, $-1.42 \pm 0.39$ and $+0.19 \pm 0.33$ at $\sigma_x = 3.0$. With four pairs per
benchmark the ResearchCodeBench effect is $-1.9 \pm 1.1$ (two draws, $\sigma_x = 2.5$), and in the dense-draw replay of
Section 8, with random partners, C alone gives $-2.64 \pm 0.48$ there ($\sigma_x = 3.0$, ten draws; $-5.44 \pm 0.54$
with eight pairs). With complete overlap a partner holding 31 labels has labelled about $31/N$ of a pair's evaluation
items (28% for the median $N = 112$), the coverage behind the projection of Section 8, which makes that projection an
upper bound.

# Appendix G. Negative results in full

The twenty negative or neutral results, by grade (A: paired replay or live contrast with an SE; B: exact calculation
or simulation; C: reasoning only).

1. **Shared level ($\tau_M$; A).** A leave-two-benchmarks-out replay costs at most $+0.6$, since public levels differ at
random. Live, the five shared-level arms are $+11.2$, $+4.0$, $+24.8$, $+6.7$ and $+3.1$ with connectors (each $\pm 6$
to 9), $-0.3$ to $+11.4$ with the pair fixed effect with connectors, and $-5.0$ to $+11.8$ ours only. The original
analysis, with SEs later found to be too small, reported $+10$ to $+12$ with B1 penalties of $+26$ to $+34$, and its
pooling-aware rerun gave $+4.2$ and $+10.3$ (B1 $+24$ and $+27$) for the two variants on top of $\tau_\mu = 1.2$.
2. **Student-t(3) level prior (A, B).** Replays gain $-0.15 \pm 0.07$ at $m_0 = -1.5$ and $-0.41 \pm 0.07$ at $-2.0$,
all at B0. Live it is $+5.9 \pm 5.6$ with connectors and $+1.0 \pm 7.1$ ours only. At budget 0 the coded
variance-matched probit predictive gives 0.308 at $m_0 = -1.5$ against 0.288 for the exact t(3) predictive and 0.271 for
the normal prior.
3. **Global recalibration (A).** A per-budget map $a\,\mathrm{logit}\,\hat p + c$, cross-fitted over held-out benchmarks
on an earlier configuration ($m_0 = -1.0$, t prior), moved Brier from 0.2171 to 0.2334 at B0, 0.2012 to 0.2183 at B1 and
0.1904 to 0.1990 at B3, with gains below $10^{-3}$ at B15 and B31: the moderated predictive is already close to
calibrated, and a global map does not transfer across benchmarks.
4. **Exact inference (A, B).** For alone pairs a 1-D quadrature predictor matches independent quadrature within 0.002,
and a cavity version for shared pairs lands within 0.002 to 0.005 of a 2-D exact answer. On a quick $k = 1$ replay (28
alone pair-draws) it lost $+2.48 \pm 2.66$ (MathArena), $+1.64 \pm 1.53$ (Multi-SWE-bench), $+1.28 \pm 1.77$ (REAL) and
$+1.99 \pm 1.23$ (ResearchCodeBench), with B0 changing by up to $-2.4$ and B1 losing $+2.1$ to $+6.3$. The calculus
gives $+1.29$ for alone pairs on the public pool and $-0.6$ to $+0.7$ under the hypotheses that Section 7.2 does not
reject. Exact inference loses because the Laplace under-reaction acts as shrinkage toward a level prior that is not
centred for every benchmark (1 of 15 gives 0.086 exact against 0.119 Laplace), so it would pay only after a joint
re-tuning of the level block.
5. **Likelihood tempering (B, A).** With a unimodal prior facing bimodal synthetic pools, regret at $\alpha = 0.5 / 1.0
/ 1.5$ was 9.31 / 3.31 / 1.74, 7.44 / 3.04 / 2.47, 12.80 / 5.16 / 2.58 and 5.91 / 2.28 / 2.14, so $\alpha$ of 1.25 or
1.5 lowers regret in all four scenarios (at $\alpha = 2$ one scenario ties $\alpha = 1$ and one is worse, 2.98 against
2.28; the best values, 1.58 to 2.47 against 2.28 to 5.16 at $\alpha = 1$, lie between $\alpha = 1.25$ and 2).
$\alpha = 1.3$ gains $-0.5$ to $-1.1$ only where $m_0$ is centred too low and loses $+0.4$ to $+1.1$ on the
narrow-core pools that matched our round-1 global summaries, a regime Section 7.2 later found inconsistent with the
data; under the surviving hypotheses it has not been evaluated. Its public $k = 1$ replay is $+0.47$ (MathArena
$-2.17 \pm 1.86$, Multi-SWE-bench $+2.18 \pm 1.13$, REAL $+0.27 \pm 1.10$, ResearchCodeBench $+1.61 \pm 0.95$).
6. **Zero-inflated mixture (A, B).** Tables C2 and D1 and Section 7.2.
7. **Ability-tilted mixture weight (A).** $-0.72 \pm 0.28$ against the failed untilted mixture (Table C2).
8. **Narrow pair priors (A, B).** d06 and all taus $\times 0.8$ (Table C2) cost from B3 on and win only under the narrow
core; d06's live upload never left the platform queue.
9. **Prior-fitted networks (C).** Trained on simulations from our prior, a transformer can at best reproduce the Bayes
rule the quadrature already computes [12]. Trained on public replays (four eligible benchmarks), it would learn the
public pool, below any task-diversity threshold for in-context learners [13]. Its numpy forward pass takes 0.3 to 0.6 ms
per call, and one training run would take 1 to 10 CPU-hours plus days of engineering.
10. **Stacking (B).** With hindsight-optimal per-benchmark weights, combining REF with smoothed, pooled or
empirical-mean predictors gains 0.0 at B0 to B7 and $-0.1$ at B15 and B31 on three of four benchmarks [14]. The whole
$-3.3$ oracle gain is MathArena at B0 to B3, before labels can inform weights. A within-draw stacker sees 9 to 27 labels
at B1 to B3.
11. **Text and item content (B, C).** Within-benchmark rank correlations between log length and item easiness are
$-0.38$, $-0.025$, $-0.25$ and $+0.10$, at most $|0.20|$ after groups. Other teams reported a 14B-model rubric at $-0.84$
and a text residual at $+0.52$ (the latter from lucaluo925's MIT-licensed repository, whose authors later withdrew it
as not established by their one shuffled-text control), and the literature finds cross-benchmark AUC of 0.57 to 0.64,
rising to 0.67 to 0.74 only with gold patches, tests and repository state [16]. All of this evidence rests on four to
six public benchmarks, so we read it as no transfer at that scale, not as a general law.
12. **Acquisition (B).** In a synthetic study that mirrors the protocol (3,000 pairs per cell, common random numbers),
uncertainty sampling changed ALC by $-0.65 \pm 0.16$ in the base case and between $+0.03$ and $-1.73$ across eleven
scenarios, against 6 to 9 for getting shrinkage strength right, about 7 for cross-pair pooling and about 4 for
moderation [19]. Those gains need item-level signal: in the same study the base case assumes item features that
correlate 0.6 with difficulty, and the gain falls to $-0.05$ (n.s.) at 0.3 and vanishes at 0 ($+0.03$, every candidate
looks alike); with partner subjects on the benchmark, fresh-first or information-based acquisition gains $-0.09$ to
$-0.47$. Item content does not transfer across benchmarks here (item 11), so the 5 to 6 alone pairs of a draw sit near
the null case and we expect about $-0.1$ overall, below anything live feedback can resolve. We kept random acquisition,
also for the final submission: it makes live comparisons deterministic (Section 2), and a hook whose call fails fails
the whole run.
13. **Within-draw empirical Bayes (B).** Learning the hard-component weight from the nine pairs of a draw recovers at
most 0.1 to 0.15, even if the true weight varies across draws.
14. **Benchmark-level hard indicator (B).** An exact two-pair model with exchangeable hardness inside a benchmark gains
at most 0.3 on shared pairs at a compromise correlation of 0.3, and loses up to $+1.9$ if hardness is a pair property.
15. **Larger $\sigma_i$ and plug-in item offsets (A).** In the decisive replay (Table C4) $\sigma_i = 2$ gains at B0
through extra moderation ($-8.9$) and loses $+2.0$ to $+3.5$ at each budget from B3 to B31 ($+1.15 \pm 0.70$ overall,
REAL $+5.1$). Plug-in MAP item offsets with SD 2 tie the cavity in ALC ($-0.54 \pm 0.24$ against $-0.52 \pm 0.13$ at
$\sigma_x = 2.0$, REAL $-0.19$) but shift the logit by $+1.3$ to $+1.4$ after a partner's success against $+0.7$ to
$+1.0$ exact; an early 10-pair check had shown them losing on REAL ($+0.75 \pm 0.77$).
16. **Item-group variance $\sigma_g$ (A).** $+0.48 \pm 0.24$ at 1.4 and $+2.31 \pm 0.57$ at 2.0 (Table C2): B0 gains
through moderation, B1 to B15 losses. It also takes away a quarter of the cavity's gain on ResearchCodeBench, where
every paper is a group.
17. **Ability weight $\lambda$ (A).** $-0.76 \pm 0.16$ at 1.3 and $-1.40 \pm 0.31$ at 1.6, mainly from MathArena and
REAL at B0 to B3; not adopted (Section 4.6).
18. **SY1.0 (A).** Section 4.6, Tables C5 and C6, and the decision records in Appendix C.
19. **Provider-free attribute ridge, NP (A).** Section 4.6, Tables C5 and C6, and the decision records in Appendix C.
20. **Late-budget de-moderation (A).** On public replays the composite over-predicts near-zero pairs at late budgets
beyond an unbiased per-pair estimate (W20 + S + C + P bias at B31 on near-zero pairs $+0.050$, against $-0.001$ for the
31-label empirical mean), because the prediction for an evaluation item whose item and group effects carry no evidence
adds their full prior variance as probit moderation, pulling an extreme pair toward 0.5 a second time. A plug-in over
the composite that drops the prior variance of unseen item effects from the moderation once the pair has at least
seven own labels (prediction only; B0 to B3 unchanged by construction) was pre-registered with the rule macro at most
$-0.15$ on the draws of Table 2 (`results/rt8_unseen.csv`). Its best variant scored $-0.092 \pm 0.042$ against
W20 + S + C + P ($z = -2.2$; B7 $-0.11$, B15 $-0.20$, B31 $-0.30$; worst benchmark ResearchCodeBench $0.0$; $-0.10$ on
REF's block),
so the bias is real but worth under 0.1, the rule failed on magnitude, and the composite is unchanged.

# Appendix H. NeurIPS Paper Checklist

The questions are those of the NeurIPS 2026 paper checklist, abridged in items 4, 5, 7, 8, 11, 14, 15 and 16, answered
Yes, No or NA with a short justification; the guideline text of the official template is omitted for length.

1. **Claims.** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions
and scope? **Yes.** The abstract and Section 1 state the model, the composite's local gain with its SE, the live
methodology, the replay-subsampling bias and the negative results; the final configuration and score are
reported from the final look (abstract, Section 7.4, Table 6), and the limits of the live evidence are stated in Section 11.
2. **Limitations.** Does the paper discuss the limitations of the work performed by the authors? **Yes.** Section 11:
the level is not identified by public data, live power is weak, the summative subset's density is unknown, one
robustness rule was overridden, and most code was written by an LLM assistant.
3. **Theory assumptions and proofs.** For each theoretical result, does the paper provide the full set of assumptions
and a complete (and correct) proof? **NA.** The paper has no theorems. The exact calculations of Sections 7.2 and 8 and
Appendix F state their assumptions (random acquisition, exchangeable revealed labels, the model classes) and are
checked numerically against brute-force computations.
4. **Experimental result reproducibility.** Does the paper fully disclose all the information needed to reproduce the
main experimental results? **Yes.** Every arm's parameters (Appendix A), the replay design (Section 5), the estimators
and rules (Section 6, Appendix D) and the script and inputs behind each number (Appendix B); the submitted model
reproduces the research code bit for bit (Section 4.5).
5. **Open access to data and code.** Does the paper provide open access to the data and code, with sufficient
instructions to faithfully reproduce the main experimental results? **Yes, with one exception.** The code, the replay
scripts and their saved public-data outputs are released (Section 12; https://github.com/rishabhsinha17/brier-rabbit-paiec); the training data
are the organizers' gated public release at a pinned revision. Our per-pair formative logs and the files derived from
them are not released, as the competition's conduct rules require, so the live tables are reproducible at arm level
only.
6. **Experimental setting/details.** Does the paper specify all the training and test details necessary to
understand the results? **Yes.** Sections 3 to 6 and Appendices A, C and D give the data, splits, hyperparameters,
how each value was chosen (a public replay rule or a live rule) and the solver.
7. **Experiment statistical significance.** Does the paper report error bars suitably and correctly defined? **Yes.**
Local replays report draw-clustered paired SEs on identical draws; live contrasts report SE$_h$, the maximum of a
stratified pair bootstrap times 1.2, a cluster-robust sandwich and an SD floor, with the null calibration that
motivates it (Section 6, Appendix D); the regime test uses a parametric bootstrap (Appendix E).
8. **Experiments compute resources.** Does the paper provide sufficient information on the computer resources needed
to reproduce the experiments? **Yes.** One laptop CPU and no GPU (Section 13); per-call and per-draw timings in
Appendix A; replay jobs ran one at a time on the same machine, and a formative draw costs about 30 s per worker.
9. **Code of ethics.** Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of
Ethics? **Yes.** No human subjects and no personal data; the competition's conduct rules were followed (no attempt to
identify hidden benchmarks or subjects, no extraction of private test data, anonymous IDs as opaque keys only;
Sections 6 and 13).
10. **Broader impacts.** Does the paper discuss both potential positive societal impacts and negative societal impacts
of the work performed? **NA.** The work is a statistical method for predicting the benchmark outcomes of AI systems
from few labels; its foreseeable use is cheaper and better-calibrated evaluation, and we see no direct path to misuse
beyond those of evaluation research in general.
11. **Safeguards.** Does the paper describe safeguards for the responsible release of data or models that have a high
risk for misuse? **NA.** No such asset is released; the predictor is a small statistical model.
12. **Licenses for existing assets.** Are the creators or original owners of assets used in the paper properly
credited, and are the license and terms of use explicitly mentioned and respected? **Yes.** measurement-db (curation
under CC BY-SA 4.0, each benchmark under its upstream licence; [3], [28] to [33]) is used at a pinned revision and not
redistributed; the other teams' MIT-licensed logs and code are cited by repository and commit and not redistributed
(Sections 3, 6, 12, 13).
13. **New assets.** Are new assets introduced in the paper well documented, and is the documentation provided
alongside the assets? **Yes.** The released code (predictor, protocol replica, analysis scripts, build and
verification tools, tests) is documented in its README and in Section 12 and Appendix B; it is released under the
MIT license.
14. **Crowdsourcing and research with human subjects.** For crowdsourcing experiments and research with human
subjects, does the paper include the full text of instructions given to participants and details about compensation?
**NA.** No crowdsourcing and no human subjects.
15. **Institutional review board (IRB) approvals or equivalent for research with human subjects.** Does the paper
describe potential risks incurred by study participants and whether such risks were disclosed to the subjects and
approved? **NA.** No research with human subjects.
16. **Declaration of LLM usage.** Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? **Yes.** An LLM coding assistant wrote most of the code
and analysis and drafted the text under the author's direction (Sections 5 and 13); no LLM is part of the predictor,
which is numpy code without models, embeddings or inference APIs.
