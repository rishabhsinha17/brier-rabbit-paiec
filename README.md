# Brier Rabbit: code for the NeurIPS 2026 Predictive AI Evaluation Competition (PAIEC)

This repository contains the predictor we submitted to PAIEC 2026, our local replica of the platform's evaluation
protocol, the replays and analysis scripts behind every local number in our technical report, and the tools that build a
submission zip and check it against the research code. Technical report: `report/paper_final.pdf` (source
`report/paper_final.md`), also submitted on OpenReview. Selected Codabench submission: 1020301, the zip
`v6_combo_w20_c1.zip` (see "The submitted model"). License: MIT (see `LICENSE`).

Our predictor is a hierarchical logistic model that refits from the revealed labels at every call, with a benchmark
level, a subject offset whose prior mean comes from public abilities and an attribute ridge, a pair deviation,
item-feature group effects and item effects. A Laplace posterior gives a posterior-predictive output. Our final
composite adds three parts: a recalibrated attribute ridge for unseen subjects (S), cavity-corrected pooling of other
subjects' labels on the same item text (C), and name and harness pooling of the pair deviation (P). Python and numpy are
its only requirements.

## Contents

| Path | Contents |
|---|---|
| `paiec/` | data loader, protocol replica (`protocol.py`), replay helpers, baselines, the model (`hier.py`, `subjects.py`) and its plug-ins (`ext/`, with the final composite in `ext/combo.py`) |
| `submission_src/<name>/` | the exact files of every zip we built: `model.py` (generated, self-contained), `params.json`, `subject_prior.json` |
| `submissions/*.zip`, `submissions/builds.csv` | every built zip and its sha256 |
| `tools/` | `make_submission.py` (generate a submission), `build_submission.py` (zip it and run the kit checks), `verify_shipped.py` (submitted code vs research code), and the code of our live-analysis tools (`feedback.py`, `swarm_evalsci.py`, `daily_report.py`, `swarm_live_design.py`), which read formative logs that this repository does not include |
| `experiments/` | replays (`paired.py` does most of the work) and the early parameter search |
| `results/` | replay outputs (per-draw pair scores on public data), `JOB_ARGS.json` with the arguments of every queued replay, and the logs `results_*.log` |
| `swarm/` | analysis scripts cited in the report's Appendix B that use public data, synthetic inputs or `results/*.csv`, with their small text outputs |
| `synthetic/` | the synthetic acquisition and Brier-choice study |
| `audit/` | runtime and malformed-input checks of the submitted zips |
| `report/` | the technical report (`paper_final.md`, `paper_final.pdf`, built by `build_pdf.sh`), the table script and the generated tables |
| `scripts/` | `fetch_measurement_db.py` (pinned data download and checksum check), `rebuild_check.py` (byte-for-byte rebuild of submissions), `run_job.py` (rerun a replay) |
| `tests/` | unit tests for the protocol replica, the replay subsampling, the submission builder and the feedback parser |

## Not included

Codabench scoring logs are visible only to the submitter and the organizers, and they list anonymous subject and
benchmark IDs. We do not publish them or any per-pair file derived from them, and the report gives arm-level contrasts
only.

The report uses two teams' public MIT-licensed formative logs as reference arms. It cites their repositories at fixed
commits instead of copying the logs.

The starting kit has no license file, so we do not redistribute it. `paiec/protocol.py` imports two of its modules,
and `tools/build_submission.py` runs its zip check and smoke test.

Downloading measurement-db requires accepting its terms on Hugging Face, so you fetch it yourself (see Setup). This
repository includes only sha256 checksums of the files.

## Setup

### Layout

Code and kit sit side by side, as in our working tree:

```
KIT/                        the official starting kit, unpacked (Codabench competition 17828, Files tab)
KIT/tools/                  prediction_workers.py, streaming_ingestion.py, check_submission_zip.py, run_smoke_test.py
KIT/data/measurement-db/    the data (see Data below)
KIT/brier_rabbit/           this repository
```

`paiec/protocol.py` looks for the kit's `tools/` one directory above the repository, and `paiec/data.py` reads
`../data/measurement-db`.

### Environment

We used CPython 3.11.2 on macOS arm64:

```
cd KIT/brier_rabbit
python3.11 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Pin the BLAS threads on numpy builds that use OpenBLAS. The model solves many tiny linear systems, and with an unpinned
thread pool the same work took 11 times longer in our test (`audit/thread_check.out`). A submission's `model.py` pins
the threads itself, but that only works when it runs before the first numpy import, as it does on the platform.

### Data

We used measurement-db at revision `bc8204d8` (47 core files, about 83 MB). Accept the dataset's terms at
https://huggingface.co/datasets/aims-foundations/measurement-db while logged in, then run:

```
huggingface-cli login          # or export HF_TOKEN=...; the script itself never stores a token
../.venv/bin/python scripts/fetch_measurement_db.py            # downloads to ../data/measurement-db
../.venv/bin/python scripts/fetch_measurement_db.py --verify   # sizes and sha256 of all 47 files
```

### Tests

`../.venv/bin/python -m pytest -q tests` takes about 20 s and should report 25 passed and 1 skipped. The skipped test
needs the other teams' logs, which this repository does not include.

## The submitted model

The final candidates differ only in the benchmark-level prior (the "level block"). Their composite settings are
identical. The selected submission, Codabench run 1020301, is the insurance build of the W20 candidate: the files of
`v6_combo_w20` unchanged plus `prediction_config.json` with `{"concurrency": 1}`, which makes the platform predict with
one worker instead of up to four. Its predictions equal its parent's bit for bit.

| Zip | Level block | sha256 |
|---|---|---|
| `v6_combo_ref.zip` | m0 -2.0, tau_mu 1.2 | `38000a70c57526e502daa259c5b7501bf4b78cc3bc95ab4d70b81bf0a7f1e44d` |
| `v6_combo_w20.zip` | m0 -2.0, tau_mu 2.0 | `dc75058601cc693861d1f79c5e4417dea51f0c3cf408d291685002bea2eb06cb` |
| `v6_combo_l15.zip` | m0 -1.5, tau_mu 1.2 | `0a09fb9c3fcb280d832355be6f216edd44ee004f3729f35f6e10d6a693a9952f` |
| `v6_combo_w20_c1.zip` (selected, run 1020301) | m0 -2.0, tau_mu 2.0, as `v6_combo_w20` | `e15a4e3f63148c63306f6d32a4cb1229e0e81720b836fa1440d53094e35f806e` |

Each zip contains `model.py` (the model code vendored into one numpy-only file by `tools/make_submission.py`),
`params.json` and `subject_prior.json` (fitted on the public training benchmarks). The insurance build adds
`prediction_config.json`.

### Rebuild byte for byte

Each rebuild takes 45 to 60 s and needs the data:

```
../.venv/bin/python scripts/rebuild_check.py                     # the four zips above
../.venv/bin/python scripts/rebuild_check.py --all --kit-checks  # every build, plus the kit's zip check and smoke tests
```

For each name, `rebuild_check.py` regenerates the three files from source and data into `build_check/`, compares them
with `submission_src/<name>/`, zips them with the recipe of `tools/build_submission.py`, and compares the zip's sha256
with `submissions/builds.csv` and with the zip in `submissions/`. On 2026-10-06 all three candidates came out identical,
and on 2026-10-30 all four zips above did. The insurance build ships its parent's `model.py`. The script regenerates it
under the parent's name and copies the hand-written `prediction_config.json`.
Zip bytes also depend on the zip and zlib versions (we used CPython's zipfile on macOS with zlib 1.2.12), while the
per-member comparison does not. The v0 and v1 builds came from an earlier builder, and the script skips them.

### Check the submitted code against the research code

```
../.venv/bin/python tools/verify_shipped.py submission_src/v6_combo_w20
# [ok] v6_combo_w20: 2166 predictions, max |shipped - research| = 0.00e+00
../.venv/bin/python tools/verify_shipped.py submission_src/v6_combo_w20_c1
# [ok] v6_combo_w20_c1: 2166 predictions, max |shipped - research| = 0.00e+00
```

`verify_shipped.py` imports the submitted `model.py` the way the platform does (repository off `sys.path`, any `paiec`
import blocked), replays the protocol on public draws and requires agreement to 1e-12. It takes about a minute.

### Build a new submission

`tools/make_submission.py NAME --params '<json>'` writes `submission_src/NAME/`. Then
`tools/build_submission.py submission_src/NAME --note "..."` zips it into `submissions/`, runs the kit's checks with 1
and 4 workers, scans the files for credentials and appends a row to `submissions/builds.csv`.

### Runtime

We timed `v6_combo_w20` on synthetic draws shaped like formative draws, in one laptop process limited to efficiency
cores. A call takes about 19 ms at 279 revealed labels, and the first call of each checkpoint refits the posterior in
about 0.2 s. All six checkpoints of a draw need about 33 s of compute, and a worker stays under 100 MB. Predictions do
not depend on call order or on the split of calls across workers, and numpy 1.26 and 2.4 agree to 5e-16. Malformed
inputs return a valid probability instead of raising. The scripts and their outputs are in `audit/`.

## Rerun the local tables

`results/JOB_ARGS.json` records the command of every queued replay, and `scripts/run_job.py` reruns one of them. The
composite ablation behind Tables 2 and 3 takes about 13 minutes with two workers:

```
../.venv/bin/python scripts/run_job.py --list
PAIEC_WORKERS=2 ../.venv/bin/python scripts/run_job.py combo_v1 --out results/combo_v1_rerun.csv
```

Each replay seeds its draws, so a rerun on the same data and environment reproduces the saved CSV. We ran the replays
from before the job queue (`results/paired.csv` and the other early `paired_*.csv`, `tune_hier*.json`,
`lobo_baselines.json`, `l2o_shared_level.csv`) directly with `experiments/*.py`. `results/paired.csv` comes from
`experiments/paired.py` with its defaults (30 runs, k 1 and 2, four held-out benchmarks). The other early replays ran
before we recorded arguments, so only their logs (`results_*.log`) document them.

`swarm/round5/writer_sims/paper_numbers_v3.py` and `paper_tables_v3.py` read `results/*.csv` and print every local
contrast of the report. Their saved outputs are the `.out` files in the same folder. `fig_ladder_v3.py` draws Figure C1,
and Appendix B of the report maps each table to its script. Table 2 of `report/tables.md` and the report's live tables
come from our formative logs, which we do not publish.

## Data and licenses

Code license: MIT (see `LICENSE`). We trained only on the six public benchmarks of measurement-db
(`aims-foundations/measurement-db`, revision `bc8204d8`). Its curation layer is CC BY-SA 4.0, and some benchmarks have
their own upstream terms (MathArena CC BY-NC-SA, MMDocRAG CC BY-NC). `results/` holds per-draw scores of our replays,
keyed by measurement-db's public subject hashes. It contains no responses or item texts.

## Acknowledgments and disclosure

We thank the PAIEC organizers for the competition, its data and its platform. We also thank the GitHub users aghasalim
(`github.com/aghasalim/predictive-eval`) and doctawho42 (`github.com/doctawho42/PAIEC`) for publishing their code and
formative logs under the MIT license, which the report uses as connector arms, and lucaluo925 for MIT-licensed public
results it cites.

The author used Claude Code (Anthropic), an LLM coding assistant, to write most of the code and analysis and to draft the
report. The author reviewed both and takes full responsibility for them. Section 13 of the report gives the details.
