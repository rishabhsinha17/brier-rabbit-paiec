# Paper 51 camera-ready corrections

The manuscript is `report/paper_final.md`. Build its PDF with `zsh report/build_pdf.sh`.
The original release is commit `8be5e86`. The camera-ready edits leave the original research model and selected
submission unchanged, so original results can still be reproduced. The corrected predictor is a separate artifact
in `camera_ready/corrected_w20/`.

From the repository root in the official starting-kit layout:

```sh
../.venv/bin/python camera_ready/verify.py
../.venv/bin/python camera_ready/inventory_audit.py
../.venv/bin/python -m pytest -q tests
../.venv/bin/python tools/verify_shipped.py submission_src/v6_combo_w20_c1
../.venv/bin/python scripts/rebuild_check.py v6_combo_w20_c1
```

Use the environment and data instructions in the root README. Verification uses only the five original public
training sources at revision `bc8204d8`. No paid API or GPU work is required. The replay ran in about 47 seconds
on this machine, excluding the initial diagnostics. Timings will vary.

The audit plan is in `PLAN.md`, raw results are in `results/`, and the review responses are in
`RESPONSE_TO_REVIEWS.md`. `build_corrected.py` makes two checked replacements in the frozen submitted model and copies
its parameter and subject-prior files unchanged. `inventory_audit.py` checks the saved official workbook, including
all cells and worksheets. It does not fit any expanded-data model.

The source inventory question remains unresolved. The author confirmed that the October 11/18 inquiry has received
no organizer clarification. Appendix J withdraws that study's supporting results and discloses the activity. An
organizer determination is still needed. This directory records the checks completed before camera-ready submission. The submission and revision
history are on [OpenReview](https://openreview.net/forum?id=5P6t3ancaQ).

Actual paid experiment spend for this revision: $0.
