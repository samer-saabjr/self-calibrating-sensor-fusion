# Verification record for Supplementary Material S2

Date of preparation: 2026-10-20. This file documents checks performed on the release package; it is not a claim of independent study replication or validation in new recording sessions.

## Reproduction checks completed

- `python code/reproduce_tables_1_3.py`: PASS; sample counts, all expert error/recall baselines, class-conditional means and sample standard deviations, and offline affine regression offsets/slopes/population residual variances reproduce Tables 1, 2 and 3 to published precision.
- `python code/reproduce_tables_9_10.py`: PASS; twelve learning arms and their paired frozen controls reproduce Tables 9–10 within displayed rounding precision; unexpected mismatches cause an assertion error.
- All six `analysis_*.ipynb` notebooks: PASS; executed from top to bottom on the packaged datasets using `nbclient` with a Python 3.13.5 kernel and the package versions in `requirements-analysis.txt`. There were no cell execution errors. The notebook-specific graphs are generated from the datasets.
- `python code/generate_figure_1_traces.py`: PASS; exports two Figure 1 trace panels from the released measurements and reconstructed annotations.
- CSV preservation check: all six original non-time binary measurement/event columns and all six original non-time counting measurement/event columns are numerically identical to the supplied original experimental CSVs; the publication CSVs contain 1851 and 2163 complete samples, respectively. Only the clock-time field is represented as elapsed seconds and appended non-measurement summary lines are omitted.
- Manuscript cutoff check: 9 binary cutoffs (200, 300, 400, 600, 800, 1000, 1250, 1500, 1600); 9 counting cutoffs (200, 300, 400, 600, 800, 1000, 1250, 1500, 1800). The minimum binary deployment error is 0.86% at cutoff 1500.

## What was not tested

- The acquisition notebooks were **not** run on physical Raspberry Pi hardware during this release check; they are included as original acquisition-source documentation.
- No additional recording sessions, hardware placements, or independent external datasets were collected or evaluated for this package.
- The regenerated Figure 1 files are equivalent data visualizations; plot styling may differ slightly from the final journal figures.

`SHA256SUMS.txt` contains file hashes for detecting accidental changes after extraction.
