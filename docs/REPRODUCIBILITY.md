# Reproducibility and private execution

## What has and has not been verified
The supplied project logs record R 4.5.2 (Windows x86_64) and Python 3.12.14. Package manifests are inherited from that environment; this editorial release does not claim that a clean R `renv::restore()` or a full database-to-model rerun was performed here. Test reports distinguish synthetic execution, static checks, private path staging, and authorized-data reproduction. Source changes in this release concern safe launch paths, parameterized PostgreSQL connection settings, missing source inclusion, labels and publication metadata, not statistical definitions or numerical model settings.

## Safe entry points
`run_all.cmd` (Windows) or `sh run_all.sh` runs only the synthetic test. There is no marker that silently starts restricted-data analysis.

To create a private workspace, outside the cloned repository:

```text
python prepare_private_workspace.py --work-dir D:\CC_PRIVATE_REPRO
```

Use a new, empty folder. The launcher refuses an existing nonempty workspace and refuses paths inside or containing the repository. It copies analysis scripts into the directory structure assumed by the original code, but does not access databases or fit models. Do not execute legacy scripts directly in `src/`: those scripts construct outputs relative to their own locations.

## Authorized local reproduction
Obtain access to MIMIC-IV v3.1 and SICdb v1.0.8 directly from PhysioNet. Verify the installed schemas against the extraction SQL. The database names and columns in SQL must match the documented local schema. PowerShell extractors require PowerShell 7 and `psql` on PATH (or an explicit `-PsqlPath`). Set `-PgHost`, `-PgUser`, `-PgPort`; enter the password only at the secure local prompt. Never add credentials to source files or upload them.

In the private workspace, run in this order, stopping on any failure:

1. `03_stage0/extract_cohort_flow.ps1` to reconstruct audited flow counts.
2. `03_stage05_signal_audit/00_scripts/extract_signal_data.ps1`, then `analyze_signal_audit.py` in that same folder.
3. `04_formal_modeling/00_scripts/extract_formal_data.ps1`, then `prepare_features.py`.
4. In the formal-modeling script folder: `run_internal_and_lock.py`, `run_external_locked.py`, `run_resolution_experiment.py`, then `assemble_results.py`.
5. `09_prewrite_work/00_scripts/run_prewrite_analysis.py` for existing pairwise and fixed-unweighted sensitivity outputs.
6. `12_final_presubmission_v2_2_work/run_level_only_comparator.py` for the added same-window comparator.
7. Clinical Table 1 extraction uses the two SQL files staged under `clinical_characteristics`. These output patient-level records and stay private. The public `tables/Table_1_clinical_characteristics.csv` is the previously audited aggregate, not a replacement for rechecking extraction provenance.

The code preserves the original candidate grids and fold logic. A rerun is not permission to search new parameters or replace original predictions with favorable variants. Compare cohort counts, outcomes, fitted setting selection and aggregate performance against the released tables, allowing only declared numerical tolerances. Hardware/library differences can affect floating-point or stochastic computations; discrepancies must be investigated, not silently overwritten.

## Public versus restricted artifacts
All row-level inputs/derived records, prediction vectors, fold assignments, NPZ/PKL/PT model objects and exact ROC/PR probability thresholds are restricted working artifacts. Never copy an entire private output directory back to GitHub. `.gitignore` is a guardrail for git; it does not filter browser uploads or Zenodo uploads. Only whitelisted aggregate tables and reviewed source code should be exported.

`figures/source_data/Figure_2_roc_pr_source.csv` is a fixed 101-point, threshold-free display approximation, rounded to five decimals. It permits an approximate public redraw but must not be used to recompute exact AUROC or average precision. Exact published curves and metrics are generated inside the authorized environment from private predictions. Full table metrics remain unchanged.

## Rendering
`figures/render_final_figures.R` accepts the public release root and an output directory, as documented by its usage message. Use an output folder outside the public repository. `figures/prepare_v22_figure_sources.R` is a private-workspace assembly step using patient-level predictions; its raw output is NOT automatically publication-safe. Apply the fixed-grid export policy before any public redistribution. No R rendering or environment restoration is silently claimed by the Python synthetic test.
