# Postoperative vital-sign summaries and sampling resolution

Analysis source for **Early postoperative vital-sign summaries for 28-day mortality prediction after major abdominal surgery: external validation and sampling resolution**.

## Scope
A 12-hour landmark prognostic study in MIMIC-IV v3.1 (development) and SICdb v1.0.8 (external evaluation), with a separate within-SICdb sampling-resolution experiment. Original predictions and aggregate results are retained. The full-summary penalized logistic model selected an L1 boundary; the added same-window median-level comparator selected an elastic-net mixture of 0.5. Cross-database validation does not imply ready-to-use clinical deployment.

## What is included
Extraction source, signal decoding/harmonization, original model and evaluation scripts, prewrite sensitivity source, the added median-level comparator, reviewed aggregate tables, figure scripts, environment manifests, MIT licence and a synthetic test. There are no raw patient records, individual predictions, fitted model objects, credentials or private fold assignments.

## Quick start
Install the documented dependencies in an isolated environment. Run `run_all.cmd` on Windows or `sh run_all.sh` elsewhere for the synthetic test only. Stage a private workspace with `python prepare_private_workspace.py --work-dir <PRIVATE_DIRECTORY_OUTSIDE_THIS_REPOSITORY>`. Read **REPRODUCIBILITY.md** before any database execution; scripts must not be run directly from the public `src` folder. Staging and synthetic tests do not establish a complete authorized-data rerun.

## Data and aggregate source precautions
Request credentialed database access directly from PhysioNet. Only approved aggregate outputs are public. ROC/PR sources are deliberately fixed-grid approximations without individual thresholds. Do not infer exact metrics from this plotting grid. The definitive metric estimates are in `tables/` and `analysis_update_level_only/`.

## Citation and release
The initial software version is 1.0.0. Before creating the GitHub release, fill the actual repository URL and release date using `prepare_release_metadata.py`. Connect the repository in Zenodo **before** publishing the release. Then cite the actual version-specific archived DOI in the manuscript. Do not move an archived tag merely to insert its own DOI; maintain subsequent metadata changes on the default branch and create a new version for changes to archived source. No repository or DOI is represented as already published in this prepared package.

## Licence
Original source is under the MIT License. This does not replace PhysioNet data-use terms. Third-party software is a dependency, not relicensed code. No original database records or restricted model artifacts are licensed by this repository.
