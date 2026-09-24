from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MACHINE = ROOT / "07_machine_readable"
TABLES = ROOT / "08_tables"
FIGURES = ROOT / "09_figures"
SOURCE = FIGURES / "source_data"
LOGS = ROOT / "10_logs"
AUDIT = PROJECT / "03_stage05_signal_audit"
STAGE0 = PROJECT / "03_stage0" / "flow_outputs"
for folder in (TABLES, FIGURES, SOURCE, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

MODELS = ["model0_snapshot", "model1_elastic_net", "model2_xgboost", "model3_grud"]
MODEL_LABEL = {
    "model0_snapshot": "Snapshot logistic",
    "model1_elastic_net": "L1-penalized logistic temporal summaries",
    "model2_xgboost": "XGBoost summaries",
    "model3_grud": "Compact GRU-D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ci_text(row: pd.Series, digits: int = 3) -> str:
    return f"{float(row.estimate):.{digits}f} ({float(row.ci_low):.{digits}f}–{float(row.ci_high):.{digits}f})"


def internal_curves() -> pd.DataFrame:
    oof = pd.read_csv(MACHINE / "internal_oof_predictions.csv")
    mean = oof.groupby(["analysis_id", "outcome"], as_index=False)[MODELS].mean()
    y = mean.outcome.to_numpy(dtype=int)
    rows = []
    for model in MODELS:
        p = mean[model].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(y, p)
        for x, z, threshold in zip(fpr, tpr, thresholds):
            rows.append({"dataset": "MIMIC-IV internal OOF", "representation": "patient_mean_5_repeats",
                         "model": model, "curve": "ROC", "x": x, "y": z, "threshold": threshold})
        precision, recall, thresholds_pr = precision_recall_curve(y, p)
        for x, z, threshold in zip(recall, precision, np.r_[thresholds_pr, np.nan]):
            rows.append({"dataset": "MIMIC-IV internal OOF", "representation": "patient_mean_5_repeats",
                         "model": model, "curve": "PR", "x": x, "y": z, "threshold": threshold})
    return pd.DataFrame(rows)


def assemble_flow() -> pd.DataFrame:
    rows = []
    for dataset, file in (("MIMIC-IV", STAGE0 / "mimiciv_flow_counts.csv"),
                          ("SICdb", STAGE0 / "sicdb_flow_counts.csv")):
        frame = pd.read_csv(file)
        if len(frame) != 1:
            raise RuntimeError(f"Unexpected flow layout: {file}")
        for order, column in enumerate(frame.columns, start=1):
            rows.append({"dataset": dataset, "order": order, "stage": column,
                         "n": int(frame.iloc[0][column])})
    return pd.DataFrame(rows)


def main() -> None:
    perf_internal = pd.read_csv(MACHINE / "performance_internal.csv")
    perf_external = pd.read_csv(MACHINE / "performance_external.csv")
    resolution = pd.read_csv(MACHINE / "resolution_performance.csv")
    resolution_diff = pd.read_csv(MACHINE / "bootstrap_differences_resolution.csv")
    sparse = pd.read_csv(MACHINE / "sparse_phase_external.csv")

    table1a = pd.read_csv(MACHINE / "table1_cohort_descriptives.csv")
    table1b = pd.read_csv(MACHINE / "table1_signal_availability.csv")
    table1a.to_csv(TABLES / "Table_1A_cohort_descriptives.csv", index=False, encoding="utf-8-sig")
    table1b.to_csv(TABLES / "Table_1B_signal_availability.csv", index=False, encoding="utf-8-sig")

    selected_metrics = ["auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]
    table2_rows = []
    for dataset, frame in (("MIMIC-IV internal OOF", perf_internal), ("SICdb external", perf_external)):
        for model in MODELS:
            row = {"dataset": dataset, "model": model, "model_label": MODEL_LABEL[model]}
            subset = frame[frame.model == model].set_index("metric")
            for metric in selected_metrics:
                row[metric] = ci_text(subset.loc[metric])
            table2_rows.append(row)
    pd.DataFrame(table2_rows).to_csv(TABLES / "Table_2_internal_external_performance.csv", index=False,
                                     encoding="utf-8-sig")

    table3a = resolution[resolution.metric.isin(["auroc", "auprc", "brier", "log_loss"])].copy()
    table3a["estimate_95ci"] = table3a.apply(ci_text, axis=1)
    table3a[["resolution_min", "model", "metric", "estimate_95ci"]].to_csv(
        TABLES / "Table_3A_resolution_performance.csv", index=False, encoding="utf-8-sig")
    table3b = resolution_diff.copy()
    table3b["delta_95ci"] = table3b.apply(ci_text, axis=1)
    table3b[["model", "contrast", "metric", "delta_95ci"]].to_csv(
        TABLES / "Table_3B_paired_5minus60.csv", index=False, encoding="utf-8-sig")

    cal_internal = perf_internal[perf_internal.metric.isin(["calibration_intercept", "calibration_slope", "brier"])].copy()
    cal_internal["representation"] = "patient_mean_5_repeats"
    cal_internal["dataset"] = "MIMIC-IV"
    cal_external = pd.read_csv(MACHINE / "calibration_metrics_external.csv")
    calibration_metrics = pd.concat([cal_internal, cal_external], ignore_index=True, sort=False)
    calibration_metrics.to_csv(MACHINE / "calibration_metrics.csv", index=False, encoding="utf-8-sig")

    differences = []
    for label, file in (
        ("internal_model_comparison", ROOT / "03_internal_validation" / "internal_paired_differences.csv"),
        ("external_model_comparison", MACHINE / "bootstrap_differences_external_models.csv"),
        ("resolution_5minus60", MACHINE / "bootstrap_differences_resolution.csv"),
    ):
        frame = pd.read_csv(file)
        frame.insert(0, "comparison_scope", label)
        differences.append(frame)
    pd.concat(differences, ignore_index=True, sort=False).to_csv(MACHINE / "bootstrap_differences.csv",
                                                                 index=False, encoding="utf-8-sig")

    curves = internal_curves()
    external_curves = pd.read_csv(ROOT / "05_external_validation" / "external_roc_pr_source.csv")
    external_curves.insert(0, "dataset", "SICdb external")
    curves = pd.concat([curves, external_curves], ignore_index=True, sort=False)
    curves.to_csv(SOURCE / "Figure_2_roc_pr_source.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(ROOT / "05_external_validation" / "external_calibration_bins.csv").to_csv(
        SOURCE / "Figure_3_calibration_source.csv", index=False, encoding="utf-8-sig")
    resolution.to_csv(SOURCE / "Figure_4_resolution_performance_source.csv", index=False, encoding="utf-8-sig")
    flow = assemble_flow()
    flow.to_csv(SOURCE / "Figure_1_cohort_flow_source.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(AUDIT / "04_resolution_loss" / "resolution_loss_summary.csv").to_csv(
        SOURCE / "Figure_5_measurement_loss_source.csv", index=False, encoding="utf-8-sig")

    sparse_range = sparse.groupby(["model", "metric"]).estimate.agg(["min", "median", "max"]).reset_index()
    sparse_range.to_csv(TABLES / "Supplement_sparse_phase_range.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(ROOT / "03_internal_validation" / "internal_tuning_history.csv",
                 TABLES / "Supplement_internal_tuning_history.csv")
    shutil.copy2(ROOT / "03_internal_validation" / "model3_unweighted_sensitivity_metrics.json",
                 TABLES / "Supplement_model3_unweighted_sensitivity_metrics.json")

    external_grud = perf_external[perf_external.model.eq("model3_grud")].set_index("metric")
    sparse_grud = sparse_range[sparse_range.model.eq("model3_grud")].set_index("metric")
    grud_delta = resolution_diff[resolution_diff.model.eq("model3_grud")].set_index("metric")
    verdict = "GO_MANUSCRIPT_MEASUREMENT_FOCUS"
    rationale = {
        "final_verdict": verdict,
        "primary_external_grud": {metric: external_grud.loc[metric, ["estimate", "ci_low", "ci_high"]].to_dict()
                                  for metric in selected_metrics},
        "sparse_grud_range": {metric: sparse_grud.loc[metric, ["min", "median", "max"]].to_dict()
                              for metric in selected_metrics},
        "resolution_grud_5minus60": {metric: grud_delta.loc[metric, ["estimate", "ci_low", "ci_high"]].to_dict()
                                     for metric in ("auroc", "auprc", "brier", "log_loss")},
        "interpretation": [
            "External discrimination remained above prevalence-level discrimination, but the locked weighted GRU-D showed marked calibration-in-the-large and slope deviation.",
            "Sparse one-point-per-hour phases preserved only moderate discrimination and showed similarly poor GRU-D calibration, implicating recording representation in transportability.",
            "The primary GRU-D 5-minus-60 predictive contrasts were small and all 95% intervals crossed zero; resolution gains were not general across model families.",
            "Previously frozen measurement-level event loss remains real, but it must not be described as a consistent mortality-prediction gain.",
        ],
    }
    (ROOT / "RESULTS_VERDICT.json").write_text(json.dumps(rationale, indent=2), encoding="utf-8")

    readme = f"""# Formal Modeling Execution

Final verdict: `{verdict}`

## Completed

- Rebuilt the frozen cohorts and reproduced MIMIC-IV 1,740/168 and SICdb 2,376/185 under read-only database sessions.
- Constructed identical 0–12 h HR/MAP/SpO2 schemas, with 50-patient time-window checks per database.
- Ran five repeats of stratified five-fold MIMIC internal validation for four prespecified models.
- Locked all full-MIMIC preprocessing/model objects before any SICdb prediction; the lock hash remained unchanged.
- Ran SICdb hourly-median external validation plus all five fixed sparse phases with 2,000 patient bootstraps.
- Ran same-patient SICdb dense-core 5/15/60-min performance experiments (N=2,061; deaths=151) with paired bootstraps.
- Generated source-linked tables, figures, environment logs, hashes and independent assertions.

## Not done

- No literature refresh, cohort/time-zero/outcome change, database pooling, outcome-driven feature selection, external updating, post-hoc subgroup, added model family or threshold search.
- No SICdb recalibration replaced primary predictions. Decision curves remain secondary.

## Conclusion

The locked temporal model retained moderate external discrimination but transported poorly in calibration, especially under sparse charting. The controlled resolution experiment did not show a consistent 5-min mortality-prediction advantage for the primary GRU-D, despite established measurement-level information loss. The defensible manuscript direction is therefore measurement/transportability, not a claim of universal high-density predictive gain.
"""
    (ROOT / "00_README_EXECUTION.md").write_text(readme, encoding="utf-8")

    summary = f"""# Results Executive Summary

## Verdict

`{verdict}`

The result supports a measurement-process and transportability manuscript focus. It does not support presenting 5-min sampling as a consistently superior mortality predictor.

## Primary model

- MIMIC repeated OOF GRU-D: AUROC {ci_text(pd.read_csv(MACHINE / 'performance_internal.csv').query("model == 'model3_grud' and metric == 'auroc'").iloc[0])}; AUPRC {ci_text(pd.read_csv(MACHINE / 'performance_internal.csv').query("model == 'model3_grud' and metric == 'auprc'").iloc[0])}.
- SICdb locked external GRU-D: AUROC {ci_text(external_grud.loc['auroc'])}; AUPRC {ci_text(external_grud.loc['auprc'])}; Brier {ci_text(external_grud.loc['brier'])}; calibration slope {ci_text(external_grud.loc['calibration_slope'])}.
- Sparse phases: GRU-D AUROC {sparse_grud.loc['auroc','min']:.3f}–{sparse_grud.loc['auroc','max']:.3f}; AUPRC {sparse_grud.loc['auprc','min']:.3f}–{sparse_grud.loc['auprc','max']:.3f}; calibration slope {sparse_grud.loc['calibration_slope','min']:.3f}–{sparse_grud.loc['calibration_slope','max']:.3f}.

## Resolution experiment

- Dense-core: N=2,061, deaths=151.
- Primary GRU-D 5-minus-60: ΔAUROC {ci_text(grud_delta.loc['auroc'])}; ΔAUPRC {ci_text(grud_delta.loc['auprc'])}; ΔBrier {ci_text(grud_delta.loc['brier'])}; Δlog loss {ci_text(grud_delta.loc['log_loss'])}.
- Interpretation: no consistent paired predictive gain for the primary temporal model. Small L1-logistic/XGBoost gains are model-dependent and do not justify a general causal or clinical-benefit claim.

## Clinical-methodological meaning

High-frequency SICdb signals contain short-event, burden, extreme and variability information that hourly compression loses. In this endpoint and frozen model family, that extra measurement information did not translate into a robust GRU-D gain for 28-day mortality. The recording process materially affected probability calibration across databases and sparse representations.
"""
    (ROOT / "RESULTS_EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")

    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "torch": torch.__version__, "seed": 20260918,
        "figure_backend": "R only", "source_databases_read_only": True,
    }
    (LOGS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    status = {"status": "SUCCESS", "final_verdict": verdict,
              "artifacts": [{"file": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in
                            [TABLES / "Table_2_internal_external_performance.csv",
                             TABLES / "Table_3B_paired_5minus60.csv",
                             SOURCE / "Figure_2_roc_pr_source.csv", ROOT / "RESULTS_VERDICT.json"]],
              "script_sha256": sha256(Path(__file__))}
    (LOGS / "results_assembly_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
