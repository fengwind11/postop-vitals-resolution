from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from modeling_common import (
    SEED, apply_sequence_preprocessor, apply_tabular_preprocessor, bootstrap_performance,
    fit_grud_fixed_epochs, fit_sequence_preprocessor, fit_tabular_preprocessor,
    metric_values, parameter_count, predict_grud,
)
from run_internal_and_lock import xgb_estimator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data_internal"
LOCKED = ROOT / "04_locked_models"
RESOLUTION = ROOT / "06_resolution_experiment"
MACHINE = ROOT / "07_machine_readable"
LOGS = ROOT / "10_logs"
for folder in (RESOLUTION, MACHINE, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_missing(row: np.ndarray) -> int:
    best = current = 0
    for observed in np.isfinite(row):
        if observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def summary_at_resolution(values: np.ndarray, age: np.ndarray, sex: np.ndarray,
                          width_min: int) -> np.ndarray:
    n, length, _ = values.shape
    step_hours = width_min / 60.0
    times = (np.arange(length, dtype=float) + 0.5) * step_hours
    columns = [age.astype(float), sex.astype(float)]
    for j, signal in enumerate(("HR", "MAP", "SpO2")):
        x = values[:, :, j].astype(float)
        valid = np.isfinite(x)
        with np.errstate(all="ignore"):
            median = np.nanmedian(x, axis=1)
            extreme = np.nanmax(x, axis=1) if signal == "HR" else np.nanmin(x, axis=1)
            sd = np.nanstd(x, axis=1, ddof=1)
        slope = np.full(n, np.nan)
        for i in range(n):
            use = valid[i]
            if use.sum() >= 2:
                slope[i] = np.polyfit(times[use], x[i, use], 1)[0]
        threshold = (x > 100) if signal == "HR" else ((x < 65) if signal == "MAP" else (x < 92))
        denom = valid.sum(axis=1)
        burden = np.full(n, np.nan)
        ok = denom > 0
        burden[ok] = (threshold & valid).sum(axis=1)[ok] / denom[ok]
        valid_fraction = denom / length
        streak_hours = np.asarray([longest_missing(row) * step_hours for row in x], dtype=float)
        columns.extend([median, extreme, sd, slope, burden, valid_fraction, streak_hours])
    raw = np.column_stack(columns).astype(np.float32)
    return np.column_stack([raw, np.isnan(raw).astype(np.float32)]).astype(np.float32)


def curves(y: np.ndarray, predictions: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for (width, model), pred in predictions.items():
        fpr, tpr, thresholds = roc_curve(y, pred)
        for x, z, threshold in zip(fpr, tpr, thresholds):
            rows.append({"resolution_min": width, "model": model, "curve": "ROC",
                         "x": x, "y": z, "threshold": threshold})
        precision, recall, thresholds_pr = precision_recall_curve(y, pred)
        for x, z, threshold in zip(recall, precision, np.r_[thresholds_pr, np.nan]):
            rows.append({"resolution_min": width, "model": model, "curve": "PR",
                         "x": x, "y": z, "threshold": threshold})
    return pd.DataFrame(rows)


def paired_resolution_differences(y: np.ndarray, predictions: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 26000)
    metrics = ("auroc", "auprc", "brier", "log_loss")
    point = {(width, model): metric_values(y, pred) for (width, model), pred in predictions.items()}
    boot = {model: {metric: [] for metric in metrics}
            for model in ("model1_elastic_net", "model2_xgboost", "model3_grud")}
    valid = 0
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        valid += 1
        for model in boot:
            a = metric_values(y[idx], predictions[(5, model)][idx])
            b = metric_values(y[idx], predictions[(60, model)][idx])
            for metric in metrics:
                boot[model][metric].append(a[metric] - b[metric])
    rows = []
    for model in boot:
        for metric in metrics:
            values = np.asarray(boot[model][metric])
            rows.append({"model": model, "contrast": "5min_minus_60min", "metric": metric,
                         "estimate": point[(5, model)][metric] - point[(60, model)][metric],
                         "ci_low": float(np.quantile(values, .025)),
                         "ci_high": float(np.quantile(values, .975)),
                         "bootstrap_replicates": valid, "paired_patient_resampling": True})
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    manifest_path = LOCKED / "LOCKED_MODEL_MANIFEST.json"
    manifest_hash_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest["selected_hyperparameters"]
    data = np.load(DATA / "sicdb_resolution_data.npz")
    ids, y = data["analysis_id"].astype(str), data["y"].astype(int)
    age, sex = data["age"].astype(np.float32), data["sex"].astype(np.float32)
    if len(y) != 2061 or int(y.sum()) != 151:
        raise RuntimeError(f"STOP_AND_DEBUG_DENSE_CORE: N={len(y)}, deaths={int(y.sum())}")
    if int(y.sum()) < 120:
        run_grud = False
    else:
        run_grud = True
    widths = (5, 15, 60)
    summaries = {width: summary_at_resolution(data[f"values_{width}min"], age, sex, width)
                 for width in widths}
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    assignments = np.full((5, len(y)), -1, dtype=int)
    oof = {(width, model): np.full((5, len(y)), np.nan)
           for width in widths for model in ("model1_elastic_net", "model2_xgboost", "model3_grud")}
    parameter_counts = set()
    for split_index, (train, test) in enumerate(splitter.split(age, y)):
        repeat, fold = divmod(split_index, 5)
        assignments[repeat, test] = fold
        seed = SEED + 30000 + repeat * 100 + fold
        print(f"RESOLUTION repeat={repeat + 1}/5 fold={fold + 1}/5", flush=True)
        for width in widths:
            summary = summaries[width]
            prep1 = fit_tabular_preprocessor(summary[train], scale=True)
            xtr1 = apply_tabular_preprocessor(summary[train], prep1)
            xte1 = apply_tabular_preprocessor(summary[test], prep1)
            p1 = selected["model1_elastic_net"]
            model1 = LogisticRegression(C=p1["C"], penalty="elasticnet", l1_ratio=p1["l1_ratio"],
                                        solver="saga", max_iter=10000, tol=1e-4, random_state=seed)
            model1.fit(xtr1, y[train])
            oof[(width, "model1_elastic_net")][repeat, test] = model1.predict_proba(xte1)[:, 1]

            prep2 = fit_tabular_preprocessor(summary[train], scale=False)
            model2 = xgb_estimator(selected["model2_xgboost"], seed)
            model2.fit(apply_tabular_preprocessor(summary[train], prep2), y[train])
            oof[(width, "model2_xgboost")][repeat, test] = model2.predict_proba(
                apply_tabular_preprocessor(summary[test], prep2))[:, 1]

            if run_grud:
                values = data[f"values_{width}min"].astype(np.float32)
                mask = data[f"mask_{width}min"].astype(np.float32)
                delta = data[f"delta_{width}min"].astype(np.float32)
                prep3 = fit_sequence_preprocessor(values[train], age[train])
                xtr3, str3 = apply_sequence_preprocessor(values[train], mask[train], age[train], sex[train], prep3)
                xte3, ste3 = apply_sequence_preprocessor(values[test], mask[test], age[test], sex[test], prep3)
                p3 = selected["model3_grud"]
                model3 = fit_grud_fixed_epochs(xtr3, mask[train], delta[train], str3, y[train],
                                               hidden_size=p3["hidden_size"], dropout=p3["dropout"],
                                               weight_decay=p3["weight_decay"], epochs=p3["epochs"],
                                               seed=seed, weighted=True)
                parameter_counts.add(parameter_count(model3))
                oof[(width, "model3_grud")][repeat, test] = predict_grud(
                    model3, xte3, mask[test], delta[test], ste3)
    if not run_grud:
        for width in widths:
            del oof[(width, "model3_grud")]
    if any(np.isnan(array).any() for array in oof.values()) or np.any(assignments < 0):
        raise RuntimeError("Resolution OOF completeness failure")
    if run_grud and len(parameter_counts) != 1:
        raise RuntimeError(f"GRU-D parameter count changed across resolutions: {parameter_counts}")

    prediction_rows = []
    models = ("model1_elastic_net", "model2_xgboost", "model3_grud") if run_grud else (
        "model1_elastic_net", "model2_xgboost")
    for repeat in range(5):
        for width in widths:
            for i, analysis_id in enumerate(ids):
                prediction_rows.append({"analysis_id": analysis_id, "outcome": int(y[i]),
                                        "repeat": repeat + 1, "fold": int(assignments[repeat, i]) + 1,
                                        "resolution_min": width,
                                        **{model: float(oof[(width, model)][repeat, i]) for model in models}})
    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame.to_csv(RESOLUTION / "resolution_oof_predictions.csv", index=False, encoding="utf-8-sig")
    mean_predictions = {(width, model): oof[(width, model)].mean(axis=0) for width in widths for model in models}

    performance_parts = []
    for width in widths:
        table, _ = bootstrap_performance(y, {model: mean_predictions[(width, model)] for model in models},
                                         replicates=2000, seed=SEED + 24000 + width,
                                         compute_differences=False)
        frame = pd.DataFrame(table)
        frame.insert(0, "resolution_min", width)
        frame.insert(1, "n", len(y))
        frame.insert(2, "events", int(y.sum()))
        performance_parts.append(frame)
    performance = pd.concat(performance_parts, ignore_index=True)
    performance.to_csv(MACHINE / "resolution_performance.csv", index=False, encoding="utf-8-sig")
    paired = paired_resolution_differences(y, mean_predictions)
    paired.to_csv(MACHINE / "bootstrap_differences_resolution.csv", index=False, encoding="utf-8-sig")
    curves(y, mean_predictions).to_csv(RESOLUTION / "resolution_roc_pr_source.csv", index=False,
                                        encoding="utf-8-sig")
    status = {
        "status": "SUCCESS", "dense_core_n": len(y), "dense_core_deaths28": int(y.sum()),
        "grud_allowed_by_event_rule": run_grud, "resolutions_min": list(widths),
        "same_patient_set": True, "same_fold_assignments": True,
        "grud_parameter_counts": sorted(parameter_counts), "bootstrap_replicates": 2000,
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": sha256(manifest_path), "elapsed_seconds": time.time() - start,
        "script_sha256": sha256(Path(__file__)),
    }
    if status["manifest_sha256_after"] != manifest_hash_before:
        raise RuntimeError("Locked manifest changed during resolution experiment")
    (LOGS / "resolution_experiment_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
