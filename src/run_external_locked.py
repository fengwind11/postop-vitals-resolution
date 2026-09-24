from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from modeling_common import (
    MODEL_NAMES, SEED, CompactGRUD, apply_sequence_preprocessor, apply_tabular_preprocessor,
    metric_values, predict_grud,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data_internal"
LOCKED = ROOT / "04_locked_models"
EXTERNAL = ROOT / "05_external_validation"
MACHINE = ROOT / "07_machine_readable"
LOGS = ROOT / "10_logs"
for folder in (EXTERNAL, MACHINE, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_table(y: np.ndarray, predictions: dict[str, np.ndarray], seed: int,
                    differences: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope")
    point = {name: metric_values(y, p) for name, p in predictions.items()}
    boot = {name: {metric: [] for metric in metrics} for name in predictions}
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        valid += 1
        for name, p in predictions.items():
            values = metric_values(y[idx], p[idx])
            for metric in metrics:
                boot[name][metric].append(values[metric])
    rows = []
    for name in predictions:
        for metric in metrics:
            values = np.asarray(boot[name][metric], dtype=float)
            rows.append({"model": name, "metric": metric, "estimate": point[name][metric],
                         "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                         "bootstrap_replicates": valid})
    diff_rows = []
    if differences:
        reference = "model3_grud"
        for comparator in predictions:
            if comparator == reference:
                continue
            for metric in ("auroc", "auprc", "brier", "log_loss"):
                values = np.asarray(boot[reference][metric]) - np.asarray(boot[comparator][metric])
                diff_rows.append({"reference_model": reference, "comparator_model": comparator,
                                  "metric": metric, "estimate": point[reference][metric] - point[comparator][metric],
                                  "ci_low": float(np.quantile(values, .025)),
                                  "ci_high": float(np.quantile(values, .975)),
                                  "bootstrap_replicates": valid, "paired": True})
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def predict_representation(data, label: str, models: dict, preprocessors: dict) -> dict[str, np.ndarray]:
    prefix = "" if label == "hourly_median" else f"phase_{int(label.split('_')[-1]):02d}_"
    snapshot = data["snapshot"] if not prefix else data[prefix + "snapshot"]
    summary = data["summary"] if not prefix else data[prefix + "summary"]
    values = data["values"] if not prefix else data[prefix + "values"]
    mask = data["mask"] if not prefix else data[prefix + "mask"]
    delta = data["delta"] if not prefix else data[prefix + "delta"]
    age, sex = data["age"], data["sex"]
    p0 = models["model0_snapshot"].predict_proba(
        apply_tabular_preprocessor(snapshot, preprocessors["model0_snapshot"]))[:, 1]
    p1 = models["model1_elastic_net"].predict_proba(
        apply_tabular_preprocessor(summary, preprocessors["model1_elastic_net"]))[:, 1]
    p2 = models["model2_xgboost"].predict_proba(
        apply_tabular_preprocessor(summary, preprocessors["model2_xgboost"]))[:, 1]
    x3, static3 = apply_sequence_preprocessor(values, mask, age, sex, preprocessors["model3_grud"])
    p3 = predict_grud(models["model3_grud"], x3, mask, delta, static3)
    return dict(zip(MODEL_NAMES, (p0, p1, p2, p3)))


def curve_rows(y: np.ndarray, predictions: dict[str, np.ndarray], representation: str) -> pd.DataFrame:
    rows = []
    for model, pred in predictions.items():
        fpr, tpr, thresholds = roc_curve(y, pred)
        for x, z, threshold in zip(fpr, tpr, thresholds):
            rows.append({"representation": representation, "model": model, "curve": "ROC",
                         "x": x, "y": z, "threshold": threshold})
        precision, recall, thresholds_pr = precision_recall_curve(y, pred)
        padded = np.r_[thresholds_pr, np.nan]
        for x, z, threshold in zip(recall, precision, padded):
            rows.append({"representation": representation, "model": model, "curve": "PR",
                         "x": x, "y": z, "threshold": threshold})
    return pd.DataFrame(rows)


def calibration_rows(y: np.ndarray, predictions: dict[str, np.ndarray], representation: str) -> pd.DataFrame:
    rows = []
    for model, pred in predictions.items():
        try:
            groups = pd.qcut(pred, q=10, labels=False, duplicates="drop")
        except ValueError:
            groups = pd.Series(np.zeros(len(pred), dtype=int))
        frame = pd.DataFrame({"group": groups, "risk": pred, "outcome": y})
        for group, part in frame.groupby("group", dropna=False):
            rows.append({"representation": representation, "model": model, "bin": int(group),
                         "n": len(part), "mean_predicted": float(part.risk.mean()),
                         "observed": float(part.outcome.mean()), "events": int(part.outcome.sum())})
    return pd.DataFrame(rows)


def dca_rows(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    n = len(y)
    prevalence = float(np.mean(y))
    for threshold in np.arange(0.02, 0.251, 0.01):
        odds = threshold / (1 - threshold)
        rows.append({"model": "treat_none", "threshold": threshold, "net_benefit": 0.0})
        rows.append({"model": "treat_all", "threshold": threshold,
                     "net_benefit": prevalence - (1 - prevalence) * odds})
        for model, pred in predictions.items():
            positive = pred >= threshold
            tp = int(np.sum(positive & (y == 1)))
            fp = int(np.sum(positive & (y == 0)))
            rows.append({"model": model, "threshold": threshold,
                         "net_benefit": tp / n - fp / n * odds})
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    manifest_path = LOCKED / "LOCKED_MODEL_MANIFEST.json"
    lock_hash_before = sha256(manifest_path)
    recorded_hash = (LOCKED / "LOCKED_MODEL_MANIFEST.sha256").read_text(encoding="ascii").split()[0]
    if lock_hash_before != recorded_hash:
        raise RuntimeError("LOCKED_MODEL_MANIFEST hash mismatch before external prediction")
    manifest = load_json(manifest_path)
    if manifest["status"] != "LOCKED_BEFORE_EXTERNAL_PREDICTION" or manifest["external_prediction_status_at_lock"] != "NOT_RUN":
        raise RuntimeError("Invalid lock status")
    for entries in manifest["artifacts"].values():
        for item in entries:
            path = LOCKED / item["file"]
            if sha256(path) != item["sha256"]:
                raise RuntimeError(f"Locked artifact hash mismatch: {path.name}")

    preprocessors = {name: load_json(LOCKED / f"model{i}_preprocessor.json")
                     for i, name in enumerate(MODEL_NAMES)}
    with (LOCKED / "model0_snapshot.pkl").open("rb") as handle:
        model0 = pickle.load(handle)
    with (LOCKED / "model1_elastic_net.pkl").open("rb") as handle:
        model1 = pickle.load(handle)
    model2 = XGBClassifier()
    model2.load_model(LOCKED / "model2_xgboost.json")
    saved3 = torch.load(LOCKED / "model3_grud.pt", map_location="cpu", weights_only=False)
    model3 = CompactGRUD(3, saved3["hidden_size"], saved3["dropout"])
    model3.load_state_dict(saved3["state_dict"])
    model3.eval()
    models = dict(zip(MODEL_NAMES, (model0, model1, model2, model3)))

    data = np.load(DATA / "sicdb_external_data.npz")
    ids, y = data["analysis_id"].astype(str), data["y"].astype(int)
    if len(y) != 2376 or int(y.sum()) != 185:
        raise RuntimeError("STOP_AND_DEBUG_COHORT at external stage")
    representations = ["hourly_median", "sparse_phase_00", "sparse_phase_15",
                       "sparse_phase_30", "sparse_phase_45", "sparse_phase_59"]
    all_predictions, prediction_rows, calibration, sparse_metrics = {}, [], [], []
    for ri, representation in enumerate(representations):
        print(f"EXTERNAL representation={representation}", flush=True)
        predictions = predict_representation(data, representation, models, preprocessors)
        all_predictions[representation] = predictions
        for i, analysis_id in enumerate(ids):
            prediction_rows.append({"analysis_id": analysis_id, "outcome": int(y[i]),
                                    "representation": representation,
                                    **{model: float(pred[i]) for model, pred in predictions.items()}})
        calibration.append(calibration_rows(y, predictions, representation))
        if representation != "hourly_median":
            table, _ = bootstrap_table(y, predictions, SEED + 10000 + ri, differences=False)
            table.insert(0, "representation", representation)
            sparse_metrics.append(table)

    pd.DataFrame(prediction_rows).to_csv(MACHINE / "external_predictions_sicdb.csv", index=False,
                                         encoding="utf-8-sig")
    primary = all_predictions["hourly_median"]
    performance, differences = bootstrap_table(y, primary, SEED + 9000, differences=True)
    performance.insert(0, "dataset", "SICdb")
    performance.insert(1, "representation", "hourly_median")
    performance.insert(2, "n", len(y))
    performance.insert(3, "events", int(y.sum()))
    performance.to_csv(MACHINE / "performance_external.csv", index=False, encoding="utf-8-sig")
    differences.insert(0, "dataset", "SICdb")
    differences.insert(1, "representation", "hourly_median")
    differences.to_csv(MACHINE / "bootstrap_differences_external_models.csv", index=False, encoding="utf-8-sig")
    sparse = pd.concat(sparse_metrics, ignore_index=True)
    sparse.to_csv(MACHINE / "sparse_phase_external.csv", index=False, encoding="utf-8-sig")
    sparse_summary = sparse.groupby(["model", "metric"]).estimate.agg(["min", "median", "max"]).reset_index()
    sparse_summary.to_csv(EXTERNAL / "sparse_phase_range_summary.csv", index=False, encoding="utf-8-sig")

    cal_bins = pd.concat(calibration, ignore_index=True)
    cal_bins.to_csv(EXTERNAL / "external_calibration_bins.csv", index=False, encoding="utf-8-sig")
    curves = curve_rows(y, primary, "SICdb_hourly_median")
    curves.to_csv(EXTERNAL / "external_roc_pr_source.csv", index=False, encoding="utf-8-sig")
    dca_rows(y, primary).to_csv(EXTERNAL / "decision_curve_external.csv", index=False, encoding="utf-8-sig")

    calibration_metric_rows = []
    for representation, predictions in all_predictions.items():
        for model, pred in predictions.items():
            metrics = metric_values(y, pred)
            calibration_metric_rows.append({"dataset": "SICdb", "representation": representation,
                                            "model": model, "n": len(y), "events": int(y.sum()),
                                            "calibration_intercept": metrics["calibration_intercept"],
                                            "calibration_slope": metrics["calibration_slope"],
                                            "brier": metrics["brier"]})
    pd.DataFrame(calibration_metric_rows).to_csv(MACHINE / "calibration_metrics_external.csv", index=False,
                                                 encoding="utf-8-sig")

    lock_hash_after = sha256(manifest_path)
    if lock_hash_after != lock_hash_before:
        raise RuntimeError("LOCKED_MODEL_MANIFEST changed during external stage")
    script_text = Path(__file__).read_text(encoding="utf-8")
    forbidden = re.findall(r"\.(?:fit|partial_fit)\s*\(|\btune\w*\s*\(", script_text)
    if forbidden:
        raise RuntimeError(f"Forbidden external-stage training token(s): {forbidden}")
    status = {
        "status": "SUCCESS", "external_n": len(y), "external_deaths28": int(y.sum()),
        "representations": representations, "bootstrap_replicates": 2000,
        "manifest_sha256_before": lock_hash_before, "manifest_sha256_after": lock_hash_after,
        "manifest_unchanged": True, "training_operations_detected": False,
        "script_sha256": sha256(Path(__file__)), "elapsed_seconds": time.time() - start,
    }
    (LOGS / "external_validation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
