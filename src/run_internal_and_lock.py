from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from modeling_common import (
    MODEL_NAMES, SEED, CompactGRUD, apply_sequence_preprocessor, apply_tabular_preprocessor,
    bootstrap_performance, fit_grud, fit_grud_fixed_epochs, fit_sequence_preprocessor,
    fit_tabular_preprocessor, metric_values, parameter_count, predict_grud, save_json, set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data_internal"
INTERNAL = ROOT / "03_internal_validation"
LOCKED = ROOT / "04_locked_models"
MACHINE = ROOT / "07_machine_readable"
LOGS = ROOT / "10_logs"
for folder in (INTERNAL, LOCKED, MACHINE, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def choose_mode(rows: pd.DataFrame, model: str) -> dict[str, Any]:
    subset = rows[rows.model == model].copy()
    counts = Counter(subset.params_key)
    top_n = max(counts.values())
    candidates = [key for key, count in counts.items() if count == top_n]
    ranking = []
    for key in candidates:
        hit = subset[subset.params_key == key]
        params = json.loads(key)
        capacity = (params.get("hidden_size", 0), params.get("dropout", 0), params.get("weight_decay", 0),
                    params.get("max_depth", 0), params.get("n_estimators", 0), params.get("C", 0))
        ranking.append((float(hit.validation_log_loss.mean()), float(hit.validation_brier.mean()), capacity, key))
    return json.loads(sorted(ranking)[0][-1])


def tune_logistic(x: np.ndarray, y: np.ndarray, model: str, seed: int) -> tuple[dict[str, Any], float, float]:
    if model == "model0_snapshot":
        candidates = [{"C": c, "penalty": "l2"} for c in (0.01, 0.1, 1.0, 10.0)]
    else:
        candidates = [{"C": c, "l1_ratio": ratio, "penalty": "elasticnet"}
                      for c, ratio in product((0.01, 0.1, 1.0, 10.0), (0.0, 0.5, 1.0))]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results = []
    for params in candidates:
        losses, briers = [], []
        for train, valid in splitter.split(x, y):
            prep = fit_tabular_preprocessor(x[train], scale=True)
            xt = apply_tabular_preprocessor(x[train], prep)
            xv = apply_tabular_preprocessor(x[valid], prep)
            if model == "model0_snapshot":
                estimator = LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs",
                                               max_iter=3000, random_state=seed)
            else:
                estimator = LogisticRegression(C=params["C"], penalty="elasticnet", l1_ratio=params["l1_ratio"],
                                               solver="saga", max_iter=2000, tol=1e-3, random_state=seed)
            estimator.fit(xt, y[train])
            p = estimator.predict_proba(xv)[:, 1]
            losses.append(log_loss(y[valid], p, labels=[0, 1]))
            briers.append(brier_score_loss(y[valid], p))
        results.append((float(np.mean(losses)), float(np.mean(briers)), params["C"],
                        params.get("l1_ratio", 0), params))
    best = sorted(results, key=lambda z: z[:4])[0]
    return best[-1], best[0], best[1]


def fit_predict_logistic(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                         model: str, params: dict[str, Any], seed: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
    prep = fit_tabular_preprocessor(x[train], scale=True)
    xt = apply_tabular_preprocessor(x[train], prep)
    xv = apply_tabular_preprocessor(x[test], prep)
    if model == "model0_snapshot":
        estimator = LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs", max_iter=3000,
                                       random_state=seed)
    else:
        estimator = LogisticRegression(C=params["C"], penalty="elasticnet", l1_ratio=params["l1_ratio"],
                                       solver="saga", max_iter=2000, tol=1e-3, random_state=seed)
    estimator.fit(xt, y[train])
    return estimator.predict_proba(xv)[:, 1], estimator, prep


def xgb_estimator(params: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=2,
        random_state=seed, max_depth=params["max_depth"], learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"], min_child_weight=params["min_child_weight"],
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    )


def tune_xgb(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, Any], float, float]:
    candidates = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=m)
                  for d, lr, n, m in product((2, 3), (0.03, 0.10), (100, 300), (5, 10))]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results = []
    for params in candidates:
        losses, briers = [], []
        for train, valid in splitter.split(x, y):
            prep = fit_tabular_preprocessor(x[train], scale=False)
            xt = apply_tabular_preprocessor(x[train], prep)
            xv = apply_tabular_preprocessor(x[valid], prep)
            estimator = xgb_estimator(params, seed)
            estimator.fit(xt, y[train])
            p = estimator.predict_proba(xv)[:, 1]
            losses.append(log_loss(y[valid], p, labels=[0, 1]))
            briers.append(brier_score_loss(y[valid], p))
        capacity = (params["max_depth"], params["n_estimators"], params["learning_rate"], params["min_child_weight"])
        results.append((float(np.mean(losses)), float(np.mean(briers)), capacity, params))
    best = sorted(results, key=lambda z: z[:3])[0]
    return best[-1], best[0], best[1]


def fit_predict_xgb(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                    params: dict[str, Any], seed: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
    prep = fit_tabular_preprocessor(x[train], scale=False)
    estimator = xgb_estimator(params, seed)
    estimator.fit(apply_tabular_preprocessor(x[train], prep), y[train])
    p = estimator.predict_proba(apply_tabular_preprocessor(x[test], prep))[:, 1]
    return p, estimator, prep


def tune_predict_grud(values: np.ndarray, mask: np.ndarray, delta: np.ndarray, age: np.ndarray, sex: np.ndarray,
                      y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int,
                      weighted: bool = True) -> tuple[np.ndarray, dict[str, Any], float, float, int, int]:
    inner_train, valid = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed)
    prep = fit_sequence_preprocessor(values[inner_train], age[inner_train])
    xtr, str_ = apply_sequence_preprocessor(values[inner_train], mask[inner_train], age[inner_train], sex[inner_train], prep)
    xva, sva = apply_sequence_preprocessor(values[valid], mask[valid], age[valid], sex[valid], prep)
    candidates = [dict(hidden_size=h, dropout=d, weight_decay=w)
                  for h, d, w in product((8, 16), (0.0, 0.2), (0.0, 1e-4))]
    fits = []
    for ci, params in enumerate(candidates):
        fit = fit_grud(xtr, mask[inner_train], delta[inner_train], str_, y[inner_train],
                       xva, mask[valid], delta[valid], sva, y[valid], seed=seed + ci,
                       weighted=weighted, **params)
        capacity = (params["hidden_size"], params["dropout"], params["weight_decay"])
        fits.append((fit.validation_log_loss, fit.validation_brier, capacity, params, fit))
    best_loss, best_brier, _, params, fit = sorted(fits, key=lambda z: z[:3])[0]
    full_prep = fit_sequence_preprocessor(values[train], age[train])
    xfull, sfull = apply_sequence_preprocessor(values[train], mask[train], age[train], sex[train], full_prep)
    xtest, stest = apply_sequence_preprocessor(values[test], mask[test], age[test], sex[test], full_prep)
    model = fit_grud_fixed_epochs(xfull, mask[train], delta[train], sfull, y[train],
                                  epochs=fit.best_epoch, seed=seed + 1000, weighted=weighted, **params)
    pred = predict_grud(model, xtest, mask[test], delta[test], stest)
    return pred, params, best_loss, best_brier, fit.best_epoch, fit.parameter_count


def main() -> None:
    start = time.time()
    set_seed(SEED)
    data = np.load(DATA / "mimic_model_data.npz")
    schema = json.loads((DATA / "feature_schema.json").read_text(encoding="utf-8"))
    ids, y = data["analysis_id"].astype(str), data["y"].astype(int)
    snapshot, summary = data["snapshot"].astype(float), data["summary"].astype(float)
    values, mask, delta = data["values"].astype(np.float32), data["mask"].astype(np.float32), data["delta"].astype(np.float32)
    age, sex = data["age"].astype(np.float32), data["sex"].astype(np.float32)
    if len(ids) != 1740 or int(y.sum()) != 168:
        raise RuntimeError("STOP_AND_DEBUG_COHORT before internal modeling")

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    oof_rows, tuning_rows, repeat_metric_rows = [], [], []
    fold_predictions = {name: np.full((5, len(y)), np.nan) for name in MODEL_NAMES}
    fold_assignment = np.full((5, len(y)), -1, dtype=int)
    for split_index, (train, test) in enumerate(splitter.split(snapshot, y)):
        repeat, fold = divmod(split_index, 5)
        fold_assignment[repeat, test] = fold
        seed = SEED + repeat * 100 + fold
        print(f"INTERNAL repeat={repeat + 1}/5 fold={fold + 1}/5", flush=True)

        params0, loss0, brier0 = tune_logistic(snapshot[train], y[train], "model0_snapshot", seed)
        p0, _, _ = fit_predict_logistic(snapshot, y, train, test, "model0_snapshot", params0, seed)
        fold_predictions["model0_snapshot"][repeat, test] = p0
        tuning_rows.append(dict(model="model0_snapshot", repeat=repeat + 1, fold=fold + 1,
                                params_key=candidate_key(params0), validation_log_loss=loss0, validation_brier=brier0))

        params1, loss1, brier1 = tune_logistic(summary[train], y[train], "model1_elastic_net", seed)
        p1, _, _ = fit_predict_logistic(summary, y, train, test, "model1_elastic_net", params1, seed)
        fold_predictions["model1_elastic_net"][repeat, test] = p1
        tuning_rows.append(dict(model="model1_elastic_net", repeat=repeat + 1, fold=fold + 1,
                                params_key=candidate_key(params1), validation_log_loss=loss1, validation_brier=brier1))

        params2, loss2, brier2 = tune_xgb(summary[train], y[train], seed)
        p2, _, _ = fit_predict_xgb(summary, y, train, test, params2, seed)
        fold_predictions["model2_xgboost"][repeat, test] = p2
        tuning_rows.append(dict(model="model2_xgboost", repeat=repeat + 1, fold=fold + 1,
                                params_key=candidate_key(params2), validation_log_loss=loss2, validation_brier=brier2))

        p3, params3, loss3, brier3, epoch3, count3 = tune_predict_grud(
            values, mask, delta, age, sex, y, train, test, seed, weighted=True)
        fold_predictions["model3_grud"][repeat, test] = p3
        params3_log = {**params3, "best_epoch": epoch3, "parameter_count": count3}
        tuning_rows.append(dict(model="model3_grud", repeat=repeat + 1, fold=fold + 1,
                                params_key=candidate_key(params3), validation_log_loss=loss3,
                                validation_brier=brier3, best_epoch=epoch3, parameter_count=count3))

    if any(np.isnan(array).any() for array in fold_predictions.values()) or np.any(fold_assignment < 0):
        raise RuntimeError("OOF completeness failure")
    for repeat in range(5):
        for i in range(len(y)):
            row = {"analysis_id": ids[i], "outcome": int(y[i]), "repeat": repeat + 1,
                   "fold": int(fold_assignment[repeat, i]) + 1}
            for name in MODEL_NAMES:
                row[name] = float(fold_predictions[name][repeat, i])
            oof_rows.append(row)
        for name in MODEL_NAMES:
            repeat_metric_rows.append({"repeat": repeat + 1, "model": name,
                                       **metric_values(y, fold_predictions[name][repeat])})

    oof = pd.DataFrame(oof_rows)
    tuning = pd.DataFrame(tuning_rows)
    oof.to_csv(MACHINE / "internal_oof_predictions.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(INTERNAL / "internal_tuning_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(repeat_metric_rows).to_csv(INTERNAL / "internal_repeat_metrics.csv", index=False, encoding="utf-8-sig")

    mean_predictions = {name: fold_predictions[name].mean(axis=0) for name in MODEL_NAMES}
    performance_rows, difference_rows = bootstrap_performance(y, mean_predictions, replicates=2000, seed=SEED + 7000)
    perf = pd.DataFrame(performance_rows)
    perf.insert(0, "dataset", "MIMIC-IV")
    perf.insert(1, "validation", "repeated_5x5_oof_patient_mean")
    perf.to_csv(MACHINE / "performance_internal.csv", index=False, encoding="utf-8-sig")
    internal_diff = pd.DataFrame(difference_rows)
    internal_diff.insert(0, "dataset", "MIMIC-IV")
    internal_diff.to_csv(INTERNAL / "internal_paired_differences.csv", index=False, encoding="utf-8-sig")

    selected = {model: choose_mode(tuning, model) for model in MODEL_NAMES}
    grud_rows = tuning[tuning.model == "model3_grud"].copy()
    chosen_key = candidate_key(selected["model3_grud"])
    epoch_source = grud_rows.loc[grud_rows.params_key == chosen_key, "best_epoch"].dropna()
    locked_epochs = int(np.median(epoch_source if len(epoch_source) else grud_rows.best_epoch.dropna()))
    selected["model3_grud"]["epochs"] = locked_epochs

    # Prespecified unweighted one-repeat sensitivity using the locked capacity and epochs.
    unweighted = np.full(len(y), np.nan)
    one_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(one_split.split(values, y)):
        prep = fit_sequence_preprocessor(values[train], age[train])
        xtr, str_ = apply_sequence_preprocessor(values[train], mask[train], age[train], sex[train], prep)
        xte, ste = apply_sequence_preprocessor(values[test], mask[test], age[test], sex[test], prep)
        params = selected["model3_grud"]
        model = fit_grud_fixed_epochs(xtr, mask[train], delta[train], str_, y[train],
                                      hidden_size=params["hidden_size"], dropout=params["dropout"],
                                      weight_decay=params["weight_decay"], epochs=locked_epochs,
                                      seed=SEED + 8000 + fold, weighted=False)
        unweighted[test] = predict_grud(model, xte, mask[test], delta[test], ste)
    sensitivity = pd.DataFrame({"analysis_id": ids, "outcome": y,
                                "model3_grud_unweighted_oof": unweighted})
    sensitivity.to_csv(INTERNAL / "model3_unweighted_sensitivity_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(metric_values(y, unweighted), INTERNAL / "model3_unweighted_sensitivity_metrics.json")

    # Full-MIMIC fits. No SICdb artifact is read in this script.
    final_artifacts: dict[str, list[Path]] = {}
    prep0 = fit_tabular_preprocessor(snapshot, scale=True)
    x0 = apply_tabular_preprocessor(snapshot, prep0)
    final0 = LogisticRegression(C=selected["model0_snapshot"]["C"], penalty="l2", solver="lbfgs",
                                max_iter=3000, random_state=SEED).fit(x0, y)
    p0_path, m0_path = LOCKED / "model0_preprocessor.json", LOCKED / "model0_snapshot.pkl"
    save_json(prep0, p0_path)
    with m0_path.open("wb") as handle:
        pickle.dump(final0, handle)
    final_artifacts["model0_snapshot"] = [p0_path, m0_path]

    prep1 = fit_tabular_preprocessor(summary, scale=True)
    x1 = apply_tabular_preprocessor(summary, prep1)
    final1 = LogisticRegression(C=selected["model1_elastic_net"]["C"], penalty="elasticnet",
                                l1_ratio=selected["model1_elastic_net"]["l1_ratio"], solver="saga",
                                max_iter=10000, tol=1e-4, random_state=SEED).fit(x1, y)
    p1_path, m1_path = LOCKED / "model1_preprocessor.json", LOCKED / "model1_elastic_net.pkl"
    save_json(prep1, p1_path)
    with m1_path.open("wb") as handle:
        pickle.dump(final1, handle)
    final_artifacts["model1_elastic_net"] = [p1_path, m1_path]

    prep2 = fit_tabular_preprocessor(summary, scale=False)
    final2 = xgb_estimator(selected["model2_xgboost"], SEED)
    final2.fit(apply_tabular_preprocessor(summary, prep2), y)
    p2_path, m2_path = LOCKED / "model2_preprocessor.json", LOCKED / "model2_xgboost.json"
    save_json(prep2, p2_path)
    final2.save_model(m2_path)
    final_artifacts["model2_xgboost"] = [p2_path, m2_path]

    prep3 = fit_sequence_preprocessor(values, age)
    x3, s3 = apply_sequence_preprocessor(values, mask, age, sex, prep3)
    params3 = selected["model3_grud"]
    final3 = fit_grud_fixed_epochs(x3, mask, delta, s3, y, hidden_size=params3["hidden_size"],
                                   dropout=params3["dropout"], weight_decay=params3["weight_decay"],
                                   epochs=locked_epochs, seed=SEED, weighted=True)
    p3_path, m3_path = LOCKED / "model3_preprocessor.json", LOCKED / "model3_grud.pt"
    save_json(prep3, p3_path)
    torch.save({"state_dict": final3.state_dict(), "hidden_size": params3["hidden_size"],
                "dropout": params3["dropout"], "weight_decay": params3["weight_decay"],
                "epochs": locked_epochs, "parameter_count": parameter_count(final3),
                "weighted": True}, m3_path)
    final_artifacts["model3_grud"] = [p3_path, m3_path]

    if parameter_count(final3) >= 20000:
        raise RuntimeError("Locked GRU-D exceeds parameter limit")
    manifest = {
        "status": "LOCKED_BEFORE_EXTERNAL_PREDICTION",
        "created_unix_time": time.time(), "seed": SEED, "development_database": "MIMIC-IV",
        "development_n": len(y), "development_deaths28": int(y.sum()),
        "external_database": "SICdb", "external_prediction_status_at_lock": "NOT_RUN",
        "feature_order": {"model0_snapshot": schema["snapshot_feature_order"],
                          "model1_elastic_net": schema["summary_feature_order"],
                          "model2_xgboost": schema["summary_feature_order"],
                          "model3_grud": {"signal_order": schema["signal_order"], "sequence_shape": [12, 3],
                                           "inputs": ["values", "mask", "time_since_last", "age", "sex_male"]}},
        "selected_hyperparameters": selected,
        "model3_parameter_count": parameter_count(final3),
        "artifacts": {
            model: [{"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]
            for model, paths in final_artifacts.items()
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "torch": torch.__version__},
        "script_sha256": sha256(Path(__file__)),
        "common_script_sha256": sha256(Path(__file__).with_name("modeling_common.py")),
    }
    manifest_path = LOCKED / "LOCKED_MODEL_MANIFEST.json"
    save_json(manifest, manifest_path)
    manifest_hash = sha256(manifest_path)
    (LOCKED / "LOCKED_MODEL_MANIFEST.sha256").write_text(
        f"{manifest_hash}  LOCKED_MODEL_MANIFEST.json\n", encoding="ascii")

    status = {
        "status": "SUCCESS", "elapsed_seconds": time.time() - start, "mimic_n": len(y),
        "mimic_deaths28": int(y.sum()), "oof_rows": len(oof), "expected_oof_rows": len(y) * 5,
        "each_patient_once_per_repeat": bool((oof.groupby(["analysis_id", "repeat"]).size() == 1).all()),
        "selected_hyperparameters": selected, "model3_parameter_count": parameter_count(final3),
        "manifest_sha256": manifest_hash, "external_data_read": False,
    }
    save_json(status, LOGS / "internal_modeling_status.json")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
