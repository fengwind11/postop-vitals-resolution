from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from xgboost import DMatrix, XGBClassifier


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
FORMAL = PROJECT / "04_formal_modeling"
SIGNAL = PROJECT / "03_stage05_signal_audit"
WORK = PROJECT / "09_prewrite_work"
OUT = WORK / "analysis_outputs"
LOCK = WORK / "unweighted_lock"
LOGS = WORK / "logs"
for folder in (OUT, LOCK, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FORMAL / "00_scripts"))
from modeling_common import (  # noqa: E402
    SEED,
    CompactGRUD,
    _irls,
    apply_sequence_preprocessor,
    apply_tabular_preprocessor,
    bootstrap_performance,
    calibration_metrics,
    fit_grud_fixed_epochs,
    fit_sequence_preprocessor,
    metric_values,
    parameter_count,
    predict_grud,
)


MODELS = ("model0_snapshot", "model1_elastic_net", "model2_xgboost", "model3_grud")
METRICS = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope")
PERFORMANCE_METRICS = ("auroc", "auprc", "brier", "log_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_est_ci(est: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{est:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def fmt_median_iqr(values: np.ndarray, digits: int = 1, percent: bool = False) -> tuple[float, float, float, str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    q25, med, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
    if percent:
        q25, med, q75 = 100 * q25, 100 * med, 100 * q75
    return float(med), float(q25), float(q75), f"{med:.{digits}f} [{q25:.{digits}f}, {q75:.{digits}f}]"


def bootstrap_one(y: np.ndarray, p: np.ndarray, seed: int) -> pd.DataFrame:
    rows, _ = bootstrap_performance(
        y, {"model3_grud_unweighted": p}, replicates=2000, seed=seed, compute_differences=False
    )
    return pd.DataFrame(rows)


def paired_contrast(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    contrast: str,
    seed: int,
    dataset: str,
    validation: str,
) -> pd.DataFrame:
    point_a = metric_values(y, a)
    point_b = metric_values(y, b)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in PERFORMANCE_METRICS}
    valid = 0
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        valid += 1
        ma = metric_values(y[idx], a[idx])
        mb = metric_values(y[idx], b[idx])
        for metric in PERFORMANCE_METRICS:
            values[metric].append(ma[metric] - mb[metric])
    rows = []
    for metric in PERFORMANCE_METRICS:
        v = np.asarray(values[metric])
        rows.append(
            {
                "dataset": dataset,
                "validation": validation,
                "contrast": contrast,
                "metric": metric,
                "estimate": point_a[metric] - point_b[metric],
                "ci_low": float(np.quantile(v, 0.025)),
                "ci_high": float(np.quantile(v, 0.975)),
                "bootstrap_replicates": valid,
                "paired_patient_resampling": True,
            }
        )
    return pd.DataFrame(rows)


def calibration_bins(y: np.ndarray, predictions: dict[str, np.ndarray], representation: str) -> pd.DataFrame:
    rows = []
    for model, pred in predictions.items():
        groups = pd.qcut(pred, q=10, labels=False, duplicates="drop")
        frame = pd.DataFrame({"bin": groups, "risk": pred, "outcome": y})
        for group, part in frame.groupby("bin", dropna=False):
            rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "bin": int(group),
                    "n": len(part),
                    "events": int(part.outcome.sum()),
                    "mean_predicted": float(part.risk.mean()),
                    "observed": float(part.outcome.mean()),
                }
            )
    return pd.DataFrame(rows)


def unweighted_internal_and_lock(manifest: dict) -> tuple[np.ndarray, dict, torch.nn.Module]:
    mimic_path = FORMAL / "02_data_internal" / "mimic_model_data.npz"
    data = np.load(mimic_path)
    ids = data["analysis_id"].astype(str)
    y = data["y"].astype(int)
    values = data["values"].astype(np.float32)
    mask = data["mask"].astype(np.float32)
    delta = data["delta"].astype(np.float32)
    age = data["age"].astype(np.float32)
    sex = data["sex"].astype(np.float32)
    assert len(y) == 1740 and int(y.sum()) == 168
    params = manifest["selected_hyperparameters"]["model3_grud"]

    # Exact reproduction of the existing one-repeat, five-fold sensitivity.
    pred = np.full(len(y), np.nan)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(values, y)):
        print(f"UNWEIGHTED_INTERNAL fold={fold + 1}/5", flush=True)
        prep = fit_sequence_preprocessor(values[train], age[train])
        xtr, str_ = apply_sequence_preprocessor(values[train], mask[train], age[train], sex[train], prep)
        xte, ste = apply_sequence_preprocessor(values[test], mask[test], age[test], sex[test], prep)
        model = fit_grud_fixed_epochs(
            xtr,
            mask[train],
            delta[train],
            str_,
            y[train],
            hidden_size=params["hidden_size"],
            dropout=params["dropout"],
            weight_decay=params["weight_decay"],
            epochs=params["epochs"],
            seed=SEED + 8000 + fold,
            weighted=False,
        )
        pred[test] = predict_grud(model, xte, mask[test], delta[test], ste)
    assert np.isfinite(pred).all()
    reproduced = pd.DataFrame(
        {"analysis_id": ids, "outcome": y, "model3_grud_unweighted_oof": pred}
    )
    reproduced.to_csv(OUT / "unweighted_internal_5fold_predictions_reproduced.csv", index=False, encoding="utf-8-sig")
    existing_path = FORMAL / "03_internal_validation" / "model3_unweighted_sensitivity_predictions.csv"
    existing = pd.read_csv(existing_path)
    merged = reproduced.merge(existing, on=["analysis_id", "outcome"], validate="one_to_one", suffixes=("_new", "_existing"))
    max_abs = float(
        np.max(
            np.abs(
                merged["model3_grud_unweighted_oof_new"].to_numpy()
                - merged["model3_grud_unweighted_oof_existing"].to_numpy()
            )
        )
    )
    if max_abs > 1e-12:
        raise RuntimeError(f"STOP_AND_DEBUG unweighted MIMIC reproduction max_abs={max_abs}")
    perf = bootstrap_one(y, pred, SEED + 45000)
    perf.insert(0, "dataset", "MIMIC-IV")
    perf.insert(1, "validation", "single_5fold_oof_locked_unweighted")
    perf.insert(2, "n", len(y))
    perf.insert(3, "events", int(y.sum()))
    perf.to_csv(OUT / "unweighted_internal_performance.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "status": "PASS_EXACT_REPRODUCTION",
            "n": len(y),
            "events": int(y.sum()),
            "max_absolute_prediction_difference": max_abs,
            "existing_prediction_sha256": sha256(existing_path),
            "reproduced_prediction_sha256": sha256(OUT / "unweighted_internal_5fold_predictions_reproduced.csv"),
            "existing_metric_values": metric_values(y, existing["model3_grud_unweighted_oof"].to_numpy()),
            "reproduced_metric_values": metric_values(y, pred),
        },
        LOGS / "unweighted_internal_reproduction.json",
    )

    # Full-MIMIC fit, locked before any SICdb artifact is opened.
    prep = fit_sequence_preprocessor(values, age)
    x, static = apply_sequence_preprocessor(values, mask, age, sex, prep)
    model = fit_grud_fixed_epochs(
        x,
        mask,
        delta,
        static,
        y,
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
        weight_decay=params["weight_decay"],
        epochs=params["epochs"],
        seed=SEED,
        weighted=False,
    )
    prep_path = LOCK / "model3_unweighted_preprocessor.json"
    model_path = LOCK / "model3_unweighted_grud.pt"
    write_json(prep, prep_path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": params["hidden_size"],
            "dropout": params["dropout"],
            "weight_decay": params["weight_decay"],
            "epochs": params["epochs"],
            "parameter_count": parameter_count(model),
            "weighted": False,
            "positive_class_weight": 1.0,
        },
        model_path,
    )
    lock_manifest = {
        "status": "LOCKED_BEFORE_EXTERNAL_PREDICTION",
        "sensitivity_role": "SUPPLEMENTARY_ONLY_DO_NOT_REPLACE_PRIMARY_WEIGHTED_GRUD",
        "seed": SEED,
        "development_database": "MIMIC-IV",
        "development_n": len(y),
        "development_deaths28": int(y.sum()),
        "external_database": "SICdb",
        "external_prediction_status_at_lock": "NOT_RUN",
        "only_change_from_primary": "positive_class_weight_set_to_1",
        "selected_hyperparameters": params,
        "parameter_count": parameter_count(model),
        "input_sha256": sha256(mimic_path),
        "primary_locked_manifest_sha256": sha256(FORMAL / "04_locked_models" / "LOCKED_MODEL_MANIFEST.json"),
        "artifacts": {
            prep_path.name: sha256(prep_path),
            model_path.name: sha256(model_path),
        },
    }
    lock_manifest_path = LOCK / "UNWEIGHTED_GRUD_LOCK_MANIFEST.json"
    write_json(lock_manifest, lock_manifest_path)
    (LOCK / "UNWEIGHTED_GRUD_LOCK_MANIFEST.sha256").write_text(
        f"{sha256(lock_manifest_path)}  {lock_manifest_path.name}\n", encoding="ascii"
    )
    return pred, prep, model


def unweighted_external(prep: dict, model: torch.nn.Module) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = FORMAL / "02_data_internal" / "sicdb_external_data.npz"
    data = np.load(data_path)
    ids = data["analysis_id"].astype(str)
    y = data["y"].astype(int)
    assert len(y) == 2376 and int(y.sum()) == 185
    representations = [
        "hourly_median",
        "sparse_phase_00",
        "sparse_phase_15",
        "sparse_phase_30",
        "sparse_phase_45",
        "sparse_phase_59",
    ]
    prediction_rows = []
    performance_rows = []
    calibration_parts = []
    for ri, representation in enumerate(representations):
        prefix = "" if representation == "hourly_median" else f"phase_{representation.split('_')[-1]}_"
        values = data["values"] if not prefix else data[prefix + "values"]
        mask = data["mask"] if not prefix else data[prefix + "mask"]
        delta = data["delta"] if not prefix else data[prefix + "delta"]
        x, static = apply_sequence_preprocessor(values, mask, data["age"], data["sex"], prep)
        pred = predict_grud(model, x, mask, delta, static)
        print(f"UNWEIGHTED_EXTERNAL representation={representation}", flush=True)
        for i, analysis_id in enumerate(ids):
            prediction_rows.append(
                {
                    "analysis_id": analysis_id,
                    "outcome": int(y[i]),
                    "representation": representation,
                    "model3_grud_unweighted": float(pred[i]),
                }
            )
        perf = bootstrap_one(y, pred, SEED + 46000 + ri)
        perf.insert(0, "dataset", "SICdb")
        perf.insert(1, "representation", representation)
        perf.insert(2, "n", len(y))
        perf.insert(3, "events", int(y.sum()))
        performance_rows.append(perf)
        calibration_parts.append(
            calibration_bins(y, {"model3_grud_unweighted": pred}, representation)
        )
    predictions = pd.DataFrame(prediction_rows)
    performance = pd.concat(performance_rows, ignore_index=True)
    predictions.to_csv(OUT / "unweighted_external_predictions.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(OUT / "unweighted_external_performance.csv", index=False, encoding="utf-8-sig")
    pd.concat(calibration_parts, ignore_index=True).to_csv(
        OUT / "unweighted_external_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )
    lock_manifest_path = LOCK / "UNWEIGHTED_GRUD_LOCK_MANIFEST.json"
    locked = json.loads(lock_manifest_path.read_text(encoding="utf-8"))
    locked["external_prediction_completed"] = True
    locked["external_input_sha256"] = sha256(data_path)
    locked["lock_manifest_sha256_used_for_external"] = sha256(lock_manifest_path)
    write_json(locked, LOGS / "unweighted_external_status.json")
    return predictions, performance


def weighted_unweighted_comparison(
    internal_unweighted: np.ndarray, external_unweighted: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    internal = pd.read_csv(FORMAL / "07_machine_readable" / "internal_oof_predictions.csv")
    first = internal[internal.repeat == 1].copy()
    assert len(first) == 1740 and first.analysis_id.nunique() == 1740
    y = first.outcome.to_numpy(int)
    weighted = first.model3_grud.to_numpy(float)
    for label, p in (("weighted", weighted), ("unweighted", internal_unweighted)):
        perf = bootstrap_one(y, p, SEED + (47000 if label == "weighted" else 47001))
        perf["dataset"] = "MIMIC-IV"
        perf["validation"] = "single_5fold_oof_matched_folds"
        perf["training_loss"] = label
        rows.append(perf)
    rows.append(
        paired_contrast(
            y,
            internal_unweighted,
            weighted,
            "unweighted_minus_weighted",
            SEED + 47002,
            "MIMIC-IV",
            "single_5fold_oof_matched_folds",
        ).assign(training_loss="paired_difference")
    )

    weighted_external = pd.read_csv(FORMAL / "07_machine_readable" / "external_predictions_sicdb.csv")
    weighted_external = weighted_external[weighted_external.representation == "hourly_median"].copy()
    unweighted_external = external_unweighted[external_unweighted.representation == "hourly_median"].copy()
    merged = weighted_external.merge(
        unweighted_external,
        on=["analysis_id", "outcome", "representation"],
        validate="one_to_one",
    )
    y = merged.outcome.to_numpy(int)
    weighted = merged.model3_grud.to_numpy(float)
    unweighted = merged.model3_grud_unweighted.to_numpy(float)
    for label, p in (("weighted", weighted), ("unweighted", unweighted)):
        perf = bootstrap_one(y, p, SEED + (48000 if label == "weighted" else 48001))
        perf["dataset"] = "SICdb"
        perf["validation"] = "locked_external_hourly_median"
        perf["training_loss"] = label
        rows.append(perf)
    rows.append(
        paired_contrast(
            y,
            unweighted,
            weighted,
            "unweighted_minus_weighted",
            SEED + 48002,
            "SICdb",
            "locked_external_hourly_median",
        ).assign(training_loss="paired_difference")
    )
    result = pd.concat(rows, ignore_index=True, sort=False)
    result.to_csv(OUT / "weighted_vs_unweighted_grud.csv", index=False, encoding="utf-8-sig")
    calibration_bins(
        y,
        {"weighted": weighted, "unweighted": unweighted},
        "SICdb_hourly_median",
    ).to_csv(OUT / "weighted_vs_unweighted_calibration_bins.csv", index=False, encoding="utf-8-sig")
    return result


def unweighted_resolution(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(FORMAL / "02_data_internal" / "sicdb_resolution_data.npz")
    ids = data["analysis_id"].astype(str)
    y = data["y"].astype(int)
    age = data["age"].astype(np.float32)
    sex = data["sex"].astype(np.float32)
    assert len(y) == 2061 and int(y.sum()) == 151
    params = manifest["selected_hyperparameters"]["model3_grud"]
    widths = (5, 15, 60)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    assignments = np.full((5, len(y)), -1, dtype=int)
    oof = {width: np.full((5, len(y)), np.nan) for width in widths}
    counts = set()
    for split_index, (train, test) in enumerate(splitter.split(age, y)):
        repeat, fold = divmod(split_index, 5)
        assignments[repeat, test] = fold
        seed = SEED + 30000 + repeat * 100 + fold
        print(f"UNWEIGHTED_RESOLUTION repeat={repeat + 1}/5 fold={fold + 1}/5", flush=True)
        for width in widths:
            values = data[f"values_{width}min"].astype(np.float32)
            mask = data[f"mask_{width}min"].astype(np.float32)
            delta = data[f"delta_{width}min"].astype(np.float32)
            prep = fit_sequence_preprocessor(values[train], age[train])
            xtr, str_ = apply_sequence_preprocessor(values[train], mask[train], age[train], sex[train], prep)
            xte, ste = apply_sequence_preprocessor(values[test], mask[test], age[test], sex[test], prep)
            model = fit_grud_fixed_epochs(
                xtr,
                mask[train],
                delta[train],
                str_,
                y[train],
                hidden_size=params["hidden_size"],
                dropout=params["dropout"],
                weight_decay=params["weight_decay"],
                epochs=params["epochs"],
                seed=seed,
                weighted=False,
            )
            counts.add(parameter_count(model))
            oof[width][repeat, test] = predict_grud(model, xte, mask[test], delta[test], ste)
    if len(counts) != 1 or any(np.isnan(v).any() for v in oof.values()) or np.any(assignments < 0):
        raise RuntimeError("STOP_AND_DEBUG unweighted resolution completeness")
    pred_rows = []
    for repeat in range(5):
        for width in widths:
            for i, analysis_id in enumerate(ids):
                pred_rows.append(
                    {
                        "analysis_id": analysis_id,
                        "outcome": int(y[i]),
                        "repeat": repeat + 1,
                        "fold": int(assignments[repeat, i]) + 1,
                        "resolution_min": width,
                        "model3_grud_unweighted": float(oof[width][repeat, i]),
                    }
                )
    pd.DataFrame(pred_rows).to_csv(
        OUT / "unweighted_resolution_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    mean = {width: oof[width].mean(axis=0) for width in widths}
    perf_parts = []
    for width in widths:
        perf = bootstrap_one(y, mean[width], SEED + 49000 + width)
        perf.insert(0, "resolution_min", width)
        perf.insert(1, "n", len(y))
        perf.insert(2, "events", int(y.sum()))
        perf_parts.append(perf)
    performance = pd.concat(perf_parts, ignore_index=True)
    performance.to_csv(OUT / "unweighted_resolution_performance.csv", index=False, encoding="utf-8-sig")
    paired = paired_contrast(
        y,
        mean[5],
        mean[60],
        "5min_minus_60min",
        SEED + 49060,
        "SICdb_dense_core",
        "repeated_5x5_oof_unweighted_grud",
    )
    paired.to_csv(OUT / "unweighted_resolution_5minus60.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "status": "SUCCESS",
            "sensitivity_role": "SUPPLEMENTARY_ONLY",
            "n": len(y),
            "events": int(y.sum()),
            "resolutions_min": list(widths),
            "same_patient_set": True,
            "same_fold_assignments": True,
            "parameter_counts": sorted(counts),
            "weighted": False,
        },
        LOGS / "unweighted_resolution_status.json",
    )
    return performance, paired


def temporal_snapshot_contrasts() -> pd.DataFrame:
    internal = pd.read_csv(FORMAL / "07_machine_readable" / "internal_oof_predictions.csv")
    internal_mean = (
        internal.groupby("analysis_id", sort=False)
        .agg(
            outcome=("outcome", "first"),
            model0_snapshot=("model0_snapshot", "mean"),
            model1_elastic_net=("model1_elastic_net", "mean"),
            model2_xgboost=("model2_xgboost", "mean"),
        )
        .reset_index()
    )
    assert len(internal_mean) == 1740 and int(internal_mean.outcome.sum()) == 168
    frames = []
    for i, model in enumerate(("model1_elastic_net", "model2_xgboost")):
        frames.append(
            paired_contrast(
                internal_mean.outcome.to_numpy(int),
                internal_mean[model].to_numpy(float),
                internal_mean.model0_snapshot.to_numpy(float),
                f"{model}_minus_model0_snapshot",
                SEED + 7000 + i,
                "MIMIC-IV",
                "repeated_5x5_oof_patient_mean",
            )
        )
    external = pd.read_csv(FORMAL / "07_machine_readable" / "external_predictions_sicdb.csv")
    external = external[external.representation == "hourly_median"].copy()
    assert len(external) == 2376 and int(external.outcome.sum()) == 185
    for i, model in enumerate(("model1_elastic_net", "model2_xgboost")):
        frames.append(
            paired_contrast(
                external.outcome.to_numpy(int),
                external[model].to_numpy(float),
                external.model0_snapshot.to_numpy(float),
                f"{model}_minus_model0_snapshot",
                SEED + 9000 + i,
                "SICdb",
                "locked_external_hourly_median",
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "temporal_vs_snapshot_paired_contrasts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"dataset": "MIMIC-IV", "n": 1740, "events": 168, "event_rate": 168 / 1740},
            {"dataset": "SICdb", "n": 2376, "events": 185, "event_rate": 185 / 2376},
        ]
    ).to_csv(OUT / "auprc_no_skill_baselines.csv", index=False, encoding="utf-8-sig")
    return result


def external_recalibration_diagnostics() -> pd.DataFrame:
    external = pd.read_csv(FORMAL / "07_machine_readable" / "external_predictions_sicdb.csv")
    external = external[external.representation == "hourly_median"].copy()
    y = external.outcome.to_numpy(float)
    rows = []
    pred_out = external[["analysis_id", "outcome"]].copy()
    for model in MODELS:
        p = np.clip(external[model].to_numpy(float), 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p))
        intercept_only = float(_irls(y, np.ones((len(y), 1)), offset=lp)[0])
        p_intercept = 1 / (1 + np.exp(-(lp + intercept_only)))
        joint = _irls(y, np.column_stack([np.ones(len(y)), lp]))
        p_joint = 1 / (1 + np.exp(-(joint[0] + joint[1] * lp)))
        for method, pred in (
            ("original_locked", p),
            ("intercept_only_apparent_diagnostic", p_intercept),
            ("intercept_plus_slope_apparent_diagnostic", p_joint),
        ):
            vals = metric_values(y, pred)
            rows.append(
                {
                    "dataset": "SICdb",
                    "representation": "hourly_median",
                    "model": model,
                    "recalibration": method,
                    "n": len(y),
                    "events": int(y.sum()),
                    "fitted_intercept": intercept_only if method == "intercept_only_apparent_diagnostic" else (float(joint[0]) if method == "intercept_plus_slope_apparent_diagnostic" else np.nan),
                    "fitted_slope": float(joint[1]) if method == "intercept_plus_slope_apparent_diagnostic" else np.nan,
                    **{metric: vals[metric] for metric in METRICS},
                }
            )
        pred_out[f"{model}__original"] = p
        pred_out[f"{model}__intercept_only"] = p_intercept
        pred_out[f"{model}__intercept_plus_slope"] = p_joint
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "external_recalibration_diagnostics.csv", index=False, encoding="utf-8-sig")
    pred_out.to_csv(OUT / "external_recalibrated_predictions_diagnostic.csv", index=False, encoding="utf-8-sig")
    return result


def feature_importance_source(manifest: dict) -> pd.DataFrame:
    schema = json.loads((FORMAL / "02_data_internal" / "feature_schema.json").read_text(encoding="utf-8"))
    names = schema["summary_feature_order"]
    with (FORMAL / "04_locked_models" / "model1_elastic_net.pkl").open("rb") as handle:
        elastic = pickle.load(handle)
    coef = np.asarray(elastic.coef_).reshape(-1)
    data = np.load(FORMAL / "02_data_internal" / "mimic_model_data.npz")
    prep = json.loads((FORMAL / "04_locked_models" / "model2_preprocessor.json").read_text(encoding="utf-8"))
    x = apply_tabular_preprocessor(data["summary"].astype(float), prep)
    xgb = XGBClassifier()
    xgb.load_model(FORMAL / "04_locked_models" / "model2_xgboost.json")
    contrib = xgb.get_booster().predict(DMatrix(x), pred_contribs=True)
    mean_abs_shap = np.mean(np.abs(contrib[:, :-1]), axis=0)
    result = pd.DataFrame(
        {
            "feature": names,
            "elastic_net_standardized_coefficient": coef,
            "elastic_net_absolute_coefficient": np.abs(coef),
            "xgboost_mean_absolute_shap": mean_abs_shap,
            "interpretation_boundary": "descriptive_model_interpretation_not_causal",
        }
    )
    result.to_csv(OUT / "feature_importance_elasticnet_shap.csv", index=False, encoding="utf-8-sig")
    return result


def descriptive_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = json.loads((FORMAL / "02_data_internal" / "feature_schema.json").read_text(encoding="utf-8"))
    names = schema["summary_feature_order"][:23]
    long_rows = []
    table1_rows = []
    datasets = {
        "MIMIC-IV": np.load(FORMAL / "02_data_internal" / "mimic_model_data.npz"),
        "SICdb": np.load(FORMAL / "02_data_internal" / "sicdb_external_data.npz"),
    }
    display = {"Characteristic": [], "MIMIC-IV": [], "SICdb": []}
    characteristics: dict[str, dict[str, str]] = {}
    for dataset, data in datasets.items():
        y, age, sex = data["y"].astype(int), data["age"].astype(float), data["sex"].astype(float)
        summary = data["summary"][:, :23].astype(float)
        counts = data["counts"].astype(float)
        n, events = len(y), int(y.sum())
        rows = {
            "N": f"{n}",
            "28-day deaths, n (%)": f"{events} ({100 * events / n:.1f})",
            "Age, years, median [IQR]": fmt_median_iqr(age, 1)[3],
            "Male sex, n (%)": f"{int(sex.sum())} ({100 * sex.mean():.1f})",
        }
        table1_rows.extend(
            [
                {"dataset": dataset, "characteristic": "N", "n": n, "events": events, "value": n},
                {"dataset": dataset, "characteristic": "28-day deaths", "n": n, "events": events, "value": events, "percent": 100 * events / n},
                {"dataset": dataset, "characteristic": "Age", "n": n, "events": events, "median": float(np.median(age)), "q25": float(np.quantile(age, .25)), "q75": float(np.quantile(age, .75))},
                {"dataset": dataset, "characteristic": "Male sex", "n": n, "events": events, "value": int(sex.sum()), "percent": 100 * sex.mean()},
            ]
        )
        for j, signal in enumerate(("HR", "MAP", "SpO2")):
            base = 2 + j * 7
            fields = [
                ("12-h median", summary[:, base], 1, False),
                ("12-h extreme", summary[:, base + 1], 1, False),
                ("threshold burden, %", summary[:, base + 4], 1, True),
                ("valid-hour fraction, %", summary[:, base + 5], 1, True),
                ("observations/hour", counts[:, :, j].sum(axis=1) / 12, 1, False),
            ]
            for label, values, digits, percent in fields:
                med, q25, q75, rendered = fmt_median_iqr(values, digits, percent)
                characteristic = f"{signal} {label}, median [IQR]"
                rows[characteristic] = rendered
                table1_rows.append(
                    {"dataset": dataset, "characteristic": characteristic, "n": n, "events": events, "median": med, "q25": q25, "q75": q75}
                )
        characteristics[dataset] = rows
        for i, name in enumerate(names):
            values = summary[:, i]
            finite = values[np.isfinite(values)]
            long_rows.append(
                {
                    "dataset": dataset,
                    "feature": name,
                    "n": n,
                    "nonmissing_n": len(finite),
                    "missing_percent": 100 * (1 - len(finite) / n),
                    "median": float(np.median(finite)),
                    "q25": float(np.quantile(finite, 0.25)),
                    "q75": float(np.quantile(finite, 0.75)),
                    "mean": float(np.mean(finite)),
                    "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                }
            )
    order = list(characteristics["MIMIC-IV"])
    for characteristic in order:
        display["Characteristic"].append(characteristic)
        display["MIMIC-IV"].append(characteristics["MIMIC-IV"][characteristic])
        display["SICdb"].append(characteristics["SICdb"][characteristic])
    pd.DataFrame(display).to_csv(OUT / "Table_1_cohort_and_measurement_profile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(table1_rows).to_csv(OUT / "Table_1_source_values.csv", index=False, encoding="utf-8-sig")
    full = pd.DataFrame(long_rows)
    full.to_csv(OUT / "Supplement_Table_S10_temporal_feature_distributions.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(table1_rows), full


def performance_tables() -> None:
    internal = pd.read_csv(FORMAL / "07_machine_readable" / "performance_internal.csv")
    external = pd.read_csv(FORMAL / "07_machine_readable" / "performance_external.csv")
    frames = []
    for dataset, n, events, perf in (("MIMIC-IV", 1740, 168, internal), ("SICdb", 2376, 185, external)):
        for model in MODELS:
            row = {"dataset": dataset, "n": n, "events": events, "event_rate_percent": 100 * events / n, "model": model}
            part = perf[perf.model == model]
            for metric in METRICS:
                hit = part[part.metric == metric].iloc[0]
                row[metric] = fmt_est_ci(hit.estimate, hit.ci_low, hit.ci_high)
            frames.append(row)
    pd.DataFrame(frames).to_csv(OUT / "Table_2_internal_external_performance.csv", index=False, encoding="utf-8-sig")

    perf = pd.read_csv(FORMAL / "07_machine_readable" / "resolution_performance.csv")
    diff = pd.read_csv(FORMAL / "07_machine_readable" / "bootstrap_differences_resolution.csv")
    rows = []
    for (width, model), part in perf.groupby(["resolution_min", "model"], sort=False):
        row = {"resolution_min": width, "n": 2061, "events": 151, "model": model}
        for metric in METRICS:
            hit = part[part.metric == metric].iloc[0]
            row[metric] = fmt_est_ci(hit.estimate, hit.ci_low, hit.ci_high)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "Table_3A_resolution_performance.csv", index=False, encoding="utf-8-sig")
    formatted = diff.copy()
    formatted["estimate_95ci"] = formatted.apply(lambda r: fmt_est_ci(r.estimate, r.ci_low, r.ci_high), axis=1)
    formatted.to_csv(OUT / "Table_3B_paired_5minus60.csv", index=False, encoding="utf-8-sig")


def supporting_tables(manifest: dict, temporal: pd.DataFrame, recal: pd.DataFrame, weighted_unweighted: pd.DataFrame) -> None:
    # S2 feature dictionary.
    schema = json.loads((FORMAL / "02_data_internal" / "feature_schema.json").read_text(encoding="utf-8"))
    dictionary = []
    for feature in schema["summary_feature_order"]:
        if feature.startswith("missing__"):
            source = feature.removeprefix("missing__")
            definition = f"Binary indicator that {source} was missing before development-set imputation"
            unit = "0/1"
        elif feature in ("age", "sex_male"):
            definition = "Age at ICU admission" if feature == "age" else "Male sex indicator"
            unit = "years" if feature == "age" else "0/1"
        else:
            signal, suffix = feature.split("_", 1)
            rules = {
                "median": "Median of the 12 hourly medians",
                "extreme": "Maximum hourly median for HR; minimum hourly median for MAP and SpO2",
                "sd": "Sample SD of available hourly medians",
                "slope_per_hour": "OLS slope across hourly midpoint times when at least two hours were available",
                "threshold_burden": "Proportion of available hours beyond the fixed threshold (HR>100, MAP<65, SpO2<92)",
                "valid_hour_fraction": "Available hourly medians divided by 12",
                "longest_missing_streak_h": "Longest run of missing hourly medians, in hours",
            }
            definition = rules[suffix]
            unit = {"HR": "beats/min", "MAP": "mmHg", "SpO2": "%"}[signal]
            if suffix in ("threshold_burden", "valid_hour_fraction"):
                unit = "proportion"
            elif suffix == "longest_missing_streak_h":
                unit = "hours"
            elif suffix == "slope_per_hour":
                unit = unit + " per hour"
        dictionary.append({"feature": feature, "definition": definition, "unit": unit, "model_use": "Elastic Net and XGBoost"})
    pd.DataFrame(dictionary).to_csv(OUT / "Supplement_Table_S2_feature_dictionary.csv", index=False, encoding="utf-8-sig")

    # S3 combines prespecified coverage and density summaries without patient-level rows.
    coverage = pd.read_csv(SIGNAL / "02_coverage" / "Table_A_signal_coverage.csv")
    density = pd.read_csv(SIGNAL / "03_density" / "Table_C_density.csv")
    dense = pd.read_csv(SIGNAL / "03_density" / "Table_D_sicdb_dense_support.csv")
    coverage.assign(section="coverage").to_csv(OUT / "Supplement_Table_S3A_signal_coverage.csv", index=False, encoding="utf-8-sig")
    density.assign(section="density").to_csv(OUT / "Supplement_Table_S3B_signal_density.csv", index=False, encoding="utf-8-sig")
    dense.assign(section="dense_core").to_csv(OUT / "Supplement_Table_S3C_dense_core.csv", index=False, encoding="utf-8-sig")

    # S4 hyperparameters and tuning-history summary.
    hp_rows = []
    for model, params in manifest["selected_hyperparameters"].items():
        hp_rows.append({"model": model, "selected_hyperparameters": json.dumps(params, sort_keys=True), "selection_rule": "modal outer-fold choice; ties resolved by validation log loss, Brier score, then lower capacity"})
    hp_rows.append({"model": "model3_grud", "selected_hyperparameters": f"parameter_count={manifest['model3_parameter_count']}", "selection_rule": "compact architecture remained below 20,000 parameters"})
    pd.DataFrame(hp_rows).to_csv(OUT / "Supplement_Table_S4A_locked_hyperparameters.csv", index=False, encoding="utf-8-sig")
    tuning = pd.read_csv(FORMAL / "03_internal_validation" / "internal_tuning_history.csv")
    tuning_summary = (
        tuning.groupby(["model", "params_key"], dropna=False)
        .agg(selection_count=("fold", "size"), validation_log_loss_mean=("validation_log_loss", "mean"), validation_brier_mean=("validation_brier", "mean"), best_epoch_median=("best_epoch", "median"), parameter_count=("parameter_count", "max"))
        .reset_index()
    )
    tuning_summary.to_csv(OUT / "Supplement_Table_S4B_tuning_history_summary.csv", index=False, encoding="utf-8-sig")

    temporal.to_csv(OUT / "Supplement_Table_S5_temporal_vs_snapshot.csv", index=False, encoding="utf-8-sig")
    recal.to_csv(OUT / "Supplement_Table_S6_external_recalibration_diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(FORMAL / "07_machine_readable" / "sparse_phase_external.csv").to_csv(
        OUT / "Supplement_Table_S7_sparse_phase_performance.csv", index=False, encoding="utf-8-sig"
    )
    weighted_unweighted.to_csv(OUT / "Supplement_Table_S8_weighted_vs_unweighted_grud.csv", index=False, encoding="utf-8-sig")
    dca = pd.read_csv(FORMAL / "05_external_validation" / "decision_curve_external.csv")
    keep = np.isclose(dca.threshold.to_numpy()[:, None], np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25])[None, :]).any(axis=1)
    dca[keep].to_csv(OUT / "Supplement_Table_S9_decision_curve_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    started = time.time()
    formal_manifest_path = FORMAL / "04_locked_models" / "LOCKED_MODEL_MANIFEST.json"
    manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    if manifest["development_n"] != 1740 or manifest["development_deaths28"] != 168:
        raise RuntimeError("STOP_AND_DEBUG MIMIC lock count mismatch")
    schema = json.loads((FORMAL / "02_data_internal" / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["dense_core_n"] != 2061 or schema["dense_core_deaths28"] != 151:
        raise RuntimeError("STOP_AND_DEBUG dense-core count mismatch")
    external_counts = pd.read_csv(FORMAL / "02_data_internal" / "sicdb_formal_counts.csv").iloc[0]
    if int(external_counts.n) != 2376 or int(external_counts.deaths28) != 185:
        raise RuntimeError("STOP_AND_DEBUG SICdb count mismatch")

    internal_unweighted, unweighted_prep, unweighted_model = unweighted_internal_and_lock(manifest)
    external_unweighted, _ = unweighted_external(unweighted_prep, unweighted_model)
    weighted_unweighted = weighted_unweighted_comparison(internal_unweighted, external_unweighted)
    unweighted_resolution(manifest)
    temporal = temporal_snapshot_contrasts()
    recal = external_recalibration_diagnostics()
    feature_importance_source(manifest)
    descriptive_tables()
    performance_tables()
    supporting_tables(manifest, temporal, recal, weighted_unweighted)

    source_hashes = {
        "formal_delivery_zip": "not_in_public_source_release; see analysis chronology",
        "signal_audit_delivery_zip": "not_in_public_source_release; see analysis chronology",
        "sap_freeze": "not_in_public_source_release; see analysis chronology",
        "locked_model_manifest": sha256(formal_manifest_path),
        "run_prewrite_analysis_script": sha256(SCRIPT),
    }
    write_json(
        {
            "status": "SUCCESS",
            "elapsed_seconds": time.time() - started,
            "counts": {
                "mimic_n": 1740,
                "mimic_deaths28": 168,
                "sicdb_n": 2376,
                "sicdb_deaths28": 185,
                "sicdb_dense_core_n": 2061,
                "sicdb_dense_core_deaths28": 151,
            },
            "source_hashes": source_hashes,
            "authorized_new_analyses_only": [
                "fixed unweighted GRU-D sensitivity",
                "temporal-summary versus snapshot paired contrasts",
                "descriptive tables, recalibration diagnostic, and model-interpretation summaries from locked artifacts",
            ],
            "forbidden_model_search_performed": False,
        },
        LOGS / "prewrite_analysis_status.json",
    )
    print(json.dumps({"status": "SUCCESS", "elapsed_seconds": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
