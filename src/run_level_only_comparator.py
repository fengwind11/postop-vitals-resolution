from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression


PROJECT = Path(__file__).resolve().parents[1]
FORMAL = PROJECT / "04_formal_modeling"
SCRIPTS = FORMAL / "00_scripts"
DATA = FORMAL / "02_data_internal"
MACHINE = FORMAL / "07_machine_readable"
OUT = Path(__file__).resolve().parent / "level_only_comparator"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))
from modeling_common import (  # noqa: E402
    SEED,
    apply_tabular_preprocessor,
    bootstrap_performance,
    fit_tabular_preprocessor,
    metric_values,
    save_json,
)
from run_internal_and_lock import (  # noqa: E402
    candidate_key,
    choose_mode,
    fit_predict_logistic,
    tune_logistic,
)


LEVEL_FEATURES = [
    "age",
    "sex_male",
    "HR_median",
    "MAP_median",
    "SpO2_median",
    "missing__age",
    "missing__sex_male",
    "missing__HR_median",
    "missing__MAP_median",
    "missing__SpO2_median",
]
MODEL_NAME = "model12h_level_only"
FULL_NAME = "model12h_full_temporal_l1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_level_matrix(summary: np.ndarray, feature_order: list[str]) -> tuple[np.ndarray, list[int]]:
    missing = [name for name in LEVEL_FEATURES if name not in feature_order]
    if missing:
        raise RuntimeError(f"Required level-only features missing from frozen schema: {missing}")
    indices = [feature_order.index(name) for name in LEVEL_FEATURES]
    return summary[:, indices].astype(float), indices


def align_patient_mean_predictions(
    table: pd.DataFrame,
    ids: np.ndarray,
    prediction_col: str,
) -> np.ndarray:
    grouped = table.groupby("analysis_id", sort=False).agg(
        outcome=("outcome", "first"), prediction=(prediction_col, "mean")
    )
    if grouped.index.duplicated().any():
        raise RuntimeError("Duplicate patient rows after prediction aggregation")
    aligned = grouped.reindex(ids)
    if aligned.isna().any().any():
        raise RuntimeError("Prediction alignment produced missing rows")
    return aligned["prediction"].to_numpy(dtype=float)


def bootstrap_tables(
    y: np.ndarray,
    full_prediction: np.ndarray,
    level_prediction: np.ndarray,
    seed: int,
    dataset: str,
    validation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance, differences = bootstrap_performance(
        y,
        {FULL_NAME: full_prediction, MODEL_NAME: level_prediction},
        replicates=2000,
        seed=seed,
        compute_differences=True,
        reference=FULL_NAME,
    )
    perf = pd.DataFrame(performance)
    perf.insert(0, "dataset", dataset)
    perf.insert(1, "validation", validation)
    perf.insert(2, "n", len(y))
    perf.insert(3, "events", int(y.sum()))
    diff = pd.DataFrame(differences)
    diff.insert(0, "dataset", dataset)
    diff.insert(1, "validation", validation)
    return perf, diff


def main() -> None:
    started = time.time()
    schema_path = DATA / "feature_schema.json"
    mimic_path = DATA / "mimic_model_data.npz"
    sicdb_path = DATA / "sicdb_external_data.npz"
    internal_oof_path = MACHINE / "internal_oof_predictions.csv"
    external_pred_path = MACHINE / "external_predictions_sicdb.csv"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_order = list(schema["summary_feature_order"])

    mimic = np.load(mimic_path)
    ids = mimic["analysis_id"].astype(str)
    y = mimic["y"].astype(int)
    level, feature_indices = get_level_matrix(mimic["summary"], feature_order)
    if len(ids) != 1740 or int(y.sum()) != 168:
        raise RuntimeError("Frozen MIMIC cohort signature mismatch")

    original_oof = pd.read_csv(internal_oof_path, dtype={"analysis_id": str})
    expected = len(ids) * 5
    if len(original_oof) != expected:
        raise RuntimeError(f"Frozen OOF row count mismatch: {len(original_oof)} != {expected}")
    if set(original_oof["analysis_id"]) != set(ids):
        raise RuntimeError("Frozen OOF identifiers do not match MIMIC NPZ")

    fold_predictions = np.full((5, len(y)), np.nan)
    tuning_rows: list[dict[str, object]] = []
    repeat_metric_rows: list[dict[str, object]] = []
    id_to_index = {analysis_id: index for index, analysis_id in enumerate(ids)}

    for repeat in range(1, 6):
        repeat_rows = original_oof.loc[original_oof["repeat"] == repeat, ["analysis_id", "fold"]]
        if len(repeat_rows) != len(ids) or repeat_rows["analysis_id"].duplicated().any():
            raise RuntimeError(f"Invalid frozen fold map for repeat {repeat}")
        fold_map = np.full(len(ids), -1, dtype=int)
        for row in repeat_rows.itertuples(index=False):
            fold_map[id_to_index[str(row.analysis_id)]] = int(row.fold)
        if np.any(fold_map < 1):
            raise RuntimeError(f"Incomplete frozen fold map for repeat {repeat}")

        for fold in range(1, 6):
            test = np.where(fold_map == fold)[0]
            train = np.where(fold_map != fold)[0]
            seed = SEED + (repeat - 1) * 100 + (fold - 1)
            params, val_loss, val_brier = tune_logistic(
                level[train], y[train], MODEL_NAME, seed
            )
            prediction, _, _ = fit_predict_logistic(
                level, y, train, test, MODEL_NAME, params, seed
            )
            fold_predictions[repeat - 1, test] = prediction
            tuning_rows.append(
                {
                    "model": MODEL_NAME,
                    "repeat": repeat,
                    "fold": fold,
                    "params_key": candidate_key(params),
                    "validation_log_loss": val_loss,
                    "validation_brier": val_brier,
                }
            )

        if np.isnan(fold_predictions[repeat - 1]).any():
            raise RuntimeError(f"Incomplete level-only OOF predictions for repeat {repeat}")
        repeat_metric_rows.append(
            {
                "dataset": "MIMIC-IV",
                "repeat": repeat,
                "model": MODEL_NAME,
                **metric_values(y, fold_predictions[repeat - 1]),
            }
        )

    tuning = pd.DataFrame(tuning_rows)
    selected = choose_mode(tuning, MODEL_NAME)
    mean_level = fold_predictions.mean(axis=0)
    mean_full = align_patient_mean_predictions(original_oof, ids, "model1_elastic_net")

    oof_rows = []
    for repeat in range(1, 6):
        for i, analysis_id in enumerate(ids):
            oof_rows.append(
                {
                    "analysis_id": analysis_id,
                    "outcome": int(y[i]),
                    "repeat": repeat,
                    MODEL_NAME: float(fold_predictions[repeat - 1, i]),
                }
            )
    pd.DataFrame(oof_rows).to_csv(OUT / "level_only_internal_oof_predictions.csv", index=False)
    tuning.to_csv(OUT / "level_only_internal_tuning_history.csv", index=False)
    pd.DataFrame(repeat_metric_rows).to_csv(OUT / "level_only_internal_repeat_metrics.csv", index=False)

    internal_perf, internal_diff = bootstrap_tables(
        y,
        mean_full,
        mean_level,
        SEED + 12000,
        "MIMIC-IV",
        "repeated_5x5_oof_patient_mean",
    )
    internal_perf.to_csv(OUT / "level_only_internal_performance.csv", index=False)
    internal_diff.to_csv(OUT / "full_vs_level_only_internal_paired_bootstrap.csv", index=False)

    prep = fit_tabular_preprocessor(level, scale=True)
    transformed = apply_tabular_preprocessor(level, prep)
    final_model = LogisticRegression(
        C=selected["C"],
        penalty="elasticnet",
        l1_ratio=selected["l1_ratio"],
        solver="saga",
        max_iter=10000,
        tol=1e-4,
        random_state=SEED,
    ).fit(transformed, y)
    prep_path = OUT / "model12h_level_only_preprocessor.json"
    model_path = OUT / "model12h_level_only.pkl"
    save_json(prep, prep_path)
    with model_path.open("wb") as handle:
        pickle.dump(final_model, handle)

    sicdb = np.load(sicdb_path)
    sicdb_ids = sicdb["analysis_id"].astype(str)
    y_ext = sicdb["y"].astype(int)
    level_ext, ext_indices = get_level_matrix(sicdb["summary"], feature_order)
    if feature_indices != ext_indices:
        raise RuntimeError("MIMIC and SICdb level-only feature index mismatch")
    if len(sicdb_ids) != 2376 or int(y_ext.sum()) != 185:
        raise RuntimeError("Frozen SICdb cohort signature mismatch")
    pred_level_ext = final_model.predict_proba(
        apply_tabular_preprocessor(level_ext, prep)
    )[:, 1]

    original_external = pd.read_csv(external_pred_path, dtype={"analysis_id": str})
    original_external = original_external.loc[
        original_external["representation"] == "hourly_median"
    ].copy()
    full_external = align_patient_mean_predictions(
        original_external, sicdb_ids, "model1_elastic_net"
    )
    aligned_outcomes = (
        original_external.set_index("analysis_id").reindex(sicdb_ids)["outcome"].to_numpy(dtype=int)
    )
    if not np.array_equal(aligned_outcomes, y_ext):
        raise RuntimeError("SICdb outcome alignment mismatch")

    pd.DataFrame(
        {
            "analysis_id": sicdb_ids,
            "outcome": y_ext,
            "representation": "hourly_median",
            MODEL_NAME: pred_level_ext,
        }
    ).to_csv(OUT / "level_only_external_predictions_sicdb.csv", index=False)

    external_perf, external_diff = bootstrap_tables(
        y_ext,
        full_external,
        pred_level_ext,
        SEED + 13000,
        "SICdb",
        "locked_external_hourly_median",
    )
    external_perf.to_csv(OUT / "level_only_external_performance.csv", index=False)
    external_diff.to_csv(OUT / "full_vs_level_only_external_paired_bootstrap.csv", index=False)

    result = {
        "status": "SUCCESS",
        "analysis_authorization": "12-h level-only comparator only",
        "seed": SEED,
        "bootstrap_replicates": 2000,
        "level_only_feature_order": LEVEL_FEATURES,
        "level_only_feature_indices_in_frozen_summary": feature_indices,
        "same_frozen_outer_folds": True,
        "same_preprocessing_framework": True,
        "original_hyperparameter_grid_only": True,
        "selected_hyperparameters": selected,
        "mimic_n": len(y),
        "mimic_events": int(y.sum()),
        "sicdb_n": len(y_ext),
        "sicdb_events": int(y_ext.sum()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "elapsed_seconds": time.time() - started,
    }
    save_json(result, OUT / "level_only_comparator_status.json")

    input_paths = [schema_path, mimic_path, sicdb_path, internal_oof_path, external_pred_path]
    output_paths = [
        prep_path,
        model_path,
        OUT / "level_only_internal_tuning_history.csv",
        OUT / "level_only_internal_repeat_metrics.csv",
        OUT / "level_only_internal_performance.csv",
        OUT / "full_vs_level_only_internal_paired_bootstrap.csv",
        OUT / "level_only_external_performance.csv",
        OUT / "full_vs_level_only_external_paired_bootstrap.csv",
        OUT / "level_only_comparator_status.json",
    ]
    manifest = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths
        ],
    }
    save_json(manifest, OUT / "level_only_comparator_manifest.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
