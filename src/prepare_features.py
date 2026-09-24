from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260918
CORE = ("HR", "MAP", "SpO2")
PHASES = (0, 15, 30, 45, 59)
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT_SCRIPT = PROJECT / "03_stage05_signal_audit" / "00_scripts" / "analyze_signal_audit.py"
FORMAL_DATA = ROOT / "02_data_internal"
QC = ROOT / "01_extract_qc"
MACHINE = ROOT / "07_machine_readable"
LOGS = ROOT / "10_logs"
for folder in (QC, MACHINE, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audit_module():
    spec = importlib.util.spec_from_file_location("signal_audit_frozen", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen signal-audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bin_signal(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    if 720 % width:
        raise ValueError(width)
    blocks = values.reshape(values.shape[0], 720 // width, width)
    counts = np.isfinite(blocks).sum(axis=2).astype(np.int16)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(blocks, axis=2).astype(np.float32)
    required = math.ceil(width / 2) if width > 1 else 1
    medians[counts < required] = np.nan
    return medians, counts


def hourly_common(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vals, counts = [], []
    for signal in CORE:
        block = signals[signal].reshape(signals[signal].shape[0], 12, 60)
        count = np.isfinite(block).sum(axis=2).astype(np.int16)
        with np.errstate(all="ignore"):
            median = np.nanmedian(block, axis=2).astype(np.float32)
        median[count == 0] = np.nan
        vals.append(median)
        counts.append(count)
    return np.stack(vals, axis=2), np.stack(counts, axis=2)


def delta_hours(mask: np.ndarray) -> np.ndarray:
    delta = np.zeros(mask.shape, dtype=np.float32)
    for t in range(1, mask.shape[1]):
        delta[:, t, :] = 1.0 + (1.0 - mask[:, t - 1, :]) * delta[:, t - 1, :]
    return delta


def longest_missing(row: np.ndarray) -> int:
    best = current = 0
    for observed in np.isfinite(row):
        if observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def sequence_summary(values: np.ndarray, age: np.ndarray, sex: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = values.shape[0]
    data: dict[str, np.ndarray] = {
        "age": age.astype(np.float32),
        "sex_male": sex.astype(np.float32),
    }
    times = np.arange(values.shape[1], dtype=float) + 0.5
    for j, signal in enumerate(CORE):
        x = values[:, :, j].astype(float)
        valid = np.isfinite(x)
        with np.errstate(all="ignore"):
            data[f"{signal}_median"] = np.nanmedian(x, axis=1)
            data[f"{signal}_extreme"] = np.nanmax(x, axis=1) if signal == "HR" else np.nanmin(x, axis=1)
            data[f"{signal}_sd"] = np.nanstd(x, axis=1, ddof=1)
        slopes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            use = valid[i]
            if use.sum() >= 2:
                slopes[i] = np.polyfit(times[use], x[i, use], 1)[0]
        data[f"{signal}_slope_per_hour"] = slopes
        if signal == "HR":
            event = x > 100
        elif signal == "MAP":
            event = x < 65
        else:
            event = x < 92
        burden = np.full(n, np.nan, dtype=float)
        denom = valid.sum(axis=1)
        ok = denom > 0
        burden[ok] = (event & valid).sum(axis=1)[ok] / denom[ok]
        data[f"{signal}_threshold_burden"] = burden
        data[f"{signal}_valid_hour_fraction"] = denom / values.shape[1]
        data[f"{signal}_longest_missing_streak_h"] = np.array([longest_missing(row) for row in x], dtype=float)
    names = list(data)
    matrix = np.column_stack([data[name] for name in names]).astype(np.float32)
    missing = np.isnan(matrix).astype(np.float32)
    missing_names = [f"missing__{name}" for name in names]
    return np.column_stack([matrix, missing]).astype(np.float32), names + missing_names


def snapshot_features(values: np.ndarray, age: np.ndarray, sex: np.ndarray) -> tuple[np.ndarray, list[str]]:
    h0 = values[:, 0, :].astype(np.float32)
    matrix = np.column_stack([age, sex, h0, np.isnan(h0).astype(np.float32)]).astype(np.float32)
    names = ["age", "sex_male", "H0_HR", "H0_MAP", "H0_SpO2",
             "missing_H0_HR", "missing_H0_MAP", "missing_H0_SpO2"]
    return matrix, names


def sparse_sequences(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for phase in PHASES:
        indexes = np.arange(12) * 60 + phase
        output[f"phase_{phase:02d}"] = np.stack([signals[s][:, indexes] for s in CORE], axis=2).astype(np.float32)
    return output


def resolution_sequences(signals: dict[str, np.ndarray], widths=(5, 15, 60)) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for width in widths:
        per_signal = [bin_signal(signals[s], width)[0] for s in CORE]
        output[width] = np.stack(per_signal, axis=2).astype(np.float32)
    return output


def sex_mimic(series: pd.Series) -> np.ndarray:
    value = series.astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    if value.isna().any():
        raise RuntimeError("STOP_AND_DEBUG_STATIC: unmapped MIMIC sex")
    return value.to_numpy(dtype=np.float32)


def sex_sicdb(series: pd.Series) -> np.ndarray:
    value = series.astype(str).str.lower().map({"male": 1.0, "female": 0.0})
    if value.isna().any():
        raise RuntimeError("STOP_AND_DEBUG_STATIC: unmapped SICdb sex")
    return value.to_numpy(dtype=np.float32)


def leakage_rows(dataset: str, ids: np.ndarray, signals: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    keys = [hashlib.sha256(f"{SEED}|leak|{dataset}|{x}".encode()).hexdigest() for x in ids]
    for i in np.argsort(keys)[:50]:
        row = {"dataset": dataset, "analysis_id": str(ids[i])}
        maximum = -1
        for signal in CORE:
            finite = np.flatnonzero(np.isfinite(signals[signal][i]))
            value = int(finite.max()) if finite.size else -1
            row[f"{signal}_max_minute"] = value
            maximum = max(maximum, value)
        row["predictor_max_minute"] = maximum
        row["strictly_before_landmark"] = maximum < 720
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    audit = load_audit_module()
    hourly = pd.read_csv(audit.INPUT / "sicdb_signals_hourly_raw.csv")
    m_base, m_combined, _, _ = audit.build_mimic()
    s_base, s_combined, _, _, _ = audit.build_sicdb(hourly)

    m_formal = pd.read_csv(FORMAL_DATA / "mimic_formal_cohort.csv")
    s_formal = pd.read_csv(FORMAL_DATA / "sicdb_formal_cohort.csv")
    m = m_base[["stay_id", "analysis_id"]].merge(m_formal, on="stay_id", validate="one_to_one")
    s = s_base[["caseid", "analysis_id"]].merge(s_formal, on="caseid", validate="one_to_one")
    if len(m) != 1740 or int(m.death28.sum()) != 168 or m.subject_id.nunique() != 1740:
        raise RuntimeError("STOP_AND_DEBUG_COHORT: MIMIC mismatch after feature merge")
    if len(s) != 2376 or int(s.death28.sum()) != 185 or s.patientid.nunique() != 2376:
        raise RuntimeError("STOP_AND_DEBUG_COHORT: SICdb mismatch after feature merge")

    m_age = pd.to_numeric(m.age_at_icu, errors="coerce").to_numpy(dtype=np.float32)
    s_age = pd.to_numeric(s.age_on_admission, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(m_age).all() or not np.isfinite(s_age).all():
        raise RuntimeError("STOP_AND_DEBUG_STATIC: age missing")
    m_sex, s_sex = sex_mimic(m.gender), sex_sicdb(s.sex_label)

    m_hour, m_counts = hourly_common(m_combined)
    s_hour, s_counts = hourly_common(s_combined)
    m_mask, s_mask = np.isfinite(m_hour).astype(np.float32), np.isfinite(s_hour).astype(np.float32)
    m_delta, s_delta = delta_hours(m_mask), delta_hours(s_mask)
    m_snap, snap_names = snapshot_features(m_hour, m_age, m_sex)
    s_snap, snap_names_s = snapshot_features(s_hour, s_age, s_sex)
    m_summary, summary_names = sequence_summary(m_hour, m_age, m_sex)
    s_summary, summary_names_s = sequence_summary(s_hour, s_age, s_sex)
    if snap_names != snap_names_s or summary_names != summary_names_s:
        raise RuntimeError("STOP_AND_DEBUG_SCHEMA: MIMIC/SICdb feature order differs")

    sparse = sparse_sequences(s_combined)
    sparse_payload: dict[str, np.ndarray] = {}
    for label, seq in sparse.items():
        snap, names0 = snapshot_features(seq, s_age, s_sex)
        summary, names1 = sequence_summary(seq, s_age, s_sex)
        if names0 != snap_names or names1 != summary_names:
            raise RuntimeError("STOP_AND_DEBUG_SCHEMA: sparse feature order differs")
        sparse_payload[f"{label}_values"] = seq
        sparse_payload[f"{label}_mask"] = np.isfinite(seq).astype(np.float32)
        sparse_payload[f"{label}_delta"] = delta_hours(np.isfinite(seq).astype(np.float32))
        sparse_payload[f"{label}_snapshot"] = snap
        sparse_payload[f"{label}_summary"] = summary

    support, dense_patient, dense_mask = audit.dense_support(s_base, s_combined)
    if int(dense_mask.sum()) != 2061:
        raise RuntimeError(f"STOP_AND_DEBUG_DENSE_CORE: got {int(dense_mask.sum())}, expected 2061")
    five_support = int(dense_patient.five_min_representation_support.sum())
    if five_support != 2105:
        raise RuntimeError(f"STOP_AND_DEBUG_5MIN_SUPPORT: got {five_support}, expected 2105")
    dense_deaths = int(s.loc[dense_mask, "death28"].sum())
    resolutions = resolution_sequences(s_combined)
    resolution_payload: dict[str, np.ndarray] = {}
    for width, seq_all in resolutions.items():
        seq = seq_all[dense_mask]
        summary, names1 = sequence_summary(seq, s_age[dense_mask], s_sex[dense_mask])
        if names1 != summary_names:
            raise RuntimeError("STOP_AND_DEBUG_SCHEMA: resolution feature order differs")
        resolution_payload[f"values_{width}min"] = seq
        resolution_payload[f"mask_{width}min"] = np.isfinite(seq).astype(np.float32)
        resolution_payload[f"delta_{width}min"] = delta_hours(np.isfinite(seq).astype(np.float32)) * (width / 60.0)
        resolution_payload[f"summary_{width}min"] = summary

    leakage = pd.concat([
        leakage_rows("MIMIC-IV", m.analysis_id.to_numpy(), m_combined),
        leakage_rows("SICdb", s.analysis_id.to_numpy(), s_combined),
    ], ignore_index=True)
    if len(leakage) != 100 or not leakage.strictly_before_landmark.all():
        raise RuntimeError("STOP_AND_DEBUG_LEAKAGE: sampled predictor at/after landmark")
    leakage.to_csv(QC / "random50_each_database_time_leakage_check.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        FORMAL_DATA / "mimic_model_data.npz",
        analysis_id=m.analysis_id.to_numpy(dtype="U16"), y=m.death28.to_numpy(dtype=np.int8),
        age=m_age, sex=m_sex, values=m_hour, counts=m_counts, mask=m_mask, delta=m_delta,
        snapshot=m_snap, summary=m_summary,
    )
    np.savez_compressed(
        FORMAL_DATA / "sicdb_external_data.npz",
        analysis_id=s.analysis_id.to_numpy(dtype="U16"), y=s.death28.to_numpy(dtype=np.int8),
        age=s_age, sex=s_sex, values=s_hour, counts=s_counts, mask=s_mask, delta=s_delta,
        snapshot=s_snap, summary=s_summary, **sparse_payload,
    )
    np.savez_compressed(
        FORMAL_DATA / "sicdb_resolution_data.npz",
        analysis_id=s.loc[dense_mask, "analysis_id"].to_numpy(dtype="U16"),
        y=s.loc[dense_mask, "death28"].to_numpy(dtype=np.int8),
        age=s_age[dense_mask], sex=s_sex[dense_mask], **resolution_payload,
    )
    schema = {
        "seed": SEED,
        "signal_order": list(CORE),
        "snapshot_feature_order": snap_names,
        "summary_feature_order": summary_names,
        "sparse_phases_minute": list(PHASES),
        "cleaning_bounds": {"HR": [20, 250], "MAP": [20, 200], "SpO2": [50, 100]},
        "primary_shape": [12, 3],
        "dense_core_n": int(dense_mask.sum()),
        "dense_core_deaths28": dense_deaths,
        "five_min_support_n": five_support,
        "max_predictor_minute": int(leakage.predictor_max_minute.max()),
    }
    (FORMAL_DATA / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    availability_rows = []
    for dataset, ids, y, values, counts in (
        ("MIMIC-IV", m.analysis_id, m.death28, m_hour, m_counts),
        ("SICdb", s.analysis_id, s.death28, s_hour, s_counts),
    ):
        for j, signal in enumerate(CORE):
            availability_rows.append({
                "dataset": dataset, "signal": signal, "n": len(ids), "deaths28": int(y.sum()),
                "available_hour_percent": float(100 * np.isfinite(values[:, :, j]).mean()),
                "patient_valid_hours_median": float(np.median(np.isfinite(values[:, :, j]).sum(axis=1))),
                "observations_per_hour_median": float(np.median(counts[:, :, j].sum(axis=1) / 12)),
            })
    pd.DataFrame(availability_rows).to_csv(MACHINE / "table1_signal_availability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"dataset": "MIMIC-IV", "n": len(m), "deaths28": int(m.death28.sum()),
         "age_median": float(np.median(m_age)), "age_q25": float(np.quantile(m_age, .25)),
         "age_q75": float(np.quantile(m_age, .75)), "male_n": int(m_sex.sum())},
        {"dataset": "SICdb", "n": len(s), "deaths28": int(s.death28.sum()),
         "age_median": float(np.median(s_age)), "age_q25": float(np.quantile(s_age, .25)),
         "age_q75": float(np.quantile(s_age, .75)), "male_n": int(s_sex.sum())},
    ]).to_csv(MACHINE / "table1_cohort_descriptives.csv", index=False, encoding="utf-8-sig")

    artifacts = [FORMAL_DATA / "mimic_model_data.npz", FORMAL_DATA / "sicdb_external_data.npz",
                 FORMAL_DATA / "sicdb_resolution_data.npz", FORMAL_DATA / "feature_schema.json",
                 QC / "random50_each_database_time_leakage_check.csv"]
    manifest = {
        "status": "SUCCESS", "seed": SEED, "mimic_n": len(m), "mimic_deaths28": int(m.death28.sum()),
        "sicdb_n": len(s), "sicdb_deaths28": int(s.death28.sum()),
        "dense_core_n": int(dense_mask.sum()), "dense_core_deaths28": dense_deaths,
        "feature_schema_identical": True, "sampled_time_leakage": False,
        "artifacts": [{"file": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts],
        "script_sha256": sha256(Path(__file__)),
        "frozen_audit_script_sha256": sha256(AUDIT_SCRIPT),
    }
    (LOGS / "feature_preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
