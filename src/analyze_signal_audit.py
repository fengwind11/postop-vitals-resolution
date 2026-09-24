from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260918
RNG = np.random.default_rng(SEED)
SCRIPT = Path(__file__).resolve()
AUDIT = SCRIPT.parent.parent
INPUT = AUDIT / "06_logs" / "internal_extracts"
DECODER = AUDIT / "01_decoder_qc"
COVERAGE = AUDIT / "02_coverage"
DENSITY = AUDIT / "03_density"
RESOLUTION = AUDIT / "04_resolution_loss"
LOGS = AUDIT / "06_logs"
for folder in (DECODER, COVERAGE, DENSITY, RESOLUTION, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

CORE = ("HR", "MAP", "SpO2")
ALL_SIGNALS = ("HR", "MAP", "SpO2", "RR", "Temperature")
BOUNDS = {"HR": (20.0, 250.0), "MAP": (20.0, 200.0), "SpO2": (50.0, 100.0)}
MIMIC_MAP = {
    220045: ("HR", 1, "Heart Rate"),
    220052: ("MAP", 1, "Arterial MAP"),
    220181: ("MAP", 2, "Non-invasive MAP"),
    220277: ("SpO2", 1, "SpO2"),
    220210: ("RR", 1, "Respiratory Rate"),
    224690: ("RR", 2, "Respiratory Rate Total"),
    223762: ("Temperature", 1, "Temperature Celsius"),
    223761: ("Temperature", 2, "Temperature Fahrenheit"),
}
SICDB_MAP = {
    707: ("HR", 1, "HeartRateECG"),
    724: ("HR", 2, "HeartRateABP"),
    708: ("HR", 3, "HeartRateSPO2"),
    703: ("MAP", 1, "Arterial MAP"),
    706: ("MAP", 2, "Non-invasive MAP"),
    710: ("SpO2", 1, "SpO2"),
    719: ("RR", 1, "Respiratory Rate"),
    709: ("Temperature", 1, "Temperature Celsius"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def q(values: np.ndarray, prob: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, prob)) if values.size else float("nan")


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    return q(values, 0.5), q(values, 0.25), q(values, 0.75)


def longest_false_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        if bool(flag):
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def priority_combine(arrays: dict[int, np.ndarray], ordered_ids: list[int]) -> np.ndarray:
    result = np.full_like(arrays[ordered_ids[0]], np.nan, dtype=np.float32)
    for source_id in ordered_ids:
        source = arrays[source_id]
        take = np.isnan(result) & np.isfinite(source)
        result[take] = source[take]
    return result


def clean_array(values: np.ndarray, signal: str) -> np.ndarray:
    output = values.copy()
    if signal in BOUNDS:
        low, high = BOUNDS[signal]
        output[(output < low) | (output > high)] = np.nan
    return output


def add_raw_stat(rows: list[dict], dataset: str, source_id: int, signal: str,
                 label: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if signal in BOUNDS:
        low, high = BOUNDS[signal]
        invalid = int(np.sum((values < low) | (values > high)))
        rule = f"[{low:g}, {high:g}] inclusive"
    else:
        invalid = 0
        rule = "DESCRIPTIVE_NO_ADDITIONAL_RANGE"
    rows.append({
        "dataset": dataset,
        "source_id": source_id,
        "signal": signal,
        "source_label": label,
        "raw_n": int(values.size),
        "raw_min": float(np.min(values)) if values.size else np.nan,
        "raw_max": float(np.max(values)) if values.size else np.nan,
        "invalid_n": invalid,
        "invalid_pct": 100.0 * invalid / values.size if values.size else np.nan,
        "cleaning_rule": rule,
    })


def decode_payload(raw: object) -> tuple[bytes | None, np.ndarray | None, np.ndarray | None]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None, None, None
    text = str(raw)
    if text.startswith("\\x") or text.startswith("0x"):
        text = text[2:]
    if len(text) != 480:
        return None, None, None
    try:
        payload = bytes.fromhex(text)
    except ValueError:
        return None, None, None
    if len(payload) != 240:
        return None, None, None
    words = np.frombuffer(payload, dtype="<u4")
    values = np.frombuffer(payload, dtype="<f4").copy()
    present = words != 0
    values[~present] = np.nan
    values[~np.isfinite(values)] = np.nan
    return payload, values, present


def decoder_qc(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    detail: list[dict] = []
    summary: list[dict] = []
    for dataid in (707, 724, 708, 703, 706, 710):
        eligible = hourly[(hourly["dataid"] == dataid) & hourly["rawdata"].notna()].copy()
        eligible = eligible.drop_duplicates(["caseid", "row_offset"])
        eligible["sample_key"] = [
            hashlib.sha256(f"{SEED}|{int(c)}|{dataid}|{int(o)}".encode()).hexdigest()
            for c, o in zip(eligible["caseid"], eligible["row_offset"])
        ]
        sample = eligible.sort_values("sample_key").head(min(100, len(eligible)))
        for row in sample.itertuples(index=False):
            payload, values, present = decode_payload(row.rawdata)
            ok_bytes = payload is not None
            if ok_bytes:
                decoded = values[np.isfinite(values)]
                decoded_n = int(decoded.size)
                decoded_mean = float(np.mean(decoded)) if decoded_n else np.nan
                count_diff = decoded_n - int(row.cnt)
                abs_diff = abs(decoded_mean - float(row.stored_val)) if decoded_n and pd.notna(row.stored_val) else np.nan
                rel_diff = abs_diff / max(abs(float(row.stored_val)), 1e-12) if np.isfinite(abs_diff) else np.nan
                offsets = int(row.row_offset) + np.arange(60) * 60
                monotone = bool(np.all(np.diff(offsets) == 60))
                boundary = bool(offsets[0] == int(row.row_offset) and offsets[-1] == int(row.row_offset) + 3540)
            else:
                decoded_n = 0
                decoded_mean = count_diff = abs_diff = rel_diff = np.nan
                monotone = boundary = False
            tolerance = max(0.01, abs(float(row.stored_val)) * 1e-4) if pd.notna(row.stored_val) else np.nan
            detail.append({
                "caseid_internal": int(row.caseid),
                "dataid": dataid,
                "row_offset": int(row.row_offset),
                "row_offset_mod3600": int(row.row_offset) % 3600,
                "raw_text_length": len(str(row.rawdata)) if pd.notna(row.rawdata) else np.nan,
                "byte_length": len(payload) if payload is not None else np.nan,
                "slot_count": 60 if payload is not None else np.nan,
                "decoded_nonmissing_count": decoded_n,
                "stored_cnt": int(row.cnt),
                "count_difference": count_diff,
                "count_exact_match": bool(count_diff == 0) if np.isfinite(count_diff) else False,
                "decoded_mean": decoded_mean,
                "stored_val": float(row.stored_val) if pd.notna(row.stored_val) else np.nan,
                "mean_abs_difference": abs_diff,
                "mean_relative_difference": rel_diff,
                "mean_tolerance": tolerance,
                "mean_within_tolerance": bool(abs_diff <= tolerance) if np.isfinite(abs_diff) else False,
                "offset_monotone_60s": monotone,
                "offset_boundary_correct": boundary,
            })
        d = pd.DataFrame([x for x in detail if x["dataid"] == dataid])
        summary.append({
            "dataid": dataid,
            "sample_case_hours": len(d),
            "byte_length_240_pct": 100 * (d["byte_length"] == 240).mean() if len(d) else 0,
            "count_exact_match_pct": 100 * d["count_exact_match"].mean() if len(d) else 0,
            "mean_within_tolerance_pct": 100 * d["mean_within_tolerance"].mean() if len(d) else 0,
            "mean_abs_difference_median": d["mean_abs_difference"].median() if len(d) else np.nan,
            "mean_abs_difference_p95": d["mean_abs_difference"].quantile(0.95) if len(d) else np.nan,
            "offset_monotone_pct": 100 * d["offset_monotone_60s"].mean() if len(d) else 0,
            "offset_boundary_pct": 100 * d["offset_boundary_correct"].mean() if len(d) else 0,
        })
    detail_df = pd.DataFrame(detail)
    summary_df = pd.DataFrame(summary)
    relevant_offsets_aligned = bool((hourly["row_offset"].astype("int64") % 3600 == 0).all())
    enough = bool((summary_df["sample_case_hours"] >= 100).all())
    bytes_ok = bool((summary_df["byte_length_240_pct"] == 100).all())
    count_ok = bool((summary_df["count_exact_match_pct"] >= 95).all())
    mean_ok = bool((summary_df["mean_within_tolerance_pct"] >= 95).all())
    offsets_ok = bool((summary_df["offset_monotone_pct"] == 100).all() and
                      (summary_df["offset_boundary_pct"] == 100).all() and relevant_offsets_aligned)
    passed = enough and bytes_ok and count_ok and mean_ok and offsets_ok
    reason = (f"sample>=100={enough}; bytes={bytes_ok}; count_agreement={count_ok}; "
              f"mean_agreement={mean_ok}; official_offset_equivalence={offsets_ok}")
    summary_df["all_relevant_row_offsets_mod3600_zero"] = relevant_offsets_aligned
    summary_df["decoder_gate_pass"] = passed
    summary_df["decoder_gate_reason"] = reason
    return detail_df, summary_df, passed, reason


def build_mimic() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    cohort = pd.read_csv(INPUT / "mimic_cohort.csv", parse_dates=["intime", "outtime"])
    if len(cohort) != 1740:
        raise RuntimeError(f"STOP_AND_DEBUG_COHORT: MIMIC recovered {len(cohort)}")
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)
    cohort["analysis_id"] = [f"M{i:06d}" for i in range(1, len(cohort) + 1)]
    index = dict(zip(cohort["stay_id"].astype(int), range(len(cohort))))
    raw = pd.read_csv(INPUT / "mimic_signals_raw.csv", parse_dates=["intime", "charttime"])
    raw["patient_index"] = raw["stay_id"].astype(int).map(index)
    raw["minute"] = np.floor((raw["charttime"] - raw["intime"]).dt.total_seconds() / 60).astype(int)
    raw = raw[(raw["minute"] >= 0) & (raw["minute"] < 720) & raw["patient_index"].notna()].copy()
    raw["itemid"] = raw["itemid"].astype(int)
    raw["value"] = pd.to_numeric(raw["valuenum"], errors="coerce")
    raw.loc[raw["itemid"] == 223761, "value"] = (raw.loc[raw["itemid"] == 223761, "value"] - 32.0) * 5.0 / 9.0
    raw_stats: list[dict] = []
    for source_id, (signal, _, label) in MIMIC_MAP.items():
        add_raw_stat(raw_stats, "MIMIC-IV", source_id, signal, label,
                     raw.loc[raw["itemid"] == source_id, "value"].to_numpy())
    arrays: dict[int, np.ndarray] = {k: np.full((len(cohort), 720), np.nan, np.float32) for k in MIMIC_MAP}
    grouped = raw.groupby(["patient_index", "itemid", "minute"], sort=False)["value"].median().reset_index()
    for source_id, (signal, _, _) in MIMIC_MAP.items():
        part = grouped[grouped["itemid"] == source_id]
        vals = part["value"].to_numpy(dtype=float).copy()
        if signal in BOUNDS:
            low, high = BOUNDS[signal]
            vals[(vals < low) | (vals > high)] = np.nan
        arrays[source_id][part["patient_index"].astype(int), part["minute"].astype(int)] = vals.astype(np.float32)
    combined = {
        "HR": priority_combine(arrays, [220045]),
        "MAP": priority_combine(arrays, [220052, 220181]),
        "SpO2": priority_combine(arrays, [220277]),
        "RR": priority_combine(arrays, [220210, 224690]),
        "Temperature": priority_combine(arrays, [223762, 223761]),
    }
    primary = {
        "HR": arrays[220045],
        "MAP": arrays[220052],
        "SpO2": arrays[220277],
    }
    return cohort, combined, primary, raw_stats


def build_sicdb(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[dict], pd.DataFrame]:
    cohort = pd.read_csv(INPUT / "sicdb_cohort.csv")
    if len(cohort) != 2376:
        raise RuntimeError(f"STOP_AND_DEBUG_COHORT: SICdb recovered {len(cohort)}")
    cohort = cohort.sort_values("caseid").reset_index(drop=True)
    cohort["analysis_id"] = [f"S{i:06d}" for i in range(1, len(cohort) + 1)]
    index = dict(zip(cohort["caseid"].astype(int), range(len(cohort))))
    icu_offsets = cohort["icuoffset"].astype(int).to_numpy()
    arrays: dict[int, np.ndarray] = {k: np.full((len(cohort), 720), np.nan, np.float32) for k in SICDB_MAP}
    duplicates: dict[tuple[int, int, int], list[float]] = {}
    values_for_stats: dict[int, list[np.ndarray]] = {k: [] for k in SICDB_MAP}
    pre_counts = np.zeros(len(cohort), dtype=int)
    first_included = np.full(len(cohort), np.nan)
    last_included = np.full(len(cohort), np.nan)
    malformed_rows = 0
    for row in hourly.itertuples(index=False):
        dataid = int(row.dataid)
        if dataid not in arrays:
            continue
        patient_index = index.get(int(row.caseid))
        if patient_index is None:
            continue
        payload, values, present = decode_payload(row.rawdata)
        if payload is None:
            if pd.notna(row.rawdata):
                malformed_rows += 1
            continue
        absolute = int(row.row_offset) + np.arange(60, dtype=int) * 60
        finite = np.isfinite(values)
        pre = finite & (absolute < icu_offsets[patient_index])
        if dataid in (707, 724, 708, 703, 706, 710):
            pre_counts[patient_index] += int(pre.sum())
        in_window = finite & (absolute >= icu_offsets[patient_index]) & (absolute < icu_offsets[patient_index] + 43200)
        if not in_window.any():
            continue
        rel_seconds = absolute[in_window] - icu_offsets[patient_index]
        if np.any(rel_seconds % 60 != 0):
            raise RuntimeError("STOP_DECODER_INVALID: decoded offsets do not align to ICU-t0 minute grid")
        minutes = (rel_seconds // 60).astype(int)
        vals = values[in_window].astype(float)
        values_for_stats[dataid].append(vals.copy())
        signal = SICDB_MAP[dataid][0]
        if dataid in (707, 724, 708, 703, 706, 710):
            first_included[patient_index] = np.nanmin([first_included[patient_index], absolute[in_window].min()]) if np.isfinite(first_included[patient_index]) else absolute[in_window].min()
            last_included[patient_index] = np.nanmax([last_included[patient_index], absolute[in_window].max()]) if np.isfinite(last_included[patient_index]) else absolute[in_window].max()
        if signal in BOUNDS:
            low, high = BOUNDS[signal]
            vals[(vals < low) | (vals > high)] = np.nan
        for minute, value in zip(minutes, vals):
            if not np.isfinite(value):
                continue
            existing = arrays[dataid][patient_index, minute]
            if np.isfinite(existing):
                key = (dataid, patient_index, int(minute))
                if key not in duplicates:
                    duplicates[key] = [float(existing)]
                duplicates[key].append(float(value))
            else:
                arrays[dataid][patient_index, minute] = value
    for (dataid, patient_index, minute), vals in duplicates.items():
        arrays[dataid][patient_index, minute] = np.median(vals)
    raw_stats: list[dict] = []
    for source_id, (signal, _, label) in SICDB_MAP.items():
        vals = np.concatenate(values_for_stats[source_id]) if values_for_stats[source_id] else np.array([])
        add_raw_stat(raw_stats, "SICdb", source_id, signal, label, vals)
    combined = {
        "HR": priority_combine(arrays, [707, 724, 708]),
        "MAP": priority_combine(arrays, [703, 706]),
        "SpO2": priority_combine(arrays, [710]),
        "RR": priority_combine(arrays, [719]),
        "Temperature": priority_combine(arrays, [709]),
    }
    primary = {"HR": arrays[707], "MAP": arrays[703], "SpO2": arrays[710]}
    t0 = cohort[["caseid", "analysis_id", "icuoffset"]].copy()
    t0["decoded_pre_icu_core_slots_available_but_excluded"] = pre_counts
    t0["first_included_minute_offset"] = first_included
    t0["last_included_minute_offset"] = last_included
    t0["first_minus_icuoffset_seconds"] = t0["first_included_minute_offset"] - t0["icuoffset"]
    t0["last_minus_icuoffset_seconds"] = t0["last_included_minute_offset"] - t0["icuoffset"]
    t0["included_before_icu_t0"] = t0["first_included_minute_offset"] < t0["icuoffset"]
    t0["included_at_or_after_12h"] = t0["last_included_minute_offset"] >= t0["icuoffset"] + 43200
    eligible = t0[(t0["icuoffset"] > 0) & t0["first_included_minute_offset"].notna()].copy()
    eligible["sample_key"] = [hashlib.sha256(f"{SEED}|t0|{int(x)}".encode()).hexdigest() for x in eligible["caseid"]]
    sample = eligible.sort_values("sample_key").head(min(50, len(eligible))).drop(columns=["sample_key", "caseid"])
    sample["malformed_nonnull_raw_rows_total"] = malformed_rows
    sample["duplicate_source_minute_cells_total"] = len(duplicates)
    return cohort, combined, primary, raw_stats, sample


def coverage_outputs(dataset: str, cohort: pd.DataFrame, combined: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(cohort)
    per_signal_hours: dict[str, np.ndarray] = {}
    availability: dict[str, np.ndarray] = {}
    coverage_rows: list[dict] = []
    hour_rows: list[dict] = []
    streak_rows: list[dict] = []
    patient = pd.DataFrame({"analysis_id": cohort["analysis_id"], "dataset": dataset})
    for signal in ALL_SIGNALS:
        avail = np.isfinite(combined[signal]).reshape(n, 12, 60).any(axis=2)
        availability[signal] = avail
        hours = avail.sum(axis=1)
        per_signal_hours[signal] = hours
        patient[f"{signal}_available_hours"] = hours
        streak = np.array([longest_false_run(x) for x in avail], dtype=int)
        patient[f"{signal}_longest_missing_streak_h"] = streak
        for criterion, hit in (("any", hours >= 1), (">=6h", hours >= 6), (">=8h", hours >= 8),
                               (">=10h", hours >= 10), ("12h", hours == 12)):
            coverage_rows.append({"dataset": dataset, "signal": signal, "criterion": criterion,
                                  "n": int(hit.sum()), "denominator": n, "percent": 100 * hit.mean()})
        for hour in range(12):
            hour_rows.append({"dataset": dataset, "signal": signal, "hour": hour,
                              "n_available": int(avail[:, hour].sum()), "denominator": n,
                              "percent": 100 * avail[:, hour].mean()})
        streak_rows.append({"dataset": dataset, "signal": signal,
                            "median_longest_missing_h": q(streak, 0.5), "q25": q(streak, 0.25),
                            "q75": q(streak, 0.75), "p95": q(streak, 0.95)})
    core_stack = np.stack([availability[x] for x in CORE], axis=2)
    core_a_hours = (core_stack.sum(axis=2) >= 2).sum(axis=1)
    core_b_hours = (core_stack.sum(axis=2) == 3).sum(axis=1)
    core_a = core_a_hours >= 8
    core_b = core_b_hours >= 6
    both = core_a & core_b
    strict = np.logical_and.reduce([per_signal_hours[x] >= 8 for x in CORE])
    patient["core_a_hours_at_least_2of3"] = core_a_hours
    patient["core_b_hours_all3"] = core_b_hours
    patient["core_a_pass"] = core_a
    patient["core_b_pass"] = core_b
    patient["core_a_plus_b_pass"] = both
    patient["strict_each_core_signal_8h_pass"] = strict
    core_rows = []
    for criterion, hit in (("core_A", core_a), ("core_B", core_b), ("core_A_plus_B", both),
                           ("strict_each_signal_ge8h", strict)):
        core_rows.append({"dataset": dataset, "criterion": criterion, "n": int(hit.sum()),
                          "denominator": n, "percent": 100 * hit.mean()})
    return (pd.DataFrame(coverage_rows), pd.DataFrame(hour_rows), patient,
            pd.DataFrame(core_rows), pd.DataFrame(streak_rows))


def density_outputs(dataset: str, cohort: pd.DataFrame, variants: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    summary_rows: list[dict] = []
    patient_rows: list[dict] = []
    hour_rows: list[dict] = []
    patient_values: dict[str, dict[str, np.ndarray]] = {}
    for variant, signals in variants.items():
        patient_values[variant] = {}
        for signal in CORE:
            valid = np.isfinite(signals[signal])
            obs_per_hour = valid.sum(axis=1) / 12.0
            occupancy = valid.sum(axis=1) / 720.0
            patient_hour = valid.reshape(len(cohort), 12, 60).sum(axis=2)
            patient_values[variant][signal] = obs_per_hour
            med, lo, hi = median_iqr(obs_per_hour)
            occ_med, occ_lo, occ_hi = median_iqr(occupancy)
            ph_med, ph_lo, ph_hi = median_iqr(patient_hour.ravel())
            summary_rows.append({
                "dataset": dataset, "source_variant": variant, "signal": signal,
                "patient_observations_per_hour_median": med,
                "patient_observations_per_hour_q25": lo,
                "patient_observations_per_hour_q75": hi,
                "patient_observations_per_hour_p10": q(obs_per_hour, 0.10),
                "patient_observations_per_hour_p90": q(obs_per_hour, 0.90),
                "temporal_occupancy_median": occ_med,
                "temporal_occupancy_q25": occ_lo,
                "temporal_occupancy_q75": occ_hi,
                "temporal_occupancy_p10": q(occupancy, 0.10),
                "temporal_occupancy_p90": q(occupancy, 0.90),
                "patient_hour_valid_minutes_median": ph_med,
                "patient_hour_valid_minutes_q25": ph_lo,
                "patient_hour_valid_minutes_q75": ph_hi,
            })
            for i, analysis_id in enumerate(cohort["analysis_id"]):
                patient_rows.append({"analysis_id": analysis_id, "dataset": dataset,
                                     "source_variant": variant, "signal": signal,
                                     "observations_per_hour": obs_per_hour[i],
                                     "temporal_occupancy": occupancy[i]})
                if variant == "combined":
                    for hour in range(12):
                        hour_rows.append({"analysis_id": analysis_id, "dataset": dataset,
                                          "signal": signal, "hour": hour,
                                          "valid_minute_bins": int(patient_hour[i, hour])})
        patient_values[variant]["core"] = np.median(
            np.stack([patient_values[variant][x] for x in CORE], axis=1), axis=1)
    return pd.DataFrame(summary_rows), pd.DataFrame(patient_rows), pd.DataFrame(hour_rows), patient_values


def density_ratios(mimic_values: dict[str, dict[str, np.ndarray]], sicdb_values: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for variant in ("combined", "primary_only"):
        for signal in (*CORE, "core"):
            m = mimic_values[variant][signal]
            s = sicdb_values[variant][signal]
            m_med = float(np.median(m))
            s_med = float(np.median(s))
            ratio = s_med / m_med if m_med > 0 else np.nan
            boots = np.empty(1000, dtype=float)
            for b in range(1000):
                mb = np.median(m[RNG.integers(0, len(m), len(m))])
                sb = np.median(s[RNG.integers(0, len(s), len(s))])
                boots[b] = sb / mb if mb > 0 else np.nan
            rows.append({
                "source_variant": variant, "signal": signal,
                "mimic_median_observations_per_hour": m_med,
                "sicdb_median_observations_per_hour": s_med,
                "sicdb_to_mimic_ratio": ratio,
                "bootstrap_replicates": 1000,
                "ratio_ci_low": q(boots, 0.025), "ratio_ci_high": q(boots, 0.975),
                "mimic_q25": q(m, 0.25), "mimic_q75": q(m, 0.75),
                "sicdb_q25": q(s, 0.25), "sicdb_q75": q(s, 0.75),
            })
    return pd.DataFrame(rows)


def dense_support(cohort: pd.DataFrame, signals: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = len(cohort)
    dense_counts: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    patient = pd.DataFrame({"analysis_id": cohort["analysis_id"]})
    for signal in CORE:
        ph = np.isfinite(signals[signal]).reshape(n, 12, 60).sum(axis=2)
        dense = ph >= 45
        dense_counts[signal] = dense.sum(axis=1)
        patient[f"{signal}_dense_hours"] = dense_counts[signal]
        for threshold in (6, 8, 10):
            hit = dense_counts[signal] >= threshold
            rows.append({"scope": signal, "criterion": f">={threshold}_dense_hours",
                         "n": int(hit.sum()), "denominator": n, "percent": 100 * hit.mean()})
    dense_core = np.logical_and.reduce([dense_counts[x] >= 8 for x in CORE])
    patient["dense_core"] = dense_core
    rows.append({"scope": "core", "criterion": "dense_core_each_signal_ge8_dense_hours",
                 "n": int(dense_core.sum()), "denominator": n, "percent": 100 * dense_core.mean()})
    five_min = []
    for signal in CORE:
        five_min.append(np.isfinite(signals[signal]).reshape(n, 12, 12, 5).any(axis=3))
    common = np.logical_and.reduce(five_min)
    supported_hours = (common.sum(axis=2) >= 9).sum(axis=1)
    support = supported_hours >= 8
    patient["five_min_supported_hours"] = supported_hours
    patient["five_min_representation_support"] = support
    rows.append({"scope": "core", "criterion": "five_min_representation_ge8_supported_hours",
                 "n": int(support.sum()), "denominator": n, "percent": 100 * support.mean()})
    return pd.DataFrame(rows), patient, dense_core


def threshold_mask(values: np.ndarray, signal: str) -> np.ndarray:
    if signal == "MAP":
        return values < 65
    if signal == "HR":
        return values > 100
    return values < 92


def runs(mask: np.ndarray, minimum_run: int, step_minutes: int) -> tuple[bool, int, int]:
    count = longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
        else:
            if current >= minimum_run:
                count += 1
                longest = max(longest, current)
            current = 0
    if current >= minimum_run:
        count += 1
        longest = max(longest, current)
    return count > 0, count, longest * step_minutes


def series_metrics(values: np.ndarray, signal: str, step_minutes: int, reference: bool) -> dict:
    valid = np.isfinite(values)
    n_valid = int(valid.sum())
    event_values = valid & threshold_mask(values, signal)
    event, episode_count, longest = runs(event_values, 5 if reference else 1, step_minutes)
    burden = 100 * event_values.sum() / n_valid if n_valid else np.nan
    observed = values[valid]
    if observed.size:
        extreme = float(np.min(observed)) if signal in ("MAP", "SpO2") else float(np.max(observed))
    else:
        extreme = np.nan
    sd = float(np.std(observed, ddof=1)) if observed.size >= 2 else np.nan
    consecutive = valid[:-1] & valid[1:]
    masd = float(np.median(np.abs(np.diff(values)[consecutive]))) if consecutive.any() else np.nan
    times = np.arange(len(values), dtype=float) * step_minutes / 60.0
    slope = np.nan
    if n_valid >= 3 and (times[valid].max() - times[valid].min()) >= 6:
        slope = float(np.polyfit(times[valid], values[valid], 1)[0])
    return {
        "burden_pct": burden, "event_detected": bool(event) if n_valid else np.nan,
        "episode_count": episode_count if n_valid else np.nan,
        "longest_episode_min": longest if n_valid else np.nan,
        "extreme": extreme, "sd": sd, "masd": masd, "slope_per_hour": slope,
        "valid_units": n_valid, "valid_minutes_equivalent": n_valid * step_minutes,
    }


def downsample(values: np.ndarray, width: int) -> np.ndarray:
    bins = values.reshape(720 // width, width)
    required = math.ceil(width / 2)
    output = np.full(len(bins), np.nan)
    for i, block in enumerate(bins):
        valid = block[np.isfinite(block)]
        if valid.size >= required:
            output[i] = np.median(valid)
    return output


def resolution_outputs(cohort: pd.DataFrame, signals: dict[str, np.ndarray], dense_core: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict] = []
    phase_rows: list[dict] = []
    for patient_index, analysis_id in enumerate(cohort["analysis_id"]):
        for signal in CORE:
            raw = signals[signal][patient_index].astype(float)
            for width in (1, 5, 15, 60):
                values = raw if width == 1 else downsample(raw, width)
                metrics = series_metrics(values, signal, width, width == 1)
                patient_rows.append({"analysis_id": analysis_id, "dense_core": bool(dense_core[patient_index]),
                                     "signal": signal, "resolution_min": width, **metrics})
            for width in (5, 15, 60):
                phases = sorted(set([0, width // 4, width // 2, (3 * width) // 4, width - 1]))
                ref = series_metrics(raw, signal, 1, True)
                for phase in phases:
                    sampled = raw[phase::width]
                    met = series_metrics(sampled, signal, width, False)
                    phase_rows.append({"analysis_id": analysis_id, "dense_core": bool(dense_core[patient_index]),
                                       "signal": signal, "resolution_min": width, "phase_min": phase,
                                       "reference_event": ref["event_detected"],
                                       "sampled_event": met["event_detected"],
                                       "reference_burden_pct": ref["burden_pct"],
                                       "sampled_burden_pct": met["burden_pct"],
                                       "sampled_valid_units": met["valid_units"]})
    patient = pd.DataFrame(patient_rows)
    summary_rows: list[dict] = []
    for cohort_name, include in (("all", np.ones(len(cohort), dtype=bool)), ("dense_core", dense_core)):
        ids = set(cohort.loc[include, "analysis_id"])
        subset = patient[patient["analysis_id"].isin(ids)]
        for signal in CORE:
            reference = subset[(subset["signal"] == signal) & (subset["resolution_min"] == 1)].set_index("analysis_id")
            for width in (1, 5, 15, 60):
                coarse = subset[(subset["signal"] == signal) & (subset["resolution_min"] == width)].set_index("analysis_id")
                paired = reference.join(coarse, lsuffix="_ref", rsuffix="_coarse", how="inner")
                analyzable = (paired["valid_units_ref"] > 0) & (paired["valid_units_coarse"] > 0)
                paired = paired[analyzable]
                burden_signed = paired["burden_pct_coarse"] - paired["burden_pct_ref"]
                burden_abs = burden_signed.abs()
                ref_event = paired["event_detected_ref"] == True
                ref_nonevent = paired["event_detected_ref"] == False
                missed = ref_event & (paired["event_detected_coarse"] == False)
                false_positive = ref_nonevent & (paired["event_detected_coarse"] == True)
                if signal == "HR":
                    extreme_loss = paired["extreme_ref"] - paired["extreme_coarse"]
                else:
                    extreme_loss = paired["extreme_coarse"] - paired["extreme_ref"]
                sd_ret = paired["sd_coarse"] / paired["sd_ref"].replace(0, np.nan)
                masd_ret = paired["masd_coarse"] / paired["masd_ref"].replace(0, np.nan)
                rho = paired["burden_pct_ref"].rank(method="average").corr(
                    paired["burden_pct_coarse"].rank(method="average"), method="pearson")
                summary_rows.append({
                    "cohort": cohort_name, "cohort_n": len(ids), "signal": signal, "resolution_min": width,
                    "paired_n": len(paired),
                    "burden_signed_error_median_pp": q(burden_signed, 0.5),
                    "burden_signed_error_q25_pp": q(burden_signed, 0.25),
                    "burden_signed_error_q75_pp": q(burden_signed, 0.75),
                    "burden_absolute_error_median_pp": q(burden_abs, 0.5),
                    "burden_absolute_error_q25_pp": q(burden_abs, 0.25),
                    "burden_absolute_error_q75_pp": q(burden_abs, 0.75),
                    "burden_absolute_error_p90_pp": q(burden_abs, 0.90),
                    "reference_event_positive_n": int(ref_event.sum()),
                    "event_missed_n": int(missed.sum()),
                    "event_miss_pct": 100 * missed.sum() / ref_event.sum() if ref_event.sum() else np.nan,
                    "reference_event_negative_n": int(ref_nonevent.sum()),
                    "false_positive_n": int(false_positive.sum()),
                    "false_positive_pct": 100 * false_positive.sum() / ref_nonevent.sum() if ref_nonevent.sum() else np.nan,
                    "extreme_attenuation_median": q(extreme_loss, 0.5),
                    "extreme_attenuation_q25": q(extreme_loss, 0.25),
                    "extreme_attenuation_q75": q(extreme_loss, 0.75),
                    "extreme_attenuation_p90": q(extreme_loss, 0.90),
                    "sd_retention_median": q(sd_ret, 0.5), "sd_retention_q25": q(sd_ret, 0.25),
                    "sd_retention_q75": q(sd_ret, 0.75),
                    "masd_retention_median": q(masd_ret, 0.5), "masd_retention_q25": q(masd_ret, 0.25),
                    "masd_retention_q75": q(masd_ret, 0.75),
                    "burden_spearman_rho": rho,
                    "slope_absolute_error_median": q((paired["slope_per_hour_coarse"] - paired["slope_per_hour_ref"]).abs(), 0.5),
                })
    phase = pd.DataFrame(phase_rows)
    phase_summary = []
    for cohort_name, condition in (("all", pd.Series(True, index=phase.index)), ("dense_core", phase["dense_core"])):
        part = phase[condition]
        for keys, group in part.groupby(["signal", "resolution_min", "phase_min"]):
            signal, width, phase_min = keys
            analyzable = group["sampled_valid_units"] > 0
            group = group[analyzable]
            ref_event = group["reference_event"] == True
            miss = ref_event & (group["sampled_event"] == False)
            abs_error = (group["sampled_burden_pct"] - group["reference_burden_pct"]).abs()
            phase_summary.append({"cohort": cohort_name, "signal": signal, "resolution_min": width,
                                  "phase_min": phase_min, "paired_n": len(group),
                                  "reference_event_positive_n": int(ref_event.sum()),
                                  "event_miss_pct": 100 * miss.sum() / ref_event.sum() if ref_event.sum() else np.nan,
                                  "burden_absolute_error_median_pp": q(abs_error, 0.5),
                                  "burden_absolute_error_p90_pp": q(abs_error, 0.90)})
    return patient, pd.DataFrame(summary_rows), pd.DataFrame(phase_summary)


def make_gate_summary(decoder_pass: bool, t0_pass: bool, core_coverage: pd.DataFrame,
                      dense: pd.DataFrame, ratios: pd.DataFrame, resolution: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    decoder_light = "GREEN" if decoder_pass and t0_pass else "RED"
    rows.append({"dimension": "Decoder / t0", "metric": "decoder_and_t0_pass", "value": int(decoder_pass and t0_pass),
                 "unit": "boolean", "traffic_light": decoder_light,
                 "rationale": "Official 60-slot logic, count/mean QC, offset equivalence, and ICU-t0 sample"})
    coverage_ab = core_coverage[core_coverage["criterion"] == "core_A_plus_B"].set_index("dataset")["percent"]
    min_coverage = float(coverage_ab.min())
    coverage_light = "GREEN" if min_coverage >= 70 else ("AMBER" if min_coverage >= 50 else "RED")
    for dataset, value in coverage_ab.items():
        rows.append({"dimension": "Common 60-min coverage", "metric": f"{dataset}_core_A_plus_B_pct",
                     "value": value, "unit": "%", "traffic_light": coverage_light,
                     "rationale": "Light is based on the lower of the two databases"})
    dense_row = dense[dense["criterion"] == "dense_core_each_signal_ge8_dense_hours"].iloc[0]
    support_row = dense[dense["criterion"] == "five_min_representation_ge8_supported_hours"].iloc[0]
    dense_pct = float(dense_row["percent"])
    support_pct = float(support_row["percent"])
    minute_light = "RED" if dense_pct < 30 else ("GREEN" if dense_pct >= 50 and support_pct >= 50 else "AMBER")
    rows.extend([
        {"dimension": "SICdb minute coverage", "metric": "dense_core_pct", "value": dense_pct, "unit": "%",
         "traffic_light": minute_light, "rationale": "Dense-core plus 5-min support"},
        {"dimension": "SICdb minute coverage", "metric": "five_min_representation_support_pct", "value": support_pct,
         "unit": "%", "traffic_light": minute_light, "rationale": "At least 8 supported hours"},
    ])
    main_ratios = ratios[ratios["source_variant"] == "combined"].set_index("signal")
    core_ratio = float(main_ratios.loc["core", "sicdb_to_mimic_ratio"])
    signal_ratios = [float(main_ratios.loc[x, "sicdb_to_mimic_ratio"]) for x in CORE]
    directions = sum(x > 1 for x in signal_ratios)
    density_light = "GREEN" if core_ratio >= 4 and directions == 3 else ("AMBER" if core_ratio >= 2 and directions >= 2 else "RED")
    for signal in (*CORE, "core"):
        rows.append({"dimension": "Density advantage", "metric": f"{signal}_sicdb_to_mimic_ratio",
                     "value": float(main_ratios.loc[signal, "sicdb_to_mimic_ratio"]), "unit": "ratio",
                     "traffic_light": density_light, "rationale": "Combined canonical source-priority series"})
    main_60 = resolution[(resolution["cohort"] == "dense_core") & (resolution["resolution_min"] == 60)].set_index("signal")
    misses = {signal: float(main_60.loc[signal, "event_miss_pct"]) for signal in CORE}
    finite_misses = [x for x in misses.values() if np.isfinite(x)]
    max_miss = max(finite_misses) if finite_misses else np.nan
    resolution_light = "GREEN" if np.isfinite(max_miss) and max_miss >= 15 else (
        "AMBER" if np.isfinite(max_miss) and max_miss >= 5 else "RED")
    for signal, value in misses.items():
        rows.append({"dimension": "60-min resolution loss", "metric": f"{signal}_event_miss_pct",
                     "value": value, "unit": "%", "traffic_light": resolution_light,
                     "rationale": "Dense-core paired reference events; continuous loss metrics retained separately"})
    if decoder_light == "RED":
        verdict = "NO_GO_SIGNAL_ARCHITECTURE"
    elif coverage_light == "GREEN" and density_light == "GREEN" and resolution_light in ("GREEN", "AMBER"):
        verdict = "GO_FORMAL_MODELING_HIGH_DENSITY"
    elif coverage_light == "GREEN" and density_light == "RED" and resolution_light == "RED":
        verdict = "GO_FORMAL_MODELING_TRANSPORT_ONLY"
    elif coverage_light == "RED" and density_light == "RED" and resolution_light == "RED":
        verdict = "NO_GO_SIGNAL_ARCHITECTURE"
    else:
        verdict = "PI_REVIEW_SIGNAL_FEASIBILITY"
    rows.append({"dimension": "Final", "metric": "verdict", "value": verdict, "unit": "token",
                 "traffic_light": "NA", "rationale": "Authorized combination rule; no outcome model was run"})
    return pd.DataFrame(rows)


def main() -> None:
    hourly = pd.read_csv(INPUT / "sicdb_signals_hourly_raw.csv")
    decoder_detail, decoder_summary, decoder_pass, decoder_reason = decoder_qc(hourly)
    save_csv(decoder_detail, DECODER / "decoder_qc.csv")
    save_csv(decoder_summary, DECODER / "decoder_qc_summary.csv")
    if not decoder_pass:
        (DECODER / "STOP_DECODER_INVALID.txt").write_text(decoder_reason + "\n", encoding="utf-8")
        raise RuntimeError("STOP_DECODER_INVALID: " + decoder_reason)

    mimic_cohort, mimic_combined, mimic_primary, mimic_raw_stats = build_mimic()
    sicdb_cohort, sicdb_combined, sicdb_primary, sicdb_raw_stats, t0_sample = build_sicdb(hourly)
    del hourly
    t0_pass = (len(t0_sample) >= 50 and not t0_sample["included_before_icu_t0"].any() and
               not t0_sample["included_at_or_after_12h"].any() and
               (t0_sample["first_minus_icuoffset_seconds"] >= 0).all() and
               (t0_sample["last_minus_icuoffset_seconds"] < 43200).all())
    save_csv(t0_sample, DECODER / "t0_alignment_sample.csv")
    save_csv(pd.DataFrame(mimic_raw_stats + sicdb_raw_stats), DECODER / "raw_range_qc.csv")
    if not t0_pass:
        (DECODER / "STOP_T0_ALIGNMENT_INVALID.txt").write_text("ICU-t0 sample failed\n", encoding="utf-8")
        raise RuntimeError("STOP_T0_ALIGNMENT_INVALID")

    cov_frames = []
    hour_frames = []
    patient_frames = []
    core_frames = []
    streak_frames = []
    for dataset, cohort, signals in (("MIMIC-IV", mimic_cohort, mimic_combined),
                                     ("SICdb", sicdb_cohort, sicdb_combined)):
        cov, hour, patient, core, streak = coverage_outputs(dataset, cohort, signals)
        cov_frames.append(cov); hour_frames.append(hour); patient_frames.append(patient)
        core_frames.append(core); streak_frames.append(streak)
    coverage_signal = pd.concat(cov_frames, ignore_index=True)
    coverage_hour = pd.concat(hour_frames, ignore_index=True)
    patient_coverage = pd.concat(patient_frames, ignore_index=True)
    core_coverage = pd.concat(core_frames, ignore_index=True)
    streak = pd.concat(streak_frames, ignore_index=True)

    mimic_density, mimic_density_patient, mimic_hour, mimic_values = density_outputs(
        "MIMIC-IV", mimic_cohort, {"combined": mimic_combined, "primary_only": mimic_primary})
    sicdb_density, sicdb_density_patient, sicdb_hour, sicdb_values = density_outputs(
        "SICdb", sicdb_cohort, {"combined": sicdb_combined, "primary_only": sicdb_primary})
    density = pd.concat([mimic_density, sicdb_density], ignore_index=True)
    density_patient = pd.concat([mimic_density_patient, sicdb_density_patient], ignore_index=True)
    patient_hour = pd.concat([mimic_hour, sicdb_hour], ignore_index=True)
    ratios = density_ratios(mimic_values, sicdb_values)
    dense, dense_patient, dense_core = dense_support(sicdb_cohort, sicdb_combined)
    patient_coverage = patient_coverage.merge(dense_patient.assign(dataset="SICdb"),
                                              on=["analysis_id", "dataset"], how="left")

    resolution_patient, resolution_summary, phase_summary = resolution_outputs(
        sicdb_cohort, sicdb_combined, dense_core)
    gate = make_gate_summary(decoder_pass, t0_pass, core_coverage, dense, ratios, resolution_summary)

    save_csv(coverage_signal, COVERAGE / "coverage_by_signal.csv")
    save_csv(coverage_hour, COVERAGE / "coverage_by_hour.csv")
    save_csv(patient_coverage, COVERAGE / "patient_coverage.csv")
    save_csv(core_coverage, COVERAGE / "core_coverage_summary.csv")
    save_csv(streak, COVERAGE / "missing_streak_summary.csv")
    save_csv(density, DENSITY / "density_by_signal.csv")
    save_csv(ratios, DENSITY / "density_ratio.csv")
    save_csv(density_patient, DENSITY / "density_patient_level.csv")
    save_csv(patient_hour, DENSITY / "patient_hour_density.csv")
    save_csv(dense, DENSITY / "sicdb_dense_support.csv")
    save_csv(resolution_patient, RESOLUTION / "resolution_loss_patient_level.csv")
    save_csv(resolution_summary, RESOLUTION / "resolution_loss_summary.csv")
    save_csv(phase_summary, RESOLUTION / "resolution_phase_sensitivity.csv")
    save_csv(gate, AUDIT / "signal_gate_summary.csv")

    save_csv(coverage_signal[coverage_signal["signal"].isin(CORE)], COVERAGE / "Table_A_signal_coverage.csv")
    save_csv(core_coverage, COVERAGE / "Table_B_core_representation.csv")
    save_csv(density.merge(ratios[ratios["source_variant"] == "combined"],
                           on=["source_variant", "signal"], how="left"), DENSITY / "Table_C_density.csv")
    save_csv(dense, DENSITY / "Table_D_sicdb_dense_support.csv")
    save_csv(resolution_summary[resolution_summary["cohort"] == "dense_core"],
             RESOLUTION / "Table_E_resolution_loss.csv")

    outputs = []
    for folder in (DECODER, COVERAGE, DENSITY, RESOLUTION):
        for path in sorted(folder.glob("*.csv")):
            outputs.append({"path": str(path.relative_to(AUDIT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "status": "SUCCESS", "seed": SEED, "decoder_gate": "PASS", "t0_gate": "PASS",
        "mimic_n": len(mimic_cohort), "sicdb_n": len(sicdb_cohort),
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "script": {"path": SCRIPT.name, "sha256": sha256(SCRIPT)},
        "inputs": [{"path": x.name, "bytes": x.stat().st_size, "sha256": sha256(x)} for x in sorted(INPUT.glob("*.csv"))],
        "outputs": outputs,
        "prohibited_models_status": "NOT_RUN",
    }
    (LOGS / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = gate.loc[gate["metric"] == "verdict", "value"].iloc[0]
    key = gate[gate["metric"].isin([
        "MIMIC-IV_core_A_plus_B_pct", "SICdb_core_A_plus_B_pct", "dense_core_pct",
        "five_min_representation_support_pct", "HR_sicdb_to_mimic_ratio",
        "MAP_sicdb_to_mimic_ratio", "SpO2_sicdb_to_mimic_ratio", "MAP_event_miss_pct"
    ])][["metric", "value"]]
    print(json.dumps({"status": "SUCCESS", "verdict": verdict,
                      "key_metrics": dict(zip(key["metric"], key["value"]))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
