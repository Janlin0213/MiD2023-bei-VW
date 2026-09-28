"""Parallel Phase 3 decision-timing sensitivity; never rewrites baseline files.

Physical HOME/AWAY states remain those in the accepted classified master.
Model AV_1/AV_2 instead represent the conceptual two-car household fleet under
unobserved prior allocation. No Phase 1/2 reconstruction is performed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (
    CHOICE_RECONSTRUCTION_SCRIPTS_DIR,
    PHASE1_DIR,
    PHASE3_DIR,
    PHASE3_SENSITIVITY_DIR,
)

CLASSIFIED_PATH = PHASE3_DIR / "vehicle_choice_occasions_classified.csv"
BASELINE_WIDE_PATH = PHASE3_DIR / "mnl_vehicle_choice_wide_base.csv"
TRIPS_PATH = PHASE1_DIR / "trips_home_chain_enriched.csv"
OUTPUT_DIR = PHASE3_SENSITIVITY_DIR
RICH_PATH = OUTPUT_DIR / "vehicle_choice_occasions_decision_timing_sensitivity.csv"
WIDE_PATH = OUTPUT_DIR / "mnl_vehicle_choice_wide_decision_timing_sensitivity_base.csv"
QA_PATH = OUTPUT_DIR / "decision_timing_sensitivity_QA.csv"
WIDE_COLUMNS = [
    "CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID",
    "CHOICE", "AV_1", "AV_2", "START_MIN", "ARRIVAL_MIN", "W_ZWECK", "W_SO1", "W_GEW",
]
NON_AVAILABILITY_FLAGS = [
    "EXCL_INCOMPLETE_FLEET", "EXCL_UNMATCHED_VEHICLE", "EXCL_DUPLICATE_IDENTIFIER",
    "EXCL_INCOMPLETE_CHOICE_TRIP_TIME", "EXCL_INVALID_SNAPSHOT", "EXCL_UNKNOWN_SNAPSHOT",
    "EXCL_AVAILABILITY_CONFLICT", "EXCL_SIMULTANEOUS_DEPARTURE",
    "EXCL_HOUSEHOLD_TIME_UNRESOLVED", "EXCL_HOUSEHOLD_TIMELINE_CONFLICT",
    "EXCL_VEHICLE_OVERLAP", "EXCL_LOCATION_TRANSITION_CONFLICT", "EXCL_ZERO_DURATION_HISTORY",
]
INSUFFICIENT = "EXCL_INSUFFICIENT_AVAILABLE_CARS"
STRICT = "VALID_STRICT_CHOICE_OCCASION"
VALID = "VALID_DECISION_TIMING_SENSITIVITY_OCCASION"
PATTERNS = {"1|2": "HOME_HOME", "1": "HOME_AWAY", "2": "AWAY_HOME"}


def read_strings(path: Path, usecols=None) -> pd.DataFrame:
    # Do not coerce or rewrite identifiers (including leading zeros).
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.strip(), errors="coerce")


def require_columns(frame: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def reject(frame: pd.DataFrame, bad: pd.Series, message: str) -> None:
    bad = bad.fillna(True)
    if bad.any():
        print(f"\n{message}\n{frame.loc[bad].head(10).to_string(index=False)}")
        raise ValueError(message)


def validate_ids(frame: pd.DataFrame) -> None:
    for col in ["CHOICE_ID", "SOURCE_ROW_ID"]:
        reject(frame, frame[col].str.strip().eq("") | frame[col].duplicated(keep=False),
               f"{col} must be non-missing and unique; identifiers must map one-to-one.")


def select_sensitivity(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(master, [*WIDE_COLUMNS[:6], "START_MIN", "ARRIVAL_MIN", "HOME_ORIGIN",
        "CHOSEN_A_ID", "AVAILABLE_CAR_SET", "N_AVAILABLE_CARS", "W_ZWECK", "W_SO1",
        "H_ANZAUTO", "pkw_fmf", "complete_fleet_baseline_flag", "snapshot_valid_flag",
        *NON_AVAILABILITY_FLAGS, INSUFFICIENT, STRICT])
    flags = master[[*NON_AVAILABILITY_FLAGS, INSUFFICIENT, STRICT]].apply(numeric)
    reject(master, ~flags.isin([0, 1]).all(axis=1), "Classification flags must be binary and populated.")
    baseline = master.loc[flags[STRICT].eq(1)].copy()
    validate_ids(baseline)
    eligible = flags[NON_AVAILABILITY_FLAGS].eq(0).all(axis=1)
    n = numeric(master["N_AVAILABLE_CARS"])
    selected = master.loc[eligible & n.isin([1, 2])].copy()
    # Fail on inconsistent classified rows, rather than silently repairing them.
    reject(selected, ~selected["AVAILABLE_CAR_SET"].isin(PATTERNS), "Unexpected physical state pattern.")
    n = numeric(selected["N_AVAILABLE_CARS"])
    chosen = numeric(selected["CHOSEN_A_ID"])
    physical = selected["AVAILABLE_CAR_SET"]
    chosen_home = (chosen.eq(1) & physical.isin(["1", "1|2"])) | (chosen.eq(2) & physical.isin(["2", "1|2"]))
    reject(selected, ~chosen_home, "Chosen car must be physically HOME and in {1,2}.")
    reject(selected, n.ne(physical.map({"1": 1, "2": 1, "1|2": 2})), "Physical set/count mismatch.")
    for col, expected in [("HOME_ORIGIN", 1), ("H_ANZAUTO", 2), ("pkw_fmf", 1),
                          ("complete_fleet_baseline_flag", 1), ("snapshot_valid_flag", 1)]:
        reject(selected, numeric(selected[col]).ne(expected), f"Invalid structural requirement: {col}.")
    start, arrival = numeric(selected["START_MIN"]), numeric(selected["ARRIVAL_MIN"])
    finite = lambda s: s.map(lambda x: pd.notna(x) and math.isfinite(x))
    reject(selected, ~finite(start) | ~finite(arrival) | arrival.le(start), "Invalid current choice-trip timing.")
    for col in WIDE_COLUMNS[:6]:
        reject(selected, selected[col].str.strip().eq(""), f"Missing identifier: {col}.")
    validate_ids(selected)
    selected["PHYS_AV_1"] = physical.isin(["1", "1|2"]).astype(int)
    selected["PHYS_AV_2"] = physical.isin(["2", "1|2"]).astype(int)
    selected["PREDEPARTURE_STATE_PATTERN"] = physical.map(PATTERNS)
    selected["SENSITIVITY_EXTENSION_FLAG"] = n.eq(1).astype(int)
    selected[VALID] = 1
    reject(baseline, ~baseline["CHOICE_ID"].isin(selected["CHOICE_ID"]), "Baseline choices missing from sensitivity.")
    is_baseline = selected["CHOICE_ID"].isin(baseline["CHOICE_ID"])
    reject(selected, is_baseline.ne(n.eq(2)), "Baseline must equal HOME_HOME; all extensions must have one HOME car.")
    reject(selected, numeric(selected[INSUFFICIENT]).ne(selected["SENSITIVITY_EXTENSION_FLAG"]),
           "Existing insufficient-availability flag disagrees with extension status.")
    reject(selected, ~selected[NON_AVAILABILITY_FLAGS].apply(numeric).eq(0).all(axis=1),
           "Non-availability strict exclusion found in sensitivity sample.")
    return selected, baseline


def merge_weights(selected: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    require_columns(trips, ["SOURCE_ROW_ID", "W_GEW"])
    reject(trips, trips["SOURCE_ROW_ID"].str.strip().eq("") | trips["SOURCE_ROW_ID"].duplicated(keep=False),
           "Trip weight source identifiers must be non-missing and unique.")
    weights = numeric(trips["W_GEW"])
    reject(trips, trips["W_GEW"].str.strip().ne("") & weights.isna(), "Non-numeric source W_GEW.")
    source = trips[["SOURCE_ROW_ID"]].copy()
    source["W_GEW"] = weights
    # Same left, one-to-one choice-level merge as accepted Phase 3.
    merged = selected.drop(columns=["W_GEW"], errors="ignore").merge(
        source, on="SOURCE_ROW_ID", how="left", validate="one_to_one", indicator=True)
    reject(merged, merged["_merge"].ne("both"), "Unmatched choice weight.")
    if not merged[WIDE_COLUMNS[:2]].reset_index(drop=True).equals(selected[WIDE_COLUMNS[:2]].reset_index(drop=True)):
        print(merged[WIDE_COLUMNS[:2]].head(10).to_string(index=False))
        raise ValueError("Weight merge changed identifiers or row order.")
    w = merged["W_GEW"]
    reject(merged, w.isna() | ~w.map(lambda x: pd.notna(x) and math.isfinite(x)) | w.le(0),
           "Selected W_GEW must be numeric, finite, non-missing and strictly positive.")
    return merged.drop(columns="_merge")


def build_outputs(master: pd.DataFrame, trips: pd.DataFrame):
    selected, baseline = select_sensitivity(master)
    rich = merge_weights(selected, trips)
    wide = rich.rename(columns={"CHOSEN_A_ID": "CHOICE"}).copy()
    wide["CHOICE"] = numeric(wide["CHOICE"]).astype(int)
    # Intentional conceptual fleet availability, NOT reconstructed physical availability.
    wide["AV_1"] = 1
    wide["AV_2"] = 1
    wide = wide[WIDE_COLUMNS]
    validate_ids(wide)
    reject(wide, ~wide["CHOICE"].isin([1, 2]) | wide["AV_1"].ne(1) | wide["AV_2"].ne(1), "Invalid model availability/choice.")
    extension = rich.loc[rich["SENSITIVITY_EXTENSION_FLAG"].eq(1)]
    increase = len(rich) - len(baseline)
    metrics = {
        "baseline strict occasions": len(baseline),
        "baseline strict households": baseline["H_ID"].nunique(),
        "sensitivity occasions": len(rich), "sensitivity households": rich["H_ID"].nunique(),
        "newly added sensitivity occasions": len(extension),
        # New households means absent from the baseline, not all households with added trips.
        "newly added sensitivity households": len(set(rich["H_ID"]) - set(baseline["H_ID"])),
        "households with extension occasions": extension["H_ID"].nunique(),
        "increase in occasions": increase,
        "increase relative to baseline (%)": 100 * increase / len(baseline) if len(baseline) else float("nan"),
    }
    for pattern in PATTERNS.values():
        metrics[pattern] = int(rich["PREDEPARTURE_STATE_PATTERN"].eq(pattern).sum())
    for value in [1, 2]:
        metrics[f"N_AVAILABLE_CARS == {value}"] = int(numeric(rich["N_AVAILABLE_CARS"]).eq(value).sum())
        metrics[f"CHOICE == {value}"] = int(wide["CHOICE"].eq(value).sum())
        metrics[f"conceptual AV_{value} == 1"] = len(wide)
        for state in [0, 1]:
            metrics[f"PHYS_AV_{value} == {state}"] = int(rich[f"PHYS_AV_{value}"].eq(state).sum())
    for a, b in [(1, 1), (1, 0), (0, 1)]:
        metrics[f"PHYS_AV_1 / PHYS_AV_2 == {a}/{b}"] = int((rich["PHYS_AV_1"].eq(a) & rich["PHYS_AV_2"].eq(b)).sum())
    w = wide["W_GEW"]
    metrics.update({"missing W_GEW": int(w.isna().sum()), "invalid W_GEW": 0,
                    "minimum W_GEW": w.min(), "maximum W_GEW": w.max(),
                    "mean W_GEW": w.mean(), "median W_GEW": w.median()})
    return rich, wide, pd.DataFrame(metrics.items(), columns=["metric", "value"])


def baseline_hashes() -> dict[Path, str]:
    paths = [
        CHOICE_RECONSTRUCTION_SCRIPTS_DIR / "03_phase3_strict_baseline.py",
        *sorted(PHASE3_DIR.glob("*")),
    ]
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main() -> None:
    if not CLASSIFIED_PATH.exists():
        raise SystemExit(
            f"Missing classified master: {CLASSIFIED_PATH}. "
            "Run scripts/02_choice_reconstruction/03_phase3_strict_baseline.py first."
        )
    before = baseline_hashes()
    master = read_strings(CLASSIFIED_PATH)
    rich, wide, qa = build_outputs(master, read_strings(TRIPS_PATH, ["SOURCE_ROW_ID", "W_GEW"]))
    if BASELINE_WIDE_PATH.exists():
        accepted = read_strings(BASELINE_WIDE_PATH)
        if list(accepted.columns) != WIDE_COLUMNS:
            raise ValueError(f"Accepted baseline wide schema differs: {list(accepted.columns)}")
        validate_ids(accepted)
        baseline_rows = wide.loc[wide["CHOICE_ID"].isin(master.loc[numeric(master[STRICT]).eq(1), "CHOICE_ID"])]
        reject(accepted, ~accepted["CHOICE_ID"].isin(baseline_rows["CHOICE_ID"]), "Accepted wide choice missing.")
        reject(baseline_rows, ~baseline_rows["CHOICE_ID"].isin(accepted["CHOICE_ID"]), "Classified baseline choice missing from accepted wide.")
        expected = accepted.set_index("CHOICE_ID").loc[baseline_rows["CHOICE_ID"], "SOURCE_ROW_ID"]
        reject(baseline_rows, baseline_rows["SOURCE_ROW_ID"].ne(expected.to_numpy()), "Baseline identifier mapping differs.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, path in [(rich, RICH_PATH), (wide, WIDE_PATH), (qa, QA_PATH)]:
        frame.to_csv(path, index=False)
    if baseline_hashes() != before:
        raise RuntimeError("Baseline files changed during sensitivity execution.")
    print("\nDECISION-TIMING SENSITIVITY\n")
    print(qa.to_string(index=False))
    print("\nOutputs:")
    for path in [RICH_PATH, WIDE_PATH, QA_PATH]:
        print(path)
    print("\nBaseline script and Phase 3 files verified unchanged (SHA-256).")


if __name__ == "__main__":
    main()
