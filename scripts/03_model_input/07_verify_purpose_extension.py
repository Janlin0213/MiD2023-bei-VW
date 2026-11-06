"""Read-only old-versus-new saved CSV audit for the purpose extension."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import REPO_ROOT
from src.thesis_pipeline.model_input.tour_features import VEHICLE_QA_COLUMNS

PURPOSE = "TOUR_LONGEST_DRIVER_TRIP_HWZWECK2"
RAW = ["TOUR_FIRST_PURPOSE", "TOUR_MAIN_PURPOSE", PURPOSE]
NEW = [f"{prefix}_{label}" for prefix in ("MAIN_PURPOSE", "LONGEST_PURPOSE")
       for label in ("WORK", "BUSINESS", "EDUCATION", "SHOPPING", "ERRAND", "LEISURE", "ESCORT", "MISSING")]
TOUR_ADDED = {
    PURPOSE, "TOUR_LONGEST_DRIVER_SOURCE_ROW_ID",
    "TOUR_NO_ELIGIBLE_HH_DRIVER_TRIP_FLAG", "TOUR_NO_VALID_DRIVER_DURATION_FLAG",
    "TOUR_LONGEST_DURATION_TIE_FLAG", "TOUR_LONGEST_TIE_COMMON_PURPOSE_FLAG",
    "TOUR_LONGEST_TIE_UNRESOLVED_FLAG", "TOUR_LONGEST_PURPOSE_MISSING_FLAG",
    "TOUR_LONGEST_RETURN_HOME_FLAG", "TOUR_LONGEST_IS_CLOSING_TRIP_FLAG",
    "TOUR_LONGEST_NOT_CLOSING_TRIP_FLAG", *VEHICLE_QA_COLUMNS,
}
PATHS = {
    "reconstruction/phase4/home_based_tour_features.csv": TOUR_ADDED,
    "reconstruction/phase4/mnl_vehicle_choice_wide_tour_features.csv": TOUR_ADDED,
    "reconstruction/phase4/tour_trip_membership.csv": set(),
    # This older sensitivity backbone also predates accepted tour accompaniment.
    "reconstruction/phase4_sensitivity/mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv": TOUR_ADDED | {"HOUSEHOLD_ACCOMPANIED"},
    "model_input/mnl_vehicle_choice_model_input.csv": {"TOUR_MAIN_PURPOSE", PURPOSE},
    "model_input/sensitivity/mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv": {"TOUR_MAIN_PURPOSE", PURPOSE},
    "model_input/allocation_candidate_model_input.csv": {"TOUR_MAIN_PURPOSE", PURPOSE},
    "model_input/single_driver_mnl_design_matrix.csv": set(RAW + NEW),
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def audit(baseline: Path) -> None:
    inventory = pd.read_csv(baseline / "inventory.csv", dtype=str)
    for record in inventory.itertuples(index=False):
        copy = baseline / record.path
        if not copy.is_file() or hashlib.sha256(copy.read_bytes()).hexdigest().upper() != record.sha256.upper():
            raise AssertionError(f"Baseline snapshot integrity failed: {copy}")
    for relative, allowed in PATHS.items():
        old_path = baseline / "data_processed" / relative
        new_path = REPO_ROOT / "data_processed" / relative
        old, new = read(old_path), read(new_path)
        if len(old) != len(new):
            raise AssertionError(f"Row count changed: {relative}")
        old_columns = list(old.columns)
        added = set(new.columns) - set(old_columns)
        if added != allowed or [c for c in new.columns if c in old_columns] != old_columns:
            raise AssertionError(f"Schema changed beyond intended additions: {relative}; added={added}")
        mismatches = [c for c in old_columns if not old[c].equals(new[c])]
        if mismatches:
            key = "CHOICE_ID" if "CHOICE_ID" in old else "TOUR_ID" if "TOUR_ID" in old else "SOURCE_ROW_ID"
            column = mismatches[0]
            indexes = old.index[old[column].ne(new[column])][:5]
            examples = [(old.at[i, key], old.at[i, column], new.at[i, column]) for i in indexes]
            raise AssertionError(f"Old field changed in {relative}: {mismatches}; examples={examples}")
        if relative.endswith("single_driver_mnl_design_matrix.csv") and (
            new.columns[len(old_columns):len(old_columns)+3].tolist() != RAW
            or new.columns[-16:].tolist() != NEW
        ):
            raise AssertionError("Purpose extension does not follow the historical matrix prefix.")
        print(f"PASS {relative}: {len(new):,} rows, {len(old_columns)} old fields unchanged, {len(added)} added")
    strict = read(REPO_ROOT / "data_processed/model_input/mnl_vehicle_choice_model_input.csv")
    sensitivity = read(REPO_ROOT / "data_processed/model_input/sensitivity/mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv")
    if list(strict.columns) != list(sensitivity.columns):
        raise AssertionError("Strict and sensitivity ordered schemas differ.")
    candidate = read(REPO_ROOT / "data_processed/model_input/allocation_candidate_model_input.csv")
    single = read(REPO_ROOT / "data_processed/model_input/single_driver_mnl_design_matrix.csv")
    selected = candidate.loc[candidate.SINGLE_ACTIVE_DRIVER_HH.eq("1")].reset_index(drop=True)
    if not single.CHOICE_ID.equals(selected.CHOICE_ID) or not single[RAW].equals(selected[RAW]):
        raise AssertionError("Single-driver raw purposes or row order differ from the candidate source.")
    print("PASS strict/sensitivity schema and single-driver purpose passthrough")


def audit_prechange(baseline: Path) -> None:
    inventory = pd.read_csv(baseline / "inventory.csv", dtype=str)
    relevant = ["reconstruction/phase4/home_based_tour_features.csv",
                "reconstruction/phase4/mnl_vehicle_choice_wide_tour_features.csv",
                "reconstruction/phase4/tour_trip_membership.csv",
                "reconstruction/phase4_sensitivity/mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv",
                "model_input/mnl_vehicle_choice_model_input.csv",
                "model_input/sensitivity/mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv",
                "model_input/allocation_candidate_model_input.csv",
                "model_input/single_driver_mnl_design_matrix.csv"]
    for relative in relevant:
        path = "data_processed/" + relative
        old_path = baseline / path
        record = inventory.loc[inventory.path.str.replace("\\", "/", regex=False).eq(path)]
        if len(record) != 1 or hashlib.sha256(old_path.read_bytes()).hexdigest().upper() != record.sha256.iloc[0].upper():
            raise AssertionError(f"Pre-change baseline integrity failed: {path}")
        old, new = read(old_path), read(REPO_ROOT / path)
        matrix = relative.endswith("single_driver_mnl_design_matrix.csv")
        ordered_old = list(new.columns[:len(old.columns)]) if matrix else [c for c in new if c in old.columns]
        if len(old) != len(new) or ordered_old != list(old.columns):
            raise AssertionError(f"Pre-change schema or row drift: {path}")
        changed = [column for column in old if not old[column].equals(new[column])]
        if changed:
            raise AssertionError(f"Pre-change protected values changed: {path}: {changed}")
        expected = NEW if relative.endswith("single_driver_mnl_design_matrix.csv") else (
            VEHICLE_QA_COLUMNS if relative.startswith("reconstruction/phase4") and
            not relative.endswith("tour_trip_membership.csv") else [])
        additions = list(new.columns[len(old.columns):]) if matrix else [c for c in new if c not in old.columns]
        if additions != expected:
            raise AssertionError(f"Unexpected pre-change additions: {path}")
        print(f"PASS pre-change {path}: {len(old):,} rows, {len(old.columns)} protected columns")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prechange-baseline", type=Path, required=True)
    args = parser.parse_args()
    audit(args.baseline.resolve())
    audit_prechange(args.prechange_baseline.resolve())


if __name__ == "__main__":
    main()
