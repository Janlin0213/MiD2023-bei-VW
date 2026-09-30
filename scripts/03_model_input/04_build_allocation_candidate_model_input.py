"""Build a feature-complete model backbone for the allocation candidate.

The accepted allocation-analysis candidate defines the row universe.  This
script attaches the already accepted tour features and reuses the canonical
vehicle-choice feature builders without applying any further sample selection.
The final schema is the current strict MNL schema plus two authoritative
household-branch flags.  No joint alternatives or persistence features are
constructed here.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.model_input import tour_features as accepted_tours
from src.thesis_pipeline.model_input import vehicle_choice as accepted_model
from src.thesis_pipeline.paths import (
    ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
    ALLOCATION_CANDIDATE_MODEL_INPUT_PATH,
    ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH,
    ALLOCATION_CANDIDATE_OCCASIONS_PATH,
    MODEL_INPUT_DIR,
    MODEL_INPUT_SENSITIVITY_DIR,
    PHASE1_DIR,
    PHASE3_SENSITIVITY_DIR,
    PHASE4_DIR,
    PHASE4_SENSITIVITY_DIR,
)
from src.thesis_pipeline.sensitivity.decision_timing import (
    CONTINUOUS_COLUMNS,
    serialized,
)


STRICT_MODEL_INPUT_PATH = MODEL_INPUT_DIR / "mnl_vehicle_choice_model_input.csv"
SENSITIVITY_MODEL_INPUT_PATH = (
    MODEL_INPUT_SENSITIVITY_DIR
    / "mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv"
)
SENSITIVITY_WIDE_PATH = (
    PHASE3_SENSITIVITY_DIR
    / "mnl_vehicle_choice_wide_decision_timing_sensitivity_base.csv"
)
SENSITIVITY_TOUR_BACKBONE_PATH = (
    PHASE4_SENSITIVITY_DIR
    / "mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv"
)
TRIPS_PATH = PHASE1_DIR / "trips_home_chain_enriched.csv"
TOUR_FEATURES_PATH = PHASE4_DIR / "home_based_tour_features.csv"

BRANCH_FLAGS = ["SINGLE_ACTIVE_DRIVER_HH", "MULTI_ACTIVE_DRIVER_HH"]
COMPARISON_RTOL = 1e-10
COMPARISON_ATOL = 1e-10

REFERENCE_COUNTS = {
    "candidate occasions": 38_779,
    "candidate households": 23_919,
    "strict overlap": 30_816,
    "sensitivity overlap": 38_143,
    "candidate-only vs sensitivity": 636,
    "single-active-driver households": 16_986,
    "multi-active-driver households": 6_933,
    "single-active-driver occasions": 20_931,
    "multi-active-driver occasions": 17_848,
}

CORE_REQUIRED_COLUMNS = [
    *accepted_model.ESTIMATION_COLUMNS,
    "START_MIN",
    "TOUR_DISTANCE_KM",
    "TOUR_TRAVEL_TIME_MIN",
    *BRANCH_FLAGS,
]


def read_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return accepted_model.read_csv_strings(path, usecols)


def require_columns(
    columns: Iterable[str], required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def assert_outputs_absent() -> None:
    existing = [
        path
        for path in [
            ALLOCATION_CANDIDATE_MODEL_INPUT_PATH,
            ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH,
        ]
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing allocation-candidate model output(s): "
            f"{existing}"
        )


def file_hashes(paths: Iterable[Path]) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def build_candidate_wide(
    occasions: pd.DataFrame,
    trips: pd.DataFrame,
    sensitivity_wide: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "CHOICE_ID",
        "SOURCE_ROW_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "CHOSEN_A_ID",
        "CHOSEN_A_ID_CANONICAL",
        "START_MIN",
        "ARRIVAL_MIN",
        "W_ZWECK",
        "W_SO1",
    ]
    require_columns(occasions.columns, required, "candidate occasion")
    require_columns(
        trips.columns,
        ["SOURCE_ROW_ID", "W_GEW", "CHAIN_ELIGIBLE", "HOME_ORIGIN"],
        "Phase-1 trip",
    )

    for column in ["CHOICE_ID", "SOURCE_ROW_ID"]:
        if occasions[column].eq("").any() or occasions[column].duplicated().any():
            raise ValueError(
                f"Candidate {column} must be populated and unique before feature construction."
            )
    for column in ["H_ID", "HP_ID", "P_ID", "W_ID"]:
        if occasions[column].eq("").any():
            raise ValueError(f"Candidate {column} must be populated.")

    chosen = numeric(occasions["CHOSEN_A_ID"])
    canonical_chosen = numeric(occasions["CHOSEN_A_ID_CANONICAL"])
    if not chosen.isin([1, 2]).all() or not chosen.equals(canonical_chosen):
        raise ValueError(
            "Candidate chosen-vehicle coding must be canonical and restricted to alternatives 1/2."
        )

    if trips["SOURCE_ROW_ID"].eq("").any() or trips["SOURCE_ROW_ID"].duplicated().any():
        raise ValueError("Phase-1 SOURCE_ROW_ID must be populated and unique.")
    weights = numeric(trips["W_GEW"])
    nonnumeric = trips["W_GEW"].ne("") & weights.isna()
    if nonnumeric.any():
        raise ValueError("Phase-1 W_GEW contains non-numeric populated values.")
    weight_source = trips[["SOURCE_ROW_ID"]].copy()
    weight_source["W_GEW"] = weights

    wide = occasions[
        [
            "CHOICE_ID",
            "SOURCE_ROW_ID",
            "H_ID",
            "HP_ID",
            "P_ID",
            "W_ID",
            "CHOSEN_A_ID",
            "START_MIN",
            "ARRIVAL_MIN",
            "W_ZWECK",
            "W_SO1",
        ]
    ].rename(columns={"CHOSEN_A_ID": "CHOICE"})
    wide["_CANDIDATE_ROW_ORDER"] = np.arange(len(wide), dtype=np.int64)
    wide["AV_1"] = 1
    wide["AV_2"] = 1
    wide = wide.merge(
        weight_source,
        on="SOURCE_ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if wide["_merge"].ne("both").any():
        print(
            wide.loc[
                wide["_merge"].ne("both"),
                ["CHOICE_ID", "SOURCE_ROW_ID", "_merge"],
            ].head(10).to_string(index=False)
        )
        raise ValueError("Every candidate occasion must match its Phase-1 survey weight.")
    wide = (
        wide.sort_values("_CANDIDATE_ROW_ORDER", kind="mergesort")
        .drop(columns=["_CANDIDATE_ROW_ORDER", "_merge"])
        .reset_index(drop=True)
    )

    expected_columns = list(sensitivity_wide.columns)
    missing = sorted(set(expected_columns) - set(wide.columns))
    extra = sorted(set(wide.columns) - set(expected_columns))
    if missing or extra:
        raise AssertionError(
            "Candidate conceptual-choice backbone differs from the accepted sensitivity "
            f"schema; missing={missing}, extra={extra}."
        )
    wide = wide[expected_columns]
    if not wide[["CHOICE_ID", "SOURCE_ROW_ID"]].equals(
        occasions[["CHOICE_ID", "SOURCE_ROW_ID"]].reset_index(drop=True)
    ):
        raise AssertionError("Candidate wide construction changed identifier order.")

    choice = numeric(wide["CHOICE"])
    weight = numeric(wide["W_GEW"])
    if (
        not choice.isin([1, 2]).all()
        or numeric(wide["AV_1"]).ne(1).any()
        or numeric(wide["AV_2"]).ne(1).any()
        or weight.isna().any()
        or not weight.map(lambda value: math.isfinite(float(value))).all()
        or weight.le(0).any()
    ):
        raise ValueError("Candidate CHOICE/availability/weight structure is invalid.")
    return wide


def build_tour_backbone(
    wide: pd.DataFrame,
    trips: pd.DataFrame,
    tours: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    backbone = accepted_tours.merge_to_strict_wide(wide, trips, tours)
    matches = numeric(backbone["TOUR_MATCH_FOUND"]).eq(1)
    matched_count = int(matches.sum())
    unmatched_count = int((~matches).sum())
    if unmatched_count:
        print(
            backbone.loc[
                ~matches,
                ["CHOICE_ID", "SOURCE_ROW_ID", "TOUR_MATCH_FOUND"],
            ].head(10).to_string(index=False)
        )
        raise AssertionError(
            "Every authoritative candidate occasion must match an accepted tour start."
        )
    return backbone, matched_count, unmatched_count


def load_model_feature_sources(
    paths: accepted_model.Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trip = read_strings(paths.trips, accepted_model.TRIP_SOURCE_COLUMNS)
    household_header = pd.read_csv(paths.households, nrows=0).columns.tolist()
    household_columns = [*accepted_model.HOUSEHOLD_SOURCE_COLUMNS]
    if "household_type" in household_header:
        household_columns.append("household_type")
    household = read_strings(paths.households, household_columns)
    person = read_strings(paths.persons, accepted_model.PERSON_SOURCE_COLUMNS)
    vehicle = read_strings(paths.vehicles, accepted_model.VEHICLE_SOURCE_COLUMNS)
    return trip, household, person, vehicle


def build_canonical_features(
    backbone: pd.DataFrame,
    canonical_columns: list[str],
    paths: accepted_model.Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], pd.Series]:
    trip_source, household_source, person_source, vehicle_source = (
        load_model_feature_sources(paths)
    )
    invariants = accepted_model.validate_backbone(backbone)
    accepted_model.canonicalize_candidate_registry(paths.candidate_registry)
    tour = accepted_model.select_tour_features(backbone)
    trip = accepted_model.select_trip_context_features(trip_source)
    household = accepted_model.select_household_features(
        household_source, person_source
    )
    person = accepted_model.select_person_features(person_source)
    vehicle = accepted_model.prepare_vehicle_features(
        vehicle_source, set(backbone["H_ID"])
    )
    merged, merge_metrics = accepted_model.merge_features(
        backbone,
        tour,
        trip,
        household,
        person,
        vehicle,
    )
    accepted_model.validate_missing_codes(merged)
    missing = sorted(set(canonical_columns) - set(merged.columns))
    if missing:
        raise AssertionError(
            f"Current canonical model columns cannot be constructed: {missing}"
        )
    canonical = merged[canonical_columns].copy()
    accepted_model.run_final_assertions(
        invariants,
        backbone,
        merged,
        canonical,
        vehicle,
    )

    merge_match_columns = [
        "_TRIP_KEY_MATCH",
        "_HOUSEHOLD_KEY_MATCH",
        "_PERSON_KEY_MATCH",
        "_VEHICLE_KEY_MATCH_1",
        "_VEHICLE_KEY_MATCH_2",
    ]
    complete_merge = merged[merge_match_columns].eq(1).all(axis=1)
    complete_merge &= numeric(backbone["TOUR_MATCH_FOUND"]).eq(1).to_numpy()
    if not complete_merge.all():
        raise AssertionError("At least one candidate row has an incomplete feature-source merge.")
    return canonical, merged, merge_metrics, complete_merge


def attach_branch_flags(
    canonical: pd.DataFrame,
    households: pd.DataFrame,
    canonical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(households.columns, ["H_ID", *BRANCH_FLAGS], "candidate household")
    if households["H_ID"].eq("").any() or households["H_ID"].duplicated().any():
        raise ValueError("Authoritative candidate-household H_ID must be populated and unique.")

    branches = households[["H_ID", *BRANCH_FLAGS]].copy()
    for column in BRANCH_FLAGS:
        values = numeric(branches[column])
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"Authoritative branch flag {column} must be populated and binary.")
        branches[column] = values.astype(int)
    if branches[BRANCH_FLAGS].sum(axis=1).ne(1).any():
        raise ValueError("Every candidate household must belong to exactly one modelling branch.")

    final = canonical.copy()
    final["_CANDIDATE_ROW_ORDER"] = np.arange(len(final), dtype=np.int64)
    final = final.merge(
        branches,
        on="H_ID",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )
    if final["_merge"].ne("both").any():
        print(
            final.loc[
                final["_merge"].ne("both"),
                ["CHOICE_ID", "H_ID", "_merge"],
            ].head(10).to_string(index=False)
        )
        raise AssertionError(
            "Every candidate occasion must map to an authoritative household branch."
        )
    final = (
        final.sort_values("_CANDIDATE_ROW_ORDER", kind="mergesort")
        .drop(columns=["_CANDIDATE_ROW_ORDER", "_merge"])
        .reset_index(drop=True)
    )
    expected_columns = [*canonical_columns, *BRANCH_FLAGS]
    if list(final.columns) != expected_columns:
        raise AssertionError("Final schema is not canonical columns plus the two branch flags.")
    if final[BRANCH_FLAGS].sum(axis=1).ne(1).any():
        raise AssertionError("Final rows must belong to exactly one modelling branch.")
    return final, branches


def compare_overlap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    canonical_columns: list[str],
    label: str,
) -> dict[str, object]:
    if list(reference.columns) != canonical_columns:
        raise AssertionError(f"{label} reference does not use the canonical ordered schema.")
    if not reference["CHOICE_ID"].is_unique or reference["CHOICE_ID"].eq("").any():
        raise AssertionError(f"{label} reference CHOICE_ID must be populated and unique.")
    missing_ids = set(reference["CHOICE_ID"]) - set(candidate["CHOICE_ID"])
    if missing_ids:
        raise AssertionError(
            f"Candidate model input is missing {len(missing_ids):,} {label} CHOICE_ID values."
        )

    old = reference.set_index("CHOICE_ID")
    new = candidate.set_index("CHOICE_ID").loc[old.index, canonical_columns[1:]]
    differences = pd.DataFrame(False, index=old.index, columns=canonical_columns)
    differences["CHOICE_ID"] = False
    for column in canonical_columns[1:]:
        old_values = old[column].astype("string").fillna("")
        new_values = new[column].astype("string").fillna("")
        if column in CONTINUOUS_COLUMNS:
            old_numeric = pd.to_numeric(old_values, errors="coerce").to_numpy(float)
            new_numeric = pd.to_numeric(new_values, errors="coerce").to_numpy(float)
            same = np.isclose(
                old_numeric,
                new_numeric,
                rtol=COMPARISON_RTOL,
                atol=COMPARISON_ATOL,
                equal_nan=False,
            )
            same |= old_values.eq("").to_numpy() & new_values.eq("").to_numpy()
            differences[column] = ~same
        else:
            differences[column] = old_values.ne(new_values).to_numpy()

    row_mismatch = differences.any(axis=1)
    mismatch_columns = [
        column for column in canonical_columns if differences[column].any()
    ]
    examples: list[dict[str, str]] = []
    if row_mismatch.any():
        for column in mismatch_columns:
            for choice_id in differences.index[differences[column]][:5]:
                examples.append(
                    {
                        "CHOICE_ID": choice_id,
                        "column": column,
                        "reference": old.at[choice_id, column],
                        "candidate": new.at[choice_id, column],
                    }
                )
            if len(examples) >= 10:
                break
        print(f"\n{label.upper()} OVERLAP MISMATCH EXAMPLES")
        print(pd.DataFrame(examples[:10]).to_string(index=False))
        raise AssertionError(
            f"{label} overlap contains canonical-feature differences; "
            f"rtol={COMPARISON_RTOL}, atol={COMPARISON_ATOL}."
        )
    return {
        "overlap_rows": len(reference),
        "mismatch_rows": int(row_mismatch.sum()),
        "mismatch_columns": len(mismatch_columns),
        "mismatch_column_names": ", ".join(mismatch_columns),
    }


def required_missing_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[CORE_REQUIRED_COLUMNS].astype("string").fillna("").eq("").any(axis=1)


def validate_final(
    final: pd.DataFrame,
    serialized_final: pd.DataFrame,
    occasions: pd.DataFrame,
    branches: pd.DataFrame,
    canonical_columns: list[str],
    sensitivity_ids: set[str],
    complete_merge: pd.Series,
) -> dict[str, object]:
    expected_columns = [*canonical_columns, *BRANCH_FLAGS]
    if list(final.columns) != expected_columns:
        raise AssertionError("Final ordered columns differ from canonical schema plus branch flags.")
    if len(final) != len(occasions):
        raise AssertionError("Final row count differs from the authoritative candidate count.")
    if final["CHOICE_ID"].eq("").any() or not final["CHOICE_ID"].is_unique:
        raise AssertionError("Final CHOICE_ID must be populated and unique.")
    if set(final["CHOICE_ID"]) != set(occasions["CHOICE_ID"]):
        raise AssertionError("Final and authoritative candidate CHOICE_ID universes differ.")
    if not final[["CHOICE_ID", "SOURCE_ROW_ID"]].equals(
        occasions[["CHOICE_ID", "SOURCE_ROW_ID"]].reset_index(drop=True)
    ):
        raise AssertionError("Final candidate row/identifier order changed.")
    if set(final["H_ID"]) != set(branches["H_ID"]):
        raise AssertionError("Final household universe differs from the branch registry.")
    for column in BRANCH_FLAGS:
        values = numeric(final[column])
        if values.isna().any() or not values.isin([0, 1]).all():
            raise AssertionError(f"Final branch flag {column} is not populated and binary.")
    if final[BRANCH_FLAGS].sum(axis=1).ne(1).any():
        raise AssertionError("Final rows do not map to exactly one household branch.")

    choice = numeric(final["CHOICE"])
    av_1 = numeric(final["AV_1"])
    av_2 = numeric(final["AV_2"])
    if not choice.isin([1, 2]).all():
        raise AssertionError("Final CHOICE is not restricted to alternatives 1/2.")
    if av_1.ne(1).any() or av_2.ne(1).any():
        raise AssertionError("Final conceptual availability must equal one for both cars.")

    candidate_only = ~serialized_final["CHOICE_ID"].isin(sensitivity_ids)
    required_missing = required_missing_mask(serialized_final)
    if required_missing.any():
        print(
            serialized_final.loc[
                required_missing,
                ["CHOICE_ID", *CORE_REQUIRED_COLUMNS],
            ].head(10).to_string(index=False)
        )
        raise AssertionError("Final rows contain missing core required model fields.")
    if not complete_merge.loc[candidate_only.to_numpy()].all():
        raise AssertionError("A candidate-only row has an incomplete feature-source merge.")

    candidate_meta = occasions.set_index("CHOICE_ID").loc[
        serialized_final.loc[candidate_only, "CHOICE_ID"]
    ]
    if numeric(candidate_meta["simultaneous_home_departure_flag"]).ne(1).any():
        raise AssertionError(
            "Candidate-only additions must be the validated simultaneous-departure occasions."
        )
    if numeric(candidate_meta["IN_DECISION_TIMING_SENSITIVITY"]).ne(0).any():
        raise AssertionError("Candidate-only additions unexpectedly belong to sensitivity.")

    single_hh = int(branches["SINGLE_ACTIVE_DRIVER_HH"].sum())
    multi_hh = int(branches["MULTI_ACTIVE_DRIVER_HH"].sum())
    single_occ = int(numeric(final["SINGLE_ACTIVE_DRIVER_HH"]).eq(1).sum())
    multi_occ = int(numeric(final["MULTI_ACTIVE_DRIVER_HH"]).eq(1).sum())
    if single_occ + multi_occ != len(final) or single_hh + multi_hh != len(branches):
        raise AssertionError("Branch household/occasion counts do not reconcile.")

    any_canonical_blank = serialized_final[canonical_columns].eq("").any(axis=1)
    return {
        "candidate_only_rows": int(candidate_only.sum()),
        "candidate_only_complete_required": int((candidate_only & ~required_missing).sum()),
        "candidate_only_missing_required": int((candidate_only & required_missing).sum()),
        "candidate_only_any_documented_missing": int(
            (candidate_only & any_canonical_blank).sum()
        ),
        "single_households": single_hh,
        "multi_households": multi_hh,
        "single_occasions": single_occ,
        "multi_occasions": multi_occ,
    }


def qa_row(
    section: str,
    metric: str,
    value: object,
    description: str,
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "count_or_value": value,
        "description": description,
    }


def build_qa(
    occasions: pd.DataFrame,
    households: pd.DataFrame,
    strict: pd.DataFrame,
    sensitivity: pd.DataFrame,
    final: pd.DataFrame,
    serialized_final: pd.DataFrame,
    canonical_columns: list[str],
    merge_metrics: dict[str, int],
    tour_matched: int,
    tour_unmatched: int,
    strict_comparison: dict[str, object],
    sensitivity_comparison: dict[str, object],
    final_metrics: dict[str, object],
    complete_merge: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object, description: str) -> None:
        rows.append(qa_row(section, metric, value, description))

    add("INPUT", "candidate occasions", len(occasions), str(ALLOCATION_CANDIDATE_OCCASIONS_PATH))
    add("INPUT", "candidate households", len(households), str(ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH))
    add("INPUT", "canonical strict input rows", len(strict), str(STRICT_MODEL_INPUT_PATH))
    add("INPUT", "canonical sensitivity input rows", len(sensitivity), str(SENSITIVITY_MODEL_INPUT_PATH))

    add("OUTPUT", "final model-input rows", len(final), "one row per authoritative candidate CHOICE_ID")
    add("OUTPUT", "unique CHOICE_ID", final["CHOICE_ID"].nunique(), "must equal final rows")
    add("OUTPUT", "column count", len(final.columns), "canonical schema plus two branch flags")
    add("OUTPUT", "canonical model column count", len(canonical_columns), "ordered columns read from current strict model input")
    add("OUTPUT", "duplicate CHOICE_ID", int(final["CHOICE_ID"].duplicated().sum()), "must be zero")
    add("OUTPUT", "missing H_ID", int(serialized_final["H_ID"].eq("").sum()), "must be zero")
    add("OUTPUT", "missing branch flag", int(serialized_final[BRANCH_FLAGS].eq("").any(axis=1).sum()), "must be zero")

    add("BRANCH STRUCTURE", "single-active-driver households", final_metrics["single_households"], "authoritative household branch registry")
    add("BRANCH STRUCTURE", "multi-active-driver households", final_metrics["multi_households"], "authoritative household branch registry")
    add("BRANCH STRUCTURE", "single-active-driver occasions", final_metrics["single_occasions"], "candidate occasions after authoritative H_ID merge")
    add("BRANCH STRUCTURE", "multi-active-driver occasions", final_metrics["multi_occasions"], "candidate occasions after authoritative H_ID merge")

    choice = numeric(final["CHOICE"])
    add("CHOICE STRUCTURE", "CHOICE == 1", int(choice.eq(1).sum()), "observed Car 1 choices")
    add("CHOICE STRUCTURE", "CHOICE == 2", int(choice.eq(2).sum()), "observed Car 2 choices")
    add("CHOICE STRUCTURE", "invalid CHOICE", int((~choice.isin([1, 2])).sum()), "must be zero")
    add("CHOICE STRUCTURE", "AV_1 != 1", int(numeric(final["AV_1"]).ne(1).sum()), "conceptual two-car availability; must be zero")
    add("CHOICE STRUCTURE", "AV_2 != 1", int(numeric(final["AV_2"]).ne(1).sum()), "conceptual two-car availability; must be zero")

    required_missing = required_missing_mask(serialized_final)
    add("FEATURE COVERAGE", "candidate rows with complete canonical feature merge", int(complete_merge.sum()), "tour, trip, household, person, and both vehicle sources matched")
    add("FEATURE COVERAGE", "candidate rows missing any required model field", int(required_missing.sum()), f"required fields: {', '.join(CORE_REQUIRED_COLUMNS)}")
    add("FEATURE COVERAGE", "tour-feature matches", tour_matched, "accepted tour-start merge")
    add("FEATURE COVERAGE", "tour-feature unmatched", tour_unmatched, "must be zero")
    add("FEATURE COVERAGE", "trip-feature matches", len(final) - merge_metrics["source_row_unmatched"], "exact SOURCE_ROW_ID merge")
    add("FEATURE COVERAGE", "trip-feature unmatched", merge_metrics["source_row_unmatched"], "must be zero")
    add("FEATURE COVERAGE", "household-feature matches", len(final) - merge_metrics["household_unmatched"], "H_ID merge")
    add("FEATURE COVERAGE", "household-feature unmatched", merge_metrics["household_unmatched"], "must be zero")
    add("FEATURE COVERAGE", "person-feature matches", len(final) - merge_metrics["person_unmatched"], "HP_ID merge with H_ID/P_ID consistency check")
    add("FEATURE COVERAGE", "person-feature unmatched", merge_metrics["person_unmatched"], "must be zero")
    add("FEATURE COVERAGE", "vehicle-1 feature matches", len(final) - merge_metrics["vehicle_1_unmatched"], "H_ID plus alternative 1")
    add("FEATURE COVERAGE", "vehicle-1 feature unmatched", merge_metrics["vehicle_1_unmatched"], "must be zero")
    add("FEATURE COVERAGE", "vehicle-2 feature matches", len(final) - merge_metrics["vehicle_2_unmatched"], "H_ID plus alternative 2")
    add("FEATURE COVERAGE", "vehicle-2 feature unmatched", merge_metrics["vehicle_2_unmatched"], "must be zero")

    for section, comparison in [
        ("STRICT REGRESSION COMPARISON", strict_comparison),
        ("SENSITIVITY REGRESSION COMPARISON", sensitivity_comparison),
    ]:
        add(section, "overlapping CHOICE_IDs", comparison["overlap_rows"], "all reference CHOICE_IDs must be present")
        add(section, "rows with any canonical-feature mismatch", comparison["mismatch_rows"], f"continuous rtol={COMPARISON_RTOL}, atol={COMPARISON_ATOL}; other fields exact")
        add(section, "columns with mismatch", comparison["mismatch_columns"], comparison["mismatch_column_names"] or "none")

    add("CANDIDATE-ONLY ADDITION", "candidate-only vs sensitivity CHOICE_IDs", final_metrics["candidate_only_rows"], "authoritative candidate minus canonical sensitivity")
    add("CANDIDATE-ONLY ADDITION", "candidate-only rows with complete required features", final_metrics["candidate_only_complete_required"], "all source merges complete and core required fields populated")
    add("CANDIDATE-ONLY ADDITION", "candidate-only rows with missing required features", final_metrics["candidate_only_missing_required"], "must be zero")
    add("CANDIDATE-ONLY ADDITION", "candidate-only rows with any documented canonical missing value", final_metrics["candidate_only_any_documented_missing"], "retained canonical missingness; no imputation or filtering")

    weights = numeric(final["W_GEW"])
    add("WEIGHT QA", "missing W_GEW", int(weights.isna().sum()), "same canonical source and treatment")
    add("WEIGHT QA", "zero W_GEW", int(weights.eq(0).sum()), "must be zero")
    add("WEIGHT QA", "negative W_GEW", int(weights.lt(0).sum()), "must be zero")
    add("WEIGHT QA", "min", float(weights.min()), "raw W_GEW")
    add("WEIGHT QA", "max", float(weights.max()), "raw W_GEW")
    add("WEIGHT QA", "mean", float(weights.mean()), "raw W_GEW")
    add("WEIGHT QA", "median", float(weights.median()), "raw W_GEW")
    return pd.DataFrame(rows, columns=["section", "metric", "count_or_value", "description"])


def warn_reference_drift(actual: dict[str, int]) -> None:
    mismatches = [
        (metric, actual[metric], reference)
        for metric, reference in REFERENCE_COUNTS.items()
        if actual[metric] != reference
    ]
    if mismatches:
        print("\nWARNING: REFERENCE COUNTS CHANGED")
        for metric, observed, reference in mismatches:
            print(f"- {metric}: observed {observed:,}; reference only {reference:,}")


def save_and_verify(final: pd.DataFrame, qa: pd.DataFrame) -> None:
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, index=False)
    qa.to_csv(ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH, index=False)

    saved_final = read_strings(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH)
    expected_final = serialized(final)
    if list(saved_final.columns) != list(expected_final.columns) or not (
        saved_final.astype("string").fillna("").reset_index(drop=True)
    ).equals(
        expected_final.astype("string").fillna("").reset_index(drop=True)
    ):
        raise AssertionError("Saved allocation-candidate model input failed read-back equality.")
    saved_qa = read_strings(ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH)
    if list(saved_qa.columns) != ["section", "metric", "count_or_value", "description"]:
        raise AssertionError("Saved allocation-candidate QA schema changed during serialization.")
    if len(saved_qa) != len(qa):
        raise AssertionError("Saved allocation-candidate QA row count changed during serialization.")


def print_summary(
    final: pd.DataFrame,
    canonical_columns: list[str],
    final_metrics: dict[str, object],
    strict_comparison: dict[str, object],
    sensitivity_comparison: dict[str, object],
    tour_matched: int,
    tour_unmatched: int,
) -> None:
    weights = numeric(final["W_GEW"])
    print("\nALLOCATION-CANDIDATE MODEL INPUT")
    print(f"final rows: {len(final):,}")
    print(f"unique CHOICE_ID: {final['CHOICE_ID'].nunique():,}")
    print(f"unique households: {final['H_ID'].nunique():,}")
    print(f"canonical model columns: {len(canonical_columns):,}")
    print(f"total columns after branch flags: {len(final.columns):,}")
    print(
        "single-active-driver households / occasions: "
        f"{int(final_metrics['single_households']):,} / "
        f"{int(final_metrics['single_occasions']):,}"
    )
    print(
        "multi-active-driver households / occasions: "
        f"{int(final_metrics['multi_households']):,} / "
        f"{int(final_metrics['multi_occasions']):,}"
    )
    print(
        "strict overlap / mismatch rows: "
        f"{int(strict_comparison['overlap_rows']):,} / "
        f"{int(strict_comparison['mismatch_rows']):,}"
    )
    print(
        "sensitivity overlap / mismatch rows: "
        f"{int(sensitivity_comparison['overlap_rows']):,} / "
        f"{int(sensitivity_comparison['mismatch_rows']):,}"
    )
    print(
        "candidate-only rows / complete required features: "
        f"{int(final_metrics['candidate_only_rows']):,} / "
        f"{int(final_metrics['candidate_only_complete_required']):,}"
    )
    print(f"tour-feature coverage: {tour_matched:,} matched / {tour_unmatched:,} unmatched")
    print(
        "W_GEW missing / zero / negative: "
        f"{int(weights.isna().sum()):,} / {int(weights.eq(0).sum()):,} / "
        f"{int(weights.lt(0).sum()):,}"
    )
    print(
        "W_GEW min / max / mean / median: "
        f"{weights.min():.12g} / {weights.max():.12g} / "
        f"{weights.mean():.12g} / {weights.median():.12g}"
    )
    print("OUTPUTS:")
    print(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH.resolve())
    print(ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH.resolve())
    print(
        "No joint alternatives, switching variables, persistence variables, "
        "or branch-specific physical files were created."
    )


def main() -> None:
    assert_outputs_absent()
    required_inputs = [
        ALLOCATION_CANDIDATE_OCCASIONS_PATH,
        ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
        STRICT_MODEL_INPUT_PATH,
        SENSITIVITY_MODEL_INPUT_PATH,
        SENSITIVITY_WIDE_PATH,
        SENSITIVITY_TOUR_BACKBONE_PATH,
        TRIPS_PATH,
        TOUR_FEATURES_PATH,
    ]
    missing_inputs = [path for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Required accepted input(s) not found: {missing_inputs}")

    protected_paths = [
        STRICT_MODEL_INPUT_PATH,
        SENSITIVITY_MODEL_INPUT_PATH,
        SENSITIVITY_WIDE_PATH,
        SENSITIVITY_TOUR_BACKBONE_PATH,
        TOUR_FEATURES_PATH,
    ]
    protected_before = file_hashes(protected_paths)

    occasions = read_strings(ALLOCATION_CANDIDATE_OCCASIONS_PATH)
    households = read_strings(ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH)
    trips = accepted_tours.read_csv_strings(TRIPS_PATH)
    tours = accepted_tours.read_csv_strings(TOUR_FEATURES_PATH)
    sensitivity_wide = read_strings(SENSITIVITY_WIDE_PATH)
    strict = read_strings(STRICT_MODEL_INPUT_PATH)
    sensitivity = read_strings(SENSITIVITY_MODEL_INPUT_PATH)

    canonical_columns = list(strict.columns)
    if list(sensitivity.columns) != canonical_columns:
        raise AssertionError("Current strict and sensitivity model schemas differ.")

    wide = build_candidate_wide(occasions, trips, sensitivity_wide)
    backbone, tour_matched, tour_unmatched = build_tour_backbone(wide, trips, tours)

    paths = accepted_model.resolve_paths()
    canonical, merged, merge_metrics, complete_merge = build_canonical_features(
        backbone,
        canonical_columns,
        paths,
    )
    final, branches = attach_branch_flags(canonical, households, canonical_columns)
    serialized_final = serialized(final)

    strict_comparison = compare_overlap(
        strict,
        serialized_final[canonical_columns],
        canonical_columns,
        "strict",
    )
    sensitivity_comparison = compare_overlap(
        sensitivity,
        serialized_final[canonical_columns],
        canonical_columns,
        "sensitivity",
    )
    final_metrics = validate_final(
        final,
        serialized_final,
        occasions,
        branches,
        canonical_columns,
        set(sensitivity["CHOICE_ID"]),
        complete_merge,
    )

    actual_counts = {
        "candidate occasions": len(final),
        "candidate households": final["H_ID"].nunique(),
        "strict overlap": int(strict_comparison["overlap_rows"]),
        "sensitivity overlap": int(sensitivity_comparison["overlap_rows"]),
        "candidate-only vs sensitivity": int(final_metrics["candidate_only_rows"]),
        "single-active-driver households": int(final_metrics["single_households"]),
        "multi-active-driver households": int(final_metrics["multi_households"]),
        "single-active-driver occasions": int(final_metrics["single_occasions"]),
        "multi-active-driver occasions": int(final_metrics["multi_occasions"]),
    }
    warn_reference_drift(actual_counts)

    qa = build_qa(
        occasions,
        households,
        strict,
        sensitivity,
        final,
        serialized_final,
        canonical_columns,
        merge_metrics,
        tour_matched,
        tour_unmatched,
        strict_comparison,
        sensitivity_comparison,
        final_metrics,
        complete_merge,
    )

    if file_hashes(protected_paths) != protected_before:
        raise AssertionError("An accepted strict/sensitivity input changed before output writing.")
    save_and_verify(final, qa)
    if file_hashes(protected_paths) != protected_before:
        raise AssertionError("An accepted strict/sensitivity input changed during execution.")

    print_summary(
        final,
        canonical_columns,
        final_metrics,
        strict_comparison,
        sensitivity_comparison,
        tour_matched,
        tour_unmatched,
    )


if __name__ == "__main__":
    main()
