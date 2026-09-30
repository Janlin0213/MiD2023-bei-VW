"""Build the household-day allocation-analysis candidate universe.

The script consumes the accepted Phase-3 classified occasion master and does
not reconstruct trips or vehicle states.  It retains physically coherent
HOME|HOME, HOME|AWAY, and AWAY|HOME home-origin occasions, including valid
simultaneous departures, and produces occasion-, household-, and QA-level
outputs in a dedicated reconstruction directory.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (
    ALLOCATION_CANDIDATE_DIR,
    ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
    ALLOCATION_CANDIDATE_OCCASIONS_PATH,
    ALLOCATION_CANDIDATE_QA_PATH,
    PHASE3_DIR,
    PHASE3_SENSITIVITY_DIR,
    SELECTED_RAW_DIR,
)


CLASSIFIED_PATH = PHASE3_DIR / "vehicle_choice_occasions_classified.csv"
SENSITIVITY_PATH = (
    PHASE3_SENSITIVITY_DIR
    / "vehicle_choice_occasions_decision_timing_sensitivity.csv"
)
CARS_PATH = SELECTED_RAW_DIR / "cars_selected_raw.csv"

REFERENCE_STRICT_OCCASIONS = 30_816
REFERENCE_SENSITIVITY_OCCASIONS = 38_143

VALID_STATE_CLASSES = {"HOME|HOME", "HOME|AWAY", "AWAY|HOME"}
STATE_CLASS_BY_AVAILABLE_SET = {
    "1|2": "HOME|HOME",
    "1": "HOME|AWAY",
    "2": "AWAY|HOME",
}
CELL_LABELS = {
    1: "Cell 1: single driver, one vehicle observed",
    2: "Cell 2: single driver, multiple vehicles",
    3: "Cell 3: multiple drivers, observed one-to-one matching",
    4: "Cell 4: multiple drivers, flexible or mixed mapping",
}

REQUIRED_MASTER_COLUMNS = [
    "CHOICE_ID",
    "TRIP_ID",
    "SOURCE_ROW_ID",
    "H_ID",
    "HP_ID",
    "P_ID",
    "W_ID",
    "START_MIN",
    "ARRIVAL_MIN",
    "HOME_ORIGIN",
    "CHOSEN_A_ID",
    "AVAILABLE_CAR_SET",
    "N_AVAILABLE_CARS",
    "snapshot_valid_flag",
    "invalid_snapshot_flag",
    "unknown_vehicle_state_at_snapshot_flag",
    "unmatched_reported_vehicle_flag",
    "duplicate_trip_identifier_flag",
    "complete_vehicle_time_flag",
    "household_vehicle_time_unresolved_flag",
    "household_zero_duration_driver_trip_flag",
    "household_vehicle_trip_overlap_flag",
    "household_vehicle_location_transition_conflict_flag",
    "household_vehicle_timeline_conflict_flag",
    "availability_conflict_flag",
    "simultaneous_home_departure_flag",
    "same_vehicle_simultaneous_departure_flag",
    "complete_fleet_baseline_flag",
    "topcoded_multicar_household_flag",
    "fleet_inventory_mismatch_flag",
    "household_car_count_conflict_flag",
    "nonstandard_A_ID_set_flag",
    "invalid_start_time_flag",
    "invalid_arrival_time_flag",
    "negative_duration_flag",
    "FLEET_CAR_SET",
    "N_RECORDED_CARS",
    "VALID_STRICT_CHOICE_OCCASION",
]

BINARY_MASTER_COLUMNS = [
    "snapshot_valid_flag",
    "invalid_snapshot_flag",
    "unknown_vehicle_state_at_snapshot_flag",
    "unmatched_reported_vehicle_flag",
    "duplicate_trip_identifier_flag",
    "complete_vehicle_time_flag",
    "household_vehicle_time_unresolved_flag",
    "household_zero_duration_driver_trip_flag",
    "household_vehicle_trip_overlap_flag",
    "household_vehicle_location_transition_conflict_flag",
    "household_vehicle_timeline_conflict_flag",
    "availability_conflict_flag",
    "simultaneous_home_departure_flag",
    "same_vehicle_simultaneous_departure_flag",
    "complete_fleet_baseline_flag",
    "topcoded_multicar_household_flag",
    "fleet_inventory_mismatch_flag",
    "household_car_count_conflict_flag",
    "nonstandard_A_ID_set_flag",
    "invalid_start_time_flag",
    "invalid_arrival_time_flag",
    "negative_duration_flag",
    "VALID_STRICT_CHOICE_OCCASION",
]

CANDIDATE_REASON_FLAGS = [
    ("INCOMPLETE_OR_MISMATCHED_FLEET", "EXCL_INVALID_TWO_CAR_FLEET"),
    ("UNIDENTIFIABLE_DRIVER", "EXCL_UNIDENTIFIABLE_DRIVER"),
    ("INVALID_CHOSEN_VEHICLE", "EXCL_INVALID_CHOSEN_VEHICLE"),
    ("DUPLICATE_OR_INVALID_IDENTIFIER", "EXCL_INVALID_OCCASION_IDENTIFIER"),
    ("INVALID_CURRENT_TRIP_TIMING", "EXCL_INVALID_CURRENT_TRIP_TIMING"),
    ("INVALID_SNAPSHOT", "EXCL_INVALID_ALLOCATION_SNAPSHOT"),
    ("UNKNOWN_SNAPSHOT", "EXCL_UNKNOWN_ALLOCATION_SNAPSHOT"),
    ("INVALID_ALLOCATION_STATE_CLASS", "EXCL_INVALID_ALLOCATION_STATE_CLASS"),
    (
        "CHOSEN_VEHICLE_STATE_INCONSISTENCY",
        "EXCL_CHOSEN_VEHICLE_STATE_INCONSISTENCY",
    ),
    ("HOUSEHOLD_TIME_UNRESOLVED", "EXCL_HOUSEHOLD_TIME_UNRESOLVED"),
    ("HOUSEHOLD_TIMELINE_CONFLICT", "EXCL_HOUSEHOLD_TIMELINE_CONFLICT"),
    ("SAME_VEHICLE_OVERLAP_CONFLICT", "EXCL_SAME_VEHICLE_OVERLAP_CONFLICT"),
    ("LOCATION_TRANSITION_CONFLICT", "EXCL_LOCATION_TRANSITION_CONFLICT"),
    ("ZERO_DURATION_HISTORY", "EXCL_ZERO_DURATION_HISTORY"),
    (
        "SIMULTANEOUS_SNAPSHOT_CONFLICT",
        "EXCL_SIMULTANEOUS_SNAPSHOT_CONFLICT",
    ),
]


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def require_columns(
    columns: Iterable[str], required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def flag_equals_one(series: pd.Series) -> pd.Series:
    return numeric(series).fillna(0).eq(1)


def finite_numeric(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))


def canonical_integer_id(series: pd.Series) -> pd.Series:
    values = numeric(series)
    whole = values.notna() & values.map(
        lambda value: math.isfinite(float(value)) and float(value).is_integer()
    )
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[whole] = values.loc[whole].astype("Int64").astype("string")
    return result


def parse_id_set(value: object) -> tuple[tuple[str, ...], int]:
    text = "" if pd.isna(value) else str(value).strip()
    if text == "":
        return tuple(), 1
    parsed: list[int] = []
    for part in text.split("|"):
        try:
            number = float(part)
        except ValueError:
            return tuple(), 1
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            return tuple(), 1
        parsed.append(int(number))
    parse_error = int(len(parsed) != len(set(parsed)) or parsed != sorted(parsed))
    return tuple(str(value) for value in parsed), parse_error


def make_driver_key(frame: pd.DataFrame) -> pd.Series:
    hp_id = frame["HP_ID"].astype("string").str.strip()
    p_id = frame["P_ID"].astype("string").str.strip()
    h_id = frame["H_ID"].astype("string").str.strip()
    key = hp_id.copy()
    fallback = key.eq("") & h_id.ne("") & p_id.ne("")
    key.loc[fallback] = h_id.loc[fallback] + "__P_ID_" + p_id.loc[fallback]
    key.loc[key.eq("")] = pd.NA
    return key


def validate_binary_columns(frame: pd.DataFrame) -> None:
    values = frame[BINARY_MASTER_COLUMNS].apply(numeric)
    invalid = ~values.isin([0, 1]).all(axis=1)
    if invalid.any():
        print("\nNON-BINARY OR MISSING MASTER FLAGS")
        print(frame.loc[invalid, ["CHOICE_ID", *BINARY_MASTER_COLUMNS]].head(10).to_string(index=False))
        raise ValueError("Required Phase-3 classification flags must be populated and binary.")


def build_car_roster(cars: pd.DataFrame) -> pd.DataFrame:
    require_columns(cars.columns, ["H_ID", "A_ID"], "vehicle inventory")
    source = cars[["H_ID", "A_ID"]].copy()
    source["A_ID_CANONICAL"] = canonical_integer_id(source["A_ID"])
    source = source.loc[
        source["H_ID"].ne("") & source["A_ID_CANONICAL"].notna()
    ].drop_duplicates(["H_ID", "A_ID_CANONICAL"])
    source["A_ID_SORT"] = numeric(source["A_ID_CANONICAL"])
    source = source.sort_values(["H_ID", "A_ID_SORT"], kind="mergesort")
    roster = (
        source.groupby("H_ID", as_index=False, sort=False)
        .agg(
            RECORDED_FLEET_FROM_CARS=(
                "A_ID_CANONICAL",
                lambda values: "|".join(values.astype(str)),
            ),
            N_RECORDED_CARS_FROM_CARS=("A_ID_CANONICAL", "nunique"),
        )
        .sort_values("H_ID", kind="mergesort")
        .reset_index(drop=True)
    )
    return roster


def add_simultaneous_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["COMMON_PREDEPARTURE_SNAPSHOT_FLAG"] = 1
    out["RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG"] = 0
    simultaneous = flag_equals_one(out["simultaneous_home_departure_flag"])
    if not simultaneous.any():
        return out

    simultaneous_rows = out.loc[simultaneous].copy()
    groups = simultaneous_rows.groupby(["H_ID", "START_MIN"], sort=False)
    source_count = groups["SOURCE_ROW_ID"].transform("nunique")
    driver_count = groups["DRIVER_KEY"].transform("nunique")
    chosen_count = groups["CHOSEN_A_ID_CANONICAL"].transform("nunique")
    common_snapshot = (
        source_count.ge(2)
        & driver_count.ge(2)
        & groups["AVAILABLE_CAR_SET"].transform("nunique").eq(1)
        & groups["N_AVAILABLE_CARS"].transform("nunique").eq(1)
    )
    same_vehicle = source_count.ge(2) & driver_count.ge(2) & chosen_count.lt(source_count)
    out.loc[simultaneous, "COMMON_PREDEPARTURE_SNAPSHOT_FLAG"] = common_snapshot.astype(int).to_numpy()
    out.loc[simultaneous, "RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG"] = same_vehicle.astype(int).to_numpy()

    comparable = simultaneous & out["DRIVER_KEY"].notna() & out["CHOSEN_A_ID_CANONICAL"].notna()
    upstream = flag_equals_one(out["same_vehicle_simultaneous_departure_flag"])
    recomputed = out["RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG"].eq(1)
    mismatch = comparable & upstream.ne(recomputed)
    if mismatch.any():
        print("\nSAME-VEHICLE SIMULTANEOUS FLAG MISMATCH")
        print(
            out.loc[
                mismatch,
                [
                    "CHOICE_ID",
                    "H_ID",
                    "START_MIN",
                    "DRIVER_KEY",
                    "CHOSEN_A_ID",
                    "same_vehicle_simultaneous_departure_flag",
                    "RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG",
                ],
            ].head(10).to_string(index=False)
        )
        raise ValueError("Recomputed same-vehicle simultaneous conflicts disagree with Phase 2.")
    return out


def add_candidate_flags(master: pd.DataFrame, car_roster: pd.DataFrame) -> pd.DataFrame:
    require_columns(master.columns, REQUIRED_MASTER_COLUMNS, "classified master")
    validate_binary_columns(master)
    out = master.copy()

    home_origin = numeric(out["HOME_ORIGIN"])
    if not home_origin.eq(1).all():
        raise ValueError("The classified master must retain only HOME_ORIGIN == 1 occasions.")

    out = out.merge(car_roster, on="H_ID", how="left", validate="many_to_one")
    out["RECORDED_FLEET_FROM_CARS"] = out["RECORDED_FLEET_FROM_CARS"].fillna("")
    out["N_RECORDED_CARS_FROM_CARS"] = numeric(
        out["N_RECORDED_CARS_FROM_CARS"]
    ).fillna(0).astype(int)
    out["DRIVER_KEY"] = make_driver_key(out)
    out["CHOSEN_A_ID_CANONICAL"] = canonical_integer_id(out["CHOSEN_A_ID"])

    fleet_parsed = out["FLEET_CAR_SET"].map(parse_id_set)
    out["_FLEET_IDS"] = fleet_parsed.map(lambda item: item[0])
    out["fleet_set_parse_error_flag"] = fleet_parsed.map(lambda item: item[1]).astype(int)
    available_parsed = out["AVAILABLE_CAR_SET"].map(parse_id_set)
    out["_AVAILABLE_IDS"] = available_parsed.map(lambda item: item[0])
    out["allocation_available_set_parse_error_flag"] = available_parsed.map(
        lambda item: item[1]
    ).astype(int)
    out["ALLOCATION_PARSED_AVAILABLE_CAR_SET"] = out["_AVAILABLE_IDS"].map(
        lambda values: "|".join(values)
    )
    out["ALLOCATION_STATE_CLASS"] = (
        out["ALLOCATION_PARSED_AVAILABLE_CAR_SET"]
        .map(STATE_CLASS_BY_AVAILABLE_SET)
        .fillna("OTHER_INVALID")
    )

    out = add_simultaneous_diagnostics(out)

    complete_fleet = flag_equals_one(out["complete_fleet_baseline_flag"])
    topcoded = flag_equals_one(out["topcoded_multicar_household_flag"])
    recorded_count = numeric(out["N_RECORDED_CARS"])
    upstream_fleet_exact = out["_FLEET_IDS"].map(lambda values: values == ("1", "2"))
    car_roster_exact = out["RECORDED_FLEET_FROM_CARS"].eq("1|2") & out[
        "N_RECORDED_CARS_FROM_CARS"
    ].eq(2)
    no_fleet_conflict = (
        ~flag_equals_one(out["fleet_inventory_mismatch_flag"])
        & ~flag_equals_one(out["household_car_count_conflict_flag"])
        & ~flag_equals_one(out["nonstandard_A_ID_set_flag"])
        & out["fleet_set_parse_error_flag"].eq(0)
    )

    out["EXCL_FLEET_BASELINE_NOT_COMPLETE"] = (~complete_fleet).astype(int)
    out["EXCL_TOPCODED_FLEET"] = topcoded.astype(int)
    out["EXCL_RECORDED_FLEET_NOT_EXACTLY_TWO"] = (
        recorded_count.ne(2) | ~upstream_fleet_exact | ~car_roster_exact
    ).astype(int)
    out["EXCL_FLEET_MISMATCH"] = (~no_fleet_conflict).astype(int)
    out["EXCL_INVALID_TWO_CAR_FLEET"] = (
        ~complete_fleet
        | topcoded
        | recorded_count.ne(2)
        | ~upstream_fleet_exact
        | ~car_roster_exact
        | ~no_fleet_conflict
    ).astype(int)

    out["EXCL_UNIDENTIFIABLE_DRIVER"] = out["DRIVER_KEY"].isna().astype(int)
    chosen_in_roster = pd.Series(
        [
            pd.notna(chosen) and str(chosen) in fleet.split("|")
            for chosen, fleet in zip(
                out["CHOSEN_A_ID_CANONICAL"], out["RECORDED_FLEET_FROM_CARS"]
            )
        ],
        index=out.index,
    )
    out["CHOSEN_IN_RECORDED_FLEET_FLAG"] = chosen_in_roster.astype(int)
    out["EXCL_INVALID_CHOSEN_VEHICLE"] = (
        out["CHOSEN_A_ID_CANONICAL"].isna()
        | ~out["CHOSEN_A_ID_CANONICAL"].isin({"1", "2"})
        | ~chosen_in_roster
        | flag_equals_one(out["unmatched_reported_vehicle_flag"])
    ).astype(int)

    duplicate_choice = out["CHOICE_ID"].eq("") | out["CHOICE_ID"].duplicated(
        keep=False
    )
    duplicate_trip = out["TRIP_ID"].eq("") | out["TRIP_ID"].duplicated(keep=False)
    out["EXCL_INVALID_OCCASION_IDENTIFIER"] = (
        duplicate_choice
        | duplicate_trip
        | flag_equals_one(out["duplicate_trip_identifier_flag"])
    ).astype(int)

    start = numeric(out["START_MIN"])
    arrival = numeric(out["ARRIVAL_MIN"])
    valid_current_time = (
        finite_numeric(out["START_MIN"])
        & finite_numeric(out["ARRIVAL_MIN"])
        & arrival.ge(start)
        & flag_equals_one(out["complete_vehicle_time_flag"])
        & ~flag_equals_one(out["invalid_start_time_flag"])
        & ~flag_equals_one(out["invalid_arrival_time_flag"])
        & ~flag_equals_one(out["negative_duration_flag"])
    )
    out["EXCL_INVALID_CURRENT_TRIP_TIMING"] = (~valid_current_time).astype(int)

    available_count_matches = numeric(out["N_AVAILABLE_CARS"]).eq(
        out["_AVAILABLE_IDS"].map(len)
    )
    out["EXCL_INVALID_ALLOCATION_SNAPSHOT"] = (
        ~flag_equals_one(out["snapshot_valid_flag"])
        | flag_equals_one(out["invalid_snapshot_flag"])
        | out["allocation_available_set_parse_error_flag"].eq(1)
        | ~available_count_matches
    ).astype(int)
    out["EXCL_UNKNOWN_ALLOCATION_SNAPSHOT"] = flag_equals_one(
        out["unknown_vehicle_state_at_snapshot_flag"]
    ).astype(int)
    out["EXCL_INVALID_ALLOCATION_STATE_CLASS"] = (
        ~out["ALLOCATION_STATE_CLASS"].isin(VALID_STATE_CLASSES)
    ).astype(int)

    chosen_home = pd.Series(
        [
            pd.notna(chosen) and str(chosen) in available
            for chosen, available in zip(
                out["CHOSEN_A_ID_CANONICAL"], out["_AVAILABLE_IDS"]
            )
        ],
        index=out.index,
    )
    out["CHOSEN_VEHICLE_HOME_FLAG"] = chosen_home.astype(int)
    out["EXCL_CHOSEN_VEHICLE_STATE_INCONSISTENCY"] = (
        ~chosen_home | flag_equals_one(out["availability_conflict_flag"])
    ).astype(int)

    out["EXCL_HOUSEHOLD_TIME_UNRESOLVED"] = flag_equals_one(
        out["household_vehicle_time_unresolved_flag"]
    ).astype(int)
    out["EXCL_HOUSEHOLD_TIMELINE_CONFLICT"] = flag_equals_one(
        out["household_vehicle_timeline_conflict_flag"]
    ).astype(int)
    out["EXCL_SAME_VEHICLE_SIMULTANEOUS_CONFLICT"] = (
        flag_equals_one(out["same_vehicle_simultaneous_departure_flag"])
        | out["RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG"].eq(1)
    ).astype(int)
    out["EXCL_SAME_VEHICLE_OVERLAP_CONFLICT"] = (
        flag_equals_one(out["household_vehicle_trip_overlap_flag"])
        | out["EXCL_SAME_VEHICLE_SIMULTANEOUS_CONFLICT"].eq(1)
    ).astype(int)
    out["EXCL_LOCATION_TRANSITION_CONFLICT"] = flag_equals_one(
        out["household_vehicle_location_transition_conflict_flag"]
    ).astype(int)
    out["EXCL_ZERO_DURATION_HISTORY"] = flag_equals_one(
        out["household_zero_duration_driver_trip_flag"]
    ).astype(int)
    out["EXCL_SIMULTANEOUS_SNAPSHOT_CONFLICT"] = (
        flag_equals_one(out["simultaneous_home_departure_flag"])
        & out["COMMON_PREDEPARTURE_SNAPSHOT_FLAG"].ne(1)
    ).astype(int)
    out["EXCL_UNRESOLVED_HOUSEHOLD_HISTORY"] = (
        out[
            [
                "EXCL_HOUSEHOLD_TIME_UNRESOLVED",
                "EXCL_HOUSEHOLD_TIMELINE_CONFLICT",
                "EXCL_SAME_VEHICLE_OVERLAP_CONFLICT",
                "EXCL_LOCATION_TRANSITION_CONFLICT",
                "EXCL_ZERO_DURATION_HISTORY",
            ]
        ]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    reason_flags = [flag for _, flag in CANDIDATE_REASON_FLAGS]
    out["VALID_ALLOCATION_ANALYSIS_CANDIDATE"] = out[reason_flags].sum(axis=1).eq(0).astype(int)
    all_reasons: list[str] = []
    primary_reasons: list[str] = []
    reason_names = [name for name, _ in CANDIDATE_REASON_FLAGS]
    for row in out[reason_flags].itertuples(index=False, name=None):
        active = [reason_names[index] for index, value in enumerate(row) if int(value) == 1]
        all_reasons.append("|".join(active) if active else "NONE")
        primary_reasons.append(active[0] if active else "NONE")
    out["ALL_ALLOCATION_EXCLUSION_REASONS"] = all_reasons
    out["PRIMARY_ALLOCATION_EXCLUSION_REASON"] = primary_reasons
    return out


def add_existing_sample_membership(
    classified: pd.DataFrame, sensitivity: pd.DataFrame
) -> pd.DataFrame:
    require_columns(sensitivity.columns, ["CHOICE_ID"], "sensitivity")
    if sensitivity["CHOICE_ID"].eq("").any() or sensitivity["CHOICE_ID"].duplicated().any():
        raise ValueError("Sensitivity CHOICE_ID must be non-missing and unique.")
    out = classified.copy()
    out["IN_STRICT_BASELINE"] = flag_equals_one(
        out["VALID_STRICT_CHOICE_OCCASION"]
    ).astype(int)
    sensitivity_ids = set(sensitivity["CHOICE_ID"])
    out["IN_DECISION_TIMING_SENSITIVITY"] = out["CHOICE_ID"].isin(
        sensitivity_ids
    ).astype(int)
    strict_not_sensitivity = out["IN_STRICT_BASELINE"].eq(1) & out[
        "IN_DECISION_TIMING_SENSITIVITY"
    ].eq(0)
    if strict_not_sensitivity.any():
        raise AssertionError("Accepted strict occasions must be contained in sensitivity membership.")
    missing_sensitivity_ids = sensitivity_ids - set(out["CHOICE_ID"])
    if missing_sensitivity_ids:
        raise AssertionError(
            "Sensitivity occasions missing from the classified master: "
            f"{sorted(missing_sensitivity_ids)[:10]}"
        )
    return out


def select_and_order_candidate(classified: pd.DataFrame) -> pd.DataFrame:
    candidate = classified.loc[
        classified["VALID_ALLOCATION_ANALYSIS_CANDIDATE"].eq(1)
    ].copy()
    candidate["_START_SORT"] = numeric(candidate["START_MIN"])
    candidate = candidate.sort_values(
        ["H_ID", "_START_SORT", "CHOICE_ID"], kind="mergesort"
    ).drop(columns="_START_SORT")

    front = [
        "CHOICE_ID",
        "TRIP_ID",
        "SOURCE_ROW_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "DRIVER_KEY",
        "W_ID",
        "START_MIN",
        "ARRIVAL_MIN",
        "CHOSEN_A_ID",
        "CHOSEN_A_ID_CANONICAL",
        "AVAILABLE_CAR_SET",
        "N_AVAILABLE_CARS",
        "ALLOCATION_STATE_CLASS",
        "simultaneous_home_departure_flag",
        "same_vehicle_simultaneous_departure_flag",
        "RECOMPUTED_SAME_VEHICLE_SIMULTANEOUS_FLAG",
        "COMMON_PREDEPARTURE_SNAPSHOT_FLAG",
        "IN_STRICT_BASELINE",
        "IN_DECISION_TIMING_SENSITIVITY",
        "VALID_ALLOCATION_ANALYSIS_CANDIDATE",
        "ALL_ALLOCATION_EXCLUSION_REASONS",
        "PRIMARY_ALLOCATION_EXCLUSION_REASON",
    ]
    remaining = [
        column
        for column in candidate.columns
        if column not in front and not column.startswith("_")
    ]
    return candidate[front + remaining].reset_index(drop=True)


def build_households(candidate: pd.DataFrame) -> pd.DataFrame:
    driver_counts = (
        candidate.groupby(["H_ID", "DRIVER_KEY"], sort=False)
        .size()
        .rename("N_DRIVER_CANDIDATE_OCCASIONS")
        .reset_index()
    )
    repeated = (
        driver_counts.groupby("H_ID", as_index=False)
        .agg(
            MAX_OCCASIONS_PER_DRIVER=("N_DRIVER_CANDIDATE_OCCASIONS", "max"),
            N_DRIVERS_WITH_2PLUS_OCCASIONS=(
                "N_DRIVER_CANDIDATE_OCCASIONS",
                lambda values: int(values.ge(2).sum()),
            ),
        )
    )

    households = (
        candidate.groupby("H_ID", as_index=False, sort=False)
        .agg(
            N_CANDIDATE_OCCASIONS=("CHOICE_ID", "size"),
            N_ACTIVE_DRIVERS=("DRIVER_KEY", "nunique"),
            N_USED_VEHICLES=("CHOSEN_A_ID_CANONICAL", "nunique"),
            N_SIMULTANEOUS_OCCASIONS=("simultaneous_home_departure_flag", lambda values: int(flag_equals_one(values).sum())),
            N_STRICT_OCCASIONS=("IN_STRICT_BASELINE", "sum"),
            N_SENSITIVITY_OCCASIONS=("IN_DECISION_TIMING_SENSITIVITY", "sum"),
        )
        .merge(repeated, on="H_ID", how="left", validate="one_to_one")
    )
    households["MEAN_OCCASIONS_PER_ACTIVE_DRIVER"] = (
        households["N_CANDIDATE_OCCASIONS"] / households["N_ACTIVE_DRIVERS"]
    )
    households["HAS_SIMULTANEOUS_DEPARTURE"] = households[
        "N_SIMULTANEOUS_OCCASIONS"
    ].gt(0).astype(int)
    households["HAS_REPEATED_DRIVER_OCCASIONS"] = households[
        "N_DRIVERS_WITH_2PLUS_OCCASIONS"
    ].ge(1).astype(int)
    households["AT_LEAST_ONE_REPEATED_DRIVER"] = households[
        "N_DRIVERS_WITH_2PLUS_OCCASIONS"
    ].ge(1).astype(int)
    households["AT_LEAST_TWO_REPEATED_DRIVERS"] = households[
        "N_DRIVERS_WITH_2PLUS_OCCASIONS"
    ].ge(2).astype(int)
    households["SINGLE_ACTIVE_DRIVER_HH"] = households["N_ACTIVE_DRIVERS"].eq(1).astype(int)
    households["MULTI_ACTIVE_DRIVER_HH"] = households["N_ACTIVE_DRIVERS"].ge(2).astype(int)
    households["HAS_STRICT_OCCASION"] = households["N_STRICT_OCCASIONS"].gt(0).astype(int)
    households["HAS_SENSITIVITY_OCCASION"] = households[
        "N_SENSITIVITY_OCCASIONS"
    ].gt(0).astype(int)

    edges = candidate[
        ["H_ID", "DRIVER_KEY", "CHOSEN_A_ID_CANONICAL"]
    ].drop_duplicates()
    driver_degree = (
        edges.groupby(["H_ID", "DRIVER_KEY"])["CHOSEN_A_ID_CANONICAL"]
        .nunique()
        .rename("DRIVER_DEGREE")
        .reset_index()
    )
    vehicle_degree = (
        edges.groupby(["H_ID", "CHOSEN_A_ID_CANONICAL"])["DRIVER_KEY"]
        .nunique()
        .rename("VEHICLE_DEGREE")
        .reset_index()
    )
    degrees = (
        driver_degree.groupby("H_ID", as_index=False)["DRIVER_DEGREE"]
        .max()
        .rename(columns={"DRIVER_DEGREE": "MAX_DRIVER_DEGREE"})
        .merge(
            vehicle_degree.groupby("H_ID", as_index=False)["VEHICLE_DEGREE"]
            .max()
            .rename(columns={"VEHICLE_DEGREE": "MAX_VEHICLE_DEGREE"}),
            on="H_ID",
            how="outer",
            validate="one_to_one",
        )
    )
    households = households.merge(degrees, on="H_ID", how="left", validate="one_to_one")

    conditions = [
        households["N_ACTIVE_DRIVERS"].eq(1) & households["MAX_DRIVER_DEGREE"].eq(1),
        households["N_ACTIVE_DRIVERS"].eq(1) & households["MAX_DRIVER_DEGREE"].ge(2),
        households["N_ACTIVE_DRIVERS"].ge(2)
        & households["MAX_DRIVER_DEGREE"].eq(1)
        & households["MAX_VEHICLE_DEGREE"].eq(1),
        households["N_ACTIVE_DRIVERS"].ge(2)
        & (
            households["MAX_DRIVER_DEGREE"].ge(2)
            | households["MAX_VEHICLE_DEGREE"].ge(2)
        ),
    ]
    membership_count = sum(condition.astype(int) for condition in conditions)
    if not membership_count.eq(1).all():
        raise AssertionError("Every candidate household must belong to exactly one candidate Cell.")
    households["CANDIDATE_CELL_ID"] = pd.NA
    for cell_id, condition in enumerate(conditions, start=1):
        households.loc[condition, "CANDIDATE_CELL_ID"] = cell_id
    households["CANDIDATE_CELL_ID"] = households["CANDIDATE_CELL_ID"].astype(int)
    households["CANDIDATE_CELL_LABEL"] = households["CANDIDATE_CELL_ID"].map(CELL_LABELS)

    columns = [
        "H_ID",
        "N_CANDIDATE_OCCASIONS",
        "N_ACTIVE_DRIVERS",
        "N_USED_VEHICLES",
        "MEAN_OCCASIONS_PER_ACTIVE_DRIVER",
        "MAX_OCCASIONS_PER_DRIVER",
        "N_DRIVERS_WITH_2PLUS_OCCASIONS",
        "HAS_REPEATED_DRIVER_OCCASIONS",
        "AT_LEAST_ONE_REPEATED_DRIVER",
        "AT_LEAST_TWO_REPEATED_DRIVERS",
        "HAS_SIMULTANEOUS_DEPARTURE",
        "N_SIMULTANEOUS_OCCASIONS",
        "SINGLE_ACTIVE_DRIVER_HH",
        "MULTI_ACTIVE_DRIVER_HH",
        "MAX_DRIVER_DEGREE",
        "MAX_VEHICLE_DEGREE",
        "CANDIDATE_CELL_ID",
        "CANDIDATE_CELL_LABEL",
        "N_STRICT_OCCASIONS",
        "N_SENSITIVITY_OCCASIONS",
        "HAS_STRICT_OCCASION",
        "HAS_SENSITIVITY_OCCASION",
    ]
    return households[columns].sort_values("H_ID", kind="mergesort").reset_index(drop=True)


def qa_row(
    section: str,
    metric: str,
    count: int,
    denominator: int,
    denominator_description: str,
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "count": int(count),
        "share": float(count / denominator) if denominator else 0.0,
        "denominator_description": denominator_description,
    }


def build_qa(
    classified: pd.DataFrame, candidate: pd.DataFrame, households: pd.DataFrame
) -> pd.DataFrame:
    input_rows = len(classified)
    input_households = classified["H_ID"].nunique()
    candidate_rows = len(candidate)
    candidate_households = len(households)
    valid_fleet = classified["EXCL_INVALID_TWO_CAR_FLEET"].eq(0)
    matched_fleet = valid_fleet & classified["CHOSEN_IN_RECORDED_FLEET_FLAG"].eq(1)
    invalid_unknown_snapshot = (
        classified["EXCL_INVALID_ALLOCATION_SNAPSHOT"].eq(1)
        | classified["EXCL_UNKNOWN_ALLOCATION_SNAPSHOT"].eq(1)
    )
    simultaneous = flag_equals_one(classified["simultaneous_home_departure_flag"])
    valid_candidate = classified["VALID_ALLOCATION_ANALYSIS_CANDIDATE"].eq(1)
    strict = classified["IN_STRICT_BASELINE"].eq(1)
    sensitivity = classified["IN_DECISION_TIMING_SENSITIVITY"].eq(1)

    rows = [
        qa_row("A. INPUT / UNIVERSE", "Phase-2 home-origin occasions in classified master", input_rows, input_rows, "all classified home-origin occasions"),
        qa_row("A. INPUT / UNIVERSE", "complete two-car household occasions", int(valid_fleet.sum()), input_rows, "all classified home-origin occasions"),
        qa_row("A. INPUT / UNIVERSE", "unique households", input_households, input_households, "all unique households in classified master"),
        qa_row("B. CANDIDATE FILTERING", "matched valid fleet", int(matched_fleet.sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "valid current timing", int(classified["EXCL_INVALID_CURRENT_TRIP_TIMING"].eq(0).sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "invalid/unknown snapshot", int(invalid_unknown_snapshot.sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "chosen-vehicle state inconsistency", int(classified["EXCL_CHOSEN_VEHICLE_STATE_INCONSISTENCY"].sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "unresolved household history", int(classified["EXCL_UNRESOLVED_HOUSEHOLD_HISTORY"].sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "same-vehicle overlap conflict", int(classified["EXCL_SAME_VEHICLE_OVERLAP_CONFLICT"].sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "simultaneous departure rows", int(simultaneous.sum()), input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "simultaneous departure rows retained", int((simultaneous & valid_candidate).sum()), int(simultaneous.sum()), "all simultaneous home-departure occasions"),
        qa_row("B. CANDIDATE FILTERING", "simultaneous departure rows excluded for physical inconsistency", int((simultaneous & classified["EXCL_SAME_VEHICLE_SIMULTANEOUS_CONFLICT"].eq(1)).sum()), int(simultaneous.sum()), "all simultaneous home-departure occasions"),
        qa_row("B. CANDIDATE FILTERING", "final valid allocation candidate occasions", candidate_rows, input_rows, "all classified home-origin occasions"),
        qa_row("B. CANDIDATE FILTERING", "final candidate households", candidate_households, input_households, "all unique households in classified master"),
        qa_row("C. RELATION TO EXISTING SAMPLES", "candidate occasions also in strict baseline", int((valid_candidate & strict).sum()), candidate_rows, "final candidate occasions"),
        qa_row("C. RELATION TO EXISTING SAMPLES", "candidate occasions also in sensitivity sample", int((valid_candidate & sensitivity).sum()), candidate_rows, "final candidate occasions"),
        qa_row("C. RELATION TO EXISTING SAMPLES", "candidate-only additional occasions", int((valid_candidate & ~sensitivity).sum()), candidate_rows, "final candidate occasions"),
        qa_row("C. RELATION TO EXISTING SAMPLES", "strict occasions absent from candidate", int((strict & ~valid_candidate).sum()), int(strict.sum()), "observed strict baseline occasions"),
        qa_row("C. RELATION TO EXISTING SAMPLES", "sensitivity occasions absent from candidate", int((sensitivity & ~valid_candidate).sum()), int(sensitivity.sum()), "observed sensitivity occasions"),
        qa_row("D. DRIVER STRUCTURE", "single-active-driver households", int(households["SINGLE_ACTIVE_DRIVER_HH"].sum()), candidate_households, "final candidate households"),
        qa_row("D. DRIVER STRUCTURE", "multi-active-driver households", int(households["MULTI_ACTIVE_DRIVER_HH"].sum()), candidate_households, "final candidate households"),
        qa_row("D. DRIVER STRUCTURE", "households with at least one repeated driver", int(households["AT_LEAST_ONE_REPEATED_DRIVER"].sum()), candidate_households, "final candidate households"),
        qa_row("D. DRIVER STRUCTURE", "households with at least two repeated drivers", int(households["AT_LEAST_TWO_REPEATED_DRIVERS"].sum()), candidate_households, "final candidate households"),
    ]

    for cell_id in range(1, 5):
        rows.append(
            qa_row(
                "E. CANDIDATE CELL DISTRIBUTION",
                f"Cell {cell_id} households",
                int(households["CANDIDATE_CELL_ID"].eq(cell_id).sum()),
                candidate_households,
                "final candidate households",
            )
        )
    for state_class in ["HOME|HOME", "HOME|AWAY", "AWAY|HOME"]:
        rows.append(
            qa_row(
                "F. STATE CLASS",
                state_class,
                int(candidate["ALLOCATION_STATE_CLASS"].eq(state_class).sum()),
                candidate_rows,
                "final candidate occasions",
            )
        )

    assertion_metrics = [
        ("duplicate candidate CHOICE_ID", int(candidate["CHOICE_ID"].duplicated(keep=False).sum())),
        ("invalid chosen A_ID", int(candidate["EXCL_INVALID_CHOSEN_VEHICLE"].sum())),
        ("same-vehicle simultaneous conflict", int(candidate["EXCL_SAME_VEHICLE_SIMULTANEOUS_CONFLICT"].sum())),
        ("household timeline conflict", int(candidate["EXCL_HOUSEHOLD_TIMELINE_CONFLICT"].sum())),
        ("location-transition conflict", int(candidate["EXCL_LOCATION_TRANSITION_CONFLICT"].sum())),
        ("unresolved time-history cases", int(candidate["EXCL_HOUSEHOLD_TIME_UNRESOLVED"].sum())),
    ]
    for metric, count in assertion_metrics:
        rows.append(
            qa_row(
                "G. ASSERTION / CONFLICT QA",
                metric,
                count,
                candidate_rows,
                "final candidate occasions",
            )
        )
    return pd.DataFrame(rows)


def validate_outputs(
    candidate: pd.DataFrame, households: pd.DataFrame, car_roster: pd.DataFrame
) -> None:
    assert not candidate["CHOICE_ID"].eq("").any(), "Candidate CHOICE_ID cannot be missing."
    assert not candidate["CHOICE_ID"].duplicated().any(), "Candidate CHOICE_ID must be unique."
    roster = car_roster.set_index("H_ID")
    candidate_roster = roster.loc[candidate["H_ID"].drop_duplicates()]
    assert candidate_roster["N_RECORDED_CARS_FROM_CARS"].eq(2).all(), (
        "Every candidate household must have exactly two recorded vehicle IDs."
    )
    assert candidate_roster["RECORDED_FLEET_FROM_CARS"].eq("1|2").all(), (
        "Every candidate household must have the standard recorded two-car fleet 1|2."
    )
    assert candidate["DRIVER_KEY"].notna().all(), "Every candidate requires DRIVER_KEY."
    assert candidate["CHOSEN_IN_RECORDED_FLEET_FLAG"].eq(1).all(), (
        "Every chosen vehicle must belong to its household fleet."
    )
    assert candidate["ALLOCATION_STATE_CLASS"].isin(VALID_STATE_CLASSES).all(), (
        "Candidate allocation states must be HOME|HOME, HOME|AWAY, or AWAY|HOME."
    )
    assert not flag_equals_one(candidate["unknown_vehicle_state_at_snapshot_flag"]).any(), (
        "Candidate rows cannot contain UNKNOWN focal vehicle state."
    )
    start = numeric(candidate["START_MIN"])
    arrival = numeric(candidate["ARRIVAL_MIN"])
    assert finite_numeric(candidate["START_MIN"]).all(), "Candidate START_MIN must be finite."
    assert finite_numeric(candidate["ARRIVAL_MIN"]).all(), "Candidate ARRIVAL_MIN must be finite."
    assert arrival.ge(start).all(), "Candidate ARRIVAL_MIN must be >= START_MIN."
    assert candidate["EXCL_SAME_VEHICLE_OVERLAP_CONFLICT"].eq(0).all(), (
        "Candidate rows cannot contain impossible same-vehicle overlaps."
    )
    assert set(households["CANDIDATE_CELL_ID"].unique()) == {1, 2, 3, 4}, (
        "Candidate Cell IDs must be exactly 1, 2, 3, and 4."
    )
    assert households["CANDIDATE_CELL_ID"].notna().all(), (
        "Candidate Cell membership must be mutually exhaustive."
    )
    assert households.loc[
        households["CANDIDATE_CELL_ID"].isin([1, 2]), "N_ACTIVE_DRIVERS"
    ].eq(1).all(), "Cell 1/2 households must have one active driver."
    assert households.loc[
        households["CANDIDATE_CELL_ID"].isin([3, 4]), "N_ACTIVE_DRIVERS"
    ].ge(2).all(), "Cell 3/4 households must have multiple active drivers."
    assert int(households["N_CANDIDATE_OCCASIONS"].sum()) == len(candidate), (
        "Household aggregation must reproduce the candidate occasion total."
    )
    assert candidate["VALID_ALLOCATION_ANALYSIS_CANDIDATE"].eq(1).all(), (
        "Occasion output may contain only valid allocation-analysis candidates."
    )
    assert candidate["ALL_ALLOCATION_EXCLUSION_REASONS"].eq("NONE").all(), (
        "Valid candidate rows cannot retain an exclusion reason."
    )


def assert_no_outputs_exist(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output file(s):\n{joined}")


def print_summary(candidate: pd.DataFrame, households: pd.DataFrame) -> None:
    print("\nALLOCATION-ANALYSIS CANDIDATE SUMMARY")
    print(f"final candidate occasions: {len(candidate):,}")
    print(f"final candidate households: {len(households):,}")
    print(f"single-active-driver households: {int(households['SINGLE_ACTIVE_DRIVER_HH'].sum()):,}")
    print(f"multi-active-driver households: {int(households['MULTI_ACTIVE_DRIVER_HH'].sum()):,}")
    print("\nCANDIDATE CELL DISTRIBUTION")
    for cell_id in range(1, 5):
        count = int(households["CANDIDATE_CELL_ID"].eq(cell_id).sum())
        share = count / len(households) if len(households) else 0.0
        print(f"Cell {cell_id}: {count:,} households ({share:.2%})")
    print("\nREPEATED-ALLOCATION SUPPORT")
    print(
        "households with >=1 repeated driver: "
        f"{int(households['AT_LEAST_ONE_REPEATED_DRIVER'].sum()):,}"
    )
    print(
        "households with >=2 repeated drivers: "
        f"{int(households['AT_LEAST_TWO_REPEATED_DRIVERS'].sum()):,}"
    )
    simultaneous_retained = int(
        flag_equals_one(candidate["simultaneous_home_departure_flag"]).sum()
    )
    print(f"valid retained simultaneous-departure occasions: {simultaneous_retained:,}")
    strict_overlap = int(candidate["IN_STRICT_BASELINE"].sum())
    sensitivity_overlap = int(candidate["IN_DECISION_TIMING_SENSITIVITY"].sum())
    print("\nRELATION TO EXISTING SAMPLES")
    print(
        f"candidate overlap with strict: {strict_overlap:,} "
        f"(REFERENCE only: {REFERENCE_STRICT_OCCASIONS:,})"
    )
    print(
        f"candidate overlap with sensitivity: {sensitivity_overlap:,} "
        f"(REFERENCE only: {REFERENCE_SENSITIVITY_OCCASIONS:,})"
    )
    print(
        "candidate-only additional occasions: "
        f"{int(candidate['IN_DECISION_TIMING_SENSITIVITY'].eq(0).sum()):,}"
    )


def main() -> None:
    started = perf_counter()
    output_paths = [
        ALLOCATION_CANDIDATE_OCCASIONS_PATH,
        ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
        ALLOCATION_CANDIDATE_QA_PATH,
    ]
    assert_no_outputs_exist(output_paths)

    master = read_csv_strings(CLASSIFIED_PATH)
    sensitivity = read_csv_strings(SENSITIVITY_PATH, usecols=["CHOICE_ID"])
    cars = read_csv_strings(CARS_PATH, usecols=["H_ID", "A_ID"])
    car_roster = build_car_roster(cars)

    classified = add_candidate_flags(master, car_roster)
    classified = add_existing_sample_membership(classified, sensitivity)
    candidate = select_and_order_candidate(classified)
    households = build_households(candidate)
    validate_outputs(candidate, households, car_roster)
    qa = build_qa(classified, candidate, households)

    ALLOCATION_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(ALLOCATION_CANDIDATE_OCCASIONS_PATH, index=False)
    households.to_csv(ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH, index=False)
    qa.to_csv(ALLOCATION_CANDIDATE_QA_PATH, index=False)

    print("INPUTS")
    for path in [CLASSIFIED_PATH, CARS_PATH, SENSITIVITY_PATH]:
        print(path)
    print_summary(candidate, households)
    print("\nOUTPUTS")
    for path in output_paths:
        print(path)
    print(f"\nCompleted in {perf_counter() - started:.2f}s.")


if __name__ == "__main__":
    main()
