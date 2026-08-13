from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

OCCASIONS_PATH = DATA_PROCESSED / "reconstruction" /"phase2"/ "vehicle_choice_occasions_all_v2.csv"
TRIPS_PATH = DATA_PROCESSED / "reconstruction" /"phase1"/ "trips_home_chain_enriched.csv"
CARS_PATH = DATA_PROCESSED /"selected_raw" / "cars_selected_raw.csv"
PHASE1_QA_PATH = DATA_PROCESSED / "reconstruction" /"phase1"/"phase1_QA_summary.csv"
PHASE2_QA_PATH = DATA_PROCESSED / "reconstruction" /"phase2"/"phase2_QA_summary_v2.csv"

CLASSIFIED_PATH = DATA_PROCESSED /"reconstruction" /"phase3"/ "vehicle_choice_occasions_classified.csv"
FUNNEL_PATH = DATA_PROCESSED / "reconstruction" /"phase3"/ "phase3_sample_funnel.csv"
WIDE_PATH = DATA_PROCESSED / "reconstruction" /"phase3"/ "mnl_vehicle_choice_wide_base.csv"
QA_PATH = DATA_PROCESSED / "reconstruction" /"phase3"/ "vehicle_availability_QA_summary.csv"

VALID_A_IDS = {1, 2, 3}
EXPECTED_ALL_OCCASIONS = 56_340
EXPECTED_CORE_OCCASIONS = 32_805
EXPECTED_STRICT_OCCASIONS = 30_816

KEEP_CONTEXT_COLUMNS = [
    "W_ZWECK",
    "zweck",
    "W_SO1",
    "H_ANZAUTO",
    "pkw_fmf",
]

WIDE_COLUMNS = [
    "CHOICE_ID",
    "SOURCE_ROW_ID",
    "H_ID",
    "HP_ID",
    "P_ID",
    "W_ID",
    "CHOICE",
    "AV_1",
    "AV_2",
    "START_MIN",
    "ARRIVAL_MIN",
    "W_ZWECK",
    "W_SO1",
    "W_GEW",
]

FORBIDDEN_WIDE_COLUMNS = {
    "TRIP_ID",
    "CHOSEN_A_ID",
    "alternative_A_ID",
    "chosen",
    "AVAILABLE_CAR_SET",
    "N_AVAILABLE_CARS",
    "H_ANZAUTO",
    "pkw_fmf",
    "VALID_STRICT_CHOICE_OCCASION",
    "zweck",
    "rbw_present_flag",
    "rbw_pkw_mode_flag",
}

COLUMN_MAP = {
    "choice_id": "CHOICE_ID",
    "trip_id": "TRIP_ID",
    "source_row_id": "SOURCE_ROW_ID",
    "h_id": "H_ID",
    "hp_id": "HP_ID",
    "p_id": "P_ID",
    "w_id": "W_ID",
    "home_origin": "HOME_ORIGIN",
    "start_min": "START_MIN",
    "arrival_min": "ARRIVAL_MIN",
    "chosen_a_id": "CHOSEN_A_ID",
    "available_car_set": "AVAILABLE_CAR_SET",
    "n_available_cars": "N_AVAILABLE_CARS",
    "complete_fleet_baseline": "complete_fleet_baseline_flag",
    "topcoded_multicar_household": "topcoded_multicar_household_flag",
    "fleet_inventory_mismatch": "fleet_inventory_mismatch_flag",
    "household_car_count_conflict": "household_car_count_conflict_flag",
    "complete_vehicle_time": "complete_vehicle_time_flag",
    "snapshot_valid": "snapshot_valid_flag",
    "invalid_snapshot": "invalid_snapshot_flag",
    "availability_conflict": "availability_conflict_flag",
    "unknown_snapshot": "unknown_vehicle_state_at_snapshot_flag",
    "simultaneous_departure": "simultaneous_home_departure_flag",
    "same_vehicle_simultaneous": "same_vehicle_simultaneous_departure_flag",
    "household_time_unresolved": "household_vehicle_time_unresolved_flag",
    "household_timeline_conflict": "household_vehicle_timeline_conflict_flag",
    "household_vehicle_overlap": "household_vehicle_trip_overlap_flag",
    "household_location_transition_conflict": "household_vehicle_location_transition_conflict_flag",
    "zero_duration_driver_trip": "zero_duration_driver_trip_flag",
    "household_zero_duration": "household_zero_duration_driver_trip_flag",
    "unmatched_reported_vehicle": "unmatched_reported_vehicle_flag",
    "duplicate_trip_identifier": "duplicate_trip_identifier_flag",
    "rbw_present": "rbw_present_flag",
    "rbw_pkw_mode": "rbw_pkw_mode_flag",
}

CORE_STEPS = [
    ("INCOMPLETE_FLEET", "EXCL_INCOMPLETE_FLEET"),
    ("UNMATCHED_VEHICLE", "EXCL_UNMATCHED_VEHICLE"),
    ("DUPLICATE_IDENTIFIER", "EXCL_DUPLICATE_IDENTIFIER"),
    ("INCOMPLETE_CHOICE_TRIP_TIME", "EXCL_INCOMPLETE_CHOICE_TRIP_TIME"),
    ("INVALID_SNAPSHOT", "EXCL_INVALID_SNAPSHOT"),
    ("UNKNOWN_SNAPSHOT", "EXCL_UNKNOWN_SNAPSHOT"),
    ("AVAILABILITY_CONFLICT", "EXCL_AVAILABILITY_CONFLICT"),
    ("INSUFFICIENT_AVAILABLE_CARS", "EXCL_INSUFFICIENT_AVAILABLE_CARS"),
    ("SIMULTANEOUS_DEPARTURE", "EXCL_SIMULTANEOUS_DEPARTURE"),
]

STRICT_STEPS = [
    ("HOUSEHOLD_TIME_UNRESOLVED", "EXCL_HOUSEHOLD_TIME_UNRESOLVED"),
    ("HOUSEHOLD_TIMELINE_CONFLICT", "EXCL_HOUSEHOLD_TIMELINE_CONFLICT"),
    ("VEHICLE_OVERLAP", "EXCL_VEHICLE_OVERLAP"),
    ("LOCATION_TRANSITION_CONFLICT", "EXCL_LOCATION_TRANSITION_CONFLICT"),
    ("ZERO_DURATION_HISTORY", "EXCL_ZERO_DURATION_HISTORY"),
]

ALL_STEPS = CORE_STEPS + STRICT_STEPS


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def integer_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def flag_not_one(series: pd.Series) -> pd.Series:
    return integer_series(series).fillna(-1).ne(1).astype(int)


def flag_equals_one(series: pd.Series) -> pd.Series:
    return integer_series(series).fillna(0).eq(1).astype(int)


def parse_available_set(value: object) -> tuple[list[int], int]:
    text = "" if pd.isna(value) else str(value).strip()
    if text == "":
        return [], 0
    parts = text.split("|")
    parsed: list[int] = []
    for part in parts:
        if part == "":
            return [], 1
        try:
            number = int(float(part))
        except ValueError:
            return [], 1
        if number not in VALID_A_IDS:
            return [], 1
        parsed.append(number)
    if len(parsed) != len(set(parsed)) or parsed != sorted(parsed):
        return parsed, 1
    return parsed, 0


def ordered_set(values: Iterable[int]) -> str:
    return "|".join(str(int(value)) for value in sorted(values))


def build_fleet_pairs(cars: pd.DataFrame) -> set[tuple[str, int]]:
    require_columns(cars, ["H_ID", "A_ID"], "vehicle inventory")
    a_id = integer_series(cars["A_ID"])
    valid = cars.loc[a_id.isin(VALID_A_IDS), ["H_ID"]].copy()
    valid["A_ID_NUM"] = a_id[a_id.isin(VALID_A_IDS)].astype(int).values
    return set(zip(valid["H_ID"].astype(str), valid["A_ID_NUM"].astype(int)))


def load_optional_qa(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    qa = read_csv_strings(path)
    if not {"metric", "count"}.issubset(qa.columns):
        return {}
    out: dict[str, int] = {}
    for row in qa.itertuples(index=False):
        try:
            out[str(row.metric)] = int(float(str(row.count)))
        except ValueError:
            continue
    return out


def validate_and_parse_occasions(occasions: pd.DataFrame) -> pd.DataFrame:
    required = [
        COLUMN_MAP["choice_id"],
        COLUMN_MAP["trip_id"],
        COLUMN_MAP["source_row_id"],
        COLUMN_MAP["h_id"],
        COLUMN_MAP["hp_id"],
        COLUMN_MAP["p_id"],
        COLUMN_MAP["w_id"],
        COLUMN_MAP["home_origin"],
        COLUMN_MAP["start_min"],
        COLUMN_MAP["arrival_min"],
        COLUMN_MAP["chosen_a_id"],
        COLUMN_MAP["available_car_set"],
        COLUMN_MAP["n_available_cars"],
        COLUMN_MAP["complete_fleet_baseline"],
        COLUMN_MAP["complete_vehicle_time"],
        COLUMN_MAP["snapshot_valid"],
        COLUMN_MAP["invalid_snapshot"],
        COLUMN_MAP["availability_conflict"],
        COLUMN_MAP["unknown_snapshot"],
        COLUMN_MAP["simultaneous_departure"],
        COLUMN_MAP["household_time_unresolved"],
        COLUMN_MAP["household_timeline_conflict"],
        COLUMN_MAP["household_vehicle_overlap"],
        COLUMN_MAP["household_location_transition_conflict"],
        COLUMN_MAP["household_zero_duration"],
        COLUMN_MAP["unmatched_reported_vehicle"],
        COLUMN_MAP["duplicate_trip_identifier"],
        COLUMN_MAP["rbw_present"],
        COLUMN_MAP["rbw_pkw_mode"],
    ]
    require_columns(occasions, required, "occasion")

    out = occasions.copy()
    home_origin = integer_series(out[COLUMN_MAP["home_origin"]])
    bad_home = out.loc[home_origin.ne(1)]
    if not bad_home.empty:
        print("\nNON-HOME-ORIGIN OCCASIONS")
        print(bad_home[[COLUMN_MAP["choice_id"], COLUMN_MAP["source_row_id"], COLUMN_MAP["home_origin"]]].to_string(index=False))
    assert bad_home.empty, "Phase 3 occasion file must contain only HOME_ORIGIN == 1 rows."

    out["duplicate_choice_id_flag"] = out[COLUMN_MAP["choice_id"]].eq("") | out[COLUMN_MAP["choice_id"]].duplicated(keep=False)
    out["duplicate_choice_id_flag"] = out["duplicate_choice_id_flag"].astype(int)
    duplicates = out.loc[out["duplicate_choice_id_flag"].eq(1)]
    if not duplicates.empty:
        print("\nDUPLICATE OR MISSING CHOICE_ID RECORDS")
        print(duplicates.to_string(index=False))

    parsed = out[COLUMN_MAP["available_car_set"]].map(parse_available_set)
    out["parsed_available_car_list"] = parsed.map(lambda item: item[0])
    out["parsed_AVAILABLE_CAR_SET"] = out["parsed_available_car_list"].map(ordered_set)
    out["available_set_parse_error_flag"] = parsed.map(lambda item: item[1]).astype(int)
    out["parsed_available_car_count"] = out["parsed_available_car_list"].map(len).astype(int)

    valid_snapshot = flag_equals_one(out[COLUMN_MAP["snapshot_valid"]]).eq(1)
    n_available = integer_series(out[COLUMN_MAP["n_available_cars"]])
    mismatched = out.loc[valid_snapshot & out["available_set_parse_error_flag"].eq(0) & n_available.ne(out["parsed_available_car_count"])]
    if not mismatched.empty:
        print("\nVALID SNAPSHOT AVAILABLE-CAR COUNT MISMATCHES")
        print(
            mismatched[
                [
                    COLUMN_MAP["choice_id"],
                    COLUMN_MAP["available_car_set"],
                    COLUMN_MAP["n_available_cars"],
                    "parsed_available_car_count",
                ]
            ].to_string(index=False)
        )
    assert mismatched.empty, "Parsed available car count must equal N_AVAILABLE_CARS for valid snapshots."
    return out


def merge_context(occasions: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    context_cols = ["SOURCE_ROW_ID", *[col for col in KEEP_CONTEXT_COLUMNS if col in trips.columns]]
    context = trips[context_cols].copy()
    return occasions.merge(context, on="SOURCE_ROW_ID", how="left", validate="many_to_one")


def canonicalize_trip_purpose(occasions: pd.DataFrame) -> pd.DataFrame:
    out = occasions.copy()
    if "W_ZWECK" not in out.columns and "zweck" not in out.columns:
        raise KeyError("Trip context must contain W_ZWECK or zweck.")
    if "W_ZWECK" not in out.columns:
        out["W_ZWECK"] = out["zweck"]
    if "zweck" in out.columns:
        canonical = out["W_ZWECK"].astype("string").str.strip()
        fallback = out["zweck"].astype("string").str.strip()
        mismatch = canonical.ne("") & fallback.ne("") & canonical.ne(fallback)
        if mismatch.any():
            print("\nW_ZWECK / zweck MISMATCHES IN PHASE 3 OCCASIONS")
            print(
                out.loc[mismatch, ["CHOICE_ID", "SOURCE_ROW_ID", "W_ZWECK", "zweck"]]
                .head(10)
                .to_string(index=False)
            )
            raise ValueError("W_ZWECK and zweck disagree in the Phase 3 occasion universe.")
        out["W_ZWECK"] = canonical.mask(canonical.eq(""), fallback)
        out = out.drop(columns="zweck")
    return out


def create_exclusion_flags(occasions: pd.DataFrame, fleet_pairs: set[tuple[str, int]]) -> pd.DataFrame:
    out = occasions.copy()
    chosen = integer_series(out[COLUMN_MAP["chosen_a_id"]])
    start = integer_series(out[COLUMN_MAP["start_min"]])
    arrival = integer_series(out[COLUMN_MAP["arrival_min"]])
    n_available = integer_series(out[COLUMN_MAP["n_available_cars"]])
    snapshot_valid = flag_equals_one(out[COLUMN_MAP["snapshot_valid"]])
    chosen_in_fleet = pd.Series(
        [
            (str(h_id), int(a_id)) in fleet_pairs if not pd.isna(a_id) else False
            for h_id, a_id in zip(out[COLUMN_MAP["h_id"]], chosen)
        ],
        index=out.index,
    )
    chosen_in_available = pd.Series(
        [
            int(a_id) in available if not pd.isna(a_id) else False
            for a_id, available in zip(chosen, out["parsed_available_car_list"])
        ],
        index=out.index,
    )

    out["EXCL_INCOMPLETE_FLEET"] = flag_not_one(out[COLUMN_MAP["complete_fleet_baseline"]])
    out["EXCL_UNMATCHED_VEHICLE"] = (
        flag_equals_one(out[COLUMN_MAP["unmatched_reported_vehicle"]]).eq(1) | ~chosen_in_fleet
    ).astype(int)
    out["EXCL_DUPLICATE_IDENTIFIER"] = (
        flag_equals_one(out[COLUMN_MAP["duplicate_trip_identifier"]]).eq(1) | out["duplicate_choice_id_flag"].eq(1)
    ).astype(int)

    if COLUMN_MAP["complete_vehicle_time"] in out.columns:
        out["EXCL_INCOMPLETE_CHOICE_TRIP_TIME"] = flag_not_one(out[COLUMN_MAP["complete_vehicle_time"]])
    else:
        out["EXCL_INCOMPLETE_CHOICE_TRIP_TIME"] = (
            start.isna() | arrival.isna() | arrival.lt(start)
        ).astype(int)

    out["EXCL_INVALID_SNAPSHOT"] = (
        snapshot_valid.ne(1)
        | flag_equals_one(out[COLUMN_MAP["invalid_snapshot"]]).eq(1)
        | out["available_set_parse_error_flag"].eq(1)
    ).astype(int)
    out["EXCL_UNKNOWN_SNAPSHOT"] = flag_equals_one(out[COLUMN_MAP["unknown_snapshot"]])
    out["EXCL_AVAILABILITY_CONFLICT"] = (
        snapshot_valid.eq(1)
        & out["available_set_parse_error_flag"].eq(0)
        & (
            flag_equals_one(out[COLUMN_MAP["availability_conflict"]]).eq(1)
            | ~chosen_in_available
        )
    ).astype(int)
    out["EXCL_INSUFFICIENT_AVAILABLE_CARS"] = n_available.fillna(-1).lt(2).astype(int)
    out["EXCL_SIMULTANEOUS_DEPARTURE"] = flag_equals_one(out[COLUMN_MAP["simultaneous_departure"]])
    out["EXCL_HOUSEHOLD_TIME_UNRESOLVED"] = flag_equals_one(out[COLUMN_MAP["household_time_unresolved"]])
    out["EXCL_HOUSEHOLD_TIMELINE_CONFLICT"] = flag_equals_one(out[COLUMN_MAP["household_timeline_conflict"]])
    out["EXCL_VEHICLE_OVERLAP"] = flag_equals_one(out[COLUMN_MAP["household_vehicle_overlap"]])
    out["EXCL_LOCATION_TRANSITION_CONFLICT"] = flag_equals_one(out[COLUMN_MAP["household_location_transition_conflict"]])
    out["EXCL_ZERO_DURATION_HISTORY"] = flag_equals_one(out[COLUMN_MAP["household_zero_duration"]])

    core_flags = [flag for _, flag in CORE_STEPS]
    strict_flags = [flag for _, flag in STRICT_STEPS]
    out["VALID_CORE_OCCASION"] = out[core_flags].sum(axis=1).eq(0).astype(int)
    out["VALID_STRICT_CHOICE_OCCASION"] = (
        out["VALID_CORE_OCCASION"].eq(1) & out[strict_flags].sum(axis=1).eq(0)
    ).astype(int)

    reason_names = [name for name, _ in ALL_STEPS]
    reason_flags = [flag for _, flag in ALL_STEPS]
    all_reasons = []
    primary_reasons = []
    for row in out[reason_flags].itertuples(index=False, name=None):
        active = [reason_names[i] for i, value in enumerate(row) if int(value) == 1]
        all_reasons.append("|".join(active) if active else "NONE")
        primary_reasons.append(active[0] if active else "NONE")
    out["ALL_EXCLUSION_REASONS"] = all_reasons
    out["PRIMARY_EXCLUSION_REASON"] = primary_reasons
    return out


def build_funnel(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current = pd.Series(True, index=classified.index)
    initial = len(classified)
    for step_number, (step_name, flag) in enumerate(ALL_STEPS, start=1):
        input_occasions = int(current.sum())
        newly_excluded = current & classified[flag].eq(1)
        current = current & classified[flag].eq(0)
        remaining = int(current.sum())
        rows.append(
            {
                "step_number": step_number,
                "step_name": step_name,
                "input_occasions": input_occasions,
                "newly_excluded_occasions": int(newly_excluded.sum()),
                "remaining_occasions": remaining,
                "remaining_households": int(classified.loc[current, COLUMN_MAP["h_id"]].nunique()),
                "share_of_initial_occasions": remaining / initial if initial else 0.0,
                "share_of_previous_step": remaining / input_occasions if input_occasions else 0.0,
            }
        )
    return pd.DataFrame(rows)


def merge_choice_weights(strict: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    require_columns(trips, ["SOURCE_ROW_ID", "W_GEW"], "trip weight source")
    source_ids = trips["SOURCE_ROW_ID"].astype("string").str.strip()
    bad_source_ids = source_ids.eq("") | source_ids.duplicated(keep=False)
    if bad_source_ids.any():
        print("\nMISSING OR DUPLICATE SOURCE_ROW_ID VALUES IN TRIP WEIGHT SOURCE")
        print(trips.loc[bad_source_ids, ["SOURCE_ROW_ID", "W_GEW"]].head(10).to_string(index=False))
        raise ValueError("SOURCE_ROW_ID must be non-missing and unique in the source trip table.")

    source_weights = numeric(trips["W_GEW"])
    nonnumeric = trips["W_GEW"].astype("string").str.strip().ne("") & source_weights.isna()
    if nonnumeric.any():
        print("\nNON-NUMERIC W_GEW VALUES IN TRIP WEIGHT SOURCE")
        print(trips.loc[nonnumeric, ["SOURCE_ROW_ID", "W_GEW"]].head(10).to_string(index=False))
        raise ValueError("W_GEW contains non-numeric values in the source trip table.")

    strict_ids = strict["SOURCE_ROW_ID"].astype("string").str.strip()
    bad_strict_ids = strict_ids.eq("") | strict_ids.duplicated(keep=False)
    if bad_strict_ids.any():
        print("\nMISSING OR DUPLICATE SOURCE_ROW_ID VALUES IN STRICT OCCASIONS")
        print(strict.loc[bad_strict_ids, ["CHOICE_ID", "SOURCE_ROW_ID"]].head(10).to_string(index=False))
        raise ValueError("SOURCE_ROW_ID must be non-missing and unique in the strict occasion sample.")

    weights = trips[["SOURCE_ROW_ID"]].copy()
    weights["W_GEW"] = source_weights
    before_choice_ids = strict["CHOICE_ID"].tolist()
    before_source_ids = strict["SOURCE_ROW_ID"].tolist()
    merged = strict.merge(weights, on="SOURCE_ROW_ID", how="left", validate="one_to_one", indicator=True)
    if (
        len(merged) != len(strict)
        or merged["CHOICE_ID"].tolist() != before_choice_ids
        or merged["SOURCE_ROW_ID"].tolist() != before_source_ids
        or merged["_merge"].ne("both").any()
    ):
        print("\nSTRICT WEIGHT MERGE FAILURES")
        print(merged.loc[merged["_merge"].ne("both"), ["CHOICE_ID", "SOURCE_ROW_ID", "_merge"]].head(10).to_string(index=False))
        raise ValueError("The one-to-one W_GEW merge lost, created, or failed to match strict occasions.")
    return merged.drop(columns="_merge")


def build_wide(strict: pd.DataFrame, fleet_pairs: set[tuple[str, int]]) -> pd.DataFrame:
    require_columns(strict, [*WIDE_COLUMNS[:6], "CHOSEN_A_ID", "parsed_available_car_list", *WIDE_COLUMNS[9:]], "strict occasion")
    # CHOICE_ID identifies the trip-based choice occasion; TRIP_ID stays in the QA master only.
    wide = strict[["CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID"]].copy()
    wide["CHOICE"] = integer_series(strict["CHOSEN_A_ID"])
    # AVAILABLE_CAR_SET remains reconstruction metadata; AV_1/AV_2 are the Biogeme representation.
    wide["AV_1"] = strict["parsed_available_car_list"].map(lambda values: int(1 in values))
    wide["AV_2"] = strict["parsed_available_car_list"].map(lambda values: int(2 in values))
    for col in ["START_MIN", "ARRIVAL_MIN", "W_ZWECK", "W_SO1", "W_GEW"]:
        wide[col] = strict[col]
    wide = wide[WIDE_COLUMNS]
    validate_wide(strict, wide, fleet_pairs)
    return wide


def validate_wide(
    strict: pd.DataFrame,
    wide: pd.DataFrame,
    fleet_pairs: set[tuple[str, int]],
) -> None:
    if list(wide.columns) != WIDE_COLUMNS:
        raise ValueError(f"Wide columns differ from the required ordered schema: {list(wide.columns)}")
    forbidden = sorted(FORBIDDEN_WIDE_COLUMNS.intersection(wide.columns))
    if forbidden:
        raise ValueError(f"Long-format or redundant columns remain in the wide output: {forbidden}")

    choice_missing = wide["CHOICE_ID"].isna() | wide["CHOICE_ID"].astype("string").str.strip().eq("")
    choice_duplicate = wide["CHOICE_ID"].duplicated(keep=False)
    source_missing = wide["SOURCE_ROW_ID"].isna() | wide["SOURCE_ROW_ID"].astype("string").str.strip().eq("")
    source_duplicate = wide["SOURCE_ROW_ID"].duplicated(keep=False)
    if choice_missing.any() or choice_duplicate.any() or source_missing.any() or source_duplicate.any():
        bad = choice_missing | choice_duplicate | source_missing | source_duplicate
        print("\nINVALID WIDE IDENTIFIERS")
        print(wide.loc[bad, ["CHOICE_ID", "SOURCE_ROW_ID"]].head(10).to_string(index=False))
        raise ValueError("Wide CHOICE_ID and SOURCE_ROW_ID must each be non-missing and unique.")

    choice_to_source = wide.groupby("CHOICE_ID")["SOURCE_ROW_ID"].nunique(dropna=False)
    source_to_choice = wide.groupby("SOURCE_ROW_ID")["CHOICE_ID"].nunique(dropna=False)
    if not choice_to_source.eq(1).all() or not source_to_choice.eq(1).all():
        raise ValueError("CHOICE_ID and SOURCE_ROW_ID must have a one-to-one mapping.")

    choice = integer_series(wide["CHOICE"])
    av_1 = integer_series(wide["AV_1"])
    av_2 = integer_series(wide["AV_2"])
    structural_bad = ~choice.isin({1, 2}) | ~av_1.isin({0, 1}) | ~av_2.isin({0, 1})
    availability_bad = av_1.ne(1) | av_2.ne(1) | ((choice.eq(1) & av_1.ne(1)) | (choice.eq(2) & av_2.ne(1)))
    if structural_bad.any() or availability_bad.any():
        print("\nINVALID CHOICE OR AVAILABILITY VALUES")
        print(wide.loc[structural_bad | availability_bad, ["CHOICE_ID", "CHOICE", "AV_1", "AV_2"]].head(10).to_string(index=False))
        raise ValueError("Strict wide rows require CHOICE in {1,2}, binary AV values, and AV_1 == AV_2 == 1.")

    complete_fleet = integer_series(strict["complete_fleet_baseline_flag"]).eq(1)
    two_cars = integer_series(strict["N_AVAILABLE_CARS"]).eq(2)
    two_car_set = strict["parsed_available_car_list"].map(lambda values: values == [1, 2])
    household_two_cars = integer_series(strict["H_ANZAUTO"]).eq(2)
    car_driver_trip = integer_series(strict["pkw_fmf"]).eq(1)
    strict_valid = integer_series(strict["VALID_STRICT_CHOICE_OCCASION"]).eq(1)
    baseline_bad = ~(complete_fleet & two_cars & two_car_set & household_two_cars & car_driver_trip & strict_valid)
    if baseline_bad.any():
        cols = [
            "CHOICE_ID",
            "H_ID",
            "complete_fleet_baseline_flag",
            "H_ANZAUTO",
            "pkw_fmf",
            "AVAILABLE_CAR_SET",
            "N_AVAILABLE_CARS",
            "CHOSEN_A_ID",
            "VALID_STRICT_CHOICE_OCCASION",
        ]
        print("\nSTRICT ROWS OUTSIDE THE COMPLETE TWO-CAR BASELINE")
        print(strict.loc[baseline_bad, cols].head(10).to_string(index=False))
        raise ValueError("All strict rows must belong to the intended complete two-car baseline.")

    bad_fleet = [
        (str(h_id), a_id)
        for h_id in strict["H_ID"]
        for a_id in (1, 2)
        if (str(h_id), a_id) not in fleet_pairs
    ]
    if bad_fleet:
        raise ValueError(f"Strict wide alternatives outside the recorded fleet: {bad_fleet[:10]}")

    weights = numeric(wide["W_GEW"])
    populated = wide["W_GEW"].astype("string").str.strip().ne("")
    weight_bad = (~populated) | weights.isna() | ~weights.map(lambda value: math.isfinite(value) if not pd.isna(value) else False) | weights.le(0)
    if weight_bad.any():
        print("\nINVALID STRICT W_GEW VALUES")
        print(wide.loc[weight_bad, ["CHOICE_ID", "SOURCE_ROW_ID", "W_GEW"]].head(10).to_string(index=False))
        raise ValueError("Strict W_GEW values must be numeric, non-missing, finite, and strictly positive.")
    wide["W_GEW"] = weights

    if len(wide) != len(strict) or wide["CHOICE_ID"].nunique() != len(strict):
        raise ValueError("Wide row and unique CHOICE_ID counts must equal the strict occasion count.")


def prevalence_rows(classified: pd.DataFrame) -> list[dict]:
    denom = len(classified)
    rows = []
    for _, flag in ALL_STEPS:
        rows.append(
            qa_row(
                "RAW EXCLUSION-FLAG PREVALENCE",
                flag,
                int(classified[flag].sum()),
                denom,
                "all home-origin choice occasions",
            )
        )
    return rows


def qa_row(section: str, metric: str, count: int, denominator: int, denominator_description: str) -> dict:
    return {
        "section": section,
        "metric": metric,
        "count": int(count),
        "share": count / denominator if denominator else 0.0,
        "denominator_description": denominator_description,
    }


def qa_stat_row(metric: str, value: float | int) -> dict:
    return {
        "section": "WEIGHT QA",
        "metric": metric,
        "count": value,
        "share": "",
        "denominator_description": "strict wide choice occasions",
    }


def weight_stats(values: pd.Series) -> dict[str, float | int]:
    weights = numeric(values)
    return {
        "missing W_GEW": int(weights.isna().sum()),
        "zero W_GEW": int(weights.eq(0).sum()),
        "negative W_GEW": int(weights.lt(0).sum()),
        "minimum W_GEW": float(weights.min()),
        "maximum W_GEW": float(weights.max()),
        "mean W_GEW": float(weights.mean()),
        "median W_GEW": float(weights.median()),
        "p01 W_GEW": float(weights.quantile(0.01)),
        "p05 W_GEW": float(weights.quantile(0.05)),
        "p95 W_GEW": float(weights.quantile(0.95)),
        "p99 W_GEW": float(weights.quantile(0.99)),
    }


def add_distribution_rows(rows: list[dict], section: str, label: str, data: pd.DataFrame, denom_desc: str) -> None:
    denom = len(data)
    n_available = integer_series(data["N_AVAILABLE_CARS"])
    for value in [0, 1, 2, 3]:
        rows.append(qa_row(section, f"{label}: {value} available cars", int(n_available.eq(value).sum()), denom, denom_desc))


def add_value_distribution_rows(
    rows: list[dict],
    section: str,
    label: str,
    series: pd.Series,
    denominator: int,
    denominator_description: str,
) -> None:
    counts = series.astype(str).value_counts().sort_index()
    for value, count in counts.items():
        rows.append(qa_row(section, f"{label}: {value}", int(count), denominator, denominator_description))


def build_qa_summary(
    classified: pd.DataFrame,
    wide: pd.DataFrame,
    funnel: pd.DataFrame,
    phase1_qa: dict[str, int],
    phase2_qa: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict] = []
    initial = len(classified)
    households = classified["H_ID"].nunique()

    phase1_metrics = [
        ("raw retained Phase 1 rows", phase1_qa.get("all retained rows", 0), phase1_qa.get("all retained rows", 0), "accepted Phase 1 retained rows"),
        ("chain-eligible detailed trips", phase1_qa.get("directly reported chain-eligible trip rows", 0), phase1_qa.get("all retained rows", 0), "accepted Phase 1 retained rows"),
        ("rbW rows excluded from chain reconstruction", phase1_qa.get("rbW rows excluded from chain reconstruction", 0), phase1_qa.get("all retained rows", 0), "accepted Phase 1 retained rows"),
        ("multi-car households", phase1_qa.get("unique households", households), phase1_qa.get("unique households", households), "accepted Phase 1 households"),
        ("H_ANZAUTO == 2 households", phase1_qa.get("H_ANZAUTO == 2 households", 0), phase1_qa.get("unique households", households), "accepted Phase 1 households"),
        ("H_ANZAUTO == 3 top-coded households", phase1_qa.get("H_ANZAUTO == 3 households", 0), phase1_qa.get("unique households", households), "accepted Phase 1 households"),
        ("complete-fleet baseline households", phase1_qa.get("complete-fleet baseline households", 0), phase1_qa.get("unique households", households), "accepted Phase 1 households"),
    ]
    for metric, count, denom, desc in phase1_metrics:
        rows.append(qa_row("PHASE 1 / INPUT CONTEXT", metric, count, denom, desc))

    vehicle_events = phase2_qa.get("vehicle departure events", 0) + phase2_qa.get("vehicle arrival events", 0)
    identifiable_trips = phase2_qa.get("identifiable household-car driver trips", 0)
    home_origin_occasions = phase2_qa.get("home-origin choice occasions", initial)
    phase2_metrics = [
        ("identifiable household-car driver trips", "identifiable household-car driver trips", identifiable_trips, "identifiable household-car driver trips"),
        ("complete-time identifiable driver trips", "complete-time identifiable driver trips", identifiable_trips, "identifiable household-car driver trips"),
        ("Phase 2 incomplete-time identifiable driver trips", "incomplete-time identifiable driver trips", identifiable_trips, "all identifiable driver trips, including non-home-origin trips"),
        ("events excluded from strict timeline", "events excluded from strict timeline", vehicle_events, "Phase 2 vehicle events"),
        ("home-origin choice occasions", "home-origin choice occasions", home_origin_occasions, "Phase 2 home-origin choice occasions"),
        ("Phase 2 invalid snapshot home-origin occasions", "invalid snapshot cases", home_origin_occasions, "Phase 2 home-origin choice occasions only"),
        ("true availability conflicts", "true availability conflicts", home_origin_occasions, "Phase 2 home-origin choice occasions"),
        ("unknown snapshot occasions", "unknown snapshot occasions", home_origin_occasions, "Phase 2 home-origin choice occasions"),
        ("simultaneous home-departure occasions", "simultaneous home-departure occasions", home_origin_occasions, "Phase 2 home-origin choice occasions"),
        ("same-vehicle simultaneous departures", "same-vehicle simultaneous departures", home_origin_occasions, "Phase 2 home-origin choice occasions"),
    ]
    for label, source_metric, denominator, description in phase2_metrics:
        rows.append(qa_row("PHASE 2 RECONSTRUCTION", label, phase2_qa.get(source_metric, 0), denominator, description))
    rows.append(qa_row("PHASE 2 RECONSTRUCTION", "vehicle events", vehicle_events, max(vehicle_events, 1), "accepted Phase 2 v2 events"))

    add_distribution_rows(rows, "AVAILABILITY DISTRIBUTION BEFORE FILTERING", "all home-origin occasions", classified, "all home-origin choice occasions")
    add_distribution_rows(
        rows,
        "AVAILABILITY DISTRIBUTION BEFORE FILTERING",
        "complete two-car baseline occasions",
        classified[classified["complete_fleet_baseline_flag"].astype(int).eq(1)],
        "complete two-car baseline occasions",
    )
    add_distribution_rows(
        rows,
        "AVAILABILITY DISTRIBUTION BEFORE FILTERING",
        "top-coded H_ANZAUTO == 3 occasions",
        classified[classified["topcoded_multicar_household_flag"].astype(int).eq(1)],
        "top-coded H_ANZAUTO == 3 occasions",
    )

    rows.extend(prevalence_rows(classified))

    core = classified[classified["VALID_CORE_OCCASION"].eq(1)]
    strict = classified[classified["VALID_STRICT_CHOICE_OCCASION"].eq(1)]
    rows.append(qa_row("SEQUENTIAL FUNNEL", "valid core occasions", len(core), initial, "all home-origin choice occasions"))
    rows.append(qa_row("SEQUENTIAL FUNNEL", "valid strict baseline occasions", len(strict), initial, "all home-origin choice occasions"))
    for row in funnel.itertuples(index=False):
        rows.append(qa_row("SEQUENTIAL FUNNEL", f"newly excluded: {row.step_name}", row.newly_excluded_occasions, initial, "all home-origin choice occasions"))
        rows.append(qa_row("SEQUENTIAL FUNNEL", f"remaining households after: {row.step_name}", row.remaining_households, households, "all home-origin choice households"))

    rows.append(qa_row("STRICT SAMPLE OUTPUT", "strict MNL CHOICE_ID values", strict["CHOICE_ID"].nunique(), len(strict), "strict valid choice occasions"))
    rows.append(qa_row("STRICT SAMPLE OUTPUT", "strict MNL households", strict["H_ID"].nunique(), households, "all home-origin choice households"))
    rows.append(qa_row("STRICT SAMPLE OUTPUT", "strict wide-format rows", len(wide), max(len(strict), 1), "one wide row per strict choice occasion"))
    rows.append(qa_row("STRICT SAMPLE OUTPUT", "strict unique CHOICE_ID", wide["CHOICE_ID"].nunique(), max(len(strict), 1), "strict valid choice occasions"))
    add_value_distribution_rows(rows, "STRICT SAMPLE OUTPUT", "CHOICE distribution", wide["CHOICE"], len(wide), "strict wide choice occasions")
    add_value_distribution_rows(rows, "STRICT SAMPLE OUTPUT", "AV_1 distribution", wide["AV_1"], len(wide), "strict wide choice occasions")
    add_value_distribution_rows(rows, "STRICT SAMPLE OUTPUT", "AV_2 distribution", wide["AV_2"], len(wide), "strict wide choice occasions")

    # rbW rows were excluded upstream; these flags describe households, not the current occasions.
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "home-origin occasions in households with any rbW record", int(classified["rbw_present_flag"].astype(int).sum()), initial, "all home-origin choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "core occasions in households with any rbW record", int(core["rbw_present_flag"].astype(int).sum()), max(len(core), 1), "valid core occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "strict occasions in households with any rbW record", int(strict["rbw_present_flag"].astype(int).sum()), max(len(strict), 1), "strict valid choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "home-origin occasions in households with any rbW Pkw-mode record", int(classified["rbw_pkw_mode_flag"].astype(int).sum()), initial, "all home-origin choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "core occasions in households with any rbW Pkw-mode record", int(core["rbw_pkw_mode_flag"].astype(int).sum()), max(len(core), 1), "valid core occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "strict occasions in households with any rbW Pkw-mode record", int(strict["rbw_pkw_mode_flag"].astype(int).sum()), max(len(strict), 1), "strict valid choice occasions"))

    rows.append(qa_stat_row("strict wide rows", len(wide)))
    rows.append(qa_stat_row("strict unique CHOICE_ID", wide["CHOICE_ID"].nunique()))
    rows.extend(qa_stat_row(metric, value) for metric, value in weight_stats(wide["W_GEW"]).items())
    return pd.DataFrame(rows)


def print_reference_checks(classified: pd.DataFrame, wide: pd.DataFrame) -> None:
    actual = {
        "all home-origin occasions": len(classified),
        "VALID_CORE_OCCASION": int(classified["VALID_CORE_OCCASION"].sum()),
        "VALID_STRICT_CHOICE_OCCASION": int(classified["VALID_STRICT_CHOICE_OCCASION"].sum()),
        "strict wide rows": len(wide),
        "strict unique CHOICE_ID": wide["CHOICE_ID"].nunique(),
    }
    expected = {
        "all home-origin occasions": EXPECTED_ALL_OCCASIONS,
        "VALID_CORE_OCCASION": EXPECTED_CORE_OCCASIONS,
        "VALID_STRICT_CHOICE_OCCASION": EXPECTED_STRICT_OCCASIONS,
        "strict wide rows": EXPECTED_STRICT_OCCASIONS,
        "strict unique CHOICE_ID": EXPECTED_STRICT_OCCASIONS,
    }
    drift = {metric: (actual[metric], target) for metric, target in expected.items() if actual[metric] != target}
    if drift:
        print("\nWARNING: PHASE 3 REFERENCE COUNTS CHANGED")
        for metric, (observed, target) in drift.items():
            print(f"{metric}: observed {observed:,}; accepted reference {target:,}")


def print_final_results(classified: pd.DataFrame, wide: pd.DataFrame, stage_times: dict[str, float]) -> None:
    strict = classified[classified["VALID_STRICT_CHOICE_OCCASION"].eq(1)]
    core = classified[classified["VALID_CORE_OCCASION"].eq(1)]
    print("\nPHASE 3 FINAL COUNTS")
    print(f"all home-origin occasions: {len(classified):,}")
    print(f"VALID_CORE_OCCASION: {len(core):,}")
    print(f"VALID_STRICT_CHOICE_OCCASION: {len(strict):,}")
    print(f"strict wide-format rows: {len(wide):,}")
    print(f"strict unique CHOICE_ID: {wide['CHOICE_ID'].nunique():,}")
    print(f"strict households: {strict['H_ID'].nunique():,}")

    print("\nCHOICE DISTRIBUTION")
    print(wide["CHOICE"].value_counts().sort_index().to_string())

    print("\nAVAILABILITY")
    print(f"AV_1 == 1: {int(integer_series(wide['AV_1']).eq(1).sum()):,}")
    print(f"AV_2 == 1: {int(integer_series(wide['AV_2']).eq(1).sum()):,}")

    stats = weight_stats(wide["W_GEW"])
    print("\nWEIGHT QA")
    print(f"missing W_GEW: {stats['missing W_GEW']:,}")
    print(
        "min / max / mean / median W_GEW: "
        f"{stats['minimum W_GEW']:.6g} / {stats['maximum W_GEW']:.6g} / "
        f"{stats['mean W_GEW']:.6g} / {stats['median W_GEW']:.6g}"
    )

    print("\nOUTPUTS")
    print(CLASSIFIED_PATH.name)
    print(FUNNEL_PATH.name)
    print(WIDE_PATH.name)
    print(QA_PATH.name)

    print("\nRUNTIME BY STAGE")
    for name, seconds in stage_times.items():
        print(f"{name}: {seconds:.2f}s")


def main() -> None:
    total_t0 = perf_counter()
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    occasions = read_csv_strings(OCCASIONS_PATH)
    trip_header = pd.read_csv(TRIPS_PATH, nrows=0).columns.tolist()
    trip_cols = ["SOURCE_ROW_ID", *[col for col in KEEP_CONTEXT_COLUMNS if col in trip_header], "W_GEW"]
    trip_cols = list(dict.fromkeys(trip_cols))
    trips = read_csv_strings(TRIPS_PATH, usecols=trip_cols)
    cars = read_csv_strings(CARS_PATH, usecols=["H_ID", "A_ID"])
    phase1_qa = load_optional_qa(PHASE1_QA_PATH)
    phase2_qa = load_optional_qa(PHASE2_QA_PATH)
    stage_times["loading"] = perf_counter() - stage_start

    stage_start = perf_counter()
    fleet_pairs = build_fleet_pairs(cars)
    classified = validate_and_parse_occasions(occasions)
    classified = merge_context(classified, trips)
    classified = canonicalize_trip_purpose(classified)
    classified = create_exclusion_flags(classified, fleet_pairs)
    funnel = build_funnel(classified)
    stage_times["classification and funnel"] = perf_counter() - stage_start

    stage_start = perf_counter()
    strict = classified[classified["VALID_STRICT_CHOICE_OCCASION"].eq(1)].copy()
    # W_GEW is merged once at the choice level; one wide row is one strict vehicle choice occasion.
    strict = merge_choice_weights(strict, trips)
    wide = build_wide(strict, fleet_pairs)
    stage_times["strict weight merge and wide construction"] = perf_counter() - stage_start

    stage_start = perf_counter()
    qa = build_qa_summary(classified, wide, funnel, phase1_qa, phase2_qa)
    classified.to_csv(CLASSIFIED_PATH, index=False)
    funnel.to_csv(FUNNEL_PATH, index=False)
    wide.to_csv(WIDE_PATH, index=False)
    qa.to_csv(QA_PATH, index=False)
    stage_times["CSV writing"] = perf_counter() - stage_start
    stage_times["total"] = perf_counter() - total_t0

    print_reference_checks(classified, wide)
    print_final_results(classified, wide, stage_times)
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
