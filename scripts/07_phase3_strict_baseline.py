from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

OCCASIONS_PATH = DATA_PROCESSED / "vehicle_choice_occasions_all_v2.csv"
TRIPS_PATH = DATA_PROCESSED / "trips_home_chain_enriched.csv"
CARS_PATH = DATA_PROCESSED / "cars_selected_raw.csv"
PHASE1_QA_PATH = DATA_PROCESSED / "phase1_QA_summary.csv"
PHASE2_QA_PATH = DATA_PROCESSED / "phase2_QA_summary_v2.csv"

CLASSIFIED_PATH = DATA_PROCESSED / "vehicle_choice_occasions_classified.csv"
FUNNEL_PATH = DATA_PROCESSED / "phase3_sample_funnel.csv"
LONG_PATH = DATA_PROCESSED / "mnl_vehicle_choice_long_base.csv"
QA_PATH = DATA_PROCESSED / "vehicle_availability_QA_summary.csv"

VALID_A_IDS = {1, 2, 3}

KEEP_CONTEXT_COLUMNS = [
    "W_ZWECK",
    "zweck",
    "W_SO1",
    "H_ANZAUTO",
    "pkw_fmf",
]

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


def print_column_inventory(occasions: pd.DataFrame, trips: pd.DataFrame, cars: pd.DataFrame) -> None:
    print("\nPHASE 3 INPUT COLUMN INVENTORY")
    print("vehicle_choice_occasions_all_v2.csv:")
    print("|".join(occasions.columns))
    print("\ntrips_home_chain_enriched.csv:")
    print("|".join(trips.columns))
    print("\ncars_selected_raw.csv:")
    print("|".join(cars.columns))
    print("\nPHASE 3 COLUMN MAPPING")
    for logical, actual in COLUMN_MAP.items():
        print(f"{logical}: {actual}")


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
    context = trips[context_cols].drop_duplicates("SOURCE_ROW_ID")
    return occasions.merge(context, on="SOURCE_ROW_ID", how="left", validate="many_to_one")


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


def expand_long(strict: pd.DataFrame, fleet_pairs: set[tuple[str, int]]) -> pd.DataFrame:
    base_cols = [
        "CHOICE_ID",
        "TRIP_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "SOURCE_ROW_ID",
        "START_MIN",
        "ARRIVAL_MIN",
        "CHOSEN_A_ID",
        "N_AVAILABLE_CARS",
        "AVAILABLE_CAR_SET",
        "rbw_present_flag",
        "rbw_pkw_mode_flag",
        "VALID_STRICT_CHOICE_OCCASION",
        *[col for col in KEEP_CONTEXT_COLUMNS if col in strict.columns],
    ]
    base_cols = list(dict.fromkeys(base_cols))
    rows = []
    for occ in strict.sort_values(["H_ID", "START_MIN", "SOURCE_ROW_ID"], kind="mergesort").itertuples(index=False):
        record = occ._asdict()
        for a_id in record["parsed_available_car_list"]:
            row = {col: record[col] for col in base_cols if col in record}
            row["alternative_A_ID"] = int(a_id)
            row["chosen"] = int(int(record["CHOSEN_A_ID"]) == int(a_id))
            rows.append(row)
    long = pd.DataFrame(rows)
    if not long.empty:
        ordered_cols = [
            "CHOICE_ID",
            "TRIP_ID",
            "H_ID",
            "HP_ID",
            "P_ID",
            "W_ID",
            "SOURCE_ROW_ID",
            "START_MIN",
            "ARRIVAL_MIN",
            "alternative_A_ID",
            "CHOSEN_A_ID",
            "chosen",
            "N_AVAILABLE_CARS",
            "AVAILABLE_CAR_SET",
            "rbw_present_flag",
            "rbw_pkw_mode_flag",
            *[col for col in KEEP_CONTEXT_COLUMNS if col in long.columns],
            "VALID_STRICT_CHOICE_OCCASION",
        ]
        long = long[[col for col in ordered_cols if col in long.columns]]
    validate_long(strict, long, fleet_pairs)
    return long


def validate_long(strict: pd.DataFrame, long: pd.DataFrame, fleet_pairs: set[tuple[str, int]]) -> None:
    if strict.empty:
        assert long.empty, "Long file must be empty when there are no strict occasions."
        return
    duplicate_choices = strict[COLUMN_MAP["choice_id"]].duplicated(keep=False)
    assert not duplicate_choices.any(), "Strict occasion sample has duplicate CHOICE_ID values."
    bad_n = strict.loc[integer_series(strict["N_AVAILABLE_CARS"]).ne(2)]
    if not bad_n.empty:
        print("\nSTRICT OCCASIONS WITHOUT EXACTLY TWO ALTERNATIVES")
        print(bad_n.to_string(index=False))
    assert bad_n.empty, "Strict baseline must have exactly two available cars."

    chosen_sum = long.groupby("CHOICE_ID")["chosen"].sum()
    assert chosen_sum.eq(1).all(), "Each CHOICE_ID must have exactly one chosen alternative."
    alt_count = long.groupby("CHOICE_ID")["alternative_A_ID"].count()
    n_available = strict.set_index("CHOICE_ID")["N_AVAILABLE_CARS"].astype(int)
    assert alt_count.equals(n_available.loc[alt_count.index]), "Long row count must equal N_AVAILABLE_CARS."
    chosen_rows = long[long["chosen"].astype(int).eq(1)]
    assert (
        chosen_rows["alternative_A_ID"].astype(int).eq(chosen_rows["CHOSEN_A_ID"].astype(int))
    ).all(), "Chosen alternative must match CHOSEN_A_ID."
    bad_fleet = [
        (str(row.H_ID), int(row.alternative_A_ID))
        for row in long.itertuples(index=False)
        if (str(row.H_ID), int(row.alternative_A_ID)) not in fleet_pairs
    ]
    assert not bad_fleet, f"Long-format alternatives outside fleet: {bad_fleet[:10]}"
    assert alt_count.eq(2).all(), "Strict baseline CHOICE_ID values must have exactly two long rows."
    duplicate_alts = long.groupby(["CHOICE_ID", "alternative_A_ID"]).size().gt(1).any()
    assert not duplicate_alts, "alternative_A_ID must be unique within CHOICE_ID."
    assert long["VALID_STRICT_CHOICE_OCCASION"].astype(int).eq(1).all(), "Long file contains excluded occasions."
    context_cols = [
        col
        for col in long.columns
        if col not in {"alternative_A_ID", "chosen"}
    ]
    inconsistent = long.groupby("CHOICE_ID")[context_cols].nunique(dropna=False).gt(1).any(axis=1)
    assert not inconsistent.any(), "Context variables must be identical within CHOICE_ID."


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
    long: pd.DataFrame,
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

    phase2_metrics = [
        "identifiable household-car driver trips",
        "complete-time identifiable driver trips",
        "incomplete-time identifiable driver trips",
        "events excluded from strict timeline",
        "home-origin choice occasions",
        "invalid snapshot cases",
        "true availability conflicts",
        "unknown snapshot occasions",
        "simultaneous home-departure occasions",
        "same-vehicle simultaneous departures",
    ]
    for metric in phase2_metrics:
        rows.append(qa_row("PHASE 2 RECONSTRUCTION", metric, phase2_qa.get(metric, 0), phase2_qa.get("home-origin choice occasions", initial), "accepted Phase 2 v2 QA"))
    vehicle_events = phase2_qa.get("vehicle departure events", 0) + phase2_qa.get("vehicle arrival events", 0)
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
    rows.append(qa_row("STRICT SAMPLE OUTPUT", "strict long-format rows", len(long), max(len(strict) * 2, 1), "expected two rows per strict choice occasion"))
    add_value_distribution_rows(rows, "STRICT SAMPLE OUTPUT", "distribution of CHOSEN_A_ID", strict["CHOSEN_A_ID"], len(strict), "strict valid choice occasions")
    add_value_distribution_rows(rows, "STRICT SAMPLE OUTPUT", "distribution of N_AVAILABLE_CARS", strict["N_AVAILABLE_CARS"], len(strict), "strict valid choice occasions")

    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "home-origin occasions in rbW-present households", int(classified["rbw_present_flag"].astype(int).sum()), initial, "all home-origin choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "core occasions in rbW-present households", int(core["rbw_present_flag"].astype(int).sum()), max(len(core), 1), "valid core occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "strict occasions in rbW-present households", int(strict["rbw_present_flag"].astype(int).sum()), max(len(strict), 1), "strict valid choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "home-origin occasions in rbW-Pkw-mode households", int(classified["rbw_pkw_mode_flag"].astype(int).sum()), initial, "all home-origin choice occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "core occasions in rbW-Pkw-mode households", int(core["rbw_pkw_mode_flag"].astype(int).sum()), max(len(core), 1), "valid core occasions"))
    rows.append(qa_row("DESCRIPTIVE rbW FLAGS", "strict occasions in rbW-Pkw-mode households", int(strict["rbw_pkw_mode_flag"].astype(int).sum()), max(len(strict), 1), "strict valid choice occasions"))
    return pd.DataFrame(rows)


def print_examples(classified: pd.DataFrame, long: pd.DataFrame) -> None:
    print("\nSTRICT CHOICE_ID EXAMPLES")
    example_cols = [
        "CHOICE_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "AVAILABLE_CAR_SET",
        "N_AVAILABLE_CARS",
        "CHOSEN_A_ID",
        "alternative_A_ID",
        "chosen",
    ]
    print(long[example_cols].head(20).to_string(index=False) if not long.empty else "(none)")

    print("\nEXCLUDED OCCASION EXAMPLES")
    targets = [
        ("invalid snapshot", "EXCL_INVALID_SNAPSHOT"),
        ("unknown snapshot", "EXCL_UNKNOWN_SNAPSHOT"),
        ("availability conflict", "EXCL_AVAILABILITY_CONFLICT"),
        ("one available car", "one_available_case_flag"),
        ("simultaneous departure", "EXCL_SIMULTANEOUS_DEPARTURE"),
        ("household time unresolved", "EXCL_HOUSEHOLD_TIME_UNRESOLVED"),
        ("vehicle overlap", "EXCL_VEHICLE_OVERLAP"),
        ("location-transition conflict", "EXCL_LOCATION_TRANSITION_CONFLICT"),
    ]
    rows = []
    for label, flag in targets:
        if flag not in classified.columns:
            continue
        sample = classified.loc[classified[flag].astype(int).eq(1)].head(2).copy()
        if sample.empty:
            continue
        sample.insert(0, "example_type", label)
        rows.append(sample)
    if rows:
        examples = pd.concat(rows, ignore_index=True)
        cols = [
            "example_type",
            "CHOICE_ID",
            "H_ID",
            "HP_ID",
            "P_ID",
            "W_ID",
            "START_MIN",
            "AVAILABLE_CAR_SET",
            "N_AVAILABLE_CARS",
            "CHOSEN_A_ID",
            "PRIMARY_EXCLUSION_REASON",
            "ALL_EXCLUSION_REASONS",
        ]
        print(examples[[col for col in cols if col in examples.columns]].head(20).to_string(index=False))
    else:
        print("(none)")


def print_final_results(classified: pd.DataFrame, funnel: pd.DataFrame, long: pd.DataFrame, stage_times: dict[str, float]) -> None:
    strict = classified[classified["VALID_STRICT_CHOICE_OCCASION"].eq(1)]
    core = classified[classified["VALID_CORE_OCCASION"].eq(1)]
    print("\nPHASE 3 OUTPUT PATHS")
    print(f"vehicle_choice_occasions_classified: {CLASSIFIED_PATH}")
    print(f"phase3_sample_funnel: {FUNNEL_PATH}")
    print(f"mnl_vehicle_choice_long_base: {LONG_PATH}")
    print(f"vehicle_availability_QA_summary: {QA_PATH}")

    print("\nPHASE 3 FINAL COUNTS")
    print(f"all home-origin occasions: {len(classified):,}")
    print(f"VALID_CORE_OCCASION: {len(core):,}")
    print(f"VALID_STRICT_CHOICE_OCCASION: {len(strict):,}")
    print(f"strict long-format rows: {len(long):,}")
    print(f"strict CHOICE_ID count: {long['CHOICE_ID'].nunique() if not long.empty else 0:,}")
    print(f"strict household count: {strict['H_ID'].nunique():,}")

    print("\nSEQUENTIAL FUNNEL")
    print(funnel.to_string(index=False))

    print("\nRAW EXCLUSION FLAG COUNTS")
    for _, flag in ALL_STEPS:
        print(f"{flag}: {int(classified[flag].sum()):,}")

    print("\nCHOSEN_A_ID DISTRIBUTION BEFORE STRICT")
    print(classified["CHOSEN_A_ID"].value_counts().sort_index().to_string())
    print("\nCHOSEN_A_ID DISTRIBUTION AFTER STRICT")
    print(strict["CHOSEN_A_ID"].value_counts().sort_index().to_string())

    print("\nN_AVAILABLE_CARS DISTRIBUTION BEFORE STRICT")
    print(classified["N_AVAILABLE_CARS"].value_counts().sort_index().to_string())
    print("\nN_AVAILABLE_CARS DISTRIBUTION AFTER STRICT")
    print(strict["N_AVAILABLE_CARS"].value_counts().sort_index().to_string())

    print("\nSTRICT rbW DESCRIPTIVE FLAGS")
    print(f"rbw_present_flag strict occasions: {int(strict['rbw_present_flag'].astype(int).sum()):,}")
    print(f"rbw_pkw_mode_flag strict occasions: {int(strict['rbw_pkw_mode_flag'].astype(int).sum()):,}")

    print("\nRUNTIME BY STAGE")
    for name, seconds in stage_times.items():
        print(f"{name}: {seconds:.2f}s")

    print_examples(classified, long)


def main() -> None:
    total_t0 = perf_counter()
    stage_times: dict[str, float] = {}

    stage_start = perf_counter()
    occasions = read_csv_strings(OCCASIONS_PATH)
    trip_header = pd.read_csv(TRIPS_PATH, nrows=0).columns.tolist()
    trip_cols = ["SOURCE_ROW_ID", *[col for col in KEEP_CONTEXT_COLUMNS if col in trip_header]]
    trips = read_csv_strings(TRIPS_PATH, usecols=trip_cols)
    cars = read_csv_strings(CARS_PATH, usecols=["H_ID", "A_ID"])
    phase1_qa = load_optional_qa(PHASE1_QA_PATH)
    phase2_qa = load_optional_qa(PHASE2_QA_PATH)
    stage_times["loading"] = perf_counter() - stage_start

    print_column_inventory(occasions, trips, cars)

    stage_start = perf_counter()
    fleet_pairs = build_fleet_pairs(cars)
    classified = validate_and_parse_occasions(occasions)
    classified = merge_context(classified, trips)
    classified = create_exclusion_flags(classified, fleet_pairs)
    funnel = build_funnel(classified)
    stage_times["classification and funnel"] = perf_counter() - stage_start

    stage_start = perf_counter()
    strict = classified[classified["VALID_STRICT_CHOICE_OCCASION"].eq(1)].copy()
    long = expand_long(strict, fleet_pairs)
    stage_times["long-format expansion"] = perf_counter() - stage_start

    stage_start = perf_counter()
    qa = build_qa_summary(classified, long, funnel, phase1_qa, phase2_qa)
    classified.to_csv(CLASSIFIED_PATH, index=False)
    funnel.to_csv(FUNNEL_PATH, index=False)
    long.to_csv(LONG_PATH, index=False)
    qa.to_csv(QA_PATH, index=False)
    stage_times["CSV writing"] = perf_counter() - stage_start
    stage_times["total"] = perf_counter() - total_t0

    print_final_results(classified, funnel, long, stage_times)
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
