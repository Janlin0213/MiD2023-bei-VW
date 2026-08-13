from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

TRIPS_PATH = DATA_PROCESSED / "reconstruction" / "phase1" / "trips_home_chain_enriched.csv"
CARS_PATH = DATA_PROCESSED / "selected_raw" / "cars_selected_raw.csv"


OUT_EVENTS_PATH = DATA_PROCESSED /"reconstruction"/"phase2"/"vehicle_events_v2.csv"
OUT_OCCASIONS_PATH = DATA_PROCESSED / "reconstruction" / "phase2" / "vehicle_choice_occasions_all_v2.csv"
OUT_QA_PATH = DATA_PROCESSED / "reconstruction" / "phase2" / "phase2_QA_summary_v2.csv"

VALID_A_IDS = {1, 2, 3}
STATE_HOME = "HOME"
STATE_AWAY = "AWAY"
STATE_UNKNOWN = "UNKNOWN"


TRIP_COLS = [
    "SOURCE_ROW_ID",
    "H_ID",
    "HP_ID",
    "P_ID",
    "W_ID",
    "CHAIN_ELIGIBLE",
    "W_RBW",
    "pkw_fmf",
    "W_WAUTO",
    "W_VM_G",
    "START_MIN",
    "ARRIVAL_MIN",
    "HOME_ORIGIN",
    "HOME_DESTINATION",
    "invalid_start_time_flag",
    "invalid_arrival_time_flag",
    "negative_duration_flag",
    "time_sequence_conflict_flag",
    "complete_fleet_baseline_flag",
    "topcoded_multicar_household_flag",
    "fleet_inventory_mismatch_flag",
    "household_car_count_conflict_flag",
    "nonstandard_A_ID_set_flag",
    "FLEET_CAR_SET",
    "N_RECORDED_CARS",
]


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def integer_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def as_int(value: object, default: int = 0) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return default
    try:
        return int(float(str(value)))
    except ValueError:
        return default


def ordered_set(values: Iterable[int]) -> str:
    return "|".join(str(int(value)) for value in sorted(values))


def parse_car_set(value: object) -> list[int]:
    text = str(value)
    if text == "" or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("|") if part != ""]


def state_string(states: dict[int, str], fleet: list[int]) -> str:
    return "|".join(f"{a_id}:{states.get(a_id, STATE_UNKNOWN)}" for a_id in fleet)


def event_home_value(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return None
    return parsed if parsed in (0, 1) else None


def add_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"{col}_NUM"] = integer_series(out[col])
    return out


def construct_recorded_fleet(cars: pd.DataFrame, retained_households: set[str]) -> pd.DataFrame:
    pairs = cars[["H_ID", "A_ID"]].copy()
    pairs["A_ID_NUM"] = integer_series(pairs["A_ID"])
    pairs = pairs[pairs["H_ID"].isin(retained_households) & pairs["A_ID_NUM"].isin(VALID_A_IDS)]
    pairs = pairs[["H_ID", "A_ID_NUM"]].drop_duplicates()

    rows = []
    for h_id, group in pairs.groupby("H_ID", sort=False):
        ids = sorted(group["A_ID_NUM"].astype(int).unique())
        rows.append({"H_ID": h_id, "FLEET_A_IDS": ids, "FLEET_CAR_SET_REBUILT": ordered_set(ids)})
    return pd.DataFrame(rows)


def fleet_pairs(fleet: pd.DataFrame) -> set[tuple[str, int]]:
    return {
        (str(row.H_ID), int(a_id))
        for row in fleet.itertuples(index=False)
        for a_id in row.FLEET_A_IDS
    }


def prepare_driver_trips(trips: pd.DataFrame, fleet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = fleet_pairs(fleet)
    candidate = (
        trips["CHAIN_ELIGIBLE_NUM"].eq(1)
        & trips["pkw_fmf_NUM"].eq(1)
        & trips["W_WAUTO_NUM"].isin(sorted(VALID_A_IDS))
    )
    in_fleet = pd.Series(
        [
            (str(h_id), int(a_id)) in pairs if not pd.isna(a_id) else False
            for h_id, a_id in zip(trips["H_ID"], trips["W_WAUTO_NUM"])
        ],
        index=trips.index,
    )
    unmatched = trips[candidate & ~in_fleet].copy()
    unmatched["unmatched_reported_vehicle_flag"] = 1

    selected = trips[candidate & in_fleet].copy()
    selected["CHOSEN_A_ID"] = selected["W_WAUTO_NUM"].astype(int)
    selected["unmatched_reported_vehicle_flag"] = 0

    valid_start = selected["invalid_start_time_flag_NUM"].eq(0) & selected["START_MIN_NUM"].notna()
    valid_arrival = selected["invalid_arrival_time_flag_NUM"].eq(0) & selected["ARRIVAL_MIN_NUM"].notna()
    nonnegative = selected["negative_duration_flag_NUM"].eq(0) & selected["ARRIVAL_MIN_NUM"].ge(selected["START_MIN_NUM"])
    selected["complete_vehicle_time_flag"] = (valid_start & valid_arrival & nonnegative).astype(int)
    selected["vehicle_time_history_unresolved_flag"] = selected["complete_vehicle_time_flag"].eq(0).astype(int)
    selected["zero_duration_driver_trip_flag"] = (
        selected["complete_vehicle_time_flag"].eq(1) & selected["START_MIN_NUM"].eq(selected["ARRIVAL_MIN_NUM"])
    ).astype(int)

    household_time_unresolved_hids = set(
        selected.loc[selected["vehicle_time_history_unresolved_flag"].eq(1), "H_ID"].astype(str)
    )
    household_zero_duration_hids = set(
        selected.loc[selected["zero_duration_driver_trip_flag"].eq(1), "H_ID"].astype(str)
    )
    household_timeline_conflict_hids = set(
        selected.loc[selected["time_sequence_conflict_flag_NUM"].eq(1), "H_ID"].astype(str)
    )
    selected["household_vehicle_time_unresolved_flag"] = (
        selected["H_ID"].astype(str).isin(household_time_unresolved_hids).astype(int)
    )
    selected["household_zero_duration_driver_trip_flag"] = (
        selected["H_ID"].astype(str).isin(household_zero_duration_hids).astype(int)
    )
    selected["household_vehicle_timeline_conflict_flag"] = (
        selected["H_ID"].astype(str).isin(household_timeline_conflict_hids).astype(int)
    )

    person_id = np.where(selected["HP_ID"].ne(""), selected["HP_ID"], selected["P_ID"])
    selected["TRIP_ID_BASE"] = selected["H_ID"] + "_" + pd.Series(person_id, index=selected.index) + "_" + selected["W_ID"]
    selected["duplicate_trip_identifier_flag"] = selected["TRIP_ID_BASE"].duplicated(keep=False).astype(int)
    selected["TRIP_ID"] = selected["TRIP_ID_BASE"]
    duplicate = selected["duplicate_trip_identifier_flag"].eq(1)
    selected.loc[duplicate, "TRIP_ID"] = (
        selected.loc[duplicate, "TRIP_ID_BASE"] + "_" + selected.loc[duplicate, "SOURCE_ROW_ID"].astype(str)
    )
    selected["CHOICE_ID"] = selected["TRIP_ID"]
    return selected, unmatched


def add_interval_diagnostics(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = selected.copy()
    out["vehicle_trip_overlap_flag"] = 0
    out["vehicle_location_transition_conflict_flag"] = 0

    complete = out[out["complete_vehicle_time_flag"].eq(1)].copy()
    complete = complete.sort_values(["H_ID", "CHOSEN_A_ID", "START_MIN_NUM", "ARRIVAL_MIN_NUM", "SOURCE_ROW_ID_NUM"], kind="mergesort")

    for (_, _), group in complete.groupby(["H_ID", "CHOSEN_A_ID"], sort=False):
        max_prev_arrival: float | None = None
        prev_destination: int | None = None
        for row in group.itertuples():
            idx = row.Index
            start = float(row.START_MIN_NUM)
            arrival = float(row.ARRIVAL_MIN_NUM)
            if max_prev_arrival is not None and start < max_prev_arrival:
                out.at[idx, "vehicle_trip_overlap_flag"] = 1
            max_prev_arrival = arrival if max_prev_arrival is None else max(max_prev_arrival, arrival)

            current_origin = event_home_value(row.HOME_ORIGIN)
            if prev_destination is not None and current_origin is not None and prev_destination != current_origin:
                out.at[idx, "vehicle_location_transition_conflict_flag"] = 1
            current_destination = event_home_value(row.HOME_DESTINATION)
            if current_destination is not None:
                prev_destination = current_destination

    household_flags = out.groupby("H_ID", sort=False)[
        [
            "vehicle_trip_overlap_flag",
            "vehicle_location_transition_conflict_flag",
            "household_vehicle_time_unresolved_flag",
            "household_zero_duration_driver_trip_flag",
            "household_vehicle_timeline_conflict_flag",
        ]
    ].max().reset_index()
    household_flags = household_flags.rename(
        columns={
            "vehicle_trip_overlap_flag": "household_vehicle_trip_overlap_flag",
            "vehicle_location_transition_conflict_flag": "household_vehicle_location_transition_conflict_flag",
        }
    )
    out = out.drop(
        columns=[
            "household_vehicle_time_unresolved_flag",
            "household_zero_duration_driver_trip_flag",
            "household_vehicle_timeline_conflict_flag",
        ]
    ).merge(household_flags, on="H_ID", how="left", validate="many_to_one")
    return out, household_flags


def add_rbw_flags(trips: pd.DataFrame, households: pd.Series) -> pd.DataFrame:
    household_set = set(households.astype(str))
    rbw = trips[trips["W_RBW_NUM"].eq(1) & trips["H_ID"].astype(str).isin(household_set)].copy()
    if rbw.empty:
        return pd.DataFrame(columns=["H_ID", "rbw_present_flag", "rbw_pkw_mode_flag"])
    flags = rbw.groupby("H_ID", sort=False).agg(
        rbw_present_flag=("W_RBW", lambda _: 1),
        rbw_pkw_mode_flag=("W_VM_G", lambda s: int(integer_series(s).eq(1).any())),
    ).reset_index()
    return flags


def create_vehicle_events(selected: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "TRIP_ID",
        "CHOICE_ID",
        "SOURCE_ROW_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "CHOSEN_A_ID",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "START_MIN",
        "ARRIVAL_MIN",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "negative_duration_flag",
        "complete_vehicle_time_flag",
        "zero_duration_driver_trip_flag",
        "vehicle_time_history_unresolved_flag",
        "household_vehicle_time_unresolved_flag",
        "household_zero_duration_driver_trip_flag",
        "vehicle_trip_overlap_flag",
        "household_vehicle_trip_overlap_flag",
        "vehicle_location_transition_conflict_flag",
        "household_vehicle_location_transition_conflict_flag",
        "household_vehicle_timeline_conflict_flag",
        "time_sequence_conflict_flag",
        "unmatched_reported_vehicle_flag",
        "duplicate_trip_identifier_flag",
        "complete_fleet_baseline_flag",
        "topcoded_multicar_household_flag",
        "fleet_inventory_mismatch_flag",
        "household_car_count_conflict_flag",
        "nonstandard_A_ID_set_flag",
        "rbw_present_flag",
        "rbw_pkw_mode_flag",
        "FLEET_CAR_SET",
        "N_RECORDED_CARS",
    ]
    base_cols = [col for col in base_cols if col in selected.columns]

    departures = selected[base_cols].copy()
    departures["A_ID"] = selected["CHOSEN_A_ID"].astype(int)
    departures["event_type"] = "DEPARTURE"
    departures["event_time"] = selected["START_MIN_NUM"]
    departures["event_home_status"] = selected["HOME_ORIGIN_NUM"]
    departures["invalid_event_time_flag"] = selected["invalid_start_time_flag_NUM"].ne(0).astype(int)

    arrivals = selected[base_cols].copy()
    arrivals["A_ID"] = selected["CHOSEN_A_ID"].astype(int)
    arrivals["event_type"] = "ARRIVAL"
    arrivals["event_time"] = selected["ARRIVAL_MIN_NUM"]
    arrivals["event_home_status"] = selected["HOME_DESTINATION_NUM"]
    arrivals["invalid_event_time_flag"] = selected["invalid_arrival_time_flag_NUM"].ne(0).astype(int)

    events = pd.concat([departures, arrivals], ignore_index=True)
    events["event_time"] = events["event_time"].astype("Int64")
    events["event_home_status"] = events["event_home_status"].astype("Int64")
    events["event_usable_for_state_flag"] = events["complete_vehicle_time_flag"].astype(int)
    events["event_excluded_from_strict_timeline_flag"] = events["event_usable_for_state_flag"].eq(0).astype(int)
    events["arrival_while_home_flag"] = 0
    events["departure_while_away_flag"] = 0
    events["vehicle_event_conflict_flag"] = 0
    events["state_before_processing"] = ""
    events["state_after_processing"] = ""
    events["AVAILABLE_CAR_SET"] = ""
    events["N_AVAILABLE_CARS"] = 0
    events["unknown_vehicle_state_at_snapshot_flag"] = 0
    events["event_type_order"] = events["event_type"].map({"ARRIVAL": 0, "DEPARTURE": 1})
    return events.sort_values(
        ["H_ID", "event_time", "event_type_order", "SOURCE_ROW_ID", "A_ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def initial_vehicle_states(events: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    strict = events[events["event_usable_for_state_flag"].eq(1)].copy()
    strict = strict.sort_values(["H_ID", "A_ID", "event_time", "event_type_order", "SOURCE_ROW_ID"], kind="mergesort")
    first_by_pair = {
        (str(h_id), int(a_id)): group.iloc[0].to_dict()
        for (h_id, a_id), group in strict.groupby(["H_ID", "A_ID"], sort=False)
    }
    event_pairs = set(zip(events["H_ID"].astype(str), integer_series(events["A_ID"]).astype(int)))
    unresolved_pairs = set(
        zip(
            events.loc[events["vehicle_time_history_unresolved_flag"].eq(1), "H_ID"].astype(str),
            integer_series(events.loc[events["vehicle_time_history_unresolved_flag"].eq(1), "A_ID"]).astype(int),
        )
    )

    rows = []
    for fleet_row in fleet.itertuples(index=False):
        h_id = str(fleet_row.H_ID)
        for a_id in fleet_row.FLEET_A_IDS:
            pair = (h_id, int(a_id))
            first = first_by_pair.get(pair)
            initial_home_assumption_flag = 0
            if first is None:
                if pair in event_pairs:
                    initial_state = STATE_UNKNOWN
                    initial_unresolved = 1
                else:
                    initial_state = STATE_HOME
                    initial_unresolved = 0
                    initial_home_assumption_flag = 1
            elif first["event_type"] == "DEPARTURE" and event_home_value(first["HOME_ORIGIN"]) == 1:
                initial_state = STATE_HOME
                initial_unresolved = 0
            else:
                initial_state = STATE_UNKNOWN
                initial_unresolved = 1

            rows.append(
                {
                    "H_ID": h_id,
                    "A_ID": int(a_id),
                    "initial_state": initial_state,
                    "initial_state_unresolved_flag": initial_unresolved,
                    "initial_home_assumption_flag": initial_home_assumption_flag,
                    "vehicle_time_history_unresolved_flag": int(pair in unresolved_pairs),
                }
            )
    return pd.DataFrame(rows)


def compact_trip_lookup(selected: pd.DataFrame) -> dict[str, dict]:
    cols = [
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
        "HOME_DESTINATION",
        "CHOSEN_A_ID",
        "complete_vehicle_time_flag",
        "zero_duration_driver_trip_flag",
        "vehicle_time_history_unresolved_flag",
        "household_vehicle_time_unresolved_flag",
        "household_zero_duration_driver_trip_flag",
        "vehicle_trip_overlap_flag",
        "household_vehicle_trip_overlap_flag",
        "vehicle_location_transition_conflict_flag",
        "household_vehicle_location_transition_conflict_flag",
        "household_vehicle_timeline_conflict_flag",
        "time_sequence_conflict_flag",
        "unmatched_reported_vehicle_flag",
        "duplicate_trip_identifier_flag",
        "complete_fleet_baseline_flag",
        "topcoded_multicar_household_flag",
        "fleet_inventory_mismatch_flag",
        "household_car_count_conflict_flag",
        "nonstandard_A_ID_set_flag",
        "rbw_present_flag",
        "rbw_pkw_mode_flag",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "negative_duration_flag",
        "FLEET_CAR_SET",
        "N_RECORDED_CARS",
    ]
    return selected.assign(SOURCE_ROW_ID_KEY=selected["SOURCE_ROW_ID"].astype(str))[
        ["SOURCE_ROW_ID_KEY", *cols]
    ].set_index("SOURCE_ROW_ID_KEY").to_dict("index")


def build_occasion_row(
    trip: dict,
    available_set: str,
    n_available: int,
    unknown_snapshot: int,
    initial_state_unresolved: int,
    simultaneous: int,
    same_vehicle_simultaneous: int,
    snapshot_valid: int,
) -> dict:
    chosen = int(trip["CHOSEN_A_ID"])
    available = set(parse_car_set(available_set))
    availability_conflict = int(snapshot_valid == 1 and chosen not in available)
    return {
        "CHOICE_ID": trip["CHOICE_ID"],
        "TRIP_ID": trip["TRIP_ID"],
        "SOURCE_ROW_ID": trip["SOURCE_ROW_ID"],
        "H_ID": trip["H_ID"],
        "HP_ID": trip["HP_ID"],
        "P_ID": trip["P_ID"],
        "W_ID": trip["W_ID"],
        "START_MIN": trip["START_MIN"],
        "ARRIVAL_MIN": trip["ARRIVAL_MIN"],
        "HOME_ORIGIN": trip["HOME_ORIGIN"],
        "HOME_DESTINATION": trip["HOME_DESTINATION"],
        "CHOSEN_A_ID": chosen,
        "AVAILABLE_CAR_SET": available_set,
        "N_AVAILABLE_CARS": int(n_available),
        "snapshot_valid_flag": int(snapshot_valid),
        "invalid_snapshot_flag": int(snapshot_valid == 0),
        "unknown_vehicle_state_at_snapshot_flag": int(unknown_snapshot),
        "initial_state_unresolved_flag": int(initial_state_unresolved),
        "unmatched_reported_vehicle_flag": int(trip["unmatched_reported_vehicle_flag"]),
        "duplicate_trip_identifier_flag": int(trip["duplicate_trip_identifier_flag"]),
        "complete_vehicle_time_flag": int(trip["complete_vehicle_time_flag"]),
        "vehicle_time_history_unresolved_flag": int(trip["vehicle_time_history_unresolved_flag"]),
        "household_vehicle_time_unresolved_flag": int(trip["household_vehicle_time_unresolved_flag"]),
        "zero_duration_driver_trip_flag": int(trip["zero_duration_driver_trip_flag"]),
        "household_zero_duration_driver_trip_flag": int(trip["household_zero_duration_driver_trip_flag"]),
        "vehicle_trip_overlap_flag": int(trip["vehicle_trip_overlap_flag"]),
        "household_vehicle_trip_overlap_flag": int(trip["household_vehicle_trip_overlap_flag"]),
        "vehicle_location_transition_conflict_flag": int(trip["vehicle_location_transition_conflict_flag"]),
        "household_vehicle_location_transition_conflict_flag": int(trip["household_vehicle_location_transition_conflict_flag"]),
        "household_vehicle_timeline_conflict_flag": int(trip["household_vehicle_timeline_conflict_flag"]),
        "availability_conflict_flag": availability_conflict,
        "zero_available_case_flag": int(snapshot_valid == 1 and n_available == 0),
        "one_available_case_flag": int(snapshot_valid == 1 and n_available == 1),
        "two_available_case_flag": int(snapshot_valid == 1 and n_available == 2),
        "three_available_case_flag": int(snapshot_valid == 1 and n_available == 3),
        "simultaneous_home_departure_flag": int(simultaneous),
        "same_vehicle_simultaneous_departure_flag": int(same_vehicle_simultaneous),
        "complete_fleet_baseline_flag": int(trip["complete_fleet_baseline_flag"]),
        "topcoded_multicar_household_flag": int(trip["topcoded_multicar_household_flag"]),
        "fleet_inventory_mismatch_flag": int(trip["fleet_inventory_mismatch_flag"]),
        "household_car_count_conflict_flag": int(trip["household_car_count_conflict_flag"]),
        "nonstandard_A_ID_set_flag": int(trip["nonstandard_A_ID_set_flag"]),
        "rbw_present_flag": int(trip["rbw_present_flag"]),
        "rbw_pkw_mode_flag": int(trip["rbw_pkw_mode_flag"]),
        "time_sequence_conflict_flag": int(trip["time_sequence_conflict_flag"]),
        "invalid_start_time_flag": int(trip["invalid_start_time_flag"]),
        "invalid_arrival_time_flag": int(trip["invalid_arrival_time_flag"]),
        "negative_duration_flag": int(trip["negative_duration_flag"]),
        "FLEET_CAR_SET": trip["FLEET_CAR_SET"],
        "N_RECORDED_CARS": trip["N_RECORDED_CARS"],
    }


def process_household_timelines(
    events: pd.DataFrame,
    selected: pd.DataFrame,
    fleet: pd.DataFrame,
    initial_states: pd.DataFrame,
    qa_households: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = perf_counter()
    trip_lookup = compact_trip_lookup(selected)
    fleet_by_h = {str(row.H_ID): list(row.FLEET_A_IDS) for row in fleet.itertuples(index=False)}
    init_by_h = {
        str(h_id): group.set_index("A_ID").to_dict("index")
        for h_id, group in initial_states.groupby("H_ID", sort=False)
    }

    strict = events[events["event_usable_for_state_flag"].eq(1)].copy()
    strict = strict.sort_values(["H_ID", "event_time", "event_type_order", "SOURCE_ROW_ID", "A_ID"], kind="mergesort")
    first_home_departure_keys: set[tuple[str, int, int]] = set()
    same_timestamp_first_event_anomalies: list[dict] = []
    for (first_h_id, first_a_id), first_group in strict.groupby(["H_ID", "A_ID"], sort=False):
        first_time = first_group["event_time"].min()
        first_time_events = first_group[first_group["event_time"].eq(first_time)]
        first_event = first_time_events.iloc[0]
        key = (str(first_h_id), int(first_a_id), int(first_time))
        if len(first_time_events) > 1:
            same_timestamp_first_event_anomalies.append(
                {
                    "H_ID": str(first_h_id),
                    "A_ID": int(first_a_id),
                    "event_time": int(first_time),
                    "n_first_timestamp_events": int(len(first_time_events)),
                    "event_types": "|".join(first_time_events["event_type"].astype(str)),
                    "trip_ids": "|".join(first_time_events["TRIP_ID"].astype(str)),
                }
            )
            continue
        if first_event["event_type"] == "DEPARTURE" and event_home_value(first_event["HOME_ORIGIN"]) == 1:
            first_home_departure_keys.add(key)

    event_updates: dict[int, dict] = {}
    occasion_rows: list[dict] = []
    trace_rows: list[dict] = []
    first_event_violations: list[dict] = []

    total_households = strict["H_ID"].nunique()
    for household_i, (h_id_value, h_events) in enumerate(strict.groupby("H_ID", sort=False), start=1):
        if household_i % 5000 == 0:
            print(f"processed {household_i:,}/{total_households:,} households in {perf_counter() - t0:.1f}s", flush=True)
        h_id = str(h_id_value)
        fleet_ids = fleet_by_h.get(h_id, [])
        if not fleet_ids:
            continue
        init = init_by_h.get(h_id, {})
        states = {a_id: init.get(a_id, {}).get("initial_state", STATE_UNKNOWN) for a_id in fleet_ids}
        initial_unresolved_any = int(any(init.get(a_id, {}).get("initial_state_unresolved_flag", 1) == 1 for a_id in fleet_ids))
        trace_enabled = h_id in qa_households
        seen_vehicle_event: set[int] = set()

        for event_time, time_group in h_events.groupby("event_time", sort=True):
            arrivals = []
            departures = []
            for event in time_group.itertuples():
                if event.event_type == "ARRIVAL":
                    arrivals.append(event)
                else:
                    departures.append(event)
            arrivals.sort(key=lambda event: (as_int(event.SOURCE_ROW_ID), as_int(event.A_ID)))
            departures.sort(key=lambda event: (as_int(event.SOURCE_ROW_ID), as_int(event.A_ID)))

            for event in arrivals:
                a_id = int(event.A_ID)
                before = state_string(states, fleet_ids)
                arrival_while_home = int(states.get(a_id, STATE_UNKNOWN) == STATE_HOME)
                destination = event_home_value(event.HOME_DESTINATION)
                if destination == 1:
                    states[a_id] = STATE_HOME
                elif destination == 0:
                    states[a_id] = STATE_AWAY
                else:
                    states[a_id] = STATE_UNKNOWN
                seen_vehicle_event.add(a_id)
                after = state_string(states, fleet_ids)
                event_updates[event.Index] = {
                    "arrival_while_home_flag": arrival_while_home,
                    "departure_while_away_flag": 0,
                    "vehicle_event_conflict_flag": arrival_while_home,
                    "state_before_processing": before,
                    "state_after_processing": after,
                }
                if trace_enabled:
                    trace_rows.append(trace_row_from_event(event, before, after))

            if departures:
                available = [a_id for a_id in fleet_ids if states.get(a_id) == STATE_HOME]
                available_set = ordered_set(available)
                unknown_snapshot = int(any(states.get(a_id) == STATE_UNKNOWN for a_id in fleet_ids))
                home_departures = [event for event in departures if event_home_value(event.HOME_ORIGIN) == 1]
                simultaneous = int(len({event.SOURCE_ROW_ID for event in home_departures}) >= 2 and len({event.HP_ID for event in home_departures}) >= 2)
                same_vehicle_simultaneous = int(simultaneous == 1 and len({event.A_ID for event in home_departures}) < len(home_departures))

                if trace_enabled:
                    snapshot_state = state_string(states, fleet_ids)
                    trace_rows.append(
                        {
                            "H_ID": h_id,
                            "event_time": int(event_time),
                            "event_type": "SNAPSHOT",
                            "TRIP_ID": "",
                            "HP_ID": "",
                            "A_ID": "",
                            "HOME_ORIGIN": "",
                            "HOME_DESTINATION": "",
                            "state_before_processing": snapshot_state,
                            "state_after_processing": snapshot_state,
                            "AVAILABLE_CAR_SET": available_set,
                        }
                    )

                for event in home_departures:
                    trip = trip_lookup[str(event.SOURCE_ROW_ID)]
                    occasion_rows.append(
                        build_occasion_row(
                            trip,
                            available_set,
                            len(available),
                            unknown_snapshot,
                            initial_unresolved_any,
                            simultaneous,
                            same_vehicle_simultaneous,
                            snapshot_valid=1,
                        )
                    )
                    first_home_departure_key = (h_id, int(event.A_ID), int(event_time))
                    if first_home_departure_key in first_home_departure_keys and states.get(int(event.A_ID)) != STATE_HOME:
                        first_event_violations.append(
                            {
                                "H_ID": h_id,
                                "A_ID": int(event.A_ID),
                                "TRIP_ID": event.TRIP_ID,
                                "event_time": int(event_time),
                                "state_before_processing": state_string(states, fleet_ids),
                            }
                        )

            for event in departures:
                a_id = int(event.A_ID)
                before = state_string(states, fleet_ids)
                departure_while_away = int(states.get(a_id, STATE_UNKNOWN) == STATE_AWAY)
                states[a_id] = STATE_AWAY
                seen_vehicle_event.add(a_id)
                after = state_string(states, fleet_ids)
                event_updates[event.Index] = {
                    "arrival_while_home_flag": 0,
                    "departure_while_away_flag": departure_while_away,
                    "vehicle_event_conflict_flag": 0,
                    "state_before_processing": before,
                    "state_after_processing": after,
                }
                if trace_enabled:
                    trace_rows.append(trace_row_from_event(event, before, after))

    incomplete_home = selected[
        selected["complete_vehicle_time_flag"].eq(0) & selected["HOME_ORIGIN_NUM"].eq(1)
    ].copy()
    for trip in incomplete_home.itertuples():
        occasion_rows.append(
            build_occasion_row(
                trip._asdict(),
                "",
                0,
                unknown_snapshot=1,
                initial_state_unresolved=1,
                simultaneous=0,
                same_vehicle_simultaneous=0,
                snapshot_valid=0,
            )
        )

    processed_events = events.copy()
    for col in [
        "arrival_while_home_flag",
        "departure_while_away_flag",
        "vehicle_event_conflict_flag",
        "state_before_processing",
        "state_after_processing",
    ]:
        if col not in processed_events.columns:
            processed_events[col] = "" if col.startswith("state") else 0
    for idx, update in event_updates.items():
        for col, value in update.items():
            processed_events.at[idx, col] = value
    excluded = processed_events["event_usable_for_state_flag"].eq(0)
    processed_events.loc[excluded, "state_before_processing"] = STATE_UNKNOWN
    processed_events.loc[excluded, "state_after_processing"] = STATE_UNKNOWN
    processed_events.loc[excluded, "vehicle_event_conflict_flag"] = 0
    processed_events["vehicle_event_conflict_flag"] = processed_events["arrival_while_home_flag"].astype(int)

    occasions = pd.DataFrame(occasion_rows)
    trace = pd.DataFrame(trace_rows)
    violations = pd.DataFrame(first_event_violations)
    if same_timestamp_first_event_anomalies:
        anomaly_trace = pd.DataFrame(same_timestamp_first_event_anomalies)
        if trace.empty:
            trace = anomaly_trace
        else:
            trace = pd.concat([trace, anomaly_trace], ignore_index=True, sort=False)
    return processed_events, occasions, trace, violations


def trace_row_from_event(event: tuple, before: str, after: str) -> dict:
    return {
        "H_ID": event.H_ID,
        "event_time": int(event.event_time),
        "event_type": event.event_type,
        "TRIP_ID": event.TRIP_ID,
        "HP_ID": event.HP_ID,
        "A_ID": event.A_ID,
        "HOME_ORIGIN": event.HOME_ORIGIN,
        "HOME_DESTINATION": event.HOME_DESTINATION,
        "state_before_processing": before,
        "state_after_processing": after,
        "AVAILABLE_CAR_SET": "",
    }


def select_qa_households(selected: pd.DataFrame) -> set[str]:
    chosen: list[str] = []
    groups = [
        selected.loc[selected["vehicle_time_history_unresolved_flag"].eq(1), "H_ID"],
        selected.loc[selected["vehicle_trip_overlap_flag"].eq(1), "H_ID"],
        selected.loc[selected["vehicle_location_transition_conflict_flag"].eq(1), "H_ID"],
        selected.loc[selected["zero_duration_driver_trip_flag"].eq(1), "H_ID"],
        selected.loc[selected["rbw_pkw_mode_flag"].eq(1), "H_ID"],
        selected["H_ID"],
    ]
    for series in groups:
        for h_id in series.astype(str).drop_duplicates():
            if h_id not in chosen:
                chosen.append(h_id)
            if len(chosen) >= 30:
                return set(chosen)
    return set(chosen)


def rbw_descriptive_counts(trips: pd.DataFrame, selected_households: set[str]) -> dict[str, int]:
    rbw = trips[trips["W_RBW_NUM"].eq(1) & trips["H_ID"].astype(str).isin(selected_households)]
    rbw_pkw = rbw[rbw["W_VM_G_NUM"].eq(1)]
    complete_two = trips[trips["complete_fleet_baseline_flag_NUM"].eq(1)]
    complete_two_hids = set(complete_two["H_ID"].astype(str))
    return {
        "unique rbW-Pkw rows": int(len(rbw_pkw)),
        "unique persons with rbW-Pkw records": int(rbw_pkw["HP_ID"].replace("", pd.NA).nunique()),
        "unique households with rbW-Pkw records": int(rbw_pkw["H_ID"].nunique()),
        "complete two-car households with rbW-Pkw records": int(rbw_pkw["H_ID"].astype(str).isin(complete_two_hids).groupby(rbw_pkw["H_ID"]).any().sum()) if not rbw_pkw.empty else 0,
    }


def build_qa_summary(
    trips: pd.DataFrame,
    selected: pd.DataFrame,
    unmatched: pd.DataFrame,
    events: pd.DataFrame,
    occasions: pd.DataFrame,
    initial_states: pd.DataFrame,
    first_event_violations: pd.DataFrame,
    stage_times: dict[str, float],
) -> pd.DataFrame:
    rbw_counts = rbw_descriptive_counts(trips, set(occasions["H_ID"].astype(str)) if not occasions.empty else set())
    rbw_pkw_hids = set(
        trips.loc[trips["W_RBW_NUM"].eq(1) & trips["W_VM_G_NUM"].eq(1), "H_ID"].astype(str)
    )
    metrics = [
        ("identifiable household-car driver trips", len(selected), max(len(selected), 1)),
        ("complete-time identifiable driver trips", int(selected["complete_vehicle_time_flag"].sum()), max(len(selected), 1)),
        ("incomplete-time identifiable driver trips", int(selected["vehicle_time_history_unresolved_flag"].sum()), max(len(selected), 1)),
        ("unmatched reported household vehicles", len(unmatched), max(len(selected) + len(unmatched), 1)),
        ("vehicle departure events", int(events["event_type"].eq("DEPARTURE").sum()), max(len(events), 1)),
        ("vehicle arrival events", int(events["event_type"].eq("ARRIVAL").sum()), max(len(events), 1)),
        ("events excluded from strict timeline", int(events["event_excluded_from_strict_timeline_flag"].sum()), max(len(events), 1)),
        ("home-origin choice occasions", len(occasions), max(len(selected), 1)),
        ("valid snapshot occasions", int(occasions["snapshot_valid_flag"].sum()), max(len(occasions), 1)),
        ("invalid snapshot cases", int(occasions["invalid_snapshot_flag"].sum()), max(len(occasions), 1)),
        ("true availability conflicts", int(occasions["availability_conflict_flag"].sum()), max(len(occasions), 1)),
        ("unknown snapshot occasions", int(occasions["unknown_vehicle_state_at_snapshot_flag"].sum()), max(len(occasions), 1)),
        ("household-time-unresolved occasions", int(occasions["household_vehicle_time_unresolved_flag"].sum()), max(len(occasions), 1)),
        ("vehicle overlap households", int(selected.loc[selected["household_vehicle_trip_overlap_flag"].eq(1), "H_ID"].nunique()), max(trips["H_ID"].nunique(), 1)),
        ("vehicle overlap occasions", int(occasions["household_vehicle_trip_overlap_flag"].sum()), max(len(occasions), 1)),
        ("location-transition-conflict households", int(selected.loc[selected["household_vehicle_location_transition_conflict_flag"].eq(1), "H_ID"].nunique()), max(trips["H_ID"].nunique(), 1)),
        ("location-transition-conflict occasions", int(occasions["household_vehicle_location_transition_conflict_flag"].sum()), max(len(occasions), 1)),
        ("zero-duration driver trips", int(selected["zero_duration_driver_trip_flag"].sum()), max(len(selected), 1)),
        ("zero-duration household occasions", int(occasions["household_zero_duration_driver_trip_flag"].sum()), max(len(occasions), 1)),
        ("arrival-while-home conflicts", int(events["arrival_while_home_flag"].sum()), max(len(events), 1)),
        ("departure-while-away diagnostic events", int(events["departure_while_away_flag"].sum()), max(len(events), 1)),
        ("simultaneous home-departure occasions", int(occasions["simultaneous_home_departure_flag"].sum()), max(len(occasions), 1)),
        ("same-vehicle simultaneous departures", int(occasions["same_vehicle_simultaneous_departure_flag"].sum()), max(len(occasions), 1)),
        ("initial-state-unresolved vehicles", int(initial_states["initial_state_unresolved_flag"].sum()), max(len(initial_states), 1)),
        ("vehicles with no observed driver events", int(initial_states["initial_home_assumption_flag"].sum()), max(len(initial_states), 1)),
        ("first-event assertion violations", len(first_event_violations), max(len(selected), 1)),
        ("rbW-Pkw-unresolved households", rbw_counts["unique households with rbW-Pkw records"], max(trips["H_ID"].nunique(), 1)),
        ("rbW-Pkw-unresolved occasions", int(occasions["H_ID"].astype(str).isin(rbw_pkw_hids).sum()), max(len(occasions), 1)),
        *[(key, value, max(value, 1)) for key, value in rbw_counts.items()],
        *[(f"runtime {key} seconds", int(round(value)), max(int(round(value)), 1)) for key, value in stage_times.items()],
    ]
    return pd.DataFrame(
        [{"metric": metric, "count": int(count), "share": count / denominator} for metric, count, denominator in metrics]
    )

def assert_phase2_v2(selected: pd.DataFrame, events: pd.DataFrame, occasions: pd.DataFrame, fleet: pd.DataFrame, first_violations: pd.DataFrame) -> None:
    counts = events.groupby(["TRIP_ID", "event_type"], sort=False).size().unstack(fill_value=0)
    assert (counts.get("DEPARTURE", 0) == 1).all(), "Each selected trip needs one departure event."
    assert (counts.get("ARRIVAL", 0) == 1).all(), "Each selected trip needs one arrival event."
    assert counts.sum(axis=1).eq(2).all(), "Each selected trip needs exactly two events."
    assert set(counts.index) == set(selected["TRIP_ID"]), "Event TRIP_ID set mismatch."
    bad_atomic = events.loc[
        events["complete_vehicle_time_flag"].eq(0) & events["event_usable_for_state_flag"].ne(0)
    ]
    assert bad_atomic.empty, "Incomplete-time events must not be usable for state."
    bad_conflicts = occasions.loc[
        occasions["snapshot_valid_flag"].eq(0) & occasions["availability_conflict_flag"].ne(0)
    ]
    assert bad_conflicts.empty, "Invalid snapshots must not be availability conflicts."
    if not occasions.empty:
        for row in occasions.itertuples(index=False):
            available = parse_car_set(row.AVAILABLE_CAR_SET)
            assert available == sorted(available), "AVAILABLE_CAR_SET is not deterministic."
            assert int(row.N_AVAILABLE_CARS) == len(available), "N_AVAILABLE_CARS mismatch."
        simultaneous = occasions[occasions["simultaneous_home_departure_flag"].eq(1)]
        inconsistent = simultaneous.groupby(["H_ID", "START_MIN"], sort=False)["AVAILABLE_CAR_SET"].nunique().gt(1).sum()
        assert inconsistent == 0, "Simultaneous departures did not share a snapshot."


def main() -> None:
    total_t0 = perf_counter()
    stage_start = perf_counter()
    trips = read_csv_strings(TRIPS_PATH, usecols=TRIP_COLS)
    trips = add_numeric_columns(
        trips,
        [
            "SOURCE_ROW_ID",
            "W_ID",
            "CHAIN_ELIGIBLE",
            "W_RBW",
            "pkw_fmf",
            "W_WAUTO",
            "W_VM_G",
            "START_MIN",
            "ARRIVAL_MIN",
            "HOME_ORIGIN",
            "HOME_DESTINATION",
            "invalid_start_time_flag",
            "invalid_arrival_time_flag",
            "negative_duration_flag",
            "time_sequence_conflict_flag",
            "complete_fleet_baseline_flag",
            "topcoded_multicar_household_flag",
            "fleet_inventory_mismatch_flag",
            "household_car_count_conflict_flag",
            "nonstandard_A_ID_set_flag",
        ],
    )
    cars = read_csv_strings(CARS_PATH, usecols=["H_ID", "A_ID"])
    stage_times = {"loading": perf_counter() - stage_start}

    stage_start = perf_counter()
    fleet = construct_recorded_fleet(cars, set(trips["H_ID"].astype(str).unique()))
    selected, unmatched = prepare_driver_trips(trips, fleet)
    selected, household_flags = add_interval_diagnostics(selected)
    rbw_flags = add_rbw_flags(trips, selected["H_ID"])
    selected = selected.merge(rbw_flags, on="H_ID", how="left")
    selected[["rbw_present_flag", "rbw_pkw_mode_flag"]] = selected[["rbw_present_flag", "rbw_pkw_mode_flag"]].fillna(0).astype(int)
    selected = selected.merge(household_flags, on="H_ID", how="left", suffixes=("", "_HH"))
    for col in [
        "household_vehicle_time_unresolved_flag",
        "household_zero_duration_driver_trip_flag",
        "household_vehicle_timeline_conflict_flag",
        "household_vehicle_trip_overlap_flag",
        "household_vehicle_location_transition_conflict_flag",
    ]:
        selected[col] = selected[col].fillna(0).astype(int)
    stage_times["driver-trip filtering"] = perf_counter() - stage_start

    stage_start = perf_counter()
    events = create_vehicle_events(selected)
    stage_times["event creation"] = perf_counter() - stage_start

    stage_start = perf_counter()
    events = events.sort_values(["H_ID", "event_time", "event_type_order", "SOURCE_ROW_ID", "A_ID"], kind="mergesort", na_position="last")
    stage_times["event sorting"] = perf_counter() - stage_start

    stage_start = perf_counter()
    initial_states = initial_vehicle_states(events, fleet)
    qa_households = select_qa_households(selected)
    processed_events, occasions, trace, first_violations = process_household_timelines(events, selected, fleet, initial_states, qa_households)
    stage_times["household state processing"] = perf_counter() - stage_start

    if not occasions.empty:
        occasions = occasions.sort_values(["H_ID", "START_MIN", "SOURCE_ROW_ID"], kind="mergesort", na_position="last").reset_index(drop=True)
    processed_events = processed_events.drop(columns=["event_type_order"], errors="ignore").sort_values(
        ["H_ID", "event_time", "event_type", "SOURCE_ROW_ID", "A_ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    assert_phase2_v2(selected, processed_events, occasions, fleet, first_violations)

    qa = build_qa_summary(trips, selected, unmatched, processed_events, occasions, initial_states, first_violations, stage_times)

    stage_start = perf_counter()
    processed_events.to_csv(OUT_EVENTS_PATH, index=False)
    occasions.to_csv(OUT_OCCASIONS_PATH, index=False)
    qa.to_csv(OUT_QA_PATH, index=False)
    stage_times["CSV writing"] = perf_counter() - stage_start
    stage_times["total"] = perf_counter() - total_t0

    print("\nPHASE 2 V2 QA SUMMARY")
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(qa.to_string(index=False, formatters={"share": "{:.6f}".format}))

    if not first_violations.empty:
        print("\nFIRST-EVENT ASSERTION VIOLATIONS")
        print(first_violations.to_string(index=False))
    else:
        print("\nFIRST-EVENT ASSERTION VIOLATIONS: none")

    print("\nSELECTED QA TRACE HOUSEHOLDS")
    if trace.empty:
        print("(no traces selected)")
    else:
        with pd.option_context("display.max_rows", 120, "display.width", 220, "display.max_columns", None):
            print(trace.head(120).to_string(index=False))

    print("\nPHASE 2 V2 OUTPUT PATHS")
    print(f"vehicle_events_v2: {OUT_EVENTS_PATH}")
    print(f"vehicle_choice_occasions_all_v2: {OUT_OCCASIONS_PATH}")
    print(f"phase2_QA_summary_v2: {OUT_QA_PATH}")
    print("\nPhase 2 v2 complete. Stopping before Phase 3.")


if __name__ == "__main__":
    main()
