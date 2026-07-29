from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

TRIPS_PATH = DATA_PROCESSED / "trips_home_chain_enriched.csv"
CARS_PATH = DATA_PROCESSED / "cars_selected_raw.csv"

OUT_EVENTS_PATH = DATA_PROCESSED / "vehicle_events.csv"
OUT_OCCASIONS_PATH = DATA_PROCESSED / "vehicle_choice_occasions_all.csv"
OUT_QA_PATH = DATA_PROCESSED / "phase2_QA_summary.csv"

VALID_A_IDS = {1, 2, 3}
STATE_HOME = "HOME"
STATE_AWAY = "AWAY"
STATE_UNKNOWN = "UNKNOWN"


def read_csv_strings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def integer_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def any_invalid_time(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    return (
        integer_series(df["invalid_start_time_flag"]).eq(1)
        | integer_series(df["invalid_arrival_time_flag"]).eq(1)
        | integer_series(df["negative_duration_flag"]).eq(1)
    )


def ordered_set(values: Iterable[int]) -> str:
    return "|".join(str(int(value)) for value in sorted(values))


def parse_car_set(value: object) -> list[int]:
    text = str(value)
    if text == "" or text.lower() == "nan":
        return []
    return [int(part) for part in text.split("|") if part != ""]


def state_string(states: dict[int, str], fleet: list[int]) -> str:
    return "|".join(f"{a_id}:{states.get(a_id, STATE_UNKNOWN)}" for a_id in fleet)


def construct_recorded_fleet(cars: pd.DataFrame, retained_households: set[str]) -> pd.DataFrame:
    for required_col in ["H_ID", "A_ID"]:
        if required_col not in cars.columns:
            raise ValueError(f"Car data is missing required column: {required_col}")
    pairs = cars[["H_ID", "A_ID"]].copy()
    pairs["A_ID_NUM"] = integer_series(pairs["A_ID"])
    pairs = pairs[pairs["H_ID"].isin(retained_households) & pairs["A_ID_NUM"].isin(VALID_A_IDS)]
    pairs = pairs[["H_ID", "A_ID_NUM"]].drop_duplicates()

    rows = []
    for h_id, group in pairs.groupby("H_ID", sort=False):
        ids = sorted(group["A_ID_NUM"].astype(int).unique())
        rows.append({"H_ID": h_id, "FLEET_CAR_SET_REBUILT": ordered_set(ids), "FLEET_A_IDS": ids})
    return pd.DataFrame(rows)


def add_numeric_columns(trips: pd.DataFrame) -> pd.DataFrame:
    out = trips.copy()
    numeric_cols = [
        "SOURCE_ROW_ID",
        "W_ID",
        "CHAIN_ELIGIBLE",
        "pkw_fmf",
        "W_WAUTO",
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
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[f"{col}_NUM"] = integer_series(out[col])
    return out


def prepare_driver_trips(trips: pd.DataFrame, fleet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fleet_pairs = {
        (row.H_ID, int(a_id))
        for row in fleet.itertuples(index=False)
        for a_id in row.FLEET_A_IDS
    }
    candidate = (
        trips["CHAIN_ELIGIBLE_NUM"].eq(1)
        & trips["pkw_fmf_NUM"].eq(1)
        & trips["W_WAUTO_NUM"].isin(sorted(VALID_A_IDS))
    )
    in_recorded_fleet = pd.Series(
        [
            (h_id, int(a_id)) in fleet_pairs if not pd.isna(a_id) else False
            for h_id, a_id in zip(trips["H_ID"], trips["W_WAUTO_NUM"])
        ],
        index=trips.index,
    )
    unmatched = trips[candidate & ~in_recorded_fleet].copy()
    unmatched["unmatched_reported_vehicle_flag"] = 1

    selected = trips[candidate & in_recorded_fleet].copy()
    selected["CHOSEN_A_ID"] = selected["W_WAUTO_NUM"].astype(int)
    selected["unmatched_reported_vehicle_flag"] = 0
    selected["vehicle_time_history_unresolved_flag"] = any_invalid_time(selected).astype(int)
    household_timeline_conflict_hids = set(
        selected.loc[selected["time_sequence_conflict_flag_NUM"].eq(1), "H_ID"].astype(str)
    )
    selected["household_vehicle_timeline_conflict_flag"] = (
        selected["H_ID"].astype(str).isin(household_timeline_conflict_hids).astype(int)
    )

    person_id = np.where(selected["HP_ID"].ne(""), selected["HP_ID"], selected["P_ID"])
    selected["TRIP_ID_BASE"] = selected["H_ID"] + "_" + pd.Series(person_id, index=selected.index) + "_" + selected["W_ID"]
    selected["duplicate_trip_identifier_flag"] = selected["TRIP_ID_BASE"].duplicated(keep=False).astype(int)
    selected["TRIP_ID"] = selected["TRIP_ID_BASE"]
    dup_mask = selected["duplicate_trip_identifier_flag"].eq(1)
    selected.loc[dup_mask, "TRIP_ID"] = (
        selected.loc[dup_mask, "TRIP_ID_BASE"] + "_" + selected.loc[dup_mask, "SOURCE_ROW_ID"].astype(str)
    )
    selected["CHOICE_ID"] = selected["TRIP_ID"]

    return selected, unmatched, pd.DataFrame({"H_ID": sorted(household_timeline_conflict_hids)})


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
        "unmatched_reported_vehicle_flag",
        "duplicate_trip_identifier_flag",
        "vehicle_time_history_unresolved_flag",
        "household_vehicle_timeline_conflict_flag",
        "time_sequence_conflict_flag",
        "complete_fleet_baseline_flag",
        "topcoded_multicar_household_flag",
        "fleet_inventory_mismatch_flag",
        "household_car_count_conflict_flag",
        "nonstandard_A_ID_set_flag",
        "FLEET_CAR_SET",
        "N_RECORDED_CARS",
    ]
    base_cols = [col for col in base_cols if col in selected.columns]

    departures = selected[base_cols].copy()
    departures["A_ID"] = selected["CHOSEN_A_ID"].astype(int)
    departures["event_type"] = "DEPARTURE"
    departures["event_time"] = selected["START_MIN_NUM"]
    departures["event_home_status"] = selected["HOME_ORIGIN_NUM"]
    departures["invalid_event_time_flag"] = selected["invalid_start_time_flag_NUM"].eq(1).astype(int)

    arrivals = selected[base_cols].copy()
    arrivals["A_ID"] = selected["CHOSEN_A_ID"].astype(int)
    arrivals["event_type"] = "ARRIVAL"
    arrivals["event_time"] = selected["ARRIVAL_MIN_NUM"]
    arrivals["event_home_status"] = selected["HOME_DESTINATION_NUM"]
    arrivals["invalid_event_time_flag"] = selected["invalid_arrival_time_flag_NUM"].eq(1).astype(int)

    events = pd.concat([departures, arrivals], ignore_index=True)
    events["event_time"] = events["event_time"].astype("Int64")
    events["event_home_status"] = events["event_home_status"].astype("Int64")
    events["arrival_while_home_flag"] = 0
    events["departure_while_away_flag"] = 0
    events["vehicle_event_conflict_flag"] = 0
    events["state_before_processing"] = ""
    events["state_after_processing"] = ""
    events["AVAILABLE_CAR_SET"] = ""
    events["N_AVAILABLE_CARS"] = 0
    events["unknown_vehicle_state_at_snapshot_flag"] = 0
    events["departure_updates_to_unknown_flag"] = (
        events["event_type"].eq("DEPARTURE")
        & integer_series(events["invalid_arrival_time_flag"]).eq(1)
    ).astype(int)
    events["event_type_order"] = events["event_type"].map({"ARRIVAL": 0, "DEPARTURE": 1})
    events = events.sort_values(
        ["H_ID", "event_time", "event_type_order", "SOURCE_ROW_ID", "A_ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return events


def initial_vehicle_states(events: pd.DataFrame, fleet: pd.DataFrame, retained_households: set[str]) -> pd.DataFrame:
    valid_events = events[events["invalid_event_time_flag"].eq(0)].copy()
    valid_events = valid_events.sort_values(["H_ID", "A_ID", "event_time", "event_type_order", "SOURCE_ROW_ID"], kind="mergesort")
    invalid_start_mask = integer_series(events["invalid_start_time_flag"]).eq(1)
    invalid_start_pairs = set(
        zip(
            events.loc[invalid_start_mask, "H_ID"].astype(str),
            integer_series(events.loc[invalid_start_mask, "A_ID"]).astype(int),
        )
    )
    all_event_pairs = set(zip(events["H_ID"].astype(str), integer_series(events["A_ID"]).astype(int)))
    unresolved_mask = events["vehicle_time_history_unresolved_flag"].eq(1)
    unresolved_pairs = set(
        zip(
            events.loc[unresolved_mask, "H_ID"].astype(str),
            integer_series(events.loc[unresolved_mask, "A_ID"]).astype(int),
        )
    )
    first_valid_by_pair: dict[tuple[str, int], dict] = {}
    for (h_id, a_id), group in valid_events.groupby(["H_ID", "A_ID"], sort=False):
        first_valid_by_pair[(str(h_id), int(a_id))] = group.iloc[0].to_dict()

    rows = []
    for row in fleet.itertuples(index=False):
        h_id = str(row.H_ID)
        if h_id not in retained_households:
            continue
        for a_id in row.FLEET_A_IDS:
            pair = (h_id, int(a_id))
            initial_home_assumption = 0
            if pair in invalid_start_pairs:
                initial_state = STATE_UNKNOWN
                initial_unresolved = 1
            elif pair not in first_valid_by_pair:
                if pair in all_event_pairs:
                    initial_state = STATE_UNKNOWN
                    initial_unresolved = 1
                else:
                    initial_state = STATE_HOME
                    initial_unresolved = 0
                    initial_home_assumption = 1
            else:
                first = first_valid_by_pair[pair]
                if first["event_type"] == "DEPARTURE" and event_home_value(first["HOME_ORIGIN"]) == 1:
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
                    "initial_home_assumption_flag": initial_home_assumption,
                    "vehicle_time_history_unresolved_flag": int(pair in unresolved_pairs),
                }
            )
    return pd.DataFrame(rows)


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


def process_household_timelines(
    events: pd.DataFrame,
    selected: pd.DataFrame,
    fleet: pd.DataFrame,
    initial_states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows = []
    occasion_rows = []
    trace_rows = []

    fleet_by_h = {str(row.H_ID): list(row.FLEET_A_IDS) for row in fleet.itertuples(index=False)}
    initial_by_h = {
        h_id: group.set_index("A_ID").to_dict("index")
        for h_id, group in initial_states.groupby("H_ID", sort=False)
    }
    trip_lookup_cols = [
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
        "unmatched_reported_vehicle_flag",
        "duplicate_trip_identifier_flag",
        "vehicle_time_history_unresolved_flag",
        "household_vehicle_timeline_conflict_flag",
        "complete_fleet_baseline_flag",
        "topcoded_multicar_household_flag",
        "fleet_inventory_mismatch_flag",
        "household_car_count_conflict_flag",
        "nonstandard_A_ID_set_flag",
        "time_sequence_conflict_flag",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "negative_duration_flag",
        "FLEET_CAR_SET",
        "N_RECORDED_CARS",
    ]
    selected_by_source = selected.assign(SOURCE_ROW_ID_KEY=selected["SOURCE_ROW_ID"].astype(str))[
        ["SOURCE_ROW_ID_KEY", *trip_lookup_cols]
    ].set_index("SOURCE_ROW_ID_KEY").to_dict("index")
    unresolved_household_vehicle = set(
        zip(
            events.loc[events["vehicle_time_history_unresolved_flag"].eq(1), "H_ID"].astype(str),
            integer_series(events.loc[events["vehicle_time_history_unresolved_flag"].eq(1), "A_ID"]).astype(int),
        )
    )

    valid_events = events[events["invalid_event_time_flag"].eq(0)].copy()
    invalid_start_home = selected[
        selected["HOME_ORIGIN_NUM"].eq(1) & selected["invalid_start_time_flag_NUM"].eq(1)
    ].copy()

    for h_id, h_events in valid_events.groupby("H_ID", sort=False):
        h_id = str(h_id)
        fleet_ids = fleet_by_h.get(h_id, [])
        if not fleet_ids:
            continue
        init = initial_by_h.get(h_id, {})
        states = {a_id: init.get(a_id, {}).get("initial_state", STATE_UNKNOWN) for a_id in fleet_ids}
        initial_unresolved_any = int(any(init.get(a_id, {}).get("initial_state_unresolved_flag", 1) == 1 for a_id in fleet_ids))
        vehicle_time_unresolved_any = int(any((h_id, a_id) in unresolved_household_vehicle for a_id in fleet_ids))
        household_event_conflict_seen = 0

        for event_time, time_group in h_events.groupby("event_time", sort=True):
            arrivals = time_group[time_group["event_type"].eq("ARRIVAL")].sort_values(
                ["SOURCE_ROW_ID", "A_ID"], kind="mergesort"
            )
            departures = time_group[time_group["event_type"].eq("DEPARTURE")].sort_values(
                ["SOURCE_ROW_ID", "A_ID"], kind="mergesort"
            )

            for _, event in arrivals.iterrows():
                a_id = int(event["A_ID"])
                before = state_string(states, fleet_ids)
                arrival_while_home = int(states.get(a_id, STATE_UNKNOWN) == STATE_HOME)
                destination = event_home_value(event["HOME_DESTINATION"])
                if destination == 1:
                    states[a_id] = STATE_HOME
                elif destination == 0:
                    states[a_id] = STATE_AWAY
                else:
                    states[a_id] = STATE_UNKNOWN
                household_event_conflict_seen = max(household_event_conflict_seen, arrival_while_home)
                after = state_string(states, fleet_ids)
                row = event.to_dict()
                row.update(
                    {
                        "arrival_while_home_flag": arrival_while_home,
                        "vehicle_event_conflict_flag": arrival_while_home,
                        "state_before_processing": before,
                        "state_after_processing": after,
                    }
                )
                event_rows.append(row)
                trace_rows.append(trace_from_event(row, event_time, before, after))

            if not departures.empty:
                available = [a_id for a_id in fleet_ids if states.get(a_id) == STATE_HOME]
                available_set = ordered_set(available)
                unknown_snapshot = int(any(states.get(a_id) == STATE_UNKNOWN for a_id in fleet_ids))
                home_departures = departures[integer_series(departures["HOME_ORIGIN"]).eq(1)]
                simultaneous = int(home_departures["SOURCE_ROW_ID"].nunique() >= 2 and home_departures["HP_ID"].nunique() >= 2)
                same_vehicle_simultaneous = int(
                    simultaneous == 1 and home_departures["A_ID"].duplicated(keep=False).any()
                )
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
                        "arrival_while_home_flag": 0,
                        "departure_while_away_flag": 0,
                        "vehicle_event_conflict_flag": household_event_conflict_seen,
                        "vehicle_time_history_unresolved_flag": vehicle_time_unresolved_any,
                    }
                )

                for _, dep in home_departures.iterrows():
                    trip = selected_by_source[str(dep["SOURCE_ROW_ID"])]
                    chosen = int(dep["A_ID"])
                    occasion = build_occasion_row(
                        trip,
                        available_set,
                        len(available),
                        unknown_snapshot,
                        initial_unresolved_any,
                        simultaneous,
                        same_vehicle_simultaneous,
                        int(chosen not in available),
                        household_event_conflict_seen,
                        vehicle_time_unresolved_any,
                        invalid_choice_snapshot_time_flag=0,
                    )
                    occasion_rows.append(occasion)

            for _, event in departures.iterrows():
                a_id = int(event["A_ID"])
                before = state_string(states, fleet_ids)
                departure_while_away = int(states.get(a_id, STATE_UNKNOWN) == STATE_AWAY)
                if int(event["departure_updates_to_unknown_flag"]) == 1:
                    states[a_id] = STATE_UNKNOWN
                else:
                    states[a_id] = STATE_AWAY
                household_event_conflict_seen = max(household_event_conflict_seen, departure_while_away)
                after = state_string(states, fleet_ids)
                row = event.to_dict()
                row.update(
                    {
                        "departure_while_away_flag": departure_while_away,
                        "vehicle_event_conflict_flag": departure_while_away,
                        "state_before_processing": before,
                        "state_after_processing": after,
                    }
                )
                event_rows.append(row)
                trace_rows.append(trace_from_event(row, event_time, before, after))

    for _, event in events[events["invalid_event_time_flag"].eq(1)].iterrows():
        row = event.to_dict()
        row["vehicle_event_conflict_flag"] = 0
        row["state_before_processing"] = STATE_UNKNOWN
        row["state_after_processing"] = STATE_UNKNOWN
        event_rows.append(row)

    for _, trip in invalid_start_home.iterrows():
        occasion_rows.append(
            build_occasion_row(
                trip.to_dict(),
                "",
                0,
                unknown_snapshot=1,
                initial_state_unresolved=1,
                simultaneous=0,
                same_vehicle_simultaneous=0,
                availability_conflict=1,
                vehicle_event_conflict=0,
                vehicle_time_unresolved=1,
                invalid_choice_snapshot_time_flag=1,
            )
        )

    processed_events = pd.DataFrame(event_rows)
    occasions = pd.DataFrame(occasion_rows)
    trace = pd.DataFrame(trace_rows)

    if not processed_events.empty:
        processed_events["vehicle_event_conflict_flag"] = (
            integer_series(processed_events["arrival_while_home_flag"]).eq(1)
            | integer_series(processed_events["departure_while_away_flag"]).eq(1)
        ).astype(int)
        household_event_conflict = processed_events.groupby("H_ID", sort=False)["vehicle_event_conflict_flag"].max()
        if not occasions.empty:
            occasions["vehicle_event_conflict_flag"] = occasions["H_ID"].map(household_event_conflict).fillna(
                occasions["vehicle_event_conflict_flag"]
            ).astype(int)

    return processed_events, occasions, trace


def build_occasion_row(
    trip: dict,
    available_set: str,
    n_available: int,
    unknown_snapshot: int,
    initial_state_unresolved: int,
    simultaneous: int,
    same_vehicle_simultaneous: int,
    availability_conflict: int,
    vehicle_event_conflict: int,
    vehicle_time_unresolved: int,
    invalid_choice_snapshot_time_flag: int,
) -> dict:
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
        "CHOSEN_A_ID": int(trip["CHOSEN_A_ID"]),
        "AVAILABLE_CAR_SET": available_set,
        "N_AVAILABLE_CARS": int(n_available),
        "unknown_vehicle_state_at_snapshot_flag": int(unknown_snapshot),
        "initial_state_unresolved_flag": int(initial_state_unresolved),
        "unmatched_reported_vehicle_flag": int(trip["unmatched_reported_vehicle_flag"]),
        "duplicate_trip_identifier_flag": int(trip["duplicate_trip_identifier_flag"]),
        "vehicle_time_history_unresolved_flag": int(max(vehicle_time_unresolved, trip["vehicle_time_history_unresolved_flag"])),
        "household_vehicle_timeline_conflict_flag": int(trip["household_vehicle_timeline_conflict_flag"]),
        "availability_conflict_flag": int(availability_conflict),
        "zero_available_case_flag": int(n_available == 0),
        "one_available_case_flag": int(n_available == 1),
        "two_available_case_flag": int(n_available == 2),
        "three_available_case_flag": int(n_available == 3),
        "simultaneous_home_departure_flag": int(simultaneous),
        "same_vehicle_simultaneous_departure_flag": int(same_vehicle_simultaneous),
        "vehicle_event_conflict_flag": int(vehicle_event_conflict),
        "invalid_choice_snapshot_time_flag": int(invalid_choice_snapshot_time_flag),
        "complete_fleet_baseline_flag": int(trip.get("complete_fleet_baseline_flag", 0)),
        "topcoded_multicar_household_flag": int(trip.get("topcoded_multicar_household_flag", 0)),
        "fleet_inventory_mismatch_flag": int(trip.get("fleet_inventory_mismatch_flag", 0)),
        "household_car_count_conflict_flag": int(trip.get("household_car_count_conflict_flag", 0)),
        "nonstandard_A_ID_set_flag": int(trip.get("nonstandard_A_ID_set_flag", 0)),
        "time_sequence_conflict_flag": int(trip.get("time_sequence_conflict_flag", 0)),
        "invalid_start_time_flag": int(trip.get("invalid_start_time_flag", 0)),
        "invalid_arrival_time_flag": int(trip.get("invalid_arrival_time_flag", 0)),
        "negative_duration_flag": int(trip.get("negative_duration_flag", 0)),
        "FLEET_CAR_SET": trip.get("FLEET_CAR_SET", ""),
        "N_RECORDED_CARS": trip.get("N_RECORDED_CARS", ""),
    }


def trace_from_event(row: dict, event_time: object, before: str, after: str) -> dict:
    return {
        "H_ID": row["H_ID"],
        "event_time": int(event_time),
        "event_type": row["event_type"],
        "TRIP_ID": row["TRIP_ID"],
        "HP_ID": row["HP_ID"],
        "A_ID": row["A_ID"],
        "HOME_ORIGIN": row["HOME_ORIGIN"],
        "HOME_DESTINATION": row["HOME_DESTINATION"],
        "state_before_processing": before,
        "state_after_processing": after,
        "AVAILABLE_CAR_SET": row.get("AVAILABLE_CAR_SET", ""),
        "arrival_while_home_flag": row.get("arrival_while_home_flag", 0),
        "departure_while_away_flag": row.get("departure_while_away_flag", 0),
        "vehicle_event_conflict_flag": row.get("vehicle_event_conflict_flag", 0),
        "vehicle_time_history_unresolved_flag": row.get("vehicle_time_history_unresolved_flag", 0),
    }


def build_qa_summary(
    trips: pd.DataFrame,
    selected: pd.DataFrame,
    unmatched: pd.DataFrame,
    events: pd.DataFrame,
    occasions: pd.DataFrame,
    initial_states: pd.DataFrame,
) -> pd.DataFrame:
    home_origin_driver = selected[selected["HOME_ORIGIN_NUM"].eq(1)]
    metrics = [
        ("identifiable household-car driver trips", len(selected), max(len(selected), 1)),
        ("unmatched reported household vehicles", len(unmatched), max(len(selected) + len(unmatched), 1)),
        ("vehicle departure events", int(events["event_type"].eq("DEPARTURE").sum()), max(len(events), 1)),
        ("vehicle arrival events", int(events["event_type"].eq("ARRIVAL").sum()), max(len(events), 1)),
        ("home-origin driver departures", len(home_origin_driver), max(len(selected), 1)),
        ("zero-available cases", int(occasions["zero_available_case_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("one-available-car cases", int(occasions["one_available_case_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("two-available-car cases", int(occasions["two_available_case_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("three-available-car cases", int(occasions["three_available_case_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("chosen vehicle absent from available set", int(occasions["availability_conflict_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("initial-state-unresolved vehicles", int(initial_states["initial_state_unresolved_flag"].sum()), max(len(initial_states), 1)),
        ("vehicles with no observed driver events", int(initial_states["initial_home_assumption_flag"].sum()), max(len(initial_states), 1)),
        ("choice occasions with unknown vehicle states", int(occasions["unknown_vehicle_state_at_snapshot_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("simultaneous home-departure occasions", int(occasions["simultaneous_home_departure_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("same-vehicle simultaneous departures", int(occasions["same_vehicle_simultaneous_departure_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
        ("arrival-while-home conflicts", int(events["arrival_while_home_flag"].sum()), max(len(events), 1)),
        ("departure-while-away conflicts", int(events["departure_while_away_flag"].sum()), max(len(events), 1)),
        ("invalid event times", int(events["invalid_event_time_flag"].sum()), max(len(events), 1)),
        ("vehicle-time-history-unresolved trips", int(selected["vehicle_time_history_unresolved_flag"].sum()), max(len(selected), 1)),
        ("household vehicle timeline conflict households", int(selected.loc[selected["household_vehicle_timeline_conflict_flag"].eq(1), "H_ID"].nunique()), max(trips["H_ID"].nunique(), 1)),
        ("invalid-start home-origin unresolved occasions", int(occasions["invalid_choice_snapshot_time_flag"].sum()) if not occasions.empty else 0, max(len(occasions), 1)),
    ]
    return pd.DataFrame(
        [{"metric": metric, "count": int(count), "share": count / denominator} for metric, count, denominator in metrics]
    )


def assert_phase2(selected: pd.DataFrame, events: pd.DataFrame, occasions: pd.DataFrame, fleet: pd.DataFrame) -> None:
    event_counts = events.groupby(["TRIP_ID", "event_type"], sort=False).size().unstack(fill_value=0)
    assert (event_counts.get("DEPARTURE", 0) == 1).all(), "Each selected TRIP_ID must have one DEPARTURE event."
    assert (event_counts.get("ARRIVAL", 0) == 1).all(), "Each selected TRIP_ID must have one ARRIVAL event."
    assert event_counts.sum(axis=1).eq(2).all(), "Each selected TRIP_ID must create exactly two events."
    assert set(event_counts.index) == set(selected["TRIP_ID"]), "Event TRIP_ID set does not match selected trips."

    fleet_sets = {row.H_ID: set(row.FLEET_A_IDS) for row in fleet.itertuples(index=False)}
    for row in occasions.itertuples(index=False):
        available = set(parse_car_set(row.AVAILABLE_CAR_SET))
        assert available.issubset(fleet_sets.get(row.H_ID, set())), "AVAILABLE_CAR_SET contains A_ID outside fleet."
        assert row.AVAILABLE_CAR_SET == ordered_set(available), "AVAILABLE_CAR_SET is not deterministically ordered."
        assert int(row.N_AVAILABLE_CARS) == len(available), "N_AVAILABLE_CARS does not match AVAILABLE_CAR_SET."

    if not occasions.empty:
        grouped = occasions.groupby(["H_ID", "START_MIN"], sort=False)
        for _, group in grouped:
            if group["simultaneous_home_departure_flag"].eq(1).any():
                assert group["AVAILABLE_CAR_SET"].nunique() == 1, "Simultaneous departures lack common snapshot."


def print_household_traces(trace: pd.DataFrame, events: pd.DataFrame, occasions: pd.DataFrame, selected: pd.DataFrame) -> None:
    chosen: list[str] = []

    def add(label: str, candidates: Iterable[str]) -> None:
        for h_id in candidates:
            if h_id in chosen:
                continue
            print(f"\nTrace selected for {label}: H_ID={h_id}")
            chosen.append(str(h_id))
            return
        print(f"\nNo trace found for {label}.")

    if not occasions.empty:
        add("normal two-car household", occasions.loc[
            occasions["N_AVAILABLE_CARS"].eq(2)
            & occasions["availability_conflict_flag"].eq(0)
            & occasions["vehicle_event_conflict_flag"].eq(0),
            "H_ID",
        ])
        arrival_home_hids = set(events.loc[events["event_type"].eq("ARRIVAL") & integer_series(events["HOME_DESTINATION"]).eq(1), "H_ID"])
        later_depart_hids = set(occasions["H_ID"])
        add("one vehicle returning home before another departure", [h for h in arrival_home_hids if h in later_depart_hids])
        add("simultaneous home departures", occasions.loc[occasions["simultaneous_home_departure_flag"].eq(1), "H_ID"])
        add("availability conflict", occasions.loc[occasions["availability_conflict_flag"].eq(1), "H_ID"])
    add("away-origin vehicle departure", selected.loc[selected["HOME_ORIGIN_NUM"].eq(0), "H_ID"])
    add("unresolved first vehicle event", events.loc[events["vehicle_time_history_unresolved_flag"].eq(1), "H_ID"])

    print("\nPHASE 2 HOUSEHOLD STATE TRACES")
    display_cols = [
        "H_ID",
        "event_time",
        "event_type",
        "TRIP_ID",
        "HP_ID",
        "A_ID",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "state_before_processing",
        "state_after_processing",
        "AVAILABLE_CAR_SET",
        "arrival_while_home_flag",
        "departure_while_away_flag",
        "vehicle_event_conflict_flag",
        "vehicle_time_history_unresolved_flag",
    ]
    for idx, h_id in enumerate(chosen[:6], start=1):
        h_trace = trace[trace["H_ID"].astype(str).eq(str(h_id))].sort_values(
            ["event_time", "event_type", "TRIP_ID"], kind="mergesort"
        )
        print(f"\n--- Household trace {idx}: H_ID={h_id}, trace rows={len(h_trace)} ---")
        with pd.option_context("display.max_rows", 80, "display.max_columns", None, "display.width", 260):
            print(h_trace[display_cols].to_string(index=False))


def main() -> None:
    t0 = perf_counter()
    trips = add_numeric_columns(read_csv_strings(TRIPS_PATH))
    cars = read_csv_strings(CARS_PATH)
    print(f"Loaded inputs in {perf_counter() - t0:.1f}s", flush=True)
    retained_households = set(trips["H_ID"].astype(str).unique())
    fleet = construct_recorded_fleet(cars, retained_households)
    print(f"Constructed fleet in {perf_counter() - t0:.1f}s", flush=True)

    selected, unmatched, _timeline_conflict_hh = prepare_driver_trips(trips, fleet)
    print(f"Prepared selected driver trips in {perf_counter() - t0:.1f}s", flush=True)
    events = create_vehicle_events(selected)
    print(f"Created event rows in {perf_counter() - t0:.1f}s", flush=True)
    initial_states = initial_vehicle_states(events, fleet, retained_households)
    print(f"Classified initial states in {perf_counter() - t0:.1f}s", flush=True)
    processed_events, occasions, trace = process_household_timelines(events, selected, fleet, initial_states)
    print(f"Processed household timelines in {perf_counter() - t0:.1f}s", flush=True)

    processed_events = processed_events.drop(columns=["event_type_order"], errors="ignore").sort_values(
        ["H_ID", "event_time", "event_type", "SOURCE_ROW_ID", "A_ID"],
        kind="mergesort",
        na_position="last",
    )
    if not occasions.empty:
        occasions = occasions.sort_values(["H_ID", "START_MIN", "SOURCE_ROW_ID"], kind="mergesort", na_position="last")

    assert_phase2(selected, processed_events, occasions, fleet)
    qa = build_qa_summary(trips, selected, unmatched, processed_events, occasions, initial_states)
    print(f"Validated Phase 2 in {perf_counter() - t0:.1f}s", flush=True)

    processed_events.to_csv(OUT_EVENTS_PATH, index=False)
    occasions.to_csv(OUT_OCCASIONS_PATH, index=False)
    qa.to_csv(OUT_QA_PATH, index=False)
    print(f"Saved Phase 2 outputs in {perf_counter() - t0:.1f}s", flush=True)

    print_household_traces(trace, processed_events, occasions, selected)
    print("\nPHASE 2 QA SUMMARY")
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(qa.to_string(index=False, formatters={"share": "{:.6f}".format}))

    print("\nPHASE 2 OUTPUT PATHS")
    print(f"vehicle_events: {OUT_EVENTS_PATH}")
    print(f"vehicle_choice_occasions_all: {OUT_OCCASIONS_PATH}")
    print(f"phase2_QA_summary: {OUT_QA_PATH}")
    print("\nFINAL PHASE 2 COUNTS")
    print(f"selected identifiable driver trips: {len(selected):,}")
    print(f"vehicle events: {len(processed_events):,}")
    print(f"home-origin choice occasions: {len(occasions):,}")
    print("\nPhase 2 complete. Stopping before Phase 3.")


if __name__ == "__main__":
    main()
