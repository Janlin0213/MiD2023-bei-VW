from __future__ import annotations

from dataclasses import dataclass
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

from src.thesis_pipeline.paths import PHASE1_DIR, SELECTED_RAW_DIR

TRIPS_PATH = SELECTED_RAW_DIR / "trips_selected_raw.csv"
CARS_PATH = SELECTED_RAW_DIR / "cars_selected_raw.csv"

OUT_TRIPS_PATH = PHASE1_DIR / "trips_home_chain_enriched.csv"
OUT_QA_PATH = PHASE1_DIR / "phase1_QA_summary.csv"

EXPECTED_A_IDS = {1, 2, 3}
TIME_MISSING_STRINGS = {"", "nan", "NaN", "NA", "N/A", "None"}


@dataclass(frozen=True)
class ColumnMap:
    h_id: str
    hp_id: str | None
    p_id: str
    w_id: str
    w_rbw: str
    w_so1: str
    dest_purpose: str
    start_combined: str | None
    arrival_combined: str | None
    start_hour: str | None
    start_minute: str | None
    arrival_hour: str | None
    arrival_minute: str | None
    folgetag: str
    h_anzauto: str


def read_csv_strings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def resolve_one(columns: Iterable[str], required_name: str, aliases: list[str] | None = None) -> str:
    aliases = aliases or []
    column_set = set(columns)
    for candidate in [required_name, *aliases]:
        if candidate in column_set:
            return candidate
    raise ValueError(f"Required column not found: {required_name} (aliases tried: {aliases})")


def resolve_optional(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def build_column_map(trips: pd.DataFrame) -> ColumnMap:
    columns = trips.columns
    start_combined = resolve_optional(columns, ["W_SZ"])
    arrival_combined = resolve_optional(columns, ["W_AZ"])
    start_hour = resolve_optional(columns, ["W_SZS"])
    start_minute = resolve_optional(columns, ["W_SZM"])
    arrival_hour = resolve_optional(columns, ["W_AZS"])
    arrival_minute = resolve_optional(columns, ["W_AZM"])

    if not ((start_combined and arrival_combined) or (start_hour and start_minute and arrival_hour and arrival_minute)):
        raise ValueError("Trip data needs W_SZ/W_AZ or W_SZS/W_SZM/W_AZS/W_AZM time columns.")

    return ColumnMap(
        h_id=resolve_one(columns, "H_ID"),
        hp_id=resolve_optional(columns, ["HP_ID"]),
        p_id=resolve_one(columns, "P_ID"),
        w_id=resolve_one(columns, "W_ID"),
        w_rbw=resolve_one(columns, "W_RBW"),
        w_so1=resolve_one(columns, "W_SO1"),
        dest_purpose=resolve_one(columns, "W_ZWECK", aliases=["zweck"]),
        start_combined=start_combined,
        arrival_combined=arrival_combined,
        start_hour=start_hour,
        start_minute=start_minute,
        arrival_hour=arrival_hour,
        arrival_minute=arrival_minute,
        folgetag=resolve_one(columns, "W_FOLGETAG"),
        h_anzauto=resolve_one(columns, "H_ANZAUTO"),
    )


def integer_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def valid_nonmissing(series: pd.Series) -> pd.Series:
    return ~series.astype("string").str.strip().isin(["", "<NA>", "nan", "NaN", "None"])


def household_car_counts(trips: pd.DataFrame, colmap: ColumnMap) -> pd.DataFrame:
    all_households = trips[[colmap.h_id]].drop_duplicates().rename(columns={colmap.h_id: "H_ID"})
    values = trips.loc[valid_nonmissing(trips[colmap.h_anzauto]), [colmap.h_id, colmap.h_anzauto]].drop_duplicates()
    values = values.rename(columns={colmap.h_id: "H_ID", colmap.h_anzauto: "H_ANZAUTO_VALUE"})
    grouped = values.groupby("H_ID", sort=False)["H_ANZAUTO_VALUE"].agg(lambda s: sorted(s.astype(str))).reset_index()

    out = all_households.merge(grouped, on="H_ID", how="left")
    out["H_ANZAUTO_VALUE"] = out["H_ANZAUTO_VALUE"].apply(lambda value: value if isinstance(value, list) else [])
    out["household_car_count_conflict_flag"] = out["H_ANZAUTO_VALUE"].apply(lambda values_: int(len(values_) > 1))
    out["H_ANZAUTO_HH"] = out["H_ANZAUTO_VALUE"].apply(lambda values_: values_[0] if len(values_) == 1 else pd.NA)
    out = out.drop(columns=["H_ANZAUTO_VALUE"])
    out["H_ANZAUTO_NUM"] = integer_series(out["H_ANZAUTO_HH"])
    return out


def ordered_car_set(values: Iterable[int]) -> str:
    ordered = sorted(int(v) for v in values)
    return "|".join(str(v) for v in ordered)


def construct_recorded_fleet(cars: pd.DataFrame, household_counts: pd.DataFrame) -> pd.DataFrame:
    for required_col in ["H_ID", "A_ID"]:
        if required_col not in cars.columns:
            raise ValueError(f"Car data is missing required column: {required_col}")

    car_pairs = cars[["H_ID", "A_ID"]].copy()
    car_pairs["A_ID_NUM"] = integer_series(car_pairs["A_ID"])
    car_pairs = car_pairs[car_pairs["A_ID_NUM"].isin(EXPECTED_A_IDS)]
    unique_pairs = car_pairs[["H_ID", "A_ID_NUM"]].drop_duplicates()

    fleet_rows: list[dict[str, object]] = []
    for h_id, group in unique_pairs.groupby("H_ID", sort=False):
        ids = sorted(int(v) for v in group["A_ID_NUM"].dropna().unique())
        expected = list(range(1, len(ids) + 1))
        fleet_rows.append(
            {
                "H_ID": h_id,
                "FLEET_CAR_SET": ordered_car_set(ids),
                "N_RECORDED_CARS": len(ids),
                "nonstandard_A_ID_set_flag": int(ids != expected),
            }
        )

    fleet = pd.DataFrame(fleet_rows)
    if fleet.empty:
        fleet = pd.DataFrame(columns=["H_ID", "FLEET_CAR_SET", "N_RECORDED_CARS", "nonstandard_A_ID_set_flag"])

    out = household_counts.merge(fleet, on="H_ID", how="left")
    out["FLEET_CAR_SET"] = out["FLEET_CAR_SET"].fillna("")
    out["N_RECORDED_CARS"] = out["N_RECORDED_CARS"].fillna(0).astype(int)
    out["nonstandard_A_ID_set_flag"] = out["nonstandard_A_ID_set_flag"].fillna(0).astype(int)

    h_num = out["H_ANZAUTO_NUM"]
    no_conflict = out["household_car_count_conflict_flag"].eq(0)
    out["topcoded_multicar_household_flag"] = (h_num.eq(3) & no_conflict).astype(int)
    out["fleet_inventory_mismatch_flag"] = (h_num.eq(2) & out["N_RECORDED_CARS"].ne(2) & no_conflict).astype(int)
    out["complete_fleet_baseline_flag"] = (
        h_num.eq(2) & out["N_RECORDED_CARS"].eq(2) & no_conflict
    ).astype(int)
    return out


def parse_combined_time(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    extracted = text.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?$")
    hour = pd.to_numeric(extracted["hour"], errors="coerce")
    minute = pd.to_numeric(extracted["minute"], errors="coerce")
    second = pd.to_numeric(extracted["second"].fillna("0"), errors="coerce")
    minute_value = hour * 60 + minute
    invalid_parts = hour.isna() | minute.isna() | second.isna() | minute.lt(0) | minute.gt(59) | second.lt(0) | second.gt(59)
    return minute_value.mask(invalid_parts)


def parse_component_time(hour_series: pd.Series, minute_series: pd.Series) -> pd.Series:
    hour = integer_series(hour_series)
    minute = integer_series(minute_series)
    minute_value = hour * 60 + minute
    invalid_parts = hour.isna() | minute.isna() | minute.lt(0) | minute.gt(59)
    return minute_value.mask(invalid_parts)


def parse_trip_times(trips: pd.DataFrame, colmap: ColumnMap) -> pd.DataFrame:
    out = trips.copy()

    use_combined = False
    if colmap.start_combined and colmap.arrival_combined:
        start_from_combined = parse_combined_time(out[colmap.start_combined])
        arrival_from_combined = parse_combined_time(out[colmap.arrival_combined])
        start_nonblank = ~out[colmap.start_combined].astype("string").str.strip().isin(TIME_MISSING_STRINGS)
        arrival_nonblank = ~out[colmap.arrival_combined].astype("string").str.strip().isin(TIME_MISSING_STRINGS)
        start_rate = start_from_combined[start_nonblank].notna().mean() if start_nonblank.any() else 0
        arrival_rate = arrival_from_combined[arrival_nonblank].notna().mean() if arrival_nonblank.any() else 0
        use_combined = bool(start_rate >= 0.99 and arrival_rate >= 0.99)

    if use_combined:
        start_base = start_from_combined
        arrival_base = arrival_from_combined
    elif colmap.start_hour and colmap.start_minute and colmap.arrival_hour and colmap.arrival_minute:
        start_base = parse_component_time(out[colmap.start_hour], out[colmap.start_minute])
        arrival_base = parse_component_time(out[colmap.arrival_hour], out[colmap.arrival_minute])
    else:
        raise ValueError("Combined trip times were not reliably parseable and component time columns are unavailable.")

    folgetag = integer_series(out[colmap.folgetag]).fillna(0)
    out["START_MIN"] = start_base
    out["ARRIVAL_MIN"] = arrival_base + np.where(folgetag.eq(1), 1440, 0)

    out["invalid_start_time_flag"] = (out["START_MIN"].isna() | out["START_MIN"].lt(0) | out["START_MIN"].ge(1440)).astype(int)
    out["invalid_arrival_time_flag"] = (
        out["ARRIVAL_MIN"].isna() | out["ARRIVAL_MIN"].lt(0) | out["ARRIVAL_MIN"].ge(2880)
    ).astype(int)
    both_valid = out["invalid_start_time_flag"].eq(0) & out["invalid_arrival_time_flag"].eq(0)
    out["negative_duration_flag"] = (both_valid & out["ARRIVAL_MIN"].lt(out["START_MIN"])).astype(int)

    out["START_MIN"] = out["START_MIN"].round().astype("Int64")
    out["ARRIVAL_MIN"] = out["ARRIVAL_MIN"].round().astype("Int64")
    return out


def make_person_keys(trips: pd.DataFrame, colmap: ColumnMap) -> pd.Series:
    if colmap.hp_id and trips[colmap.hp_id].astype("string").str.strip().ne("").any():
        hp = trips[colmap.hp_id].astype("string").str.strip()
        fallback = trips[colmap.h_id].astype("string") + "|" + trips[colmap.p_id].astype("string")
        return np.where(hp.ne(""), "HP|" + hp, "HHP|" + fallback)
    return "HHP|" + trips[colmap.h_id].astype("string") + "|" + trips[colmap.p_id].astype("string")


def known_home_code(value: object) -> bool:
    return not pd.isna(value) and int(value) in (0, 1)


def purpose_to_destination(
    purpose: object, previous_home_origin: int | pd._libs.missing.NAType | None
) -> int | pd._libs.missing.NAType:
    if pd.isna(purpose):
        return pd.NA
    purpose_int = int(purpose)
    if purpose_int == 8:
        return 1
    if purpose_int == 9:
        return int(previous_home_origin) if known_home_code(previous_home_origin) else pd.NA
    if purpose_int in {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16}:
        return 0
    return pd.NA


def explicit_origin_from_w_so1(code: object) -> int | pd._libs.missing.NAType:
    if pd.isna(code):
        return pd.NA
    if int(code) == 1:
        return 1
    if int(code) == 2:
        return 0
    return pd.NA


def reconstruct_person_chains(trips: pd.DataFrame, colmap: ColumnMap) -> pd.DataFrame:
    out = trips.copy()
    out["PERSON_KEY"] = make_person_keys(out, colmap)
    out["W_ID_NUM"] = integer_series(out[colmap.w_id])
    out["W_RBW_NUM"] = integer_series(out[colmap.w_rbw])
    out["W_SO1_NUM"] = integer_series(out[colmap.w_so1])
    out["DEST_PURPOSE_NUM"] = integer_series(out[colmap.dest_purpose])
    out["CHAIN_ELIGIBLE"] = out["W_RBW_NUM"].eq(0).astype(int)
    out["excluded_from_chain_reason"] = np.where(out["W_RBW_NUM"].eq(1), "rbW_aggregate", "")
    out["CHAIN_SEQUENCE_NUM"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    hp_sort = colmap.hp_id if colmap.hp_id else colmap.p_id
    out = out.sort_values(
        [colmap.h_id, hp_sort, colmap.p_id, "W_ID_NUM", "SOURCE_ROW_ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    n = len(out)
    home_origin: list[object] = [pd.NA] * n
    home_destination: list[object] = [pd.NA] * n
    duplicate_w_id = np.zeros(n, dtype=np.int8)
    sequence_gap = np.zeros(n, dtype=np.int8)
    time_conflict = np.zeros(n, dtype=np.int8)
    so1_conflict = np.zeros(n, dtype=np.int8)

    for _, group in out.groupby("PERSON_KEY", sort=False):
        chain_group = group[group["CHAIN_ELIGIBLE"].eq(1)]
        positions = chain_group.index.to_numpy()
        if len(chain_group) == 0:
            continue

        w_ids = chain_group["W_ID_NUM"].dropna().astype(int)
        if not w_ids.empty:
            counts = w_ids.value_counts()
            has_duplicate = counts.gt(1).any()
            unique_wids = sorted(w_ids.unique())
            has_gap = len(unique_wids) > 1 and unique_wids != list(range(unique_wids[0], unique_wids[-1] + 1))
        else:
            has_duplicate = False
            has_gap = False

        starts = chain_group["START_MIN"].to_numpy(dtype=object)
        arrivals = chain_group["ARRIVAL_MIN"].to_numpy(dtype=object)
        invalid_start = chain_group["invalid_start_time_flag"].to_numpy()
        invalid_arrival = chain_group["invalid_arrival_time_flag"].to_numpy()
        has_time_conflict = False
        previous_start: int | None = None
        previous_arrival: int | None = None
        for start, arrival, bad_start, bad_arrival in zip(starts, arrivals, invalid_start, invalid_arrival):
            if not bad_start and previous_start is not None and int(start) < previous_start:
                has_time_conflict = True
            if not bad_start and previous_arrival is not None and int(start) < previous_arrival:
                has_time_conflict = True
            if not bad_start:
                previous_start = int(start)
            if not bad_arrival:
                previous_arrival = int(arrival)

        duplicate_w_id[positions] = int(has_duplicate)
        sequence_gap[positions] = int(has_gap)
        time_conflict[positions] = int(has_time_conflict)
        out.loc[positions, "CHAIN_SEQUENCE_NUM"] = pd.Series(range(1, len(positions) + 1), index=positions, dtype="Int64")

        previous_origin: int | pd._libs.missing.NAType | None = None
        previous_destination: int | pd._libs.missing.NAType | None = None
        so1_values = chain_group["W_SO1_NUM"].to_numpy(dtype=object)
        purpose_values = chain_group["DEST_PURPOSE_NUM"].to_numpy(dtype=object)
        for seq_idx, (pos, so1_value, purpose_value) in enumerate(zip(positions, so1_values, purpose_values)):
            explicit_origin = explicit_origin_from_w_so1(so1_value)
            if seq_idx == 0:
                origin = explicit_origin
            else:
                origin = int(previous_destination) if known_home_code(previous_destination) else pd.NA
                if known_home_code(explicit_origin) and known_home_code(origin) and int(explicit_origin) != int(origin):
                    so1_conflict[pos] = 1

            destination = purpose_to_destination(purpose_value, previous_origin)
            home_origin[pos] = origin
            home_destination[pos] = destination
            previous_origin = origin
            previous_destination = destination

    out["HOME_ORIGIN"] = pd.Series(home_origin, dtype="Int64")
    out["HOME_DESTINATION"] = pd.Series(home_destination, dtype="Int64")
    out["duplicate_W_ID_flag"] = duplicate_w_id
    out["trip_sequence_gap_flag"] = sequence_gap
    out["time_sequence_conflict_flag"] = time_conflict
    out["W_SO1_sequence_conflict_flag"] = so1_conflict
    out["home_origin_unknown_flag"] = out["HOME_ORIGIN"].isna().astype(int)
    out["home_destination_unknown_flag"] = out["HOME_DESTINATION"].isna().astype(int)
    out = out.drop(columns=["W_RBW_NUM", "W_SO1_NUM", "DEST_PURPOSE_NUM"])
    return out


def build_qa_summary(
    raw_trips: pd.DataFrame,
    retained: pd.DataFrame,
    household_fleet: pd.DataFrame,
) -> pd.DataFrame:
    retained_households = retained["H_ID"].nunique()
    retained_persons = retained["PERSON_KEY"].nunique()
    chain_eligible = retained[retained["CHAIN_ELIGIBLE"].eq(1)]
    rbw = retained[retained["excluded_from_chain_reason"].eq("rbW_aggregate")]
    household_den = max(retained_households, 1)
    person_den = max(retained_persons, 1)
    row_den = max(len(retained), 1)
    chain_row_den = max(len(chain_eligible), 1)
    rbw_den = max(len(rbw), 1)

    person_flags = chain_eligible.groupby("PERSON_KEY", sort=False)[
        ["duplicate_W_ID_flag", "trip_sequence_gap_flag", "time_sequence_conflict_flag"]
    ].max()
    retained_household_fleet = household_fleet[household_fleet["H_ID"].isin(retained["H_ID"].unique())]
    purpose_num = integer_series(retained["W_ZWECK"]) if "W_ZWECK" in retained.columns else integer_series(retained["zweck"])
    first_chain = retained["CHAIN_SEQUENCE_NUM"].eq(1)
    identifiable = identifiable_household_car_driver_mask(retained, household_fleet)
    identifiable_den = max(int(identifiable.sum()), 1)

    metrics = [
        ("raw trip rows", len(raw_trips), max(len(raw_trips), 1)),
        ("all retained rows", len(retained), row_den),
        ("directly reported chain-eligible trip rows", len(chain_eligible), row_den),
        ("rbW rows excluded from chain reconstruction", len(rbw), row_den),
        ("unique households", retained_households, household_den),
        ("unique persons", retained_persons, person_den),
        ("H_ANZAUTO == 2 households", int(retained_household_fleet["H_ANZAUTO_NUM"].eq(2).sum()), household_den),
        ("H_ANZAUTO == 3 households", int(retained_household_fleet["H_ANZAUTO_NUM"].eq(3).sum()), household_den),
        (
            "complete-fleet baseline households",
            int(retained_household_fleet["complete_fleet_baseline_flag"].sum()),
            household_den,
        ),
        (
            "household car-count conflicts",
            int(household_fleet["household_car_count_conflict_flag"].sum()),
            max(household_fleet["H_ID"].nunique(), 1),
        ),
        (
            "two-car fleet inventory mismatches",
            int(retained_household_fleet["fleet_inventory_mismatch_flag"].sum()),
            household_den,
        ),
        (
            "nonstandard A_ID sets",
            int(retained_household_fleet["nonstandard_A_ID_set_flag"].sum()),
            household_den,
        ),
        ("duplicate W_ID person chains", int(person_flags["duplicate_W_ID_flag"].sum()), person_den),
        ("trip-sequence-gap person chains", int(person_flags["trip_sequence_gap_flag"].sum()), person_den),
        ("time-sequence-conflict person chains", int(person_flags["time_sequence_conflict_flag"].sum()), person_den),
        ("unknown home origins", int(retained["home_origin_unknown_flag"].sum()), row_den),
        ("unknown home destinations", int(retained["home_destination_unknown_flag"].sum()), row_den),
        (
            "unknown HOME_ORIGIN among chain-eligible trips",
            int(chain_eligible["home_origin_unknown_flag"].sum()),
            chain_row_den,
        ),
        (
            "unknown HOME_DESTINATION among chain-eligible trips",
            int(chain_eligible["home_destination_unknown_flag"].sum()),
            chain_row_den,
        ),
        ("W_ZWECK == 99 rows", int(purpose_num.eq(99).sum()), row_den),
        (
            "W_ZWECK == 9 without a preceding chain-eligible trip",
            int((purpose_num.eq(9) & first_chain).sum()),
            chain_row_den,
        ),
        (
            "first chain-eligible trips with W_SO1 == 810",
            int((first_chain & integer_series(retained["W_SO1"]).eq(810)).sum()),
            max(int(first_chain.sum()), 1),
        ),
        (
            "first chain-eligible trips with W_SO1 == 9",
            int((first_chain & integer_series(retained["W_SO1"]).eq(9)).sum()),
            max(int(first_chain.sum()), 1),
        ),
        ("invalid start times", int(retained["invalid_start_time_flag"].sum()), row_den),
        ("invalid arrival times", int(retained["invalid_arrival_time_flag"].sum()), row_den),
        (
            "invalid-time rows among all retained rows",
            int(any_invalid_time(retained).sum()),
            row_den,
        ),
        ("invalid-time rows among rbW rows", int(any_invalid_time(rbw).sum()), rbw_den),
        (
            "invalid-time rows among chain-eligible trips",
            int(any_invalid_time(chain_eligible).sum()),
            chain_row_den,
        ),
        (
            "invalid-time rows among identifiable household-car driver trips",
            int(any_invalid_time(retained[identifiable]).sum()),
            identifiable_den,
        ),
        ("negative durations", int(retained["negative_duration_flag"].sum()), row_den),
    ]
    return pd.DataFrame(
        [{"metric": metric, "count": int(count), "share": count / denominator} for metric, count, denominator in metrics]
    )


def has_noncar_before_car(group: pd.DataFrame) -> bool:
    if "pkw_fmf" not in group.columns:
        return False
    seen_noncar = False
    for value in group["pkw_fmf"].astype("string"):
        if value != "1":
            seen_noncar = True
        if seen_noncar and value == "1":
            return True
    return False


def is_normal_home_away_home(group: pd.DataFrame) -> bool:
    group = group[group["CHAIN_ELIGIBLE"].eq(1)]
    qa_flags = [
        "duplicate_W_ID_flag",
        "trip_sequence_gap_flag",
        "time_sequence_conflict_flag",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
    ]
    return (
        len(group) >= 2
        and bool(group["HOME_ORIGIN"].eq(1).fillna(False).iloc[0])
        and bool(group["HOME_DESTINATION"].eq(0).fillna(False).any())
        and bool(group["HOME_DESTINATION"].eq(1).fillna(False).any())
        and int(group[qa_flags].sum().sum()) == 0
    )


def has_purpose_10_to_16(group: pd.DataFrame) -> bool:
    return integer_series(group["W_ZWECK"]).isin([10, 11, 12, 13, 14, 15, 16]).any()


def has_return_trip_9(group: pd.DataFrame) -> bool:
    return integer_series(group["W_ZWECK"]).eq(9).any()


def has_rbw_rows(group: pd.DataFrame) -> bool:
    return group["excluded_from_chain_reason"].eq("rbW_aggregate").any()


def has_first_chain_so1(group: pd.DataFrame, value: int) -> bool:
    return (group["CHAIN_SEQUENCE_NUM"].eq(1) & integer_series(group["W_SO1"]).eq(value)).any()


def has_purpose_99(group: pd.DataFrame) -> bool:
    return integer_series(group["W_ZWECK"]).eq(99).any()


def select_example_persons(enriched: pd.DataFrame, colmap: ColumnMap) -> list[str]:
    selected: list[str] = []

    def add_first(label: str, predicate, max_rows: int = 25) -> None:
        for key, group in enriched.groupby("PERSON_KEY", sort=False):
            if key in selected or len(group) > max_rows:
                continue
            if predicate(group):
                print(f"\nExample selected for {label}: {key}")
                selected.append(key)
                return
        print(f"\nNo compact example found for {label}.")

    purpose_num = pd.to_numeric(enriched[colmap.dest_purpose], errors="coerce")
    enriched = enriched.assign(_PURPOSE_NUM=purpose_num)

    add_first("normal home-away-home chain", is_normal_home_away_home)
    add_first("chain containing W_ZWECK values 10-16", has_purpose_10_to_16)
    add_first("return trip with W_ZWECK == 9", has_return_trip_9)
    add_first("person who also has rbW rows", has_rbw_rows, max_rows=60)
    add_first("first chain-eligible trip with W_SO1 == 810", lambda g: has_first_chain_so1(g, 810))
    add_first("genuine W_ZWECK == 99 case", has_purpose_99)
    add_first("non-car trip before a car trip", has_noncar_before_car)
    add_first(
        "sequence or time conflict",
        lambda g: g["duplicate_W_ID_flag"].eq(1).any()
        or g["trip_sequence_gap_flag"].eq(1).any()
        or g["time_sequence_conflict_flag"].eq(1).any(),
    )

    for key, group in enriched.groupby("PERSON_KEY", sort=False):
        if len(selected) >= 10:
            break
        if key not in selected and len(group) <= 10:
            selected.append(key)

    return selected[:10]


def print_example_chains(enriched: pd.DataFrame, colmap: ColumnMap) -> None:
    qa_cols = [
        "duplicate_W_ID_flag",
        "trip_sequence_gap_flag",
        "time_sequence_conflict_flag",
        "W_SO1_sequence_conflict_flag",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "negative_duration_flag",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
    ]
    display_cols = [
        "SOURCE_ROW_ID",
        "H_ID",
        *(["HP_ID"] if "HP_ID" in enriched.columns else []),
        "P_ID",
        "W_ID",
        colmap.w_rbw,
        "CHAIN_ELIGIBLE",
        "excluded_from_chain_reason",
        "CHAIN_SEQUENCE_NUM",
        colmap.w_so1,
        colmap.dest_purpose,
        "START_MIN",
        "ARRIVAL_MIN",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        *qa_cols,
    ]
    example_keys = select_example_persons(enriched, colmap)
    print("\nPHASE 1 QA EXAMPLE PERSON CHAINS")
    for idx, key in enumerate(example_keys, start=1):
        chain = enriched.loc[enriched["PERSON_KEY"].eq(key), display_cols]
        print(f"\n--- Example chain {idx}: PERSON_KEY={key}, rows={len(chain)} ---")
        with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_rows", None):
            print(chain.to_string(index=False))


def assert_phase1(enriched: pd.DataFrame, household_fleet: pd.DataFrame) -> None:
    assert enriched["SOURCE_ROW_ID"].is_unique, "SOURCE_ROW_ID is not unique."
    assert len(enriched) == len(enriched.index), "Enriched row count does not match retained Phase 1 data."
    assert set(enriched["CHAIN_ELIGIBLE"].dropna().unique()).issubset({0, 1}), "CHAIN_ELIGIBLE contains values outside 0/1."
    rbw_rows = enriched["excluded_from_chain_reason"].eq("rbW_aggregate")
    assert enriched.loc[rbw_rows, "CHAIN_ELIGIBLE"].eq(0).all(), "rbW rows must not be chain eligible."
    assert enriched.loc[rbw_rows, "trip_sequence_gap_flag"].eq(0).all(), "rbW rows must not carry sequence-gap flags."

    complete_households = household_fleet.loc[household_fleet["complete_fleet_baseline_flag"].eq(1), "H_ID"]
    invalid_complete = household_fleet[
        household_fleet["H_ID"].isin(complete_households) & household_fleet["N_RECORDED_CARS"].ne(2)
    ]
    assert invalid_complete.empty, "A complete-fleet baseline household lacks exactly two recorded A_ID values."

    for value in enriched["FLEET_CAR_SET"].dropna().unique():
        if value == "":
            continue
        parts = [int(part) for part in str(value).split("|")]
        assert parts == sorted(parts), f"FLEET_CAR_SET is not deterministically ordered: {value}"


def any_invalid_time(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return (
        df["invalid_start_time_flag"].eq(1)
        | df["invalid_arrival_time_flag"].eq(1)
        | df["negative_duration_flag"].eq(1)
    )


def identifiable_household_car_driver_mask(trips: pd.DataFrame, household_fleet: pd.DataFrame) -> pd.Series:
    fleet_pairs: set[tuple[str, int]] = set()
    for _, row in household_fleet.iterrows():
        car_set = str(row["FLEET_CAR_SET"])
        if not car_set:
            continue
        for part in car_set.split("|"):
            fleet_pairs.add((str(row["H_ID"]), int(part)))

    w_wauto = integer_series(trips["W_WAUTO"]) if "W_WAUTO" in trips.columns else pd.Series(np.nan, index=trips.index)
    pkw_fmf = integer_series(trips["pkw_fmf"]) if "pkw_fmf" in trips.columns else pd.Series(np.nan, index=trips.index)
    in_fleet = pd.Series(
        [
            (str(h_id), int(a_id)) in fleet_pairs if not pd.isna(a_id) else False
            for h_id, a_id in zip(trips["H_ID"], w_wauto)
        ],
        index=trips.index,
    )
    return trips["CHAIN_ELIGIBLE"].eq(1) & pkw_fmf.eq(1) & w_wauto.isin([1, 2, 3]) & in_fleet


def read_previous_phase1_counts(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    cols = [
        "PERSON_KEY",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "negative_duration_flag",
        "trip_sequence_gap_flag",
    ]
    try:
        previous = pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False, na_values=[])
    except ValueError:
        return None
    for col in cols:
        if col != "PERSON_KEY":
            previous[col] = integer_series(previous[col]).fillna(0).astype(int)

    return {
        "unknown HOME_ORIGIN": int(previous["home_origin_unknown_flag"].sum()),
        "unknown HOME_DESTINATION": int(previous["home_destination_unknown_flag"].sum()),
        "invalid-time rows": int(any_invalid_time(previous).sum()),
        "trip-sequence-gap person chains": int(
            previous.groupby("PERSON_KEY", sort=False)["trip_sequence_gap_flag"].max().sum()
        ),
    }


def print_before_after(previous_counts: dict[str, int] | None, revised: pd.DataFrame) -> None:
    revised_counts = {
        "unknown HOME_ORIGIN": int(revised["home_origin_unknown_flag"].sum()),
        "unknown HOME_DESTINATION": int(revised["home_destination_unknown_flag"].sum()),
        "invalid-time rows": int(any_invalid_time(revised).sum()),
        "trip-sequence-gap person chains": int(
            revised.groupby("PERSON_KEY", sort=False)["trip_sequence_gap_flag"].max().sum()
        ),
    }
    print("\nBEFORE-AFTER PHASE 1 COMPARISON")
    rows = []
    for metric, revised_count in revised_counts.items():
        previous_count = previous_counts.get(metric) if previous_counts else pd.NA
        delta = revised_count - previous_count if previous_counts else pd.NA
        rows.append({"metric": metric, "previous_count": previous_count, "revised_count": revised_count, "delta": delta})
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    trips_raw = read_csv_strings(TRIPS_PATH)
    cars_raw = read_csv_strings(CARS_PATH)
    colmap = build_column_map(trips_raw)
    previous_counts = read_previous_phase1_counts(OUT_TRIPS_PATH)

    trips_raw.insert(0, "SOURCE_ROW_ID", np.arange(len(trips_raw), dtype=np.int64))
    assert trips_raw["SOURCE_ROW_ID"].is_unique, "SOURCE_ROW_ID is not unique after creation."

    household_counts = household_car_counts(trips_raw, colmap)
    household_fleet = construct_recorded_fleet(cars_raw, household_counts)

    retained_households = household_fleet.loc[
        household_fleet["household_car_count_conflict_flag"].eq(0)
        & household_fleet["H_ANZAUTO_NUM"].isin([2, 3]),
        "H_ID",
    ]
    retained = trips_raw[trips_raw["H_ID"].isin(set(retained_households))].copy()
    retained = retained.merge(
        household_fleet[
            [
                "H_ID",
                "H_ANZAUTO_HH",
                "FLEET_CAR_SET",
                "N_RECORDED_CARS",
                "complete_fleet_baseline_flag",
                "topcoded_multicar_household_flag",
                "fleet_inventory_mismatch_flag",
                "household_car_count_conflict_flag",
                "nonstandard_A_ID_set_flag",
            ]
        ],
        on="H_ID",
        how="left",
        validate="many_to_one",
    )
    retained[colmap.h_anzauto] = retained["H_ANZAUTO_HH"]
    retained = retained.drop(columns=["H_ANZAUTO_HH"])

    retained = parse_trip_times(retained, colmap)
    enriched = reconstruct_person_chains(retained, colmap)
    assert_phase1(enriched, household_fleet)

    qa_summary = build_qa_summary(trips_raw, enriched, household_fleet)

    OUT_TRIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_TRIPS_PATH, index=False)
    qa_summary.to_csv(OUT_QA_PATH, index=False)

    print_before_after(previous_counts, enriched)
    print_example_chains(enriched, colmap)

    print("\nPHASE 1 QA SUMMARY")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(qa_summary.to_string(index=False, formatters={"share": "{:.6f}".format}))

    unresolved = {
        "household car-count conflicts": int(household_fleet["household_car_count_conflict_flag"].sum()),
        "two-car fleet inventory mismatches": int(household_fleet["fleet_inventory_mismatch_flag"].sum()),
        "nonstandard A_ID sets": int(household_fleet["nonstandard_A_ID_set_flag"].sum()),
        "time-sequence-conflict chain-eligible rows": int(
            enriched.loc[enriched["CHAIN_ELIGIBLE"].eq(1), "time_sequence_conflict_flag"].sum()
        ),
        "unknown home-origin chain-eligible rows": int(
            enriched.loc[enriched["CHAIN_ELIGIBLE"].eq(1), "home_origin_unknown_flag"].sum()
        ),
        "unknown home-destination chain-eligible rows": int(
            enriched.loc[enriched["CHAIN_ELIGIBLE"].eq(1), "home_destination_unknown_flag"].sum()
        ),
        "invalid-time chain-eligible rows": int(any_invalid_time(enriched[enriched["CHAIN_ELIGIBLE"].eq(1)]).sum()),
    }
    print("\nPHASE 1 OUTPUT PATHS")
    print(f"trips_home_chain_enriched: {OUT_TRIPS_PATH}")
    print(f"phase1_QA_summary: {OUT_QA_PATH}")
    print("\nFINAL PHASE 1 COUNTS")
    print(f"retained trip rows: {len(enriched):,}")
    print(f"retained households: {enriched['H_ID'].nunique():,}")
    print("\nUNRESOLVED QA ISSUES")
    for label, count in unresolved.items():
        print(f"{label}: {count:,}")
    print("\nPhase 1 complete. Stopping before Phase 2.")


if __name__ == "__main__":
    main()
