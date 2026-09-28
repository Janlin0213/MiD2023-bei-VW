from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import PHASE1_DIR, SELECTED_RAW_DIR

ENRICHED_PATH = PHASE1_DIR / "trips_home_chain_enriched.csv"
CARS_PATH = SELECTED_RAW_DIR / "cars_selected_raw.csv"
TIME_MISSING_STRINGS = {"", "nan", "NaN", "NA", "N/A", "None"}


def read_required_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [col for col in columns if col not in header]
    if missing:
        raise ValueError(f"{path.name} is missing required diagnostic column(s): {missing}")
    df = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    if df.empty:
        print("(none)")
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
            print(df.to_string(index=False))


def count_share(labels: pd.Series, denominator: int) -> pd.DataFrame:
    counts = labels.value_counts(dropna=False).rename_axis("reason").reset_index(name="count")
    counts["share_of_group"] = counts["count"] / max(denominator, 1)
    return counts


def raw_distribution(df: pd.DataFrame, column: str, n: int = 30) -> pd.DataFrame:
    out = df[column].fillna("<NA>").replace("", "<blank>").value_counts(dropna=False).head(n).reset_index()
    out.columns = [column, "count"]
    out["share"] = out["count"] / max(len(df), 1)
    return out


def build_base_frame() -> pd.DataFrame:
    columns = [
        "SOURCE_ROW_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "W_SO1",
        "W_ZWECK",
        "W_SZ",
        "W_AZ",
        "W_FOLGETAG",
        "pkw_fmf",
        "W_WAUTO",
        "START_MIN",
        "ARRIVAL_MIN",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
        "trip_sequence_gap_flag",
        "time_sequence_conflict_flag",
        "FLEET_CAR_SET",
    ]
    df = read_required_columns(ENRICHED_PATH, columns)
    numeric_cols = [
        "SOURCE_ROW_ID",
        "W_ID",
        "W_SO1",
        "W_ZWECK",
        "W_FOLGETAG",
        "pkw_fmf",
        "W_WAUTO",
        "START_MIN",
        "ARRIVAL_MIN",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "invalid_start_time_flag",
        "invalid_arrival_time_flag",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
        "trip_sequence_gap_flag",
        "time_sequence_conflict_flag",
    ]
    for col in numeric_cols:
        df[f"{col}_NUM"] = as_num(df[col])
    df["PERSON_KEY_DIAG"] = np.where(
        df["HP_ID"].ne(""),
        "HP|" + df["HP_ID"],
        "HHP|" + df["H_ID"] + "|" + df["P_ID"],
    )
    df = df.sort_values(["H_ID", "HP_ID", "P_ID", "W_ID_NUM", "SOURCE_ROW_ID_NUM"], kind="mergesort")
    df["ROW_IN_PERSON"] = df.groupby("PERSON_KEY_DIAG", sort=False).cumcount() + 1
    df["PREV_HOME_ORIGIN_NUM"] = df.groupby("PERSON_KEY_DIAG", sort=False)["HOME_ORIGIN_NUM"].shift(1)
    df["PREV_HOME_DESTINATION_NUM"] = df.groupby("PERSON_KEY_DIAG", sort=False)["HOME_DESTINATION_NUM"].shift(1)
    df["PREV_W_ID_NUM"] = df.groupby("PERSON_KEY_DIAG", sort=False)["W_ID_NUM"].shift(1)
    return df


def recorded_fleet_pairs() -> set[tuple[str, int]]:
    cars = read_required_columns(CARS_PATH, ["H_ID", "A_ID"])
    cars["A_ID_NUM"] = as_num(cars["A_ID"])
    cars = cars[cars["A_ID_NUM"].isin([1, 2, 3])]
    return set(zip(cars["H_ID"].astype(str), cars["A_ID_NUM"].astype(int)))


def diagnose_unknown_origins(df: pd.DataFrame) -> pd.DataFrame:
    unknown = df[df["home_origin_unknown_flag_NUM"].eq(1)].copy()
    conditions = [
        unknown["ROW_IN_PERSON"].eq(1) & ~unknown["W_SO1_NUM"].isin([1, 2]),
        unknown["ROW_IN_PERSON"].gt(1) & unknown["PREV_HOME_DESTINATION_NUM"].isna(),
    ]
    choices = [
        "first trip with missing or invalid W_SO1",
        "propagated from previous unknown HOME_DESTINATION",
    ]
    unknown["reason"] = np.select(conditions, choices, default="other reasons")
    return count_share(unknown["reason"], len(unknown))


def diagnose_unknown_destinations(df: pd.DataFrame) -> pd.DataFrame:
    unknown = df[df["home_destination_unknown_flag_NUM"].eq(1)].copy()
    valid_purpose = unknown["W_ZWECK_NUM"].isin(list(range(1, 10)))
    conditions = [
        ~valid_purpose,
        unknown["W_ZWECK_NUM"].eq(9) & unknown["ROW_IN_PERSON"].eq(1),
        unknown["W_ZWECK_NUM"].eq(9) & unknown["ROW_IN_PERSON"].gt(1) & unknown["PREV_HOME_ORIGIN_NUM"].isna(),
    ]
    choices = [
        "missing or invalid W_ZWECK",
        "W_ZWECK == 9 without a preceding trip",
        "W_ZWECK == 9 with unknown preceding HOME_ORIGIN",
    ]
    unknown["reason"] = np.select(conditions, choices, default="other reasons")
    return count_share(unknown["reason"], len(unknown))


def explain_time_parsing(df: pd.DataFrame) -> None:
    start_pattern = df["W_SZ"].str.match(r"^\d{1,2}:\d{1,2}(?::\d{1,2})?$", na=False)
    arrival_pattern = df["W_AZ"].str.match(r"^\d{1,2}:\d{1,2}(?::\d{1,2})?$", na=False)
    start_nonblank = ~df["W_SZ"].isin(TIME_MISSING_STRINGS)
    arrival_nonblank = ~df["W_AZ"].isin(TIME_MISSING_STRINGS)
    start_rate = start_pattern[start_nonblank].mean()
    arrival_rate = arrival_pattern[arrival_nonblank].mean()
    print("\nTIME PARSING METHOD")
    print("Phase 1 used combined variables W_SZ and W_AZ.")
    print("Parser: H:MM[:SS] or HH:MM[:SS] -> hour * 60 + minute; seconds validated but not added.")
    print("If W_FOLGETAG == 1, Phase 1 added 1440 minutes to ARRIVAL_MIN.")
    print(f"W_SZ combined parse-pattern rate among nonblank retained trips: {start_rate:.6f}")
    print(f"W_AZ combined parse-pattern rate among nonblank retained trips: {arrival_rate:.6f}")
    print("Separate hour/minute variables were not used because the combined variables were reliably parseable.")


def invalid_time_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    invalid = df[df["invalid_start_time_flag_NUM"].eq(1) | df["invalid_arrival_time_flag_NUM"].eq(1)].copy()
    labels = np.select(
        [
            invalid["invalid_start_time_flag_NUM"].eq(1) & invalid["invalid_arrival_time_flag_NUM"].eq(1),
            invalid["invalid_start_time_flag_NUM"].eq(1) & invalid["invalid_arrival_time_flag_NUM"].eq(0),
            invalid["invalid_start_time_flag_NUM"].eq(0) & invalid["invalid_arrival_time_flag_NUM"].eq(1),
        ],
        ["both START_MIN and ARRIVAL_MIN invalid", "only START_MIN invalid", "only ARRIVAL_MIN invalid"],
        default="other",
    )
    return count_share(pd.Series(labels), len(invalid))


def invalid_time_combinations(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    invalid = df[df["invalid_start_time_flag_NUM"].eq(1) | df["invalid_arrival_time_flag_NUM"].eq(1)].copy()
    combo_cols = ["W_SZ", "W_AZ", "W_FOLGETAG", "invalid_start_time_flag", "invalid_arrival_time_flag"]
    out = invalid.groupby(combo_cols, dropna=False).size().reset_index(name="count")
    out["share_of_invalid_time_rows"] = out["count"] / max(len(invalid), 1)
    return out.sort_values(["count", *combo_cols], ascending=[False, True, True, True, True, True]).head(n)


def subgroup_rates(df: pd.DataFrame) -> pd.DataFrame:
    fleet_pairs = recorded_fleet_pairs()
    chosen_pair = list(zip(df["H_ID"].astype(str), df["W_WAUTO_NUM"]))
    in_recorded_fleet = pd.Series([
        (h, int(aid)) in fleet_pairs if not pd.isna(aid) else False for h, aid in chosen_pair
    ], index=df.index)
    identifiable = df["pkw_fmf_NUM"].eq(1) & df["W_WAUTO_NUM"].isin([1, 2, 3]) & in_recorded_fleet
    groups = {
        "all trips": pd.Series(True, index=df.index),
        "pkw_fmf == 1 trips": df["pkw_fmf_NUM"].eq(1),
        "identifiable household-car driver trips": identifiable,
        "home-origin household-car driver trips": identifiable & df["HOME_ORIGIN_NUM"].eq(1),
    }

    rows = []
    for label, mask in groups.items():
        sub = df[mask]
        n = len(sub)
        rows.append(
            {
                "group": label,
                "rows": n,
                "invalid_start_time_count": int(sub["invalid_start_time_flag_NUM"].eq(1).sum()),
                "invalid_start_time_rate": sub["invalid_start_time_flag_NUM"].eq(1).mean() if n else 0,
                "invalid_arrival_time_count": int(sub["invalid_arrival_time_flag_NUM"].eq(1).sum()),
                "invalid_arrival_time_rate": sub["invalid_arrival_time_flag_NUM"].eq(1).mean() if n else 0,
                "any_invalid_time_count": int(
                    (sub["invalid_start_time_flag_NUM"].eq(1) | sub["invalid_arrival_time_flag_NUM"].eq(1)).sum()
                ),
                "any_invalid_time_rate": (
                    (sub["invalid_start_time_flag_NUM"].eq(1) | sub["invalid_arrival_time_flag_NUM"].eq(1)).mean()
                    if n
                    else 0
                ),
                "unknown_HOME_ORIGIN_count": int(sub["home_origin_unknown_flag_NUM"].eq(1).sum()),
                "unknown_HOME_ORIGIN_rate": sub["home_origin_unknown_flag_NUM"].eq(1).mean() if n else 0,
                "unknown_HOME_DESTINATION_count": int(sub["home_destination_unknown_flag_NUM"].eq(1).sum()),
                "unknown_HOME_DESTINATION_rate": sub["home_destination_unknown_flag_NUM"].eq(1).mean() if n else 0,
                "any_unknown_home_status_count": int(
                    (sub["home_origin_unknown_flag_NUM"].eq(1) | sub["home_destination_unknown_flag_NUM"].eq(1)).sum()
                ),
                "any_unknown_home_status_rate": (
                    (sub["home_origin_unknown_flag_NUM"].eq(1) | sub["home_destination_unknown_flag_NUM"].eq(1)).mean()
                    if n
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def gap_distribution(df: pd.DataFrame) -> pd.DataFrame:
    person_gaps = df[df["W_ID_NUM"].notna() & df["PREV_W_ID_NUM"].notna()].copy()
    person_gaps["GAP_SIZE"] = person_gaps["W_ID_NUM"] - person_gaps["PREV_W_ID_NUM"] - 1
    person_gaps = person_gaps[person_gaps["GAP_SIZE"].gt(0)]
    out = person_gaps["GAP_SIZE"].astype(int).value_counts().sort_index().reset_index()
    out.columns = ["missing_W_ID_gap_size", "occurrences"]
    out["share_of_gap_occurrences"] = out["occurrences"] / max(len(person_gaps), 1)
    return out


def print_gap_chains(df: pd.DataFrame, n: int = 10) -> None:
    gap_rows = df[df["W_ID_NUM"].notna() & df["PREV_W_ID_NUM"].notna()].copy()
    gap_rows["GAP_SIZE"] = gap_rows["W_ID_NUM"] - gap_rows["PREV_W_ID_NUM"] - 1
    keys = gap_rows[gap_rows["GAP_SIZE"].gt(0)]["PERSON_KEY_DIAG"].drop_duplicates().head(n).tolist()
    cols = [
        "SOURCE_ROW_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "W_ID",
        "W_SO1",
        "W_ZWECK",
        "W_SZ",
        "W_AZ",
        "START_MIN",
        "ARRIVAL_MIN",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "trip_sequence_gap_flag",
        "time_sequence_conflict_flag",
        "home_origin_unknown_flag",
        "home_destination_unknown_flag",
    ]
    print("\nTEN REPRESENTATIVE TRIP-SEQUENCE-GAP CHAINS")
    for i, key in enumerate(keys, start=1):
        chain = df[df["PERSON_KEY_DIAG"].eq(key)][cols]
        print(f"\n--- Gap chain {i}: PERSON_KEY={key}, rows={len(chain)} ---")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
            print(chain.to_string(index=False))


def main() -> None:
    df = build_base_frame()

    origin_unknown = df[df["home_origin_unknown_flag_NUM"].eq(1)]
    destination_unknown = df[df["home_destination_unknown_flag_NUM"].eq(1)]

    print(f"Read-only diagnostics over: {ENRICHED_PATH}")
    print(f"Rows diagnosed: {len(df):,}")

    print_table("UNKNOWN HOME_ORIGIN BREAKDOWN", diagnose_unknown_origins(df))
    print_table("UNKNOWN HOME_DESTINATION BREAKDOWN", diagnose_unknown_destinations(df))

    print_table("RAW W_SO1 DISTRIBUTION FOR UNKNOWN HOME_ORIGIN ROWS", raw_distribution(origin_unknown, "W_SO1", 30))
    print_table("RAW W_ZWECK DISTRIBUTION FOR UNKNOWN HOME_DESTINATION ROWS", raw_distribution(destination_unknown, "W_ZWECK", 30))
    print_table("RAW W_ZWECK DISTRIBUTION FOR UNKNOWN HOME_ORIGIN ROWS", raw_distribution(origin_unknown, "W_ZWECK", 30))
    print_table("RAW W_SO1 DISTRIBUTION FOR UNKNOWN HOME_DESTINATION ROWS", raw_distribution(destination_unknown, "W_SO1", 30))

    explain_time_parsing(df)
    print_table("INVALID TIME BREAKDOWN", invalid_time_breakdown(df))
    print_table("MOST FREQUENT RAW TIME COMBINATIONS CAUSING INVALID FLAGS", invalid_time_combinations(df, 25))
    print_table("INVALID-TIME AND UNKNOWN-HOME-STATUS RATES BY TRIP GROUP", subgroup_rates(df))
    print_table("TRIP-SEQUENCE GAP SIZE DISTRIBUTION", gap_distribution(df))
    print_gap_chains(df, 10)


if __name__ == "__main__":
    main()
