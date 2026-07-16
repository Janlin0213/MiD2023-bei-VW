"""Construct reconstructed household type from MiD person age groups.

This script adds a household-level ``household_type`` column to the selected
household CSV using person-level ``alter_gr5`` from the selected person CSV.

By default, it skips reconstruction when ``household_type`` already exists.
Use ``--force`` to replace an existing column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ALTER_GR5_CHILD = {1, 2, 3}
ALTER_GR5_YOUNG = {4, 5, 6}
ALTER_GR5_MIDDLE = {7, 8, 9, 10, 11, 12}
ALTER_GR5_OLD = {13, 14, 15}
ALTER_GR5_ADULT = ALTER_GR5_YOUNG | ALTER_GR5_MIDDLE | ALTER_GR5_OLD

HOUSEHOLD_TYPE_ORDER = [
    "family_household",
    "young_household",
    "adult_household",
    "senior_household",
    "unknown",
]


def classify_mid_hh_type(age_group_codes: pd.Series) -> str:
    """Classify one household from person-level ``alter_gr5`` codes."""
    codes = [int(code) for code in age_group_codes.dropna()]

    if len(codes) == 0:
        return "unknown"

    has_child = any(code in ALTER_GR5_CHILD for code in codes)
    all_18_34 = all(code in ALTER_GR5_YOUNG for code in codes)
    all_65plus = all(code in ALTER_GR5_OLD for code in codes)
    all_adults = all(code in ALTER_GR5_ADULT for code in codes)
    has_middle = any(code in ALTER_GR5_MIDDLE for code in codes)
    has_young = any(code in ALTER_GR5_YOUNG for code in codes)
    has_old = any(code in ALTER_GR5_OLD for code in codes)

    if has_child:
        return "family_household"
    if all_18_34:
        return "young_household"
    if all_65plus:
        return "senior_household"
    if all_adults and (has_middle or (has_young and has_old)):
        return "adult_household"
    return "unknown"


def resolve_first_existing(paths: list[Path]) -> Path:
    """Return the first existing path from a list of candidates."""
    for path in paths:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the candidate files exists: {candidates}")


def build_household_type(persons: pd.DataFrame) -> pd.DataFrame:
    """Return ``H_ID`` and reconstructed ``household_type``."""
    required = {"H_ID", "alter_gr5"}
    missing = required.difference(persons.columns)
    if missing:
        raise KeyError(f"Missing required person column(s): {sorted(missing)}")

    person_age_groups = persons.loc[persons["H_ID"].notna(), ["H_ID", "alter_gr5"]].copy()
    person_age_groups["alter_gr5"] = pd.to_numeric(
        person_age_groups["alter_gr5"], errors="coerce"
    ).astype("Int64")

    return (
        person_age_groups.groupby("H_ID", as_index=False)["alter_gr5"]
        .agg(classify_mid_hh_type)
        .rename(columns={"alter_gr5": "household_type"})
    )


def add_household_type(
    households_path: Path,
    persons_path: Path,
    *,
    force: bool = False,
) -> pd.Series:
    """Add ``household_type`` to the household CSV and return value counts."""
    households = pd.read_csv(households_path, dtype={"H_ID": "string"})

    if "household_type" in households.columns and not force:
        return (
            households["household_type"]
            .value_counts(dropna=False)
            .reindex(HOUSEHOLD_TYPE_ORDER, fill_value=0)
        )

    persons = pd.read_csv(persons_path, usecols=["H_ID", "alter_gr5"], dtype={"H_ID": "string"})
    household_type = build_household_type(persons)

    households = households.drop(columns=["household_type"], errors="ignore")
    households = households.merge(household_type, on="H_ID", how="left")
    households["household_type"] = households["household_type"].fillna("unknown")
    households.to_csv(households_path, index=False)

    return (
        households["household_type"]
        .value_counts(dropna=False)
        .reindex(HOUSEHOLD_TYPE_ORDER, fill_value=0)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--households-csv",
        type=Path,
        default=None,
        help="Selected household CSV. Defaults to data_processed/01_hh_selected_raw.csv if present, otherwise data_processed/hh_selected_raw.csv.",
    )
    parser.add_argument(
        "--persons-csv",
        type=Path,
        default=Path("data_processed/persons_selected_raw.csv"),
        help="Selected person CSV with H_ID and alter_gr5.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace household_type if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    households_path = args.households_csv or resolve_first_existing(
        [
            Path("data_processed/01_hh_selected_raw.csv"),
            Path("data_processed/hh_selected_raw.csv"),
        ]
    )

    counts = add_household_type(
        households_path=households_path,
        persons_path=args.persons_csv,
        force=args.force,
    )
    print(f"household_type counts in {households_path}:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
