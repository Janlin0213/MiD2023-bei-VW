"""Build the reusable Theme 1 household-level analytical backbone.

The canonical household universe comes from ``hh_selected_raw.csv``. Person
licence information is classified once and aggregated before a validated
one-to-one merge. Raw/selected inputs are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

HOUSEHOLDS_PATH = ROOT / "data_processed" / "selected_raw" / "hh_selected_raw.csv"
PERSONS_PATH = ROOT / "data_processed" / "selected_raw" / "persons_selected_raw.csv"

BACKBONE_DIR = ROOT / "data_processed" / "eda" / "statistical_test"
BACKBONE_PATH = BACKBONE_DIR / "theme1_household_backbone.csv"
QA_PATH = BACKBONE_DIR / "theme1_household_backbone_QA.csv"

HOUSEHOLD_COLUMNS = [
    "H_ID",
    "H_GEW",
    "H_ANZAUTO",
    "RegioStaR4",
    "household_type",
]
PERSON_COLUMNS = ["H_ID", "P_ID", "P_FS_PKW"]

CAR_GROUP_MAP = {1: "single_car", 2: "multi_car", 3: "multi_car"}
REGIOSTAR4_ORDER = [11, 12, 21, 22]
HOUSEHOLD_TYPE_ORDER = [
    "family_household",
    "young_household",
    "adult_household",
    "senior_household",
    "unknown",
]
LICENSE_STATE_MAP = {
    1: "licensed",
    2: "explicit_no",
    403: "structural_not_eligible",
    9: "unresolved",
    206: "unresolved",
}
LICENSE_STATES = [
    "licensed",
    "explicit_no",
    "structural_not_eligible",
    "unresolved",
]


def require_columns(columns: list[str], required: list[str], source: str) -> None:
    """Raise a clear error when a required upstream column is absent."""
    missing = sorted(set(required).difference(columns))
    if not missing:
        return
    if source == "household" and "household_type" in missing:
        raise RuntimeError(
            "hh_selected_raw.csv is missing 'household_type'. Run the upstream "
            "household-type construction step (src/build_household_type.py) first."
        )
    raise KeyError(f"{source} input is missing required column(s): {missing}")


def numeric(series: pd.Series) -> pd.Series:
    """Parse a series without silently replacing invalid values."""
    return pd.to_numeric(series, errors="coerce")


def qa_row(
    section: str,
    metric: str,
    *,
    category: Any = "",
    count: Any = np.nan,
    share: Any = np.nan,
    value: Any = np.nan,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "section": section,
        "metric": metric,
        "category": category,
        "count": count,
        "share": share,
        "value": value,
        "notes": notes,
    }


def distribution_rows(
    section: str,
    metric: str,
    series: pd.Series,
    *,
    denominator: int | None = None,
    notes: str = "",
) -> list[dict[str, Any]]:
    """Return deterministic tidy QA rows for a raw or derived distribution."""
    total = len(series) if denominator is None else denominator
    counts = series.value_counts(dropna=False)
    rows: list[dict[str, Any]] = []
    for category, count in counts.items():
        label = "<missing>" if pd.isna(category) else category
        rows.append(
            qa_row(
                section,
                metric,
                category=label,
                count=int(count),
                share=(float(count) / total if total else np.nan),
                notes=notes,
            )
        )
    return rows


def valid_weight_mask(weight: pd.Series) -> pd.Series:
    values = numeric(weight)
    return values.notna() & np.isfinite(values) & values.gt(0)


def weighted_share(mask: pd.Series, universe: pd.Series, weights: pd.Series) -> float:
    valid = universe & valid_weight_mask(weights)
    denominator = float(numeric(weights.loc[valid]).sum())
    if denominator <= 0:
        return np.nan
    numerator = float(numeric(weights.loc[valid & mask]).sum())
    return numerator / denominator


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HOUSEHOLDS_PATH.exists():
        raise FileNotFoundError(f"Household input not found: {HOUSEHOLDS_PATH}")
    if not PERSONS_PATH.exists():
        raise FileNotFoundError(f"Person input not found: {PERSONS_PATH}")

    household_header = pd.read_csv(HOUSEHOLDS_PATH, nrows=0).columns.tolist()
    person_header = pd.read_csv(PERSONS_PATH, nrows=0).columns.tolist()
    require_columns(household_header, HOUSEHOLD_COLUMNS, "household")
    require_columns(person_header, PERSON_COLUMNS, "person")

    households = pd.read_csv(
        HOUSEHOLDS_PATH,
        usecols=HOUSEHOLD_COLUMNS,
        dtype={"H_ID": "string", "H_GEW": "string", "household_type": "string"},
        low_memory=False,
    )
    persons = pd.read_csv(
        PERSONS_PATH,
        usecols=PERSON_COLUMNS,
        dtype={"H_ID": "string", "P_ID": "string"},
        low_memory=False,
    )
    return households, persons


def validate_households(households: pd.DataFrame) -> None:
    if households["H_ID"].isna().any():
        sample = households.loc[households["H_ID"].isna()].head(10)
        raise RuntimeError(f"Household H_ID contains missing values:\n{sample}")
    duplicates = households["H_ID"].duplicated(keep=False)
    if duplicates.any():
        sample = households.loc[duplicates].sort_values("H_ID").head(10)
        raise RuntimeError(f"Household H_ID is not unique:\n{sample}")


def build_person_summary(persons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if persons["H_ID"].isna().any() or persons["P_ID"].isna().any():
        sample = persons.loc[persons[["H_ID", "P_ID"]].isna().any(axis=1)].head(10)
        raise RuntimeError(f"Person input has missing H_ID/P_ID keys:\n{sample}")
    duplicate_people = persons.duplicated(["H_ID", "P_ID"], keep=False)
    if duplicate_people.any():
        sample = persons.loc[duplicate_people].sort_values(["H_ID", "P_ID"]).head(10)
        raise RuntimeError(f"Person H_ID/P_ID keys are not unique:\n{sample}")

    persons = persons.copy()
    persons["P_FS_PKW_numeric"] = numeric(persons["P_FS_PKW"])
    persons["license_state"] = persons["P_FS_PKW_numeric"].map(LICENSE_STATE_MAP)

    unknown_code = persons["license_state"].isna()
    if unknown_code.any():
        diagnostic = (
            persons.loc[unknown_code, "P_FS_PKW"]
            .value_counts(dropna=False)
            .rename_axis("P_FS_PKW")
            .reset_index(name="count")
        )
        raise RuntimeError(
            "Unmapped P_FS_PKW values were observed. Establish their MiD meaning "
            f"before continuing:\n{diagnostic.head(20).to_string(index=False)}"
        )

    state_indicators = pd.get_dummies(persons["license_state"])
    state_indicators = state_indicators.reindex(columns=LICENSE_STATES, fill_value=0)
    if not state_indicators.sum(axis=1).eq(1).all():
        sample = persons.loc[~state_indicators.sum(axis=1).eq(1)].head(10)
        raise AssertionError(f"Each person must have exactly one licence state:\n{sample}")

    persons = pd.concat([persons, state_indicators.astype("int8")], axis=1)
    summary = (
        persons.groupby("H_ID", as_index=False, sort=False)
        .agg(
            n_person_records=("P_ID", "size"),
            n_licensed_drivers=("licensed", "sum"),
            n_license_explicit_no=("explicit_no", "sum"),
            n_license_structural_not_eligible=("structural_not_eligible", "sum"),
            n_license_unresolved=("unresolved", "sum"),
        )
    )
    check_total = summary[LICENSE_COUNT_COLUMNS].sum(axis=1)
    if not check_total.eq(summary["n_person_records"]).all():
        sample = summary.loc[~check_total.eq(summary["n_person_records"])].head(10)
        raise AssertionError(f"Person-to-household licence aggregation failed:\n{sample}")
    if summary["H_ID"].duplicated().any():
        raise AssertionError("Person aggregation did not produce one row per H_ID.")
    return summary, persons


LICENSE_COUNT_COLUMNS = [
    "n_licensed_drivers",
    "n_license_explicit_no",
    "n_license_structural_not_eligible",
    "n_license_unresolved",
]


def construct_backbone(
    households: pd.DataFrame,
    persons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_households(households)
    person_summary, classified_persons = build_person_summary(persons)

    household_ids = set(households["H_ID"])
    orphan_person_summary = person_summary.loc[~person_summary["H_ID"].isin(household_ids)].copy()
    mergeable_summary = person_summary.loc[person_summary["H_ID"].isin(household_ids)].copy()

    backbone = households.merge(
        mergeable_summary,
        on="H_ID",
        how="left",
        validate="one_to_one",
        indicator="_person_merge",
    )
    if len(backbone) != len(households):
        raise AssertionError("Household merge changed the household row count.")
    if backbone["H_ID"].duplicated().any():
        raise AssertionError("Household merge duplicated H_ID values.")

    backbone["person_roster_present_flag"] = backbone["_person_merge"].eq("both").astype("int8")
    backbone = backbone.drop(columns="_person_merge")
    for column in ["n_person_records", *LICENSE_COUNT_COLUMNS]:
        backbone[column] = numeric(backbone[column]).astype("Int64")

    backbone["license_information_complete_flag"] = (
        backbone["person_roster_present_flag"].eq(1)
        & backbone["n_license_unresolved"].eq(0)
    ).astype("int8")

    backbone["H_GEW"] = numeric(backbone["H_GEW"])
    backbone["H_ANZAUTO"] = numeric(backbone["H_ANZAUTO"]).astype("Int64")
    backbone["RegioStaR4"] = numeric(backbone["RegioStaR4"]).astype("Int64")

    backbone["car_ownership_group"] = backbone["H_ANZAUTO"].map(CAR_GROUP_MAP)
    backbone["exact_car_count_flag"] = backbone["H_ANZAUTO"].isin([1, 2]).astype("int8")

    licensed = numeric(backbone["n_licensed_drivers"])
    exact_denominator = numeric(backbone["H_ANZAUTO"]).where(
        backbone["exact_car_count_flag"].eq(1)
    )
    backbone["driver_car_ratio_exact"] = licensed / exact_denominator
    backbone["driver_car_ratio_exact_flag"] = backbone["exact_car_count_flag"].copy()

    capped_denominator = numeric(backbone["H_ANZAUTO"]).where(
        backbone["H_ANZAUTO"].isin([1, 2, 3])
    )
    backbone["driver_car_ratio_capped3_upper_bound"] = licensed / capped_denominator

    ordered_columns = [
        "H_ID",
        "H_GEW",
        "H_ANZAUTO",
        "car_ownership_group",
        "exact_car_count_flag",
        "RegioStaR4",
        "household_type",
        "n_person_records",
        "person_roster_present_flag",
        "n_licensed_drivers",
        "n_license_explicit_no",
        "n_license_structural_not_eligible",
        "n_license_unresolved",
        "license_information_complete_flag",
        "driver_car_ratio_exact",
        "driver_car_ratio_capped3_upper_bound",
        "driver_car_ratio_exact_flag",
    ]
    backbone = backbone[ordered_columns]

    if len(backbone) != backbone["H_ID"].nunique(dropna=False):
        raise AssertionError("Backbone is not exactly one row per H_ID.")
    return backbone, classified_persons, orphan_person_summary


def build_qa(
    households: pd.DataFrame,
    backbone: pd.DataFrame,
    classified_persons: pd.DataFrame,
    orphan_person_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_households = len(backbone)

    rows.extend(
        [
            qa_row("household_structure", "input household rows", count=len(households)),
            qa_row(
                "household_structure",
                "unique input H_ID",
                count=households["H_ID"].nunique(dropna=False),
            ),
            qa_row("household_structure", "output backbone rows", count=n_households),
            qa_row(
                "household_structure",
                "duplicate output H_ID",
                count=int(backbone["H_ID"].duplicated(keep=False).sum()),
            ),
            qa_row(
                "household_structure",
                "households without person records",
                count=int(backbone["person_roster_present_flag"].eq(0).sum()),
                share=float(backbone["person_roster_present_flag"].eq(0).mean()),
                notes="Retained in backbone; licence completeness set to 0.",
            ),
            qa_row(
                "household_structure",
                "orphan person household IDs",
                count=len(orphan_person_summary),
                notes="Person H_ID values absent from canonical household input; not merged.",
            ),
        ]
    )

    car = backbone["H_ANZAUTO"]
    multi = car.isin([2, 3])
    rows.extend(
        [
            qa_row("car_ownership", "zero-car households", count=int(car.eq(0).sum()), share=float(car.eq(0).mean())),
            qa_row("car_ownership", "single-car households", count=int(car.eq(1).sum()), share=float(car.eq(1).mean())),
            qa_row("car_ownership", "exactly-two-car households", count=int(car.eq(2).sum()), share=float(car.eq(2).mean())),
            qa_row("car_ownership", "3+-car households", count=int(car.eq(3).sum()), share=float(car.eq(3).mean()), notes="H_ANZAUTO=3 means three or more cars."),
            qa_row("car_ownership", "all multi-car households", count=int(multi.sum()), share=float(multi.mean())),
            qa_row("car_ownership", "missing/invalid H_ANZAUTO", count=int((car.isna() | ~car.isin([0, 1, 2, 3])).sum())),
            qa_row(
                "car_ownership",
                "unweighted share of 3+ among multi-car households",
                share=float(car.eq(3).sum() / multi.sum()) if multi.any() else np.nan,
            ),
            qa_row(
                "car_ownership",
                "H_GEW-weighted share of 3+ among multi-car households",
                share=weighted_share(car.eq(3), multi, backbone["H_GEW"]),
                notes="Weight sum is not an estimated German household count.",
            ),
        ]
    )

    rows.extend(distribution_rows("spatial_context", "RegioStaR4 distribution", backbone["RegioStaR4"]))
    rows.extend(
        [
            qa_row("spatial_context", "valid expected RegioStaR4 codes", count=int(backbone["RegioStaR4"].isin(REGIOSTAR4_ORDER).sum())),
            qa_row("spatial_context", "missing RegioStaR4", count=int(backbone["RegioStaR4"].isna().sum())),
            qa_row("spatial_context", "unexpected/unmapped RegioStaR4 codes", count=int((backbone["RegioStaR4"].notna() & ~backbone["RegioStaR4"].isin(REGIOSTAR4_ORDER)).sum()), notes=f"Expected codes: {REGIOSTAR4_ORDER}"),
        ]
    )

    rows.extend(distribution_rows("household_type", "household_type distribution", backbone["household_type"]))
    rows.extend(
        [
            qa_row("household_type", "missing household_type", count=int(backbone["household_type"].isna().sum())),
            qa_row("household_type", "unexpected household_type values", count=int((backbone["household_type"].notna() & ~backbone["household_type"].isin(HOUSEHOLD_TYPE_ORDER)).sum()), notes=f"Expected upstream labels: {HOUSEHOLD_TYPE_ORDER}"),
        ]
    )

    rows.extend(distribution_rows("driving_licence", "P_FS_PKW raw-code distribution", classified_persons["P_FS_PKW_numeric"]))
    person_state_counts = classified_persons["license_state"].value_counts()
    state_metric = {
        "licensed": "licensed persons",
        "explicit_no": "explicit-no persons",
        "structural_not_eligible": "structural-not-eligible persons",
        "unresolved": "unresolved persons",
    }
    for state in LICENSE_STATES:
        count = int(person_state_counts.get(state, 0))
        rows.append(qa_row("driving_licence", state_metric[state], count=count, share=count / len(classified_persons)))

    complete = backbone["license_information_complete_flag"].eq(1)
    rows.extend(
        [
            qa_row("driving_licence", "households with complete licence information", count=int(complete.sum()), share=float(complete.mean())),
            qa_row("driving_licence", "households with unresolved licence information", count=int((~complete).sum()), share=float((~complete).mean()), notes="Includes households without any person-roster record."),
        ]
    )
    rows.extend(distribution_rows("driving_licence", "n_licensed_drivers distribution", backbone["n_licensed_drivers"]))

    eligible_car = backbone["car_ownership_group"].isin(["single_car", "multi_car"])
    valid_w = valid_weight_mask(backbone["H_GEW"])
    incomplete_eligible = eligible_car & ~complete & valid_w
    rows.extend(
        [
            qa_row(
                "driving_licence",
                "licence-analysis households excluded due to unresolved information",
                count=int(incomplete_eligible.sum()),
                share=weighted_share(~complete, eligible_car, backbone["H_GEW"]),
                notes="Share is H_GEW-weighted among Single/Multi households with valid H_GEW.",
            ),
            qa_row("driver_car_ratio", "valid exact driver/car ratios", count=int(backbone["driver_car_ratio_exact"].notna().sum())),
            qa_row("driver_car_ratio", "exact-ratio cases excluded because H_ANZAUTO == 3", count=int(car.eq(3).sum())),
            qa_row("driver_car_ratio", "valid capped3 upper-bound ratios", count=int(backbone["driver_car_ratio_capped3_upper_bound"].notna().sum()), notes="For H_ANZAUTO=3 this is an upper bound, not an exact ratio."),
        ]
    )

    raw_weight = households["H_GEW"]
    parsed_weight = numeric(raw_weight)
    nonnumeric_weight = raw_weight.notna() & parsed_weight.isna()
    finite_weight = parsed_weight.notna() & np.isfinite(parsed_weight)
    valid_weight = finite_weight & parsed_weight.gt(0)
    weight_stats = parsed_weight.loc[valid_weight]
    rows.extend(
        [
            qa_row("weight_qa", "missing H_GEW", count=int(raw_weight.isna().sum())),
            qa_row("weight_qa", "non-numeric H_GEW", count=int(nonnumeric_weight.sum())),
            qa_row("weight_qa", "non-finite H_GEW", count=int((parsed_weight.notna() & ~np.isfinite(parsed_weight)).sum())),
            qa_row("weight_qa", "zero H_GEW", count=int(parsed_weight.eq(0).sum())),
            qa_row("weight_qa", "negative H_GEW", count=int(parsed_weight.lt(0).sum())),
            qa_row("weight_qa", "minimum valid H_GEW", value=float(weight_stats.min()) if len(weight_stats) else np.nan),
            qa_row("weight_qa", "maximum valid H_GEW", value=float(weight_stats.max()) if len(weight_stats) else np.nan),
            qa_row("weight_qa", "mean valid H_GEW", value=float(weight_stats.mean()) if len(weight_stats) else np.nan),
        ]
    )

    return pd.DataFrame(rows, columns=["section", "metric", "category", "count", "share", "value", "notes"])


def print_summary(backbone: pd.DataFrame) -> None:
    car = backbone["H_ANZAUTO"]
    multi = car.isin([2, 3])
    complete = backbone["license_information_complete_flag"].eq(1)
    eligible = backbone["car_ownership_group"].notna()
    exclusion_count = int((eligible & ~complete & valid_weight_mask(backbone["H_GEW"])).sum())
    exclusion_share = weighted_share(~complete, eligible, backbone["H_GEW"])

    print("\nTHEME 1 HOUSEHOLD BACKBONE")
    print(f"\nHouseholds: {len(backbone):,}")
    print(f"Single-car: {car.eq(1).sum():,}")
    print(f"Exactly two-car: {car.eq(2).sum():,}")
    print(f"3+-car: {car.eq(3).sum():,}")
    print(f"All multi-car: {multi.sum():,}")
    print("\n3+ share among multi-car:")
    print(f"unweighted: {car.eq(3).sum() / multi.sum():.3%}")
    print(f"H_GEW weighted: {weighted_share(car.eq(3), multi, backbone['H_GEW']):.3%}")
    print("\nLicence information:")
    print(f"complete households: {complete.sum():,}")
    print(f"unresolved/incomplete households: {(~complete).sum():,}")
    print(f"excluded from Single/Multi licence analysis: {exclusion_count:,}")
    print(f"H_GEW-weighted share excluded: {exclusion_share:.3%}")
    print(f"\nValid exact driver/car ratios: {backbone['driver_car_ratio_exact'].notna().sum():,}")
    print(f"Valid capped3 upper-bound ratios: {backbone['driver_car_ratio_capped3_upper_bound'].notna().sum():,}")
    print("\nOutput:")
    print(BACKBONE_PATH)
    print(QA_PATH)


def main() -> None:
    households, persons = read_inputs()
    backbone, classified_persons, orphan_person_summary = construct_backbone(households, persons)
    qa = build_qa(households, backbone, classified_persons, orphan_person_summary)

    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    backbone.to_csv(BACKBONE_PATH, index=False)
    qa.to_csv(QA_PATH, index=False)

    saved = pd.read_csv(BACKBONE_PATH, dtype={"H_ID": "string"}, low_memory=False)
    if len(saved) != len(backbone) or saved["H_ID"].duplicated().any():
        raise AssertionError("Saved backbone failed one-row-per-H_ID verification.")
    print_summary(backbone)


if __name__ == "__main__":
    main()
