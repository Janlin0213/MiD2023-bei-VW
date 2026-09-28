"""Theme 1: overall non-rbW reference-day mobility intensity by car ownership.

This descriptive cross-sectional person-day analysis compares persons in accepted
Single-car and Multi-car households on total non-rbW trips/distance and the amount
estimated to be performed as a Pkw driver. Estimates use raw positive ``P_GEW`` and
Gaussian identity-link GEE with household-clustered robust covariance. They describe
reference-day associations and Pkw driving participation, not habitual, longitudinal,
causal, household-car-allocation, or specific-vehicle-use effects. The estimator is
not a complete complex-survey-design estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import SELECTED_RAW_DIR, THEME1_BACKBONE_DIR, THEME1_OUTPUT_DIR

PERSON_PATH = SELECTED_RAW_DIR / "persons_selected_raw.csv"
TRIP_PATH = SELECTED_RAW_DIR / "trips_selected_raw.csv"
HOUSEHOLD_BACKBONE_PATH = THEME1_BACKBONE_DIR / "theme1_household_backbone.csv"
OUTPUT_DIR = THEME1_OUTPUT_DIR / "04_reference_day_mobility_intensity"

STATSMODELS_VERSION = "0.14.6"
CAR_ORDER = ["single_car", "multi_car"]
CAR_LABELS = {"single_car": "Single-car", "multi_car": "Multi-car"}
CAR_STYLES = {
    "single_car": {"color": "#2F5D7C"},
    "multi_car": {"color": "#D98245"},
}
MOBILITY_SCOPE_STYLES = {
    "total": {"marker": "o", "label": "Total mobility"},
    "pkw_driver": {"marker": "D", "label": "Pkw-driver mobility"},
}
OUTCOMES = {
    "anzwege2": {
        "label": "Trips per person/day",
        "panel_title": "A  Trips per person/day",
        "unit": "trips per person/day",
        "decimals": 2,
        "dimension": "trips",
        "mobility_scope": "total",
    },
    "pkw_driver_trips_day": {
        "label": "Pkw-driver trips per person/day",
        "panel_title": "A  Trips per person/day",
        "unit": "trips per person/day",
        "decimals": 2,
        "dimension": "trips",
        "mobility_scope": "pkw_driver",
    },
    "perskm2": {
        "label": "Distance per person/day (km)",
        "panel_title": "B  Distance per person/day (km)",
        "unit": "km per person/day",
        "decimals": 1,
        "dimension": "distance",
        "mobility_scope": "total",
    },
    "pkw_driver_km_day": {
        "label": "Pkw-driver distance per person/day (km)",
        "panel_title": "B  Distance per person/day (km)",
        "unit": "km per person/day",
        "decimals": 1,
        "dimension": "distance",
        "mobility_scope": "pkw_driver",
    },
}
PERSON_REQUIRED = ["HP_ID", "H_ID", "P_GEW", "anzwege2", "perskm2"]
TRIP_REQUIRED = ["HP_ID", "H_ID", "W_ID", "W_RBW", "pkw_fmf", "wegkm_imp"]
HOUSEHOLD_REQUIRED = ["H_ID", "H_ANZAUTO", "car_ownership_group"]
Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class MeanGEEResult:
    outcome: str
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    converged: bool
    iterations: int | None
    weights_verified: bool


def require_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise KeyError(
            f"{source} is missing required column(s): {missing}. "
            "Rerun the accepted selected-raw extraction if the person source is stale."
        )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_identifier(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def valid_weight(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.notna() & np.isfinite(values) & values.gt(0)


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PERSON_PATH.exists():
        raise FileNotFoundError(f"Accepted person selected-raw source not found: {PERSON_PATH}")
    if not HOUSEHOLD_BACKBONE_PATH.exists():
        raise FileNotFoundError(
            f"Accepted Theme 1 household backbone not found: {HOUSEHOLD_BACKBONE_PATH}"
        )

    person_header = pd.read_csv(PERSON_PATH, nrows=0).columns.tolist()
    household_header = pd.read_csv(HOUSEHOLD_BACKBONE_PATH, nrows=0).columns.tolist()
    require_columns(person_header, PERSON_REQUIRED, "Person selected-raw source")
    require_columns(household_header, HOUSEHOLD_REQUIRED, "Theme 1 household backbone")

    person_columns = [*PERSON_REQUIRED]
    if "P_ID" in person_header:
        person_columns.insert(2, "P_ID")
    person_id_columns = {column: "string" for column in ["HP_ID", "H_ID", "P_ID"] if column in person_columns}
    persons = pd.read_csv(
        PERSON_PATH,
        usecols=person_columns,
        dtype=person_id_columns,
        low_memory=False,
    )
    households = pd.read_csv(
        HOUSEHOLD_BACKBONE_PATH,
        usecols=HOUSEHOLD_REQUIRED,
        dtype={"H_ID": "string"},
        low_memory=False,
    )

    valid_hp = valid_identifier(persons["HP_ID"])
    duplicate_hp = valid_hp & persons["HP_ID"].duplicated(keep=False)
    if duplicate_hp.any():
        diagnostic_columns = [column for column in ["HP_ID", "H_ID", "P_ID"] if column in persons]
        raise RuntimeError(
            "Person selected-raw source is not one row per valid canonical HP_ID. "
            f"Diagnostic sample:\n{persons.loc[duplicate_hp, diagnostic_columns].head(10).to_string(index=False)}"
        )
    invalid_household_id = ~valid_identifier(households["H_ID"])
    duplicate_household = households["H_ID"].duplicated(keep=False)
    if invalid_household_id.any() or duplicate_household.any():
        bad = invalid_household_id | duplicate_household
        raise RuntimeError(
            "Theme 1 household backbone is not one row per valid H_ID. "
            f"Diagnostic sample:\n{households.loc[bad].head(10).to_string(index=False)}"
        )

    persons["P_GEW_source"] = persons["P_GEW"]
    persons["P_GEW"] = numeric(persons["P_GEW"])
    households["H_ANZAUTO"] = numeric(households["H_ANZAUTO"])
    return persons, households


def clean_outcomes(persons: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    data = persons.copy()
    data["anzwege2_source"] = data["anzwege2"]
    data["perskm2_source"] = data["perskm2"]
    data["anzwege2_numeric"] = numeric(data["anzwege2"])
    data["perskm2_numeric"] = numeric(data["perskm2"])

    rbw_only = data["perskm2_numeric"].eq(80802)
    rbw_inconsistent = rbw_only & ~data["anzwege2_numeric"].eq(0)
    if rbw_inconsistent.any():
        diagnostic_columns = [
            column
            for column in ["HP_ID", "H_ID", "P_ID", "anzwege2_source", "perskm2_source"]
            if column in data
        ]
        diagnostic = data.loc[rbw_inconsistent, diagnostic_columns].head(20)
        raise RuntimeError(
            "perskm2 == 80802 was not consistently paired with zero non-rbW trips; "
            "refusing to recode. Diagnostic sample:\n"
            + diagnostic.to_string(index=False)
        )

    data["perskm2_clean"] = data["perskm2_numeric"].mask(rbw_only, 0.0)
    finite_anz = data["anzwege2_numeric"].notna() & np.isfinite(data["anzwege2_numeric"])
    integer_anz = finite_anz & np.isclose(
        data["anzwege2_numeric"], np.round(data["anzwege2_numeric"]), rtol=0.0, atol=1e-12
    )
    data["valid_anzwege2"] = integer_anz & data["anzwege2_numeric"].between(0, 50)
    data["valid_perskm2"] = (
        data["perskm2_clean"].notna()
        & np.isfinite(data["perskm2_clean"])
        & data["perskm2_clean"].between(0, 2189.56)
    )
    data["anzwege2_clean"] = data["anzwege2_numeric"].where(data["valid_anzwege2"])
    data["perskm2_recode_80802"] = rbw_only

    special_counts = {
        "anzwege2_code_803": int(data["anzwege2_numeric"].eq(803).sum()),
        "anzwege2_code_804": int(data["anzwege2_numeric"].eq(804).sum()),
        "perskm2_code_80802_rbw_only": int(rbw_only.sum()),
        "perskm2_code_80803": int(data["perskm2_numeric"].eq(80803).sum()),
        "perskm2_code_80804": int(data["perskm2_numeric"].eq(80804).sum()),
        "perskm2_80802_inconsistent_with_zero_trips": int(rbw_inconsistent.sum()),
        "perskm2_80802_recoded_to_zero": int(rbw_only.sum()),
    }
    return data, special_counts


def merge_person_household(persons: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    merged = persons.merge(
        households,
        on="H_ID",
        how="left",
        validate="many_to_one",
        indicator="person_household_merge",
    )
    if len(merged) != len(persons):
        raise AssertionError("Person-household merge changed the number of source person rows.")
    valid_hp = valid_identifier(merged["HP_ID"])
    if merged.loc[valid_hp, "HP_ID"].duplicated().any():
        raise AssertionError("Person-household merge duplicated valid canonical HP_ID values.")
    return merged


def make_analytical_sample(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    masks: dict[str, pd.Series] = {}
    masks["valid_person_id"] = valid_identifier(merged["HP_ID"])
    masks["matched_household"] = masks["valid_person_id"] & merged[
        "person_household_merge"
    ].eq("both")
    masks["accepted_car_group"] = masks["matched_household"] & merged[
        "car_ownership_group"
    ].isin(CAR_ORDER)
    masks["valid_weight"] = masks["accepted_car_group"] & valid_weight(merged["P_GEW"])
    masks["valid_anzwege2"] = masks["valid_weight"] & merged["valid_anzwege2"]
    masks["valid_perskm2"] = masks["valid_weight"] & merged["valid_perskm2"]
    masks["common_complete"] = (
        masks["valid_weight"] & merged["valid_anzwege2"] & merged["valid_perskm2"]
    )
    sample = merged.loc[masks["common_complete"]].copy()
    sample["anzwege2"] = sample["anzwege2_clean"].astype(float)
    sample["perskm2"] = sample["perskm2_clean"].astype(float)

    if sample.empty:
        raise RuntimeError("The common complete reference-day analytical sample is empty.")
    if sample["HP_ID"].duplicated().any():
        raise AssertionError("Final analytical sample contains duplicate HP_ID values.")
    if not sample["car_ownership_group"].isin(CAR_ORDER).all():
        raise AssertionError("Final sample contains an unaccepted car-ownership group.")
    if set(sample["car_ownership_group"].unique()) != set(CAR_ORDER):
        raise RuntimeError("Final sample does not contain both accepted car-ownership groups.")
    if not valid_weight(sample["P_GEW"]).all():
        raise AssertionError("Final sample contains a nonpositive or non-finite P_GEW.")
    if not sample["anzwege2"].between(0, 50).all():
        raise AssertionError("Final sample contains anzwege2 outside 0-50.")
    if not sample["perskm2"].between(0, 2189.56).all():
        raise AssertionError("Final sample contains perskm2 outside 0-2189.56 km.")
    return sample, masks


def read_trip_source() -> pd.DataFrame:
    if not TRIP_PATH.exists():
        raise FileNotFoundError(f"Accepted trip selected-raw source not found: {TRIP_PATH}")
    trip_header = pd.read_csv(TRIP_PATH, nrows=0).columns.tolist()
    require_columns(trip_header, TRIP_REQUIRED, "Trip selected-raw source")
    trips = pd.read_csv(
        TRIP_PATH,
        usecols=TRIP_REQUIRED,
        dtype={"HP_ID": "string", "H_ID": "string", "W_ID": "string"},
        low_memory=False,
    )
    for column in ["W_RBW", "pkw_fmf", "wegkm_imp"]:
        trips[f"{column}_source"] = trips[column]
        trips[column] = numeric(trips[column])

    invalid_rbw = trips["W_RBW"].isna() | ~trips["W_RBW"].isin([0, 1])
    if invalid_rbw.any():
        diagnostic = trips.loc[
            invalid_rbw,
            ["HP_ID", "H_ID", "W_ID", "W_RBW_source", "pkw_fmf_source"],
        ].head(20)
        raise RuntimeError(
            "Trip source contains missing or unexpected W_RBW values, so the authoritative "
            f"rbW split cannot be applied. Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )
    valid_trip_key = valid_identifier(trips["HP_ID"]) & valid_identifier(trips["W_ID"])
    duplicate_trip_key = valid_trip_key & trips.duplicated(["HP_ID", "W_ID"], keep=False)
    if duplicate_trip_key.any():
        diagnostic = trips.loc[
            duplicate_trip_key, ["HP_ID", "H_ID", "W_ID", "W_RBW", "pkw_fmf"]
        ].head(20)
        raise RuntimeError(
            "Trip selected-raw source contains duplicate canonical HP_ID/W_ID keys. "
            f"Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )
    return trips


def add_pkw_driver_outcomes(
    sample: pd.DataFrame, trips: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    original_n = len(sample)
    original_group_n = sample["car_ownership_group"].value_counts().to_dict()
    nonrbw_source = trips["W_RBW"].eq(0)
    rbw_source = trips["W_RBW"].eq(1)
    primary_source = nonrbw_source & trips["pkw_fmf"].eq(1)
    switching_source = nonrbw_source & trips["pkw_fmf"].eq(3)
    valid_distance_source = (
        trips["wegkm_imp"].notna()
        & np.isfinite(trips["wegkm_imp"])
        & trips["wegkm_imp"].between(0.01, 950)
    )

    sample_ids = sample[["HP_ID", "H_ID"]].rename(columns={"H_ID": "person_H_ID"})
    final_trips = trips.merge(
        sample_ids,
        on="HP_ID",
        how="inner",
        validate="many_to_one",
    )
    h_id_mismatch = ~final_trips["H_ID"].eq(final_trips["person_H_ID"])
    if h_id_mismatch.any():
        diagnostic = final_trips.loc[
            h_id_mismatch,
            ["HP_ID", "H_ID", "person_H_ID", "W_ID", "W_RBW", "pkw_fmf"],
        ].head(20)
        raise RuntimeError(
            "Trip H_ID does not match the accepted person-sample H_ID for the same HP_ID. "
            f"Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )

    nonrbw_final = final_trips.loc[final_trips["W_RBW"].eq(0)].copy()
    invalid_final_trip_id = ~valid_identifier(nonrbw_final["W_ID"])
    if invalid_final_trip_id.any():
        diagnostic = nonrbw_final.loc[
            invalid_final_trip_id, ["HP_ID", "H_ID", "W_ID", "W_RBW", "pkw_fmf"]
        ].head(20)
        raise RuntimeError(
            "A final-sample non-rbW detailed trip lacks a valid W_ID. "
            f"Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )

    primary_final = nonrbw_final["pkw_fmf"].eq(1)
    valid_distance_final = (
        nonrbw_final["wegkm_imp"].notna()
        & np.isfinite(nonrbw_final["wegkm_imp"])
        & nonrbw_final["wegkm_imp"].between(0.01, 950)
    )
    invalid_primary_final = primary_final & ~valid_distance_final
    if invalid_primary_final.any():
        diagnostic = nonrbw_final.loc[
            invalid_primary_final,
            [
                "HP_ID",
                "H_ID",
                "W_ID",
                "W_RBW_source",
                "pkw_fmf_source",
                "wegkm_imp_source",
            ],
        ].head(20)
        raise RuntimeError(
            f"Found {int(invalid_primary_final.sum()):,} final-sample non-rbW pkw_fmf == 1 "
            "trip(s) with invalid/non-finite wegkm_imp outside 0.01-950 km; refusing to "
            f"underestimate driver distance. Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )

    detailed_counts = nonrbw_final.groupby("HP_ID", observed=True).size().rename(
        "n_detailed_nonrbw_trips"
    )
    driver_aggregate = nonrbw_final.loc[primary_final].groupby("HP_ID", observed=True).agg(
        observed_pkw_driver_trip_count=("W_ID", "size"),
        observed_pkw_driver_km=("wegkm_imp", "sum"),
    )
    person_trip_aggregate = pd.concat([detailed_counts, driver_aggregate], axis=1).fillna(0)
    person_trip_aggregate = person_trip_aggregate.reset_index()
    extended = sample.merge(
        person_trip_aggregate,
        on="HP_ID",
        how="left",
        validate="one_to_one",
    )
    aggregate_columns = [
        "n_detailed_nonrbw_trips",
        "observed_pkw_driver_trip_count",
        "observed_pkw_driver_km",
    ]
    extended[aggregate_columns] = extended[aggregate_columns].fillna(0.0)

    if len(extended) != original_n or extended["HP_ID"].duplicated().any():
        raise AssertionError("Trip aggregation merge changed or duplicated the accepted person sample.")
    final_group_n = extended["car_ownership_group"].value_counts().to_dict()
    if final_group_n != original_group_n:
        raise AssertionError(
            f"Single/Multi person counts changed after trip aggregation: {original_group_n} "
            f"versus {final_group_n}."
        )

    has_detailed = extended["n_detailed_nonrbw_trips"].gt(0)
    zero_total_zero_detail = extended["anzwege2"].eq(0) & ~has_detailed
    undefined_factor = extended["anzwege2"].gt(0) & ~has_detailed
    if undefined_factor.any():
        diagnostic = extended.loc[
            undefined_factor,
            ["HP_ID", "H_ID", "car_ownership_group", "anzwege2", "n_detailed_nonrbw_trips"],
        ].head(20)
        raise RuntimeError(
            "A final-sample person has positive anzwege2 but no detailed non-rbW trip row, "
            f"so the further-trip factor is undefined. Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )
    extended["further_trip_factor"] = np.where(
        has_detailed,
        extended["anzwege2"] / extended["n_detailed_nonrbw_trips"],
        np.where(zero_total_zero_detail, 1.0, np.nan),
    )
    factor_below_one = extended["further_trip_factor"].lt(1.0 - 1e-12)
    if factor_below_one.any():
        diagnostic = extended.loc[
            factor_below_one,
            [
                "HP_ID",
                "H_ID",
                "car_ownership_group",
                "anzwege2",
                "n_detailed_nonrbw_trips",
                "further_trip_factor",
            ],
        ].head(20)
        raise RuntimeError(
            "Further-trip factor is below 1 beyond numerical tolerance; refusing to correct "
            f"silently. Diagnostic sample:\n{diagnostic.to_string(index=False)}"
        )

    extended["pkw_driver_trips_day"] = (
        extended["observed_pkw_driver_trip_count"] * extended["further_trip_factor"]
    )
    extended["pkw_driver_km_day"] = (
        extended["observed_pkw_driver_km"] * extended["further_trip_factor"]
    )
    for outcome in ["pkw_driver_trips_day", "pkw_driver_km_day"]:
        if not np.isfinite(extended[outcome]).all() or extended[outcome].lt(0).any():
            raise AssertionError(f"Derived outcome {outcome} is non-finite or negative.")
    zero_primary = extended["observed_pkw_driver_trip_count"].eq(0)
    if not extended.loc[zero_primary, ["pkw_driver_trips_day", "pkw_driver_km_day"]].eq(0).all().all():
        raise AssertionError("A person without a primary driver trip received a nonzero outcome.")
    trips_above_total = extended["pkw_driver_trips_day"].gt(extended["anzwege2"] + 1e-10)
    if trips_above_total.any():
        diagnostic = extended.loc[
            trips_above_total,
            [
                "HP_ID",
                "H_ID",
                "anzwege2",
                "n_detailed_nonrbw_trips",
                "observed_pkw_driver_trip_count",
                "further_trip_factor",
                "pkw_driver_trips_day",
            ],
        ].head(20)
        raise RuntimeError(
            "Derived Pkw-driver trips exceed total non-rbW trips. Diagnostic sample:\n"
            + diagnostic.to_string(index=False)
        )

    distance_excess = extended["pkw_driver_km_day"] - extended["perskm2"]
    distance_above_total = distance_excess.gt(1e-8)
    if distance_above_total.any():
        diagnostic = extended.loc[
            distance_above_total,
            [
                "HP_ID",
                "H_ID",
                "anzwege2",
                "perskm2",
                "n_detailed_nonrbw_trips",
                "observed_pkw_driver_trip_count",
                "observed_pkw_driver_km",
                "further_trip_factor",
                "pkw_driver_km_day",
            ],
        ].sort_values("pkw_driver_km_day", ascending=False).head(10)
        print(
            "\nPKW-DRIVER DISTANCE UPLIFT DIAGNOSTIC\n"
            f"persons above perskm2: {int(distance_above_total.sum()):,}\n"
            f"maximum excess km: {float(distance_excess.max()):,.4f}\n"
            "No values were capped or corrected. Diagnostic sample:\n"
            + diagnostic.to_string(index=False)
        )

    factor = extended["further_trip_factor"].to_numpy(dtype=float)
    metrics: dict[str, float | int] = {
        "source_trip_rows": len(trips),
        "source_trip_persons": trips["HP_ID"].nunique(),
        "nonrbw_detailed_rows": int(nonrbw_source.sum()),
        "rbw_rows_excluded": int(rbw_source.sum()),
        "nonrbw_pkw_fmf_1_rows": int(primary_source.sum()),
        "persons_with_nonrbw_pkw_fmf_1": trips.loc[primary_source, "HP_ID"].nunique(),
        "nonrbw_pkw_fmf_3_rows": int(switching_source.sum()),
        "persons_with_nonrbw_pkw_fmf_3": trips.loc[switching_source, "HP_ID"].nunique(),
        "invalid_wegkm_imp_primary_rows_source": int((primary_source & ~valid_distance_source).sum()),
        "trip_rows_matched_to_final_sample": len(final_trips),
        "nonrbw_rows_in_final_sample": len(nonrbw_final),
        "primary_driver_rows_in_final_sample": int(primary_final.sum()),
        "primary_driver_persons_in_final_sample": nonrbw_final.loc[primary_final, "HP_ID"].nunique(),
        "switching_rows_in_final_sample": int(nonrbw_final["pkw_fmf"].eq(3).sum()),
        "switching_persons_in_final_sample": nonrbw_final.loc[
            nonrbw_final["pkw_fmf"].eq(3), "HP_ID"
        ].nunique(),
        "invalid_wegkm_imp_primary_rows_final_sample": int(invalid_primary_final.sum()),
        "aggregated_persons_matched_to_final_sample": len(person_trip_aggregate),
        "persons_with_zero_primary_driver_trips": int(zero_primary.sum()),
        "persons_with_undefined_factor": int(undefined_factor.sum()),
        "persons_with_factor_below_1": int(factor_below_one.sum()),
        "persons_with_factor_equal_1": int(np.isclose(factor, 1.0, rtol=0.0, atol=1e-12).sum()),
        "persons_with_factor_above_1": int((factor > 1.0 + 1e-12).sum()),
        "factor_min": float(np.min(factor)),
        "factor_max": float(np.max(factor)),
        "factor_mean": float(np.mean(factor)),
        "factor_p95": float(np.quantile(factor, 0.95)),
        "factor_p99": float(np.quantile(factor, 0.99)),
        "pkw_driver_trips_above_total_persons": int(trips_above_total.sum()),
        "pkw_driver_km_above_total_persons": int(distance_above_total.sum()),
        "pkw_driver_km_max_excess": float(max(distance_excess.max(), 0.0)),
        "final_analytical_persons_after_trip_merge": len(extended),
        "final_single_car_persons_after_trip_merge": int(
            extended["car_ownership_group"].eq("single_car").sum()
        ),
        "final_multi_car_persons_after_trip_merge": int(
            extended["car_ownership_group"].eq("multi_car").sum()
        ),
    }
    return extended, metrics


def fit_group_means(sample: pd.DataFrame, outcome: str) -> MeanGEEResult:
    endog = sample[outcome].to_numpy(dtype=float)
    exog = np.column_stack(
        [sample["car_ownership_group"].eq(car_group).to_numpy(dtype=float) for car_group in CAR_ORDER]
    )
    weights = sample["P_GEW"].to_numpy(dtype=float)
    clusters, levels = pd.factorize(sample["H_ID"], sort=False)
    if (clusters < 0).any() or len(levels) != sample["H_ID"].nunique():
        raise AssertionError("Could not create one household cluster code per accepted H_ID.")
    if np.linalg.matrix_rank(exog) != len(CAR_ORDER):
        raise RuntimeError(f"No-intercept group design is rank deficient for {outcome}.")

    model = sm.GEE(
        endog=endog,
        exog=exog,
        groups=clusters,
        family=sm.families.Gaussian(sm.families.links.Identity()),
        cov_struct=sm.cov_struct.Independence(),
        weights=weights,
    )
    stored_weights = np.asarray(model.weights, dtype=float)
    weights_verified = (
        stored_weights.shape == weights.shape
        and np.isfinite(stored_weights).all()
        and np.array_equal(stored_weights, weights)
    )
    if not weights_verified:
        raise AssertionError(f"GEE did not retain raw positive P_GEW for {outcome}.")

    fitted = model.fit(cov_type="robust", maxiter=100)
    beta = np.asarray(fitted.params, dtype=float)
    covariance = np.asarray(fitted.cov_params(), dtype=float)
    standard_error = np.asarray(fitted.bse, dtype=float)
    converged = bool(getattr(fitted, "converged", False))
    fit_history = getattr(fitted, "fit_history", {}) or {}
    iterations = len(fit_history.get("params", [])) or None
    if not converged:
        raise RuntimeError(f"Weighted Gaussian GEE did not converge for {outcome}.")
    if covariance.shape != (2, 2) or not (
        np.isfinite(beta).all()
        and np.isfinite(covariance).all()
        and np.isfinite(standard_error).all()
    ):
        raise RuntimeError(f"Weighted Gaussian GEE returned invalid estimates for {outcome}.")

    for index, car_group in enumerate(CAR_ORDER):
        group = sample["car_ownership_group"].eq(car_group)
        direct_mean = float(np.average(endog[group], weights=weights[group]))
        if not np.isclose(beta[index], direct_mean, rtol=1e-10, atol=1e-10):
            raise AssertionError(
                f"GEE/direct weighted mean mismatch for {outcome}, {car_group}: "
                f"{beta[index]} versus {direct_mean}."
            )

    return MeanGEEResult(
        outcome=outcome,
        beta=beta.copy(),
        covariance=covariance.copy(),
        standard_error=standard_error.copy(),
        converged=converged,
        iterations=iterations,
        weights_verified=weights_verified,
    )


def summarize_results(
    sample: pd.DataFrame, results: dict[str, MeanGEEResult]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast = np.array([-1.0, 1.0])
    for outcome, specification in OUTCOMES.items():
        result = results[outcome]
        for index, car_group in enumerate(CAR_ORDER):
            group = sample.loc[sample["car_ownership_group"].eq(car_group)]
            direct_mean = float(np.average(group[outcome], weights=group["P_GEW"]))
            estimate = float(result.beta[index])
            standard_error = float(result.standard_error[index])
            summary_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": specification["label"],
                    "unit": specification["unit"],
                    "dimension": specification["dimension"],
                    "mobility_scope": specification["mobility_scope"],
                    "car_ownership_group": car_group,
                    "car_ownership_label": CAR_LABELS[car_group],
                    "unweighted_person_n": len(group),
                    "household_n": group["H_ID"].nunique(),
                    "raw_P_GEW_sum": float(group["P_GEW"].sum()),
                    "weighted_mean": estimate,
                    "household_clustered_robust_se": standard_error,
                    "ci95_low": estimate - Z_95 * standard_error,
                    "ci95_high": estimate + Z_95 * standard_error,
                    "direct_weighted_mean_QA": direct_mean,
                    "GEE_direct_mean_agree": bool(
                        np.isclose(estimate, direct_mean, rtol=1e-10, atol=1e-10)
                    ),
                }
            )

        estimate = float(contrast @ result.beta)
        variance = float(contrast @ result.covariance @ contrast)
        if variance < -1e-10:
            raise RuntimeError(f"Negative robust contrast variance for {outcome}: {variance}")
        standard_error = float(np.sqrt(max(variance, 0.0)))
        z_statistic = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2.0 * norm.sf(abs(z_statistic))) if standard_error > 0 else np.nan
        contrast_rows.append(
            {
                "outcome": outcome,
                "outcome_label": specification["label"],
                "unit": specification["unit"],
                "dimension": specification["dimension"],
                "mobility_scope": specification["mobility_scope"],
                "contrast": "multi_car_minus_single_car",
                "estimate": estimate,
                "household_clustered_robust_se": standard_error,
                "ci95_low": estimate - Z_95 * standard_error,
                "ci95_high": estimate + Z_95 * standard_error,
                "z_statistic": z_statistic,
                "p_value": p_value,
                "interpretation_scope": "descriptive cross-sectional association",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["mean_share_of_total_mobility"] = np.nan
    for dimension in ["trips", "distance"]:
        for car_group in CAR_ORDER:
            total_mask = (
                summary["dimension"].eq(dimension)
                & summary["mobility_scope"].eq("total")
                & summary["car_ownership_group"].eq(car_group)
            )
            pkw_mask = (
                summary["dimension"].eq(dimension)
                & summary["mobility_scope"].eq("pkw_driver")
                & summary["car_ownership_group"].eq(car_group)
            )
            if int(total_mask.sum()) != 1 or int(pkw_mask.sum()) != 1:
                raise AssertionError(f"Could not identify total/Pkw rows for {dimension}, {car_group}.")
            total_mean = float(summary.loc[total_mask, "weighted_mean"].iloc[0])
            pkw_mean = float(summary.loc[pkw_mask, "weighted_mean"].iloc[0])
            summary.loc[pkw_mask, "mean_share_of_total_mobility"] = (
                pkw_mean / total_mean if total_mean > 0 else np.nan
            )
    return summary, pd.DataFrame(contrast_rows)


def qa_row(
    section: str,
    metric: str,
    count: int,
    denominator: int | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "section": section,
        "metric": metric,
        "count": int(count),
        "denominator": denominator,
        "share": np.nan if denominator in (None, 0) else float(count) / denominator,
        "notes": notes,
    }


def build_sample_qa(
    merged: pd.DataFrame,
    sample: pd.DataFrame,
    masks: dict[str, pd.Series],
    special_counts: dict[str, int],
) -> pd.DataFrame:
    source_n = len(merged)
    valid_id_n = int(masks["valid_person_id"].sum())
    matched_n = int(masks["matched_household"].sum())
    accepted_n = int(masks["accepted_car_group"].sum())
    weight_n = int(masks["valid_weight"].sum())
    rows = [
        qa_row("sample_flow", "source_person_rows", source_n, source_n, "Accepted person selected-raw rows."),
        qa_row("sample_flow", "valid_canonical_person_rows", valid_id_n, source_n, "Nonmissing, nonblank HP_ID."),
        qa_row("sample_flow", "invalid_canonical_person_rows", source_n - valid_id_n, source_n, "Missing or blank HP_ID; excluded."),
        qa_row("sample_flow", "matched_persons", matched_n, valid_id_n, "Valid-HP_ID persons matched many-to-one to the accepted household backbone."),
        qa_row("sample_flow", "unmatched_persons", valid_id_n - matched_n, valid_id_n, "Valid-HP_ID persons not matched to the accepted household backbone; excluded."),
        qa_row("sample_flow", "accepted_car_ownership_persons", accepted_n, matched_n, "Matched persons in the backbone-defined single_car or multi_car groups."),
        qa_row("sample_flow", "other_car_ownership_persons", matched_n - accepted_n, matched_n, "Matched persons outside the two accepted car-ownership groups; excluded."),
        qa_row("sample_flow", "invalid_or_nonpositive_weights", accepted_n - weight_n, accepted_n, "Non-numeric, non-finite, zero, or negative P_GEW; excluded."),
        qa_row("sample_flow", "positive_finite_weight_persons", weight_n, accepted_n, "Persons retained after the P_GEW rule."),
        qa_row("outcome_validity", "valid_anzwege2", int(masks["valid_anzwege2"].sum()), weight_n, "Integer-valued anzwege2 in the substantive 0-50 range."),
        qa_row("outcome_validity", "invalid_anzwege2", weight_n - int(masks["valid_anzwege2"].sum()), weight_n, "Missing, non-numeric, non-integer, or outside 0-50; excluded from the common sample."),
        qa_row("outcome_validity", "valid_perskm2", int(masks["valid_perskm2"].sum()), weight_n, "perskm2 in 0-2189.56 km after the explicit 80802 recode."),
        qa_row("outcome_validity", "invalid_perskm2", weight_n - int(masks["valid_perskm2"].sum()), weight_n, "Missing, non-numeric, or outside 0-2189.56 km; 80803/80804 remain invalid."),
        qa_row("sample_flow", "valid_both_outcomes", int(masks["common_complete"].sum()), weight_n, "Common complete sample valid for both outcomes."),
        qa_row("final_sample", "final_analytical_persons", len(sample), source_n, "One row per valid HP_ID in the common complete sample."),
        qa_row("final_sample", "final_households", sample["H_ID"].nunique(), None, "Unique H_ID clusters in the final sample."),
    ]
    for car_group in CAR_ORDER:
        group = sample.loc[sample["car_ownership_group"].eq(car_group)]
        rows.extend(
            [
                qa_row("final_sample", f"final_{car_group}_persons", len(group), len(sample), f"Final persons in accepted {car_group}."),
                qa_row("final_sample", f"final_{car_group}_households", group["H_ID"].nunique(), sample["H_ID"].nunique(), f"Final households contributing to {car_group}."),
            ]
        )
    for metric, count in special_counts.items():
        notes = {
            "anzwege2_code_803": "Source design/nonresponse code; excluded, never treated as trips.",
            "anzwege2_code_804": "Source design/nonresponse code; excluded, never treated as trips.",
            "perskm2_code_80802_rbw_only": "Source rbW-only code; verified against anzwege2 == 0.",
            "perskm2_code_80803": "Source design/nonresponse code; excluded, never treated as distance.",
            "perskm2_code_80804": "Source design/nonresponse code; excluded, never treated as distance.",
            "perskm2_80802_inconsistent_with_zero_trips": "Must equal zero; otherwise analysis stops before recoding.",
            "perskm2_80802_recoded_to_zero": "Explicitly recoded to 0.0 km for the non-rbW metric after the consistency assertion.",
        }[metric]
        rows.append(qa_row("special_values_source", metric, count, source_n, notes))
    final_rbw = int(sample["perskm2_recode_80802"].sum())
    rows.append(
        qa_row(
            "special_values_final",
            "perskm2_80802_recoded_to_zero_in_final_sample",
            final_rbw,
            len(sample),
            "rbW-only persons retained as 0.0 km in the final non-rbW common sample.",
        )
    )
    return pd.DataFrame(rows)


def build_trip_aggregation_qa(metrics: dict[str, float | int]) -> pd.DataFrame:
    source_rows = int(metrics["source_trip_rows"])
    source_persons = int(metrics["source_trip_persons"])
    nonrbw_rows = int(metrics["nonrbw_detailed_rows"])
    final_n = int(metrics["final_analytical_persons_after_trip_merge"])
    specifications = [
        ("trip_universe", "source_trip_rows", source_rows, "count", "Accepted full selected-raw trip rows."),
        ("trip_universe", "source_trip_persons", None, "count", "Unique HP_ID values in the accepted trip source."),
        ("trip_universe", "nonrbw_detailed_rows", source_rows, "count", "Detailed rows retained using the authoritative W_RBW == 0 rule."),
        ("trip_universe", "rbw_rows_excluded", source_rows, "count", "Rows excluded using W_RBW == 1; no W_ID proxy used."),
        ("primary_definition", "nonrbw_pkw_fmf_1_rows", nonrbw_rows, "count", "Primary detailed Pkw-driver rows: W_RBW == 0 and pkw_fmf == 1."),
        ("primary_definition", "persons_with_nonrbw_pkw_fmf_1", source_persons, "count", "Unique source persons with at least one primary driver row."),
        ("switching_QA", "nonrbw_pkw_fmf_3_rows", nonrbw_rows, "count", "Driver/passenger switching rows; excluded from primary outcomes."),
        ("switching_QA", "persons_with_nonrbw_pkw_fmf_3", source_persons, "count", "Unique source persons with switching rows; excluded from primary outcomes."),
        ("distance_validity", "invalid_wegkm_imp_primary_rows_source", int(metrics["nonrbw_pkw_fmf_1_rows"]), "count", "Primary source rows with non-finite or out-of-range wegkm_imp; never converted to zero."),
        ("final_sample_trip_merge", "trip_rows_matched_to_final_sample", source_rows, "count", "Trip rows belonging to the accepted common-complete person sample."),
        ("final_sample_trip_merge", "nonrbw_rows_in_final_sample", int(metrics["trip_rows_matched_to_final_sample"]), "count", "Final-sample trip rows satisfying W_RBW == 0."),
        ("final_sample_trip_merge", "primary_driver_rows_in_final_sample", int(metrics["nonrbw_rows_in_final_sample"]), "count", "Final-sample non-rbW rows with pkw_fmf == 1."),
        ("final_sample_trip_merge", "primary_driver_persons_in_final_sample", final_n, "count", "Final-sample persons with at least one primary driver row."),
        ("final_sample_trip_merge", "switching_rows_in_final_sample", int(metrics["nonrbw_rows_in_final_sample"]), "count", "Final-sample non-rbW pkw_fmf == 3 rows excluded from primary outcomes."),
        ("final_sample_trip_merge", "switching_persons_in_final_sample", final_n, "count", "Final-sample persons with at least one switching row."),
        ("distance_validity", "invalid_wegkm_imp_primary_rows_final_sample", int(metrics["primary_driver_rows_in_final_sample"]), "count", "Must be zero; otherwise the script stops and prints diagnostics."),
        ("final_sample_trip_merge", "aggregated_persons_matched_to_final_sample", final_n, "count", "Final persons with at least one detailed non-rbW row aggregated and matched one-to-one."),
        ("primary_definition", "persons_with_zero_primary_driver_trips", final_n, "count", "Final persons assigned zero Pkw-driver trips and km before/after uplift."),
        ("further_trip_factor", "persons_with_undefined_factor", final_n, "count", "Must be zero; positive anzwege2 with zero detailed non-rbW rows stops the script."),
        ("further_trip_factor", "persons_with_factor_below_1", final_n, "count", "Must be zero beyond tolerance 1e-12; otherwise the script stops."),
        ("further_trip_factor", "persons_with_factor_equal_1", final_n, "count", "Factor equals 1 within absolute tolerance 1e-12."),
        ("further_trip_factor", "persons_with_factor_above_1", final_n, "count", "Persons with additional/undetailed-trip uplift above 1."),
        ("further_trip_factor", "factor_min", None, "factor", "Minimum person-level anzwege2 / detailed non-rbW trip count factor."),
        ("further_trip_factor", "factor_max", None, "factor", "Maximum person-level further-trip factor."),
        ("further_trip_factor", "factor_mean", None, "factor", "Arithmetic mean of person-level further-trip factors."),
        ("further_trip_factor", "factor_p95", None, "factor", "95th percentile of person-level further-trip factors."),
        ("further_trip_factor", "factor_p99", None, "factor", "99th percentile of person-level further-trip factors."),
        ("outcome_consistency", "pkw_driver_trips_above_total_persons", final_n, "count", "Must be zero; Pkw-driver trips are not capped to total trips."),
        ("outcome_consistency", "pkw_driver_km_above_total_persons", final_n, "count", "Reported, not capped; possible under independent proportional uplift/rounding."),
        ("outcome_consistency", "pkw_driver_km_max_excess", None, "km", "Maximum positive pkw_driver_km_day minus perskm2; no correction applied."),
        ("final_sample", "final_analytical_persons_after_trip_merge", final_n, "count", "Must equal the accepted total-mobility analytical N."),
        ("final_sample", "final_single_car_persons_after_trip_merge", final_n, "count", "Must equal the accepted Single-car analytical N."),
        ("final_sample", "final_multi_car_persons_after_trip_merge", final_n, "count", "Must equal the accepted Multi-car analytical N."),
    ]
    rows: list[dict[str, Any]] = []
    for section, metric, denominator, value_kind, notes in specifications:
        value = metrics[metric]
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "value_kind": value_kind,
                "denominator": denominator,
                "share": (
                    np.nan
                    if denominator in (None, 0) or value_kind != "count"
                    else float(value) / float(denominator)
                ),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_metadata(
    sample: pd.DataFrame,
    results: dict[str, MeanGEEResult],
    special_counts: dict[str, int],
) -> Path:
    rows = [
        ("analytical_unit", "person; one row per valid canonical HP_ID"),
        ("reference_period", "MiD reference day"),
        ("person_source", str(PERSON_PATH)),
        ("trip_source", str(TRIP_PATH)),
        ("household_backbone", str(HOUSEHOLD_BACKBONE_PATH)),
        ("car_ownership_definition", "Accepted backbone values single_car and multi_car; not reconstructed here"),
        ("weight", "P_GEW: raw, positive, finite, not normalized"),
        ("cluster", "H_ID"),
        ("method", "P_GEW-weighted Gaussian identity-link GEE with Independence working correlation"),
        ("covariance", "household-clustered robust / sandwich"),
        ("outcomes", "Total anzwege2/perskm2 and derived Pkw-driver trips/km per person/day; all exclude rbW"),
        ("anzwege2_validity", "Integer values in the substantive 0-50 range; 803/804 excluded"),
        ("perskm2_validity", "Values in 0-2189.56 km; 80803/80804 excluded"),
        ("perskm2_80802_handling", f"{special_counts['perskm2_code_80802_rbw_only']:,} rbW-only cases verified with anzwege2 == 0 and recoded to 0.0 km"),
        ("common_sample", "Both accepted total outcomes are valid on the same final person population; derived Pkw-driver outcomes retain that identical denominator and assign zero to persons without a qualifying driver trip"),
        ("pkw_driver_primary_definition", "Directly reported detailed trips with W_RBW == 0 and pkw_fmf == 1; pkw_fmf == 3 excluded"),
        ("pkw_driver_distance", "wegkm_imp in the accepted 0.01-950 km range on primary driver trips"),
        ("further_trip_uplift", "Person-level anzwege2 / directly reported non-rbW detailed trip rows; 1.0 when both counts are zero"),
        ("further_trip_uplift_interpretation", "Proportionally assigns undetailed weitere Wege to the person's observed detailed-trip mobility composition because additional trips have no detailed mode/driver information"),
        ("pkw_driver_interpretation", "Pkw-driver metrics are derived from directly reported non-rbW trips with pkw_fmf == 1. Undetailed weitere Wege are represented using the MiD-style person-level proportional uplift factor. These metrics describe Pkw driving participation, not confirmed use of a specific recorded household vehicle."),
        ("vehicle_choice_reconstruction", "No MNL, W_WAUTO, household-fleet matching, vehicle-state, HOME_ORIGIN, or strict-choice reconstruction logic used"),
        ("final_persons", f"{len(sample):,}"),
        ("final_households", f"{sample['H_ID'].nunique():,}"),
        ("interpretation", "Descriptive cross-sectional association; not usual, habitual, longitudinal, or causal mobility behaviour"),
        ("complex_survey_caveat", "This is not a complete complex-survey-design estimator"),
        ("statsmodels_version", statsmodels.__version__),
    ]
    for outcome, result in results.items():
        rows.extend(
            [
                (f"GEE_converged_{outcome}", str(result.converged)),
                (f"GEE_iterations_{outcome}", str(result.iterations)),
                (f"raw_P_GEW_verified_{outcome}", str(result.weights_verified)),
            ]
        )
    return write_csv(pd.DataFrame(rows, columns=["item", "value"]), "analysis_metadata.csv")


def plot_primary_figure(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8))
    panel_specifications = [
        ("trips", "A  Trips per person/day", "Trips per person/day", 2),
        ("distance", "B  Distance per person/day (km)", "Distance per person/day (km)", 1),
    ]
    scope_offsets = {"total": -0.11, "pkw_driver": 0.11}
    for ax, (dimension, panel_title, y_label, decimals) in zip(axes, panel_specifications):
        panel = summary.loc[summary["dimension"].eq(dimension)].copy()
        if len(panel) != 4:
            raise AssertionError(f"Expected four plotting rows for {dimension}; found {len(panel)}.")
        x_positions = np.arange(len(CAR_ORDER), dtype=float)
        for x_value, car_group in zip(x_positions, CAR_ORDER):
            for mobility_scope in ["total", "pkw_driver"]:
                row = panel.loc[
                    panel["car_ownership_group"].eq(car_group)
                    & panel["mobility_scope"].eq(mobility_scope)
                ]
                if len(row) != 1:
                    raise AssertionError(
                        f"Expected one plotting row for {dimension}, {car_group}, {mobility_scope}."
                    )
                row = row.iloc[0]
                estimate = float(row["weighted_mean"])
                lower = float(row["ci95_low"])
                upper = float(row["ci95_high"])
                color = CAR_STYLES[car_group]["color"]
                marker = MOBILITY_SCOPE_STYLES[mobility_scope]["marker"]
                plotted_x = x_value + scope_offsets[mobility_scope]
                ax.errorbar(
                    plotted_x,
                    estimate,
                    yerr=np.array([[estimate - lower], [upper - estimate]]),
                    fmt=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=8,
                    elinewidth=1.4,
                    capsize=4,
                    capthick=1.2,
                    zorder=3,
                )
                ax.annotate(
                    f"{estimate:,.{decimals}f}",
                    xy=(plotted_x, estimate),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                    color=color,
                    fontweight="bold",
                )
        ax.set_title(panel_title, loc="left", fontsize=11.5, fontweight="bold")
        group_n: dict[str, int] = {}
        for car_group in CAR_ORDER:
            n_values = panel.loc[
                panel["car_ownership_group"].eq(car_group), "unweighted_person_n"
            ].unique()
            if len(n_values) != 1:
                raise AssertionError(f"Plotting Ns differ across mobility scopes for {car_group}.")
            group_n[car_group] = int(n_values[0])
        ax.set_xticks(
            x_positions,
            [
                f"{CAR_LABELS[car_group]}\n(n = {group_n[car_group]:,})"
                for car_group in CAR_ORDER
            ],
        )
        ax.set_ylabel(y_label)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(0, float(panel["ci95_high"].max()) * 1.20)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#808080")
        ax.tick_params(axis="y", length=0, colors="#404040")
        ax.tick_params(axis="x", length=0, pad=8)

    fig.suptitle(
        "Reference-day mobility intensity by household car ownership",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MOBILITY_SCOPE_STYLES[mobility_scope]["marker"],
            linestyle="None",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=MOBILITY_SCOPE_STYLES[mobility_scope]["label"],
        )
        for mobility_scope in ["total", "pkw_driver"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.08,
        0.025,
        "Weighted means with household-clustered robust 95% CIs. All outcomes exclude rbW; "
        "Pkw-driver measures use pkw_fmf = 1 and the MiD further-trip uplift.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.27, wspace=0.30)
    output_path = OUTPUT_DIR / "reference_day_mobility_intensity_overall.png"
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_weighted_ecdf(sample: pd.DataFrame) -> Path:
    def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
        order = np.argsort(values.to_numpy())
        sorted_values = values.to_numpy()[order]
        sorted_weights = weights.to_numpy()[order]
        cumulative_weights = np.cumsum(sorted_weights)
        threshold = quantile * cumulative_weights[-1]
        return float(sorted_values[np.searchsorted(cumulative_weights, threshold, side="left")])

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        return float(np.average(values.to_numpy(), weights=weights.to_numpy()))

    def within_range_share(values: pd.Series, weights: pd.Series, upper: float) -> float:
        mask = values.to_numpy() <= upper
        total_weight = weights.to_numpy().sum()
        if total_weight <= 0:
            return 0.0
        return float(weights.to_numpy()[mask].sum() / total_weight * 100.0)

    panel_specifications = [
        ("total", "anzwege2", "A  Total mobility — Trips per person/day", "Trips per person/day", "trips"),
        (
            "total",
            "perskm2",
            "B  Total mobility — Distance per person/day (km)",
            "Distance per person/day (km)",
            "distance",
        ),
        (
            "pkw_driver",
            "pkw_driver_trips_day",
            "C  Pkw-driver mobility — Trips per person/day",
            "Trips per person/day",
            "trips",
        ),
        (
            "pkw_driver",
            "pkw_driver_km_day",
            "D  Pkw-driver mobility — Distance per person/day (km)",
            "Distance per person/day (km)",
            "distance",
        ),
    ]
    distance_series = []
    for car_group in CAR_ORDER:
        for outcome in ["perskm2", "pkw_driver_km_day"]:
            group = sample.loc[sample["car_ownership_group"].eq(car_group), [outcome, "P_GEW"]]
            distance_series.append(
                (
                    group[outcome].astype(float),
                    group["P_GEW"].astype(float),
                )
            )
    distance_display_limit = max(
        weighted_quantile(values, weights, 0.95) for values, weights in distance_series
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharey=True)
    for ax, (mobility_scope, outcome, panel_title, x_label, dimension) in zip(
        axes.flat, panel_specifications
    ):
        annotation_blocks = []
        for car_group in CAR_ORDER:
            group = sample.loc[sample["car_ownership_group"].eq(car_group), [outcome, "P_GEW"]]
            values = group[outcome].astype(float)
            weights = group["P_GEW"].astype(float)
            order = np.argsort(values.to_numpy(), kind="stable")
            sorted_values = values.to_numpy()[order]
            sorted_weights = weights.to_numpy()[order]
            ecdf = np.cumsum(sorted_weights) / sorted_weights.sum()
            color = CAR_STYLES[car_group]["color"]
            ax.step(
                sorted_values,
                ecdf,
                where="post",
                color=color,
                linewidth=2.1,
                zorder=2,
            )

            mean_value = weighted_mean(values, weights)
            ax.axvline(
                mean_value,
                color=color,
                linewidth=1.0,
                alpha=0.35,
                zorder=1,
            )

            display_upper = 10.0 if dimension == "trips" else distance_display_limit
            share_in_range = within_range_share(values, weights, display_upper)
            max_value = float(values.max())
            annotation_blocks.append(
                (
                    CAR_LABELS[car_group],
                    f"mean = {mean_value:.2f}",
                    f"{share_in_range:.1f}% within displayed range; max = {max_value:.1f}",
                )
            )

        ax.set_title(panel_title, loc="left", fontsize=11.5, fontweight="bold")
        ax.set_xlabel(x_label)
        if dimension == "trips":
            ax.set_xlim(0, 10)
        else:
            ax.set_xlim(0, distance_display_limit * 1.05)

        annotation_y_positions = [0.90, 0.56]
        for (label, mean_text, share_text), y_position in zip(annotation_blocks, annotation_y_positions):
            ax.text(
                0.98,
                y_position,
                f"{label}\n{mean_text}\n{share_text}",
                transform=ax.transAxes,
                fontsize=8.0,
                color="#404040",
                ha="right",
                va="top",
                linespacing=1.35,
            )

        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#B0B0B0")
        ax.spines["bottom"].set_color("#808080")
        ax.tick_params(length=0, colors="#404040")

    axes[0, 0].set_ylabel("Cumulative share of persons")
    axes[1, 0].set_ylabel("Cumulative share of persons")
    fig.suptitle(
        "Reference-day mobility intensity distributions by household car ownership",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color=CAR_STYLES[car_group]["color"], linewidth=2.0, label=CAR_LABELS[car_group])
        for car_group in CAR_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.08,
        0.018,
        "Survey-weighted ECDFs; the distance x-axis is truncated for visualization only at the maximum weighted P95 across all four distance series, while all observations are retained in the calculations.",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.19, wspace=0.24, hspace=0.34)
    output_path = OUTPUT_DIR / "reference_day_mobility_intensity_weighted_ecdf.png"
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_accepted_total_output_baselines() -> dict[str, dict[str, Any]]:
    filenames = [
        "01_reference_day_mobility_summary.csv",
        "02_reference_day_mobility_contrasts.csv",
        "03_reference_day_mobility_sample_QA.csv",
    ]
    baselines: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        path = OUTPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Accepted pre-extension output required for regression validation is missing: {path}"
            )
        frame = pd.read_csv(path, low_memory=False)
        if filename == "01_reference_day_mobility_summary.csv" and "mobility_scope" in frame:
            frame = frame.loc[frame["mobility_scope"].eq("total")].drop(
                columns=["dimension", "mobility_scope", "mean_share_of_total_mobility"]
            )
        if filename == "02_reference_day_mobility_contrasts.csv" and "mobility_scope" in frame:
            frame = frame.loc[frame["mobility_scope"].eq("total")].drop(
                columns=["dimension", "mobility_scope"]
            )
        frame = frame.reset_index(drop=True)
        baselines[filename] = {
            "text": frame.to_csv(index=False, float_format="%.10g").replace("\r\n", "\n"),
            "columns": frame.columns.tolist(),
        }
    return baselines


def validate_accepted_total_outputs_unchanged(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    sample_qa: pd.DataFrame,
    baselines: dict[str, dict[str, Any]],
) -> None:
    projections = {
        "01_reference_day_mobility_summary.csv": summary.loc[
            summary["mobility_scope"].eq("total")
        ],
        "02_reference_day_mobility_contrasts.csv": contrasts.loc[
            contrasts["mobility_scope"].eq("total")
        ],
        "03_reference_day_mobility_sample_QA.csv": sample_qa,
    }
    for filename, frame in projections.items():
        columns = baselines[filename]["columns"]
        projected = frame.loc[:, columns].reset_index(drop=True)
        candidate_text = projected.to_csv(index=False, float_format="%.10g").replace(
            "\r\n", "\n"
        )
        if candidate_text != baselines[filename]["text"]:
            raise AssertionError(
                f"Accepted total-mobility output changed unexpectedly before write: {filename}"
            )


def print_console_summary(
    persons: pd.DataFrame,
    sample: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    special_counts: dict[str, int],
    trip_metrics: dict[str, float | int],
) -> None:
    print("\nTHEME 1 - REFERENCE-DAY MOBILITY INTENSITY")
    print("\nINPUT")
    print(f"person selected raw: {PERSON_PATH}")
    print(f"trip selected raw: {TRIP_PATH}")
    print(f"household backbone: {HOUSEHOLD_BACKBONE_PATH}")
    print("\nSAMPLE")
    print(f"source persons: {len(persons):,}")
    print(f"final analytical persons: {len(sample):,}")
    print(f"final households: {sample['H_ID'].nunique():,}")
    for car_group in CAR_ORDER:
        group = sample.loc[sample["car_ownership_group"].eq(car_group)]
        print(f"{CAR_LABELS[car_group]} persons: {len(group):,}")

    print("\nWEIGHTED GROUP MEANS")
    for row in summary.itertuples(index=False):
        print(
            f"{row.outcome} - {row.car_ownership_label}: {row.weighted_mean:,.4f} "
            f"(95% CI {row.ci95_low:,.4f} to {row.ci95_high:,.4f}; "
            f"n={row.unweighted_person_n:,})"
        )
    print("\nMULTI MINUS SINGLE CONTRASTS")
    for row in contrasts.itertuples(index=False):
        print(
            f"{row.outcome}: {row.estimate:,.4f} "
            f"(95% CI {row.ci95_low:,.4f} to {row.ci95_high:,.4f}; "
            f"p={row.p_value:.6g})"
        )
    print("\nWEIGHTED PKW-DRIVER SHARE OF TOTAL MOBILITY")
    pkw_rows = summary.loc[summary["mobility_scope"].eq("pkw_driver")]
    for row in pkw_rows.itertuples(index=False):
        print(
            f"{row.dimension} - {row.car_ownership_label}: "
            f"{row.mean_share_of_total_mobility:.2%} "
            "(ratio of weighted group means)"
        )
    print("\nSPECIAL-VALUE HANDLING")
    for metric, count in special_counts.items():
        print(f"{metric}: {count:,}")
    print(f"perskm2_80802_recoded_to_zero_in_final_sample: {int(sample['perskm2_recode_80802'].sum()):,}")
    print("\nTRIP AGGREGATION AND FURTHER-TRIP UPLIFT QA")
    for metric, value in trip_metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{metric}: {int(value):,}")
        else:
            print(f"{metric}: {float(value):,.10g}")
    print("\nSCOPE")
    print("Reference-day, non-rbW, descriptive/associational person analysis; not habitual or causal.")
    print("Pkw-driver metrics describe driving participation, not confirmed use of a recorded household vehicle.")
    print("No MNL vehicle-choice reconstruction logic was reused or changed.")
    print("This is not a complete complex-survey-design estimator.")
    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)


def main() -> None:
    assert statsmodels.__version__ == STATSMODELS_VERSION, (
        f"This module requires statsmodels {STATSMODELS_VERSION}; found {statsmodels.__version__}."
    )
    assert hasattr(sm, "GEE")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accepted_baselines = load_accepted_total_output_baselines()

    persons, households = read_sources()
    persons, special_counts = clean_outcomes(persons)
    merged = merge_person_household(persons, households)
    sample, masks = make_analytical_sample(merged)
    trips = read_trip_source()
    sample, trip_metrics = add_pkw_driver_outcomes(sample, trips)
    results = {outcome: fit_group_means(sample, outcome) for outcome in OUTCOMES}
    summary, contrasts = summarize_results(sample, results)
    qa = build_sample_qa(merged, sample, masks, special_counts)
    trip_qa = build_trip_aggregation_qa(trip_metrics)
    validate_accepted_total_outputs_unchanged(
        summary, contrasts, qa, accepted_baselines
    )

    write_csv(summary, "01_reference_day_mobility_summary.csv")
    write_csv(contrasts, "02_reference_day_mobility_contrasts.csv")
    write_csv(qa, "03_reference_day_mobility_sample_QA.csv")
    write_csv(trip_qa, "04_pkw_driver_mobility_QA.csv")
    write_metadata(sample, results, special_counts)
    plot_primary_figure(summary)
    plot_weighted_ecdf(sample)
    print_console_summary(
        persons, sample, summary, contrasts, special_counts, trip_metrics
    )


if __name__ == "__main__":
    main()
