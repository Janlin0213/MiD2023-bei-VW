"""Estimate the Stage-0 V1-only unlabelled household vehicle-use MNL.

V1 contains generic powertrain, segment, and continuous vehicle-age
coefficients for both household vehicles. Car 1 and Car 2 are arbitrary
identifiers, so the model has no alternative-specific constant and no
car-label-specific coefficient.

The input is the accepted single-active-driver design matrix produced by
``scripts/03_model_input/05_build_single_driver_mnl_design_matrix.py``. This
script validates and filters that matrix in memory; it never changes the source
CSV or any upstream data-generation logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (  # noqa: E402
    OUTPUTS_DIR,
    SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH,
)

from biogeme.biogeme import BIOGEME  # noqa: E402
from biogeme.database import Database  # noqa: E402
from biogeme.expressions import Beta, Variable  # noqa: E402
import biogeme.models as models  # noqa: E402
from biogeme.parameters import Parameters  # noqa: E402
from biogeme.results_processing import EstimateVarianceCovariance  # noqa: E402
from biogeme.version import get_version as get_biogeme_version  # noqa: E402


RESULTS_DIR = OUTPUTS_DIR / "mnl" / "v1_only"
V2_RESULTS_DIR = OUTPUTS_DIR / "mnl" / "v2_only"

POWERTRAINS = ("ICE", "HEV", "PHEV", "BEV")
POWERTRAIN_FULL = (*POWERTRAINS, "OTHER", "MISSING")
SEGMENTS = ("SMALL", "COMPACT", "MEDIUM", "LARGE")
SEGMENT_FULL = (*SEGMENTS, "MISSING")

PARAMETERS = (
    "B_PT_HEV",
    "B_PT_PHEV",
    "B_PT_BEV",
    "B_SEG_COMPACT",
    "B_SEG_MEDIUM",
    "B_SEG_LARGE",
    "B_VEHICLE_AGE",
)

DELTA_X_SPECS = (
    ("D_PT_HEV", "B_PT_HEV", "PT1_HEV", "PT2_HEV"),
    ("D_PT_PHEV", "B_PT_PHEV", "PT1_PHEV", "PT2_PHEV"),
    ("D_PT_BEV", "B_PT_BEV", "PT1_BEV", "PT2_BEV"),
    ("D_SEG_COMPACT", "B_SEG_COMPACT", "SEG1_COMPACT", "SEG2_COMPACT"),
    ("D_SEG_MEDIUM", "B_SEG_MEDIUM", "SEG1_MEDIUM", "SEG2_MEDIUM"),
    ("D_SEG_LARGE", "B_SEG_LARGE", "SEG1_LARGE", "SEG2_LARGE"),
    ("D_VEHICLE_AGE", "B_VEHICLE_AGE", "VEHICLE_AGE_1", "VEHICLE_AGE_2"),
)

MODEL_COLUMNS = [
    "CHOICE",
    "AV_1",
    "AV_2",
    "W_GEW",
    *[f"PT{car}_{category}" for car in (1, 2) for category in POWERTRAINS],
    *[f"SEG{car}_{category}" for car in (1, 2) for category in SEGMENTS],
    "VEHICLE_AGE_1",
    "VEHICLE_AGE_2",
]

ABS_COEFFICIENT_WARNING = 10.0
ROBUST_SE_WARNING = 5.0
CORRELATION_WARNING = 0.95
HESSIAN_EIGENVALUE_WARNING = 1.0e-5
HESSIAN_CONDITION_WARNING = 1.0e8
LABEL_ATOL = 1.0e-8
LABEL_RTOL = 1.0e-6


def sha256(path: Path) -> str:
    """Return a SHA-256 digest without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_required_columns(data: pd.DataFrame) -> None:
    """Stop if the accepted design matrix lacks a required V1 field."""

    required = {
        "CHOICE_ID",
        "CHOICE",
        "AV_1",
        "AV_2",
        "W_GEW",
        *[f"PT{car}_{category}" for car in (1, 2) for category in POWERTRAIN_FULL],
        *[f"SEG{car}_{category}" for car in (1, 2) for category in SEGMENT_FULL],
        "VEHICLE_AGE_1",
        "VEHICLE_AGE_1_MISSING",
        "VEHICLE_AGE_2",
        "VEHICLE_AGE_2_MISSING",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise AssertionError(f"Required V1 input columns are missing: {missing}")


def assert_binary(data: pd.DataFrame, columns: list[str], label: str) -> None:
    """Assert that indicator columns contain only numeric 0/1."""

    for column in columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        if invalid.any():
            examples = data.loc[invalid, ["CHOICE_ID", column]].head(5).to_dict("records")
            raise AssertionError(
                f"{label}: {column} is not complete binary 0/1. Examples: {examples}"
            )
        data[column] = numeric.astype(np.int8)


def assert_one_hot(data: pd.DataFrame, columns: list[str], label: str) -> None:
    """Assert exactly one active category per row for a dummy block."""

    sums = data[columns].sum(axis=1)
    invalid = sums.ne(1)
    if invalid.any():
        examples = data.loc[invalid, ["CHOICE_ID", *columns]].head(5).to_dict("records")
        raise AssertionError(
            f"{label} is not exactly one-hot for {int(invalid.sum()):,} rows. "
            f"Examples: {examples}"
        )


def validate_source(data: pd.DataFrame) -> None:
    """Validate upstream invariants before applying V1 exclusion rules."""

    assert_required_columns(data)
    choice_ids = data["CHOICE_ID"]
    if choice_ids.isna().any() or choice_ids.astype(str).str.strip().eq("").any():
        raise AssertionError("CHOICE_ID contains missing or blank values.")
    if not choice_ids.is_unique:
        duplicates = data.loc[choice_ids.duplicated(False), "CHOICE_ID"].head(10).tolist()
        raise AssertionError(f"CHOICE_ID is not unique. Examples: {duplicates}")

    data["CHOICE"] = pd.to_numeric(data["CHOICE"], errors="coerce")
    if data["CHOICE"].isna().any() or not data["CHOICE"].isin([1, 2]).all():
        raise AssertionError("CHOICE must contain only 1 or 2.")
    data["CHOICE"] = data["CHOICE"].astype(np.int8)

    assert_binary(data, ["AV_1", "AV_2"], "Availability")
    chosen_available = np.where(data["CHOICE"].eq(1), data["AV_1"], data["AV_2"])
    if not np.equal(chosen_available, 1).all():
        raise AssertionError("At least one chosen alternative is unavailable.")

    for car in (1, 2):
        pt_columns = [f"PT{car}_{category}" for category in POWERTRAIN_FULL]
        segment_columns = [f"SEG{car}_{category}" for category in SEGMENT_FULL]
        age_column = f"VEHICLE_AGE_{car}"
        age_missing_column = f"{age_column}_MISSING"
        assert_binary(data, pt_columns, f"Car-{car} powertrain block")
        assert_binary(data, segment_columns, f"Car-{car} segment block")
        assert_binary(data, [age_missing_column], f"Car-{car} vehicle-age missing flag")
        assert_one_hot(data, pt_columns, f"Car-{car} full powertrain block")
        assert_one_hot(data, segment_columns, f"Car-{car} full segment block")

        raw = data[age_column]
        direct_missing = raw.isna() | raw.astype("string").str.strip().eq("")
        numeric = pd.to_numeric(raw, errors="coerce")
        nonnumeric_populated = ~direct_missing & numeric.isna()
        if nonnumeric_populated.any():
            examples = data.loc[nonnumeric_populated, ["CHOICE_ID", age_column]].head(5)
            raise AssertionError(
                f"{age_column} contains populated non-numeric values. Examples: "
                f"{examples.to_dict('records')}"
            )
        indicator_missing = data[age_missing_column].eq(1)
        mismatch = indicator_missing.ne(direct_missing)
        if mismatch.any():
            examples = data.loc[
                mismatch, ["CHOICE_ID", age_column, age_missing_column]
            ].head(5)
            raise AssertionError(
                f"{age_missing_column} disagrees with direct missingness in {age_column}. "
                f"Examples: {examples.to_dict('records')}"
            )
        data[age_column] = numeric.astype(float)

    data["W_GEW"] = pd.to_numeric(data["W_GEW"], errors="coerce").astype(float)


def build_v1_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply V1 complete-case exclusions and return transparent sample flow."""

    invalid_powertrain = data[
        ["PT1_OTHER", "PT1_MISSING", "PT2_OTHER", "PT2_MISSING"]
    ].eq(1).any(axis=1)
    invalid_segment = data[["SEG1_MISSING", "SEG2_MISSING"]].eq(1).any(axis=1)

    age1 = data["VEHICLE_AGE_1"]
    age2 = data["VEHICLE_AGE_2"]
    invalid_age = (
        data[["VEHICLE_AGE_1_MISSING", "VEHICLE_AGE_2_MISSING"]].eq(1).any(axis=1)
        | age1.isna()
        | age2.isna()
        | ~np.isfinite(age1)
        | ~np.isfinite(age2)
        | age1.lt(0)
        | age2.lt(0)
    )
    excluded = invalid_powertrain | invalid_segment | invalid_age
    retained = data.loc[~excluded].copy()

    initial_n = len(data)
    counts = {
        "excluded_invalid_or_other_powertrain": int(invalid_powertrain.sum()),
        "excluded_missing_or_invalid_segment": int(invalid_segment.sum()),
        "excluded_missing_or_invalid_vehicle_age": int(invalid_age.sum()),
        "overlap_powertrain_and_segment": int((invalid_powertrain & invalid_segment).sum()),
        "overlap_powertrain_and_vehicle_age": int((invalid_powertrain & invalid_age).sum()),
        "overlap_segment_and_vehicle_age": int((invalid_segment & invalid_age).sum()),
        "overlap_all_three_reasons": int(
            (invalid_powertrain & invalid_segment & invalid_age).sum()
        ),
        "total_excluded_unique_occasions": int(excluded.sum()),
        "final_v1_estimation_occasions": len(retained),
    }
    notes = {
        "excluded_invalid_or_other_powertrain": (
            "Non-exclusive: either vehicle has PT*_OTHER=1 or PT*_MISSING=1."
        ),
        "excluded_missing_or_invalid_segment": (
            "Non-exclusive: either vehicle has SEG*_MISSING=1; full source blocks were validated one-hot."
        ),
        "excluded_missing_or_invalid_vehicle_age": (
            "Non-exclusive: either raw age is missing, non-finite, or negative, or its explicit missing flag is active."
        ),
        "overlap_powertrain_and_segment": "Overlap of powertrain and segment reasons, including triple overlap.",
        "overlap_powertrain_and_vehicle_age": "Overlap of powertrain and age reasons, including triple overlap.",
        "overlap_segment_and_vehicle_age": "Overlap of segment and age reasons, including triple overlap.",
        "overlap_all_three_reasons": "Intersection of all three exclusion reasons.",
        "total_excluded_unique_occasions": "Union of the three non-exclusive exclusion reasons.",
        "final_v1_estimation_occasions": (
            "Both vehicles have accepted powertrain and segment and numeric finite non-negative age."
        ),
    }
    rows = [
        {
            "metric": "initial_single_active_driver_occasions",
            "count": initial_n,
            "share_of_initial": 1.0,
            "notes": "One row per unique CHOICE_ID in the accepted single-driver design matrix.",
        }
    ]
    rows.extend(
        {
            "metric": metric,
            "count": count,
            "share_of_initial": count / initial_n,
            "notes": notes[metric],
        }
        for metric, count in counts.items()
    )
    validate_retained_sample(retained)
    return retained, pd.DataFrame(rows)


def validate_retained_sample(data: pd.DataFrame) -> None:
    """Assert all model-specific conditions on the retained V1 sample."""

    if data["CHOICE_ID"].isna().any() or not data["CHOICE_ID"].is_unique:
        raise AssertionError("Retained CHOICE_ID values must be non-missing and unique.")
    if not data["CHOICE"].isin([1, 2]).all():
        raise AssertionError("Retained CHOICE values are not limited to 1 and 2.")
    if not data[["AV_1", "AV_2"]].isin([0, 1]).all().all():
        raise AssertionError("Retained availability values are not binary.")
    chosen_available = np.where(data["CHOICE"].eq(1), data["AV_1"], data["AV_2"])
    if not np.equal(chosen_available, 1).all():
        raise AssertionError("A retained chosen alternative is unavailable.")

    for car in (1, 2):
        assert_one_hot(
            data,
            [f"PT{car}_{category}" for category in POWERTRAINS],
            f"Retained Car-{car} accepted powertrain block",
        )
        assert_one_hot(
            data,
            [f"SEG{car}_{category}" for category in SEGMENTS],
            f"Retained Car-{car} accepted segment block",
        )
        if data[f"VEHICLE_AGE_{car}_MISSING"].ne(0).any():
            raise AssertionError(f"Retained Car-{car} vehicle-age missing flag is active.")
        ages = data[f"VEHICLE_AGE_{car}"].to_numpy(dtype=float)
        if not np.isfinite(ages).all() or not np.greater_equal(ages, 0).all():
            raise AssertionError(
                f"Retained VEHICLE_AGE_{car} must be numeric, finite, and non-negative."
            )

    weights = data["W_GEW"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or not np.greater(weights, 0).all():
        raise AssertionError("W_GEW must be finite and positive on the weighted sample.")


def dummy_category(
    data: pd.DataFrame, prefix: str, categories: tuple[str, ...]
) -> pd.Series:
    """Recover category labels from a validated one-hot block."""

    columns = [f"{prefix}_{category}" for category in categories]
    positions = data[columns].to_numpy().argmax(axis=1)
    return pd.Series(np.asarray(categories, dtype=object)[positions], index=data.index)


def build_identification_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-choice V1 variation and retained vehicle age."""

    n = len(data)
    weights = data["W_GEW"].to_numpy(dtype=float)
    weight_total = float(weights.sum())
    pt1 = dummy_category(data, "PT1", POWERTRAINS)
    pt2 = dummy_category(data, "PT2", POWERTRAINS)
    seg1 = dummy_category(data, "SEG1", SEGMENTS)
    seg2 = dummy_category(data, "SEG2", SEGMENTS)
    age1 = data["VEHICLE_AGE_1"].to_numpy(dtype=float)
    age2 = data["VEHICLE_AGE_2"].to_numpy(dtype=float)
    delta_age = age1 - age2
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        metric: str,
        category: str,
        mask: np.ndarray | pd.Series,
        denominator_count: float = n,
        weighted_denominator: float = weight_total,
        notes: str = "",
    ) -> None:
        mask_array = np.asarray(mask, dtype=bool)
        count = int(mask_array.sum())
        weighted_count = float(weights[mask_array].sum())
        rows.append(
            {
                "section": section,
                "metric": metric,
                "category": category,
                "count": count,
                "share": count / denominator_count,
                "weighted_count": weighted_count,
                "weighted_share": weighted_count / weighted_denominator,
                "denominator_count": denominator_count,
                "weighted_denominator": weighted_denominator,
                "value": np.nan,
                "notes": notes,
            }
        )

    def add_value(section: str, metric: str, category: str, value: float, notes: str) -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "category": category,
                "count": np.nan,
                "share": np.nan,
                "weighted_count": np.nan,
                "weighted_share": np.nan,
                "denominator_count": np.nan,
                "weighted_denominator": np.nan,
                "value": float(value),
                "notes": notes,
            }
        )

    for category in POWERTRAINS:
        car1 = pt1.eq(category).to_numpy()
        car2 = pt2.eq(category).to_numpy()
        rows.append(
            {
                "section": "A_alternative_powertrain_composition",
                "metric": "powertrain",
                "category": category,
                "count": int(car1.sum() + car2.sum()),
                "share": float((car1.sum() + car2.sum()) / (2 * n)),
                "weighted_count": float(weights[car1].sum() + weights[car2].sum()),
                "weighted_share": float(
                    (weights[car1].sum() + weights[car2].sum()) / (2 * weight_total)
                ),
                "denominator_count": 2 * n,
                "weighted_denominator": 2 * weight_total,
                "value": np.nan,
                "notes": "Alternative-level composition; each occasion contributes two vehicles.",
            }
        )

    same_pt = pt1.eq(pt2).to_numpy()
    add("B_choice_set_powertrain", "same_vs_different", "same", same_pt)
    add("B_choice_set_powertrain", "same_vs_different", "different", ~same_pt)
    for first_index, first in enumerate(POWERTRAINS):
        for second in POWERTRAINS[first_index:]:
            if first == second:
                pair_mask = pt1.eq(first) & pt2.eq(second)
            else:
                pair_mask = (pt1.eq(first) & pt2.eq(second)) | (
                    pt1.eq(second) & pt2.eq(first)
                )
            add("B_choice_set_powertrain", "unordered_pair", f"{first}-{second}", pair_mask)

    for category in SEGMENTS:
        car1 = seg1.eq(category).to_numpy()
        car2 = seg2.eq(category).to_numpy()
        rows.append(
            {
                "section": "C_alternative_segment_composition",
                "metric": "segment",
                "category": category,
                "count": int(car1.sum() + car2.sum()),
                "share": float((car1.sum() + car2.sum()) / (2 * n)),
                "weighted_count": float(weights[car1].sum() + weights[car2].sum()),
                "weighted_share": float(
                    (weights[car1].sum() + weights[car2].sum()) / (2 * weight_total)
                ),
                "denominator_count": 2 * n,
                "weighted_denominator": 2 * weight_total,
                "value": np.nan,
                "notes": "Alternative-level composition; each occasion contributes two vehicles.",
            }
        )

    same_segment = seg1.eq(seg2).to_numpy()
    add("D_choice_set_segment", "same_vs_different", "same", same_segment)
    add("D_choice_set_segment", "same_vs_different", "different", ~same_segment)

    same_age = np.equal(age1, age2)
    add("E_vehicle_age_variation", "same_vs_different", "same_age", same_age)
    add("E_vehicle_age_variation", "same_vs_different", "different_age", ~same_age)
    add("E_vehicle_age_variation", "delta_nonzero", "D_VEHICLE_AGE_not_zero", delta_age != 0)
    for label, values in (
        ("VEHICLE_AGE_1", age1),
        ("VEHICLE_AGE_2", age2),
        ("D_VEHICLE_AGE", delta_age),
    ):
        for statistic, value in (
            ("min", np.min(values)),
            ("mean", np.mean(values)),
            ("median", np.median(values)),
            ("max", np.max(values)),
        ):
            add_value(
                "E_vehicle_age_descriptive_summary",
                "raw_years_no_transformation",
                f"{label}_{statistic}",
                value,
                "Retained values; no cap, winsorisation, standardisation, or other transformation.",
            )

    variation_map = {
        "B_PT_HEV": data["PT1_HEV"].ne(data["PT2_HEV"]),
        "B_PT_PHEV": data["PT1_PHEV"].ne(data["PT2_PHEV"]),
        "B_PT_BEV": data["PT1_BEV"].ne(data["PT2_BEV"]),
        "B_SEG_COMPACT": data["SEG1_COMPACT"].ne(data["SEG2_COMPACT"]),
        "B_SEG_MEDIUM": data["SEG1_MEDIUM"].ne(data["SEG2_MEDIUM"]),
        "B_SEG_LARGE": data["SEG1_LARGE"].ne(data["SEG2_LARGE"]),
        "B_VEHICLE_AGE": data["VEHICLE_AGE_1"].ne(data["VEHICLE_AGE_2"]),
    }
    for parameter, mask in variation_map.items():
        add(
            "F_parameter_specific_variation",
            "X_car1_not_equal_X_car2",
            parameter,
            mask,
            notes="Unequal alternative-specific values directly identify this coefficient.",
        )

    noninformative = same_pt & same_segment & same_age
    add(
        "G_non_informative_v1_occasions",
        "same_powertrain_segment_and_age",
        "non_informative_for_v1_coefficients",
        noninformative,
        notes=(
            "Kept in estimation. With no ASC and all V1 attributes identical, utilities are equal and the occasion does not identify V1 coefficients."
        ),
    )
    return pd.DataFrame(rows)


def build_delta_x_diagnostics(
    data: pd.DataFrame, diagnostics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the effective unlabelled design and Pearson correlation/VIF."""

    delta_x = pd.DataFrame(
        {
            predictor: data[car1_column].to_numpy(dtype=float)
            - data[car2_column].to_numpy(dtype=float)
            for predictor, _, car1_column, car2_column in DELTA_X_SPECS
        },
        index=data.index,
    )
    if not np.isfinite(delta_x.to_numpy()).all():
        raise AssertionError("The effective delta-X design contains non-finite values.")

    variation_counts = (
        diagnostics.loc[
            diagnostics["section"].eq("F_parameter_specific_variation"),
            ["category", "count"],
        ]
        .set_index("category")["count"]
        .astype(int)
    )
    for predictor, parameter, _, _ in DELTA_X_SPECS:
        observed = int(delta_x[predictor].ne(0).sum())
        expected = int(variation_counts.loc[parameter])
        if observed != expected:
            raise AssertionError(
                f"Diagnostic count mismatch for {predictor}/{parameter}: "
                f"delta-X nonzero {observed:,} != identifying variation {expected:,}."
            )

    correlation = delta_x.corr(method="pearson")
    design = delta_x.to_numpy(dtype=float)
    n = len(delta_x)
    rows: list[dict[str, Any]] = []
    for index, (predictor, _, _, _) in enumerate(DELTA_X_SPECS):
        target = design[:, index]
        centered = target - target.mean()
        total_sum_squares = float(centered @ centered)
        other_indices = [item for item in range(design.shape[1]) if item != index]
        auxiliary = np.column_stack([np.ones(n, dtype=float), design[:, other_indices]])
        if total_sum_squares == 0.0:
            vif = float("inf")
        else:
            try:
                coefficients, _, _, _ = np.linalg.lstsq(auxiliary, target, rcond=None)
                residuals = target - auxiliary @ coefficients
                residual_sum_squares = float(residuals @ residuals)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    f"VIF auxiliary regression failed numerically for {predictor}."
                ) from error
            if not np.isfinite(residual_sum_squares):
                raise RuntimeError(
                    f"VIF auxiliary regression returned non-finite RSS for {predictor}."
                )
            unexplained_share = residual_sum_squares / total_sum_squares
            vif = (
                float("inf")
                if unexplained_share <= np.finfo(float).eps
                else 1.0 / unexplained_share
            )

        other_correlations = correlation.loc[predictor].drop(index=predictor).abs().dropna()
        if other_correlations.empty:
            most_correlated_with = "unavailable"
            maximum_correlation = float("nan")
        else:
            most_correlated_with = str(other_correlations.idxmax())
            maximum_correlation = float(other_correlations.loc[most_correlated_with])
        if not np.isfinite(vif) or vif >= 10.0:
            flag = "strong multicollinearity warning"
        elif vif >= 5.0:
            flag = "moderate multicollinearity warning"
        else:
            flag = "no notable multicollinearity flag"
        if total_sum_squares == 0.0:
            flag += "; zero-variance predictor (VIF undefined/infinite)"
        if np.isfinite(maximum_correlation) and maximum_correlation >= 0.8:
            flag += "; high pairwise delta-X correlation"
        nonzero = int(np.count_nonzero(target))
        rows.append(
            {
                "predictor": predictor,
                "nonzero_count": nonzero,
                "nonzero_share": nonzero / n,
                "max_abs_correlation": maximum_correlation,
                "most_correlated_with": most_correlated_with,
                "VIF": vif,
                "diagnostic_flag": flag,
            }
        )
    return correlation, pd.DataFrame(rows)


def estimate_model(data: pd.DataFrame, model_name: str, weighted: bool):
    """Estimate one generic-coefficient V1 MNL using Biogeme 3.3.x."""

    model_data = data[MODEL_COLUMNS].copy()
    database = Database(model_name, model_data)
    choice = Variable("CHOICE")
    availability = {1: Variable("AV_1"), 2: Variable("AV_2")}

    beta = {name: Beta(name, 0, None, None, 0) for name in PARAMETERS}
    utilities = {
        car: (
            beta["B_PT_HEV"] * Variable(f"PT{car}_HEV")
            + beta["B_PT_PHEV"] * Variable(f"PT{car}_PHEV")
            + beta["B_PT_BEV"] * Variable(f"PT{car}_BEV")
            + beta["B_SEG_COMPACT"] * Variable(f"SEG{car}_COMPACT")
            + beta["B_SEG_MEDIUM"] * Variable(f"SEG{car}_MEDIUM")
            + beta["B_SEG_LARGE"] * Variable(f"SEG{car}_LARGE")
            + beta["B_VEHICLE_AGE"] * Variable(f"VEHICLE_AGE_{car}")
        )
        for car in (1, 2)
    }
    log_probability = models.loglogit(utilities, availability, choice)
    formulas = (
        {"log_like": log_probability, "weight": Variable("W_GEW")}
        if weighted
        else {"log_like": log_probability}
    )
    biogeme = BIOGEME(
        database,
        formulas,
        parameters=Parameters(),
        generate_html=False,
        generate_yaml=False,
        save_iterations=False,
    )
    biogeme.model_name = model_name
    available_count = model_data[["AV_1", "AV_2"]].sum(axis=1).to_numpy(dtype=float)
    null_contribution = -np.log(available_count)
    if weighted:
        null_contribution *= model_data["W_GEW"].to_numpy(dtype=float)
    biogeme.null_loglikelihood = float(null_contribution.sum())
    return biogeme.estimate()


def parameter_table(results: Any) -> pd.DataFrame:
    """Return estimates with robust inference."""

    return pd.DataFrame(
        [
            {
                "parameter": parameter,
                "estimate": results.get_parameter_value(parameter),
                "robust_standard_error": results.get_parameter_std_err(
                    parameter, EstimateVarianceCovariance.ROBUST
                ),
                "robust_t_statistic": results.get_parameter_t_test(
                    parameter, EstimateVarianceCovariance.ROBUST
                ),
                "robust_p_value": results.get_parameter_p_value(
                    parameter, EstimateVarianceCovariance.ROBUST
                ),
            }
            for parameter in PARAMETERS
        ]
    )


def robust_correlation_details(results: Any) -> tuple[float, str]:
    """Return the largest absolute off-diagonal robust correlation."""

    covariance = np.asarray(results.robust_variance_covariance_matrix, dtype=float)
    standard_deviations = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(standard_deviations, standard_deviations)
    np.fill_diagonal(correlation, np.nan)
    if not np.isfinite(correlation).any():
        return float("nan"), "unavailable"
    flat_index = int(np.nanargmax(np.abs(correlation)))
    row, column = np.unravel_index(flat_index, correlation.shape)
    return (
        abs(float(correlation[row, column])),
        f"{results.beta_names[row]} vs {results.beta_names[column]}",
    )


def model_warnings(
    model_name: str, results: Any, parameters: pd.DataFrame, data: pd.DataFrame
) -> list[str]:
    """Apply the V2 stability philosophy to all seven V1 coefficients."""

    warnings: list[str] = []
    if not results.algorithm_has_converged:
        warnings.append("Convergence failure: Biogeme reports that the optimizer did not converge.")
    numeric = parameters[
        ["estimate", "robust_standard_error", "robust_t_statistic", "robust_p_value"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        warnings.append("At least one estimate or robust inference statistic is NaN/infinite.")
    large_coefficients = parameters.loc[
        parameters["estimate"].abs().ge(ABS_COEFFICIENT_WARNING), "parameter"
    ].tolist()
    if large_coefficients:
        warnings.append(
            f"Absolute coefficient >= {ABS_COEFFICIENT_WARNING:g}: {', '.join(large_coefficients)}."
        )
    large_standard_errors = parameters.loc[
        parameters["robust_standard_error"].abs().ge(ROBUST_SE_WARNING), "parameter"
    ].tolist()
    if large_standard_errors:
        warnings.append(
            f"Robust standard error >= {ROBUST_SE_WARNING:g}: {', '.join(large_standard_errors)}."
        )
    max_correlation, pair = robust_correlation_details(results)
    if np.isfinite(max_correlation) and max_correlation >= CORRELATION_WARNING:
        warnings.append(
            f"Suspicious robust parameter correlation: |r|={max_correlation:.4f} for {pair}."
        )
    try:
        smallest_eigenvalue = float(results.smallest_eigenvalue)
        condition_number = float(results.condition_number)
        if smallest_eigenvalue <= HESSIAN_EIGENVALUE_WARNING:
            warnings.append(
                "Identification warning: smallest information-matrix eigenvalue "
                f"{smallest_eigenvalue:.4g} <= {HESSIAN_EIGENVALUE_WARNING:g}."
            )
        if not np.isfinite(condition_number) or condition_number >= HESSIAN_CONDITION_WARNING:
            warnings.append(
                "Hessian conditioning warning: condition number "
                f"{condition_number:.4g} >= {HESSIAN_CONDITION_WARNING:g}."
            )
    except Exception as error:
        warnings.append(f"Hessian diagnostics unavailable: {type(error).__name__}: {error}")

    variation_masks = {
        parameter: data[car1].ne(data[car2])
        for _, parameter, car1, car2 in DELTA_X_SPECS
    }
    n = len(data)
    variation_threshold = max(100, math.ceil(0.01 * n))
    for parameter, mask in variation_masks.items():
        count = int(mask.sum())
        if count < variation_threshold:
            warnings.append(
                f"Very small identifying variation for {parameter}: {count:,}/{n:,} occasions "
                f"(< {variation_threshold:,} threshold)."
            )

    alternative_total = 2 * n
    ice_share = float((data["PT1_ICE"].sum() + data["PT2_ICE"].sum()) / alternative_total)
    if ice_share >= 0.80:
        warnings.append(
            f"High ICE prevalence ({ice_share:.1%} of vehicle alternatives) limits information for non-ICE effects."
        )
    for category in ("HEV", "PHEV", "BEV"):
        share = float(
            (data[f"PT1_{category}"].sum() + data[f"PT2_{category}"].sum())
            / alternative_total
        )
        if share < 0.025:
            warnings.append(
                f"Sparse alternative-level {category} support ({share:.2%}); interpret {model_name} uncertainty cautiously."
            )
    for category in SEGMENTS:
        share = float(
            (data[f"SEG1_{category}"].sum() + data[f"SEG2_{category}"].sum())
            / alternative_total
        )
        if share < 0.025:
            warnings.append(
                f"Sparse alternative-level {category} segment support ({share:.2%}); interpret {model_name} uncertainty cautiously."
            )
    return warnings


def fit_summary(
    model_name: str,
    weighted: bool,
    results: Any,
    parameters: pd.DataFrame,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a one-row fit and numerical-stability summary."""

    null_ll = float(results.null_log_likelihood)
    final_ll = float(results.final_log_likelihood)
    parameter_count = int(results.number_of_parameters)
    max_correlation, correlation_pair = robust_correlation_details(results)
    warnings = model_warnings(model_name, results, parameters, data)
    raw = results.raw_estimation_results
    row = {
        "model": model_name,
        "weighted": weighted,
        "N": len(data),
        "sum_of_weights": float(data["W_GEW"].sum()) if weighted else float(len(data)),
        "number_of_parameters": parameter_count,
        "null_log_likelihood": null_ll,
        "initial_log_likelihood": float(results.initial_log_likelihood),
        "final_log_likelihood": final_ll,
        "rho_square_null": 1.0 - final_ll / null_ll,
        "adjusted_rho_square_null": 1.0 - (final_ll - parameter_count) / null_ll,
        "AIC": float(results.akaike_information_criterion),
        "BIC": float(results.bayesian_information_criterion),
        "convergence_status": "converged" if results.algorithm_has_converged else "failed",
        "final_gradient_norm": float(results.gradient_norm),
        "smallest_information_eigenvalue": float(results.smallest_eigenvalue),
        "hessian_condition_number": float(results.condition_number),
        "max_abs_robust_parameter_correlation": max_correlation,
        "most_correlated_parameter_pair": correlation_pair,
        "optimizer_messages": json.dumps(raw.optimization_messages, sort_keys=True, default=str),
        "warning_count": len(warnings),
        "warnings": " | ".join(warnings) if warnings else "none",
    }
    return pd.DataFrame([row]), warnings


def swap_vehicle_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Relabel all V1 attributes, availability, and choice in memory."""

    swapped = data.copy(deep=True)
    paired_columns = (
        [(f"PT1_{category}", f"PT2_{category}") for category in POWERTRAIN_FULL]
        + [(f"SEG1_{category}", f"SEG2_{category}") for category in SEGMENT_FULL]
        + [
            ("VEHICLE_AGE_1", "VEHICLE_AGE_2"),
            ("VEHICLE_AGE_1_MISSING", "VEHICLE_AGE_2_MISSING"),
            ("AV_1", "AV_2"),
        ]
    )
    for car1_column, car2_column in paired_columns:
        swapped[car1_column] = data[car2_column].to_numpy()
        swapped[car2_column] = data[car1_column].to_numpy()
    swapped["CHOICE"] = 3 - data["CHOICE"]
    validate_retained_sample(swapped)
    return swapped


def label_invariance_table(
    original_parameters: pd.DataFrame, swapped_results: Any
) -> tuple[pd.DataFrame, bool]:
    """Compare original and relabelled unweighted coefficients."""

    originals = original_parameters.set_index("parameter")["estimate"]
    rows = []
    all_passed = True
    for parameter in PARAMETERS:
        original = float(originals.loc[parameter])
        swapped = float(swapped_results.get_parameter_value(parameter))
        passed = bool(np.isclose(original, swapped, atol=LABEL_ATOL, rtol=LABEL_RTOL))
        all_passed = all_passed and passed
        rows.append(
            {
                "parameter": parameter,
                "original_estimate": original,
                "swapped_estimate": swapped,
                "absolute_difference": abs(original - swapped),
                "absolute_tolerance": LABEL_ATOL,
                "relative_tolerance": LABEL_RTOL,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows), all_passed


def compare_with_v2(source: pd.DataFrame, retained_v1: pd.DataFrame) -> dict[str, Any]:
    """Compare V1 IDs with the current V2-rule universe without estimating V2."""

    comparison: dict[str, Any] = {
        "v2_n": None,
        "same_n": None,
        "same_choice_id_universe": None,
        "source": "unavailable",
    }
    required = {
        "STATUS1_MISSING",
        "STATUS2_MISSING",
        "PT1_OTHER",
        "PT1_MISSING",
        "PT2_OTHER",
        "PT2_MISSING",
    }
    if required.issubset(source.columns):
        v2_excluded = source[
            ["PT1_OTHER", "PT1_MISSING", "PT2_OTHER", "PT2_MISSING"]
        ].eq(1).any(axis=1) | source[["STATUS1_MISSING", "STATUS2_MISSING"]].eq(1).any(axis=1)
        v2_ids = set(source.loc[~v2_excluded, "CHOICE_ID"])
        comparison.update(
            {
                "v2_n": len(v2_ids),
                "same_n": len(v2_ids) == len(retained_v1),
                "same_choice_id_universe": v2_ids == set(retained_v1["CHOICE_ID"]),
                "source": "recomputed from accepted source using the existing V2 exclusion rules",
            }
        )

    flow_path = V2_RESULTS_DIR / "v2_estimation_sample_flow.csv"
    if flow_path.exists():
        flow = pd.read_csv(flow_path)
        match = flow.loc[flow["metric"].eq("final_v2_estimation_occasions"), "count"]
        if len(match) == 1:
            existing_n = int(match.iloc[0])
            if comparison["v2_n"] is not None and existing_n != comparison["v2_n"]:
                raise AssertionError(
                    f"Existing V2 sample flow reports {existing_n:,}, but current V2 rules yield "
                    f"{comparison['v2_n']:,}."
                )
            comparison["v2_n"] = existing_n
            comparison["same_n"] = existing_n == len(retained_v1)
            comparison["source"] += "; confirmed by existing V2 sample-flow output"
    return comparison


def interpretation_text(parameter: str, estimate: float) -> str:
    """Return the requested sign-aware mechanical interpretation."""

    direction = "higher" if estimate > 0 else "lower" if estimate < 0 else "unchanged"
    if parameter.startswith("B_PT_"):
        category = parameter.removeprefix("B_PT_")
        return (
            f"{parameter} ({estimate:.6g}): Relative to ICE, {category} is associated with "
            f"{direction} relative utility for household vehicle use, holding Segment and Vehicle Age constant."
        )
    if parameter.startswith("B_SEG_"):
        category = parameter.removeprefix("B_SEG_").title()
        return (
            f"{parameter} ({estimate:.6g}): Relative to Small, {category} is associated with "
            f"{direction} relative utility for household vehicle use, holding Powertrain and Vehicle Age constant."
        )
    return (
        f"{parameter} ({estimate:.6g}): A one-year increase in vehicle age is associated with "
        f"{direction} relative utility for household vehicle use, holding Powertrain and Segment constant."
    )


def markdown_parameter_table(parameters: pd.DataFrame) -> str:
    display = parameters.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    return display.to_markdown(index=False)


def markdown_delta_x_table(delta_x_vif: pd.DataFrame) -> str:
    display = delta_x_vif.rename(
        columns={
            "predictor": "ΔX predictor",
            "nonzero_count": "non-zero occasions",
            "nonzero_share": "non-zero share",
            "max_abs_correlation": "maximum absolute correlation",
            "most_correlated_with": "most-correlated predictor",
            "diagnostic_flag": "diagnostic flag",
        }
    ).copy()
    display["non-zero occasions"] = display["non-zero occasions"].map(lambda value: f"{value:,}")
    display["non-zero share"] = display["non-zero share"].map(lambda value: f"{value:.2%}")
    display["maximum absolute correlation"] = display["maximum absolute correlation"].map(
        lambda value: f"{value:.4f}" if np.isfinite(value) else "undefined"
    )
    display["VIF"] = display["VIF"].map(
        lambda value: f"{value:.4f}" if np.isfinite(value) else "infinite"
    )
    return display.to_markdown(index=False)


def build_markdown_summary(
    sample_flow: pd.DataFrame,
    diagnostics: pd.DataFrame,
    delta_x_vif: pd.DataFrame,
    unweighted_parameters: pd.DataFrame,
    weighted_parameters: pd.DataFrame,
    unweighted_fit: pd.DataFrame,
    weighted_fit: pd.DataFrame,
    unweighted_warnings: list[str],
    weighted_warnings: list[str],
    invariance: pd.DataFrame,
    invariance_passed: bool,
    v2_comparison: dict[str, Any],
) -> str:
    """Generate the thesis-facing methodological and result summary."""

    counts = sample_flow.set_index("metric")["count"]
    n = int(counts["final_v1_estimation_occasions"])
    diag = diagnostics.set_index(["section", "metric", "category"])

    def diagnostic_value(section: str, metric: str, category: str, column: str) -> float:
        return float(diag.loc[(section, metric, category), column])

    same_pt = diagnostic_value("B_choice_set_powertrain", "same_vs_different", "same", "count")
    different_pt = diagnostic_value("B_choice_set_powertrain", "same_vs_different", "different", "count")
    same_seg = diagnostic_value("D_choice_set_segment", "same_vs_different", "same", "count")
    different_seg = diagnostic_value("D_choice_set_segment", "same_vs_different", "different", "count")
    same_age = diagnostic_value("E_vehicle_age_variation", "same_vs_different", "same_age", "count")
    different_age = diagnostic_value("E_vehicle_age_variation", "same_vs_different", "different_age", "count")
    noninformative = diagnostic_value(
        "G_non_informative_v1_occasions",
        "same_powertrain_segment_and_age",
        "non_informative_for_v1_coefficients",
        "count",
    )
    variation_lines = []
    for parameter in PARAMETERS:
        count = diagnostic_value(
            "F_parameter_specific_variation", "X_car1_not_equal_X_car2", parameter, "count"
        )
        weighted_count = diagnostic_value(
            "F_parameter_specific_variation",
            "X_car1_not_equal_X_car2",
            parameter,
            "weighted_count",
        )
        variation_lines.append(
            f"- {parameter}: {int(count):,} occasions ({count / n:.2%}); weighted count {weighted_count:,.3f}."
        )
    age_rows = diagnostics.loc[
        diagnostics["section"].eq("E_vehicle_age_descriptive_summary"), ["category", "value"]
    ]
    age_summary = "; ".join(
        f"{row.category}={row.value:.3f}" for row in age_rows.itertuples(index=False)
    )
    interpretations = "\n".join(
        f"- {interpretation_text(row.parameter, row.estimate)}"
        for row in unweighted_parameters.itertuples(index=False)
    )
    warning_block = lambda items: "\n".join(f"- {item}" for item in items) if items else "- None."
    maximum_difference = float(invariance["absolute_difference"].max())
    v2_n = "unavailable" if v2_comparison["v2_n"] is None else f"{v2_comparison['v2_n']:,}"
    sample_relation = (
        "same N and same CHOICE_ID universe"
        if v2_comparison["same_choice_id_universe"] is True
        else "different N and/or different CHOICE_ID universe"
    )
    comparison_warning = ""
    if v2_comparison["same_choice_id_universe"] is not True:
        comparison_warning = (
            "\n\nV1 and V2 are currently model-specific Stage-0 benchmarks. "
            "Their fit statistics should not be directly compared as if they were estimated on the same observations. "
            "A common-sample comparison can be performed later for vehicle-representation selection."
        )

    return f"""# Stage-0 V1-only unlabelled MNL

## Estimation sample

- Source: `{SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve()}`.
- Source branch: single-active-driver design matrix.
- Initial occasions: {int(counts['initial_single_active_driver_occasions']):,}.
- Invalid/OTHER powertrain reason count: {int(counts['excluded_invalid_or_other_powertrain']):,}.
- Missing/invalid segment reason count: {int(counts['excluded_missing_or_invalid_segment']):,}.
- Missing/invalid vehicle-age reason count: {int(counts['excluded_missing_or_invalid_vehicle_age']):,}.
- Pairwise overlaps: powertrain/segment {int(counts['overlap_powertrain_and_segment']):,}; powertrain/age {int(counts['overlap_powertrain_and_vehicle_age']):,}; segment/age {int(counts['overlap_segment_and_vehicle_age']):,}.
- All-three overlap: {int(counts['overlap_all_three_reasons']):,}.
- Total unique excluded occasions: {int(counts['total_excluded_unique_occasions']):,}.
- Final V1 estimation sample: {n:,} choice occasions.
- No missing value was imputed. Vehicle Age remains in raw years with no cap, winsorisation, spline, category, standardisation, or nonlinear term.

Biogeme version: {get_biogeme_version()}.

## Specification and identification

This unlabelled two-alternative household vehicle-use model has no alternative-specific constant and applies the same seven generic coefficients to both vehicles. Powertrain reference: ICE. Segment reference: SMALL. Vehicle Age is continuous in years.

`Vj = B_PT_HEV*PTj_HEV + B_PT_PHEV*PTj_PHEV + B_PT_BEV*PTj_BEV + B_SEG_COMPACT*SEGj_COMPACT + B_SEG_MEDIUM*SEGj_MEDIUM + B_SEG_LARGE*SEGj_LARGE + B_VEHICLE_AGE*VEHICLE_AGE_j`, for j in {{1, 2}}.

- Same powertrain: {int(same_pt):,} ({same_pt / n:.2%}); different: {int(different_pt):,} ({different_pt / n:.2%}).
- Same segment: {int(same_seg):,} ({same_seg / n:.2%}); different: {int(different_seg):,} ({different_seg / n:.2%}).
- Same age: {int(same_age):,} ({same_age / n:.2%}); different age / `D_VEHICLE_AGE != 0`: {int(different_age):,} ({different_age / n:.2%}).
- Retained age summary: {age_summary}.
- Same powertrain, segment, and age: {int(noninformative):,} ({noninformative / n:.2%}). These valid occasions remain in estimation but do not identify the V1 coefficients.

Parameter-specific identifying variation:

{chr(10).join(variation_lines)}

## Effective ΔX diagnostics

Identification operates through `X_car1 - X_car2`; Pearson correlation and VIF therefore use the seven effective predictors.

{markdown_delta_x_table(delta_x_vif)}

Flags are descriptive only and never remove a predictor.

## Unweighted results

{markdown_parameter_table(unweighted_parameters)}

Fit: null LL {unweighted_fit.at[0, 'null_log_likelihood']:.6f}; final LL {unweighted_fit.at[0, 'final_log_likelihood']:.6f}; rho-square {unweighted_fit.at[0, 'rho_square_null']:.6f}; adjusted rho-square {unweighted_fit.at[0, 'adjusted_rho_square_null']:.6f}; AIC {unweighted_fit.at[0, 'AIC']:.6f}; BIC {unweighted_fit.at[0, 'BIC']:.6f}; convergence {unweighted_fit.at[0, 'convergence_status']}.

## W_GEW-weighted results

{markdown_parameter_table(weighted_parameters)}

Fit: sum of weights {weighted_fit.at[0, 'sum_of_weights']:.6f}; weighted null LL {weighted_fit.at[0, 'null_log_likelihood']:.6f}; final weighted LL {weighted_fit.at[0, 'final_log_likelihood']:.6f}; rho-square {weighted_fit.at[0, 'rho_square_null']:.6f}; adjusted rho-square {weighted_fit.at[0, 'adjusted_rho_square_null']:.6f}; AIC {weighted_fit.at[0, 'AIC']:.6f}; BIC {weighted_fit.at[0, 'BIC']:.6f}; convergence {weighted_fit.at[0, 'convergence_status']}.

## Mechanical interpretation templates

{interpretations}

These are conditional within-household vehicle-use effects. They are not market shares or vehicle-purchase preferences.

## Stability warnings

Unweighted:

{warning_block(unweighted_warnings)}

Weighted:

{warning_block(weighted_warnings)}

Warnings never remove a variable, age value, or occasion.

## Label-invariance check

Result: **{'PASS' if invariance_passed else 'FAIL'}**. Maximum absolute coefficient difference after swapping all Car-1/Car-2 V1 attributes, availability, and CHOICE labels in memory: {maximum_difference:.3g}. Tolerances: absolute {LABEL_ATOL:g}, relative {LABEL_RTOL:g}. The source CSV was not changed.

## Stage-0 comparison scope

- Initial source N: {int(counts['initial_single_active_driver_occasions']):,}.
- Final V1 N: {n:,}.
- Existing final V2 N: {v2_n} ({v2_comparison['source']}).
- Sample relationship: {sample_relation}.{comparison_warning}
"""


def write_results(
    sample_flow: pd.DataFrame,
    diagnostics: pd.DataFrame,
    delta_x_correlation: pd.DataFrame,
    delta_x_vif: pd.DataFrame,
    unweighted_parameters: pd.DataFrame,
    weighted_parameters: pd.DataFrame,
    unweighted_fit: pd.DataFrame,
    weighted_fit: pd.DataFrame,
    invariance: pd.DataFrame,
    summary: str,
) -> list[Path]:
    """Write only V1-specific outputs."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        RESULTS_DIR / "v1_estimation_sample_flow.csv": sample_flow,
        RESULTS_DIR / "v1_identification_diagnostics.csv": diagnostics,
        RESULTS_DIR / "v1_delta_x_vif.csv": delta_x_vif,
        RESULTS_DIR / "v1_only_unweighted_parameters.csv": unweighted_parameters,
        RESULTS_DIR / "v1_only_unweighted_fit_summary.csv": unweighted_fit,
        RESULTS_DIR / "v1_only_weighted_parameters.csv": weighted_parameters,
        RESULTS_DIR / "v1_only_weighted_fit_summary.csv": weighted_fit,
        RESULTS_DIR / "v1_label_invariance_check.csv": invariance,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    correlation_path = RESULTS_DIR / "v1_delta_x_correlation.csv"
    delta_x_correlation.to_csv(correlation_path, index=True, index_label="predictor")
    summary_path = RESULTS_DIR / "v1_only_model_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    written = [*outputs, correlation_path, summary_path]
    for path in written:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected non-empty result file was not written: {path}")
    return written


def print_run_summary(
    sample_flow: pd.DataFrame,
    delta_x_vif: pd.DataFrame,
    unweighted_parameters: pd.DataFrame,
    weighted_parameters: pd.DataFrame,
    unweighted_fit: pd.DataFrame,
    weighted_fit: pd.DataFrame,
    invariance_passed: bool,
    v2_comparison: dict[str, Any],
    outputs: list[Path],
) -> None:
    """Print compact execution and QA results."""

    print("\nSTAGE-0 V1-ONLY UNLABELLED MNL")
    print(f"Biogeme version: {get_biogeme_version()}")
    print(f"Input: {SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve()}")
    print("\nESTIMATION SAMPLE")
    print(sample_flow[["metric", "count", "share_of_initial"]].to_string(index=False))
    print("\nEFFECTIVE DELTA-X VIF")
    print(delta_x_vif.to_string(index=False))
    print("\nUNWEIGHTED PARAMETERS")
    print(unweighted_parameters.to_string(index=False))
    print("\nWEIGHTED PARAMETERS")
    print(weighted_parameters.to_string(index=False))
    print("\nFIT")
    print(
        pd.concat([unweighted_fit, weighted_fit], ignore_index=True)[
            [
                "model",
                "N",
                "sum_of_weights",
                "null_log_likelihood",
                "final_log_likelihood",
                "rho_square_null",
                "adjusted_rho_square_null",
                "AIC",
                "BIC",
                "convergence_status",
                "warning_count",
            ]
        ].to_string(index=False)
    )
    print(f"\nLABEL INVARIANCE: {'PASS' if invariance_passed else 'FAIL'}")
    print(
        "V2 COMPARISON: "
        f"N={v2_comparison['v2_n']}; same N={v2_comparison['same_n']}; "
        f"same CHOICE_ID universe={v2_comparison['same_choice_id_universe']}"
    )
    print("\nOUTPUTS")
    for path in outputs:
        print(path.resolve())


def main() -> None:
    if not SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.exists():
        raise FileNotFoundError(
            "Accepted single-driver design matrix not found: "
            f"{SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve()}"
        )

    source_hash_before = sha256(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH)
    source = pd.read_csv(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, low_memory=False)
    validate_source(source)
    retained, sample_flow = build_v1_sample(source)
    diagnostics = build_identification_diagnostics(retained)
    delta_x_correlation, delta_x_vif = build_delta_x_diagnostics(retained, diagnostics)
    v2_comparison = compare_with_v2(source, retained)

    unweighted_results = estimate_model(retained, "v1_only_unweighted", weighted=False)
    weighted_results = estimate_model(retained, "v1_only_weighted", weighted=True)
    unweighted_parameters = parameter_table(unweighted_results)
    weighted_parameters = parameter_table(weighted_results)
    unweighted_fit, unweighted_warnings = fit_summary(
        "v1_only_unweighted", False, unweighted_results, unweighted_parameters, retained
    )
    weighted_fit, weighted_warnings = fit_summary(
        "v1_only_weighted", True, weighted_results, weighted_parameters, retained
    )

    swapped = swap_vehicle_labels(retained)
    swapped_results = estimate_model(swapped, "v1_only_unweighted_label_swap", weighted=False)
    invariance, invariance_passed = label_invariance_table(
        unweighted_parameters, swapped_results
    )
    summary = build_markdown_summary(
        sample_flow,
        diagnostics,
        delta_x_vif,
        unweighted_parameters,
        weighted_parameters,
        unweighted_fit,
        weighted_fit,
        unweighted_warnings,
        weighted_warnings,
        invariance,
        invariance_passed,
        v2_comparison,
    )

    if sha256(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH) != source_hash_before:
        raise AssertionError("The accepted source CSV changed during estimation; no results written.")
    outputs = write_results(
        sample_flow,
        diagnostics,
        delta_x_correlation,
        delta_x_vif,
        unweighted_parameters,
        weighted_parameters,
        unweighted_fit,
        weighted_fit,
        invariance,
        summary,
    )
    if sha256(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH) != source_hash_before:
        raise AssertionError("The accepted source CSV changed while results were written.")
    if not invariance_passed:
        raise AssertionError("The V1 Car-1/Car-2 label-invariance test failed.")

    print_run_summary(
        sample_flow,
        delta_x_vif,
        unweighted_parameters,
        weighted_parameters,
        unweighted_fit,
        weighted_fit,
        invariance_passed,
        v2_comparison,
        outputs,
    )


if __name__ == "__main__":
    main()
