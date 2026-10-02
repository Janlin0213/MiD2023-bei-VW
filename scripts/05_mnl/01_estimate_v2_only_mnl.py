"""Estimate the Stage-0 V2-only unlabelled household vehicle-use MNL.

The model uses generic powertrain and vehicle-status coefficients for both
household vehicles. Car 1 and Car 2 are arbitrary identifiers, so no
alternative-specific constant or car-label-specific coefficient is estimated.

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


RESULTS_DIR = OUTPUTS_DIR / "mnl" / "v2_only"

POWERTRAINS = ("ICE", "HEV", "PHEV", "BEV")
POWERTRAIN_FULL = (*POWERTRAINS, "OTHER", "MISSING")
STATUSES = ("LOW", "MEDIUM", "HIGH")
STATUS_FULL = (*STATUSES, "MISSING")

PARAMETERS = (
    "B_PT_HEV",
    "B_PT_PHEV",
    "B_PT_BEV",
    "B_STATUS_LOW",
    "B_STATUS_HIGH",
)

DELTA_X_SPECS = (
    ("D_PT_HEV", "B_PT_HEV", "PT1_HEV", "PT2_HEV"),
    ("D_PT_PHEV", "B_PT_PHEV", "PT1_PHEV", "PT2_PHEV"),
    ("D_PT_BEV", "B_PT_BEV", "PT1_BEV", "PT2_BEV"),
    ("D_STATUS_LOW", "B_STATUS_LOW", "STATUS1_LOW", "STATUS2_LOW"),
    ("D_STATUS_HIGH", "B_STATUS_HIGH", "STATUS1_HIGH", "STATUS2_HIGH"),
)

MODEL_COLUMNS = [
    "CHOICE",
    "AV_1",
    "AV_2",
    "W_GEW",
    *[f"PT{car}_{category}" for car in (1, 2) for category in POWERTRAINS],
    *[f"STATUS{car}_{category}" for car in (1, 2) for category in STATUSES],
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
    """Stop if the accepted design matrix lacks a required V2 field."""

    required = {
        "CHOICE_ID",
        "CHOICE",
        "AV_1",
        "AV_2",
        "W_GEW",
        *[f"PT{car}_{category}" for car in (1, 2) for category in POWERTRAIN_FULL],
        *[f"STATUS{car}_{category}" for car in (1, 2) for category in STATUS_FULL],
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise AssertionError(f"Required V2 input columns are missing: {missing}")


def assert_binary(data: pd.DataFrame, columns: list[str], label: str) -> None:
    """Assert that a set of indicator columns contains only numeric 0/1."""

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
    """Validate upstream invariants before applying the V2 exclusion rules."""

    assert_required_columns(data)
    if data["CHOICE_ID"].isna().any() or data["CHOICE_ID"].astype(str).str.strip().eq("").any():
        raise AssertionError("CHOICE_ID contains missing or blank values.")
    if not data["CHOICE_ID"].is_unique:
        duplicates = data.loc[data["CHOICE_ID"].duplicated(False), "CHOICE_ID"].head(10).tolist()
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
        status_columns = [f"STATUS{car}_{category}" for category in STATUS_FULL]
        assert_binary(data, pt_columns, f"Car-{car} powertrain block")
        assert_binary(data, status_columns, f"Car-{car} status block")
        assert_one_hot(data, pt_columns, f"Car-{car} full powertrain block")
        assert_one_hot(data, status_columns, f"Car-{car} full status block")

    weights = pd.to_numeric(data["W_GEW"], errors="coerce")
    data["W_GEW"] = weights.astype(float)


def build_v2_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the prespecified V2 exclusions and return sample-flow details."""

    invalid_powertrain = (
        data[["PT1_OTHER", "PT1_MISSING", "PT2_OTHER", "PT2_MISSING"]]
        .eq(1)
        .any(axis=1)
    )
    missing_status = data[["STATUS1_MISSING", "STATUS2_MISSING"]].eq(1).any(axis=1)
    excluded = invalid_powertrain | missing_status
    retained = data.loc[~excluded].copy()

    initial_n = len(data)
    invalid_n = int(invalid_powertrain.sum())
    missing_status_n = int(missing_status.sum())
    overlap_n = int((invalid_powertrain & missing_status).sum())
    total_excluded_n = int(excluded.sum())
    final_n = len(retained)

    sample_flow = pd.DataFrame(
        [
            {
                "metric": "initial_single_active_driver_occasions",
                "count": initial_n,
                "share_of_initial": 1.0,
                "notes": "One row per unique CHOICE_ID in the accepted single-driver MNL design matrix.",
            },
            {
                "metric": "excluded_invalid_or_other_powertrain",
                "count": invalid_n,
                "share_of_initial": invalid_n / initial_n,
                "notes": "Non-exclusive reason count: either vehicle has PT*_OTHER=1 or PT*_MISSING=1.",
            },
            {
                "metric": "excluded_missing_status",
                "count": missing_status_n,
                "share_of_initial": missing_status_n / initial_n,
                "notes": "Non-exclusive reason count: either vehicle has STATUS*_MISSING=1.",
            },
            {
                "metric": "excluded_for_both_reasons",
                "count": overlap_n,
                "share_of_initial": overlap_n / initial_n,
                "notes": "Overlap between the two non-exclusive exclusion-reason counts.",
            },
            {
                "metric": "total_excluded_unique_occasions",
                "count": total_excluded_n,
                "share_of_initial": total_excluded_n / initial_n,
                "notes": "Union of invalid/OTHER powertrain and missing-status exclusions.",
            },
            {
                "metric": "final_v2_estimation_occasions",
                "count": final_n,
                "share_of_initial": final_n / initial_n,
                "notes": "Both vehicles have powertrain in ICE/HEV/PHEV/BEV and status in LOW/MEDIUM/HIGH.",
            },
        ]
    )

    validate_retained_sample(retained)
    return retained, sample_flow


def validate_retained_sample(data: pd.DataFrame) -> None:
    """Assert all model-specific conditions on the retained V2 sample."""

    if not data["CHOICE_ID"].is_unique:
        raise AssertionError("Retained CHOICE_ID values are not unique.")
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
            [f"STATUS{car}_{category}" for category in STATUSES],
            f"Retained Car-{car} accepted status block",
        )

    weights = data["W_GEW"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or not np.greater(weights, 0).all():
        raise AssertionError("W_GEW must be finite and positive on the weighted estimation sample.")


def dummy_category(
    data: pd.DataFrame,
    prefix: str,
    categories: tuple[str, ...],
) -> pd.Series:
    """Recover the category label from a validated one-hot block."""

    columns = [f"{prefix}_{category}" for category in categories]
    positions = data[columns].to_numpy().argmax(axis=1)
    return pd.Series(np.asarray(categories, dtype=object)[positions], index=data.index)


def build_identification_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-choice V2 variation, unweighted and W_GEW-weighted."""

    n = len(data)
    weights = data["W_GEW"].to_numpy(dtype=float)
    weight_total = float(weights.sum())
    pt1 = dummy_category(data, "PT1", POWERTRAINS)
    pt2 = dummy_category(data, "PT2", POWERTRAINS)
    status1 = dummy_category(data, "STATUS1", STATUSES)
    status2 = dummy_category(data, "STATUS2", STATUSES)

    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        metric: str,
        category: str,
        mask: np.ndarray | pd.Series,
        denominator_count: float,
        weighted_denominator: float,
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
                "notes": notes,
            }
        )

    for category in POWERTRAINS:
        car1_mask = pt1.eq(category).to_numpy()
        car2_mask = pt2.eq(category).to_numpy()
        alternative_count = int(car1_mask.sum() + car2_mask.sum())
        alternative_weight = float(weights[car1_mask].sum() + weights[car2_mask].sum())
        rows.append(
            {
                "section": "A_alternative_powertrain_composition",
                "metric": "powertrain",
                "category": category,
                "count": alternative_count,
                "share": alternative_count / (2 * n),
                "weighted_count": alternative_weight,
                "weighted_share": alternative_weight / (2 * weight_total),
                "denominator_count": 2 * n,
                "weighted_denominator": 2 * weight_total,
                "notes": "Alternative-level composition; each occasion contributes two vehicles.",
            }
        )

    same_pt = pt1.eq(pt2).to_numpy()
    add("B_choice_set_powertrain", "same_vs_different", "same", same_pt, n, weight_total)
    add("B_choice_set_powertrain", "same_vs_different", "different", ~same_pt, n, weight_total)

    for first_index, first in enumerate(POWERTRAINS):
        for second in POWERTRAINS[first_index:]:
            if first == second:
                pair_mask = pt1.eq(first) & pt2.eq(second)
            else:
                pair_mask = (pt1.eq(first) & pt2.eq(second)) | (
                    pt1.eq(second) & pt2.eq(first)
                )
            add(
                "B_choice_set_powertrain",
                "unordered_pair",
                f"{first}-{second}",
                pair_mask,
                n,
                weight_total,
            )

    same_status = status1.eq(status2).to_numpy()
    add("C_status_variation", "same_vs_different", "same", same_status, n, weight_total)
    add(
        "C_status_variation",
        "same_vs_different",
        "different",
        ~same_status,
        n,
        weight_total,
    )

    variation_map = {
        "B_PT_HEV": data["PT1_HEV"].ne(data["PT2_HEV"]),
        "B_PT_PHEV": data["PT1_PHEV"].ne(data["PT2_PHEV"]),
        "B_PT_BEV": data["PT1_BEV"].ne(data["PT2_BEV"]),
        "B_STATUS_LOW": data["STATUS1_LOW"].ne(data["STATUS2_LOW"]),
        "B_STATUS_HIGH": data["STATUS1_HIGH"].ne(data["STATUS2_HIGH"]),
    }
    for parameter, mask in variation_map.items():
        add(
            "D_parameter_specific_variation",
            "X_car1_not_equal_X_car2",
            parameter,
            mask,
            n,
            weight_total,
            notes="Only occasions with unequal alternative-specific dummy values directly identify this coefficient.",
        )

    noninformative = same_pt & same_status
    # Under the no-ASC unlabelled V2-only model, these valid occasions have
    # identical systematic utilities for both alternatives and therefore do
    # not identify the V2 coefficients. They remain in the estimation sample.
    add(
        "E_non_informative_v2_occasions",
        "same_powertrain_and_same_status",
        "non_informative_for_v2_coefficients",
        noninformative,
        n,
        weight_total,
        notes=(
            "Kept in estimation. With no ASC and identical V2 attributes, V1=V2, so these occasions do not identify V2 coefficients."
        ),
    )
    return pd.DataFrame(rows)


def build_delta_x_diagnostics(
    data: pd.DataFrame,
    identification_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the effective MNL design and its correlation/VIF diagnostics."""

    # For a two-alternative unlabelled logit, P(choice=1) depends on V1 - V2.
    # The effective regressors for identification are therefore X1 - X2, so
    # multicollinearity is checked on these delta-X columns rather than only on
    # pooled vehicle-level categorical associations.
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
        identification_diagnostics.loc[
            identification_diagnostics["section"].eq("D_parameter_specific_variation"),
            ["category", "count"],
        ]
        .set_index("category")["count"]
        .astype(int)
    )
    for predictor, parameter, _, _ in DELTA_X_SPECS:
        nonzero_count = int(delta_x[predictor].ne(0).sum())
        expected_count = int(variation_counts.loc[parameter])
        if nonzero_count != expected_count:
            raise AssertionError(
                f"Diagnostic count mismatch for {predictor}/{parameter}: "
                f"delta-X nonzero count {nonzero_count:,} != identifying-variation "
                f"count {expected_count:,}."
            )

    correlation = delta_x.corr(method="pearson")
    rows: list[dict[str, Any]] = []
    design = delta_x.to_numpy(dtype=float)
    n = len(delta_x)

    for column_index, (predictor, _, _, _) in enumerate(DELTA_X_SPECS):
        target = design[:, column_index]
        centered_target = target - target.mean()
        total_sum_squares = float(centered_target @ centered_target)

        other_indices = [index for index in range(design.shape[1]) if index != column_index]
        auxiliary_design = np.column_stack(
            [np.ones(n, dtype=float), design[:, other_indices]]
        )
        if total_sum_squares == 0.0:
            vif = float("inf")
        else:
            try:
                coefficients, _, _, _ = np.linalg.lstsq(
                    auxiliary_design, target, rcond=None
                )
                residuals = target - auxiliary_design @ coefficients
                residual_sum_squares = float(residuals @ residuals)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    f"VIF auxiliary regression failed numerically for {predictor}."
                ) from error
            if not np.isfinite(residual_sum_squares):
                raise RuntimeError(
                    f"VIF auxiliary regression returned a non-finite residual sum "
                    f"of squares for {predictor}."
                )
            unexplained_share = residual_sum_squares / total_sum_squares
            vif = (
                float("inf")
                if unexplained_share <= np.finfo(float).eps
                else 1.0 / unexplained_share
            )

        other_correlations = correlation.loc[predictor].drop(index=predictor).abs().dropna()
        if other_correlations.empty:
            max_abs_correlation = float("nan")
            most_correlated_with = "unavailable"
        else:
            most_correlated_with = str(other_correlations.idxmax())
            max_abs_correlation = float(other_correlations.loc[most_correlated_with])

        if not np.isfinite(vif) or vif >= 10.0:
            diagnostic_flag = "strong multicollinearity warning"
        elif vif >= 5.0:
            diagnostic_flag = "moderate multicollinearity warning"
        else:
            diagnostic_flag = "no notable multicollinearity flag"
        if total_sum_squares == 0.0:
            diagnostic_flag += "; zero-variance predictor (VIF undefined/infinite)"
        if np.isfinite(max_abs_correlation) and max_abs_correlation >= 0.8:
            diagnostic_flag += "; high pairwise delta-X correlation"

        nonzero_count = int(np.count_nonzero(target))
        rows.append(
            {
                "predictor": predictor,
                "nonzero_count": nonzero_count,
                "nonzero_share": nonzero_count / n,
                "max_abs_correlation": max_abs_correlation,
                "most_correlated_with": most_correlated_with,
                "VIF": vif,
                "diagnostic_flag": diagnostic_flag,
            }
        )

    return correlation, pd.DataFrame(rows)


def estimate_model(data: pd.DataFrame, model_name: str, weighted: bool):
    """Estimate one generic-coefficient V2 MNL using Biogeme 3.3.x."""

    model_data = data[MODEL_COLUMNS].copy()
    database = Database(model_name, model_data)

    choice = Variable("CHOICE")
    av1 = Variable("AV_1")
    av2 = Variable("AV_2")

    b_pt_hev = Beta("B_PT_HEV", 0, None, None, 0)
    b_pt_phev = Beta("B_PT_PHEV", 0, None, None, 0)
    b_pt_bev = Beta("B_PT_BEV", 0, None, None, 0)
    b_status_low = Beta("B_STATUS_LOW", 0, None, None, 0)
    b_status_high = Beta("B_STATUS_HIGH", 0, None, None, 0)

    v1 = (
        b_pt_hev * Variable("PT1_HEV")
        + b_pt_phev * Variable("PT1_PHEV")
        + b_pt_bev * Variable("PT1_BEV")
        + b_status_low * Variable("STATUS1_LOW")
        + b_status_high * Variable("STATUS1_HIGH")
    )
    v2 = (
        b_pt_hev * Variable("PT2_HEV")
        + b_pt_phev * Variable("PT2_PHEV")
        + b_pt_bev * Variable("PT2_BEV")
        + b_status_low * Variable("STATUS2_LOW")
        + b_status_high * Variable("STATUS2_HIGH")
    )
    utilities = {1: v1, 2: v2}
    availability = {1: av1, 2: av2}
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
        null_contribution = null_contribution * model_data["W_GEW"].to_numpy(dtype=float)
    biogeme.null_loglikelihood = float(null_contribution.sum())
    return biogeme.estimate()


def parameter_table(results: Any) -> pd.DataFrame:
    """Return the requested compact table with robust inference."""

    rows = []
    for parameter in PARAMETERS:
        rows.append(
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
        )
    return pd.DataFrame(rows)


def robust_correlation_details(results: Any) -> tuple[float, str]:
    """Return the largest absolute off-diagonal robust correlation."""

    covariance = np.asarray(results.robust_variance_covariance_matrix, dtype=float)
    standard_deviations = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviations, standard_deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    np.fill_diagonal(correlation, np.nan)
    if not np.isfinite(correlation).any():
        return float("nan"), "unavailable"
    flat_index = int(np.nanargmax(np.abs(correlation)))
    row, column = np.unravel_index(flat_index, correlation.shape)
    value = float(correlation[row, column])
    pair = f"{results.beta_names[row]} vs {results.beta_names[column]}"
    return abs(value), pair


def model_warnings(
    model_name: str,
    results: Any,
    parameters: pd.DataFrame,
    data: pd.DataFrame,
) -> list[str]:
    """Apply transparent stability flags without changing the specification."""

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
    except Exception as error:  # Biogeme may omit second derivatives after failure.
        warnings.append(f"Hessian diagnostics unavailable: {type(error).__name__}: {error}")

    n = len(data)
    variation_masks = {
        "B_PT_HEV": data["PT1_HEV"].ne(data["PT2_HEV"]),
        "B_PT_PHEV": data["PT1_PHEV"].ne(data["PT2_PHEV"]),
        "B_PT_BEV": data["PT1_BEV"].ne(data["PT2_BEV"]),
        "B_STATUS_LOW": data["STATUS1_LOW"].ne(data["STATUS2_LOW"]),
        "B_STATUS_HIGH": data["STATUS1_HIGH"].ne(data["STATUS2_HIGH"]),
    }
    variation_warning_threshold = max(100, math.ceil(0.01 * n))
    for parameter, mask in variation_masks.items():
        count = int(mask.sum())
        if count < variation_warning_threshold:
            warnings.append(
                f"Very small identifying variation for {parameter}: {count:,}/{n:,} occasions "
                f"(< {variation_warning_threshold:,} threshold)."
            )

    alternative_total = 2 * n
    ice_share = float(
        (data["PT1_ICE"].sum() + data["PT2_ICE"].sum()) / alternative_total
    )
    if ice_share >= 0.80:
        warnings.append(
            f"High ICE prevalence ({ice_share:.1%} of vehicle alternatives) limits information for non-ICE effects."
        )
    for category in ("HEV", "PHEV", "BEV"):
        category_share = float(
            (data[f"PT1_{category}"].sum() + data[f"PT2_{category}"].sum())
            / alternative_total
        )
        if category_share < 0.025:
            warnings.append(
                f"Sparse alternative-level {category} support ({category_share:.2%}); interpret {model_name} uncertainty cautiously."
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
    rho_square = 1.0 - final_ll / null_ll
    adjusted_rho_square = 1.0 - (final_ll - parameter_count) / null_ll
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
        "rho_square_null": rho_square,
        "adjusted_rho_square_null": adjusted_rho_square,
        "AIC": float(results.akaike_information_criterion),
        "BIC": float(results.bayesian_information_criterion),
        "convergence_status": "converged" if results.algorithm_has_converged else "failed",
        "final_gradient_norm": float(results.gradient_norm),
        "smallest_information_eigenvalue": float(results.smallest_eigenvalue),
        "hessian_condition_number": float(results.condition_number),
        "max_abs_robust_parameter_correlation": max_correlation,
        "most_correlated_parameter_pair": correlation_pair,
        "optimizer_messages": json.dumps(
            raw.optimization_messages, sort_keys=True, default=str
        ),
        "warning_count": len(warnings),
        "warnings": " | ".join(warnings) if warnings else "none",
    }
    return pd.DataFrame([row]), warnings


def swap_vehicle_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Create the in-memory relabelled data used for the invariance test."""

    swapped = data.copy(deep=True)
    paired_columns = [
        (f"PT1_{category}", f"PT2_{category}") for category in POWERTRAIN_FULL
    ] + [
        (f"STATUS1_{category}", f"STATUS2_{category}") for category in STATUS_FULL
    ] + [("AV_1", "AV_2")]
    for car1_column, car2_column in paired_columns:
        car1_values = data[car1_column].copy()
        swapped[car1_column] = data[car2_column].to_numpy()
        swapped[car2_column] = car1_values.to_numpy()
    swapped["CHOICE"] = 3 - data["CHOICE"]
    validate_retained_sample(swapped)
    return swapped


def label_invariance_table(
    original_parameters: pd.DataFrame,
    swapped_results: Any,
) -> tuple[pd.DataFrame, bool]:
    """Compare original and relabelled unweighted generic coefficients."""

    rows = []
    all_passed = True
    originals = original_parameters.set_index("parameter")["estimate"]
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


def interpretation_text(parameter: str, estimate: float) -> str:
    """Return a sign-aware but deliberately mechanical interpretation."""

    direction = "higher" if estimate > 0 else "lower" if estimate < 0 else "unchanged"
    if parameter.startswith("B_PT_"):
        category = parameter.removeprefix("B_PT_")
        return (
            f"{parameter} ({estimate:.6g}): Relative to ICE, {category} is associated with "
            f"{direction} utility for household vehicle use, conditional on both vehicles being "
            "in the household choice set and holding vehicle status constant."
        )
    category = parameter.removeprefix("B_STATUS_").lower()
    return (
        f"{parameter} ({estimate:.6g}): Relative to medium-status vehicles, {category}-status "
        f"vehicles are associated with {direction} relative utility for household vehicle use, "
        "holding powertrain constant."
    )


def markdown_parameter_table(parameters: pd.DataFrame) -> str:
    """Format estimates compactly for the human-readable model summary."""

    display = parameters.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    return display.to_markdown(index=False)


def markdown_delta_x_table(delta_x_vif: pd.DataFrame) -> str:
    """Format the effective-design diagnostics for the Markdown summary."""

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
    display["non-zero occasions"] = display["non-zero occasions"].map(
        lambda value: f"{value:,}"
    )
    display["non-zero share"] = display["non-zero share"].map(
        lambda value: f"{value:.2%}"
    )
    display["maximum absolute correlation"] = display[
        "maximum absolute correlation"
    ].map(
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
) -> str:
    """Generate the concise thesis-facing methodological and result summary."""

    counts = sample_flow.set_index("metric")["count"]
    n = int(counts["final_v2_estimation_occasions"])
    diag = diagnostics.set_index(["section", "metric", "category"])

    def diagnostic_value(section: str, metric: str, category: str, column: str) -> float:
        return float(diag.loc[(section, metric, category), column])

    same_pt = diagnostic_value("B_choice_set_powertrain", "same_vs_different", "same", "count")
    different_pt = diagnostic_value("B_choice_set_powertrain", "same_vs_different", "different", "count")
    same_status = diagnostic_value("C_status_variation", "same_vs_different", "same", "count")
    different_status = diagnostic_value("C_status_variation", "same_vs_different", "different", "count")
    noninformative = diagnostic_value(
        "E_non_informative_v2_occasions",
        "same_powertrain_and_same_status",
        "non_informative_for_v2_coefficients",
        "count",
    )

    variation_lines = []
    for parameter in PARAMETERS:
        count = diagnostic_value(
            "D_parameter_specific_variation",
            "X_car1_not_equal_X_car2",
            parameter,
            "count",
        )
        weighted_count = diagnostic_value(
            "D_parameter_specific_variation",
            "X_car1_not_equal_X_car2",
            parameter,
            "weighted_count",
        )
        variation_lines.append(
            f"- {parameter}: {int(count):,} occasions ({count / n:.2%}); weighted count {weighted_count:,.3f}."
        )

    def warning_block(warnings: list[str]) -> str:
        return "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None."

    interpretations = "\n".join(
        f"- {interpretation_text(row.parameter, row.estimate)}"
        for row in unweighted_parameters.itertuples(index=False)
    )

    invariance_max_difference = float(invariance["absolute_difference"].max())
    return f"""# Stage-0 V2-only unlabelled MNL

## Estimation sample

- Source: `{SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve()}`.
- Source branch: single-active-driver subset; it is not the 30,816-observation strict-baseline sample.
- Initial occasions: {int(counts['initial_single_active_driver_occasions']):,}.
- Invalid/OTHER powertrain reason count: {int(counts['excluded_invalid_or_other_powertrain']):,}.
- Missing-status reason count: {int(counts['excluded_missing_status']):,}.
- Overlap between the two reason counts: {int(counts['excluded_for_both_reasons']):,}.
- Total unique excluded occasions: {int(counts['total_excluded_unique_occasions']):,}.
- Final V2 estimation sample: {n:,} choice occasions.
- Inclusion rule: both vehicles have powertrain ICE, HEV, PHEV, or BEV and status LOW, MEDIUM, or HIGH. No missing value was imputed.

Biogeme version: {get_biogeme_version()}.

## Specification and identification

This is an unlabelled two-alternative household vehicle-use model. Car 1 and Car 2 are arbitrary identifiers. The model has no alternative-specific constant and applies the same five generic coefficients to both vehicles.

Powertrain reference: ICE. Estimated effects: HEV, PHEV, BEV. Status reference: MEDIUM. Estimated effects: LOW, HIGH.

`Vj = B_PT_HEV*PTj_HEV + B_PT_PHEV*PTj_PHEV + B_PT_BEV*PTj_BEV + B_STATUS_LOW*STATUSj_LOW + B_STATUS_HIGH*STATUSj_HIGH`, for j in {{1, 2}}.

- Same powertrain: {int(same_pt):,} ({same_pt / n:.2%}); different powertrain: {int(different_pt):,} ({different_pt / n:.2%}).
- Same status: {int(same_status):,} ({same_status / n:.2%}); different status: {int(different_status):,} ({different_status / n:.2%}).
- Same powertrain and same status: {int(noninformative):,} ({noninformative / n:.2%}). These valid occasions remain in estimation, but V1=V2 under this specification and they do not identify the V2 coefficients.

Parameter-specific within-choice variation:

{chr(10).join(variation_lines)}

## Effective ΔX design and multicollinearity

Because this is an unlabelled two-alternative MNL, identification operates through `X_car1 - X_car2`. The correlation and VIF diagnostics below therefore assess the actual effective MNL design matrix. This differs from pooled vehicle-level association screening.

{markdown_delta_x_table(delta_x_vif)}

VIF is a pre-estimation specification diagnostic and is not derived from the estimated MNL coefficients. The flags are descriptive only and do not remove predictors.

## Unweighted results

{markdown_parameter_table(unweighted_parameters)}

Fit: null LL {unweighted_fit.at[0, 'null_log_likelihood']:.6f}; final LL {unweighted_fit.at[0, 'final_log_likelihood']:.6f}; rho-square {unweighted_fit.at[0, 'rho_square_null']:.6f}; adjusted rho-square {unweighted_fit.at[0, 'adjusted_rho_square_null']:.6f}; AIC {unweighted_fit.at[0, 'AIC']:.6f}; BIC {unweighted_fit.at[0, 'BIC']:.6f}; convergence {unweighted_fit.at[0, 'convergence_status']}.

## W_GEW-weighted results

{markdown_parameter_table(weighted_parameters)}

Fit: sum of weights {weighted_fit.at[0, 'sum_of_weights']:.6f}; weighted null LL {weighted_fit.at[0, 'null_log_likelihood']:.6f}; final weighted LL {weighted_fit.at[0, 'final_log_likelihood']:.6f}; rho-square {weighted_fit.at[0, 'rho_square_null']:.6f}; adjusted rho-square {weighted_fit.at[0, 'adjusted_rho_square_null']:.6f}; AIC {weighted_fit.at[0, 'AIC']:.6f}; BIC {weighted_fit.at[0, 'BIC']:.6f}; convergence {weighted_fit.at[0, 'convergence_status']}.

## Mechanical interpretation templates

{interpretations}

Coefficients are relative utility effects. They do not represent market shares or vehicle-purchase preferences. Powertrain effects are identified only where the two household vehicles differ in the relevant indicators. High ICE prevalence reduces effective information for non-ICE coefficients and can increase standard errors. Statistical insignificance can reflect limited within-choice variation and is not, by itself, evidence of a zero behavioural effect.

## Stability flags

Unweighted:

{warning_block(unweighted_warnings)}

Weighted:

{warning_block(weighted_warnings)}

The warning rules flag numerical or support concerns only. They never remove a variable or occasion.

## Label-invariance check

Result: **{'PASS' if invariance_passed else 'FAIL'}**. Maximum absolute coefficient difference after swapping all Car-1/Car-2 V2 attributes, availability, and CHOICE labels in memory: {invariance_max_difference:.3g}. Tolerances: absolute {LABEL_ATOL:g}, relative {LABEL_RTOL:g}. The source CSV was not changed.
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
    """Write only the compact result set requested for this benchmark."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        RESULTS_DIR / "v2_estimation_sample_flow.csv": sample_flow,
        RESULTS_DIR / "v2_identification_diagnostics.csv": diagnostics,
        RESULTS_DIR / "v2_delta_x_vif.csv": delta_x_vif,
        RESULTS_DIR / "v2_only_unweighted_parameters.csv": unweighted_parameters,
        RESULTS_DIR / "v2_only_unweighted_fit_summary.csv": unweighted_fit,
        RESULTS_DIR / "v2_only_weighted_parameters.csv": weighted_parameters,
        RESULTS_DIR / "v2_only_weighted_fit_summary.csv": weighted_fit,
        RESULTS_DIR / "v2_label_invariance_check.csv": invariance,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    correlation_path = RESULTS_DIR / "v2_delta_x_correlation.csv"
    delta_x_correlation.to_csv(correlation_path, index=True, index_label="predictor")
    summary_path = RESULTS_DIR / "v2_only_model_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    written = [*outputs, correlation_path, summary_path]
    for path in written:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected non-empty result file was not written: {path}")
    return written


def print_run_summary(
    sample_flow: pd.DataFrame,
    diagnostics: pd.DataFrame,
    delta_x_correlation: pd.DataFrame,
    delta_x_vif: pd.DataFrame,
    unweighted_parameters: pd.DataFrame,
    weighted_parameters: pd.DataFrame,
    unweighted_fit: pd.DataFrame,
    weighted_fit: pd.DataFrame,
    invariance_passed: bool,
    outputs: list[Path],
) -> None:
    """Print the sample, diagnostics, estimates, fit, and output locations."""

    print("\nSTAGE-0 V2-ONLY UNLABELLED MNL")
    print(f"Biogeme version: {get_biogeme_version()}")
    print(f"Input: {SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve()}")
    print("\nESTIMATION SAMPLE")
    print(sample_flow[["metric", "count", "share_of_initial"]].to_string(index=False))

    selected_diagnostics = diagnostics.loc[
        diagnostics["section"].isin(
            [
                "A_alternative_powertrain_composition",
                "B_choice_set_powertrain",
                "C_status_variation",
                "D_parameter_specific_variation",
                "E_non_informative_v2_occasions",
            ]
        )
        & (
            ~diagnostics["section"].eq("B_choice_set_powertrain")
            | diagnostics["metric"].eq("same_vs_different")
        )
    ]
    print("\nKEY IDENTIFICATION DIAGNOSTICS")
    print(
        selected_diagnostics[
            ["section", "metric", "category", "count", "share", "weighted_count", "weighted_share"]
        ].to_string(index=False)
    )
    print("\nEFFECTIVE DELTA-X CORRELATION")
    print(delta_x_correlation.to_string())
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
    retained, sample_flow = build_v2_sample(source)
    diagnostics = build_identification_diagnostics(retained)
    delta_x_correlation, delta_x_vif = build_delta_x_diagnostics(
        retained, diagnostics
    )

    unweighted_results = estimate_model(retained, "v2_only_unweighted", weighted=False)
    weighted_results = estimate_model(retained, "v2_only_weighted", weighted=True)
    unweighted_parameters = parameter_table(unweighted_results)
    weighted_parameters = parameter_table(weighted_results)
    unweighted_fit, unweighted_warnings = fit_summary(
        "v2_only_unweighted", False, unweighted_results, unweighted_parameters, retained
    )
    weighted_fit, weighted_warnings = fit_summary(
        "v2_only_weighted", True, weighted_results, weighted_parameters, retained
    )

    swapped = swap_vehicle_labels(retained)
    swapped_results = estimate_model(swapped, "v2_only_unweighted_label_swap", weighted=False)
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

    print_run_summary(
        sample_flow,
        diagnostics,
        delta_x_correlation,
        delta_x_vif,
        unweighted_parameters,
        weighted_parameters,
        unweighted_fit,
        weighted_fit,
        invariance_passed,
        outputs,
    )


if __name__ == "__main__":
    main()
