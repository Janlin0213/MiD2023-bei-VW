"""Theme 1: habitual car-use orientation in single- vs multi-car households.

Research questions, in analytical order:
1. How does weighted habitual car-use frequency differ by car ownership group?
2. Is multi-car household membership associated with a different ordinal position?
3. Does that association vary across spatial contexts?
4. Does that association vary across household types?

The analytical unit is the person. All estimates use the raw positive MiD person
weight ``P_GEW``. Inferential models are P_GEW-weighted proportional-odds
marginal logistic regressions implemented through threshold expansion and GEE,
with household-clustered robust covariance. This is not a complete complex-survey
design estimator and results are interpreted associationally, not causally.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm
import statsmodels
import statsmodels.api as sm

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import SELECTED_RAW_DIR, THEME1_BACKBONE_DIR, THEME1_OUTPUT_DIR

PERSON_PATH = SELECTED_RAW_DIR / "persons_selected_raw.csv"
HOUSEHOLD_BACKBONE_PATH = THEME1_BACKBONE_DIR / "theme1_household_backbone.csv"
OUTPUT_DIR = THEME1_OUTPUT_DIR / "02_habitual_car_use_orientation"

STATSMODELS_VERSION = "0.14.6"
METHOD = (
    "P_GEW-weighted proportional-odds marginal logistic regression implemented "
    "through threshold expansion and GEE, with household-clustered robust covariance"
)
WORKING_CORRELATION = "Independence"
COVARIANCE = "robust / sandwich"
ANALYTICAL_UNIT = "person"

PERSON_COLUMNS = ["HP_ID", "H_ID", "P_ID", "P_GEW", "P_NUTZ_AUTO"]
HOUSEHOLD_COLUMNS = [
    "H_ID",
    "H_ANZAUTO",
    "car_ownership_group",
    "RegioStaR7",
    "spatial_context_3",
    "household_type",
]

CAR_ORDER = ["single_car", "multi_car"]
CAR_LABELS = {"single_car": "Single-car", "multi_car": "Multi-car"}
SPATIAL_ORDER = ["urban_core", "intermediate_urban", "small_town_rural"]
SPATIAL_LABELS = {
    "urban_core": "Urban core",
    "intermediate_urban": "Intermediate urban",
    "small_town_rural": "Small-town / rural",
}
# The completed licensed-driver module uses the first element of this accepted
# order as its treatment reference; therefore family_household remains the
# household-type reference here for cross-module consistency.
HOUSEHOLD_TYPE_ORDER = [
    "family_household",
    "young_household",
    "adult_household",
    "senior_household",
]
HOUSEHOLD_TYPE_LABELS = {
    "family_household": "Family household",
    "young_household": "Young household",
    "adult_household": "Adult household",
    "senior_household": "Senior household",
}

OUTCOME_CODES = [1, 2, 3, 4, 5]
OUTCOME_LABELS = {
    1: "Daily / almost daily",
    2: "1–3 days/week",
    3: "1–3 days/month",
    4: "Less than monthly",
    5: "Never / almost never",
}
OUTCOME_COLORS = {
    1: "#2F5D7C",
    2: "#4C8C9B",
    3: "#E0B44C",
    4: "#D98245",
    5: "#9B6A88",
}
CAR_DOT_STYLES = {
    "single_car": {"color": "#2F5D7C", "marker": "o", "label": "Single-car household"},
    "multi_car": {"color": "#D98245", "marker": "s", "label": "Multi-car household"},
}
PROPORTION_DOT_FILENAMES = {
    "overall": "habitual_car_use_proportion_dots_overall",
    "household_type": "habitual_car_use_proportion_dots_household_type",
    "spatial": "habitual_car_use_proportion_dots_spatial",
}
PLOT_SOURCE_FILENAMES = {
    "overall": "01_weighted_distribution_overall.csv",
    "spatial": "02_weighted_distribution_spatial.csv",
    "household_type": "03_weighted_distribution_household_type.csv",
}
THRESHOLD_INTERPRETATIONS = {
    1: "At least less-than-monthly rather than never",
    2: "At least monthly",
    3: "At least weekly",
    4: "Daily / almost daily",
}
THRESHOLD_TERMS = [f"C(threshold)[{k}]" for k in range(1, 5)]
MULTI_TERM = "multi_car_dummy"
Z_95 = float(norm.ppf(0.975))
SMALL_CELL_THRESHOLD = 100


@dataclass(frozen=True)
class CompactGEEResult:
    model_name: str
    terms: list[str]
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    n_persons: int
    n_households: int
    person_weight_sum: float
    expanded_rows: int
    converged: bool
    iterations: int | None
    weights_verified: bool


def require_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise KeyError(f"{source} is missing required column(s): {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_weight_mask(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.notna() & np.isfinite(values) & values.gt(0)


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PERSON_PATH.exists():
        raise FileNotFoundError(f"Accepted person-level source not found: {PERSON_PATH}")
    if not HOUSEHOLD_BACKBONE_PATH.exists():
        raise FileNotFoundError(
            f"Accepted Theme 1 household backbone not found: {HOUSEHOLD_BACKBONE_PATH}. "
            "Run 01_build_theme1_household_backbone.py first."
        )

    person_header = pd.read_csv(PERSON_PATH, nrows=0).columns.tolist()
    household_header = pd.read_csv(HOUSEHOLD_BACKBONE_PATH, nrows=0).columns.tolist()
    require_columns(person_header, PERSON_COLUMNS, "Person-level processed dataset")
    require_columns(household_header, HOUSEHOLD_COLUMNS, "Theme 1 household backbone")

    persons = pd.read_csv(
        PERSON_PATH,
        usecols=PERSON_COLUMNS,
        dtype={"HP_ID": "string", "H_ID": "string", "P_ID": "string"},
        low_memory=False,
    )
    households = pd.read_csv(
        HOUSEHOLD_BACKBONE_PATH,
        usecols=HOUSEHOLD_COLUMNS,
        dtype={"H_ID": "string"},
        low_memory=False,
    )

    duplicate_person = persons["HP_ID"].duplicated(keep=False) | persons["HP_ID"].isna()
    if duplicate_person.any():
        sample = persons.loc[duplicate_person, PERSON_COLUMNS].head(10)
        raise RuntimeError(
            "Accepted person source is not one row per nonmissing canonical HP_ID. "
            f"Diagnostic sample:\n{sample.to_string(index=False)}"
        )
    if households["H_ID"].isna().any() or households["H_ID"].duplicated().any():
        bad = households["H_ID"].isna() | households["H_ID"].duplicated(keep=False)
        sample = households.loc[bad].head(10)
        raise RuntimeError(
            "Theme 1 household backbone is not one row per nonmissing H_ID. "
            f"Diagnostic sample:\n{sample.to_string(index=False)}"
        )

    for column in ["P_GEW", "P_NUTZ_AUTO"]:
        persons[f"{column}_source"] = persons[column]
        persons[column] = numeric(persons[column])
    for column in ["H_ANZAUTO", "RegioStaR7"]:
        households[column] = numeric(households[column])
    return persons, households


def merge_person_household(persons: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    merged = persons.merge(
        households,
        on="H_ID",
        how="left",
        validate="many_to_one",
        indicator="person_household_merge",
    )
    if len(merged) != len(persons):
        raise AssertionError("Person-household merge changed the number of person rows.")
    if merged["HP_ID"].duplicated().any():
        raise AssertionError("Person-household merge duplicated canonical person identifiers.")
    merged["multi_car_dummy"] = merged["car_ownership_group"].map(
        {"single_car": 0.0, "multi_car": 1.0}
    )
    merged["car_use_intensity"] = 6 - merged["P_NUTZ_AUTO"]
    return merged


def observed_outcome_distribution(persons: pd.DataFrame) -> pd.DataFrame:
    source = persons["P_NUTZ_AUTO"]
    counts = source.value_counts(dropna=False).to_dict()
    expected = [1, 2, 3, 4, 5, 9, 206, 402]
    rows: list[dict[str, Any]] = []
    for value in expected:
        rows.append(
            {
                "observed_value": value,
                "value_class": "expected substantive" if value in OUTCOME_CODES else "expected non-substantive",
                "count": int(counts.get(value, 0)),
                "share_of_source_person_rows": float(counts.get(value, 0) / len(persons)),
            }
        )
    missing_count = int(source.isna().sum())
    rows.append(
        {
            "observed_value": "<missing>",
            "value_class": "missing",
            "count": missing_count,
            "share_of_source_person_rows": missing_count / len(persons),
        }
    )
    unexpected = source.notna() & ~source.isin(expected)
    unexpected_counts = source.loc[unexpected].value_counts().sort_index()
    if unexpected_counts.empty:
        rows.append(
            {
                "observed_value": "<other unexpected>",
                "value_class": "other unexpected",
                "count": 0,
                "share_of_source_person_rows": 0.0,
            }
        )
    else:
        for value, count in unexpected_counts.items():
            rows.append(
                {
                    "observed_value": value,
                    "value_class": "other unexpected",
                    "count": int(count),
                    "share_of_source_person_rows": float(count / len(persons)),
                }
            )
    output = pd.DataFrame(rows)
    if int(output["count"].sum()) != len(persons):
        raise AssertionError("Saved P_NUTZ_AUTO distribution does not exhaust the person source.")
    return output


def make_samples(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall_mask = (
        merged["HP_ID"].notna()
        & merged["H_ID"].notna()
        & merged["person_household_merge"].eq("both")
        & valid_weight_mask(merged["P_GEW"])
        & merged["P_NUTZ_AUTO"].isin(OUTCOME_CODES)
        & merged["car_ownership_group"].isin(CAR_ORDER)
    )
    overall = merged.loc[overall_mask].copy()
    spatial = overall.loc[overall["spatial_context_3"].isin(SPATIAL_ORDER)].copy()
    household_type = overall.loc[
        overall["household_type"].isin(HOUSEHOLD_TYPE_ORDER)
    ].copy()
    samples = {"overall": overall, "spatial": spatial, "household_type": household_type}

    for name, sample in samples.items():
        if sample.empty:
            raise RuntimeError(f"{name} analytical sample is empty.")
        if sample["HP_ID"].duplicated().any():
            raise AssertionError(f"{name} sample contains duplicate HP_ID values.")
        assert sample["P_NUTZ_AUTO"].isin(OUTCOME_CODES).all()
        assert sample["car_use_intensity"].isin(OUTCOME_CODES).all()
        assert valid_weight_mask(sample["P_GEW"]).all()
        assert sample["car_ownership_group"].isin(CAR_ORDER).all()
        assert sample["multi_car_dummy"].isin([0.0, 1.0]).all()

    if set(spatial["spatial_context_3"].unique()) != set(SPATIAL_ORDER):
        raise RuntimeError("Not all expected spatial contexts occur in the spatial sample.")
    if set(household_type["household_type"].unique()) != set(HOUSEHOLD_TYPE_ORDER):
        raise RuntimeError("Not all expected household types occur in the household-type sample.")
    return samples


def weighted_distribution(
    sample: pd.DataFrame,
    group_columns: list[str],
    orders: dict[str, list[str]],
    analysis_scope: str,
) -> pd.DataFrame:
    index_levels = [orders[column] for column in group_columns] + [OUTCOME_CODES]
    complete_index = pd.MultiIndex.from_product(index_levels, names=[*group_columns, "P_NUTZ_AUTO"])
    grouped = (
        sample.groupby([*group_columns, "P_NUTZ_AUTO"], observed=True, sort=False)
        .agg(unweighted_n=("HP_ID", "size"), weighted_sum=("P_GEW", "sum"))
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    cell = (
        sample.groupby(group_columns, observed=True, sort=False)
        .agg(cell_unweighted_n=("HP_ID", "size"), cell_weight_sum=("P_GEW", "sum"))
        .reset_index()
    )
    output = grouped.merge(cell, on=group_columns, how="left", validate="many_to_one")
    if output["cell_weight_sum"].isna().any() or (output["cell_weight_sum"] <= 0).any():
        raise RuntimeError(f"Missing or nonpositive descriptive comparison cell in {analysis_scope}.")
    output["weighted_share"] = output["weighted_sum"] / output["cell_weight_sum"]
    output["category_label"] = output["P_NUTZ_AUTO"].map(OUTCOME_LABELS)
    output.insert(0, "analysis_scope", analysis_scope)
    shares = output.groupby(group_columns, observed=True)["weighted_share"].sum()
    if not np.allclose(shares.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"Weighted shares do not sum to one in {analysis_scope}:\n{shares}")
    return output[
        [
            "analysis_scope",
            *group_columns,
            "P_NUTZ_AUTO",
            "category_label",
            "unweighted_n",
            "weighted_sum",
            "weighted_share",
            "cell_unweighted_n",
            "cell_weight_sum",
        ]
    ]


def publication_table(
    distribution: pd.DataFrame,
    context: str | None,
    context_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    index_columns = ["car_ownership_group"] if context is None else [context, "car_ownership_group"]
    pivot = distribution.pivot(index=index_columns, columns="category_label", values="weighted_share")
    pivot = pivot[[OUTCOME_LABELS[code] for code in OUTCOME_CODES]] * 100.0
    cell_n = distribution.groupby(index_columns, observed=True)["cell_unweighted_n"].first()
    pivot["Unweighted N"] = cell_n
    pivot = pivot.reset_index()
    pivot["_car_order"] = pivot["car_ownership_group"].map(
        {value: rank for rank, value in enumerate(CAR_ORDER)}
    )
    sort_columns = ["_car_order"]
    if context is not None:
        context_order = distribution[context].drop_duplicates().tolist()
        pivot["_context_order"] = pivot[context].map(
            {value: rank for rank, value in enumerate(context_order)}
        )
        sort_columns = ["_context_order", "_car_order"]
    pivot = pivot.sort_values(sort_columns).drop(columns=sort_columns).reset_index(drop=True)
    pivot["car_ownership_group"] = pivot["car_ownership_group"].map(CAR_LABELS)
    if context is not None and context_labels is not None:
        pivot[context] = pivot[context].map(context_labels)
    return pivot.rename(columns={context: "Context"} if context is not None else {})


def build_substantive_design(
    sample: pd.DataFrame,
    context: str | None = None,
    order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    multi = sample["multi_car_dummy"].to_numpy(dtype=float)
    columns = [multi]
    terms = [MULTI_TERM]
    if context is not None:
        if order is None or len(order) < 2:
            raise ValueError("Context models require an explicit deterministic order.")
        observed = set(sample[context].dropna().unique())
        if observed != set(order):
            raise RuntimeError(f"Unexpected or missing {context} categories: {sorted(observed)}")
        cells = pd.crosstab(sample[context], sample["car_ownership_group"])
        missing_cells = [
            (level, car)
            for level in order
            for car in CAR_ORDER
            if level not in cells.index or car not in cells.columns or cells.loc[level, car] == 0
        ]
        if missing_cells:
            raise RuntimeError(f"Empty {context} × car-ownership cells: {missing_cells}")
        for level in order[1:]:
            dummy = sample[context].eq(level).to_numpy(dtype=float)
            columns.append(dummy)
            terms.append(f"C({context})[T.{level}]")
        for level in order[1:]:
            dummy = sample[context].eq(level).to_numpy(dtype=float)
            columns.append(multi * dummy)
            terms.append(f"{MULTI_TERM}:C({context})[T.{level}]")
    x = np.column_stack(columns)
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x) != x.shape[1]:
        raise RuntimeError(f"Substantive design is non-finite or rank deficient: {terms}")
    return x, terms


def household_cluster_codes(sample: pd.DataFrame) -> np.ndarray:
    codes, levels = pd.factorize(sample["H_ID"], sort=False)
    if (codes < 0).any() or len(levels) != sample["H_ID"].nunique():
        raise AssertionError("Could not create a one-to-one household cluster code mapping.")
    return codes.astype(np.int64, copy=False)


def fit_ordinal_gee(
    sample: pd.DataFrame,
    model_name: str,
    context: str | None = None,
    order: list[str] | None = None,
) -> CompactGEEResult:
    substantive_x, substantive_terms = build_substantive_design(sample, context, order)
    n_persons = len(sample)
    person_index = np.repeat(np.arange(n_persons, dtype=np.int64), 4)
    threshold = np.tile(np.arange(1, 5, dtype=np.int8), n_persons)
    intensity = sample["car_use_intensity"].to_numpy(dtype=np.int8)
    endog = (intensity[person_index] > threshold).astype(float)
    threshold_x = np.eye(4, dtype=float)[threshold - 1]
    exog = np.column_stack([threshold_x, substantive_x[person_index]])
    terms = [*THRESHOLD_TERMS, *substantive_terms]
    weights = np.repeat(sample["P_GEW"].to_numpy(dtype=float), 4)
    clusters = np.repeat(household_cluster_codes(sample), 4)

    expected_rows = 4 * n_persons
    assert len(endog) == expected_rows
    assert np.all(np.bincount(person_index, minlength=n_persons) == 4)
    assert set(np.unique(threshold)) == {1, 2, 3, 4}
    assert set(np.unique(endog)).issubset({0.0, 1.0})
    assert exog.shape == (expected_rows, len(terms))
    assert np.isfinite(exog).all() and np.isfinite(weights).all() and (weights > 0).all()

    model = sm.GEE(
        endog=endog,
        exog=exog,
        groups=clusters,
        family=sm.families.Binomial(),
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
        raise AssertionError(f"Generic GEE did not retain raw repeated P_GEW in {model_name}.")

    fitted = model.fit(cov_type="robust", maxiter=100)
    beta = np.asarray(fitted.params, dtype=float)
    se = np.asarray(fitted.bse, dtype=float)
    covariance = np.asarray(fitted.cov_params(), dtype=float)
    converged = bool(getattr(fitted, "converged", False))
    fit_history = getattr(fitted, "fit_history", {}) or {}
    iterations = len(fit_history.get("params", [])) or None
    if not converged:
        raise RuntimeError(f"{model_name} GEE did not converge after {iterations} iterations.")
    if not (np.isfinite(beta).all() and np.isfinite(se).all() and np.isfinite(covariance).all()):
        raise RuntimeError(f"{model_name} produced non-finite parameters, SEs, or covariance.")
    if covariance.shape != (len(terms), len(terms)) or np.any(np.diag(covariance) < -1e-10):
        raise RuntimeError(f"{model_name} produced an invalid robust covariance matrix.")

    compact = CompactGEEResult(
        model_name=model_name,
        terms=terms,
        beta=beta.copy(),
        covariance=covariance.copy(),
        standard_error=se.copy(),
        n_persons=n_persons,
        n_households=int(sample["H_ID"].nunique()),
        person_weight_sum=float(sample["P_GEW"].sum()),
        expanded_rows=expected_rows,
        converged=converged,
        iterations=iterations,
        weights_verified=weights_verified,
    )
    del fitted, model, exog, endog, weights, clusters, threshold_x, person_index
    gc.collect()
    return compact


def coefficient_table(result: CompactGEEResult) -> pd.DataFrame:
    statistic = np.divide(
        result.beta,
        result.standard_error,
        out=np.full_like(result.beta, np.nan),
        where=result.standard_error > 0,
    )
    low = result.beta - Z_95 * result.standard_error
    high = result.beta + Z_95 * result.standard_error
    parameter_type = ["threshold" if term in THRESHOLD_TERMS else "predictor" for term in result.terms]
    is_predictor = np.array([kind == "predictor" for kind in parameter_type])
    odds_ratio = np.where(is_predictor, np.exp(result.beta), np.nan)
    or_low = np.where(is_predictor, np.exp(low), np.nan)
    or_high = np.where(is_predictor, np.exp(high), np.nan)
    return pd.DataFrame(
        {
            "model": result.model_name,
            "term": result.terms,
            "parameter_type": parameter_type,
            "estimate": result.beta,
            "robust_se": result.standard_error,
            "statistic": statistic,
            "p_value": 2 * norm.sf(np.abs(statistic)),
            "ci95_low": low,
            "ci95_high": high,
            "odds_ratio": odds_ratio,
            "or_ci95_low": or_low,
            "or_ci95_high": or_high,
            "n_persons": result.n_persons,
            "n_households": result.n_households,
            "person_weight_sum": result.person_weight_sum,
            "expanded_model_rows": result.expanded_rows,
            "method": METHOD,
            "working_correlation": WORKING_CORRELATION,
            "covariance": COVARIANCE,
            "converged": result.converged,
            "iterations": result.iterations,
            "raw_P_GEW_verified": result.weights_verified,
        }
    )


def joint_wald(
    result: CompactGEEResult,
    context: str,
    order: list[str],
) -> dict[str, Any]:
    interaction_terms = [f"{MULTI_TERM}:C({context})[T.{level}]" for level in order[1:]]
    indices = [result.terms.index(term) for term in interaction_terms]
    beta = result.beta[indices]
    covariance = result.covariance[np.ix_(indices, indices)]
    rank = int(np.linalg.matrix_rank(covariance))
    if rank != len(indices):
        raise RuntimeError(f"Interaction covariance is rank deficient in {result.model_name}.")
    statistic = float(beta.T @ np.linalg.solve(covariance, beta))
    return {
        "test_name": f"Joint Multi × {context} interaction Wald test",
        "context_variable": context,
        "wald_statistic": statistic,
        "degrees_of_freedom": len(indices),
        "p_value": float(chi2.sf(statistic, len(indices))),
        "n_persons": result.n_persons,
        "n_households": result.n_households,
        "person_weight_sum": result.person_weight_sum,
        "method": "Joint Wald chi-square test using the GEE robust covariance matrix",
    }


def substantive_vector(
    result: CompactGEEResult,
    car_group: str,
    context: str | None = None,
    level: str | None = None,
) -> np.ndarray:
    terms = result.terms[4:]
    values = np.zeros(len(terms), dtype=float)
    multi = car_group == "multi_car"
    values[terms.index(MULTI_TERM)] = float(multi)
    if context is not None and level is not None:
        main = f"C({context})[T.{level}]"
        interaction = f"{MULTI_TERM}:C({context})[T.{level}]"
        if main in terms:
            values[terms.index(main)] = 1.0
        if multi and interaction in terms:
            values[terms.index(interaction)] = 1.0
    return values


def context_contrasts(
    result: CompactGEEResult,
    sample: pd.DataFrame,
    context: str,
    order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    beta = result.beta[4:]
    covariance = result.covariance[4:, 4:]
    for level in order:
        single = substantive_vector(result, "single_car", context, level)
        multi = substantive_vector(result, "multi_car", context, level)
        contrast = multi - single
        estimate = float(contrast @ beta)
        variance = float(contrast @ covariance @ contrast)
        if variance < -1e-10:
            raise RuntimeError(f"Negative contrast variance in {result.model_name}, {level}.")
        se = float(np.sqrt(max(variance, 0.0)))
        low, high = estimate - Z_95 * se, estimate + Z_95 * se
        cell = sample.loc[sample[context].eq(level)]
        rows.append(
            {
                context: level,
                "estimate_multi_minus_single": estimate,
                "robust_se": se,
                "ci95_low": low,
                "ci95_high": high,
                "odds_ratio": float(np.exp(estimate)),
                "or_ci95_low": float(np.exp(low)),
                "or_ci95_high": float(np.exp(high)),
                "p_value": float(2 * norm.sf(abs(estimate / se))) if se > 0 else np.nan,
                "n_persons_single": int(cell["car_ownership_group"].eq("single_car").sum()),
                "n_persons_multi": int(cell["car_ownership_group"].eq("multi_car").sum()),
                "n_households": int(cell["H_ID"].nunique()),
                "method": METHOD,
            }
        )
    return pd.DataFrame(rows)


def fit_binary_diagnostic(sample: pd.DataFrame, threshold: int) -> dict[str, Any]:
    endog = (sample["car_use_intensity"].to_numpy(dtype=np.int8) > threshold).astype(float)
    exog = np.column_stack(
        [np.ones(len(sample), dtype=float), sample["multi_car_dummy"].to_numpy(dtype=float)]
    )
    weights = sample["P_GEW"].to_numpy(dtype=float)
    clusters = household_cluster_codes(sample)
    model = sm.GEE(
        endog=endog,
        exog=exog,
        groups=clusters,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
        weights=weights,
    )
    if not np.array_equal(np.asarray(model.weights, dtype=float), weights):
        raise AssertionError(f"Diagnostic threshold {threshold} did not retain raw P_GEW.")
    fitted = model.fit(cov_type="robust", maxiter=100)
    if not bool(getattr(fitted, "converged", False)):
        raise RuntimeError(f"Diagnostic threshold {threshold} GEE failed to converge.")
    beta = float(fitted.params[1])
    se = float(fitted.bse[1])
    if not np.isfinite([beta, se, *np.asarray(fitted.cov_params()).ravel()]).all():
        raise RuntimeError(f"Diagnostic threshold {threshold} produced non-finite output.")
    low, high = beta - Z_95 * se, beta + Z_95 * se
    row = {
        "threshold": threshold,
        "threshold_interpretation": THRESHOLD_INTERPRETATIONS[threshold],
        "estimate": beta,
        "robust_se": se,
        "ci95_low": low,
        "ci95_high": high,
        "odds_ratio": float(np.exp(beta)),
        "or_ci95_low": float(np.exp(low)),
        "or_ci95_high": float(np.exp(high)),
        "p_value": float(2 * norm.sf(abs(beta / se))),
        "n_persons": len(sample),
        "n_households": int(sample["H_ID"].nunique()),
        "person_weight_sum": float(sample["P_GEW"].sum()),
        "method": "P_GEW-weighted binary GEE with household-clustered robust covariance",
        "raw_P_GEW_verified": True,
    }
    del fitted, model
    gc.collect()
    return row


def predicted_probabilities(
    result: CompactGEEResult,
    context: str | None = None,
    order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    scenarios: list[tuple[str | None, str]] = []
    if context is None:
        scenarios = [(None, car) for car in CAR_ORDER]
    else:
        if order is None:
            raise ValueError("Predictions for a context model require an explicit order.")
        scenarios = [(level, car) for level in order for car in CAR_ORDER]

    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    alpha = result.beta[:4]
    beta = result.beta[4:]
    for level, car in scenarios:
        x = substantive_vector(result, car, context, level)
        q = expit(alpha + x @ beta)
        label = car if level is None else f"{level} × {car}"
        monotonic = bool(np.all(q[:-1] >= q[1:] - 1e-10))
        probabilities = np.array([1 - q[0], q[0] - q[1], q[1] - q[2], q[2] - q[3], q[3]])
        valid = (
            monotonic
            and np.isfinite(probabilities).all()
            and bool(np.all(probabilities >= -1e-10))
            and bool(np.all(probabilities <= 1 + 1e-10))
            and bool(np.isclose(probabilities.sum(), 1.0, rtol=1e-10, atol=1e-10))
        )
        if not valid:
            invalid.append(label)
            continue
        # Output follows the natural thesis display order (P_NUTZ_AUTO 1..5),
        # while probabilities were recovered on intensity order Y=1..5.
        for code in OUTCOME_CODES:
            intensity = 6 - code
            probability = float(probabilities[intensity - 1])
            row = {
                "car_ownership_group": car,
                "P_NUTZ_AUTO": code,
                "category_label": OUTCOME_LABELS[code],
                "car_use_intensity": intensity,
                "predicted_probability": probability,
                "weekly_or_more_probability": float(q[2]),
                "cumulative_probabilities_monotonic": monotonic,
                "method": METHOD,
            }
            if context is not None:
                row[context] = level
            rows.append(row)
    output = pd.DataFrame(rows)
    if invalid:
        return output, invalid
    group_columns = ["car_ownership_group"] if context is None else [context, "car_ownership_group"]
    sums = output.groupby(group_columns, observed=True)["predicted_probability"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
        raise AssertionError("Predicted category probabilities do not sum to one.")
    if not output["predicted_probability"].between(-1e-10, 1 + 1e-10).all():
        raise AssertionError("Predicted category probabilities fall outside [0, 1].")
    return output, []


def plot_distribution(
    distribution: pd.DataFrame,
    path_stem: Path,
    title: str,
    context: str | None = None,
    order: list[str] | None = None,
    context_labels: dict[str, str] | None = None,
) -> None:
    if context is None:
        cells = [(None, car) for car in CAR_ORDER]
        positions = np.array([0.0, 1.0])
        tick_labels = [CAR_LABELS[car] for car in CAR_ORDER]
        figsize = (7.2, 5.2)
    else:
        if order is None or context_labels is None:
            raise ValueError("Context figures require deterministic order and labels.")
        cells = [(level, car) for level in order for car in CAR_ORDER]
        positions = np.array([3 * i + j for i in range(len(order)) for j in range(2)], dtype=float)
        tick_labels = [CAR_LABELS[car].replace("-car", "") for _, car in cells]
        figsize = (10.8 if len(order) == 3 else 12.2, 5.8)

    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(cells), dtype=float)
    for code in OUTCOME_CODES:
        heights: list[float] = []
        for level, car in cells:
            mask = distribution["car_ownership_group"].eq(car) & distribution["P_NUTZ_AUTO"].eq(code)
            if context is not None:
                mask &= distribution[context].eq(level)
            row = distribution.loc[mask]
            if len(row) != 1:
                raise AssertionError(f"Expected one distribution row for {level}, {car}, {code}.")
            heights.append(float(row.iloc[0]["weighted_share"]))
        values = np.asarray(heights)
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.78,
            color=OUTCOME_COLORS[code],
            edgecolor="white",
            linewidth=0.55,
            label=OUTCOME_LABELS[code],
        )
        bottoms += values

    for x, (level, car) in zip(positions, cells):
        mask = distribution["car_ownership_group"].eq(car)
        if context is not None:
            mask &= distribution[context].eq(level)
        n = int(distribution.loc[mask, "cell_unweighted_n"].iloc[0])
        ax.text(x, 1.014, f"N={n:,}", ha="center", va="bottom", fontsize=7.5, color="#404040")

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted share of persons")
    ax.set_ylim(0, 1.075)
    ax.set_xticks(positions, tick_labels)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if context is not None and order is not None and context_labels is not None:
        for i, level in enumerate(order):
            center = 3 * i + 0.5
            ax.text(
                center,
                -0.105,
                context_labels[level],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.5,
                fontweight="bold",
            )
        bottom_margin = 0.29
    else:
        bottom_margin = 0.24
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18 if context is None else -0.22),
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=bottom_margin)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)


def validate_proportion_dot_distribution(
    distribution: pd.DataFrame,
    *,
    context: str | None = None,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Validate a saved weighted distribution before descriptive plotting."""
    required = {"car_ownership_group", "P_NUTZ_AUTO", "category_label", "weighted_share"}
    if context is not None:
        required.add(context)
    missing = sorted(required.difference(distribution.columns))
    if missing:
        raise KeyError(f"Proportion-dot plotting input is missing column(s): {missing}")

    data = distribution.copy()
    data["P_NUTZ_AUTO"] = numeric(data["P_NUTZ_AUTO"])
    data["weighted_share"] = numeric(data["weighted_share"])
    if data[["P_NUTZ_AUTO", "weighted_share"]].isna().any().any():
        raise RuntimeError("Proportion-dot plotting input contains non-numeric categories or shares.")
    if not np.isfinite(data["weighted_share"]).all() or not data["weighted_share"].between(0, 1).all():
        raise RuntimeError("Weighted shares for proportion-dot plotting must be finite and within [0, 1].")
    if set(data["P_NUTZ_AUTO"].astype(int).unique()) != set(OUTCOME_CODES):
        raise RuntimeError("Proportion-dot plotting input does not contain exactly the five accepted categories.")
    if set(data["car_ownership_group"].dropna().unique()) != set(CAR_ORDER):
        raise RuntimeError("Proportion-dot plotting input does not contain both accepted car groups.")

    data["P_NUTZ_AUTO"] = data["P_NUTZ_AUTO"].astype(int)
    expected_labels = data["P_NUTZ_AUTO"].map(OUTCOME_LABELS)
    if not data["category_label"].eq(expected_labels).all():
        raise RuntimeError("Saved outcome labels do not match the accepted thesis category labels.")

    key_columns = ["car_ownership_group", "P_NUTZ_AUTO"]
    context_levels: list[str | None] = [None]
    if context is not None:
        if order is None:
            raise ValueError("Faceted proportion-dot plots require an explicit deterministic facet order.")
        observed = set(data[context].dropna().unique())
        if observed != set(order):
            raise RuntimeError(f"Unexpected or missing {context} facets: {sorted(observed)}")
        key_columns = [context, *key_columns]
        context_levels = list(order)
    if data.duplicated(key_columns).any():
        raise RuntimeError(f"Duplicate plotting rows found for keys {key_columns}.")

    for level in context_levels:
        panel = data if context is None else data.loc[data[context].eq(level)]
        for code in OUTCOME_CODES:
            for car_group in CAR_ORDER:
                rows = panel.loc[
                    panel["P_NUTZ_AUTO"].eq(code)
                    & panel["car_ownership_group"].eq(car_group)
                ]
                if len(rows) != 1:
                    raise RuntimeError(
                        f"Expected one plotting row for {context}={level}, "
                        f"category={code}, car_group={car_group}; found {len(rows)}."
                    )
    return data


def dot_legend_handles(labels: dict[str, str] | None = None) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=CAR_DOT_STYLES[car_group]["marker"],
            linestyle="None",
            markerfacecolor=CAR_DOT_STYLES[car_group]["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=(labels or {}).get(car_group, CAR_DOT_STYLES[car_group]["label"]),
        )
        for car_group in CAR_ORDER
    ]


def draw_proportion_dot_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    annotate: bool = False,
    show_y_labels: bool = True,
) -> None:
    """Draw one five-category Single-vs-Multi weighted proportion panel."""
    y_positions = np.arange(len(OUTCOME_CODES), dtype=float)
    offsets = {"single_car": -0.09, "multi_car": 0.09}
    values_by_group: dict[str, np.ndarray] = {}
    for car_group in CAR_ORDER:
        values = []
        for code in OUTCOME_CODES:
            row = panel.loc[
                panel["P_NUTZ_AUTO"].eq(code)
                & panel["car_ownership_group"].eq(car_group),
                "weighted_share",
            ]
            if len(row) != 1:
                raise AssertionError(f"Expected one dot value for {car_group}, category {code}.")
            values.append(float(row.iloc[0]) * 100.0)
        values_by_group[car_group] = np.asarray(values)

    for index, y_value in enumerate(y_positions):
        single_x = values_by_group["single_car"][index]
        multi_x = values_by_group["multi_car"][index]
        ax.plot(
            [single_x, multi_x],
            [y_value + offsets["single_car"], y_value + offsets["multi_car"]],
            color="#B8B8B8",
            linewidth=1.25,
            solid_capstyle="round",
            zorder=1,
        )

    for car_group in CAR_ORDER:
        style = CAR_DOT_STYLES[car_group]
        y_values = y_positions + offsets[car_group]
        ax.scatter(
            values_by_group[car_group],
            y_values,
            s=52,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        if annotate:
            horizontal_offset = -1.4 if car_group == "single_car" else 1.4
            alignment = "right" if car_group == "single_car" else "left"
            for x_value, y_value in zip(values_by_group[car_group], y_values):
                ax.text(
                    x_value + horizontal_offset,
                    y_value,
                    f"{x_value:.1f}%",
                    ha=alignment,
                    va="center",
                    fontsize=8.2,
                    color=style["color"],
                )

    ax.set_xlim(0, 100)
    ax.set_ylim(len(OUTCOME_CODES) - 0.45, -0.45)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels([OUTCOME_LABELS[code] for code in OUTCOME_CODES])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#404040")


def save_figure_png(fig: plt.Figure, path_stem: Path) -> list[Path]:
    png_path = path_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [png_path]


def plot_habitual_overall_proportion_dots(distribution: pd.DataFrame) -> list[Path]:
    data = validate_proportion_dot_distribution(distribution)
    if "cell_unweighted_n" not in data.columns:
        raise KeyError("Overall plotting input is missing cell_unweighted_n for heading sample sizes.")
    group_n = data.groupby("car_ownership_group", observed=True)["cell_unweighted_n"].agg(
        ["nunique", "first"]
    )
    if not group_n["nunique"].eq(1).all():
        raise RuntimeError("Overall cell_unweighted_n is inconsistent within a car-ownership group.")
    group_counts = {}
    for car_group in CAR_ORDER:
        value = numeric(pd.Series([group_n.loc[car_group, "first"]])).iloc[0]
        if not np.isfinite(value) or value <= 0 or not float(value).is_integer():
            raise RuntimeError(f"Invalid overall cell_unweighted_n for {car_group}: {value}")
        group_counts[car_group] = int(value)

    y_positions = np.arange(len(OUTCOME_CODES))
    shares = {
        car_group: np.array([
            float(data.loc[
                data["car_ownership_group"].eq(car_group)
                & data["P_NUTZ_AUTO"].eq(code),
                "weighted_share",
            ].iloc[0]) * 100.0
            for code in OUTCOME_CODES
        ])
        for car_group in CAR_ORDER
    }
    axis_limit = int(np.ceil(max(values.max() for values in shares.values()) / 20.0) * 20)
    axis_limit = max(axis_limit, 20)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(y_positions, -shares["single_car"], height=0.56,
            color=CAR_DOT_STYLES["single_car"]["color"], zorder=2)
    ax.barh(y_positions, shares["multi_car"], height=0.56,
            color=CAR_DOT_STYLES["multi_car"]["color"], zorder=2)

    for car_group in CAR_ORDER:
        left_side = car_group == "single_car"
        for y_value, value in zip(y_positions, shares[car_group]):
            inside = value >= 15.0
            x_value = (-value + 1.7 if inside else -value - 1.2) if left_side else (
                value - 1.7 if inside else value + 1.2
            )
            alignment = ("left" if inside else "right") if left_side else (
                "right" if inside else "left"
            )
            ax.text(x_value, y_value, f"{value:.1f}%", ha=alignment, va="center",
                    fontsize=9.2, fontweight="bold",
                    color="white" if inside else CAR_DOT_STYLES[car_group]["color"],
                    zorder=4)

    ax.set_title(
        "Habitual car-use frequency by household car ownership",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=54,
    )
    for car_group, side in [("single_car", -1), ("multi_car", 1)]:
        ax.text(side * axis_limit / 2, 1.055,
                f"{CAR_LABELS[car_group]} households (n = {group_counts[car_group]:,})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold",
                color=CAR_DOT_STYLES[car_group]["color"])
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(len(OUTCOME_CODES) - 0.5, -0.5)
    ax.set_yticks(y_positions, [OUTCOME_LABELS[code] for code in OUTCOME_CODES])
    ax.set_xticks(np.arange(-axis_limit, axis_limit + 1, 20))
    ax.xaxis.set_major_formatter(lambda value, _: f"{abs(value):.0f}%")
    ax.set_xlabel("Weighted share of persons (%)", labelpad=8)
    ax.grid(axis="x", color="#DFE3E6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#69747C", linewidth=1.1, zorder=3)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", length=0, colors="#404040", labelsize=9)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.77, bottom=0.13)
    return save_figure_png(fig, OUTPUT_DIR / PROPORTION_DOT_FILENAMES["overall"])


def plot_habitual_faceted_proportion_dots(
    distribution: pd.DataFrame,
    *,
    context: str,
    order: list[str],
    labels: dict[str, str],
    title: str,
    output_key: str,
) -> list[Path]:
    data = validate_proportion_dot_distribution(distribution, context=context, order=order)
    if len(order) == 4:
        nrows, ncols = 2, 2
        figsize = (12.6, 7.4)
        left_margin, bottom_margin = 0.18, 0.14
    elif len(order) == 3:
        nrows, ncols = 1, 3
        figsize = (13.8, 5.2)
        left_margin, bottom_margin = 0.15, 0.20
    else:
        raise ValueError("Faceted proportion-dot plotting supports the accepted three or four facets.")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    for index, level in enumerate(order):
        ax = flat_axes[index]
        panel = data.loc[data[context].eq(level)]
        show_y_labels = index % ncols == 0
        draw_proportion_dot_panel(ax, panel, annotate=False, show_y_labels=show_y_labels)
        ax.set_title(labels[level], loc="left", fontsize=11, fontweight="bold")
    for ax in flat_axes[len(order):]:
        ax.set_visible(False)

    fig.suptitle(title, x=left_margin, ha="left", fontsize=13, fontweight="bold")
    fig.supxlabel("Weighted share of persons (%)", y=0.065 if nrows == 2 else 0.10, fontsize=11)
    fig.supylabel("Habitual car-use frequency", x=0.018, fontsize=11)
    fig.legend(
        handles=dot_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=left_margin,
        right=0.99,
        top=0.88,
        bottom=bottom_margin,
        wspace=0.22,
        hspace=0.26,
    )
    return save_figure_png(fig, OUTPUT_DIR / PROPORTION_DOT_FILENAMES[output_key])


def create_proportion_dot_figures(
    overall_distribution: pd.DataFrame,
    household_distribution: pd.DataFrame,
    spatial_distribution: pd.DataFrame,
) -> list[Path]:
    """Create all three descriptive PNGs as optional appendix diagnostics."""
    paths = plot_habitual_overall_proportion_dots(overall_distribution)
    paths.extend(
        plot_habitual_faceted_proportion_dots(
            household_distribution,
            context="household_type",
            order=HOUSEHOLD_TYPE_ORDER,
            labels=HOUSEHOLD_TYPE_LABELS,
            title="Habitual car-use frequency by household type",
            output_key="household_type",
        )
    )
    paths.extend(
        plot_habitual_faceted_proportion_dots(
            spatial_distribution,
            context="spatial_context_3",
            order=SPATIAL_ORDER,
            labels=SPATIAL_LABELS,
            title="Habitual car-use frequency by spatial context",
            output_key="spatial",
        )
    )
    return paths


def read_saved_plot_distributions() -> dict[str, pd.DataFrame]:
    distributions: dict[str, pd.DataFrame] = {}
    for key, filename in PLOT_SOURCE_FILENAMES.items():
        path = OUTPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Saved weighted descriptive plotting input not found: {path}")
        distributions[key] = pd.read_csv(path, low_memory=False)
    return distributions


def plot_only_main() -> None:
    """Regenerate the thesis-facing overall figure without fitting any model."""
    distributions = read_saved_plot_distributions()
    saved_paths = plot_habitual_overall_proportion_dots(distributions["overall"])
    print("\nHABITUAL CAR-USE FREQUENCY - FIGURE UPDATE COMPLETE")
    print("\nLoaded weighted descriptive files:")
    print(f"  {OUTPUT_DIR / PLOT_SOURCE_FILENAMES['overall']}")
    print("\nSaved figures:")
    for path in saved_paths:
        print(f"  {path}")


def qa_row(
    section: str,
    metric: str,
    count: float | int | None,
    denominator: int | None,
    denominator_description: str,
    notes: str = "",
) -> dict[str, Any]:
    share = np.nan if denominator in (None, 0) or count is None else float(count) / denominator
    return {
        "section": section,
        "metric": metric,
        "count": count,
        "share": share,
        "denominator_description": denominator_description,
        "notes": notes,
    }


def build_sample_qa(
    persons: pd.DataFrame,
    merged: pd.DataFrame,
    samples: dict[str, pd.DataFrame],
    model_results: dict[str, CompactGEEResult],
    probability_issues: dict[str, list[str]],
) -> pd.DataFrame:
    n_source = len(persons)
    matched = merged["person_household_merge"].eq("both")
    outcome = merged["P_NUTZ_AUTO"]
    weight = numeric(merged["P_GEW"])
    unexpected_outcome = outcome.notna() & ~outcome.isin([1, 2, 3, 4, 5, 9, 206, 402])
    rows = [
        qa_row("source", "source person rows", n_source, n_source, "source person rows"),
        qa_row("source", "unique persons", persons["HP_ID"].nunique(), n_source, "source person rows"),
        qa_row("source", "unique households", persons["H_ID"].nunique(), n_source, "source person rows"),
        qa_row("merge", "persons matched to HH backbone", int(matched.sum()), n_source, "source person rows"),
        qa_row("merge", "persons with unmatched H_ID", int((~matched).sum()), n_source, "source person rows"),
    ]
    for code in [1, 2, 3, 4, 5, 9, 206, 402]:
        rows.append(
            qa_row("outcome", f"P_NUTZ_AUTO == {code}", int(outcome.eq(code).sum()), n_source, "source person rows")
        )
    rows.extend(
        [
            qa_row("outcome", "missing P_NUTZ_AUTO", int(outcome.isna().sum()), n_source, "source person rows"),
            qa_row(
                "outcome",
                "other unexpected P_NUTZ_AUTO",
                int(unexpected_outcome.sum()),
                n_source,
                "source person rows",
            ),
            qa_row("weight", "missing/non-numeric P_GEW", int(weight.isna().sum()), n_source, "source person rows"),
            qa_row("weight", "zero P_GEW", int(weight.eq(0).sum()), n_source, "source person rows"),
            qa_row("weight", "negative P_GEW", int(weight.lt(0).sum()), n_source, "source person rows"),
            qa_row("weight", "non-finite P_GEW", int(np.isinf(weight).sum()), n_source, "source person rows"),
            qa_row(
                "car ownership",
                "persons in zero-car households",
                int((matched & merged["H_ANZAUTO"].eq(0)).sum()),
                n_source,
                "source person rows",
            ),
            qa_row(
                "car ownership",
                "persons in single-car households",
                int((matched & merged["car_ownership_group"].eq("single_car")).sum()),
                n_source,
                "source person rows",
            ),
            qa_row(
                "car ownership",
                "persons in multi-car households",
                int((matched & merged["car_ownership_group"].eq("multi_car")).sum()),
                n_source,
                "source person rows",
                "H_ANZAUTO=3 means three or more cars.",
            ),
            qa_row(
                "car ownership",
                "persons with missing/invalid car ownership",
                int((~matched | ~merged["car_ownership_group"].isin(CAR_ORDER)).sum()),
                n_source,
                "source person rows",
                "Includes unmatched H_ID; categories are not silently reassigned.",
            ),
        ]
    )
    for key, label in [
        ("overall", "overall valid analysis"),
        ("spatial", "spatial valid analysis"),
        ("household_type", "HH-type valid analysis"),
    ]:
        sample = samples[key]
        rows.append(qa_row("samples", f"{label} persons", len(sample), n_source, "source person rows"))
        rows.append(
            qa_row(
                "samples",
                f"{label} households",
                sample["H_ID"].nunique(),
                persons["H_ID"].nunique(),
                "unique source households",
            )
        )
    unknown = samples["overall"]["household_type"].eq("unknown") | samples["overall"]["household_type"].isna()
    rows.append(
        qa_row(
            "samples",
            "unknown HH-type persons",
            int(unknown.sum()),
            len(samples["overall"]),
            "overall analysis persons",
            "Excluded only from the household-type interaction sample.",
        )
    )
    for key, label in [
        ("overall", "overall"),
        ("spatial", "spatial"),
        ("household_type", "HH-type"),
    ]:
        result = model_results[key]
        rows.append(
            qa_row(
                "threshold expansion",
                f"ordinal expanded {label} model rows",
                result.expanded_rows,
                None,
                "model-internal rows; not a substantive sample size",
            )
        )
        rows.append(
            qa_row(
                "threshold expansion",
                f"expected ordinal expanded {label} rows",
                4 * len(samples[key]),
                None,
                "four thresholds per original person",
            )
        )
    for context, order in [("spatial_context_3", SPATIAL_ORDER), ("household_type", HOUSEHOLD_TYPE_ORDER)]:
        sample_key = "spatial" if context == "spatial_context_3" else "household_type"
        for level in order:
            for car in CAR_ORDER:
                count = int(
                    (samples[sample_key][context].eq(level) & samples[sample_key]["car_ownership_group"].eq(car)).sum()
                )
                rows.append(
                    qa_row(
                        "context cells",
                        f"{context}: {level} × {car}",
                        count,
                        len(samples[sample_key]),
                        f"{sample_key} analysis persons",
                        "SMALL CELL" if count < SMALL_CELL_THRESHOLD else "",
                    )
                )
    for model_name, issues in probability_issues.items():
        rows.append(
            qa_row(
                "predicted probabilities",
                f"invalid/non-monotonic scenarios: {model_name}",
                len(issues),
                None,
                "model prediction scenarios",
                "; ".join(issues) if issues else "All predicted probabilities valid and monotonic.",
            )
        )
    return pd.DataFrame(rows)


def build_weight_qa(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall_weight = samples["overall"]["P_GEW"]
    metrics = {
        "minimum P_GEW": overall_weight.min(),
        "maximum P_GEW": overall_weight.max(),
        "mean P_GEW": overall_weight.mean(),
        "median P_GEW": overall_weight.median(),
        "p01 P_GEW": overall_weight.quantile(0.01),
        "p05 P_GEW": overall_weight.quantile(0.05),
        "p95 P_GEW": overall_weight.quantile(0.95),
        "p99 P_GEW": overall_weight.quantile(0.99),
        "sum P_GEW": overall_weight.sum(),
    }
    for metric, value in metrics.items():
        rows.append(
            {
                "analysis_sample": "overall",
                "metric": metric,
                "value": float(value),
                "n_persons": len(samples["overall"]),
                "n_households": samples["overall"]["H_ID"].nunique(),
                "person_weight_sum": float(overall_weight.sum()),
                "notes": "Raw, unnormalized positive P_GEW on one-row-per-person data.",
            }
        )
    for name in ["spatial", "household_type"]:
        sample = samples[name]
        for metric, value in [
            ("n_persons", len(sample)),
            ("n_households", sample["H_ID"].nunique()),
            ("sum P_GEW", sample["P_GEW"].sum()),
        ]:
            rows.append(
                {
                    "analysis_sample": name,
                    "metric": metric,
                    "value": float(value),
                    "n_persons": len(sample),
                    "n_households": sample["H_ID"].nunique(),
                    "person_weight_sum": float(sample["P_GEW"].sum()),
                    "notes": "Original one-row-per-person analytical sample.",
                }
            )
    return pd.DataFrame(rows)


def write_metadata() -> None:
    rows = [
        ("research_question_1", "How is habitual car-use frequency distributed among persons in Single- and Multi-car households?"),
        ("research_question_2", "Is Multi-car household membership associated with a systematically different ordinal habitual car-use frequency?"),
        ("research_question_3", "Does the Single-vs-Multi association differ across spatial contexts?"),
        ("research_question_4", "Does the Single-vs-Multi association differ across household types?"),
        ("analytical_unit", ANALYTICAL_UNIT),
        ("weight", "P_GEW (raw positive values; not normalized)"),
        ("cluster", "H_ID"),
        ("working_correlation", WORKING_CORRELATION),
        ("covariance", COVARIANCE),
        ("method", METHOD),
        ("person_source", str(PERSON_PATH)),
        ("household_backbone", str(HOUSEHOLD_BACKBONE_PATH)),
        ("statsmodels", statsmodels.__version__),
        ("interpretation", "Associational EDA; not causal and not a complete complex-survey design estimator."),
    ]
    write_csv(pd.DataFrame(rows, columns=["item", "value"]), "analysis_metadata.csv")


def print_console_summary(
    persons: pd.DataFrame,
    samples: dict[str, pd.DataFrame],
    overall_distribution: pd.DataFrame,
    model_results: dict[str, CompactGEEResult],
    wald: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    overall_table = coefficient_table(model_results["overall"])
    multi = overall_table.loc[overall_table["term"].eq(MULTI_TERM)].iloc[0]
    print("\nTHEME 1 — HABITUAL CAR-USE ORIENTATION")
    print("\nINPUT")
    print(f"person-level source: {PERSON_PATH}")
    print(f"HH backbone source: {HOUSEHOLD_BACKBONE_PATH}")
    print("\nENVIRONMENT")
    print(f"statsmodels: {statsmodels.__version__}")
    print("estimator: threshold-expanded weighted GEE")
    print("\nSAMPLE")
    print(f"source persons: {len(persons):,}")
    for name, label in [("overall", "overall valid"), ("spatial", "spatial-model"), ("household_type", "HH-type-model")]:
        print(f"{label} persons: {len(samples[name]):,}")
        print(f"{label} households: {samples[name]['H_ID'].nunique():,}")
    print("\nWEIGHTED DISTRIBUTION")
    for car in CAR_ORDER:
        print(f"{CAR_LABELS[car]}:")
        data = overall_distribution.loc[overall_distribution["car_ownership_group"].eq(car)]
        for code in OUTCOME_CODES:
            share = float(data.loc[data["P_NUTZ_AUTO"].eq(code), "weighted_share"].iloc[0])
            print(f"  {OUTCOME_LABELS[code]}: {share:.1%}")
    print("\nOVERALL PROPORTIONAL-ODDS MODEL")
    print(f"Multi-car coefficient: {multi['estimate']:.4f}")
    print(f"OR: {multi['odds_ratio']:.4f}")
    print(f"95% CI: [{multi['or_ci95_low']:.4f}, {multi['or_ci95_high']:.4f}]")
    print(f"p-value: {multi['p_value']:.6g}")
    for context, label in [("spatial_context_3", "SPATIAL INTERACTION"), ("household_type", "HOUSEHOLD-TYPE INTERACTION")]:
        row = wald.loc[wald["context_variable"].eq(context)].iloc[0]
        print(f"\n{label}")
        print(f"joint Wald statistic: {row['wald_statistic']:.4f}")
        print(f"df: {int(row['degrees_of_freedom'])}")
        print(f"p-value: {row['p_value']:.6g}")
    print("\nPROPORTIONAL-ODDS DIAGNOSTIC")
    print("threshold-specific Multi-car ORs: " + ", ".join(f"k={int(r.threshold)}: {r.odds_ratio:.3f}" for r in diagnostic.itertuples()))
    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)


def main() -> None:
    assert statsmodels.__version__ == STATSMODELS_VERSION, (
        f"This module requires statsmodels {STATSMODELS_VERSION}; found {statsmodels.__version__}."
    )
    assert hasattr(sm, "GEE")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    persons, households = read_sources()
    outcome_observed = observed_outcome_distribution(persons)
    write_csv(outcome_observed, "00_P_NUTZ_AUTO_observed_distribution.csv")
    merged = merge_person_household(persons, households)
    samples = make_samples(merged)

    overall_distribution = weighted_distribution(
        samples["overall"], ["car_ownership_group"], {"car_ownership_group": CAR_ORDER}, "overall"
    )
    spatial_distribution = weighted_distribution(
        samples["spatial"],
        ["spatial_context_3", "car_ownership_group"],
        {"spatial_context_3": SPATIAL_ORDER, "car_ownership_group": CAR_ORDER},
        "spatial_context_3",
    )
    household_distribution = weighted_distribution(
        samples["household_type"],
        ["household_type", "car_ownership_group"],
        {"household_type": HOUSEHOLD_TYPE_ORDER, "car_ownership_group": CAR_ORDER},
        "household_type",
    )
    write_csv(overall_distribution, "01_weighted_distribution_overall.csv")
    write_csv(spatial_distribution, "02_weighted_distribution_spatial.csv")
    write_csv(household_distribution, "03_weighted_distribution_household_type.csv")
    write_csv(publication_table(overall_distribution, None), "publication_table_overall.csv")
    write_csv(
        publication_table(spatial_distribution, "spatial_context_3", SPATIAL_LABELS),
        "publication_table_spatial.csv",
    )
    write_csv(
        publication_table(household_distribution, "household_type", HOUSEHOLD_TYPE_LABELS),
        "publication_table_household_type.csv",
    )

    plot_habitual_overall_proportion_dots(overall_distribution)

    model_results = {
        "overall": fit_ordinal_gee(samples["overall"], "overall_proportional_odds"),
        "spatial": fit_ordinal_gee(
            samples["spatial"], "spatial_context_interaction", "spatial_context_3", SPATIAL_ORDER
        ),
        "household_type": fit_ordinal_gee(
            samples["household_type"],
            "household_type_interaction",
            "household_type",
            HOUSEHOLD_TYPE_ORDER,
        ),
    }
    write_csv(coefficient_table(model_results["overall"]), "04_ordered_model_overall.csv")
    write_csv(coefficient_table(model_results["spatial"]), "05_ordered_model_spatial.csv")
    spatial_contrasts = context_contrasts(
        model_results["spatial"], samples["spatial"], "spatial_context_3", SPATIAL_ORDER
    )
    write_csv(spatial_contrasts, "06_spatial_multicar_contrasts.csv")
    write_csv(coefficient_table(model_results["household_type"]), "07_ordered_model_household_type.csv")
    household_contrasts = context_contrasts(
        model_results["household_type"],
        samples["household_type"],
        "household_type",
        HOUSEHOLD_TYPE_ORDER,
    )
    write_csv(household_contrasts, "08_household_type_multicar_contrasts.csv")
    wald = pd.DataFrame(
        [
            joint_wald(model_results["spatial"], "spatial_context_3", SPATIAL_ORDER),
            joint_wald(model_results["household_type"], "household_type", HOUSEHOLD_TYPE_ORDER),
        ]
    )
    write_csv(wald, "09_joint_wald_tests.csv")

    diagnostic = pd.DataFrame([fit_binary_diagnostic(samples["overall"], k) for k in range(1, 5)])
    write_csv(diagnostic, "10_proportional_odds_diagnostic.csv")

    prediction_specs = [
        ("overall", None, None, "13_predicted_probabilities_overall.csv"),
        ("spatial", "spatial_context_3", SPATIAL_ORDER, "14_predicted_probabilities_spatial.csv"),
        (
            "household_type",
            "household_type",
            HOUSEHOLD_TYPE_ORDER,
            "15_predicted_probabilities_household_type.csv",
        ),
    ]
    probability_issues: dict[str, list[str]] = {}
    for model_name, context, order, filename in prediction_specs:
        predictions, issues = predicted_probabilities(model_results[model_name], context, order)
        probability_issues[model_name] = issues
        if not issues:
            write_csv(predictions, filename)

    sample_qa = build_sample_qa(persons, merged, samples, model_results, probability_issues)
    write_csv(sample_qa, "11_sample_QA.csv")
    write_csv(build_weight_qa(samples), "12_weight_QA.csv")
    write_metadata()
    print_console_summary(persons, samples, overall_distribution, model_results, wald, diagnostic)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate descriptive figures from saved weighted distributions without fitting models.",
    )
    arguments = argument_parser.parse_args()
    if arguments.plots_only:
        plot_only_main()
    else:
        main()
