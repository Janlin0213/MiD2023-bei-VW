"""Analyse driver/car ratios in single- vs multi-car households.

This script consumes the canonical Theme 1 household backbone. It implements
household-weighted linear regression with HC3 heteroskedasticity-robust
standard errors directly from the transformed-WLS estimating equations, so
the analysis has no dependency on a particular regression package.

The robust intervals are household-weighted/HC3 intervals; they are not fully
survey-design-adjusted because PSU and stratum information is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[3]
BACKBONE_PATH = (
    ROOT
    / "data_processed"
    / "eda"
    / "statistical_test"
    / "theme1_household_backbone.csv"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "eda"
    / "statistical_tests"
    / "theme1_single_vs_multicar"
    / "01_licensed_driver_resources"
)

CAR_ORDER = ["single_car", "multi_car"]
REGIOSTAR4_ORDER = [11, 12, 21, 22]
HOUSEHOLD_TYPE_ORDER = [
    "family_household",
    "young_household",
    "adult_household",
    "senior_household",
    "unknown",
]

CAR_TERM = "C(car_ownership_group)[T.multi_car]"
Z_95 = float(norm.ppf(0.975))

EXACT_SCOPE = (
    "Exact driver/car ratio: Single = exactly one car; Multi = exactly two cars; "
    "3+ households excluded because the denominator is top-coded; complete licence "
    "information and valid positive finite H_GEW. Household-weighted estimates with "
    "HC3 robust SEs; not fully survey-design-adjusted."
)
CAPPED_SCOPE = (
    "Capped3 upper-bound sensitivity including 3+ households: Single = one car; "
    "Multi = two or 3+ cars. For 3+ households, licensed drivers / 3 is an upper-bound "
    "proxy, not the exact driver/car ratio. Complete licence information and valid "
    "positive finite H_GEW. Household-weighted estimates with HC3 robust SEs; not "
    "fully survey-design-adjusted."
)


@dataclass(frozen=True)
class WLSHC3Result:
    beta: np.ndarray
    covariance: np.ndarray
    terms: list[str]
    nobs: int
    weight_sum: float


def require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(
            f"Canonical backbone is missing required column(s): {missing}. "
            "Run 01_build_theme1_household_backbone.py first."
        )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_weight_mask(weight: pd.Series) -> pd.Series:
    values = numeric(weight)
    return values.notna() & np.isfinite(values) & values.gt(0)


def read_backbone() -> pd.DataFrame:
    if not BACKBONE_PATH.exists():
        raise FileNotFoundError(
            f"Canonical backbone not found: {BACKBONE_PATH}. "
            "Run 01_build_theme1_household_backbone.py first."
        )
    frame = pd.read_csv(BACKBONE_PATH, dtype={"H_ID": "string"}, low_memory=False)
    required = [
        "H_ID",
        "H_GEW",
        "H_ANZAUTO",
        "car_ownership_group",
        "RegioStaR4",
        "household_type",
        "n_licensed_drivers",
        "license_information_complete_flag",
        "driver_car_ratio_exact",
        "driver_car_ratio_capped3_upper_bound",
        "driver_car_ratio_exact_flag",
    ]
    require_columns(frame, required)
    if frame["H_ID"].isna().any() or frame["H_ID"].duplicated().any():
        sample = frame.loc[frame["H_ID"].isna() | frame["H_ID"].duplicated(keep=False)].head(10)
        raise RuntimeError(f"Backbone is not one row per nonmissing H_ID:\n{sample}")

    for column in [
        "H_GEW",
        "H_ANZAUTO",
        "RegioStaR4",
        "n_licensed_drivers",
        "license_information_complete_flag",
        "driver_car_ratio_exact",
        "driver_car_ratio_capped3_upper_bound",
        "driver_car_ratio_exact_flag",
    ]:
        frame[column] = numeric(frame[column])
    return frame


def analysis_sample(frame: pd.DataFrame, outcome: str, *, exact: bool = False) -> pd.DataFrame:
    mask = (
        frame["car_ownership_group"].isin(CAR_ORDER)
        & frame["license_information_complete_flag"].eq(1)
        & valid_weight_mask(frame["H_GEW"])
        & numeric(frame[outcome]).notna()
        & np.isfinite(numeric(frame[outcome]))
    )
    if exact:
        mask &= frame["driver_car_ratio_exact_flag"].eq(1) & frame["H_ANZAUTO"].isin([1, 2])
    sample = frame.loc[mask].copy()
    if sample.empty:
        raise RuntimeError(f"No valid observations for outcome '{outcome}'.")
    return sample


def fit_wls_hc3(y: np.ndarray, x: np.ndarray, weights: np.ndarray, terms: list[str]) -> WLSHC3Result:
    """Fit WLS and the standard HC3 sandwich covariance on transformed data.

    WLS is OLS on ``sqrt(w) * y`` and ``sqrt(w) * X``. The HC3 meat uses
    squared transformed residuals divided by ``(1 - leverage)^2``. This is the
    same estimating-equation definition used by standard WLS/HC3 software.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(y) != x.shape[0] or len(weights) != len(y):
        raise ValueError("Incompatible WLS array shapes.")
    if not (np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(weights).all()):
        raise ValueError("WLS inputs must be finite.")
    if not (weights > 0).all():
        raise ValueError("WLS weights must be strictly positive.")
    nobs, n_terms = x.shape
    if nobs <= n_terms:
        raise RuntimeError(f"WLS model has insufficient observations: n={nobs}, p={n_terms}.")
    rank = int(np.linalg.matrix_rank(x))
    if rank != n_terms:
        raise RuntimeError(f"WLS design matrix is rank deficient: rank={rank}, terms={n_terms}.")

    xtwx = x.T @ (weights[:, None] * x)
    bread = np.linalg.inv(xtwx)
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    leverage = weights * np.einsum("ij,jk,ik->i", x, bread, x)
    if np.any(leverage >= 1 - 1e-10) or np.any(leverage < -1e-10):
        raise RuntimeError(
            "HC3 leverage values are outside the stable [0, 1) range; inspect model cells."
        )

    # On transformed WLS data, x*=sqrt(w)x and u*=sqrt(w)u. Therefore the
    # HC3 meat multiplier on x_i x_i' is w_i^2 u_i^2 / (1-h_i)^2.
    meat_scale = (weights**2) * (residual**2) / ((1.0 - leverage) ** 2)
    meat = x.T @ (meat_scale[:, None] * x)
    covariance = bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    return WLSHC3Result(beta, covariance, terms, nobs, float(weights.sum()))


def coefficient_table(
    result: WLSHC3Result,
    *,
    outcome: str,
    model: str,
    scope: str,
) -> pd.DataFrame:
    variance = np.diag(result.covariance)
    if np.any(variance < -1e-10):
        raise RuntimeError(f"Negative robust variance in {model}: {variance}")
    se = np.sqrt(np.maximum(variance, 0))
    statistic = np.divide(result.beta, se, out=np.full_like(result.beta, np.nan), where=se > 0)
    p_value = 2 * norm.sf(np.abs(statistic))
    return pd.DataFrame(
        {
            "outcome": outcome,
            "model": model,
            "term": result.terms,
            "estimate": result.beta,
            "robust_standard_error": se,
            "ci95_low": result.beta - Z_95 * se,
            "ci95_high": result.beta + Z_95 * se,
            "statistic": statistic,
            "p_value": p_value,
            "unweighted_n": result.nobs,
            "weight_sum": result.weight_sum,
            "analysis_scope": scope,
            "method": "household-weighted linear regression; HC3 robust covariance",
        }
    )


def build_design(
    frame: pd.DataFrame,
    *,
    context: str | None = None,
    requested_order: list[Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[Any]]:
    """Create treatment-coded Single-reference interaction design matrices."""
    data = frame.copy()
    multi = data["car_ownership_group"].eq("multi_car").astype(float)
    columns = [np.ones(len(data), dtype=float), multi.to_numpy()]
    terms = ["Intercept", CAR_TERM]
    used_order: list[Any] = []

    if context is not None:
        if requested_order is None:
            raise ValueError("An explicit deterministic context order is required.")
        observed = set(data[context].dropna().tolist())
        used_order = [level for level in requested_order if level in observed]
        unexpected = observed.difference(requested_order)
        if unexpected:
            raise RuntimeError(f"Unexpected {context} values in model sample: {sorted(unexpected)}")
        data = data.loc[data[context].isin(used_order)].copy()
        multi = data["car_ownership_group"].eq("multi_car").astype(float)
        columns = [np.ones(len(data), dtype=float), multi.to_numpy()]
        terms = ["Intercept", CAR_TERM]
        if len(used_order) < 2:
            raise RuntimeError(f"Interaction model requires at least two observed {context} levels.")
        for level in used_order[1:]:
            dummy = data[context].eq(level).astype(float)
            columns.append(dummy.to_numpy())
            terms.append(f"C({context})[T.{level}]")
        for level in used_order[1:]:
            interaction = multi * data[context].eq(level).astype(float)
            columns.append(interaction.to_numpy())
            terms.append(f"{CAR_TERM}:C({context})[T.{level}]")

        # A missing ownership group in any context cell makes the requested
        # within-context interaction unidentifiable, so fail with a compact diagnostic.
        cells = pd.crosstab(data[context], data["car_ownership_group"])
        missing_cells = [
            (level, group)
            for level in used_order
            for group in CAR_ORDER
            if level not in cells.index or group not in cells.columns or cells.loc[level, group] == 0
        ]
        if missing_cells:
            raise RuntimeError(f"Empty car-group/context cells for {context}: {missing_cells}")

    x = np.column_stack(columns)
    return data, x, terms, used_order


def fit_model(
    frame: pd.DataFrame,
    outcome: str,
    *,
    model: str,
    scope: str,
    context: str | None = None,
    order: list[Any] | None = None,
) -> tuple[WLSHC3Result, pd.DataFrame, pd.DataFrame, list[Any]]:
    data, x, terms, used_order = build_design(frame, context=context, requested_order=order)
    result = fit_wls_hc3(
        numeric(data[outcome]).to_numpy(),
        x,
        numeric(data["H_GEW"]).to_numpy(),
        terms,
    )
    table = coefficient_table(result, outcome=outcome, model=model, scope=scope)
    return result, table, data, used_order


def joint_wald_test(
    result: WLSHC3Result,
    term_indices: list[int],
    *,
    outcome: str,
    model: str,
    tested_term: str,
    scope: str,
) -> dict[str, Any]:
    if not term_indices:
        raise RuntimeError(f"No interaction terms found for joint Wald test in {model}.")
    estimates = result.beta[term_indices]
    covariance = result.covariance[np.ix_(term_indices, term_indices)]
    df = int(np.linalg.matrix_rank(covariance))
    if df != len(term_indices):
        raise RuntimeError(f"Interaction covariance is rank deficient in {model}: rank={df}.")
    statistic = float(estimates.T @ np.linalg.inv(covariance) @ estimates)
    return {
        "outcome": outcome,
        "model": model,
        "tested_term": tested_term,
        "wald_statistic": statistic,
        "degrees_of_freedom": df,
        "p_value": float(chi2.sf(statistic, df)),
        "unweighted_n": result.nobs,
        "weight_sum": result.weight_sum,
        "analysis_scope": scope,
        "method": "joint Wald chi-square test using WLS HC3 covariance",
    }


def prediction_vector(
    terms: list[str],
    *,
    car_group: str,
    context: str,
    level: Any,
) -> np.ndarray:
    values = np.zeros(len(terms), dtype=float)
    values[terms.index("Intercept")] = 1.0
    if car_group == "multi_car":
        values[terms.index(CAR_TERM)] = 1.0
    main_term = f"C({context})[T.{level}]"
    interaction_term = f"{CAR_TERM}:C({context})[T.{level}]"
    if main_term in terms:
        values[terms.index(main_term)] = 1.0
    if car_group == "multi_car" and interaction_term in terms:
        values[terms.index(interaction_term)] = 1.0
    return values


def context_contrasts(
    result: WLSHC3Result,
    *,
    outcome: str,
    model: str,
    context: str,
    order: list[Any],
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in order:
        single_x = prediction_vector(result.terms, car_group="single_car", context=context, level=level)
        multi_x = prediction_vector(result.terms, car_group="multi_car", context=context, level=level)
        contrast = multi_x - single_x
        single_mean = float(single_x @ result.beta)
        multi_mean = float(multi_x @ result.beta)
        difference = float(contrast @ result.beta)
        variance = float(contrast @ result.covariance @ contrast)
        if variance < -1e-10:
            raise RuntimeError(f"Negative contrast variance in {model}, {level}: {variance}")
        se = float(np.sqrt(max(variance, 0)))
        statistic = difference / se if se > 0 else np.nan
        rows.append(
            {
                "outcome": outcome,
                "model": model,
                "context_variable": context,
                "context": level,
                "single_estimated_mean": single_mean,
                "multi_estimated_mean": multi_mean,
                "difference_multi_minus_single": difference,
                "standard_error": se,
                "ci95_low": difference - Z_95 * se,
                "ci95_high": difference + Z_95 * se,
                "p_value": float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan,
                "unweighted_n": result.nobs,
                "weight_sum": result.weight_sum,
                "analysis_scope": scope,
            }
        )
    return pd.DataFrame(rows)


def intercept_only_summary(group: pd.DataFrame, outcome: str) -> tuple[float, float, float, float]:
    """Weighted mean and HC3 uncertainty via intercept-only WLS.

    The WLS intercept is exactly sum(H_GEW*y)/sum(H_GEW). Using the same HC3
    covariance implementation as the main models keeps group uncertainty
    internally consistent without implying full survey-design adjustment.
    """
    y = numeric(group[outcome]).to_numpy()
    weights = numeric(group["H_GEW"]).to_numpy()
    result = fit_wls_hc3(y, np.ones((len(group), 1)), weights, ["Intercept"])
    mean = float(result.beta[0])
    direct_mean = float(np.average(y, weights=weights))
    if not np.isclose(mean, direct_mean, rtol=1e-12, atol=1e-12):
        raise AssertionError("Intercept-only WLS did not reproduce the weighted mean.")
    se = float(np.sqrt(max(result.covariance[0, 0], 0)))
    return mean, se, mean - Z_95 * se, mean + Z_95 * se


def weighted_summary(
    frame: pd.DataFrame,
    outcome: str,
    group_columns: list[str],
    *,
    scope: str,
    orders: dict[str, list[Any]],
) -> pd.DataFrame:
    data = frame.dropna(subset=[outcome, "H_GEW", *group_columns]).copy()
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in data.groupby(grouper, sort=False, observed=True):
        key_tuple = (keys,) if len(group_columns) == 1 else tuple(keys)
        mean, se, low, high = intercept_only_summary(group, outcome)
        row = {column: value for column, value in zip(group_columns, key_tuple)}
        row.update(
            {
                "outcome": outcome,
                "unweighted_n": len(group),
                "weight_sum": float(numeric(group["H_GEW"]).sum()),
                "weighted_mean": mean,
                "robust_standard_error": se,
                "ci95_low": low,
                "ci95_high": high,
                "analysis_scope": scope,
            }
        )
        rows.append(row)

    output = pd.DataFrame(rows)
    sort_columns: list[str] = []
    for column in group_columns:
        order = orders[column]
        rank_column = f"_{column}_order"
        output[rank_column] = output[column].map({value: i for i, value in enumerate(order)})
        if output[rank_column].isna().any():
            unexpected = output.loc[output[rank_column].isna(), column].unique().tolist()
            raise RuntimeError(f"Unexpected {column} categories in summary: {unexpected}")
        sort_columns.append(rank_column)
    output = output.sort_values(sort_columns).drop(columns=sort_columns).reset_index(drop=True)
    return output


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT_DIR / filename, index=False, float_format="%.10g")


def create_descriptive_outputs(
    sample: pd.DataFrame,
    outcome: str,
    prefix: str,
    scope: str,
) -> dict[str, pd.DataFrame]:
    specifications = {
        "overall": (["car_ownership_group"], {"car_ownership_group": CAR_ORDER}),
        "regiostar4": (
            ["RegioStaR4", "car_ownership_group"],
            {"RegioStaR4": REGIOSTAR4_ORDER, "car_ownership_group": CAR_ORDER},
        ),
        "household_type": (
            ["household_type", "car_ownership_group"],
            {"household_type": HOUSEHOLD_TYPE_ORDER, "car_ownership_group": CAR_ORDER},
        ),
    }
    outputs: dict[str, pd.DataFrame] = {}
    for label, (groups, orders) in specifications.items():
        summary = weighted_summary(sample, outcome, groups, scope=scope, orders=orders)
        outputs[label] = summary
        write_csv(summary, f"{prefix}_weighted_summary_{label}.csv")
    return outputs


def run_model_set(
    sample: pd.DataFrame,
    outcome: str,
    prefix: str,
    scope: str,
) -> dict[str, Any]:
    overall_result, overall_table, _, _ = fit_model(
        sample,
        outcome,
        model="overall_difference",
        scope=scope,
    )
    write_csv(overall_table, f"model_{prefix}_overall.csv")

    model_specs = [
        ("regiostar4", "RegioStaR4", REGIOSTAR4_ORDER, "RegioStaR4_interaction"),
        ("household_type", "household_type", HOUSEHOLD_TYPE_ORDER, "household_type_interaction"),
    ]
    results: dict[str, WLSHC3Result] = {"overall": overall_result}
    used_orders: dict[str, list[Any]] = {}
    wald_rows: list[dict[str, Any]] = []
    contrast_outputs: dict[str, pd.DataFrame] = {}

    for file_label, context, order, model_name in model_specs:
        result, table, _, used_order = fit_model(
            sample,
            outcome,
            model=model_name,
            scope=scope,
            context=context,
            order=order,
        )
        results[file_label] = result
        used_orders[file_label] = used_order
        write_csv(table, f"model_{prefix}_{file_label}.csv")

        interaction_indices = [i for i, term in enumerate(result.terms) if ":" in term]
        wald_rows.append(
            joint_wald_test(
                result,
                interaction_indices,
                outcome=outcome,
                model=model_name,
                tested_term=f"car_ownership_group × {context}",
                scope=scope,
            )
        )

        contrasts = context_contrasts(
            result,
            outcome=outcome,
            model=model_name,
            context=context,
            order=used_order,
            scope=scope,
        )
        contrast_outputs[file_label] = contrasts
        write_csv(contrasts, f"contrasts_{prefix}_{file_label}.csv")

    wald = pd.DataFrame(wald_rows)
    write_csv(wald, f"wald_{prefix}.csv")
    return {
        "results": results,
        "wald": wald,
        "contrasts": contrast_outputs,
        "used_orders": used_orders,
    }


def plot_weighted_means(
    summary: pd.DataFrame,
    *,
    context: str,
    order: list[Any],
    labels: dict[Any, str],
    title: str,
    y_label: str,
    path: Path,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x_positions = np.arange(len(order), dtype=float)
    offsets = {"single_car": -0.13, "multi_car": 0.13}
    colors = {"single_car": "#4C78A8", "multi_car": "#F58518"}
    display = {"single_car": "Single-car", "multi_car": "Multi-car"}

    for car_group in CAR_ORDER:
        group = summary.loc[summary["car_ownership_group"].eq(car_group)].set_index(context)
        missing = [level for level in order if level not in group.index]
        if missing:
            raise RuntimeError(f"Cannot plot {path.name}; missing {context} cells: {missing}")
        group = group.loc[order]
        means = group["weighted_mean"].to_numpy(dtype=float)
        lower = means - group["ci95_low"].to_numpy(dtype=float)
        upper = group["ci95_high"].to_numpy(dtype=float) - means
        point_x = x_positions + offsets[car_group]
        ax.errorbar(
            point_x,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=6,
            capsize=4,
            linewidth=1.5,
            color=colors[car_group],
            label=display[car_group],
        )
        annotation_x = 10 if car_group == "single_car" else -10
        horizontal_alignment = "left" if car_group == "single_car" else "right"
        for x_value, mean, unweighted_n in zip(
            point_x,
            means,
            group["unweighted_n"].to_numpy(dtype=int),
        ):
            ax.annotate(
                f"{mean:.3f}",
                xy=(x_value, mean),
                xytext=(annotation_x, 5),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=colors[car_group],
            )
            ax.annotate(
                f"n={unweighted_n:,}",
                xy=(x_value, mean),
                xytext=(annotation_x, -6),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="top",
                fontsize=7.5,
                color="#666666",
            )

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_xticks(x_positions, [labels[level] for level in order])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.18)
    ax.legend(frameon=False, ncol=2, loc="best")
    # Fixed margins keep long analytical y-axis labels fully visible and make
    # all figures render to a consistent publication-ready canvas.
    bottom = 0.32 if note else 0.15
    fig.subplots_adjust(left=0.16, right=0.98, bottom=bottom, top=0.88)
    if note:
        wrapped_note = "\n".join(textwrap.fill(line, width=100) for line in note.splitlines())
        fig.text(0.16, 0.04, wrapped_note, fontsize=8.5, linespacing=1.35)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_sample_qa(frame: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible = frame["car_ownership_group"].isin(CAR_ORDER) & valid_weight_mask(frame["H_GEW"])
    incomplete = eligible & ~frame["license_information_complete_flag"].eq(1)
    eligible_weight_sum = float(frame.loc[eligible, "H_GEW"].sum())
    multi = eligible & frame["car_ownership_group"].eq("multi_car")
    three_plus = multi & frame["H_ANZAUTO"].eq(3)
    rows = [
        {
            "metric": "Single/Multi households excluded due to unresolved licence information",
            "count": int(incomplete.sum()),
            "share": float(frame.loc[incomplete, "H_GEW"].sum() / eligible_weight_sum),
            "notes": "H_GEW-weighted share among Single/Multi households with valid H_GEW; includes absent person roster.",
        },
        {
            "metric": "unweighted share of 3+ among multi-car households",
            "count": int(three_plus.sum()),
            "share": float(three_plus.sum() / multi.sum()),
            "notes": "H_ANZAUTO=3 means three or more cars.",
        },
        {
            "metric": "H_GEW-weighted share of 3+ among multi-car households",
            "count": int(three_plus.sum()),
            "share": float(frame.loc[three_plus, "H_GEW"].sum() / frame.loc[multi, "H_GEW"].sum()),
            "notes": "Weight sums are not estimated German household counts.",
        },
    ]
    for name, sample in samples.items():
        rows.append(
            {
                "metric": f"analysis sample: {name}",
                "count": len(sample),
                "share": float(sample["H_GEW"].sum() / eligible_weight_sum),
                "notes": "Share is an H_GEW-weighted coverage measure relative to valid-weight Single/Multi households.",
            }
        )
    return pd.DataFrame(rows)


def extract_overall_statistics(
    descriptive: pd.DataFrame,
    model_output: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    means = descriptive.set_index("car_ownership_group")["weighted_mean"]
    result = model_output["results"]["overall"]
    term_index = result.terms.index(CAR_TERM)
    difference = float(result.beta[term_index])
    se = float(np.sqrt(result.covariance[term_index, term_index]))
    p_value = float(2 * norm.sf(abs(difference / se)))
    return float(means["single_car"]), float(means["multi_car"]), difference, se, p_value


def find_wald(model_output: dict[str, Any], model: str) -> pd.Series:
    row = model_output["wald"].loc[model_output["wald"]["model"].eq(model)]
    if len(row) != 1:
        raise AssertionError(f"Expected exactly one Wald row for {model}.")
    return row.iloc[0]


def print_wald(label: str, row: pd.Series) -> None:
    print(label)
    print(f"statistic: {row['wald_statistic']:.4f}")
    print(f"df: {int(row['degrees_of_freedom'])}")
    print(f"p: {row['p_value']:.6g}")


def print_console_summary(
    samples: dict[str, pd.DataFrame],
    descriptions: dict[str, dict[str, pd.DataFrame]],
    models: dict[str, dict[str, Any]],
) -> None:
    exact_single, exact_multi, exact_diff, exact_se, exact_p = extract_overall_statistics(
        descriptions["exact"]["overall"], models["exact"]
    )
    capped_single, capped_multi, capped_diff, capped_se, capped_p = extract_overall_statistics(
        descriptions["capped"]["overall"], models["capped"]
    )

    print("\nDRIVER/CAR RATIO ANALYSIS")
    print("\nPRIMARY EXACT DRIVER/CAR RATIO")
    print(f"Single-car households: {samples['exact']['H_ANZAUTO'].eq(1).sum():,}")
    print(f"Exactly-two-car households: {samples['exact']['H_ANZAUTO'].eq(2).sum():,}")
    print(f"3+ excluded: {samples['capped']['H_ANZAUTO'].eq(3).sum():,}")
    print("\nWeighted ratio:")
    print(f"Single: {exact_single:.4f}")
    print(f"Two-car: {exact_multi:.4f}")
    print(f"Difference (Two-car - Single): {exact_diff:.4f} (SE {exact_se:.4f}, p {exact_p:.6g})")
    print()
    print_wald("RegioStaR4 interaction joint Wald:", find_wald(models["exact"], "RegioStaR4_interaction"))
    print()
    print_wald("Household-type interaction joint Wald:", find_wald(models["exact"], "household_type_interaction"))

    print("\nCAPPED-3 UPPER-BOUND SENSITIVITY")
    print(f"3+ households included: {samples['capped']['H_ANZAUTO'].eq(3).sum():,}")
    print(f"Weighted ratio Single: {capped_single:.4f}")
    print(f"Weighted ratio Multi: {capped_multi:.4f}")
    print(f"Difference (Multi - Single): {capped_diff:.4f} (SE {capped_se:.4f}, p {capped_p:.6g})")
    print("Comparison statistics (interpretation left to the researcher):")
    print(f"exact vs capped overall differences: {exact_diff:.4f} vs {capped_diff:.4f}")
    print(
        "exact vs capped RegioStaR4 Wald p-values: "
        f"{find_wald(models['exact'], 'RegioStaR4_interaction')['p_value']:.6g} vs "
        f"{find_wald(models['capped'], 'RegioStaR4_interaction')['p_value']:.6g}"
    )
    print(
        "exact vs capped household-type Wald p-values: "
        f"{find_wald(models['exact'], 'household_type_interaction')['p_value']:.6g} vs "
        f"{find_wald(models['capped'], 'household_type_interaction')['p_value']:.6g}"
    )
    print("\nOutputs:")
    print(OUTPUT_DIR)


def main() -> None:
    backbone = read_backbone()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "exact": analysis_sample(backbone, "driver_car_ratio_exact", exact=True),
        "capped": analysis_sample(backbone, "driver_car_ratio_capped3_upper_bound"),
    }
    sample_qa = build_sample_qa(backbone, samples)
    write_csv(sample_qa, "analysis_sample_QA.csv")

    descriptions = {
        "exact": create_descriptive_outputs(
            samples["exact"], "driver_car_ratio_exact", "driver_car_ratio_exact", EXACT_SCOPE
        ),
        "capped": create_descriptive_outputs(
            samples["capped"],
            "driver_car_ratio_capped3_upper_bound",
            "driver_car_ratio_capped3",
            CAPPED_SCOPE,
        ),
    }

    models = {
        "exact": run_model_set(
            samples["exact"], "driver_car_ratio_exact", "driver_car_ratio_exact", EXACT_SCOPE
        ),
        "capped": run_model_set(
            samples["capped"],
            "driver_car_ratio_capped3_upper_bound",
            "driver_car_ratio_capped3",
            CAPPED_SCOPE,
        ),
    }

    regiostar4_labels = {
        11: "Metropolitan\ncity region",
        12: "Regiopolitan\ncity region",
        21: "Rural region near\na city region",
        22: "Peripheral\nrural region",
    }
    household_labels = {
        "family_household": "Family",
        "young_household": "Young",
        "adult_household": "Adult",
        "senior_household": "Senior",
        "unknown": "Unknown",
    }
    observed_household_order = [
        level
        for level in HOUSEHOLD_TYPE_ORDER
        if level in set(descriptions["exact"]["household_type"]["household_type"])
    ]

    plot_weighted_means(
        descriptions["exact"]["regiostar4"],
        context="RegioStaR4",
        order=REGIOSTAR4_ORDER,
        labels=regiostar4_labels,
        title="Exact driver/car ratio by RegioStaR4",
        y_label="Weighted mean licensed drivers per car",
        path=OUTPUT_DIR / "driver_car_ratio_exact_by_regiostar4.png",
        note=(
            "Exact specification: Multi-car = exactly two cars; 3+ households excluded.\n"
            "Points show H_GEW-weighted means; whiskers show 95% HC3 confidence intervals; "
            "n denotes unweighted household observations."
        ),
    )
    plot_weighted_means(
        descriptions["exact"]["household_type"],
        context="household_type",
        order=observed_household_order,
        labels=household_labels,
        title="Exact driver/car ratio by household type",
        y_label="Weighted mean licensed drivers per car",
        path=OUTPUT_DIR / "driver_car_ratio_exact_by_household_type.png",
        note=(
            "Exact specification: Multi-car = exactly two cars; 3+ households excluded.\n"
            "Points show H_GEW-weighted means; whiskers show 95% HC3 confidence intervals; "
            "n denotes unweighted household observations."
        ),
    )
    plot_weighted_means(
        descriptions["capped"]["regiostar4"],
        context="RegioStaR4",
        order=REGIOSTAR4_ORDER,
        labels=regiostar4_labels,
        title="Capped-3 upper-bound driver/car ratio by RegioStaR4",
        y_label="Weighted mean upper-bound proxy",
        path=OUTPUT_DIR / "driver_car_ratio_capped3_by_regiostar4.png",
        note=(
            "Sensitivity includes 3+ households; licensed drivers / 3 is an upper-bound proxy, "
            "not an exact ratio for that group.\n"
            "Points show H_GEW-weighted means; whiskers show 95% HC3 confidence intervals; "
            "n denotes unweighted household observations."
        ),
    )

    print_console_summary(samples, descriptions, models)


if __name__ == "__main__":
    main()
