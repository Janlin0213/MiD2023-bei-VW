"""Theme 1: annual mileage of recorded vehicles in single- vs multi-car households.

The statistical unit is the recorded household vehicle. All descriptives and
models use raw positive ``A_GEW``. Inferential estimates are Gaussian identity-
link GEE marginal regressions with household-clustered robust covariance. The
analysis is observational and the 3+ household category is fleet-count top-coded:
detailed information is available for at most three vehicles per household.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
VEHICLE_PATH = ROOT / "data_processed" / "selected_raw" / "cars_selected_raw.csv"
BACKBONE_PATH = ROOT / "data_processed" / "eda" / "statistical_test" / "theme1_household_backbone.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda" / "statistical_tests" / "theme1_single_vs_multicar" / "03_annual_vehicle_mileage"

STATSMODELS_VERSION = "0.14.6"
METHOD = "A_GEW-weighted Gaussian marginal regression with identity link, estimated using GEE with household-clustered robust covariance"
CAR_ORDER = ["single_car", "multi_car"]
CAR_LABELS = {"single_car": "Single-car", "multi_car": "Multi-car"}
SPATIAL_ORDER = ["urban_core", "intermediate_urban", "small_town_rural"]
SPATIAL_LABELS = {"urban_core": "Urban core", "intermediate_urban": "Intermediate urban", "small_town_rural": "Small-town / rural"}
# Earlier Theme 1 modules explicitly use family households as the HH-type reference.
HOUSEHOLD_TYPE_ORDER = ["family_household", "young_household", "adult_household", "senior_household"]
HOUSEHOLD_TYPE_LABELS = {"family_household": "Family", "young_household": "Young", "adult_household": "Adult", "senior_household": "Senior"}
DETAIL_ORDER = ["one_car", "exactly_two_cars", "three_plus_cars"]
MILEAGE_CODES = list(range(1, 8))
MILEAGE_LABELS = {1: "<5k", 2: "5–<10k", 3: "10–<15k", 4: "15–<20k", 5: "20–<25k", 6: "25–<50k", 7: "50k+"}
MILEAGE_COLORS = ["#D9EAF0", "#B9D7E2", "#86B7C8", "#4C8C9B", "#E0B44C", "#D98245", "#9B5B56"]
Z95 = float(norm.ppf(0.975))
SMALL_CELL_THRESHOLD = 100
MATERIAL_CHANGE_RELATIVE = 0.20  # Flag a direction reversal or >20% change in the raw-scale estimate.

VEHICLE_COLUMNS = ["H_ID", "A_ID", "A_GEW", "H_ANZAUTO", "A_JAHRESFL", "jahresfl_gr"]
HOUSEHOLD_COLUMNS = ["H_ID", "H_ANZAUTO", "car_ownership_group", "RegioStaR7", "spatial_context_3", "household_type"]


@dataclass(frozen=True)
class GEEResult:
    name: str
    outcome: str
    predictor: str
    terms: list[str]
    beta: np.ndarray
    covariance: np.ndarray
    se: np.ndarray
    n_vehicles: int
    n_households: int
    weight_sum: float
    converged: bool
    iterations: int | None
    weights_verified: bool


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise KeyError(f"{source} is missing required column(s): {missing}")


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_csv(path, index=False)
    return path


def weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, q: float) -> float:
    """Return the first sorted value whose normalized cumulative weight reaches q."""
    if not 0 <= q <= 1:
        raise ValueError("q must lie in [0, 1].")
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return np.nan
    order = np.argsort(x[valid], kind="mergesort")
    xs, ws = x[valid][order], w[valid][order]
    cumulative = np.cumsum(ws) / ws.sum()
    return float(xs[min(np.searchsorted(cumulative, q, side="left"), len(xs) - 1)])


def validate_weighted_quantile() -> None:
    x = np.array([20.0, 0.0, 10.0])
    w = np.array([1.0, 1.0, 2.0])
    assert weighted_quantile(x, w, 0.25) == 0.0
    assert weighted_quantile(x, w, 0.50) == 10.0
    assert weighted_quantile(x, w, 0.90) == 20.0


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not VEHICLE_PATH.exists() or not BACKBONE_PATH.exists():
        raise FileNotFoundError(f"Missing accepted input: {VEHICLE_PATH} or {BACKBONE_PATH}")
    vh = pd.read_csv(VEHICLE_PATH, nrows=0).columns.tolist()
    hh = pd.read_csv(BACKBONE_PATH, nrows=0).columns.tolist()
    require_columns(vh, VEHICLE_COLUMNS, "vehicle source")
    require_columns(hh, HOUSEHOLD_COLUMNS, "household backbone")
    vehicles = pd.read_csv(VEHICLE_PATH, usecols=VEHICLE_COLUMNS, dtype={"H_ID": "string", "A_ID": "string"}, low_memory=False)
    households = pd.read_csv(BACKBONE_PATH, usecols=HOUSEHOLD_COLUMNS, dtype={"H_ID": "string"}, low_memory=False)
    for frame in (vehicles, households):
        frame["H_ID"] = frame["H_ID"].str.strip().replace("", pd.NA)
    vehicles["A_ID"] = vehicles["A_ID"].str.strip().replace("", pd.NA)
    if vehicles["H_ID"].isna().any():
        raise RuntimeError("Vehicle source contains missing H_ID values.")
    duplicates = vehicles.duplicated(["H_ID", "A_ID"], keep=False)
    if duplicates.any():
        print(vehicles.loc[duplicates].sort_values(["H_ID", "A_ID"]).head(20).to_string(index=False))
        raise RuntimeError("Unexpected duplicate H_ID + A_ID vehicle keys; no silent repair was made.")
    if households["H_ID"].isna().any() or households["H_ID"].duplicated().any():
        raise RuntimeError("Household backbone must contain one nonmissing row per H_ID.")
    return vehicles, households


def observed_mileage_qa(vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = numeric(vehicles["A_JAHRESFL"])
    valid = raw.between(0, 250000, inclusive="both")
    unexpected = raw.notna() & ~valid & ~raw.isin([999994, 999999])
    categories = [
        ("valid_0_to_250000", valid), ("zero_mileage", raw.eq(0)),
        ("special_999994_implausible", raw.eq(999994)), ("special_999999_no_answer", raw.eq(999999)),
        ("missing_or_nonnumeric", raw.isna()), ("other_unexpected", unexpected),
    ]
    rows = []
    for label, mask in categories:
        rows.append({"category": label, "count": int(mask.sum()), "share_of_source_rows": float(mask.mean()), "observed_values": ";".join(map(str, sorted(raw.loc[mask].dropna().unique())[:30])) if label != "valid_0_to_250000" else "0–250000 inclusive"})
    if unexpected.any():
        diagnostic = raw.loc[unexpected].value_counts().sort_index()
        raise RuntimeError(f"Unexpected A_JAHRESFL values require codebook resolution:\n{diagnostic.to_string()}")
    return pd.DataFrame(rows), raw, valid


def merge_sources(vehicles: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    source = vehicles.rename(columns={"H_ANZAUTO": "source_H_ANZAUTO"})
    merged = source.merge(households, on="H_ID", how="left", validate="many_to_one", indicator="vehicle_household_merge")
    if len(merged) != len(vehicles) or merged.duplicated(["H_ID", "A_ID"]).any():
        raise AssertionError("Household merge changed vehicle rows or duplicated vehicle keys.")
    both = merged["vehicle_household_merge"].eq("both")
    inconsistent = both & numeric(merged["source_H_ANZAUTO"]).ne(numeric(merged["H_ANZAUTO"]))
    if inconsistent.any():
        raise RuntimeError(f"Source/backbone H_ANZAUTO disagreement in {int(inconsistent.sum())} vehicle rows.")
    merged["A_GEW"] = numeric(merged["A_GEW"])
    merged["A_JAHRESFL"] = numeric(merged["A_JAHRESFL"])
    merged["H_ANZAUTO"] = numeric(merged["H_ANZAUTO"])
    merged["multi_car_dummy"] = merged["car_ownership_group"].map({"single_car": 0.0, "multi_car": 1.0})
    merged["car_count_detail"] = merged["H_ANZAUTO"].map({1: "one_car", 2: "exactly_two_cars", 3: "three_plus_cars"})
    derived_group = pd.cut(merged["A_JAHRESFL"], bins=[0, 5000, 10000, 15000, 20000, 25000, 50000, np.inf], labels=MILEAGE_CODES, right=False, include_lowest=True).astype("Int64")
    valid_raw_group = merged["A_JAHRESFL"].between(0, 250000) & numeric(merged["jahresfl_gr"]).isin(MILEAGE_CODES)
    inconsistent_group = valid_raw_group & numeric(merged["jahresfl_gr"]).ne(derived_group.astype(float))
    if inconsistent_group.any():
        diagnostic = merged.loc[inconsistent_group, ["H_ID", "A_ID", "A_JAHRESFL", "jahresfl_gr"]].copy()
        diagnostic["derived_annual_mileage_group"] = derived_group.loc[inconsistent_group]
        print(diagnostic.head(20).to_string(index=False))
        raise RuntimeError("jahresfl_gr is inconsistent with official groups derived from A_JAHRESFL.")
    # The accepted official group is reused after the raw-mileage consistency check.
    merged["annual_mileage_group"] = numeric(merged["jahresfl_gr"]).where(valid_raw_group).astype("Int64")
    return merged


def make_samples(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid_key = merged["H_ID"].notna() & merged["A_ID"].notna()
    matched = merged["vehicle_household_merge"].eq("both")
    weight = merged["A_GEW"].notna() & np.isfinite(merged["A_GEW"]) & merged["A_GEW"].gt(0)
    mileage = merged["A_JAHRESFL"].between(0, 250000, inclusive="both")
    ownership = merged["car_ownership_group"].isin(CAR_ORDER) & merged["multi_car_dummy"].isin([0, 1])
    primary = merged.loc[valid_key & matched & weight & mileage & ownership].copy()
    spatial = primary.loc[primary["spatial_context_3"].isin(SPATIAL_ORDER)].copy()
    hh_type = primary.loc[primary["household_type"].isin(HOUSEHOLD_TYPE_ORDER)].copy()
    exact = primary.loc[primary["H_ANZAUTO"].isin([1, 2])].copy()
    exact["exact_two_dummy"] = exact["H_ANZAUTO"].eq(2).astype(float)
    exact_spatial = exact.loc[exact["spatial_context_3"].isin(SPATIAL_ORDER)].copy()
    exact_hh = exact.loc[exact["household_type"].isin(HOUSEHOLD_TYPE_ORDER)].copy()
    for sample in [primary, spatial, hh_type, exact, exact_spatial, exact_hh]:
        assert sample["A_JAHRESFL"].between(0, 250000).all()
        assert np.isfinite(sample["A_GEW"]).all() and sample["A_GEW"].gt(0).all()
    assert primary["car_ownership_group"].isin(CAR_ORDER).all() and primary["multi_car_dummy"].isin([0, 1]).all()
    assert exact["H_ANZAUTO"].isin([1, 2]).all()
    return {"primary": primary, "spatial": spatial, "household_type": hh_type, "exact": exact, "exact_spatial": exact_spatial, "exact_household_type": exact_hh}


def weighted_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, cell in frame.groupby(grouper, observed=True, sort=False):
        keys = (keys,) if len(group_cols) == 1 else keys
        x, w = cell["A_JAHRESFL"], cell["A_GEW"]
        row = dict(zip(group_cols, keys))
        row.update({
            "unweighted_n_vehicles": len(cell), "unique_households": cell["H_ID"].nunique(), "weight_sum": float(w.sum()),
            "weighted_mean_km_per_year": float(np.average(x, weights=w)), "weighted_median_km_per_year": weighted_quantile(x, w, .5),
            "weighted_p25": weighted_quantile(x, w, .25), "weighted_p75": weighted_quantile(x, w, .75), "weighted_p90": weighted_quantile(x, w, .9), "weighted_p95": weighted_quantile(x, w, .95),
            "weighted_share_zero_mileage": float(w.loc[x.eq(0)].sum() / w.sum()), "minimum": float(x.min()), "maximum": float(x.max()),
            "n_above_50000": int(x.gt(50000).sum()), "n_above_100000": int(x.gt(100000).sum()), "n_above_200000": int(x.gt(200000).sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_distribution(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, cell in frame.groupby(grouper, observed=True, sort=False):
        keys = (keys,) if len(group_cols) == 1 else keys
        total = float(cell["A_GEW"].sum())
        for code in MILEAGE_CODES:
            part = cell.loc[cell["annual_mileage_group"].eq(code)]
            rows.append({**dict(zip(group_cols, keys)), "annual_mileage_group": code, "category_label": MILEAGE_LABELS[code], "unweighted_n": len(part), "weight_sum": float(part["A_GEW"].sum()), "weighted_share": float(part["A_GEW"].sum() / total)})
    result = pd.DataFrame(rows)
    sums = result.groupby(group_cols, observed=True)["weighted_share"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-10):
        raise AssertionError(f"Grouped mileage shares do not sum to one:\n{sums}")
    return result


def build_design(sample: pd.DataFrame, predictor: str, context: str | None = None, order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    p = sample[predictor].to_numpy(float)
    columns, terms = [np.ones(len(sample)), p], ["Intercept", predictor]
    if context:
        assert order is not None
        observed = set(sample[context].dropna().unique())
        if observed != set(order):
            raise RuntimeError(f"Unexpected/missing {context} levels: observed={sorted(observed)} expected={order}")
        cells = pd.crosstab(sample[context], p)
        missing = [(level, value) for level in order for value in [0.0, 1.0] if level not in cells.index or value not in cells.columns or cells.loc[level, value] == 0]
        if missing:
            raise RuntimeError(f"Empty expected {context} × ownership cells: {missing}")
        for level in order[1:]:
            d = sample[context].eq(level).to_numpy(float)
            columns.append(d); terms.append(f"C({context})[T.{level}]")
        for level in order[1:]:
            d = sample[context].eq(level).to_numpy(float)
            columns.append(p * d); terms.append(f"{predictor}:C({context})[T.{level}]")
    x = np.column_stack(columns)
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x) != x.shape[1]:
        raise RuntimeError(f"Invalid/rank-deficient design for {terms}")
    return x, terms


def fit_gee(sample: pd.DataFrame, outcome: str, predictor: str, name: str, context: str | None = None, order: list[str] | None = None) -> GEEResult:
    x, terms = build_design(sample, predictor, context, order)
    y = sample[outcome].to_numpy(float)
    w = sample["A_GEW"].to_numpy(float)
    groups, levels = pd.factorize(sample["H_ID"], sort=False)
    assert len(levels) == sample["H_ID"].nunique()
    model = sm.GEE(y, x, groups=groups, weights=w, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
    stored = np.asarray(model.weights, dtype=float)
    weights_verified = stored.shape == w.shape and np.array_equal(stored, w)
    if not weights_verified:
        raise AssertionError(f"Raw A_GEW vector not retained by GEE model {name}.")
    fitted = model.fit(cov_type="robust", maxiter=100)
    beta, se, covariance = map(lambda z: np.asarray(z, dtype=float), [fitted.params, fitted.bse, fitted.cov_params()])
    converged = bool(getattr(fitted, "converged", False))
    history = getattr(fitted, "fit_history", {}) or {}
    iterations = len(history.get("params", [])) or None
    if not converged or not (np.isfinite(beta).all() and np.isfinite(se).all() and np.isfinite(covariance).all()):
        raise RuntimeError(f"GEE failure/non-finite output in {name}; converged={converged}")
    if covariance.shape != (len(terms), len(terms)) or np.any(np.diag(covariance) < -1e-10):
        raise RuntimeError(f"Invalid robust covariance in {name}.")
    return GEEResult(name, outcome, predictor, terms, beta, covariance, se, len(sample), sample["H_ID"].nunique(), float(w.sum()), converged, iterations, weights_verified)


def coefficient_table(result: GEEResult) -> pd.DataFrame:
    statistic = np.divide(result.beta, result.se, out=np.full_like(result.beta, np.nan), where=result.se > 0)
    return pd.DataFrame({"model": result.name, "outcome": result.outcome, "term": result.terms, "estimate_km_per_year" if result.outcome == "A_JAHRESFL" else "estimate_log1p_scale": result.beta, "robust_se": result.se, "z_or_wald_statistic": statistic, "p_value": 2 * norm.sf(abs(statistic)), "ci95_low": result.beta - Z95 * result.se, "ci95_high": result.beta + Z95 * result.se, "n_vehicles": result.n_vehicles, "n_households": result.n_households, "weight_sum": result.weight_sum, "method": METHOD, "converged": result.converged, "iterations": result.iterations, "raw_A_GEW_verified": result.weights_verified})


def scenario_vector(result: GEEResult, predictor_value: float, context: str | None = None, level: str | None = None) -> np.ndarray:
    v = np.zeros(len(result.terms)); v[result.terms.index("Intercept")] = 1; v[result.terms.index(result.predictor)] = predictor_value
    if context and level:
        main, inter = f"C({context})[T.{level}]", f"{result.predictor}:C({context})[T.{level}]"
        if main in result.terms: v[result.terms.index(main)] = 1
        if predictor_value and inter in result.terms: v[result.terms.index(inter)] = 1
    return v


def linear_estimate(result: GEEResult, vector: np.ndarray) -> tuple[float, float, float, float, float]:
    est = float(vector @ result.beta); variance = float(vector @ result.covariance @ vector)
    if variance < -1e-10: raise RuntimeError(f"Negative linear-combination variance in {result.name}")
    se = float(np.sqrt(max(0.0, variance))); low, high = est - Z95 * se, est + Z95 * se
    p = float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan
    return est, se, low, high, p


def overall_estimates(result: GEEResult, sample: pd.DataFrame) -> pd.DataFrame:
    single = scenario_vector(result, 0); multi = scenario_vector(result, 1); contrast = multi - single
    group_one_label = "Multi-car" if result.predictor == "multi_car_dummy" else "Exact-two"
    difference_label = "Multi minus Single difference" if result.predictor == "multi_car_dummy" else "Exact-two minus Single difference"
    rows = []
    for label, vector in [("Single-car predicted mean", single), (f"{group_one_label} predicted mean", multi), (difference_label, contrast)]:
        est, se, low, high, p = linear_estimate(result, vector)
        rows.append({"result": label, "estimate": est, "robust_se": se, "ci95_low": low, "ci95_high": high, "p_value": p, "n_vehicles": len(sample), "n_households": sample.H_ID.nunique(), "weight_sum": sample.A_GEW.sum(), "method": METHOD, "raw_A_GEW_verified": result.weights_verified})
    return pd.DataFrame(rows)


def joint_wald(result: GEEResult, context: str, order: list[str]) -> dict[str, Any]:
    terms = [f"{result.predictor}:C({context})[T.{level}]" for level in order[1:]]
    idx = [result.terms.index(t) for t in terms]; b = result.beta[idx]; cov = result.covariance[np.ix_(idx, idx)]
    if np.linalg.matrix_rank(cov) != len(idx): raise RuntimeError(f"Rank-deficient interaction covariance in {result.name}")
    statistic = float(b.T @ np.linalg.solve(cov, b))
    return {"test_name": f"Joint ownership × {context} interaction", "context_variable": context, "wald_statistic": statistic, "degrees_of_freedom": len(idx), "p_value": float(chi2.sf(statistic, len(idx))), "model": result.name, "n_vehicles": result.n_vehicles, "n_households": result.n_households, "weight_sum": result.weight_sum, "method": "Joint Wald chi-square test using GEE robust covariance"}


def context_estimates(result: GEEResult, sample: pd.DataFrame, context: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for level in order:
        single, multi = scenario_vector(result, 0, context, level), scenario_vector(result, 1, context, level)
        s = linear_estimate(result, single); m = linear_estimate(result, multi); d = linear_estimate(result, multi - single)
        if not np.isclose(m[0] - s[0], d[0], atol=1e-8): raise AssertionError("Context contrast validation failed.")
        cell = sample.loc[sample[context].eq(level)]
        rows.append({context: level, "predicted_single_mean": s[0], "single_robust_se": s[1], "single_ci95_low": s[2], "single_ci95_high": s[3], "predicted_multi_mean": m[0], "multi_robust_se": m[1], "multi_ci95_low": m[2], "multi_ci95_high": m[3], "difference_multi_minus_single": d[0], "difference_robust_se": d[1], "difference_ci95_low": d[2], "difference_ci95_high": d[3], "difference_p_value": d[4], "relative_difference_percent": 100 * d[0] / s[0] if s[0] != 0 else np.nan, "n_vehicles_single": int((cell[result.predictor] == 0).sum()), "n_vehicles_multi": int((cell[result.predictor] == 1).sum()), "n_households": cell.H_ID.nunique(), "method": METHOD, "raw_A_GEW_verified": result.weights_verified})
    return pd.DataFrame(rows)


def validate_overall_model(result: GEEResult, summary: pd.DataFrame) -> None:
    model_values = {
        "Single-car predicted mean": linear_estimate(result, scenario_vector(result, 0))[0],
        "Multi-car predicted mean": linear_estimate(result, scenario_vector(result, 1))[0],
    }
    direct = summary.set_index("car_ownership_group")["weighted_mean_km_per_year"]
    if not (np.isclose(model_values["Single-car predicted mean"], direct["single_car"], atol=1e-6) and np.isclose(model_values["Multi-car predicted mean"], direct["multi_car"], atol=1e-6)):
        raise AssertionError(f"Overall GEE means disagree with direct weighted means: model={model_values}, direct={direct.to_dict()}")


def plot_distribution(distribution: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); bottom = np.zeros(2); x = np.arange(2)
    for code, color in zip(MILEAGE_CODES, MILEAGE_COLORS):
        vals = [float(distribution.loc[(distribution.car_ownership_group == car) & (distribution.annual_mileage_group == code), "weighted_share"].iloc[0]) for car in CAR_ORDER]
        ax.bar(x, vals, bottom=bottom, width=.62, color=color, edgecolor="white", linewidth=.6, label=MILEAGE_LABELS[code]); bottom += vals
    ax.set_xticks(x, [CAR_LABELS[c] for c in CAR_ORDER]); ax.set_ylabel("Weighted share of recorded vehicles"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylim(0, 1.04); ax.grid(axis="y", color="#dddddd"); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False); ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(OUTPUT_DIR / f"annual_vehicle_mileage_distribution_overall.{ext}", dpi=220 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def plot_context(estimates: pd.DataFrame, context: str, order: list[str], labels: dict[str, str], stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(order)); offset = .13
    for car, sign, color in [("single", -1, "#2F5D7C"), ("multi", 1, "#D98245")]:
        mean = estimates[f"predicted_{car}_mean"].to_numpy(); low = estimates[f"{car}_ci95_low"].to_numpy(); high = estimates[f"{car}_ci95_high"].to_numpy()
        ax.errorbar(x + sign * offset, mean, yerr=np.vstack([mean-low, high-mean]), fmt="o", color=color, capsize=3, linewidth=1.4, label="Single-car" if car == "single" else "Multi-car")
    ax.set_xticks(x, [labels[v] for v in order]); ax.set_ylabel("Predicted mean annual mileage [km/year]"); ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}")); ax.grid(axis="y", color="#dddddd"); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False); ax.legend(frameon=False); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", dpi=220 if ext == "png" else None, facecolor="white")
    plt.close(fig)


# Scientific distribution figures. The fixed cap applies only to the raw jitter
# display layer; weighted boxes and saved GEE estimates always use the complete
# validated analytical cell.
FIGURE_COLORS = {"single_car": "#2F5D7C", "multi_car": "#D98245"}
FIGURE_POINT_SEED = 20260818
JITTER_MAX_PER_CELL = 2500
FIGURE_NOTE = (
    "Boxes: A_GEW-weighted quartiles; diamonds/whiskers: weighted GEE means and 95% CIs.\n"
    "Points: deterministic raw-data display subsample; valid values above the range are counted."
)
MAIN_FIGURE_NOTE = "Points and whiskers show A_GEW-weighted GEE means and 95% CIs."


def weighted_box_statistics(cell: pd.DataFrame) -> dict[str, float]:
    """Build weighted Tukey-style box statistics with observed-value whiskers."""
    values, weights = cell["A_JAHRESFL"], cell["A_GEW"]
    q1 = weighted_quantile(values, weights, 0.25)
    median = weighted_quantile(values, weights, 0.50)
    q3 = weighted_quantile(values, weights, 0.75)
    if not (np.isfinite([q1, median, q3]).all() and q1 <= median <= q3):
        raise AssertionError(f"Invalid weighted quartiles: q1={q1}, median={median}, q3={q3}")
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    observed = values.to_numpy(float)
    lower_candidates = observed[observed >= lower_fence]
    upper_candidates = observed[observed <= upper_fence]
    if len(lower_candidates) == 0 or len(upper_candidates) == 0:
        raise AssertionError("Weighted box fences contain no observed whisker candidates.")
    whislo, whishi = float(lower_candidates.min()), float(upper_candidates.max())
    if not (whislo <= q1 <= median <= q3 <= whishi):
        raise AssertionError(
            f"Invalid weighted box ordering: {(whislo, q1, median, q3, whishi)}"
        )
    return {
        "whislo": whislo,
        "q1": q1,
        "med": median,
        "q3": q3,
        "whishi": whishi,
        "fliers": [],
    }


def deterministic_jitter_sample(
    visible_frame: pd.DataFrame,
    group_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    """Fixed-seed, cell-balanced sample for raw jitter display only."""
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, cell in visible_frame.groupby(grouper, observed=True, sort=False):
        index = cell.index.to_numpy()
        if len(index) > JITTER_MAX_PER_CELL:
            index = rng.choice(index, size=JITTER_MAX_PER_CELL, replace=False)
        selected.append(index)
    if not selected:
        raise RuntimeError("No visible observations available for the jitter layer.")
    return visible_frame.loc[np.concatenate(selected)].copy()


def validate_figure_cell(cell: pd.DataFrame, label: str) -> None:
    if not cell["A_JAHRESFL"].between(0, 250000, inclusive="both").all():
        raise AssertionError(f"Invalid annual-mileage value in plotting cell {label}.")
    if set(cell["car_ownership_group"].dropna().unique()) != set(CAR_ORDER):
        raise RuntimeError(f"Plotting cell {label} does not contain Single and Multi.")


def sensible_y_ticks(display_ymax: float) -> np.ndarray:
    if display_ymax <= 75000:
        interval = 10000
    elif display_ymax <= 125000:
        interval = 25000
    else:
        interval = 50000
    return np.arange(0, display_ymax + 0.1, interval, dtype=float)


def draw_scientific_mileage_panel(
    ax: plt.Axes,
    full_cell: pd.DataFrame,
    jitter_cell: pd.DataFrame,
    estimates: dict[str, tuple[float, float, float]],
    display_ymax: float,
    seed: int,
) -> dict[str, int]:
    """Draw jitter, weighted boxes, GEE mean/CIs, and upper-range counts."""
    validate_figure_cell(full_cell, ax.get_title() or "panel")
    positions = np.arange(len(CAR_ORDER), dtype=float)
    rng = np.random.default_rng(seed)
    above_counts: dict[str, int] = {}

    # Background layer: actual raw mileage values within the visible range only.
    for position, car in zip(positions, CAR_ORDER):
        points = jitter_cell.loc[
            jitter_cell["car_ownership_group"].eq(car), "A_JAHRESFL"
        ].to_numpy(float)
        if len(points) == 0:
            raise RuntimeError(f"Empty jitter display cell for {car}.")
        ax.scatter(
            position + rng.uniform(-0.15, 0.15, size=len(points)),
            points,
            s=4,
            alpha=0.075,
            color=FIGURE_COLORS[car],
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

    # Middle layer: explicitly supplied A_GEW-weighted quartiles and real-value whiskers.
    for position, car in zip(positions, CAR_ORDER):
        group = full_cell.loc[full_cell["car_ownership_group"].eq(car)]
        stats = weighted_box_statistics(group)
        artists = ax.bxp(
            [stats],
            positions=[position],
            widths=0.50,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            boxprops={
                "facecolor": FIGURE_COLORS[car],
                "edgecolor": FIGURE_COLORS[car],
                "alpha": 0.25,
                "linewidth": 1.4,
            },
            medianprops={"color": "#202020", "linewidth": 1.7},
            whiskerprops={"color": FIGURE_COLORS[car], "linewidth": 1.15},
            capprops={"color": FIGURE_COLORS[car], "linewidth": 1.15},
            zorder=3,
        )
        for collection in artists.values():
            for artist in collection:
                artist.set_zorder(3)

        # Foreground layer: saved A_GEW-weighted household-clustered GEE estimate.
        mean, low, high = estimates[car]
        if not (np.isfinite([mean, low, high]).all() and low <= mean <= high):
            raise AssertionError(f"Invalid saved mean/CI overlay for {car}: {(mean, low, high)}")
        ax.errorbar(
            position,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            markersize=7.2,
            markerfacecolor=FIGURE_COLORS[car],
            markeredgecolor="white",
            markeredgewidth=1.0,
            ecolor=FIGURE_COLORS[car],
            elinewidth=2.0,
            capsize=4,
            capthick=1.6,
            zorder=5,
        )

        count = int(group["A_JAHRESFL"].gt(display_ymax).sum())
        above_counts[car] = count
        if count > 0:
            ax.text(
                position,
                display_ymax * 0.965,
                f"↑ n={count:,} above range",
                ha="center",
                va="top",
                fontsize=7.4,
                color=FIGURE_COLORS[car],
                fontweight="bold",
                zorder=6,
            )

    ax.set_xticks(positions, [CAR_LABELS[car] for car in CAR_ORDER])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, display_ymax)
    ax.set_yticks(sensible_y_ticks(display_ymax))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return above_counts


def load_scientific_figure_inputs() -> tuple[
    dict[str, tuple[float, float, float]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Read saved summaries and validated GEE overlays without fitting models."""
    paths = {
        "overall_model": OUTPUT_DIR / "07_model_overall.csv",
        "spatial_model": OUTPUT_DIR / "09_spatial_multicar_contrasts.csv",
        "household_model": OUTPUT_DIR / "11_household_type_multicar_contrasts.csv",
        "overall_summary": OUTPUT_DIR / "01_weighted_summary_overall.csv",
        "spatial_summary": OUTPUT_DIR / "03_weighted_summary_spatial.csv",
        "household_summary": OUTPUT_DIR / "05_weighted_summary_household_type.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing validated figure inputs: {missing}")
    overall_table = pd.read_csv(paths["overall_model"])
    overall: dict[str, tuple[float, float, float]] = {}
    for car, label in [
        ("single_car", "Single-car predicted mean"),
        ("multi_car", "Multi-car predicted mean"),
    ]:
        row = overall_table.loc[overall_table["result"].eq(label)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one saved estimate for {label}.")
        record = row.iloc[0]
        overall[car] = (
            float(record["estimate_km_per_year"]),
            float(record["ci95_low"]),
            float(record["ci95_high"]),
        )
    spatial_model = pd.read_csv(paths["spatial_model"]).set_index("spatial_context_3").loc[SPATIAL_ORDER].reset_index()
    household_model = pd.read_csv(paths["household_model"]).set_index("household_type").loc[HOUSEHOLD_TYPE_ORDER].reset_index()
    return (
        overall,
        spatial_model,
        household_model,
        pd.read_csv(paths["overall_summary"]),
        pd.read_csv(paths["spatial_summary"]),
        pd.read_csv(paths["household_summary"]),
    )


def context_mean_ci(row: pd.Series) -> dict[str, tuple[float, float, float]]:
    return {
        "single_car": (
            float(row["predicted_single_mean"]),
            float(row["single_ci95_low"]),
            float(row["single_ci95_high"]),
        ),
        "multi_car": (
            float(row["predicted_multi_mean"]),
            float(row["multi_ci95_low"]),
            float(row["multi_ci95_high"]),
        ),
    }


def validate_main_figure_inputs(
    overall: dict[str, tuple[float, float, float]],
    spatial: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    """Validate completeness and CI ordering in the saved GEE figure inputs."""
    if set(overall) != set(CAR_ORDER):
        raise RuntimeError("Overall main-figure input must contain Single and Multi estimates.")
    for label, estimates in overall.items():
        mean, low, high = estimates
        if not (np.isfinite([mean, low, high]).all() and low <= mean <= high):
            raise AssertionError(f"Invalid overall saved mean/CI for {label}: {estimates}")
    for frame, context, order in [
        (spatial, "spatial_context_3", SPATIAL_ORDER),
        (household, "household_type", HOUSEHOLD_TYPE_ORDER),
    ]:
        if frame[context].tolist() != order:
            raise RuntimeError(f"Saved {context} figure rows are missing, duplicated, or out of order.")
        for row in frame.itertuples(index=False):
            estimates = context_mean_ci(pd.Series(row._asdict()))
            for car, values in estimates.items():
                mean, low, high = values
                if not (np.isfinite([mean, low, high]).all() and low <= mean <= high):
                    raise AssertionError(
                        f"Invalid saved mean/CI for {context}, {getattr(row, context)}, {car}: {values}"
                    )


def mean_ci_y_limits(
    estimate_sets: list[dict[str, tuple[float, float, float]]],
) -> tuple[float, float]:
    """Return a readable, rounded linear range spanning every saved 95% CI."""
    lows = [values[1] for estimates in estimate_sets for values in estimates.values()]
    highs = [values[2] for estimates in estimate_sets for values in estimates.values()]
    lower, upper = float(min(lows)), float(max(highs))
    span = upper - lower
    if not (np.isfinite([lower, upper, span]).all() and span > 0):
        raise AssertionError(f"Cannot derive main-figure y range from CI bounds: {(lower, upper)}")
    margin = max(400.0, 0.12 * span)
    rounded_lower = max(0.0, np.floor((lower - margin) / 500.0) * 500.0)
    rounded_upper = np.ceil((upper + margin) / 500.0) * 500.0
    if rounded_upper <= rounded_lower:
        raise AssertionError("Derived main-figure y-axis range is empty.")
    return float(rounded_lower), float(rounded_upper)


def draw_mean_ci_panel(
    ax: plt.Axes,
    estimates: dict[str, tuple[float, float, float]],
    y_limits: tuple[float, float],
    label_means: bool = False,
) -> None:
    """Draw one paired Single/Multi GEE mean-and-CI panel."""
    positions = np.arange(len(CAR_ORDER), dtype=float)
    means = np.array([estimates[car][0] for car in CAR_ORDER], dtype=float)
    ax.plot(positions, means, color="#A6A6A6", linewidth=1.1, zorder=1)
    markers = {"single_car": "o", "multi_car": "D"}
    for position, car in zip(positions, CAR_ORDER):
        mean, low, high = estimates[car]
        ax.errorbar(
            position,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt=markers[car],
            markersize=8.0,
            markerfacecolor=FIGURE_COLORS[car],
            markeredgecolor="white",
            markeredgewidth=1.1,
            ecolor=FIGURE_COLORS[car],
            elinewidth=2.0,
            capsize=5,
            capthick=1.5,
            zorder=3,
        )
        if label_means:
            ax.annotate(
                f"{mean / 1000:.1f}k",
                (position, high),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=FIGURE_COLORS[car],
            )
    ax.set_xticks(positions, [CAR_LABELS[car] for car in CAR_ORDER])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_main_mean_ci_overall(
    estimates: dict[str, tuple[float, float, float]],
) -> None:
    y_limits = mean_ci_y_limits([estimates])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    draw_mean_ci_panel(ax, estimates, y_limits, label_means=True)
    ax.set_title("Annual Vehicle Mileage - Overall", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Household car-ownership group")
    ax.set_ylabel("Annual vehicle mileage (km/year)")
    fig.text(0.5, 0.018, MAIN_FIGURE_NOTE, ha="center", va="bottom", fontsize=7.6, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_scientific_figure(fig, "annual_vehicle_mileage_mean_ci_overall")


def plot_main_mean_ci_household_type(saved_model: pd.DataFrame) -> None:
    estimate_sets = [
        context_mean_ci(saved_model.loc[saved_model["household_type"].eq(level)].iloc[0])
        for level in HOUSEHOLD_TYPE_ORDER
    ]
    y_limits = mean_ci_y_limits(estimate_sets)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharey=True)
    for ax, level, estimates in zip(axes.flat, HOUSEHOLD_TYPE_ORDER, estimate_sets):
        draw_mean_ci_panel(ax, estimates, y_limits)
        ax.set_title(
            f"{HOUSEHOLD_TYPE_LABELS[level]} household",
            loc="left",
            fontsize=10.8,
            fontweight="bold",
        )
    fig.suptitle("Annual Vehicle Mileage by Household Type", x=0.055, ha="left", fontsize=13, fontweight="bold")
    fig.supylabel("Annual vehicle mileage (km/year)", x=0.012, fontsize=10.5)
    fig.supxlabel("Household car-ownership group", y=0.072, fontsize=10.5)
    fig.text(0.5, 0.015, MAIN_FIGURE_NOTE, ha="center", va="bottom", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0.035, 0.105, 1, 0.94))
    save_scientific_figure(fig, "annual_vehicle_mileage_mean_ci_household_type")


def plot_main_mean_ci_spatial(saved_model: pd.DataFrame) -> None:
    estimate_sets = [
        context_mean_ci(saved_model.loc[saved_model["spatial_context_3"].eq(level)].iloc[0])
        for level in SPATIAL_ORDER
    ]
    y_limits = mean_ci_y_limits(estimate_sets)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    for ax, level, estimates in zip(axes, SPATIAL_ORDER, estimate_sets):
        draw_mean_ci_panel(ax, estimates, y_limits)
        ax.set_title(SPATIAL_LABELS[level], loc="left", fontsize=10.8, fontweight="bold")
    fig.suptitle("Annual Vehicle Mileage by Spatial Context", x=0.045, ha="left", fontsize=13, fontweight="bold")
    fig.supylabel("Annual vehicle mileage (km/year)", x=0.006, fontsize=10.5)
    fig.supxlabel("Household car-ownership group", y=0.085, fontsize=10.5)
    fig.text(0.5, 0.018, MAIN_FIGURE_NOTE, ha="center", va="bottom", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0.02, 0.13, 1, 0.92))
    save_scientific_figure(fig, "annual_vehicle_mileage_mean_ci_spatial")


def main_result_figures_only() -> None:
    """Regenerate primary mean/CI figures from validated saved GEE outputs only."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall, spatial, household, _, _, _ = load_scientific_figure_inputs()
    validate_main_figure_inputs(overall, spatial, household)
    plot_main_mean_ci_overall(overall)
    plot_main_mean_ci_household_type(household)
    plot_main_mean_ci_spatial(spatial)
    print("\nANNUAL VEHICLE MILEAGE - MAIN FIGURES UPDATED")
    print("\nMain figures generated:")
    print("- overall weighted GEE mean + 95% CI")
    print("- household-type weighted GEE mean + 95% CI")
    print("- spatial weighted GEE mean + 95% CI")
    print("\nOptional diagnostic box/jitter figures:\nleft unchanged")
    print("\nMean/CI source:")
    print("existing A_GEW-weighted household-clustered GEE outputs")
    print("07_model_overall.csv")
    print("09_spatial_multicar_contrasts.csv")
    print("11_household_type_multicar_contrasts.csv")
    print("\nStatistical logic changed:\nNO")
    print("\nSaved:")
    for stem in [
        "annual_vehicle_mileage_mean_ci_overall",
        "annual_vehicle_mileage_mean_ci_household_type",
        "annual_vehicle_mileage_mean_ci_spatial",
    ]:
        print(f"- {stem}.png")
        print(f"- {stem}.pdf")


def validate_weighted_boxes_against_saved(
    sample: pd.DataFrame,
    saved: pd.DataFrame,
    context: str | None = None,
    order: list[str] | None = None,
) -> None:
    levels: list[str | None] = [None] if context is None else list(order or [])
    for level in levels:
        context_cell = sample if context is None else sample.loc[sample[context].eq(level)]
        for car in CAR_ORDER:
            cell = context_cell.loc[context_cell["car_ownership_group"].eq(car)]
            box = weighted_box_statistics(cell)
            row = saved.loc[saved["car_ownership_group"].eq(car)]
            if context is not None:
                row = row.loc[row[context].eq(level)]
            if len(row) != 1:
                raise RuntimeError(f"Expected one saved weighted-summary row for {context}, {level}, {car}.")
            record = row.iloc[0]
            expected = np.array([
                record["weighted_p25"],
                record["weighted_median_km_per_year"],
                record["weighted_p75"],
            ], dtype=float)
            actual = np.array([box["q1"], box["med"], box["q3"]], dtype=float)
            if not np.allclose(actual, expected, atol=1e-9):
                raise AssertionError(
                    f"Weighted box/saved summary mismatch for {context}, {level}, {car}: "
                    f"actual={actual}, expected={expected}"
                )


def save_scientific_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ["png", "pdf"]:
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            dpi=240 if extension == "png" else None,
            facecolor="white",
        )
    plt.close(fig)


def plot_scientific_overall(
    primary: pd.DataFrame,
    estimates: dict[str, tuple[float, float, float]],
    display_ymax: float,
) -> tuple[int, dict[str, int]]:
    visible = primary.loc[primary["A_JAHRESFL"].le(display_ymax)]
    points = deterministic_jitter_sample(visible, ["car_ownership_group"], FIGURE_POINT_SEED)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    above = draw_scientific_mileage_panel(
        ax, primary, points, estimates, display_ymax, FIGURE_POINT_SEED
    )
    ax.set_title("Annual Vehicle Mileage — Overall", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Household car-ownership group")
    ax.set_ylabel("Annual vehicle mileage (km/year)")
    fig.text(0.5, 0.012, FIGURE_NOTE, ha="center", va="bottom", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_scientific_figure(fig, "annual_vehicle_mileage_box_jitter_mean_overall")
    return len(points), above


def plot_scientific_household_type(
    sample: pd.DataFrame,
    saved_model: pd.DataFrame,
    display_ymax: float,
) -> int:
    visible = sample.loc[sample["A_JAHRESFL"].le(display_ymax)]
    points = deterministic_jitter_sample(
        visible,
        ["household_type", "car_ownership_group"],
        FIGURE_POINT_SEED + 1,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5), sharey=True)
    for panel, (ax, level) in enumerate(zip(axes.flat, HOUSEHOLD_TYPE_ORDER)):
        full_cell = sample.loc[sample["household_type"].eq(level)]
        point_cell = points.loc[points["household_type"].eq(level)]
        row = saved_model.loc[saved_model["household_type"].eq(level)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one saved household-type mean/CI row for {level}.")
        draw_scientific_mileage_panel(
            ax,
            full_cell,
            point_cell,
            context_mean_ci(row.iloc[0]),
            display_ymax,
            FIGURE_POINT_SEED + 10 + panel,
        )
        ax.set_title(f"{HOUSEHOLD_TYPE_LABELS[level]} household", loc="left", fontsize=10.5, fontweight="bold")
    fig.supylabel("Annual vehicle mileage (km/year)", x=0.012, fontsize=10.5)
    fig.supxlabel("Household car-ownership group", y=0.082, fontsize=10.5)
    fig.text(0.5, 0.012, FIGURE_NOTE, ha="center", va="bottom", fontsize=7.1, color="#555555")
    fig.tight_layout(rect=(0.028, 0.12, 1, 1))
    save_scientific_figure(fig, "annual_vehicle_mileage_box_jitter_mean_household_type")
    return len(points)


def plot_scientific_spatial(
    sample: pd.DataFrame,
    saved_model: pd.DataFrame,
    display_ymax: float,
) -> int:
    visible = sample.loc[sample["A_JAHRESFL"].le(display_ymax)]
    points = deterministic_jitter_sample(
        visible,
        ["spatial_context_3", "car_ownership_group"],
        FIGURE_POINT_SEED + 2,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7), sharey=True)
    for panel, (ax, level) in enumerate(zip(axes, SPATIAL_ORDER)):
        full_cell = sample.loc[sample["spatial_context_3"].eq(level)]
        point_cell = points.loc[points["spatial_context_3"].eq(level)]
        row = saved_model.loc[saved_model["spatial_context_3"].eq(level)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one saved spatial mean/CI row for {level}.")
        draw_scientific_mileage_panel(
            ax,
            full_cell,
            point_cell,
            context_mean_ci(row.iloc[0]),
            display_ymax,
            FIGURE_POINT_SEED + 20 + panel,
        )
        ax.set_title(SPATIAL_LABELS[level], loc="left", fontsize=10.5, fontweight="bold")
    fig.supylabel("Annual vehicle mileage (km/year)", x=0.006, fontsize=10.5)
    fig.supxlabel("Household car-ownership group", y=0.09, fontsize=10.5)
    fig.text(0.5, 0.012, FIGURE_NOTE, ha="center", va="bottom", fontsize=7.1, color="#555555")
    fig.tight_layout(rect=(0.018, 0.135, 1, 1))
    save_scientific_figure(fig, "annual_vehicle_mileage_box_jitter_mean_spatial")
    return len(points)


def scientific_figures_only_main() -> None:
    """Generate the requested figures without fitting or rewriting analyses."""
    assert statsmodels.__version__ == STATSMODELS_VERSION
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vehicles, households = read_sources()
    samples = make_samples(merge_sources(vehicles, households))
    primary, spatial, household = samples["primary"], samples["spatial"], samples["household_type"]
    weighted_p99 = weighted_quantile(primary["A_JAHRESFL"], primary["A_GEW"], 0.99)
    display_ymax = float(max(50000, np.ceil(weighted_p99 / 5000) * 5000))
    if display_ymax >= 250000:
        raise AssertionError("P99-derived display window did not create a distribution-focused zoom.")

    (
        overall_model,
        spatial_model,
        household_model,
        overall_summary,
        spatial_summary,
        household_summary,
    ) = load_scientific_figure_inputs()
    validate_weighted_boxes_against_saved(primary, overall_summary)
    validate_weighted_boxes_against_saved(
        spatial, spatial_summary, "spatial_context_3", SPATIAL_ORDER
    )
    validate_weighted_boxes_against_saved(
        household, household_summary, "household_type", HOUSEHOLD_TYPE_ORDER
    )
    validate_figure_cell(primary, "overall")
    for context, sample, order in [
        ("spatial_context_3", spatial, SPATIAL_ORDER),
        ("household_type", household, HOUSEHOLD_TYPE_ORDER),
    ]:
        for level in order:
            validate_figure_cell(sample.loc[sample[context].eq(level)], f"{context}={level}")

    _, overall_above = plot_scientific_overall(primary, overall_model, display_ymax)
    plot_scientific_household_type(household, household_model, display_ymax)
    plot_scientific_spatial(spatial, spatial_model, display_ymax)
    print("\nANNUAL VEHICLE MILEAGE - SCIENTIFIC FIGURES COMPLETE")
    print(f"\nWeighted P99:\n{weighted_p99:,.0f} km/year")
    print(f"\nCommon display upper limit:\n{display_ymax:,.0f} km/year")
    print("\nWeighted boxes:\nyes")
    print("\nMean/95% CI source:\nexisting A_GEW-weighted household-clustered GEE outputs")
    print("\nJitter:\ndeterministic display subsample")
    print(f"maximum displayed points per cell: {JITTER_MAX_PER_CELL:,}")
    print("\nAbove-range observations:")
    print(f"overall Single: {overall_above['single_car']:,}")
    print(f"overall Multi: {overall_above['multi_car']:,}")
    print("\nGenerated:")
    print("annual_vehicle_mileage_box_jitter_mean_overall")
    print("annual_vehicle_mileage_box_jitter_mean_household_type")
    print("annual_vehicle_mileage_box_jitter_mean_spatial")
    print("\nStatistical logic changed:\nNO")


def qa_row(section: str, metric: str, count: int | float, denominator: int | None, denominator_description: str, notes: str = "") -> dict[str, Any]:
    return {"section": section, "metric": metric, "count": count, "share": np.nan if not denominator else float(count) / denominator, "denominator_description": denominator_description, "notes": notes}


def build_fleet_qa(merged: pd.DataFrame) -> pd.DataFrame:
    matched = merged.loc[merged.vehicle_household_merge.eq("both")].copy()
    hh = matched.groupby(["H_ID", "H_ANZAUTO"], as_index=False).agg(n_recorded_vehicle_rows=("A_ID", "size"), n_valid_mileage_vehicle_rows=("A_JAHRESFL", lambda x: int(numeric(x).between(0, 250000).sum())))
    rows = []
    for count in [1, 2, 3]:
        label = {1: "one_car", 2: "exactly_two_cars", 3: "three_plus_cars"}[count]
        cell = hh.loc[hh.H_ANZAUTO.eq(count)]
        for (recorded, valid), n in cell.groupby(["n_recorded_vehicle_rows", "n_valid_mileage_vehicle_rows"]).size().items():
            rows.append({"H_ANZAUTO": count, "car_count_detail": label, "n_recorded_vehicle_rows": recorded, "n_valid_mileage_vehicle_rows": valid, "n_households": n, "notes": "H_ANZAUTO=3 means 3+; at most three detailed vehicle rows are available, so three rows do not prove exactly three owned cars." if count == 3 else ""})
    return pd.DataFrame(rows)


def exact_two_coverage(merged: pd.DataFrame) -> pd.DataFrame:
    cell = merged.loc[merged.vehicle_household_merge.eq("both") & merged.H_ANZAUTO.eq(2)]
    hh = cell.groupby("H_ID").agg(n_recorded=("A_ID", "size"), n_valid=("A_JAHRESFL", lambda x: int(numeric(x).between(0, 250000).sum())))
    state = np.select([hh.n_valid.ge(2), hh.n_valid.eq(1)], ["both_mileages_valid", "one_mileage_valid"], default="no_mileage_valid")
    return pd.Series(state).value_counts().reindex(["both_mileages_valid", "one_mileage_valid", "no_mileage_valid"], fill_value=0).rename_axis("coverage").reset_index(name="n_households")


def build_sample_qa(vehicles: pd.DataFrame, merged: pd.DataFrame, samples: dict[str, pd.DataFrame], coverage: pd.DataFrame) -> pd.DataFrame:
    n = len(vehicles); raw_mileage = numeric(vehicles.A_JAHRESFL); weight = numeric(vehicles.A_GEW); matched = merged.vehicle_household_merge.eq("both")
    rows = [qa_row("source", "source vehicle rows", n, n, "source vehicle rows"), qa_row("source", "unique vehicle keys H_ID + A_ID", vehicles[["H_ID", "A_ID"]].drop_duplicates().shape[0], n, "source vehicle rows"), qa_row("source", "unique source households", vehicles.H_ID.nunique(), vehicles.H_ID.nunique(), "unique source households"), qa_row("merge", "matched vehicle rows", int(matched.sum()), n, "source vehicle rows"), qa_row("merge", "unmatched vehicle rows", int((~matched).sum()), n, "source vehicle rows", f"unmatched household IDs={merged.loc[~matched, 'H_ID'].nunique()}")]
    for aid in ["1", "2", "3"]: rows.append(qa_row("vehicle identity", f"A_ID == {aid}", int(vehicles.A_ID.eq(aid).sum()), n, "source vehicle rows"))
    rows.append(qa_row("vehicle identity", "unexpected A_ID", int((~vehicles.A_ID.isin(["1", "2", "3"])).sum()), n, "source vehicle rows", ";".join(sorted(vehicles.loc[~vehicles.A_ID.isin(["1", "2", "3"]), "A_ID"].dropna().unique()))))
    mileage_metrics = [("A_JAHRESFL valid 0–250000", raw_mileage.between(0,250000)), ("A_JAHRESFL == 0", raw_mileage.eq(0)), ("A_JAHRESFL == 999994", raw_mileage.eq(999994)), ("A_JAHRESFL == 999999", raw_mileage.eq(999999)), ("missing A_JAHRESFL", raw_mileage.isna()), ("unexpected A_JAHRESFL", raw_mileage.notna() & ~raw_mileage.between(0,250000) & ~raw_mileage.isin([999994,999999]))]
    for metric, mask in mileage_metrics: rows.append(qa_row("mileage", metric, int(mask.sum()), n, "source vehicle rows"))
    for metric, mask in [("A_GEW valid positive", weight.notna() & np.isfinite(weight) & weight.gt(0)), ("missing A_GEW", weight.isna()), ("zero A_GEW", weight.eq(0)), ("negative A_GEW", weight.lt(0)), ("nonfinite A_GEW", np.isinf(weight))]: rows.append(qa_row("weight", metric, int(mask.sum()), n, "source vehicle rows"))
    for count, label in [(1,"one-car"),(2,"exactly-two-car"),(3,"three-plus")]:
        rows.append(qa_row("ownership", f"{label} vehicle rows", int((matched & merged.H_ANZAUTO.eq(count)).sum()), n, "source vehicle rows", "Recorded vehicles; 3+ is not a complete fleet distribution." if count == 3 else "")); rows.append(qa_row("ownership", f"{label} households", merged.loc[matched & merged.H_ANZAUTO.eq(count), "H_ID"].nunique(), vehicles.H_ID.nunique(), "unique source households"))
    for key, label in [("primary","primary valid"),("spatial","spatial valid"),("household_type","HH-type valid"),("exact","exact-two sensitivity")]: rows.extend([qa_row("samples", f"{label} vehicles", len(samples[key]), n, "source vehicle rows"), qa_row("samples", f"{label} households", samples[key].H_ID.nunique(), vehicles.H_ID.nunique(), "unique source households")])
    for row in coverage.itertuples(): rows.append(qa_row("exact-two mileage coverage", f"exact-two HH with {row.coverage.replace('_',' ')}", int(row.n_households), int(coverage.n_households.sum()), "exactly-two-car source households"))
    for context, key, order in [("spatial_context_3","spatial",SPATIAL_ORDER),("household_type","household_type",HOUSEHOLD_TYPE_ORDER)]:
        for level in order:
            for car in CAR_ORDER:
                count = int((samples[key][context].eq(level) & samples[key].car_ownership_group.eq(car)).sum()); rows.append(qa_row("context cells", f"{context}: {level} × {car}", count, len(samples[key]), f"{key} vehicles", "SMALL CELL" if count < SMALL_CELL_THRESHOLD else ""))
    return pd.DataFrame(rows)


def weight_qa(merged: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    w = primary.A_GEW
    metrics = {"min":w.min(),"max":w.max(),"mean":w.mean(),"median":w.median(),"p01":w.quantile(.01),"p05":w.quantile(.05),"p95":w.quantile(.95),"p99":w.quantile(.99),"sum":w.sum()}
    variation = merged.loc[merged.vehicle_household_merge.eq("both")].groupby("H_ID").A_GEW.nunique(dropna=False); affected = variation.gt(1)
    rows = [{"metric": k, "value": float(v), "n_vehicles":len(primary), "n_households":primary.H_ID.nunique(), "notes":"Raw unnormalized positive A_GEW in primary sample."} for k,v in metrics.items()]
    rows.append({"metric":"households with nonconstant A_GEW across recorded cars","value":int(affected.sum()),"n_vehicles":len(primary),"n_households":primary.H_ID.nunique(),"notes":"Weights were not averaged; compact diagnostic printed if nonzero."})
    if affected.any(): print(merged.loc[merged.H_ID.isin(affected.index[affected]), ["H_ID","A_ID","A_GEW"]].head(20).to_string(index=False))
    return pd.DataFrame(rows)


def publication_overall(summary: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy(); table["Group"] = table.car_ownership_group.map(CAR_LABELS)
    columns = ["Group","weighted_mean_km_per_year","weighted_median_km_per_year","weighted_p25","weighted_p75","weighted_share_zero_mileage","unweighted_n_vehicles","unique_households"]
    diff = overall.loc[overall.result.eq("Multi minus Single difference")].iloc[0]; means = summary.set_index("car_ownership_group").weighted_mean_km_per_year
    table["multi_minus_single_km_per_year"] = np.nan; table["difference_ci95_low"] = np.nan; table["difference_ci95_high"] = np.nan; table["difference_p_value"] = np.nan; table["relative_difference_percent"] = np.nan
    table.loc[table.Group.eq("Multi-car"), ["multi_minus_single_km_per_year","difference_ci95_low","difference_ci95_high","difference_p_value","relative_difference_percent"]] = [diff.estimate,diff.ci95_low,diff.ci95_high,diff.p_value,100*(means.multi_car-means.single_car)/means.single_car]
    return table[columns + ["multi_minus_single_km_per_year","difference_ci95_low","difference_ci95_high","difference_p_value","relative_difference_percent"]]


def sensitivity_comparison(primary: GEEResult, exact: GEEResult, p_wald: list[dict[str,Any]], e_wald: list[dict[str,Any]]) -> pd.DataFrame:
    p = linear_estimate(primary, scenario_vector(primary,1)-scenario_vector(primary,0))[0]; e = linear_estimate(exact, scenario_vector(exact,1)-scenario_vector(exact,0))[0]
    material = np.sign(p) != np.sign(e) or abs(e-p) > MATERIAL_CHANGE_RELATIVE * max(abs(p),1)
    rows = [{"result":"overall ownership difference km/year","primary_2plus_estimate":p,"exact_two_estimate":e,"same_direction":bool(np.sign(p)==np.sign(e)),"material_change_flag":bool(material),"criterion":"direction reversal or absolute estimate change >20% of primary absolute estimate"}]
    for pw, ew in zip(p_wald,e_wald): rows.append({"result":f"{pw['context_variable']} interaction p-value","primary_2plus_estimate":pw["p_value"],"exact_two_estimate":ew["p_value"],"same_direction":np.nan,"material_change_flag":bool((pw["p_value"]<.05)!=(ew["p_value"]<.05)),"criterion":"material if alpha=.05 joint-interaction conclusion changes"})
    return pd.DataFrame(rows)


def log_comparison(raw: dict[str,GEEResult], log: dict[str,GEEResult], raw_wald: list[dict[str,Any]], log_wald: list[dict[str,Any]]) -> pd.DataFrame:
    rb = raw["overall"].beta[raw["overall"].terms.index("multi_car_dummy")]; lb = log["overall"].beta[log["overall"].terms.index("multi_car_dummy")]
    rows = [{"result":"overall","raw_scale_estimate_km_per_year":rb,"log1p_scale_coefficient":lb,"approx_multiplicative_difference_percent_on_1_plus_scale":100*np.expm1(lb),"direction_consistent":bool(np.sign(rb)==np.sign(lb)),"interaction_conclusion_consistent":np.nan}]
    for rw,lw in zip(raw_wald,log_wald): rows.append({"result":rw["context_variable"],"raw_scale_estimate_km_per_year":np.nan,"log1p_scale_coefficient":np.nan,"approx_multiplicative_difference_percent_on_1_plus_scale":np.nan,"direction_consistent":np.nan,"interaction_conclusion_consistent":bool((rw["p_value"]<.05)==(lw["p_value"]<.05)),"raw_joint_wald_p_value":rw["p_value"],"log1p_joint_wald_p_value":lw["p_value"]})
    return pd.DataFrame(rows)


def write_metadata(results: list[GEEResult]) -> None:
    rows = [("analytical_unit","vehicle"),("vehicle_source",str(VEHICLE_PATH)),("household_backbone",str(BACKBONE_PATH)),("weight","A_GEW: raw, positive, finite, not normalized"),("cluster","H_ID"),("outcome","A_JAHRESFL [km/year]"),("method",METHOD),("working_correlation","Independence"),("covariance","robust / sandwich"),("weighted_quantiles","Sort by A_JAHRESFL and select first value where normalized cumulative positive A_GEW reaches q; deterministic synthetic test passed."),("3plus_limitation","H_ANZAUTO=3 means 3+ cars; Autos contains detailed records for at most three, so the estimand is recorded vehicles in multi-car households."),("causal_scope","Cross-sectional associational EDA; not causal and not a complete complex-survey estimator."),("statsmodels",statsmodels.__version__), ("upstream_vehicle_selection_extended","no; accepted source already contained A_GEW and A_JAHRESFL")]
    rows.extend((f"raw_A_GEW_verified_{r.name}", str(r.weights_verified)) for r in results)
    write_csv(pd.DataFrame(rows,columns=["item","value"]),"analysis_metadata.csv")


def main() -> None:
    assert statsmodels.__version__ == STATSMODELS_VERSION, f"Requires statsmodels {STATSMODELS_VERSION}; found {statsmodels.__version__}"
    validate_weighted_quantile(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vehicles, households = read_sources(); observed, _, _ = observed_mileage_qa(vehicles); write_csv(observed,"00_A_JAHRESFL_observed_QA.csv")
    merged = merge_sources(vehicles, households); samples = make_samples(merged)
    primary, spatial, hh_type = samples["primary"], samples["spatial"], samples["household_type"]

    summary = weighted_summary(primary,["car_ownership_group"]); dist = weighted_distribution(primary,["car_ownership_group"])
    summary_sp = weighted_summary(spatial,["spatial_context_3","car_ownership_group"]); dist_sp = weighted_distribution(spatial,["spatial_context_3","car_ownership_group"])
    summary_hh = weighted_summary(hh_type,["household_type","car_ownership_group"]); dist_hh = weighted_distribution(hh_type,["household_type","car_ownership_group"])
    for frame,name in [(summary,"01_weighted_summary_overall.csv"),(dist,"02_weighted_distribution_overall.csv"),(summary_sp,"03_weighted_summary_spatial.csv"),(dist_sp,"04_weighted_distribution_spatial.csv"),(summary_hh,"05_weighted_summary_household_type.csv"),(dist_hh,"06_weighted_distribution_household_type.csv")]: write_csv(frame,name)

    raw = {"overall":fit_gee(primary,"A_JAHRESFL","multi_car_dummy","primary_overall"), "spatial":fit_gee(spatial,"A_JAHRESFL","multi_car_dummy","primary_spatial","spatial_context_3",SPATIAL_ORDER), "household_type":fit_gee(hh_type,"A_JAHRESFL","multi_car_dummy","primary_household_type","household_type",HOUSEHOLD_TYPE_ORDER)}
    validate_overall_model(raw["overall"],summary); overall = overall_estimates(raw["overall"],primary)
    write_csv(pd.concat([coefficient_table(raw["overall"]), overall.rename(columns={"estimate":"estimate_km_per_year"})],ignore_index=True,sort=False),"07_model_overall.csv")
    write_csv(coefficient_table(raw["spatial"]),"08_model_spatial.csv"); spatial_est = context_estimates(raw["spatial"],spatial,"spatial_context_3",SPATIAL_ORDER); write_csv(spatial_est,"09_spatial_multicar_contrasts.csv")
    write_csv(coefficient_table(raw["household_type"]),"10_model_household_type.csv"); hh_est = context_estimates(raw["household_type"],hh_type,"household_type",HOUSEHOLD_TYPE_ORDER); write_csv(hh_est,"11_household_type_multicar_contrasts.csv")
    raw_wald = [joint_wald(raw["spatial"],"spatial_context_3",SPATIAL_ORDER),joint_wald(raw["household_type"],"household_type",HOUSEHOLD_TYPE_ORDER)]; write_csv(pd.DataFrame(raw_wald),"12_joint_wald_tests.csv")

    exact = {"overall":fit_gee(samples["exact"],"A_JAHRESFL","exact_two_dummy","exact_two_overall"), "spatial":fit_gee(samples["exact_spatial"],"A_JAHRESFL","exact_two_dummy","exact_two_spatial","spatial_context_3",SPATIAL_ORDER), "household_type":fit_gee(samples["exact_household_type"],"A_JAHRESFL","exact_two_dummy","exact_two_household_type","household_type",HOUSEHOLD_TYPE_ORDER)}
    exact_wald = [joint_wald(exact["spatial"],"spatial_context_3",SPATIAL_ORDER),joint_wald(exact["household_type"],"household_type",HOUSEHOLD_TYPE_ORDER)]
    exact_overall = overall_estimates(exact["overall"], samples["exact"]).rename(columns={"estimate": "estimate_km_per_year"})
    write_csv(pd.concat([coefficient_table(exact["overall"]), exact_overall],ignore_index=True,sort=False),"13_exact_two_sensitivity_overall.csv")
    exact_renames = {"predicted_multi_mean":"predicted_exact_two_mean", "multi_robust_se":"exact_two_robust_se", "multi_ci95_low":"exact_two_ci95_low", "multi_ci95_high":"exact_two_ci95_high", "difference_multi_minus_single":"difference_exact_two_minus_single", "n_vehicles_multi":"n_vehicles_exact_two"}
    ex_sp = context_estimates(exact["spatial"],samples["exact_spatial"],"spatial_context_3",SPATIAL_ORDER).rename(columns=exact_renames); ex_sp["joint_interaction_p_value"] = exact_wald[0]["p_value"]; write_csv(ex_sp,"14_exact_two_sensitivity_spatial.csv")
    ex_hh = context_estimates(exact["household_type"],samples["exact_household_type"],"household_type",HOUSEHOLD_TYPE_ORDER).rename(columns=exact_renames); ex_hh["joint_interaction_p_value"] = exact_wald[1]["p_value"]; write_csv(ex_hh,"15_exact_two_sensitivity_household_type.csv"); exact_comparison=sensitivity_comparison(raw["overall"],exact["overall"],raw_wald,exact_wald); write_csv(exact_comparison,"16_exact_two_sensitivity_comparison.csv")

    for key in ["primary","spatial","household_type"]: samples[key]["log1p_annual_mileage"] = np.log1p(samples[key]["A_JAHRESFL"])
    log = {"overall":fit_gee(primary,"log1p_annual_mileage","multi_car_dummy","log1p_overall"), "spatial":fit_gee(spatial,"log1p_annual_mileage","multi_car_dummy","log1p_spatial","spatial_context_3",SPATIAL_ORDER), "household_type":fit_gee(hh_type,"log1p_annual_mileage","multi_car_dummy","log1p_household_type","household_type",HOUSEHOLD_TYPE_ORDER)}
    log_wald=[joint_wald(log["spatial"],"spatial_context_3",SPATIAL_ORDER),joint_wald(log["household_type"],"household_type",HOUSEHOLD_TYPE_ORDER)]
    log_overall=coefficient_table(log["overall"]); log_overall["approx_multiplicative_difference_on_1_plus_scale"] = np.expm1(log_overall["estimate_log1p_scale"]); write_csv(log_overall,"17_log1p_sensitivity_overall.csv")
    log_renames = {"predicted_single_mean":"predicted_single_mean_log1p", "predicted_multi_mean":"predicted_multi_mean_log1p", "difference_multi_minus_single":"difference_multi_minus_single_log1p", "relative_difference_percent":"relative_difference_percent_not_applicable"}
    log_sp=context_estimates(log["spatial"],spatial,"spatial_context_3",SPATIAL_ORDER); log_sp["approx_multiplicative_difference_on_1_plus_scale"] = np.expm1(log_sp.difference_multi_minus_single); log_sp["relative_difference_percent"] = np.nan; log_sp["joint_interaction_p_value"]=log_wald[0]["p_value"]; log_sp=log_sp.rename(columns=log_renames); write_csv(log_sp,"18_log1p_sensitivity_spatial.csv")
    log_hh=context_estimates(log["household_type"],hh_type,"household_type",HOUSEHOLD_TYPE_ORDER); log_hh["approx_multiplicative_difference_on_1_plus_scale"] = np.expm1(log_hh.difference_multi_minus_single); log_hh["relative_difference_percent"] = np.nan; log_hh["joint_interaction_p_value"]=log_wald[1]["p_value"]; log_hh=log_hh.rename(columns=log_renames); write_csv(log_hh,"19_log1p_sensitivity_household_type.csv"); log_comp=log_comparison(raw,log,raw_wald,log_wald); write_csv(log_comp,"20_log1p_sensitivity_comparison.csv")

    detail=weighted_summary(primary,["car_count_detail"]); detail["interpretation_note"] = np.where(detail.car_count_detail.eq("three_plus_cars"),"Recorded vehicles in 3+ car households; not a complete fleet distribution.",""); write_csv(detail,"21_car_count_detail_summary.csv")
    coverage=exact_two_coverage(merged); fleet=build_fleet_qa(merged); write_csv(fleet,"24_fleet_record_QA.csv")
    qa=build_sample_qa(vehicles,merged,samples,coverage); write_csv(qa,"22_sample_QA.csv"); write_csv(weight_qa(merged,primary),"23_weight_QA.csv")
    # Only run the optional both-mileages-valid exact-two QA if incomplete coverage exceeds 5%.
    incomplete=1-float(coverage.loc[coverage.coverage.eq("both_mileages_valid"),"n_households"].iloc[0]/coverage.n_households.sum())
    if incomplete>.05:
        complete_ids=set(merged.loc[merged.H_ANZAUTO.eq(2)].groupby("H_ID").filter(lambda x:numeric(x.A_JAHRESFL).between(0,250000).sum()>=2).H_ID)
        complete=primary.loc[primary.H_ANZAUTO.eq(1)|primary.H_ID.isin(complete_ids)].copy(); complete["exact_two_dummy"]=complete.H_ANZAUTO.eq(2).astype(float); comp=fit_gee(complete,"A_JAHRESFL","exact_two_dummy","exact_two_complete_mileage_optional"); write_csv(coefficient_table(comp),"25_optional_exact_two_complete_mileage.csv")

    write_csv(publication_overall(summary,overall),"publication_table_overall.csv"); pub_sp=spatial_est.copy(); pub_sp["context_label"]=pub_sp.spatial_context_3.map(SPATIAL_LABELS); write_csv(pub_sp,"publication_table_spatial.csv"); pub_hh=hh_est.copy(); pub_hh["household_type_label"]=pub_hh.household_type.map(HOUSEHOLD_TYPE_LABELS); write_csv(pub_hh,"publication_table_household_type.csv")
    plot_distribution(dist); plot_context(spatial_est,"spatial_context_3",SPATIAL_ORDER,SPATIAL_LABELS,"annual_vehicle_mileage_spatial"); plot_context(hh_est,"household_type",HOUSEHOLD_TYPE_ORDER,HOUSEHOLD_TYPE_LABELS,"annual_vehicle_mileage_household_type")
    main_overall, main_spatial, main_household, _, _, _ = load_scientific_figure_inputs()
    validate_main_figure_inputs(main_overall, main_spatial, main_household)
    plot_main_mean_ci_overall(main_overall)
    plot_main_mean_ci_household_type(main_household)
    plot_main_mean_ci_spatial(main_spatial)
    all_results=[*raw.values(),*exact.values(),*log.values()]; write_metadata(all_results)

    means=summary.set_index("car_ownership_group"); difference=overall.loc[overall.result.eq("Multi minus Single difference")].iloc[0]
    print("\nTHEME 1 — ANNUAL VEHICLE MILEAGE\n\nINPUT"); print(f"vehicle source: {VEHICLE_PATH}\nHH backbone: {BACKBONE_PATH}"); print(f"\nSAMPLE\nsource vehicles: {len(vehicles):,}\nsource households: {vehicles.H_ID.nunique():,}\nvalid mileage vehicles: {int(numeric(vehicles.A_JAHRESFL).between(0,250000).sum()):,}\nprimary vehicles: {len(primary):,}\nprimary households: {primary.H_ID.nunique():,}")
    print(f"\nCAR OWNERSHIP\nSingle-car vehicles: {(primary.car_ownership_group=='single_car').sum():,}\nMulti-car vehicles: {(primary.car_ownership_group=='multi_car').sum():,}\nExact-two vehicles: {(primary.H_ANZAUTO==2).sum():,}\n3+ recorded vehicles: {(primary.H_ANZAUTO==3).sum():,}")
    for car in CAR_ORDER: print(f"\n{CAR_LABELS[car]}:\n  mean: {means.loc[car,'weighted_mean_km_per_year']:,.1f} km/year\n  median: {means.loc[car,'weighted_median_km_per_year']:,.1f} km/year\n  zero share: {means.loc[car,'weighted_share_zero_mileage']:.2%}")
    rel=100*(means.loc['multi_car','weighted_mean_km_per_year']-means.loc['single_car','weighted_mean_km_per_year'])/means.loc['single_car','weighted_mean_km_per_year']; print(f"\nPRIMARY MODEL\nMulti minus Single: {difference.estimate:,.1f} km/year\n95% CI: [{difference.ci95_low:,.1f}, {difference.ci95_high:,.1f}]\np-value: {difference.p_value:.6g}\nrelative difference: {rel:.2f}%")
    print(f"\nSPATIAL INTERACTION\nWald statistic: {raw_wald[0]['wald_statistic']:.3f}\ndf: {raw_wald[0]['degrees_of_freedom']}\np-value: {raw_wald[0]['p_value']:.6g}\n\nHOUSEHOLD-TYPE INTERACTION\nWald statistic: {raw_wald[1]['wald_statistic']:.3f}\ndf: {raw_wald[1]['degrees_of_freedom']}\np-value: {raw_wald[1]['p_value']:.6g}")
    exact_diff=exact_comparison.iloc[0]; print(f"\nEXACT-TWO SENSITIVITY\noverall difference: {exact_diff.exact_two_estimate:,.1f}\nspatial interaction p: {exact_wald[0]['p_value']:.6g}\nHH-type interaction p: {exact_wald[1]['p_value']:.6g}\n\nLOG1P SENSITIVITY\ndirection consistent: {log_comp.iloc[0].direction_consistent}\nspatial conclusion consistent: {log_comp.iloc[1].interaction_conclusion_consistent}\nHH-type conclusion consistent: {log_comp.iloc[2].interaction_conclusion_consistent}\n\nOUTPUT DIRECTORY\n{OUTPUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        if "--main-figures-only" in sys.argv[1:]:
            main_result_figures_only()
        elif "--figures-only" in sys.argv[1:]:
            scientific_figures_only_main()
        else:
            main()
