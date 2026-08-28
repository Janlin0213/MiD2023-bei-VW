"""Composition diagnostic for Theme 1 annual vehicle mileage.

This focused extension reads the completed full-sample result, constructs one
common vehicle sample, and compares additive household-type and powertrain
adjustments. It neither reruns nor modifies the completed mileage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
CAR_PATH = ROOT / "data_processed" / "selected_raw" / "cars_selected_raw.csv"
BACKBONE_PATH = ROOT / "data_processed" / "eda" / "statistical_test" / "theme1_household_backbone.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda" / "statistical_tests" / "theme1_single_vs_multicar" / "03_annual_vehicle_mileage"
FULL_MODEL_PATH = OUTPUT_DIR / "07_model_overall.csv"
EXACT_TWO_PATH = OUTPUT_DIR / "13_exact_two_sensitivity_overall.csv"
LOG1P_PATH = OUTPUT_DIR / "20_log1p_sensitivity_comparison.csv"

HH_OUTPUT = OUTPUT_DIR / "25_composition_diagnostic_hh_type.csv"
POWERTRAIN_OUTPUT = OUTPUT_DIR / "26_composition_diagnostic_powertrain.csv"
MODEL_OUTPUT = OUTPUT_DIR / "27_composition_diagnostic_nested_models.csv"
QA_OUTPUT = OUTPUT_DIR / "28_composition_diagnostic_QA.csv"
MARKDOWN_OUTPUT = OUTPUT_DIR / "annual_vehicle_mileage_composition_diagnostic.md"
NEW_OUTPUTS = {HH_OUTPUT, POWERTRAIN_OUTPUT, MODEL_OUTPUT, QA_OUTPUT, MARKDOWN_OUTPUT}

STATSMODELS_VERSION = "0.14.6"
METHOD = "A_GEW-weighted Gaussian marginal regression with identity link, estimated using GEE with household-clustered robust covariance"
Z95 = float(norm.ppf(0.975))
CAR_ORDER = ["single_car", "multi_car"]
HH_ORDER = ["family_household", "young_household", "adult_household", "senior_household"]
HH_LABELS = {
    "family_household": "Family household",
    "young_household": "Young household",
    "adult_household": "Adult household",
    "senior_household": "Senior household",
}

# Official MiD 2023 detailed A_ANTRIEB categories (code plan F2), retained
# without a BEV/non-BEV collapse and in their accepted source order.
POWERTRAIN_ORDER = [1, 2, 3, 4, 5, 6, 7]
POWERTRAIN_LABELS = {
    1: "Petrol",
    2: "Diesel",
    3: "Hybrid without vehicle charging connection",
    4: "Plug-in hybrid",
    5: "Battery-electric",
    6: "Gas",
    7: "Other",
    94: "Implausible value",
    99: "No answer",
}
INVALID_POWERTRAIN_CODES = [94, 99]


@dataclass(frozen=True)
class DiagnosticResult:
    model_id: str
    description: str
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    covariance: np.ndarray
    n_vehicles: int
    n_households: int
    weight_sum: float
    keys: tuple[str, ...]
    weights: np.ndarray
    converged: bool
    iterations: int | None
    weights_verified: bool


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_output_hashes() -> dict[Path, str]:
    """Fingerprint pre-existing outputs so the extension cannot alter them silently."""
    return {path: file_hash(path) for path in OUTPUT_DIR.iterdir() if path.is_file() and path not in NEW_OUTPUTS}


def assert_protected_outputs_unchanged(before: dict[Path, str]) -> None:
    after = {path: file_hash(path) for path in before}
    changed = [str(path) for path in before if before[path] != after[path]]
    if changed:
        raise AssertionError(f"Pre-existing annual-mileage outputs changed: {changed}")


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    x, w = values.to_numpy(float), weights.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    order = np.argsort(x[valid], kind="mergesort")
    xs, ws = x[valid][order], w[valid][order]
    cumulative = np.cumsum(ws) / ws.sum()
    return float(xs[min(np.searchsorted(cumulative, q, side="left"), len(xs) - 1)])


def read_full_result() -> dict[str, Any]:
    table = pd.read_csv(FULL_MODEL_PATH)
    row = table.loc[table["term"].eq("multi_car_dummy")]
    if len(row) != 1:
        raise RuntimeError("Could not identify the authoritative M0_full multi_car_dummy row.")
    record = row.iloc[0]
    return {
        "estimate": float(record["estimate_km_per_year"]),
        "se": float(record["robust_se"]),
        "ci_low": float(record["ci95_low"]),
        "ci_high": float(record["ci95_high"]),
        "p_value": float(record["p_value"]),
        "n_vehicles": int(record["n_vehicles"]),
        "n_households": int(record["n_households"]),
        "weight_sum": float(record["weight_sum"]),
    }


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    car_columns = ["H_ID", "A_ID", "A_GEW", "A_JAHRESFL", "A_ANTRIEB"]
    hh_columns = ["H_ID", "H_ANZAUTO", "car_ownership_group", "household_type", "spatial_context_3"]
    require_columns(pd.read_csv(CAR_PATH, nrows=0).columns, car_columns, "car source")
    require_columns(pd.read_csv(BACKBONE_PATH, nrows=0).columns, hh_columns, "household backbone")
    cars = pd.read_csv(CAR_PATH, usecols=car_columns, dtype={"H_ID": "string", "A_ID": "string"}, low_memory=False)
    households = pd.read_csv(BACKBONE_PATH, usecols=hh_columns, dtype={"H_ID": "string"}, low_memory=False)
    for frame in (cars, households):
        frame["H_ID"] = frame["H_ID"].str.strip().replace("", pd.NA)
    cars["A_ID"] = cars["A_ID"].str.strip().replace("", pd.NA)
    duplicates = cars.duplicated(["H_ID", "A_ID"], keep=False)
    if cars["H_ID"].isna().any() or duplicates.any():
        if duplicates.any():
            print(cars.loc[duplicates].head(20).to_string(index=False))
        raise RuntimeError("Invalid or duplicate source vehicle keys.")
    if households["H_ID"].isna().any() or households["H_ID"].duplicated().any():
        raise RuntimeError("Household backbone is not one nonmissing row per H_ID.")
    return cars, households


def prepare_samples(cars: pd.DataFrame, households: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    merged = cars.merge(households, on="H_ID", how="left", validate="many_to_one", indicator="vehicle_household_merge")
    if len(merged) != len(cars) or merged.duplicated(["H_ID", "A_ID"]).any():
        raise AssertionError("Household merge changed or duplicated vehicle rows.")
    merged["A_GEW"] = numeric(merged["A_GEW"])
    merged["A_JAHRESFL"] = numeric(merged["A_JAHRESFL"])
    merged["A_ANTRIEB"] = numeric(merged["A_ANTRIEB"]).astype("Int64")
    merged["multi_car_dummy"] = merged["car_ownership_group"].map({"single_car": 0.0, "multi_car": 1.0})
    primary_mask = (
        merged["H_ID"].notna()
        & merged["A_ID"].notna()
        & merged["vehicle_household_merge"].eq("both")
        & merged["A_GEW"].notna()
        & np.isfinite(merged["A_GEW"])
        & merged["A_GEW"].gt(0)
        & merged["A_JAHRESFL"].between(0, 250000, inclusive="both")
        & merged["car_ownership_group"].isin(CAR_ORDER)
        & merged["multi_car_dummy"].isin([0, 1])
    )
    primary = merged.loc[primary_mask].copy()
    valid_hh = primary["household_type"].isin(HH_ORDER)
    valid_powertrain = primary["A_ANTRIEB"].isin(POWERTRAIN_ORDER)
    common = primary.loc[valid_hh & valid_powertrain].copy()
    common["vehicle_key"] = common["H_ID"].astype(str) + "::" + common["A_ID"].astype(str)
    common = common.sort_values(["H_ID", "A_ID"], kind="mergesort").reset_index(drop=True)
    if common["vehicle_key"].duplicated().any():
        raise AssertionError("Composition common sample has duplicate vehicle keys.")
    assert common["A_JAHRESFL"].between(0, 250000).all()
    assert np.isfinite(common["A_GEW"]).all() and common["A_GEW"].gt(0).all()
    assert common["household_type"].isin(HH_ORDER).all()
    assert common["A_ANTRIEB"].isin(POWERTRAIN_ORDER).all()
    counts = {
        "valid_hh": int(valid_hh.sum()),
        "invalid_hh": int((~valid_hh).sum()),
        "valid_powertrain": int(valid_powertrain.sum()),
        "invalid_powertrain": int((~valid_powertrain).sum()),
        "powertrain_excluded_after_hh": int((valid_hh & ~valid_powertrain).sum()),
    }
    return primary, common, counts


def weighted_mean(cell: pd.DataFrame) -> float:
    return float(np.average(cell["A_JAHRESFL"], weights=cell["A_GEW"]))


def household_type_table(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_weight = common.groupby("car_ownership_group", observed=True)["A_GEW"].sum()
    for hh_type in HH_ORDER:
        cell = common.loc[common["household_type"].eq(hh_type)]
        single = cell.loc[cell["car_ownership_group"].eq("single_car")]
        multi = cell.loc[cell["car_ownership_group"].eq("multi_car")]
        single_share = float(single["A_GEW"].sum() / group_weight["single_car"])
        multi_share = float(multi["A_GEW"].sum() / group_weight["multi_car"])
        rows.append({
            "household_type": hh_type,
            "household_type_label": HH_LABELS[hh_type],
            "unweighted_n_single_vehicles": len(single),
            "unweighted_n_multi_vehicles": len(multi),
            "weighted_share_single": single_share,
            "weighted_share_multi": multi_share,
            "multi_minus_single_share_percentage_points": 100 * (multi_share - single_share),
            "weighted_mean_mileage_overall": weighted_mean(cell),
            "weighted_median_mileage_overall": weighted_quantile(cell["A_JAHRESFL"], cell["A_GEW"], 0.5),
            "unweighted_n_vehicles_overall": len(cell),
            "unique_households_overall": cell["H_ID"].nunique(),
            "weighted_mean_mileage_single": weighted_mean(single),
            "weighted_mean_mileage_multi": weighted_mean(multi),
        })
    result = pd.DataFrame(rows)
    assert np.isclose(result["weighted_share_single"].sum(), 1.0)
    assert np.isclose(result["weighted_share_multi"].sum(), 1.0)
    return result


def powertrain_table(primary: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_weight_sum = float(primary["A_GEW"].sum())
    common_group_weight = common.groupby("car_ownership_group", observed=True)["A_GEW"].sum()
    observed_order: list[int | str] = [*POWERTRAIN_ORDER, *INVALID_POWERTRAIN_CODES, "<missing>"]
    for code in observed_order:
        source_mask = primary["A_ANTRIEB"].isna() if code == "<missing>" else primary["A_ANTRIEB"].eq(code)
        source = primary.loc[source_mask]
        valid = code in POWERTRAIN_ORDER
        label = "Missing/non-numeric" if code == "<missing>" else POWERTRAIN_LABELS[int(code)]
        row: dict[str, Any] = {
            "row_type": "substantive_profile" if valid else "source_QA_invalid_or_missing",
            "powertrain_code": code,
            "powertrain_label": label,
            "valid_substantive_category": valid,
            "source_primary_unweighted_n": len(source),
            "source_primary_weight_sum": float(source["A_GEW"].sum()),
            "source_primary_weighted_share": float(source["A_GEW"].sum() / primary_weight_sum),
        }
        if valid:
            cell = common.loc[common["A_ANTRIEB"].eq(code)]
            single = cell.loc[cell["car_ownership_group"].eq("single_car")]
            multi = cell.loc[cell["car_ownership_group"].eq("multi_car")]
            single_share = float(single["A_GEW"].sum() / common_group_weight["single_car"])
            multi_share = float(multi["A_GEW"].sum() / common_group_weight["multi_car"])
            row.update({
                "unweighted_n_single_vehicles": len(single),
                "unweighted_n_multi_vehicles": len(multi),
                "weighted_share_single": single_share,
                "weighted_share_multi": multi_share,
                "multi_minus_single_share_percentage_points": 100 * (multi_share - single_share),
                "weighted_mean_mileage_overall": weighted_mean(cell),
                "weighted_median_mileage_overall": weighted_quantile(cell["A_JAHRESFL"], cell["A_GEW"], 0.5),
                "unweighted_n_vehicles_overall": len(cell),
                "unique_households_overall": cell["H_ID"].nunique(),
                "weighted_mean_mileage_single": weighted_mean(single),
                "weighted_mean_mileage_multi": weighted_mean(multi),
            })
        rows.append(row)
    result = pd.DataFrame(rows)
    valid_rows = result["valid_substantive_category"].eq(True)
    assert np.isclose(result.loc[valid_rows, "weighted_share_single"].sum(), 1.0)
    assert np.isclose(result.loc[valid_rows, "weighted_share_multi"].sum(), 1.0)
    assert np.isclose(result["source_primary_weighted_share"].sum(), 1.0)
    return result


def build_design(sample: pd.DataFrame, model_id: str) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(sample)), sample["multi_car_dummy"].to_numpy(float)]
    terms = ["Intercept", "multi_car_dummy"]
    # The completed interactions ask whether the ownership difference varies by
    # context. Here the separate question is coefficient change after accounting
    # for group composition, so deliberately additive adjustment is appropriate.
    if model_id in {"M1_HH_type", "M3_HH_type_powertrain"}:
        for level in HH_ORDER[1:]:
            columns.append(sample["household_type"].eq(level).to_numpy(float))
            terms.append(f"C(household_type)[T.{level}]")
    if model_id in {"M2_powertrain", "M3_HH_type_powertrain"}:
        for code in POWERTRAIN_ORDER[1:]:
            columns.append(sample["A_ANTRIEB"].eq(code).to_numpy(float))
            terms.append(f"C(A_ANTRIEB)[T.{code}]")
    design = np.column_stack(columns)
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError(f"Non-finite or rank-deficient design for {model_id}: {terms}")
    return design, terms


def fit_model(sample: pd.DataFrame, model_id: str, description: str) -> DiagnosticResult:
    design, terms = build_design(sample, model_id)
    y = sample["A_JAHRESFL"].to_numpy(float)
    weights = sample["A_GEW"].to_numpy(float)
    groups, levels = pd.factorize(sample["H_ID"], sort=False)
    assert len(levels) == sample["H_ID"].nunique()
    model = sm.GEE(
        endog=y,
        exog=design,
        groups=groups,
        weights=weights,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Independence(),
    )
    stored_weights = np.asarray(model.weights, dtype=float)
    weights_verified = stored_weights.shape == weights.shape and np.array_equal(stored_weights, weights)
    if not weights_verified:
        raise AssertionError(f"GEE did not retain raw A_GEW in {model_id}.")
    fitted = model.fit(cov_type="robust", maxiter=100)
    beta = np.asarray(fitted.params, dtype=float)
    se = np.asarray(fitted.bse, dtype=float)
    covariance = np.asarray(fitted.cov_params(), dtype=float)
    converged = bool(getattr(fitted, "converged", False))
    history = getattr(fitted, "fit_history", {}) or {}
    iterations = len(history.get("params", [])) or None
    if not converged or not (np.isfinite(beta).all() and np.isfinite(se).all() and np.isfinite(covariance).all()):
        raise RuntimeError(f"Non-converged/non-finite GEE result for {model_id}.")
    if covariance.shape != (len(terms), len(terms)):
        raise RuntimeError(f"Invalid covariance shape for {model_id}.")
    return DiagnosticResult(
        model_id=model_id,
        description=description,
        terms=terms,
        beta=beta,
        se=se,
        covariance=covariance,
        n_vehicles=len(sample),
        n_households=sample["H_ID"].nunique(),
        weight_sum=float(weights.sum()),
        keys=tuple(sample["vehicle_key"]),
        weights=weights.copy(),
        converged=converged,
        iterations=iterations,
        weights_verified=weights_verified,
    )


def validate_identical_model_samples(results: list[DiagnosticResult]) -> None:
    reference = results[0]
    for result in results[1:]:
        assert result.n_vehicles == reference.n_vehicles
        assert result.n_households == reference.n_households
        assert result.keys == reference.keys
        assert np.array_equal(result.weights, reference.weights)


def result_row(result: DiagnosticResult) -> dict[str, Any]:
    index = result.terms.index("multi_car_dummy")
    estimate, se = float(result.beta[index]), float(result.se[index])
    statistic = estimate / se
    return {
        "model_id": result.model_id,
        "model_description": result.description,
        "multi_car_estimate_km": estimate,
        "robust_se": se,
        "ci95_low": estimate - Z95 * se,
        "ci95_high": estimate + Z95 * se,
        "p_value": float(2 * norm.sf(abs(statistic))),
        "n_vehicles": result.n_vehicles,
        "n_households": result.n_households,
        "weight_sum": result.weight_sum,
        "method": METHOD,
        "converged": result.converged,
        "iterations": result.iterations,
        "raw_A_GEW_verified": result.weights_verified,
    }


def nested_model_table(full: dict[str, Any], results: list[DiagnosticResult], single_mean_common: float) -> pd.DataFrame:
    rows = [{
        "model_id": "M0_full",
        "model_description": "Completed primary full-sample crude model; read from prior output",
        "multi_car_estimate_km": full["estimate"],
        "robust_se": full["se"],
        "ci95_low": full["ci_low"],
        "ci95_high": full["ci_high"],
        "p_value": full["p_value"],
        "n_vehicles": full["n_vehicles"],
        "n_households": full["n_households"],
        "weight_sum": full["weight_sum"],
        "method": "Read from completed 07_model_overall.csv",
        "converged": True,
        "iterations": np.nan,
        "raw_A_GEW_verified": True,
    }]
    rows.extend(result_row(result) for result in results)
    table = pd.DataFrame(rows)
    baseline = float(table.loc[table["model_id"].eq("M0_common"), "multi_car_estimate_km"].iloc[0])
    table["change_from_M0_common_km"] = table["multi_car_estimate_km"] - baseline
    table["retained_fraction"] = table["multi_car_estimate_km"] / baseline
    table["attenuation_percent"] = 100 * (1 - table["retained_fraction"])
    table["same_direction_as_M0_common"] = np.sign(table["multi_car_estimate_km"]) == np.sign(baseline)
    table["sign_reversal"] = ~table["same_direction_as_M0_common"]
    table.loc[table["sign_reversal"], "attenuation_percent"] = np.nan
    table.loc[table["model_id"].eq("M0_full"), "attenuation_percent"] = np.nan
    table["relative_multi_difference_percent"] = 100 * table["multi_car_estimate_km"] / single_mean_common
    table["attenuation_denominator"] = "M0_common"
    table["single_common_weighted_mean_km"] = single_mean_common
    return table


def qa_row(section: str, metric: str, count: int | float, denominator: int | None, notes: str = "") -> dict[str, Any]:
    return {
        "section": section,
        "metric": metric,
        "count": count,
        "share": np.nan if denominator in (None, 0) else float(count) / denominator,
        "denominator_description": "not applicable" if denominator is None else f"N={denominator:,}",
        "notes": notes,
    }


def build_qa(primary: pd.DataFrame, common: pd.DataFrame, counts: dict[str, int], results: list[DiagnosticResult]) -> pd.DataFrame:
    rows = [
        qa_row("primary source", "source primary valid vehicles", len(primary), len(primary)),
        qa_row("primary source", "source primary valid households", primary["H_ID"].nunique(), primary["H_ID"].nunique()),
        qa_row("household type", "valid HH type vehicles", counts["valid_hh"], len(primary)),
        qa_row("household type", "invalid/missing HH type vehicles", counts["invalid_hh"], len(primary)),
        qa_row("powertrain", "valid powertrain vehicles", counts["valid_powertrain"], len(primary)),
        qa_row("powertrain", "invalid/missing powertrain vehicles", counts["invalid_powertrain"], len(primary)),
        qa_row("common sample", "excluded because HH type invalid/missing", counts["invalid_hh"], len(primary)),
        qa_row("common sample", "excluded because powertrain invalid/missing after valid HH type", counts["powertrain_excluded_after_hh"], len(primary), "Stepwise exclusion; avoids double-counting overlap."),
        qa_row("common sample", "composition common-sample vehicles", len(common), len(primary)),
        qa_row("common sample", "composition common-sample households", common["H_ID"].nunique(), primary["H_ID"].nunique()),
        qa_row("common sample", "Single common-sample vehicles", int(common["car_ownership_group"].eq("single_car").sum()), len(common)),
        qa_row("common sample", "Multi common-sample vehicles", int(common["car_ownership_group"].eq("multi_car").sum()), len(common)),
        qa_row("categories", "number of HH-type categories", common["household_type"].nunique(), None),
        qa_row("categories", "number of powertrain categories", common["A_ANTRIEB"].nunique(), None),
    ]
    for result in results:
        rows.append(qa_row("models", f"{result.model_id} vehicle N", result.n_vehicles, len(common), "Identical common-sample keys asserted."))
        rows.append(qa_row("models", f"{result.model_id} household N", result.n_households, common["H_ID"].nunique(), "Identical common-sample clusters asserted."))
        rows.append(qa_row("models", f"{result.model_id} raw A_GEW retained", int(result.weights_verified), 1, "Exact vector equality inside generic GEE."))
    return pd.DataFrame(rows)


def fmt_km(value: float) -> str:
    return f"{value:,.0f}"


def fmt_p(value: float) -> str:
    return f"{value:.3g}"


def write_markdown(
    full: dict[str, Any],
    primary: pd.DataFrame,
    common: pd.DataFrame,
    hh_table: pd.DataFrame,
    powertrain: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    model = models.set_index("model_id")
    exact = pd.read_csv(EXACT_TWO_PATH)
    exact_row = exact.loc[exact["result"].eq("Exact-two minus Single difference")].iloc[0]
    log = pd.read_csv(LOG1P_PATH)
    log_row = log.loc[log["result"].eq("overall")].iloc[0]
    hh_lines = ["| Household type | Single share | Multi share | Difference (pp) | Overall mean km/year |", "|---|---:|---:|---:|---:|"]
    for row in hh_table.itertuples():
        hh_lines.append(f"| {row.household_type_label} | {row.weighted_share_single:.1%} | {row.weighted_share_multi:.1%} | {row.multi_minus_single_share_percentage_points:+.1f} | {row.weighted_mean_mileage_overall:,.0f} |")
    valid_powertrain = powertrain.loc[powertrain["valid_substantive_category"].eq(True)]
    pt_lines = ["| Powertrain | Single share | Multi share | Difference (pp) | Overall mean km/year |", "|---|---:|---:|---:|---:|"]
    for row in valid_powertrain.itertuples():
        pt_lines.append(f"| {row.powertrain_label} | {row.weighted_share_single:.1%} | {row.weighted_share_multi:.1%} | {row.multi_minus_single_share_percentage_points:+.1f} | {row.weighted_mean_mileage_overall:,.0f} |")
    model_lines = ["| Model | Adjustment | Multi − Single km/year | 95% CI | p-value | Change from M0 common | Retained | Attenuation | Relative difference |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id in ["M0_common", "M1_HH_type", "M2_powertrain", "M3_HH_type_powertrain"]:
        row = model.loc[model_id]
        attenuation = "—" if pd.isna(row.attenuation_percent) else f"{row.attenuation_percent:.1f}%"
        model_lines.append(f"| {model_id} | {row.model_description} | {row.multi_car_estimate_km:+,.0f} | {row.ci95_low:+,.0f} to {row.ci95_high:+,.0f} | {fmt_p(row.p_value)} | {row.change_from_M0_common_km:+,.0f} | {row.retained_fraction:.3f} | {attenuation} | {row.relative_multi_difference_percent:+.1f}% |")

    m0, m1, m2, m3 = (model.loc[key] for key in ["M0_common", "M1_HH_type", "M2_powertrain", "M3_HH_type_powertrain"])
    if bool(m3.sign_reversal):
        interpretation = "The combined adjustment reverses the sign of the crude common-sample association, revealing a different within-composition pattern; a conventional attenuation percentage is therefore not interpreted."
    elif abs(m3.retained_fraction) < 0.25:
        interpretation = "Once broad household-type and powertrain composition are accounted for, little of the common-sample crude Single–Multi mileage difference remains."
    elif abs(m3.retained_fraction) < 0.75:
        interpretation = "Broad household-type and powertrain composition account for a substantial portion of the crude common-sample association, although a remaining adjusted difference persists."
    else:
        interpretation = "The positive mileage association remains close to the common-sample crude estimate and is not explained primarily by these two broad composition dimensions."
    m1_text = "substantially" if abs(m1.retained_fraction) < 0.75 else "only modestly"
    m2_text = "substantially" if abs(m2.retained_fraction) < 0.75 else "only modestly"
    restriction_change = m0.multi_car_estimate_km - full["estimate"]
    markdown = f"""# Annual Vehicle Mileage — Composition Diagnostic

## 1. Why this diagnostic was added

The completed crude vehicle-level analysis found higher annual mileage among recorded vehicles in Multi-car households (M0 full: {full['estimate']:+,.0f} km/year), while the previously completed household-type-specific contrasts were much smaller. This diagnostic examines whether the aggregate result is primarily consistent with differences in household-type and vehicle-powertrain composition. It does not imply that additional household cars themselves increase vehicle use.

## 2. Common analytical sample

M0 full used {full['n_vehicles']:,} vehicles in {full['n_households']:,} households. The diagnostic common sample contains {len(common):,} vehicles in {common['H_ID'].nunique():,} households, compared with {len(primary):,} valid primary vehicles. Restriction to complete household type and substantive A_ANTRIEB changes the crude coefficient by {restriction_change:+,.0f} km/year (M0 common: {m0.multi_car_estimate_km:+,.0f} km/year), before any adjustment.

## 3. Household-type composition

{chr(10).join(hh_lines)}

Household-type shares are vehicle-level A_GEW-weighted shares. Household-type adjustment {m1_text} changes the common-sample coefficient, from {m0.multi_car_estimate_km:+,.0f} to {m1.multi_car_estimate_km:+,.0f} km/year.

## 4. Powertrain composition

{chr(10).join(pt_lines)}

The groups have visibly different powertrain mixes where the share differences are large. Powertrain adjustment {m2_text} changes the coefficient, from {m0.multi_car_estimate_km:+,.0f} to {m2.multi_car_estimate_km:+,.0f} km/year. This describes fleet composition rather than a nuisance variable that must be removed from the primary descriptive result.

## 5. Nested-model comparison

{chr(10).join(model_lines)}

M0 common—not M0 full—is the denominator for coefficient-change and attenuation calculations. Relative differences use the weighted Single-car mean in the common sample.

## 6. Interpretation

{interpretation} Household-type and powertrain terms are descriptive composition adjustments, not mediators or causally identified confounders.

## 7. Relation to existing sensitivities

The saved exact-two sensitivity estimated {exact_row['estimate_km_per_year']:+,.0f} km/year and addresses 3+ fleet-record truncation. The saved log1p sensitivity retained the positive overall direction (approximately {log_row['approx_multiplicative_difference_percent_on_1_plus_scale']:.2f}% on the 1 + mileage scale) and addresses outcome skewness and scale. The present diagnostic addresses group composition; these are distinct robustness questions, and neither prior sensitivity was re-estimated here.

## 8. Thesis implication

Describe the +1,312 km/year result as a crude weighted vehicle-level association for recorded vehicles. Present the M0-common-to-M3 coefficient change alongside it to show how much of that aggregate contrast is consistent with broad household-type and powertrain composition, without causal wording.
"""
    MARKDOWN_OUTPUT.write_text(markdown, encoding="utf-8")


def print_summary(models: pd.DataFrame, common: pd.DataFrame) -> None:
    table = models.set_index("model_id")
    print("\nANNUAL VEHICLE MILEAGE - COMPOSITION DIAGNOSTIC")
    print("\nPRIMARY FULL-SAMPLE RESULT")
    print(f"M0_full Multi - Single: {table.loc['M0_full', 'multi_car_estimate_km']:,.1f} km/year")
    print("\nCOMMON SAMPLE")
    print(f"vehicles: {len(common):,}")
    print(f"households: {common['H_ID'].nunique():,}")
    print(f"M0_common Multi - Single: {table.loc['M0_common', 'multi_car_estimate_km']:,.1f} km/year")
    for model_id, title in [("M1_HH_type", "HOUSEHOLD-TYPE ADJUSTMENT"), ("M2_powertrain", "POWERTRAIN ADJUSTMENT"), ("M3_HH_type_powertrain", "COMBINED ADJUSTMENT")]:
        row = table.loc[model_id]
        print(f"\n{title}")
        print(f"{model_id} Multi - Single: {row.multi_car_estimate_km:,.1f} km/year")
        if model_id == "M3_HH_type_powertrain":
            print(f"95% CI: [{row.ci95_low:,.1f}, {row.ci95_high:,.1f}]")
            print(f"p-value: {row.p_value:.6g}")
        print(f"change vs M0_common: {row.change_from_M0_common_km:+,.1f} km")
        attenuation = "not interpreted (sign reversal)" if row.sign_reversal else f"{row.attenuation_percent:.2f}%"
        print(f"attenuation: {attenuation}")
    print("\nOUTPUTS")
    print(OUTPUT_DIR)


def main() -> None:
    assert statsmodels.__version__ == STATSMODELS_VERSION
    required = [CAR_PATH, BACKBONE_PATH, FULL_MODEL_PATH, EXACT_TWO_PATH, LOG1P_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing accepted inputs/prior results: {missing}")
    protected_before = protected_output_hashes()
    full = read_full_result()
    cars, households = read_sources()
    primary, common, counts = prepare_samples(cars, households)
    if len(primary) != full["n_vehicles"] or primary["H_ID"].nunique() != full["n_households"]:
        raise AssertionError("Reconstructed valid primary universe does not match saved M0_full sample counts.")

    hh_table = household_type_table(common)
    pt_table = powertrain_table(primary, common)
    descriptions = {
        "M0_common": "Common-sample crude",
        "M1_HH_type": "Household-type adjusted",
        "M2_powertrain": "Powertrain adjusted",
        "M3_HH_type_powertrain": "Household-type + powertrain adjusted",
    }
    results = [fit_model(common, model_id, description) for model_id, description in descriptions.items()]
    validate_identical_model_samples(results)
    single_mean_common = weighted_mean(common.loc[common["car_ownership_group"].eq("single_car")])
    models = nested_model_table(full, results, single_mean_common)
    qa = build_qa(primary, common, counts, results)

    hh_table.to_csv(HH_OUTPUT, index=False)
    pt_table.to_csv(POWERTRAIN_OUTPUT, index=False)
    models.to_csv(MODEL_OUTPUT, index=False)
    qa.to_csv(QA_OUTPUT, index=False)
    write_markdown(full, primary, common, hh_table, pt_table, models)
    assert_protected_outputs_unchanged(protected_before)
    print_summary(models, common)


if __name__ == "__main__":
    main()
