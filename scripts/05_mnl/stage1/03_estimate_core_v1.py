"""Stage-5 estimation of the frozen Core–V1 difference-design MNL."""
from __future__ import annotations

import importlib.util
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from biogeme.biogeme import BIOGEME  # noqa: E402
from biogeme.database import Database  # noqa: E402
from biogeme.expressions import Beta, Variable  # noqa: E402
from biogeme.parameters import Parameters  # noqa: E402
from biogeme.version import get_version as get_biogeme_version  # noqa: E402
import biogeme.models as models  # noqa: E402

from src.thesis_pipeline.paths import OUTPUTS_DIR  # noqa: E402

PREPARE_PATH = Path(__file__).with_name("01_prepare_core_v1.py")
spec = importlib.util.spec_from_file_location("core_v1_stage5_prepare_readonly", PREPARE_PATH)
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)

ROOT = OUTPUTS_DIR / "mnl" / "stage1" / "core_v1"
OUT = ROOT / "estimation"
BASELINE_PATH = OUT / "review" / "core_v1_stage5_protected_baseline.json"
PREDICTORS = tuple(prep.PREDICTORS)
PARAMETERS = tuple("B_" + name[2:] for name in PREDICTORS)
assert len(PREDICTORS) == len(PARAMETERS) == 26


def verify_protected() -> dict:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for section in ("stage1_files_and_code", "upstream_and_stage0"):
        for filename, expected in baseline[section].items():
            if prep.sha256(prep.protected_path(filename)) != expected:
                raise AssertionError(f"Protected {section} file changed: {filename}")
    return baseline


def load_accepted() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    verify_protected()
    data_dir = ROOT / "data"
    metadata = json.loads((data_dir / "core_v1_transformations.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(data_dir / "core_v1_context_sample.csv",
                         dtype={key: "string" for key in prep.JOIN_KEYS}, low_memory=False)
    effective = pd.read_csv(data_dir / "core_v1_effective_design.csv",
                            dtype={"CHOICE_ID": "string", "H_ID": "string"}, low_memory=False)
    if len(sample) != metadata["sample"]["stage1_n"] or sample.H_ID.nunique() != metadata["sample"]["households"]:
        raise AssertionError("Stage-1 metadata/sample mismatch")
    if sample.CHOICE_ID.isna().any() or not sample.CHOICE_ID.is_unique or sample.H_ID.isna().any():
        raise AssertionError("Missing or repeated occasion/household identifiers")
    if list(effective.columns) != ["CHOICE_ID", "H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW", *PREDICTORS]:
        raise AssertionError(f"Frozen {len(PREDICTORS)}-column predictor names or order changed")
    if len(effective) != len(sample) or not effective.CHOICE_ID.equals(sample.CHOICE_ID):
        raise AssertionError("Stage-1 sample and difference-design rows differ")
    for name in ("H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW"):
        if not effective[name].equals(sample[name]):
            raise AssertionError(f"Stage-1 control differs between sample and design: {name}")
    if not sample.CHOICE.isin([1, 2]).all() or not sample.AV_1.eq(1).all() or not sample.AV_2.eq(1).all():
        raise AssertionError("Accepted sample must have valid choices and both alternatives available")
    w = sample.W_GEW.to_numpy(float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise AssertionError("W_GEW must be finite and positive")
    x = effective[list(PREDICTORS)].to_numpy(float)
    if not np.isfinite(x).all() or not np.allclose(x, prep.effective_design(sample).to_numpy(float), rtol=0, atol=1e-12):
        raise AssertionError(f"Saved {len(PREDICTORS)}-column effective design failed independent reconstruction")
    return sample, effective, metadata


def binary_quantities(beta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent stable LL, score, negative Hessian, probabilities, contributions."""
    eta = x @ beta
    p = expit(eta)
    contributions = w * (y * eta - np.logaddexp(0, eta))
    scores = (w * (y - p))[:, None] * x
    information = x.T @ ((w * p * (1 - p))[:, None] * x)
    return float(contributions.sum()), scores.sum(axis=0), information, p, contributions


def scipy_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, start: np.ndarray | None = None) -> dict:
    """Independent optimizer for restarts, relabeling, and prespecified subsets."""
    if start is None:
        start = np.zeros(x.shape[1])
    def objective(beta):
        ll, gradient, _, _, _ = binary_quantities(beta, x, y, w)
        return -ll, -gradient
    def hessian(beta):
        return binary_quantities(beta, x, y, w)[2]
    result = minimize(objective, np.asarray(start, float), method="trust-exact", jac=True,
                      hess=hessian, options={"gtol": 1e-5, "maxiter": 250})
    ll, gradient, information, probabilities, contributions = binary_quantities(result.x, x, y, w)
    return {"beta": result.x, "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "ll": ll, "gradient": gradient,
            "information": information, "probabilities": probabilities,
            "contributions": contributions}


def biogeme_fit(effective: pd.DataFrame, weighted: bool, name: str):
    """Main model follows the installed Biogeme 3.3.3 Stage-0 weight convention."""
    columns = ["CHOICE", "AV_1", "AV_2", *PREDICTORS]
    if weighted:
        columns.append("W_GEW")
    database = Database(name, effective[columns].copy())
    betas = {parameter: Beta(parameter, 0, None, None, 0) for parameter in PARAMETERS}
    utility1 = sum((betas[parameter] * Variable(predictor)
                    for parameter, predictor in zip(PARAMETERS, PREDICTORS)), 0)
    utilities = {1: utility1, 2: 0}
    availability = {1: Variable("AV_1"), 2: Variable("AV_2")}
    log_probability = models.loglogit(utilities, availability, Variable("CHOICE"))
    formulas = ({"log_like": log_probability, "weight": Variable("W_GEW")}
                if weighted else {"log_like": log_probability})
    engine = BIOGEME(database, formulas, parameters=Parameters(), generate_html=False,
                     generate_yaml=False, save_iterations=False)
    engine.model_name = name
    weights = effective.W_GEW.to_numpy(float) if weighted else np.ones(len(effective))
    engine.null_loglikelihood = float(-np.log(2) * weights.sum())
    result = engine.estimate()
    beta = np.asarray([result.get_parameter_value(parameter) for parameter in PARAMETERS], float)
    raw = result.raw_estimation_results
    return result, beta, raw


def validate_biogeme(result, beta: np.ndarray, raw, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    ll, gradient, information, p, contributions = binary_quantities(beta, x, y, w)
    ll_difference = abs(ll - float(result.final_log_likelihood))
    raw_names = list(raw.beta_names)
    raw_gradient = np.asarray(raw.gradient, float)
    raw_hessian = np.asarray(raw.hessian, float)
    indices = [raw_names.index(parameter) for parameter in PARAMETERS]
    raw_gradient = raw_gradient[indices]
    raw_hessian = raw_hessian[np.ix_(indices, indices)]
    gradient_difference = float(np.max(np.abs(gradient - raw_gradient)))
    hessian_difference = float(np.max(np.abs(information + raw_hessian)))
    score_scale = max(1.0, float(w.sum()))
    if ll_difference > 1e-5 or gradient_difference / score_scale > 1e-7 or hessian_difference / score_scale > 1e-7:
        raise AssertionError(f"BLOCKER: Biogeme/independent objective mismatch: LL={ll_difference:g}, gradient={gradient_difference:g}, Hessian={hessian_difference:g}")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Nonfinite or invalid predicted probability")
    return {"ll": ll, "gradient": gradient, "information": information, "probabilities": p,
            "contributions": contributions, "ll_abs_difference": ll_difference,
            "gradient_max_abs_difference": gradient_difference,
            "hessian_max_abs_difference": hessian_difference}


def covariance_qa(x: np.ndarray, y: np.ndarray, w: np.ndarray, hid: pd.Series, beta: np.ndarray) -> dict:
    ll, gradient, information, p, contributions = binary_quantities(beta, x, y, w)
    eigen = np.linalg.eigvalsh(information)
    h_condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else float("inf")
    if eigen[0] <= 1e-8 or not np.isfinite(h_condition) or h_condition > 1e12:
        raise np.linalg.LinAlgError(f"Observed information not stably invertible; min eigenvalue={eigen[0]}, condition={h_condition}")
    bread = np.linalg.solve(information, np.eye(len(beta)))
    scores = (w * (y - p))[:, None] * x
    if hid.isna().any() or hid.astype("string").str.strip().eq("").any():
        raise AssertionError("Invalid H_ID for household clustering")
    names = PARAMETERS if x.shape[1] == len(PARAMETERS) else tuple(f"P_{i}" for i in range(x.shape[1]))
    grouped = pd.DataFrame(scores, columns=names).assign(H_ID=hid.to_numpy()).groupby("H_ID", sort=True)
    cluster_scores = grouped[list(names)].sum().to_numpy(float)
    cluster_counts = grouped.size().to_numpy(int)
    if cluster_counts.sum() != len(x) or not np.allclose(cluster_scores.sum(axis=0), scores.sum(axis=0), rtol=0, atol=1e-8):
        raise AssertionError("Household score aggregation failed")
    model = bread
    hc0 = bread @ (scores.T @ scores) @ bread
    cluster = bread @ (cluster_scores.T @ cluster_scores) @ bread
    covariances = {"model": model, "HC0": hc0, "HID_cluster_HC0": cluster}
    for name, cov in covariances.items():
        if not np.allclose(cov, cov.T, rtol=0, atol=1e-10):
            raise AssertionError(f"{name} covariance asymmetric")
        if not np.isfinite(cov).all() or (np.diag(cov) < -1e-12).any():
            raise AssertionError(f"{name} covariance has invalid diagonal")
        if not np.isfinite(np.sqrt(np.maximum(np.diag(cov), 0))).all():
            raise AssertionError(f"{name} standard error invalid")
    return {"ll": ll, "gradient": gradient, "information": information, "eigenvalues": eigen,
            "hessian_condition": h_condition, "scores": scores, "cluster_scores": cluster_scores,
            "cluster_counts": cluster_counts, "households": len(cluster_counts),
            "covariances": covariances, "probabilities": p, "contributions": contributions}


def parameter_table(beta: np.ndarray, covariance: np.ndarray, method: str) -> pd.DataFrame:
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    if not np.isfinite(se).all() or (se <= 0).any():
        raise AssertionError(f"Invalid standard errors: {method}")
    z = beta / se
    ci_low, ci_high = beta - norm.ppf(.975) * se, beta + norm.ppf(.975) * se
    with np.errstate(over="ignore", under="ignore"):
        odds = np.exp(beta)
        odds_low, odds_high = np.exp(ci_low), np.exp(ci_high)
    return pd.DataFrame({"order": np.arange(1, len(beta) + 1), "parameter": PARAMETERS,
                         "predictor": PREDICTORS, "estimate": beta, "standard_error": se,
                         "z": z, "p_two_sided": 2 * norm.sf(np.abs(z)),
                         "ci95_low": ci_low, "ci95_high": ci_high,
                         "odds_ratio": odds, "odds_ratio_ci95_low": odds_low,
                         "odds_ratio_ci95_high": odds_high, "covariance_method": method})


def biogeme_probabilities(effective: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    database = Database("core_v1_probability_validation", effective[["AV_1", "AV_2", *PREDICTORS]].copy())
    coefficients = {name: Beta(name, 0, None, None, 0) for name in PARAMETERS}
    u1 = sum((coefficients[name] * Variable(predictor)
              for name, predictor in zip(PARAMETERS, PREDICTORS)), 0)
    expression = models.logit({1: u1, 2: 0},
                              {1: Variable("AV_1"), 2: Variable("AV_2")}, 1)
    engine = BIOGEME(database, {"P1": expression}, parameters=Parameters(),
                     generate_html=False, generate_yaml=False, save_iterations=False)
    return engine.simulate(dict(zip(PARAMETERS, beta)))["P1"].to_numpy(float)


def prediction_row(model: str, p: np.ndarray, y: np.ndarray, ll: float, w: np.ndarray) -> dict:
    q = np.quantile(p, [.01, .05, .25, .5, .75, .95, .99])
    return {"model": model, "n": len(p), "sum_weights": float(w.sum()),
            "mean_p_car1": float(p.mean()), "weighted_mean_p_car1": float(np.average(p, weights=w)),
            "observed_car1_share": float(y.mean()), "weighted_observed_car1_share": float(np.average(y, weights=w)),
            "p_min": float(p.min()), "p01": q[0], "p05": q[1], "p25": q[2],
            "p_median": q[3], "p75": q[4], "p95": q[5], "p99": q[6], "p_max": float(p.max()),
            "p_below_1e_minus_6_n": int((p < 1e-6).sum()),
            "p_above_1_minus_1e_minus_6_n": int((p > 1-1e-6).sum()),
            "objective_log_loss": -ll / float(w.sum())}


def parameter_correlation(cov: np.ndarray, model: str, method: str) -> tuple[pd.DataFrame, dict]:
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    correlation = cov / np.outer(se, se)
    if not np.isfinite(correlation).all():
        raise AssertionError(f"Non-finite parameter correlation: {model}/{method}")
    table = pd.DataFrame(correlation, index=PARAMETERS, columns=PARAMETERS)
    upper = np.triu(np.abs(correlation), 1)
    row, col = np.unravel_index(int(np.argmax(upper)), upper.shape)
    return table, {"model": model, "method": method,
                   "max_abs_parameter_correlation": float(upper[row, col]),
                   "parameter_1": PARAMETERS[row], "parameter_2": PARAMETERS[col]}


def label_invariance(sample: pd.DataFrame, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                     beta: np.ndarray) -> tuple[pd.DataFrame, dict]:
    swapped_design = prep.effective_design(sample, swap=True).to_numpy(float)
    if not np.allclose(swapped_design, -x, rtol=0, atol=1e-12):
        raise AssertionError("Full vehicle-label swap did not negate effective design")
    if not np.array_equal(3 - sample.CHOICE.to_numpy(int), np.where(y == 1, 2, 1)):
        raise AssertionError("Choice swap failed")
    x_swap, y_swap = swapped_design, 1-y
    original = binary_quantities(beta, x, y, w)
    swapped = binary_quantities(beta, x_swap, y_swap, w)
    p_error = float(np.max(np.abs(swapped[3] - (1-original[3]))))
    chosen_original = np.where(y == 1, original[3], 1-original[3])
    chosen_swapped = np.where(y_swap == 1, swapped[3], 1-swapped[3])
    chosen_error = float(np.max(np.abs(chosen_original-chosen_swapped)))
    contribution_error = float(np.max(np.abs(original[4]-swapped[4])))
    score_original = (w * (y-original[3]))[:, None] * x
    score_swapped = (w * (y_swap-swapped[3]))[:, None] * x_swap
    score_error = float(np.max(np.abs(score_original-score_swapped)))
    refit = scipy_fit(x_swap, y_swap, w, beta)
    beta_error = float(np.max(np.abs(refit["beta"]-beta)))
    ll_error = abs(refit["ll"]-original[0])
    passed = (p_error < 1e-12 and chosen_error < 1e-12 and contribution_error < 1e-9
              and score_error < 1e-9 and beta_error < 5e-4 and ll_error < 1e-4
              and np.max(np.abs(refit["gradient"])) / w.sum() < 1e-5)
    rows = pd.DataFrame({"parameter": PARAMETERS, "original_estimate": beta,
                         "swapped_refit_estimate": refit["beta"],
                         "difference": refit["beta"]-beta})
    details = {"status": "PASS" if passed else "BLOCKER", "probability_max_abs_error": p_error,
               "chosen_probability_max_abs_error": chosen_error,
               "contribution_max_abs_error": contribution_error,
               "individual_score_max_abs_error": score_error,
               "refit_beta_max_abs_error": beta_error, "refit_ll_abs_error": ll_error,
               "refit_success": refit["success"], "refit_message": refit["message"]}
    return rows, details


def sensitivity_fit(label: str, mask: np.ndarray, sample: pd.DataFrame, x: np.ndarray,
                    y: np.ndarray, w: np.ndarray, main_beta: np.ndarray,
                    main_cov: np.ndarray) -> tuple[pd.DataFrame, dict]:
    n, excluded = int(mask.sum()), int((~mask).sum())
    x_sub, y_sub, w_sub = x[mask], y[mask], w[mask]
    hid_sub = sample.loc[mask, "H_ID"].reset_index(drop=True)
    rank = int(np.linalg.matrix_rank(x_sub))
    summary = {"sensitivity": label, "n": n, "excluded_n": excluded,
               "households": int(hid_sub.nunique()), "columns": len(PREDICTORS), "rank": rank}
    if rank < len(PREDICTORS):
        summary.update(status="FAILED_RANK_DEFICIENT", message="No predictor removed")
        return pd.DataFrame(), summary
    fit = scipy_fit(x_sub, y_sub, w_sub, main_beta)
    restarted_from_zero = False
    if not fit["success"]:
        restart = scipy_fit(x_sub, y_sub, w_sub, np.zeros(len(PREDICTORS)))
        if restart["success"] and restart["ll"] >= fit["ll"] - 1e-5:
            fit = restart
            restarted_from_zero = True
    if (not fit["success"] or not np.isfinite(fit["beta"]).all() or
            np.max(np.abs(fit["gradient"])) / w_sub.sum() > 1e-5):
        summary.update(status="FAILED_OPTIMIZATION", message=fit["message"])
        return pd.DataFrame(), summary
    try:
        covariance = covariance_qa(x_sub, y_sub, w_sub, hid_sub, fit["beta"])
    except (np.linalg.LinAlgError, AssertionError) as error:
        summary.update(status="FAILED_COVARIANCE", message=str(error))
        return pd.DataFrame(), summary
    se = np.sqrt(np.diag(covariance["covariances"]["HID_cluster_HC0"]))
    main_se = np.sqrt(np.diag(main_cov))
    difference = fit["beta"]-main_beta
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(np.abs(main_beta) > 1e-10, difference / np.abs(main_beta), np.nan)
    table = pd.DataFrame({"parameter": PARAMETERS, "predictor": PREDICTORS,
                          "main_weighted_estimate": main_beta, "sensitivity_estimate": fit["beta"],
                          "absolute_change": np.abs(difference), "signed_change": difference,
                          "relative_change_to_abs_main": relative,
                          "sign_changed": np.sign(fit["beta"]) != np.sign(main_beta),
                          "main_cluster_se": main_se, "sensitivity_cluster_se": se,
                          "cluster_se_change": se-main_se})
    summary.update(status="PASS", message=fit["message"], converged=fit["success"],
                   restarted_from_zero=restarted_from_zero,
                   final_ll=fit["ll"], gradient_norm=float(np.linalg.norm(fit["gradient"])),
                   hessian_condition=covariance["hessian_condition"],
                   max_abs_coefficient_change=float(np.max(np.abs(difference))),
                   sign_changes=int(table.sign_changed.sum()))
    return table, summary


def save_covariance(matrix: np.ndarray, path: Path) -> None:
    pd.DataFrame(matrix, index=PARAMETERS, columns=PARAMETERS).to_csv(path)


def estimate_main_model(name: str, weighted: bool, sample: pd.DataFrame,
                        effective: pd.DataFrame) -> dict:
    x = effective[list(PREDICTORS)].to_numpy(float)
    y = sample.CHOICE.eq(1).to_numpy(float)
    w = sample.W_GEW.to_numpy(float) if weighted else np.ones(len(sample))
    result, beta, raw = biogeme_fit(effective, weighted, f"core_v1_{name}")
    if len(beta) != len(PARAMETERS) or int(result.number_of_parameters) != len(PARAMETERS):
        raise AssertionError(f"Biogeme did not estimate all {len(PARAMETERS)} coefficients: {name}")
    check = validate_biogeme(result, beta, raw, x, y, w)
    biogeme_p = biogeme_probabilities(effective, beta)
    probability_error = float(np.max(np.abs(biogeme_p-check["probabilities"])))
    if probability_error > 1e-10:
        raise AssertionError(f"BLOCKER: Biogeme predicted probability mismatch for {name}: {probability_error}")
    covariance = covariance_qa(x, y, w, sample.H_ID, beta)
    if np.max(np.abs(check["information"]-covariance["information"])) > 1e-8:
        raise AssertionError("Independent information matrix changed between validation and inference")
    biogeme_names = list(raw.beta_names)
    indices = [biogeme_names.index(parameter) for parameter in PARAMETERS]
    biogeme_hc0 = np.asarray(result.robust_variance_covariance_matrix, float)[np.ix_(indices, indices)]
    biogeme_model = np.asarray(result.rao_cramer_variance_covariance_matrix, float)[np.ix_(indices, indices)]
    biogeme_bhhh = np.asarray(raw.bhhh, float)[np.ix_(indices, indices)]
    residual = y-check["probabilities"]
    meat_w2 = x.T @ ((w*w*residual*residual)[:, None] * x)
    meat_w1 = x.T @ ((w*residual*residual)[:, None] * x)
    bhhh_w2_difference = float(np.max(np.abs(biogeme_bhhh-meat_w2)))
    bhhh_w1_difference = float(np.max(np.abs(biogeme_bhhh-meat_w1)))
    hc0_difference = float(np.max(np.abs(biogeme_hc0-covariance["covariances"]["HC0"])))
    model_difference = float(np.max(np.abs(biogeme_model-covariance["covariances"]["model"])))
    model_root = OUT / "parameters"
    inference_root = OUT / "inference"
    fit_root = OUT / "fit"
    diag_root = OUT / "diagnostics"
    for folder in (model_root, inference_root, fit_root, diag_root):
        folder.mkdir(parents=True, exist_ok=True)
    methods = {"model_based": "model", "individual_HC0": "HC0", "HID_cluster_HC0": "HID_cluster_HC0"}
    tables = {}
    correlations = []
    for label, key in methods.items():
        matrix = covariance["covariances"][key]
        table = parameter_table(beta, matrix, key)
        table.to_csv(model_root / f"core_v1_{name}_{label}_parameters.csv", index=False)
        save_covariance(matrix, inference_root / f"core_v1_{name}_{label}_covariance.csv")
        tables[label] = table
        corr, maximum = parameter_correlation(matrix, name, key)
        correlations.append(maximum)
        if key == "HID_cluster_HC0":
            corr.to_csv(diag_root / f"core_v1_{name}_cluster_parameter_correlation.csv")
    fit_row = {
        "model": name, "weighted": weighted, "N": len(sample), "households": sample.H_ID.nunique(),
        "parameters": len(beta), "sum_weights": float(w.sum()),
        "weighting_method": "raw W_GEW observation weight; no normalization" if weighted else "each occasion weight 1",
        "starting_values": f"all {len(PARAMETERS)} zero", "optimizer": raw.optimization_messages.get("Algorithm", "Biogeme default"),
        "optimizer_messages": json.dumps({key: value for key, value in raw.optimization_messages.items()
                                           if "time" not in key.lower()}, sort_keys=True, default=str),
        "converged": bool(result.algorithm_has_converged),
        "iterations": raw.optimization_messages.get("Number of iterations", "unavailable"),
        "initial_ll": float(result.initial_log_likelihood),
        "final_ll": float(result.final_log_likelihood),
        "null_ll": float(result.null_log_likelihood),
        "rho_square": 1-float(result.final_log_likelihood)/float(result.null_log_likelihood),
        "adjusted_rho_square": 1-(float(result.final_log_likelihood)-len(PARAMETERS))/float(result.null_log_likelihood),
        "AIC": float(result.akaike_information_criterion),
        "BIC": float(result.bayesian_information_criterion),
        "biogeme_gradient_norm": float(result.gradient_norm),
        "independent_gradient_norm": float(np.linalg.norm(check["gradient"])),
        "independent_gradient_max_abs_per_weight_sum": float(np.max(np.abs(check["gradient"]))/w.sum()),
        "biogeme_version": get_biogeme_version(), "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "pandas_version": pd.__version__,
    }
    validation_row = {"model": name, "independent_ll": check["ll"],
                      "biogeme_ll": float(result.final_log_likelihood),
                      "ll_abs_difference": check["ll_abs_difference"],
                      "gradient_max_abs_difference": check["gradient_max_abs_difference"],
                      "hessian_max_abs_difference": check["hessian_max_abs_difference"],
                      "probability_max_abs_difference": probability_error,
                      "HC0_covariance_max_abs_difference_from_biogeme": hc0_difference,
                      "model_covariance_max_abs_difference_from_biogeme": model_difference,
                      "biogeme_BHHH_minus_w_squared_score_meat_max_abs": bhhh_w2_difference,
                      "biogeme_BHHH_minus_w_linear_score_meat_max_abs": bhhh_w1_difference,
                      "robust_convention_note": ("Biogeme weighted BHHH uses one factor of W_GEW; independent pseudo-score HC0 uses squared W_GEW"
                                                 if weighted else "Unweighted BHHH and individual HC0 agree"),
                      "ll_tolerance": 1e-5, "gradient_hessian_tolerance_per_sum_weights": 1e-7,
                      "probability_tolerance": 1e-10}
    max_corr = next(item for item in correlations if item["method"] == "HID_cluster_HC0")
    hessian_row = {"model": name, "min_information_eigenvalue": covariance["eigenvalues"][0],
                   "max_information_eigenvalue": covariance["eigenvalues"][-1],
                   "information_condition_number": covariance["hessian_condition"],
                   **max_corr}
    cluster_row = {"model": name, "cluster_variable": "H_ID", "N": len(sample),
                   "households": covariance["households"],
                   "sum_household_counts": int(covariance["cluster_counts"].sum()),
                   "multi_occasion_households": int((covariance["cluster_counts"] > 1).sum()),
                   "max_occasions_per_household": int(covariance["cluster_counts"].max()),
                   "max_abs_score_sum_discrepancy": float(np.max(np.abs(covariance["cluster_scores"].sum(axis=0)-covariance["scores"].sum(axis=0)))),
                   "method": "uncorrected household-cluster HC0 sandwich; no finite-sample factor"}
    pd.DataFrame([fit_row]).to_csv(fit_root / f"core_v1_{name}_fit_summary.csv", index=False)
    pd.DataFrame([validation_row]).to_csv(diag_root / f"core_v1_{name}_independent_likelihood_QA.csv", index=False)
    pd.DataFrame([hessian_row]).to_csv(diag_root / f"core_v1_{name}_hessian_diagnostics.csv", index=False)
    pd.DataFrame([cluster_row]).to_csv(inference_root / f"core_v1_{name}_cluster_aggregation_QA.csv", index=False)
    pd.DataFrame([prediction_row(name, check["probabilities"], y, check["ll"], w)]).to_csv(
        fit_root / f"core_v1_{name}_prediction_diagnostics.csv", index=False)
    return {"name": name, "weighted": weighted, "result": result, "beta": beta,
            "fit": fit_row, "validation": validation_row, "hessian": hessian_row,
            "cluster": cluster_row, "correlations": correlations,
            "tables": tables, "covariance": covariance,
            "prediction": prediction_row(name, check["probabilities"], y, check["ll"], w)}


def interpretation_table(metadata: dict) -> pd.DataFrame:
    rows = []
    for item in metadata["main_effects"] + metadata["interactions"]:
        predictor = item["predictor"]
        parameter = "B_" + predictor[2:]
        if predictor == "D_VEHICLE_AGE":
            meaning, unit = "Additional utility of one more vehicle year in the reference contexts, holding modeled attributes fixed", "per vehicle year"
        elif predictor.startswith("D_TIME_"):
            meaning, unit = "Change in target powertrain contrast associated with one more hour of realized tour time", "per tour hour"
        elif predictor.endswith("VEHAGE"):
            meaning, unit = "Change in vehicle-age slope for active context versus reference context", "per vehicle year"
        elif predictor in PREDICTORS[:7]:
            meaning, unit = "Generic vehicle-attribute contrast at the reference levels of its included contexts", "per attribute indicator"
        else:
            meaning, unit = "Additional target vehicle-attribute contrast for active context versus reference context", "per context indicator"
        rows.append({"order": len(rows)+1, "parameter": parameter, "predictor": predictor,
                     "effective_formula": item["formula"], "interpretation": meaning,
                     "unit": unit, "causal_claim": "none; observational association"})
    return pd.DataFrame(rows)


def rerun_stability(model: dict, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    stage0_file = OUTPUTS_DIR / "mnl" / "v1_only" / f"v1_only_{model['name']}_parameters.csv"
    stage0 = pd.read_csv(stage0_file).set_index("parameter")["estimate"]
    stage0_start = np.zeros(len(PARAMETERS))
    for i, parameter in enumerate(PARAMETERS[:7]):
        stage0_start[i] = float(stage0.loc[parameter])
    rows = []
    for label, start in [("all_zero", np.zeros(len(PARAMETERS))), ("stage0_main_effects_then_zero_interactions", stage0_start)]:
        fit = scipy_fit(x, y, w, start)
        rows.append({"model": model["name"], "start": label, "optimizer": "SciPy trust-exact independent check",
                     "converged": fit["success"], "message": fit["message"],
                     "iterations": fit["iterations"], "final_ll": fit["ll"],
                     "ll_abs_difference_from_biogeme": abs(fit["ll"]-model["fit"]["final_ll"]),
                     "max_abs_beta_difference_from_biogeme": float(np.max(np.abs(fit["beta"]-model["beta"]))),
                     "gradient_max_abs_per_weight_sum": float(np.max(np.abs(fit["gradient"]))/w.sum())})
    return pd.DataFrame(rows)


def stage5_report(metadata: dict, sample: pd.DataFrame, models_out: dict,
                  restarts: pd.DataFrame, label_qa: dict, sensitivities: list[dict],
                  test_result: dict, warnings: list[str]) -> str:
    weighted = models_out["weighted"]
    unweighted = models_out["unweighted"]
    primary = weighted["tables"]["HID_cluster_HC0"]
    w_hc0 = weighted["tables"]["individual_HC0"]
    max_se_difference = float(np.max(np.abs(primary.standard_error-w_hc0.standard_error)))
    max_se_row = primary.iloc[int(np.argmax(np.abs(primary.standard_error-w_hc0.standard_error)))]
    significant = int(primary.p_two_sided.lt(.05).sum())
    age_s = next(s for s in sensitivities if s["sensitivity"] == "exclude_abs_age_delta_gt40")
    long_s = next(s for s in sensitivities if s["sensitivity"] == "exclude_tour_time_gt240")
    long_table = pd.read_csv(OUT / "sensitivity" / "core_v1_sensitivity_long_tours.csv")
    long_flip = long_table.loc[long_table.sign_changed]
    flip_text = (
        "; ".join(f"{r.parameter}: {r.main_weighted_estimate:.4f} to {r.sensitivity_estimate:.4f} "
                  f"(cluster SE {r.main_cluster_se:.4f} to {r.sensitivity_cluster_se:.4f})"
                  for _, r in long_flip.iterrows()) if not long_flip.empty else "none"
    )
    lines = ["# Stage-5 Core–V1 estimation report", "", "**Decision: ESTIMATED_PENDING_MANUAL_REVIEW**", "",
             "## 1. Input version and specification", "",
             "The controlled canonical revision rebuilt Stage-1 from verified upstream inputs. Its context sample, effective design, diagnostics, and unchanged Stage-0 results matched the revised SHA-256 baseline before and after estimation. The current input and output hashes are retained in the Stage-5 manifest.", "",
             f"- N={len(sample):,} occasions; H_ID households={sample.H_ID.nunique():,}; both alternatives available for every occasion.",
             f"- Frozen design: seven generic vehicle main effects and {len(prep.INTERACTIONS)} context interactions, {len(PARAMETERS)} free coefficients, no ASC or generic context main effect. Tour time remains centered at {metadata['centering_mean_minutes']:.6f} minutes.",
             "- Primary model uses the original unnormalized `W_GEW` as an observation weight, consistent with Stage 0. Both main fits use the same rows and predictors.", "",
             "## 2. Main estimation and fit", "",
             "| Model | Converged | Final LL | Null LL | AIC | BIC | Independent gradient norm |", "|---|---|---:|---:|---:|---:|---:|",
             *[f"| {m['name']} | {m['fit']['converged']} | {m['fit']['final_ll']:.3f} | {m['fit']['null_ll']:.3f} | {m['fit']['AIC']:.2f} | {m['fit']['BIC']:.2f} | {m['fit']['independent_gradient_norm']:.4g} |" for m in (unweighted, weighted)], "",
             "The weighted fit statistics describe a pseudo-likelihood objective. Weighted and unweighted AIC/BIC are not mechanically comparable, and no model was selected by fit statistics.", "",
             "## 3. Primary weighted, household-clustered inference", "",
             "The following uses weighted coefficients and uncorrected H_ID-clustered HC0 sandwich standard errors. Two-sided p<0.05 is counted descriptively and does not select predictors.", "",
             "| Parameter | Estimate | Cluster SE | z | p | 95% CI |", "|---|---:|---:|---:|---:|---|",
             *[f"| {r.parameter} | {r.estimate:.4f} | {r.standard_error:.4f} | {r.z:.3f} | {r.p_two_sided:.4g} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] |" for _, r in primary.iterrows()], "",
             f"Descriptive p<0.05 count: {significant}/{len(PARAMETERS)}. Full model-based, individual HC0, and household-cluster covariance matrices and parameter tables are saved separately. The largest absolute clustered-versus-individual-HC0 SE difference is {max_se_difference:.5f} for {max_se_row.parameter}.", "",
             "## 4. Numerical and covariance validation", "",
             f"- Independent LL discrepancy: unweighted {unweighted['validation']['ll_abs_difference']:.3g}; weighted {weighted['validation']['ll_abs_difference']:.3g}. Independent gradient, negative Hessian, and Biogeme predicted probabilities also agree within the recorded tolerances.",
             f"- Biogeme's built-in weighted robust covariance differs from the observation-level HC0 defined here (largest matrix entry difference {weighted['validation']['HC0_covariance_max_abs_difference_from_biogeme']:.5f}). Its BHHH meat matches one factor of `W_GEW` (max difference {weighted['validation']['biogeme_BHHH_minus_w_linear_score_meat_max_abs']:.3g}), while the requested weighted pseudo-score `s_i=W_GEW_i D_i(y_i-p_i)` gives squared weights in individual HC0 and household-summed scores in clustered HC0. The unweighted matrices agree to {unweighted['validation']['HC0_covariance_max_abs_difference_from_biogeme']:.3g}. The weighted Biogeme robust covariance is therefore not substituted for the specified sandwich.",
             f"- Weighted observed-information condition number: {weighted['hessian']['information_condition_number']:.4g}; smallest eigenvalue: {weighted['hessian']['min_information_eigenvalue']:.4g}. Largest absolute clustered parameter correlation: {weighted['hessian']['max_abs_parameter_correlation']:.4f} ({weighted['hessian']['parameter_1']}, {weighted['hessian']['parameter_2']}). The revised pre-estimation scaled design condition number is recorded in the diagnostics and measures a different property.",
             f"- Household score aggregation: {weighted['cluster']['households']:,} clusters; {weighted['cluster']['multi_occasion_households']:,} have multiple occasions; household counts sum to N. No pseudo-inverse or finite-cluster correction was used.",
             f"- Independent zero and Stage-0-start refits: maximum coefficient discrepancy {restarts.max_abs_beta_difference_from_biogeme.max():.4g}; see optimization diagnostics. Biogeme 3.3.3 and SciPy versions are recorded in fit files.", "",
             "## 5. Predictions and label invariance", "",
             f"- Weighted model: mean predicted P(Car 1)={weighted['prediction']['mean_p_car1']:.4f}; median={weighted['prediction']['p_median']:.4f}; range=[{weighted['prediction']['p_min']:.4g}, {weighted['prediction']['p_max']:.4g}]; objective log-loss={weighted['prediction']['objective_log_loss']:.4f}.",
             f"- Vehicle-label swap: {label_qa['status']}. Maximum probability error={label_qa['probability_max_abs_error']:.3g}; max coefficient discrepancy after swapped refit={label_qa['refit_beta_max_abs_error']:.3g}. Individual chosen-alternative probabilities, LL contributions, and scores were checked.", "",
             "## 6. Prespecified sensitivity analyses", "",
             "| Subset | Excluded | N | Rank | Status | Largest absolute coefficient change | Sign changes |", "|---|---:|---:|---:|---|---:|---:|",
             *[f"| {s['sensitivity']} | {s['excluded_n']:,} | {s['n']:,} | {s['rank']}/{len(PARAMETERS)} | {s['status']} | {s.get('max_abs_coefficient_change', float('nan')):.4f} | {s.get('sign_changes', 'NA')} |" for s in sensitivities], "",
             f"Both subsets are temporary in-memory sensitivity checks. The original {len(PARAMETERS)} columns and full-sample time-centering mean were retained. They do not redefine the primary sample or authorize future exclusions. The largest age-tail coefficient change is {age_s.get('max_abs_coefficient_change', float('nan')):.4f}; the largest long-tour change is {long_s.get('max_abs_coefficient_change', float('nan')):.4f}. Sign changes in the long-tour subset: {flip_text}. A sign change near zero relative to its clustered SE does not support a stable directional claim.", "",
             "## 7. Interpretation and limitations", "",
             "The BEV main effect refers to the modeled baseline context: mean tour time, 0–1 stops, no confirmed household companion during focal-person tour driving, male driver, age 30–49, non-family household, and rural region, while holding segment and vehicle age constant. LARGE and vehicle-age main effects likewise depend on their included interactions. `B_TIME_PHEV` and `B_TIME_BEV` change the corresponding powertrain contrast per additional tour hour; vehicle-age slopes are per vehicle year. The interpretation CSV gives every parameter's formula and unit.", "",
             f"In the weighted fit, the baseline-context BEV contrast is {primary.set_index('parameter').loc['B_PT_BEV', 'estimate']:.3f} (conditional odds multiplier {math.exp(primary.set_index('parameter').loc['B_PT_BEV', 'estimate']):.2f}); the household-accompanied increment to the LARGE contrast is {primary.set_index('parameter').loc['B_HHACCOMP_LARGE', 'estimate']:.3f}; and the female increment to LARGE is {primary.set_index('parameter').loc['B_FEMALE_LARGE', 'estimate']:.3f}. The baseline vehicle-age slope is {primary.set_index('parameter').loc['B_VEHICLE_AGE', 'estimate']:.4f} per vehicle year. These terms are conditional on the other modeled interactions and are not population-wide unconditional effects.", "",
             "These are observational vehicle-use associations. Realized tour-driving accompaniment, tour time and stops may be jointly determined with vehicle choice. Household clustering allows within-household score correlation but does not estimate random preferences, resolve allocation constraints or endogeneity, or fully represent MiD's complex survey design. Odds ratios and illustrative pair-choice probabilities are conditional model quantities, not purchase probabilities or market shares.", "",
             "## 8. Manual review", "",
             f"- Automated tests: {test_result['status']} ({test_result['tests_run']} tests).", 
             *[f"- REVIEW: {warning}" for warning in warnings],
             f"- PASS: frozen inputs and {len(PARAMETERS)}-column design verified; independent objective and covariance checks completed.",
             "- BLOCKER: none detected in the completed primary estimation.",
             "- Final thesis model selection and substantive interpretation remain subject to manual review. No Core–V2 or later-stage model was started.", ""]
    return "\n".join(lines)


def run_automated_tests() -> tuple[dict, str]:
    """Keep Stage-1's manifest scan from seeing Stage-5 output during its tests."""
    repo = Path(__file__).resolve().parents[3]
    hold = ROOT.parent / "core_v1_stage5_test_hold"
    if hold.exists() or not OUT.is_dir():
        raise RuntimeError("Unsafe test isolation target or missing estimation output directory")
    stage1_command = [sys.executable, "-m", "unittest", "scripts/05_mnl/stage1/test_core_v1.py", "-v"]
    stage5_command = [sys.executable, "-m", "unittest", "scripts/05_mnl/stage1/test_core_v1_estimation.py", "-v"]
    OUT.rename(hold)
    try:
        first = subprocess.run(stage1_command, cwd=repo, capture_output=True, text=True, check=False)
    finally:
        hold.rename(OUT)
    verify_protected()
    second = subprocess.run(stage5_command, cwd=repo, capture_output=True, text=True, check=False)
    verify_protected()
    log = "STAGE-1 TESTS\n" + first.stdout + first.stderr + "\nSTAGE-5 TESTS\n" + second.stdout + second.stderr
    import re
    counts = [int(x) for x in re.findall(r"Ran (\d+) tests", log)]
    result = {"status": "PASS" if first.returncode == 0 and second.returncode == 0 else "BLOCKER",
              "tests_run": sum(counts), "stage1_return_code": first.returncode,
              "stage5_return_code": second.returncode,
              "commands": [" ".join(stage1_command), " ".join(stage5_command)]}
    return result, log


def main() -> None:
    baseline = verify_protected()
    sample, effective, metadata = load_accepted()
    x = effective[list(PREDICTORS)].to_numpy(float)
    y = sample.CHOICE.eq(1).to_numpy(float)
    weights = sample.W_GEW.to_numpy(float)
    for folder in ("parameters", "fit", "inference", "diagnostics", "sensitivity", "review"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    interpretation_table(metadata).to_csv(OUT / "parameters" / "core_v1_parameter_interpretation.csv", index=False)

    models_out = {}
    for name, weighted in (("unweighted", False), ("weighted", True)):
        models_out[name] = estimate_main_model(name, weighted, sample, effective)
        print(f"Biogeme {name}: converged={models_out[name]['fit']['converged']}; LL={models_out[name]['fit']['final_ll']:.6f}", flush=True)
    unweighted, weighted = models_out["unweighted"], models_out["weighted"]
    pd.DataFrame([unweighted["fit"], weighted["fit"]]).to_csv(OUT / "fit" / "core_v1_model_fit_summary.csv", index=False)
    pd.DataFrame([unweighted["validation"], weighted["validation"]]).to_csv(
        OUT / "diagnostics" / "core_v1_independent_likelihood_QA.csv", index=False)
    pd.DataFrame([unweighted["hessian"], weighted["hessian"]]).to_csv(
        OUT / "diagnostics" / "core_v1_hessian_diagnostics.csv", index=False)
    pd.DataFrame([unweighted["cluster"], weighted["cluster"]]).to_csv(
        OUT / "inference" / "core_v1_cluster_aggregation_QA.csv", index=False)
    pd.DataFrame([unweighted["prediction"], weighted["prediction"]]).to_csv(
        OUT / "fit" / "core_v1_prediction_diagnostics.csv", index=False)
    pd.DataFrame(unweighted["correlations"] + weighted["correlations"]).to_csv(
        OUT / "diagnostics" / "core_v1_parameter_correlation_summary.csv", index=False)

    comparison = pd.DataFrame({"parameter": PARAMETERS, "predictor": PREDICTORS,
                               "unweighted_estimate": unweighted["beta"],
                               "weighted_estimate": weighted["beta"],
                               "weighted_minus_unweighted": weighted["beta"]-unweighted["beta"],
                               "unweighted_cluster_se": unweighted["tables"]["HID_cluster_HC0"].standard_error,
                               "weighted_cluster_se": weighted["tables"]["HID_cluster_HC0"].standard_error})
    comparison.to_csv(OUT / "parameters" / "core_v1_parameter_comparison.csv", index=False)

    restart_tables = [rerun_stability(unweighted, x, y, np.ones(len(sample))),
                      rerun_stability(weighted, x, y, weights)]
    restarts = pd.concat(restart_tables, ignore_index=True)
    restarts.to_csv(OUT / "fit" / "core_v1_optimization_diagnostics.csv", index=False)
    invariance_table, invariance_details = label_invariance(sample, x, y, weights, weighted["beta"])
    invariance_table.to_csv(OUT / "diagnostics" / "core_v1_label_invariance.csv", index=False)
    (OUT / "diagnostics" / "core_v1_label_invariance_summary.json").write_text(
        json.dumps(invariance_details, indent=2), encoding="utf-8")
    if invariance_details["status"] != "PASS":
        raise AssertionError(f"BLOCKER: vehicle-label invariance failed: {invariance_details}")

    age_exclude = (sample.VEHICLE_AGE_1-sample.VEHICLE_AGE_2).abs().gt(40).to_numpy()
    long_exclude = sample.TOUR_TRAVEL_TIME_MIN.gt(240).to_numpy()
    age_qa = pd.read_csv(ROOT / "diagnostics" / "core_v1_vehicle_age_extreme_summary.csv")
    if age_exclude.sum() != int(age_qa.iloc[0].abs_difference_gt40_n):
        raise AssertionError("Vehicle-age sensitivity count differs from revised Stage-1 diagnostics")
    sensitivity_results = []
    main_cluster_cov = weighted["covariance"]["covariances"]["HID_cluster_HC0"]
    for label, mask, filename in [
        ("exclude_abs_age_delta_gt40", ~age_exclude, "core_v1_sensitivity_vehicle_age.csv"),
        ("exclude_tour_time_gt240", ~long_exclude, "core_v1_sensitivity_long_tours.csv"),
    ]:
        table, summary = sensitivity_fit(label, mask, sample, x, y, weights,
                                         weighted["beta"], main_cluster_cov)
        if not table.empty:
            table.to_csv(OUT / "sensitivity" / filename, index=False)
        sensitivity_results.append(summary)
    pd.DataFrame(sensitivity_results).to_csv(
        OUT / "sensitivity" / "core_v1_sensitivity_summary.csv", index=False)

    warnings = []
    for model in (unweighted, weighted):
        if not model["fit"]["converged"]:
            warnings.append(f"{model['name']} Biogeme optimizer did not declare convergence")
        if model["fit"]["independent_gradient_max_abs_per_weight_sum"] > 1e-5:
            warnings.append(f"{model['name']} independent gradient remains material")
        if model["hessian"]["information_condition_number"] > 1e8:
            warnings.append(f"{model['name']} observed information is ill-conditioned")
        if model["hessian"]["max_abs_parameter_correlation"] > .95:
            warnings.append(f"{model['name']} parameter correlation exceeds .95")
        if np.max(np.abs(model["beta"])) > 10:
            warnings.append(f"{model['name']} coefficient magnitude exceeds 10")
        if model["prediction"]["p_below_1e_minus_6_n"] + model["prediction"]["p_above_1_minus_1e_minus_6_n"]:
            warnings.append(f"{model['name']} has extreme predicted probabilities")
    if not restarts.ll_abs_difference_from_biogeme.le(1e-4).all() or not restarts.max_abs_beta_difference_from_biogeme.le(5e-4).all():
        warnings.append("Independent restart estimates differ beyond numerical comparison tolerances")
    for summary in sensitivity_results:
        if summary["status"] != "PASS":
            warnings.append(f"{summary['sensitivity']} sensitivity unresolved: {summary['status']}")
    warnings.extend([
        "Biogeme weighted built-in robust covariance uses linear-weight BHHH and differs from squared-weight pseudo-score HC0; specified H_ID-clustered covariance is primary",
        f"{int(age_exclude.sum())} >40-year vehicle-age contrasts remain in the full-sample primary fit; original construction years await upstream validation",
        f"PHEV contrasts above 240 tour minutes: {int(((sample.PT1_PHEV != sample.PT2_PHEV) & sample.TOUR_TRAVEL_TIME_MIN.gt(240)).sum())}; the long-tour sensitivity is reported separately",
        f"All {len(prep.INTERACTIONS)} interaction subsets had descriptive Kish/N below 0.5 in pre-estimation QA; this is not a design-based effective sample size",
    ])
    (OUT / "review" / "core_v1_stage5_warnings.json").write_text(json.dumps(warnings, indent=2), encoding="utf-8")

    test_result, test_log = run_automated_tests()
    (OUT / "review" / "core_v1_stage5_test_log.txt").write_text(test_log, encoding="utf-8")
    if test_result["status"] != "PASS":
        warnings.append("Automated test suite failed; see Stage-5 test log")
    (OUT / "review" / "core_v1_stage5_manual_review_checklist.md").write_text(
        "# Stage-5 Core–V1 manual review checklist\n\nDecision: ESTIMATED_PENDING_MANUAL_REVIEW\n\n"
        + "\n".join(f"- [ ] {item}" for item in [
            f"Protected input hashes and exact {len(PARAMETERS)}-column design reviewed",
            "Biogeme convergence and independent objective validation reviewed",
            "Weighted H_ID-clustered parameter table and uncertainty reviewed",
            "Weighted versus unweighted fit interpretation reviewed",
            "Hessian, predicted probabilities, and label invariance reviewed",
            "Vehicle-age and long-tour sensitivity comparisons reviewed",
            "Extreme age source verification considered",
            "Observational and complex-survey limitations documented",
            "Final thesis specification decision documented",
        ]) + "\n", encoding="utf-8")
    (OUT / "review" / "core_v1_stage5_estimation_report.md").write_text(
        stage5_report(metadata, sample, models_out, restarts, invariance_details,
                      sensitivity_results, test_result, warnings), encoding="utf-8")
    verify_protected()
    output_files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "core_v1_stage5_manifest.json")
    manifest = {"decision_status": "ESTIMATED_PENDING_MANUAL_REVIEW",
                "protected_baseline_path": prep.repo_relative(BASELINE_PATH),
                "protected_file_count": len(baseline["stage1_files_and_code"]) + len(baseline["upstream_and_stage0"]),
                "stage5_script_sha256": prep.sha256(Path(__file__)),
                "input_N": len(sample), "households": sample.H_ID.nunique(),
                "predictors": list(PREDICTORS), "parameters": list(PARAMETERS),
                "biogeme_version": get_biogeme_version(),
                "test_result": test_result, "label_invariance": invariance_details,
                "sensitivity_statuses": sensitivity_results,
                "output_file_sha256": {prep.repo_relative(path): prep.sha256(path) for path in output_files}}
    (OUT / "review" / "core_v1_stage5_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    primary = weighted["tables"]["HID_cluster_HC0"]
    se_diff = np.abs(primary.standard_error.to_numpy()-weighted["tables"]["individual_HC0"].standard_error.to_numpy())
    print(f"N={len(sample):,}; households={sample.H_ID.nunique():,}; coefficients={len(PARAMETERS)}; Biogeme={get_biogeme_version()}")
    print(f"Converged: U={unweighted['fit']['converged']}, W={weighted['fit']['converged']}; LL: U={unweighted['fit']['final_ll']:.6f}, W={weighted['fit']['final_ll']:.6f}")
    print(f"Weighted gradient norm={weighted['fit']['independent_gradient_norm']:.6g}; Hessian condition={weighted['hessian']['information_condition_number']:.6g}; max |parameter correlation|={weighted['hessian']['max_abs_parameter_correlation']:.4f}")
    print(f"Primary clustered p<.05 descriptive count={int(primary.p_two_sided.lt(.05).sum())}; largest HC0-vs-cluster SE difference={se_diff.max():.6g} ({PARAMETERS[int(se_diff.argmax())]})")
    print("Sensitivities:", ", ".join(f"{s['sensitivity']}={s['status']}" for s in sensitivity_results))
    print(f"Label invariance={invariance_details['status']}; tests={test_result['status']} ({test_result['tests_run']}); output={OUT.resolve()}")
    print(f"PASS: frozen design/source integrity and independent objective/covariance validation. REVIEW: {len(warnings)} warnings. BLOCKER: {'none' if test_result['status']=='PASS' else 'test failure'}")
    if test_result["status"] != "PASS":
        raise AssertionError("Stage-5 automated tests failed")


if __name__ == "__main__":
    main()
