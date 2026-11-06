"""Estimate the isolated, fixed 21-coefficient Stage-1 Core-V2 MNL."""
from __future__ import annotations

import importlib.util
import json
import math
import platform
from pathlib import Path
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

DIAGNOSE_PATH = Path(__file__).with_name("02_diagnose_core_v2.py")
spec = importlib.util.spec_from_file_location("core_v2_diagnostics_for_estimation", DIAGNOSE_PATH)
diagnose = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnose
spec.loader.exec_module(diagnose)
prep = diagnose.prep
ROOT = prep.ROOT
OUT = ROOT / "estimation"
PREDICTORS = prep.PREDICTORS
PARAMETERS = tuple("B_" + name[2:] for name in PREDICTORS)
assert len(PREDICTORS) == len(PARAMETERS) == 21


def verify_gate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if OUT.exists():
        raise FileExistsError(f"Core-V2 estimation output collision: {OUT}")
    sample, effective, metadata, input_manifest = diagnose.load_prepared()
    gate_path = diagnose.REVIEW / "core_v2_diagnostic_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "PASS_NO_BLOCKER" or gate["blockers"]:
        raise AssertionError("Pre-estimation Core-V2 diagnostic gate has unresolved blockers.")
    if gate["prepared_input_manifest_sha256"] != prep.sha256(prep.DATA / "core_v2_input_manifest.json"):
        raise AssertionError("Prepared Core-V2 input manifest changed after diagnostics.")
    if gate["prepare_code_sha256"] != prep.sha256(prep.Path(__file__).with_name("01_prepare_core_v2.py")):
        raise AssertionError("Preparation code changed after diagnostics.")
    if gate["diagnostic_code_sha256"] != prep.sha256(DIAGNOSE_PATH):
        raise AssertionError("Diagnostic code changed after the gate was issued.")
    for relative, expected in gate["diagnostic_hashes_sha256"].items():
        if prep.sha256(ROOT / relative) != expected:
            raise AssertionError(f"Core-V2 diagnostic artifact changed: {relative}")
    if gate["matrix_metrics"]["rank"] != 21 or gate["matrix_metrics"]["scaled_rank"] != 21:
        raise AssertionError("Full-rank gate is absent.")
    return sample, effective, gate


def binary_quantities(beta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    eta = x @ beta
    p = expit(eta)
    contributions = w * (y*eta - np.logaddexp(0, eta))
    individual_scores = (w*(y-p))[:, None] * x
    information = x.T @ ((w*p*(1-p))[:, None] * x)
    return float(contributions.sum()), individual_scores.sum(axis=0), information, p, contributions, individual_scores


def scipy_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, start: np.ndarray) -> dict:
    def objective(beta):
        ll, gradient, _, _, _, _ = binary_quantities(beta, x, y, w)
        return -ll, -gradient
    def hessian(beta):
        return binary_quantities(beta, x, y, w)[2]
    result = minimize(objective, np.asarray(start, float), method="trust-exact", jac=True,
                      hess=hessian, options={"gtol": 1e-5, "maxiter": 250})
    ll, gradient, information, p, _, _ = binary_quantities(result.x, x, y, w)
    return {"beta": result.x, "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "ll": ll, "gradient": gradient,
            "information": information, "probabilities": p}


def biogeme_fit(effective: pd.DataFrame, weighted: bool, label: str) -> tuple:
    columns = ["CHOICE", "AV_1", "AV_2", *PREDICTORS, *(["W_GEW"] if weighted else [])]
    database = Database(f"core_v2_{label}", effective[columns].copy())
    betas = {name: Beta(name, 0, None, None, 0) for name in PARAMETERS}
    u1 = sum((betas[name]*Variable(predictor) for name, predictor in zip(PARAMETERS, PREDICTORS)), 0)
    availability = {1: Variable("AV_1"), 2: Variable("AV_2")}
    log_probability = models.loglogit({1: u1, 2: 0}, availability, Variable("CHOICE"))
    formulas = ({"log_like": log_probability, "weight": Variable("W_GEW")}
                if weighted else {"log_like": log_probability})
    engine = BIOGEME(database, formulas, parameters=Parameters(), generate_html=False,
                     generate_yaml=False, save_iterations=False)
    engine.model_name = f"core_v2_{label}"
    weight_sum = float(effective.W_GEW.sum()) if weighted else float(len(effective))
    engine.null_loglikelihood = -math.log(2)*weight_sum
    result = engine.estimate()
    beta = np.asarray([result.get_parameter_value(name) for name in PARAMETERS], float)
    if len(beta) != 21 or int(result.number_of_parameters) != 21:
        raise AssertionError("Biogeme did not estimate exactly 21 coefficients.")
    return result, beta, result.raw_estimation_results


def biogeme_probabilities(effective: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    database = Database("core_v2_probability_check", effective[["AV_1", "AV_2", *PREDICTORS]].copy())
    coefficients = {name: Beta(name, 0, None, None, 0) for name in PARAMETERS}
    u1 = sum((coefficients[name]*Variable(predictor)
              for name, predictor in zip(PARAMETERS, PREDICTORS)), 0)
    expression = models.logit({1: u1, 2: 0},
                              {1: Variable("AV_1"), 2: Variable("AV_2")}, 1)
    engine = BIOGEME(database, {"P1": expression}, parameters=Parameters(),
                     generate_html=False, generate_yaml=False, save_iterations=False)
    return engine.simulate(dict(zip(PARAMETERS, beta)))["P1"].to_numpy(float)


def covariance_qa(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                  hid: pd.Series, beta: np.ndarray) -> dict:
    ll, gradient, info, p, contributions, scores = binary_quantities(beta, x, y, w)
    eigen = np.linalg.eigvalsh(info)
    condition = float(eigen[-1]/eigen[0]) if eigen[0] > 0 else float("inf")
    if eigen[0] <= 1e-8 or not np.isfinite(condition) or condition > 1e12:
        raise np.linalg.LinAlgError("Core-V2 observed information is not stably invertible.")
    bread = np.linalg.solve(info, np.eye(len(beta)))
    if hid.isna().any() or hid.astype("string").str.strip().eq("").any():
        raise AssertionError("Household identifier missing for clustered covariance.")
    grouped = pd.DataFrame(scores, columns=PARAMETERS).assign(H_ID=hid.to_numpy()).groupby("H_ID", sort=True)
    clusters = grouped[list(PARAMETERS)].sum().to_numpy(float)
    if not np.allclose(clusters.sum(axis=0), scores.sum(axis=0), rtol=0, atol=1e-8):
        raise AssertionError("Household score aggregation failed.")
    cov = {"model_based": bread,
           "individual_HC0": bread @ (scores.T@scores) @ bread,
           "HID_cluster_HC0": bread @ (clusters.T@clusters) @ bread}
    for label, matrix in cov.items():
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10) or (np.diag(matrix) < -1e-12).any():
            raise AssertionError(f"Invalid {label} covariance.")
    return {"covariances": cov, "information": info, "eigenvalues": eigen,
            "condition": condition, "clusters": len(clusters), "probabilities": p,
            "contributions": contributions, "scores": scores, "gradient": gradient, "ll": ll}


def parameter_table(beta: np.ndarray, cov: np.ndarray, model: str, method: str) -> pd.DataFrame:
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    if not np.isfinite(se).all() or (se <= 0).any():
        raise AssertionError(f"Invalid {method} standard errors for {model}.")
    z = beta/se
    return pd.DataFrame({"model": model, "order": range(1, 22), "parameter": PARAMETERS,
                         "predictor": PREDICTORS, "estimate": beta, "standard_error": se,
                         "z": z, "p_two_sided": 2*norm.sf(np.abs(z)),
                         "ci95_low": beta-1.959963984540054*se,
                         "ci95_high": beta+1.959963984540054*se, "covariance_method": method})


def model_result(label: str, weighted: bool, sample: pd.DataFrame, effective: pd.DataFrame) -> dict:
    x = effective[list(PREDICTORS)].to_numpy(float)
    y = sample.CHOICE.eq(1).to_numpy(float)
    w = sample.W_GEW.to_numpy(float) if weighted else np.ones(len(sample))
    result, beta, raw = biogeme_fit(effective, weighted, label)
    ll, gradient, info, p, contributions, scores = binary_quantities(beta, x, y, w)
    names = list(raw.beta_names)
    indices = [names.index(name) for name in PARAMETERS]
    raw_gradient = np.asarray(raw.gradient, float)[indices]
    raw_hessian = np.asarray(raw.hessian, float)[np.ix_(indices, indices)]
    ll_error = abs(ll-float(result.final_log_likelihood))
    gradient_error = float(np.max(np.abs(gradient-raw_gradient)))
    hessian_error = float(np.max(np.abs(info+raw_hessian)))
    if ll_error > 1e-5 or gradient_error/w.sum() > 1e-7 or hessian_error/w.sum() > 1e-7:
        raise AssertionError(f"Biogeme and independent Core-V2 objective differ: {label}.")
    simulation_error = float(np.max(np.abs(biogeme_probabilities(effective, beta)-p)))
    if simulation_error > 1e-10:
        raise AssertionError(f"Biogeme probability simulation differs: {label}.")
    inference = covariance_qa(x, y, w, sample.H_ID, beta)
    biogeme_model = np.asarray(result.rao_cramer_variance_covariance_matrix, float)[np.ix_(indices, indices)]
    biogeme_hc0 = np.asarray(result.robust_variance_covariance_matrix, float)[np.ix_(indices, indices)]
    bhhh = np.asarray(raw.bhhh, float)[np.ix_(indices, indices)]
    residual = y-p
    meat_linear_w = x.T @ ((w*residual*residual)[:, None]*x)
    meat_squared_w = scores.T @ scores
    covariance_checks = {
        "model_covariance_max_abs_difference_from_biogeme": float(np.max(np.abs(inference["covariances"]["model_based"]-biogeme_model))),
        "individual_HC0_max_abs_difference_from_biogeme": float(np.max(np.abs(inference["covariances"]["individual_HC0"]-biogeme_hc0))),
        "biogeme_BHHH_minus_linear_weight_meat_max_abs": float(np.max(np.abs(bhhh-meat_linear_w))),
        "biogeme_BHHH_minus_squared_weight_meat_max_abs": float(np.max(np.abs(bhhh-meat_squared_w))),
    }
    if covariance_checks["model_covariance_max_abs_difference_from_biogeme"] > 1e-5:
        raise AssertionError("Model-based covariance differs from Biogeme.")
    if not weighted and covariance_checks["individual_HC0_max_abs_difference_from_biogeme"] > 1e-5:
        raise AssertionError("Unweighted HC0 covariance differs from Biogeme.")
    null_ll = -math.log(2)*float(w.sum())
    fit = {"model": label, "weighted": weighted, "N": len(sample), "households": int(sample.H_ID.nunique()),
           "coefficients": 21, "sum_weights": float(w.sum()), "weight_rule": "raw W_GEW" if weighted else "unit weights",
           "converged": bool(result.algorithm_has_converged), "iterations": raw.optimization_messages.get("Number of iterations", "unavailable"),
           "final_ll": ll, "null_ll": null_ll, "rho_square": 1-ll/null_ll,
           "AIC": 2*21-2*ll, "BIC": math.log(len(sample))*21-2*ll,
           "AIC_BIC_note": "Weighted values are pseudo-information criteria, descriptive only" if weighted else "Same-sample unweighted likelihood",
           "gradient_norm": float(np.linalg.norm(gradient)), "gradient_max_abs_per_weight_sum": float(np.max(np.abs(gradient))/w.sum()),
           "min_information_eigenvalue": float(inference["eigenvalues"][0]),
           "max_information_eigenvalue": float(inference["eigenvalues"][-1]),
           "hessian_condition": inference["condition"], "weighted_log_loss": -ll/w.sum(),
           "biogeme_version": get_biogeme_version(), "python_version": platform.python_version(),
           "numpy_version": np.__version__, "scipy_version": scipy.__version__, "pandas_version": pd.__version__}
    validation = {"model": label, "ll_abs_difference": ll_error,
                  "gradient_max_abs_difference": gradient_error, "hessian_max_abs_difference": hessian_error,
                  "probability_max_abs_difference": simulation_error, **covariance_checks,
                  "robust_covariance_note": "Biogeme weighted robust uses linear-weight BHHH; selected pseudo-score HC0 and household cluster meat use squared W_GEW" if weighted else "Unweighted HC0 agrees with Biogeme"}
    prediction = {"model": label, "mean_p_car1": float(p.mean()),
                  "weighted_mean_p_car1": float(np.average(p, weights=w)),
                  "observed_car1_share": float(y.mean()), "weighted_observed_car1_share": float(np.average(y, weights=w)),
                  "p_min": float(p.min()), "p_median": float(np.median(p)), "p_max": float(p.max()),
                  "p01": float(np.quantile(p, .01)), "p99": float(np.quantile(p, .99)),
                  "extreme_probability_n": int(((p < 1e-6) | (p > 1-1e-6)).sum()),
                  "weighted_brier": float(np.average((y-p)**2, weights=w)),
                  "weighted_log_loss": -ll/w.sum()}
    parameter_corr = inference["covariances"]["HID_cluster_HC0"]
    se = np.sqrt(np.diag(parameter_corr))
    parameter_corr = parameter_corr/np.outer(se, se)
    upper = np.triu(np.abs(parameter_corr), 1)
    row, col = np.unravel_index(np.argmax(upper), upper.shape)
    correlation = {"model": label, "max_abs_cluster_parameter_correlation": float(upper[row, col]),
                   "parameter_1": PARAMETERS[row], "parameter_2": PARAMETERS[col]}
    return {"label": label, "beta": beta, "fit": fit, "validation": validation,
            "inference": inference, "prediction": prediction, "correlation": correlation,
            "cluster_parameter_correlation": parameter_corr}


def restart_validation(model: dict, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[dict]:
    rows = []
    for label, start in (("zero", np.zeros(21)), ("small_positive", np.full(21, .1))):
        result = scipy_fit(x, y, w, start)
        rows.append({"model": model["label"], "start": label, "success": result["success"],
                     "message": result["message"], "iterations": result["iterations"],
                     "ll": result["ll"], "ll_abs_difference_from_biogeme": abs(result["ll"]-model["fit"]["final_ll"]),
                     "max_abs_beta_difference_from_biogeme": float(np.max(np.abs(result["beta"]-model["beta"]))),
                     "gradient_max_abs_per_weight_sum": float(np.max(np.abs(result["gradient"]))/w.sum()),
                     "stationary_despite_optimizer_flag": bool(not result["success"] and
                                                               np.max(np.abs(result["gradient"]))/w.sum() < 1e-7)})
    return rows


def label_invariance(sample: pd.DataFrame, x: np.ndarray, y: np.ndarray,
                     w: np.ndarray, beta: np.ndarray) -> dict:
    swapped = prep.effective_design(sample, swap=True).to_numpy(float)
    if not np.allclose(swapped, -x, rtol=0, atol=1e-12):
        raise AssertionError("Core-V2 design does not negate under label swap.")
    old = binary_quantities(beta, x, y, w)
    new = binary_quantities(beta, swapped, 1-y, w)
    refit = scipy_fit(swapped, 1-y, w, beta)
    normalized_refit_score = float(np.max(np.abs(refit["gradient"]))/w.sum())
    result = {"probability_max_abs_error": float(np.max(np.abs(new[3]-(1-old[3])))),
              "contribution_max_abs_error": float(np.max(np.abs(new[4]-old[4]))),
              "score_max_abs_error": float(np.max(np.abs(new[5]-old[5]))),
              "refit_ll_abs_error": abs(refit["ll"]-old[0]),
              "refit_beta_max_abs_error": float(np.max(np.abs(refit["beta"]-beta))),
              "refit_success": refit["success"], "refit_message": refit["message"],
              "refit_gradient_max_abs_per_weight_sum": normalized_refit_score,
              "refit_stationary_despite_optimizer_flag": bool(not refit["success"] and normalized_refit_score < 1e-7)}
    result["status"] = "PASS" if (result["probability_max_abs_error"] < 1e-12 and
                                    result["contribution_max_abs_error"] < 1e-8 and
                                    result["score_max_abs_error"] < 1e-8 and
                                    result["refit_beta_max_abs_error"] < 5e-4 and
                                    result["refit_ll_abs_error"] < 1e-4 and
                                    (refit["success"] or normalized_refit_score < 1e-7)) else "BLOCKER"
    return result


def interpretation_table() -> pd.DataFrame:
    context_rules = {
        "TOUR_DISTANCE_C_10KM": ("TOUR_DISTANCE_KM", "final-sample mean distance", "per 10 km"),
        "MULTI_PURPOSE": ("TOUR_N_DISTINCT_PURPOSES", "0-1 reliably known distinct purposes", "2+ versus 0-1 distinct purposes"),
        "MAIN_PURPOSE_WORK": ("TOUR_MAIN_PURPOSE", "Business/Education/Errand/Leisure pooled", "Work versus pooled valid reference"),
        "MAIN_PURPOSE_ESCORT": ("TOUR_MAIN_PURPOSE", "Business/Education/Errand/Leisure pooled", "Escort versus pooled valid reference"),
        "MAIN_PURPOSE_SHOPPING": ("TOUR_MAIN_PURPOSE", "Business/Education/Errand/Leisure pooled", "Shopping versus pooled valid reference"),
        "MALE": ("HP_SEX", "female", "male versus female"),
        "EMPLOYED": ("erwerb", "not employed", "employed versus not employed"),
        "AGE_UNDER30": ("alter_gr5", "valid ages 30+", "ages 18-29 versus 30+"),
        "VERY_HIGH_ECON": ("oek_status", "valid levels 1-4", "very high level 5 versus levels 1-4"),
        "URBAN_REGION": ("RegioStaR4", "codes 21/22", "codes 11/12 versus 21/22"),
    }
    rows = []
    for name, a, b in prep.MAIN:
        source = "POWERTRAIN" if name.startswith("D_PT") else "STATUS"
        reference = "ICE" if source == "POWERTRAIN" else "Medium status"
        rows.append({"parameter": "B_"+name[2:], "predictor": name, "formula": f"{a} - {b}",
                     "source_attribute": f"{source}_1/2", "context_reference": reference,
                     "interpretation_unit": f"Car-1 minus Car-2 {name.removeprefix('D_')} indicator; status is a segment/year class"})
    for name, context, target in prep.INTERACTIONS:
        source, reference, unit = context_rules[context]
        rows.append({"parameter": "B_"+name[2:], "predictor": name,
                     "formula": f"{context} * {target}", "source_attribute": source,
                     "context_reference": reference,
                     "interpretation_unit": f"Change in {target} log-odds contrast per {unit}"})
    return pd.DataFrame(rows)


def estimate() -> dict:
    sample, effective, gate = verify_gate()
    x = effective[list(PREDICTORS)].to_numpy(float)
    y = sample.CHOICE.eq(1).to_numpy(float)
    weights = sample.W_GEW.to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any() or not sample.AV_1.eq(1).all() or not sample.AV_2.eq(1).all():
        raise AssertionError("Core-V2 estimation controls are invalid.")
    models_out = {}
    for label, weighted in (("unweighted", False), ("weighted", True)):
        models_out[label] = model_result(label, weighted, sample, effective)
        print(f"Biogeme {label}: converged={models_out[label]['fit']['converged']} LL={models_out[label]['fit']['final_ll']:.6f}", flush=True)
    restarts = restart_validation(models_out["unweighted"], x, y, np.ones(len(sample)))
    restarts += restart_validation(models_out["weighted"], x, y, weights)
    invariance = label_invariance(sample, x, y, weights, models_out["weighted"]["beta"])
    if invariance["status"] != "PASS":
        raise AssertionError(f"Core-V2 label-invariance blocker: {invariance}")
    if any(not result["fit"]["converged"] or result["fit"]["gradient_max_abs_per_weight_sum"] > 1e-5
           for result in models_out.values()):
        raise AssertionError("Core-V2 primary Biogeme fit did not converge stably.")
    if any((not row["success"] and row["gradient_max_abs_per_weight_sum"] > 1e-7) or
           row["ll_abs_difference_from_biogeme"] > 1e-4 or
           row["max_abs_beta_difference_from_biogeme"] > 5e-4 for row in restarts):
        raise AssertionError("Independent optimizer starts do not agree with Biogeme.")
    # Prespecified descriptive long-distance sensitivity; never changes the primary fit.
    long_mask = prep.numeric(sample.TOUR_DISTANCE_KM).le(300).to_numpy()
    sensitivity = {"subset": "tour_distance_at_most_300km", "excluded_n": int((~long_mask).sum()),
                   "n": int(long_mask.sum()), "rank": int(np.linalg.matrix_rank(x[long_mask]))}
    if sensitivity["rank"] == 21:
        attempts = [("primary_beta", scipy_fit(x[long_mask], y[long_mask], weights[long_mask],
                                                models_out["weighted"]["beta"])),
                    ("zero", scipy_fit(x[long_mask], y[long_mask], weights[long_mask], np.zeros(21)))]
        sensitivity["optimizer_attempts"] = [
            {"start": start, "success": fit["success"], "message": fit["message"],
             "ll": fit["ll"],
             "gradient_max_abs_per_weight_sum": float(np.max(np.abs(fit["gradient"]))/weights[long_mask].sum())}
            for start, fit in attempts]
        successful = [(start, fit) for start, fit in attempts if fit["success"]]
        start, fit = max(successful or attempts, key=lambda row: row[1]["ll"])
        sensitivity.update(status="PASS" if fit["success"] else "REVIEW_OPTIMIZER",
                           selected_start=start, optimizer_success=fit["success"],
                           optimizer_message=fit["message"],
                           gradient_max_abs_per_weight_sum=float(np.max(np.abs(fit["gradient"]))/weights[long_mask].sum()),
                           ll=fit["ll"], max_abs_beta_change=float(np.max(np.abs(fit["beta"]-models_out["weighted"]["beta"]))))
    else:
        sensitivity["status"] = "REVIEW_RANK_DEFICIENT"
    # Reconfirm protected inputs before the first estimation-result write.
    verify_gate()
    OUT.mkdir(parents=True, exist_ok=False)
    for sub in ("parameters", "inference", "fit", "diagnostics", "review"):
        (OUT/sub).mkdir(exist_ok=False)
    interpretation_table().to_csv(OUT/"parameters/core_v2_parameter_interpretation.csv", index=False)
    tables = {}
    for label, model in models_out.items():
        tables[label] = {}
        for method, covariance in model["inference"]["covariances"].items():
            table = parameter_table(model["beta"], covariance, label, method)
            table.to_csv(OUT/"parameters"/f"core_v2_{label}_{method}_parameters.csv", index=False)
            pd.DataFrame(covariance, index=PARAMETERS, columns=PARAMETERS).to_csv(
                OUT/"inference"/f"core_v2_{label}_{method}_covariance.csv")
            tables[label][method] = table
        pd.DataFrame(model["cluster_parameter_correlation"], index=PARAMETERS, columns=PARAMETERS).to_csv(
            OUT/"diagnostics"/f"core_v2_{label}_cluster_parameter_correlation.csv")
    pd.DataFrame([model["fit"] for model in models_out.values()]).to_csv(OUT/"fit/core_v2_model_fit_summary.csv", index=False)
    pd.DataFrame([model["prediction"] for model in models_out.values()]).to_csv(OUT/"fit/core_v2_prediction_diagnostics.csv", index=False)
    pd.DataFrame([model["validation"] for model in models_out.values()]).to_csv(OUT/"diagnostics/core_v2_independent_validation.csv", index=False)
    pd.DataFrame([model["correlation"] for model in models_out.values()]).to_csv(OUT/"diagnostics/core_v2_parameter_correlation_summary.csv", index=False)
    pd.DataFrame(restarts).to_csv(OUT/"fit/core_v2_optimizer_restarts.csv", index=False)
    (OUT/"diagnostics/core_v2_label_invariance.json").write_text(json.dumps(invariance, indent=2)+"\n", encoding="utf-8")
    (OUT/"fit/core_v2_long_distance_sensitivity.json").write_text(json.dumps(sensitivity, indent=2)+"\n", encoding="utf-8")
    primary = tables["weighted"]["HID_cluster_HC0"]
    report = ["# Core-V2 Stage-1 estimation", "",
              "Decision: ESTIMATED_PENDING_MANUAL_REVIEW. Five V2 vehicle contrasts and sixteen interactions; no alternative-specific constant.",
              f"N={len(sample):,}, H_ID={sample.H_ID.nunique():,}; weighted LL={models_out['weighted']['fit']['final_ll']:.6f}; unweighted LL={models_out['unweighted']['fit']['final_ll']:.6f}.",
              f"Weighted and unweighted Biogeme convergence: {models_out['weighted']['fit']['converged']} / {models_out['unweighted']['fit']['converged']}.",
              "Primary uncertainty is household-clustered HC0 from summed W_GEW-weighted pseudo-scores. Biogeme's built-in weighted robust covariance uses a different linear-weight BHHH convention and is not substituted.",
              f"Independent objective, gradient, Hessian, probability, covariance and optimizer-start validations passed. Label swap: {invariance['status']}.",
              f"Independent restart optimizer flags: {sum(not row['success'] for row in restarts)} non-success; any such solution was accepted only when its normalized score, likelihood and coefficients matched the Biogeme optimum. See restart CSV.",
              f"Weighted observed-information condition={models_out['weighted']['fit']['hessian_condition']:.6g}; max clustered parameter correlation={models_out['weighted']['correlation']['max_abs_cluster_parameter_correlation']:.5f}.",
              "The weighted AIC/BIC are descriptive pseudo-information criteria; model comparisons require the same observations and appropriate weight interpretation.",
              "Distance enters linearly by prespecification. VERY_HIGH_ECON compares level 5 with valid levels 1-4; MULTI_PURPOSE compares 2+ distinct substantive purposes with reliably known 0-1. Purpose effects compare Work/Escort/Shopping with a pooled Business/Education/Errand/Leisure reference. Vehicle status is not direct capacity.",
              "Coefficients describe conditional allocation associations, not causal effects or vehicle-purchase preferences. Survey weights and household clustering do not fully reproduce the MiD complex sample design.",
              f"Long-distance sensitivity: {sensitivity['status']} ({sensitivity['excluded_n']} valid >300 km Tours omitted only in the sensitivity).", "",
              "## Weighted household-clustered coefficients", "",
              "| Parameter | Estimate | Cluster SE | 95% CI |", "|---|---:|---:|---:|",
              *[f"| {r.parameter} | {r.estimate:.5f} | {r.standard_error:.5f} | [{r.ci95_low:.5f}, {r.ci95_high:.5f}] |"
                for r in primary.itertuples()], ""]
    (OUT/"review/core_v2_estimation_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT/"review/core_v2_manual_review_checklist.md").write_text(
        "# Core-V2 manual review\n\nDecision: ESTIMATED_PENDING_MANUAL_REVIEW\n\n"
        "- [x] Current shared inputs, sample, diagnostic gate, and 21-column formulas verified.\n"
        "- [x] Biogeme weighted/unweighted fits and independent validation passed.\n"
        "- [ ] Review clustered estimates, weak effects, very-high-economic contrast and long-tour sensitivity.\n"
        "- [ ] Record the final thesis-model interpretation and limitations.\n", encoding="utf-8")
    # Reconfirm the current-run protected inputs after estimation.
    sample2, effective2, _ = verify_gate_without_collision()
    if not sample.CHOICE_ID.equals(sample2.CHOICE_ID) or not effective.CHOICE_ID.equals(effective2.CHOICE_ID):
        raise AssertionError("Protected Core-V2 inputs changed during estimation.")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "core_v2_output_manifest.json")
    output_manifest = {"status": "ESTIMATED_PENDING_MANUAL_REVIEW", "N": len(sample),
                       "households": int(sample.H_ID.nunique()), "predictors": list(PREDICTORS),
                       "parameters": list(PARAMETERS), "input_manifest_sha256": prep.sha256(prep.DATA/"core_v2_input_manifest.json"),
                       "diagnostic_gate_sha256": prep.sha256(diagnose.REVIEW/"core_v2_diagnostic_gate.json"),
                       "estimation_code_sha256": prep.sha256(Path(__file__)),
                       "biogeme_version": get_biogeme_version(), "label_invariance": invariance,
                       "sensitivity": sensitivity,
                       "output_hashes_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): prep.sha256(p) for p in files}}
    (OUT/"review/core_v2_output_manifest.json").write_text(json.dumps(output_manifest, indent=2)+"\n", encoding="utf-8")
    return output_manifest


def verify_gate_without_collision() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Re-read protected inputs after estimation outputs exist."""
    sample, effective, metadata, _ = diagnose.load_prepared()
    gate = json.loads((diagnose.REVIEW/"core_v2_diagnostic_gate.json").read_text(encoding="utf-8"))
    if gate["status"] != "PASS_NO_BLOCKER":
        raise AssertionError("Diagnostic gate changed during estimation.")
    return sample, effective, metadata


if __name__ == "__main__":
    result = estimate()
    print(json.dumps({"status": result["status"], "N": result["N"],
                      "households": result["households"], "label_invariance": result["label_invariance"]}, indent=2))
