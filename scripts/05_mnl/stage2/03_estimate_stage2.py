"""Estimate manually selected Stage 2 weighted two-vehicle MNL specifications.

Edit the block below, then run this file. Every execution gets a fresh UTC run
directory. The --smoke option estimates V1 main effects on the same frozen pool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from src.thesis_pipeline.paths import OUTPUTS_DIR, REPO_ROOT  # noqa: E402


# ================= USER-EDITABLE MODEL SPECIFICATION =================
# Use "V1", "V2", or both. Empty context lists give main-effects-only models.
MODELS = ["V1", "V2"]
MAIN_EFFECTS = {
    "V1": ["D_PT_HEV", "D_PT_PHEV", "D_PT_BEV", "D_SEG_COMPACT",
           "D_SEG_MEDIUM", "D_SEG_LARGE", "D_VEHICLE_AGE"],
    "V2": ["D_PT_HEV", "D_PT_PHEV", "D_PT_BEV", "D_STATUS_LOW", "D_STATUS_HIGH"],
}
# None selects all 17 contexts from the saved manifest. [] selects none.
INTERACTION_CONTEXTS = {"V1": None, "V2": None}
# Remove an entire context group from the selected contexts (including "all").
EXCLUDE_CONTEXTS = {"V1": [], "V2": []}
# Feature-level overrides use exact names from stage2_feature_manifest.json.
EXCLUDE_INTERACTIONS = {"V1": [], "V2": []}
ADD_INTERACTIONS = {"V1": [], "V2": []}
# ================================================================


PREP = OUTPUTS_DIR / "mnl" / "stage2" / "preparation"
RUNS = OUTPUTS_DIR / "mnl" / "stage2" / "estimation"
INPUT_FILES = ("stage2_common_pool.csv", "stage2_v1_design.csv", "stage2_v2_design.csv",
               "stage2_feature_manifest.json", "stage2_preparation_qa.json")
SETTINGS = {"optimization_algorithm": "simple_bounds", "max_iterations": 1000,
            "save_iterations": False, "generate_yaml": True}
# Biogeme 3.3.3's HTML writer repeatedly recalculates covariance for every
# parameter pair; the 109-parameter V1 report stalled. Native YAML is retained.
NATIVE_HTML_MAX_PARAMETERS = 30
LOWER_CONTEXTS = {"FAMILY_X_ESCORT": ("FAMILY", "MAIN_ESCORT"),
                  "VERY_HIGH_ECON_X_WORK": ("VERY_HIGH_ECON", "MAIN_WORK"),
                  "URBAN_X_LEISURE": ("URBAN_REGION", "MAIN_LEISURE")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def unique_list(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries in {label}")


def resolve(model: str, manifest: dict, smoke: bool) -> tuple[list[str], list[str], list[str]]:
    specs = manifest[f"{model.lower()}_features"]
    prepared = [row["feature"] for row in specs]
    mains = [row["feature"] for row in specs if row["kind"] == "main"]
    interactions = {row["feature"]: row for row in specs if row["kind"] == "interaction"}
    contexts = [row["feature"] for row in manifest["contexts"]]
    selected_main = MAIN_EFFECTS[model].copy()
    selected_contexts = [] if smoke else INTERACTION_CONTEXTS[model]
    selected_contexts = contexts if selected_contexts is None else selected_contexts
    excluded_contexts = [] if smoke else EXCLUDE_CONTEXTS[model]
    excluded = [] if smoke else EXCLUDE_INTERACTIONS[model]
    added = [] if smoke else ADD_INTERACTIONS[model]
    for label, values in (("main effects", selected_main), ("contexts", selected_contexts),
                          ("context exclusions", excluded_contexts),
                          ("interaction exclusions", excluded), ("interaction additions", added)):
        unique_list(values, f"{model} {label}")
    if set(selected_main) - set(mains) or (set(selected_contexts) | set(excluded_contexts)) - set(contexts):
        raise ValueError(f"{model}: unknown main effect or context")
    if (set(excluded) | set(added)) - set(interactions):
        raise ValueError(f"{model}: unknown or cross-model interaction override")
    if set(excluded) & set(added):
        raise ValueError(f"{model}: same interaction both excluded and added")
    selected = set(selected_main)
    selected.update(name for name, spec in interactions.items()
                    if spec["context"] in selected_contexts and spec["context"] not in excluded_contexts
                    and name not in excluded)
    selected.update(added)
    if not selected:
        raise ValueError(f"{model}: no predictors selected")
    resolved = [name for name in prepared if name in selected]
    warnings = []
    for name in resolved:
        if name not in interactions:
            continue
        spec = interactions[name]
        vehicle_main = f"D_{spec['vehicle_attribute']}"
        if vehicle_main not in selected:
            warnings.append(f"{name}: vehicle main effect {vehicle_main} is absent")
        for lower in LOWER_CONTEXTS.get(spec["context"], ()):
            lower_name = f"D_{lower}__{spec['vehicle_attribute']}"
            if lower_name not in selected:
                warnings.append(f"{name}: lower-order interaction {lower_name} is absent")
    return resolved, [name for name in prepared if name not in selected], warnings


def load_preparation() -> tuple[dict[str, pd.DataFrame], dict, dict, dict]:
    paths = {name: PREP / name for name in INPUT_FILES}
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = json.loads(paths["stage2_feature_manifest.json"].read_text(encoding="utf-8"))
    qa = json.loads(paths["stage2_preparation_qa.json"].read_text(encoding="utf-8"))
    for name in INPUT_FILES[:3]:
        if hashes[name] != qa["output_sha256"][name]:
            raise ValueError(f"Frozen preparation hash mismatch: {name}")
    if manifest["references"]["powertrain"] != "ICE" or \
            manifest["references"]["segment_v1"] != "SMALL" or \
            manifest["references"]["status_v2"] != "MEDIUM":
        raise ValueError("Unexpected vehicle reference categories")
    data = {"COMMON": pd.read_csv(paths[INPUT_FILES[0]], dtype={"CHOICE_ID": str, "H_ID": str}),
            "V1": pd.read_csv(paths[INPUT_FILES[1]], dtype={"CHOICE_ID": str, "H_ID": str}),
            "V2": pd.read_csv(paths[INPUT_FILES[2]], dtype={"CHOICE_ID": str, "H_ID": str})}
    common = data["COMMON"]
    if len(common) != qa["final_n"] or not common.CHOICE_ID.is_unique:
        raise ValueError("Frozen common sample does not match preparation QA")
    controls = ["CHOICE_ID", "H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW"]
    if not common.CHOICE.isin([1, 2]).all() or not common[["AV_1", "AV_2"]].eq(1).all().all() or \
            not np.isfinite(common.W_GEW).all() or not common.W_GEW.gt(0).all():
        raise ValueError("Invalid choice, availability, or weight in frozen pool")
    for model in ("V1", "V2"):
        design = data[model]
        declared = [row["feature"] for row in manifest[f"{model.lower()}_features"]]
        if list(design.columns[10:]) != declared or not design[controls].equals(common[controls]):
            raise ValueError(f"{model} design differs from common sample or manifest")
        if not np.isfinite(design[declared].to_numpy(float)).all():
            raise ValueError(f"{model} design has non-finite predictors")
    return data, manifest, qa, hashes


def make_run_directory() -> tuple[Path, str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    RUNS.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"{timestamp}_{uuid4().hex[:8]}"
        path = RUNS / run_id
        try:
            path.mkdir(exist_ok=False)
            return path, run_id, timestamp
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a unique estimation run directory")


def estimate_model(model: str, design: pd.DataFrame, features: list[str],
                   output_dir: Path, run_id: str, timestamp: str,
                   warnings: list[str]) -> dict:
    from biogeme.biogeme import BIOGEME
    from biogeme.database import Database
    from biogeme.expressions import Beta, Variable
    from biogeme.parameters import Parameters
    from biogeme.results_processing import EstimateVarianceCovariance
    import biogeme.models as models
    from biogeme.version import get_version

    output_dir.mkdir(exist_ok=False)
    database = Database(f"stage2_{model.lower()}", design[["CHOICE", "W_GEW", *features]].copy())
    coefficients = {name: Beta(f"B_{name[2:]}", 0, None, None, 0) for name in features}
    utility_1 = sum((coefficients[name] * Variable(name) for name in features), 0)
    log_probability = models.loglogit({1: utility_1, 2: 0},
                                      {1: 1, 2: 1}, Variable("CHOICE"))
    generate_html = len(features) <= NATIVE_HTML_MAX_PARAMETERS
    if not generate_html:
        warnings.append("Native Biogeme HTML disabled for large model: its pairwise covariance writer stalled; native YAML retained.")
    engine = BIOGEME(database, {"log_like": log_probability, "weight": Variable("W_GEW")},
                     parameters=Parameters(), generate_html=generate_html, **SETTINGS)
    engine.model_name = str((output_dir / f"stage2_{model.lower()}").resolve())
    engine.null_loglikelihood = -math.log(2) * float(design.W_GEW.sum())
    result = engine.estimate()  # full estimation, including covariance calculations
    if len(result.beta_names) != len(features):
        raise AssertionError("Biogeme parameter count differs from selected predictors")
    rows = []
    covariance = EstimateVarianceCovariance.ROBUST
    robust_matrix = result.get_variance_covariance_matrix(covariance)
    from scipy.stats import norm
    for name in features:
        parameter = f"B_{name[2:]}"
        beta = float(result.get_parameter_value(parameter))
        index = result.get_parameter_index(parameter)
        variance = float(robust_matrix[index, index])
        se = math.sqrt(variance) if variance >= 0 and math.isfinite(variance) else None
        t = beta / se if se is not None and se > 0 else None
        p = float(2 * norm.sf(abs(t))) if t is not None else None
        lower, upper = ((beta - 1.96 * se, beta + 1.96 * se)
                        if se is not None else (None, None))
        if se is None:
            warnings.append(f"{parameter}: robust variance unavailable")
        rows.append({"predictor": name, "parameter": parameter, "coefficient": beta,
                     "robust_standard_error": se, "robust_t": t, "robust_p_value": p,
                     "robust_ci95_lower": lower, "robust_ci95_upper": upper,
                     "covariance_estimator": "Biogeme ROBUST; not household-clustered"})
    pd.DataFrame(rows).to_csv(output_dir / "coefficients.csv", index=False)
    beta_vector = np.array([result.get_parameter_value(f"B_{name[2:]}") for name in features])
    delta = design[features].to_numpy(float) @ beta_vector
    from scipy.special import expit
    probability_1 = expit(delta)
    y = design.CHOICE.eq(1).to_numpy()
    manual_ll = float(np.dot(design.W_GEW.to_numpy(float),
                             np.where(y, -np.logaddexp(0, -delta), -np.logaddexp(0, delta))))
    final_ll = float(result.final_log_likelihood)
    if not np.isfinite(beta_vector).all() or not math.isfinite(final_ll) or \
            abs(manual_ll - final_ll) > 1e-5 or not np.isfinite(probability_1).all():
        raise AssertionError("Coefficient, sign, weight, or likelihood verification failed")
    summary = {"model": model, "run_id": run_id, "timestamp_utc": timestamp,
               "observations": len(design), "households": int(design.H_ID.nunique()),
               "weight_sum": float(design.W_GEW.sum()), "parameters": len(features),
               "converged": bool(result.algorithm_has_converged),
               "weighted_null_log_likelihood": float(result.null_log_likelihood),
               "weighted_final_log_likelihood": final_ll,
               "manual_weighted_log_likelihood": manual_ll,
               "weighted_aic": float(result.akaike_information_criterion),
               "weighted_bic_n_observations": float(result.bayesian_information_criterion),
               "rho_square_null": float(result.rho_square_null),
               "gradient_norm": float(result.gradient_norm),
               "probability_checks": [{"choice": int(design.CHOICE.iloc[i]),
                                       "delta_utility": float(delta[i]),
                                       "p_vehicle_1": float(probability_1[i])}
                                      for i in range(min(3, len(design)))],
               "biogeme_version": get_version(), "optimizer": SETTINGS["optimization_algorithm"],
               "native_html_enabled": generate_html,
               "native_html": Path(engine.html_filename).name if engine.html_filename else None,
               "native_yaml": Path(engine.yaml_filename).name if engine.yaml_filename else None,
               "covariance_estimator": "Biogeme ROBUST; not household-clustered",
               "warnings": warnings + ([] if result.algorithm_has_converged else ["Optimizer did not converge"]),
               "likelihood_note": "Survey-weighted objective; ordinary LR p-values are not reported."}
    summary["status"] = "SUCCESS" if summary["converged"] else "NON_CONVERGED"
    write_json(output_dir / "model_summary.json", summary)
    required = [output_dir / "coefficients.csv", output_dir / "model_summary.json",
                output_dir / summary["native_yaml"]]
    if generate_html:
        required.append(output_dir / summary["native_html"])
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError(f"{model}: required estimation output is missing or empty")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="V1 main effects only; same frozen sample")
    args = parser.parse_args()
    data, manifest, qa, hashes = load_preparation()
    models = ["V1"] if args.smoke else MODELS
    unique_list(models, "MODELS")
    if not models or set(models) - {"V1", "V2"}:
        raise ValueError("MODELS must contain V1, V2, or both")
    resolved = {model: resolve(model, manifest, args.smoke) for model in models}
    if {name: sha256(PREP / name) for name in INPUT_FILES} != hashes:
        raise AssertionError("Preparation inputs changed before estimation")
    run_dir, run_id, timestamp = make_run_directory()
    from biogeme.version import get_version
    config = {"run_id": run_id, "timestamp_utc": timestamp, "mode": "smoke" if args.smoke else "configured",
              "models": models, "source_input_sha256": manifest["source_sha256"],
              "preparation_input_sha256": hashes, "estimator_code_sha256": sha256(Path(__file__)),
              "biogeme_version": get_version(), "settings": SETTINGS,
              "native_html_max_parameters": NATIVE_HTML_MAX_PARAMETERS,
              "weight": "W_GEW applied once via Biogeme weight formula; original values",
              "sample_n": qa["final_n"], "references": manifest["references"],
              "utility": "V1 and V2 are model specifications; Car 1 and Car 2 are alternatives. D_k = X_car1,k - X_car2,k; U_car1 = sum(beta_k * D_k); U_car2 = 0; CHOICE 1/2.",
              "editable_selection": {"main_effects": MAIN_EFFECTS,
                                     "interaction_contexts": INTERACTION_CONTEXTS,
                                     "excluded_contexts": EXCLUDE_CONTEXTS,
                                     "excluded_interactions": EXCLUDE_INTERACTIONS,
                                     "added_interactions": ADD_INTERACTIONS},
              "specifications": {model: {"selected_predictors": selected,
                                         "excluded_prepared_predictors": excluded,
                                         "hierarchy_warnings": warnings}
                                 for model, (selected, excluded, warnings) in resolved.items()}}
    write_json(run_dir / "run_configuration.json", config)
    results = {}
    for index, model in enumerate(models):
        selected, _, warnings = resolved[model]
        try:
            results[model] = estimate_model(model, data[model], selected,
                                            run_dir / model, run_id, timestamp, warnings.copy())
        except KeyboardInterrupt:
            results[model] = {"status": "INTERRUPTED", "error": "KeyboardInterrupt during estimation"}
            for not_started in models[index + 1:]:
                results[not_started] = {"status": "INTERRUPTED", "error": "Not started because the run was interrupted"}
            write_json(run_dir / "run_summary.json", results)
            raise SystemExit(130)
        except Exception as error:
            results[model] = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
            (run_dir / model).mkdir(exist_ok=True)
            write_json(run_dir / model / "failure.json", results[model])
            print(f"{model} FAILED: {error}", flush=True)
    write_json(run_dir / "run_summary.json", results)
    if {name: sha256(PREP / name) for name in INPUT_FILES} != hashes:
        raise AssertionError("Preparation input changed during estimation")
    print(run_dir)
    print(json.dumps({model: {"converged": value.get("converged"),
                              "weighted_final_log_likelihood": value.get("weighted_final_log_likelihood"),
                              "status": value["status"]}
                      for model, value in results.items()}, indent=2))
    if any(value["status"] != "SUCCESS" for value in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
