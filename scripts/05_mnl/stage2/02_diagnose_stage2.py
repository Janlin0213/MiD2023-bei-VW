"""Diagnose the frozen Stage 2 utility-difference designs before estimation.

Reports support, Pearson correlations, intercept-auxiliary VIF, raw design
rank, and weighted column-normalized conditioning. Never selects predictors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import svd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from src.thesis_pipeline.paths import OUTPUTS_DIR, REPO_ROOT  # noqa: E402


PREP = OUTPUTS_DIR / "mnl" / "stage2" / "preparation"
OUT = OUTPUTS_DIR / "mnl" / "stage2" / "diagnostics"
CONTROLS = ("CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID",
            "CHOICE", "AV_1", "AV_2", "W_GEW")
CORRELATION_WARNING = 0.8
VIF_WARNING = 10.0
LOW_SUPPORT_WARNING = 30  # descriptive flag only; never an exclusion rule
CONDITION_WARNING = 1e8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict, dict, dict]:
    names = ("stage2_common_pool.csv", "stage2_v1_design.csv", "stage2_v2_design.csv",
             "stage2_feature_manifest.json", "stage2_preparation_qa.json")
    paths = {name: PREP / name for name in names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = json.loads(paths[names[3]].read_text(encoding="utf-8"))
    qa = json.loads(paths[names[4]].read_text(encoding="utf-8"))
    for name in names[:3]:
        if hashes[name] != qa["output_sha256"][name]:
            raise ValueError(f"Preparation fingerprint mismatch: {name}")
    if manifest["references"] != {"powertrain": "ICE", "segment_v1": "SMALL",
                                   "status_v2": "MEDIUM", "driver_age_context": "30-49",
                                   "sex_context": "MALE"}:
        raise ValueError("Unexpected Stage 2 reference conventions")
    tables = {name: pd.read_csv(paths[name], dtype={"CHOICE_ID": str, "H_ID": str})
              for name in names[:3]}
    common = tables[names[0]]
    designs = {"v1": tables[names[1]], "v2": tables[names[2]]}
    if len(common) != qa["final_n"] or common.CHOICE_ID.isna().any() or not common.CHOICE_ID.is_unique:
        raise ValueError("Invalid frozen common pool")
    for model, design in designs.items():
        specs = manifest[f"{model}_features"]
        features = [row["feature"] for row in specs]
        if len(features) != len(set(features)) or list(design.columns) != [*CONTROLS, *features]:
            raise ValueError(f"{model}: predictor columns differ from manifest")
        if len(design) != len(common) or not design[list(CONTROLS)].equals(common[list(CONTROLS)]):
            raise ValueError(f"{model}: ordered controls differ from common pool")
        if not np.isfinite(design[features].to_numpy(float)).all():
            raise ValueError(f"{model}: missing or non-finite predictor")
        if sum(row["kind"] == "interaction" for row in specs) != qa[f"{model}_interactions"]:
            raise ValueError(f"{model}: interaction count differs from preparation QA")
        if sum(row["kind"] == "main" for row in specs) != qa[f"{model}_main"]:
            raise ValueError(f"{model}: main-effect count differs from preparation QA")
    if not common.CHOICE.isin([1, 2]).all() or not common[["AV_1", "AV_2"]].eq(1).all().all():
        raise ValueError("Invalid choice or availability")
    if not np.isfinite(common.W_GEW).all() or not common.W_GEW.gt(0).all():
        raise ValueError("Invalid survey weights")
    return common, designs, manifest, qa, hashes


def interaction_support(common: pd.DataFrame, design: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        if spec["kind"] != "interaction":
            continue
        context = spec["context"]
        values = common[context].to_numpy(float)
        binary = np.isin(values, [0, 1]).all() and context != "TOUR_TIME_C_H"
        active = int(np.count_nonzero(values == 1)) if binary else None
        informative = design[spec["feature"]].to_numpy(float) != 0
        chosen = design.loc[informative, "CHOICE"]
        count = int(informative.sum())
        rows.append({"feature": spec["feature"], "context": context,
                     "vehicle_attribute": spec["vehicle_attribute"],
                     "context_type": "binary" if binary else "continuous",
                     "context_active_n": active,
                     "informative_n": count,
                     "informative_households": int(design.loc[informative, "H_ID"].nunique()),
                     "informative_weight_sum": float(design.loc[informative, "W_GEW"].sum()),
                     "choice_1_n": int(chosen.eq(1).sum()),
                     "choice_2_n": int(chosen.eq(2).sum()),
                     "support_flag": "ZERO" if count == 0 else
                                     "LOW" if count < LOW_SUPPORT_WARNING else "OK"})
    return pd.DataFrame(rows)


def correlation_table(design: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = design[features].to_numpy(float)
    centered = x - x.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    constant = [features[i] for i in np.flatnonzero(norms == 0)]
    rows = []
    valid = np.flatnonzero(norms > 0)
    z = centered[:, valid] / norms[valid]
    corr = z.T @ z
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            value = float(corr[a, b])
            if abs(value) >= CORRELATION_WARNING:
                rows.append((features[valid[a]], features[valid[b]], value))
    # Also preserve the ten strongest positive and negative pairs when below threshold.
    upper = np.triu_indices(len(valid), k=1)
    pairs = sorted(zip(corr[upper], upper[0], upper[1]), key=lambda row: row[0])
    for value, a, b in pairs[:10] + pairs[-10:]:
        rows.append((features[valid[a]], features[valid[b]], float(value)))
    unique = {(a, b): r for a, b, r in rows}
    table = pd.DataFrame([{"feature_a": a, "feature_b": b, "pearson_r": r,
                           "abs_r": abs(r), "high_correlation": abs(r) >= CORRELATION_WARNING}
                          for (a, b), r in unique.items()])
    if not table.empty:
        table = table.sort_values(["abs_r", "feature_a", "feature_b"],
                                  ascending=[False, True, True]).reset_index(drop=True)
    return table, constant


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    _, singular, vectors = svd(x, full_matrices=False, lapack_driver="gesdd")
    tolerance = float(np.finfo(float).eps * max(x.shape) * singular[0])
    return singular, vectors, int((singular > tolerance).sum()), tolerance


def rank_and_condition(design: pd.DataFrame, features: list[str]) -> dict:
    x = design[features].to_numpy(float)
    singular, vectors, rank, tolerance = spectrum(x)
    zero_columns = [features[i] for i in np.flatnonzero(np.linalg.norm(x, axis=0) == 0)]
    dependencies = []
    if rank < x.shape[1]:
        for i in np.flatnonzero(singular <= tolerance)[:5]:
            direction = vectors[i]
            top = np.argsort(np.abs(direction))[-8:][::-1]
            dependencies.append({"singular_value": float(singular[i]),
                                 "leading_terms": [{"feature": features[j],
                                                    "coefficient": float(direction[j])}
                                                   for j in top if abs(direction[j]) > 1e-6]})
    # Weighted columns are divided by their Euclidean norm. No demeaning or ASC.
    weighted = x * np.sqrt(design.W_GEW.to_numpy(float))[:, None]
    scales = np.linalg.norm(weighted, axis=0)
    scaled = np.divide(weighted, scales, out=np.zeros_like(weighted), where=scales > 0)
    scaled_singular, scaled_vectors, scaled_rank, scaled_tolerance = spectrum(scaled)
    condition = (float(scaled_singular[0] / scaled_singular[-1])
                 if scaled_rank == x.shape[1] else None)
    weakest = np.argsort(np.abs(scaled_vectors[-1]))[-8:][::-1]
    return {"n": x.shape[0], "predictors": x.shape[1], "rank": rank,
            "rank_deficiency": x.shape[1] - rank, "rank_tolerance": tolerance,
            "largest_singular": float(singular[0]), "smallest_singular": float(singular[-1]),
            "zero_columns": zero_columns, "dependencies": dependencies,
            "condition_convention": "sqrt(W_GEW) * X; each column divided by its weighted L2 norm; no demeaning or ASC",
            "scaled_rank": scaled_rank, "scaled_tolerance": scaled_tolerance,
            "scaled_largest_singular": float(scaled_singular[0]),
            "scaled_smallest_singular": float(scaled_singular[-1]),
            "condition_number": condition,
            "unstable_direction": [{"feature": features[j],
                                    "coefficient": float(scaled_vectors[-1, j])}
                                   for j in weakest]}


def vif_table(design: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, str]:
    x = design[features].to_numpy(float)
    centered = x - x.mean(axis=0)  # intercept in each auxiliary regression
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms == 0):
        reason = "Centered design has constant predictor(s); intercept-auxiliary VIF undefined."
        values = [None] * len(features)
    else:
        standardized = centered / norms
        singular, vectors, rank, _ = spectrum(standardized)
        if rank < len(features):
            reason = "Centered auxiliary design is rank-deficient; VIF undefined without dropping predictors."
            values = [None] * len(features)
        else:
            reason = "Intercept-auxiliary OLS: center and unit-normalize each predictor; VIF is diagonal of inverse correlation matrix."
            values = np.sum((vectors.T / singular) ** 2, axis=1).tolist()
    return pd.DataFrame({"feature": features, "vif": values,
                         "high_vif": [v is not None and v > VIF_WARNING for v in values],
                         "status": ["UNDEFINED" if v is None else "WARNING" if v > VIF_WARNING else "OK"
                                    for v in values]}), reason


def main() -> None:
    common, designs, manifest, qa, input_hashes = read_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for model, design in designs.items():
        specs = manifest[f"{model}_features"]
        features = [row["feature"] for row in specs]
        support = interaction_support(common, design, specs)
        correlations, constant = correlation_table(design, features)
        structure = rank_and_condition(design, features)  # before VIF
        vif, vif_convention = vif_table(design, features)
        support.to_csv(OUT / f"{model}_interaction_support.csv", index=False)
        correlations.to_csv(OUT / f"{model}_correlations.csv", index=False)
        vif.to_csv(OUT / f"{model}_vif.csv", index=False)
        warnings = []
        if support.support_flag.ne("OK").any(): warnings.append("low interaction support")
        if not correlations.empty and correlations.high_correlation.any(): warnings.append("high pairwise correlation")
        if vif.high_vif.any(): warnings.append("VIF > 10")
        if structure["condition_number"] is not None and structure["condition_number"] > CONDITION_WARNING:
            warnings.append("large scaled condition number")
        if structure["rank_deficiency"]: warnings.append("rank-deficient predictor matrix")
        if constant: warnings.append("constant predictor(s)")
        results[model] = {"classification": "BLOCK" if structure["rank_deficiency"] else
                          "WARNING" if warnings else "PASS", "warnings": warnings,
                          "main_predictors": sum(row["kind"] == "main" for row in specs),
                          "interactions": len(support), "zero_support": int(support.support_flag.eq("ZERO").sum()),
                          "low_support": int(support.support_flag.eq("LOW").sum()),
                          "lowest_support": support.nsmallest(5, "informative_n")[
                              ["feature", "informative_n", "informative_households"]].to_dict("records"),
                          "high_correlation_pairs": int(correlations.high_correlation.sum()) if not correlations.empty else 0,
                          "strongest_positive": correlations.nlargest(5, "pearson_r").to_dict("records"),
                          "strongest_negative": correlations.nsmallest(5, "pearson_r").to_dict("records"),
                          "constant_predictors": constant, "vif_convention": vif_convention,
                          "highest_vif": vif.dropna(subset=["vif"]).nlargest(10, "vif")[
                              ["feature", "vif"]].to_dict("records"),
                          "structure": structure}
    if {name: sha256(PREP / name) for name in input_hashes} != input_hashes:
        raise AssertionError("Preparation input changed during diagnostics")
    summary = {"source": "frozen Stage 2 preparation outputs", "input_sha256": input_hashes,
               "upstream_source_sha256": manifest["source_sha256"],
               "preparation_code_sha256": manifest["code_sha256"],
               "diagnostics_code_sha256": sha256(Path(__file__)),
               "thresholds_for_warning_only": {"interaction_informative_n_below": LOW_SUPPORT_WARNING,
                                               "absolute_pearson_at_least": CORRELATION_WARNING,
                                               "vif_above": VIF_WARNING,
                                               "scaled_condition_above": CONDITION_WARNING},
               "models": results}
    (OUT / "stage2_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Stage 2 preparation diagnostics", "",
             "Five read-only diagnostic categories; no predictors were removed.", ""]
    for model, result in results.items():
        s = result["structure"]
        lines += [f"## {model.upper()} — {result['classification']}", "",
                  f"- Interaction support: {result['zero_support']} zero and {result['low_support']} low (<{LOW_SUPPORT_WARNING}) of {result['interactions']}.",
                  f"- Weakest interaction support: {result['lowest_support'][0]['feature']} ({result['lowest_support'][0]['informative_n']} occasions).",
                  f"- Pairwise correlation: {result['high_correlation_pairs']} pairs with |r| ≥ {CORRELATION_WARNING}.",
                  f"- Strongest positive: {result['strongest_positive'][0]['feature_a']} / {result['strongest_positive'][0]['feature_b']} (r={result['strongest_positive'][0]['pearson_r']:.4f}).",
                  f"- Strongest negative: {result['strongest_negative'][0]['feature_a']} / {result['strongest_negative'][0]['feature_b']} (r={result['strongest_negative'][0]['pearson_r']:.4f}).",
                  f"- VIF: {result['vif_convention']}",
                  f"- Highest VIF: {result['highest_vif'][0]['feature']} ({result['highest_vif'][0]['vif']:.2f})." if result['highest_vif'] else "- Highest VIF: undefined.",
                  f"- Matrix rank: {s['rank']}/{s['predictors']}; tolerance {s['rank_tolerance']:.6g}.",
                  f"- Condition number: {s['condition_number'] if s['condition_number'] is not None else 'infinite/undefined (rank deficiency)'}; weighted column-normalized without demeaning.",
                  "- Weakest scaled direction: " + ", ".join(term["feature"] for term in s["unstable_direction"][:3]) + ".",
                  f"- Warnings: {', '.join(result['warnings']) if result['warnings'] else 'none'}.", ""]
    (OUT / "stage2_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({model: {"classification": r["classification"],
                              "rank": r["structure"]["rank"],
                              "predictors": r["structure"]["predictors"],
                              "condition_number": r["structure"]["condition_number"]}
                      for model, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
