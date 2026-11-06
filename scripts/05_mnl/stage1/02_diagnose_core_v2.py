"""Pre-estimation support and identification audit for isolated Core-V2 inputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import qr

PREPARE = Path(__file__).with_name("01_prepare_core_v2.py")
spec = importlib.util.spec_from_file_location("core_v2_prepare_for_diagnostics", PREPARE)
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)

ROOT = prep.ROOT
DATA = prep.DATA
DIAG = ROOT / "diagnostics"
REVIEW = ROOT / "review"


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    manifest = json.loads((DATA / "core_v2_input_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PREPARED_PENDING_DIAGNOSTICS" or manifest["code_sha256"] != prep.sha256(PREPARE):
        raise AssertionError("Core-V2 preparation code or manifest changed.")
    prep.verify_current_sources(manifest["source_hashes_sha256"])
    for name, expected in manifest["output_hashes_sha256"].items():
        if prep.sha256(DATA / name) != expected:
            raise AssertionError(f"Prepared Core-V2 artifact changed: {name}")
    sample = pd.read_csv(DATA / "core_v2_context_sample.csv",
                         dtype={key: "string" for key in prep.KEYS}, low_memory=False)
    effective = pd.read_csv(DATA / "core_v2_effective_design.csv",
                            dtype={"CHOICE_ID": "string", "H_ID": "string"}, low_memory=False)
    metadata = json.loads((DATA / "core_v2_transformations.json").read_text(encoding="utf-8"))
    if len(sample) != metadata["sample"]["final_n"] or len(effective) != len(sample):
        raise AssertionError("Prepared Core-V2 sample size differs from metadata.")
    if list(effective) != ["CHOICE_ID", "H_ID", *prep.CONTROLS, *prep.PREDICTORS]:
        raise AssertionError("Prepared effective design is not the fixed 27-column schema.")
    if not effective.CHOICE_ID.equals(sample.CHOICE_ID):
        raise AssertionError("Prepared sample and design CHOICE_ID order differs.")
    for field in ("H_ID", *prep.CONTROLS):
        a, b = sample[field], effective[field]
        equal = prep.numeric(a).eq(prep.numeric(b)) if field in prep.CONTROLS else a.eq(b)
        if not equal.all():
            raise AssertionError(f"Prepared control alignment failed: {field}")
    rebuilt = prep.effective_design(sample)
    if not np.allclose(rebuilt.to_numpy(float), effective[list(prep.PREDICTORS)].to_numpy(float),
                       rtol=0, atol=1e-12):
        raise AssertionError("Saved Core-V2 design does not reconstruct.")
    if not np.allclose(rebuilt.to_numpy(float), -prep.effective_design(sample, swap=True).to_numpy(float),
                       rtol=0, atol=1e-12):
        raise AssertionError("Core-V2 label-swap invariance failed.")
    return sample, effective, metadata, manifest


def pre_exclusion_composition(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    matrix = pd.read_csv(prep.SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, dtype=str, keep_default_na=False)
    candidate = pd.read_csv(prep.ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, dtype=str, keep_default_na=False)
    tours = pd.read_csv(prep.TOURS_PATH, usecols=["TOUR_START_SOURCE_ROW_ID", "H_ID", "TOUR_CLOSED",
                                                   "TOUR_DISTANCE_KM", "TOUR_DISTANCE_COMPLETE_FLAG",
                                                   "TOUR_N_TRIPS", "TOUR_N_NONHOME_STOPS",
                                                   "TOUR_N_DISTINCT_PURPOSES", "TOUR_MULTI_PURPOSE_FLAG",
                                                   "TOUR_PURPOSE_UNKNOWN_FLAG",
                                                   "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"],
                        dtype=str, keep_default_na=False)
    prep.validate_matrix_blocks(matrix)
    joined = prep.join_sources(matrix, candidate, tours)
    prep.validate_coding(joined)
    before, reasons = prep.context_and_reasons(joined)
    retained = before.loc[~before[list(reasons)].any(axis=1)]
    if not retained.CHOICE_ID.astype("string").reset_index(drop=True).equals(sample.CHOICE_ID.reset_index(drop=True)):
        raise AssertionError("Reconstructed Core-V2 sample differs from frozen preparation.")
    fields = ["TOUR_MAIN_PURPOSE", "TOUR_N_NONHOME_STOPS", "TOUR_N_DISTINCT_PURPOSES",
              "TOUR_DISTANCE_COMPLETE_FLAG", "TOUR_PURPOSE_UNKNOWN_FLAG", "MULTI_PURPOSE",
              "VERY_HIGH_ECON",
              "HP_SEX", "erwerb", "alter_gr5", "oek_status", "RegioStaR4",
              "POWERTRAIN_1", "POWERTRAIN_2", "STATUS_1", "STATUS_2"]
    rows = []
    for field in fields:
        for value in sorted(set(before[field].astype(str)) | set(sample[field].astype(str))):
            n0 = int(before[field].astype(str).eq(value).sum())
            n1 = int(sample[field].astype(str).eq(value).sum())
            rows.append({"field": field, "category": value, "candidate_n": n0,
                         "candidate_share": n0/len(before), "core_v2_n": n1,
                         "core_v2_share": n1/len(sample)})
    return pd.DataFrame(rows), before, reasons


def support(sample: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    weight = prep.numeric(sample.W_GEW).to_numpy(float)
    choice = prep.numeric(sample.CHOICE).to_numpy(int)
    both = prep.numeric(sample.AV_1).eq(1).to_numpy() & prep.numeric(sample.AV_2).eq(1).to_numpy()
    rows = []
    specifications = [(name, "ALL", name, "main") for name, _, _ in prep.MAIN]
    specifications += [(name, context, target, "interaction") for name, context, target in prep.INTERACTIONS]
    for name, context, target, kind in specifications:
        contrast = design[target].to_numpy(float)
        x = design[name].to_numpy(float)
        active = np.ones(len(sample), bool) if context == "ALL" else prep.numeric(sample[context]).to_numpy(float) != 0
        informative = both & (np.abs(x) > 1e-12)
        contrast_nonzero = both & active & (np.abs(contrast) > 1e-12)
        chosen_target = informative & (((contrast > 0) & (choice == 1)) | ((contrast < 0) & (choice == 2)))
        chosen_other = informative & (((contrast > 0) & (choice == 2)) | ((contrast < 0) & (choice == 1)))
        support_weights = np.sort(weight[informative])[::-1]
        top_count = max(1, int(np.ceil(len(support_weights)*.01)))
        rows.append({
            "predictor": name, "kind": kind, "context": context, "vehicle_target": target,
            "context_active_n": int(active.sum()), "context_active_both_available_n": int((active & both).sum()),
            "nonzero_vehicle_contrast_n": int(contrast_nonzero.sum()),
            "total_target_contrast_n": int((both & (np.abs(contrast) > 1e-12)).sum()),
            "nonzero_effective_predictor_n": int(informative.sum()),
            "supporting_households": int(sample.loc[informative, "H_ID"].nunique()),
            "positive_vehicle_contrast_n": int((informative & (contrast > 0)).sum()),
            "negative_vehicle_contrast_n": int((informative & (contrast < 0)).sum()),
            "chosen_target_n": int(chosen_target.sum()), "chosen_nontarget_n": int(chosen_other.sum()),
            "support_weight_sum": float(weight[informative].sum()),
            "weighted_support_share": float(weight[informative].sum()/weight.sum()),
            "max_support_weight_share": float(support_weights[0]/support_weights.sum()) if len(support_weights) else np.nan,
            "top_1pct_support_weight_share": float(support_weights[:top_count].sum()/support_weights.sum()) if len(support_weights) else np.nan,
            "warning": "NO_IDENTIFYING_SUPPORT" if not informative.any() else
                       "ONE_SIDED_CHOICE" if min(chosen_target.sum(), chosen_other.sum()) == 0 else
                       "LOW_RESPONSE_COUNT" if min(chosen_target.sum(), chosen_other.sum()) < 30 else "",
        })
    return pd.DataFrame(rows)


def distance_support(sample: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    distance = prep.numeric(sample.TOUR_DISTANCE_KM)
    bins = [0, 10, 30, 100, 300, np.inf]
    labels = ["0-10", "10-30", "30-100", "100-300", ">300"]
    band = pd.cut(distance, bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        mask = band.eq(label)
        for target in ("D_PT_BEV", "D_PT_PHEV"):
            contrast = design[target]
            rows.append({"distance_band_km": label, "vehicle_target": target,
                         "occasions": int(mask.sum()), "households": int(sample.loc[mask, "H_ID"].nunique()),
                         "nonzero_contrast": int((mask & contrast.ne(0)).sum()),
                         "weight_sum": float(prep.numeric(sample.loc[mask, "W_GEW"]).sum()),
                         "distance_median_km": float(distance.loc[mask].median()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def exclusion_overlap(before: pd.DataFrame, reasons: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{"reason_a": a, "reason_b": b,
                          "both_n": int((before[a] & before[b]).sum())}
                         for i, a in enumerate(reasons) for b in reasons[i:]])


def multi_purpose_vs_stops(before: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in (("initial", before), ("retained", sample)):
        distinct = prep.numeric(frame.TOUR_N_DISTINCT_PURPOSES).ge(2)
        stops = prep.numeric(frame.TOUR_N_NONHOME_STOPS).ge(2)
        for purpose_value in (0, 1):
            for stop_value in (0, 1):
                mask = distinct.eq(purpose_value) & stops.eq(stop_value)
                rows.append({"sample": label, "multi_purpose": purpose_value,
                             "two_plus_stops": stop_value, "occasions": int(mask.sum()),
                             "weighted_share": float(prep.numeric(frame.loc[mask, "W_GEW"]).sum() /
                                                     prep.numeric(frame.W_GEW).sum())})
    return pd.DataFrame(rows)


def purpose_uncertainty(before: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in (("initial", before), ("retained", sample)):
        count = prep.numeric(frame.TOUR_N_DISTINCT_PURPOSES)
        unknown = prep.numeric(frame.TOUR_PURPOSE_UNKNOWN_FLAG)
        for n in sorted(count.dropna().unique()):
            for flag in (0, 1):
                mask = count.eq(n) & unknown.eq(flag)
                rows.append({"sample": label, "distinct_purposes": int(n),
                             "unknown_purpose_flag": flag, "occasions": int(mask.sum()),
                             "ambiguous_n": int((mask & frame.uncertain_or_invalid_purpose_count).sum())})
    return pd.DataFrame(rows)


def economic_contrast_support(sample: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    weights = prep.numeric(sample.W_GEW).to_numpy(float)
    choice = prep.numeric(sample.CHOICE).to_numpy(int)
    groups = prep.numeric(sample.VERY_HIGH_ECON).to_numpy(int)
    rows = []
    for group in (0, 1):
        group_mask = groups == group
        for target in ("D_PT_BEV", "D_STATUS_HIGH"):
            contrast = design[target].to_numpy(float)
            informative = group_mask & (contrast != 0)
            chosen_target = informative & (((contrast > 0) & (choice == 1)) | ((contrast < 0) & (choice == 2)))
            support_weights = np.sort(weights[informative])[::-1]
            top_count = max(1, int(np.ceil(len(support_weights)*.01)))
            rows.append({"very_high_econ": group, "vehicle_target": target,
                         "group_occasions": int(group_mask.sum()),
                         "group_households": int(sample.loc[group_mask, "H_ID"].nunique()),
                         "both_available_contrasts": int(informative.sum()),
                         "supporting_households": int(sample.loc[informative, "H_ID"].nunique()),
                         "positive_contrasts": int((informative & (contrast > 0)).sum()),
                         "negative_contrasts": int((informative & (contrast < 0)).sum()),
                         "chosen_target": int(chosen_target.sum()),
                         "chosen_nontarget": int(informative.sum()-chosen_target.sum()),
                         "weighted_support_sum": float(weights[informative].sum()),
                         "weighted_support_share": float(weights[informative].sum()/weights.sum()),
                         "top_1pct_support_weight_share": float(support_weights[:top_count].sum()/support_weights.sum())
                         if len(support_weights) else np.nan})
    return pd.DataFrame(rows)


def weighted_composition(sample: pd.DataFrame) -> pd.DataFrame:
    fields = ["TOUR_MAIN_PURPOSE", "TOUR_N_DISTINCT_PURPOSES", "MULTI_PURPOSE",
              "TOUR_DISTANCE_COMPLETE_FLAG", "oek_status", "VERY_HIGH_ECON",
              "HP_SEX", "erwerb", "AGE_UNDER30", "URBAN_REGION",
              "POWERTRAIN_1", "POWERTRAIN_2", "STATUS_1", "STATUS_2"]
    total = prep.numeric(sample.W_GEW).sum()
    rows = []
    for field in fields:
        for category, group in sample.groupby(field, dropna=False):
            rows.append({"field": field, "category": str(category), "occasions": len(group),
                         "weighted_share": float(prep.numeric(group.W_GEW).sum()/total)})
    return pd.DataFrame(rows)


def matrix_diagnostics(design: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = design[list(prep.PREDICTORS)].to_numpy(float)
    if not np.isfinite(x).all():
        raise AssertionError("Nonfinite predictor in effective matrix.")
    singular = np.linalg.svd(x, compute_uv=False)
    norms = np.linalg.norm(x, axis=0)
    if (norms <= 0).any():
        raise AssertionError("At least one Core-V2 predictor has no support.")
    scaled = x/norms
    scaled_singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(x))
    scaled_rank = int(np.linalg.matrix_rank(scaled))
    _, r, pivots = qr(scaled, mode="economic", pivoting=True)
    qrd = [prep.PREDICTORS[i] for i in pivots[scaled_rank:]]
    corr = pd.DataFrame(np.corrcoef(x, rowvar=False), index=prep.PREDICTORS, columns=prep.PREDICTORS)
    vifs = []
    for i, name in enumerate(prep.PREDICTORS):
        y = x[:, i]
        other = np.column_stack([np.ones(len(x)), np.delete(x, i, axis=1)])
        fitted = other @ np.linalg.lstsq(other, y, rcond=None)[0]
        total = float(((y-y.mean())**2).sum())
        r2 = 1-float(((y-fitted)**2).sum())/total if total > 0 else np.nan
        vifs.append({"predictor": name, "VIF": 1/(1-r2) if np.isfinite(r2) and r2 < 1 else np.inf,
                     "auxiliary_R2": r2})
    pairs = [{"left": a, "right": b, "correlation": float(corr.loc[a, b])}
             for i, a in enumerate(prep.PREDICTORS) for b in prep.PREDICTORS[i+1:]]
    metrics = {"n": len(x), "p": len(prep.PREDICTORS), "rank": rank, "scaled_rank": scaled_rank,
               "raw_condition": float(singular[0]/singular[-1]),
               "scaled_condition": float(scaled_singular[0]/scaled_singular[-1]),
               "smallest_singular_value": float(singular[-1]),
               "qr_dependent_candidates": qrd,
               "max_vif": float(max(row["VIF"] for row in vifs)),
               "near_zero_variance_predictors": [name for name, std in zip(prep.PREDICTORS, x.std(axis=0))
                                                 if std < 1e-5]}
    values = pd.DataFrame({"index": range(1, len(singular)+1), "raw_singular_value": singular,
                           "scaled_singular_value": scaled_singular})
    return metrics, values, pd.DataFrame(vifs), pd.DataFrame(pairs)


def diagnose() -> dict:
    if DIAG.exists() or REVIEW.exists():
        raise FileExistsError("Core-V2 diagnostic/review output collision.")
    sample, effective, metadata, manifest = load_prepared()
    design = effective[list(prep.PREDICTORS)]
    comp, before, reasons = pre_exclusion_composition(sample)
    supports = support(sample, design)
    bands = distance_support(sample, design)
    overlaps = exclusion_overlap(before, reasons)
    purpose_stops = multi_purpose_vs_stops(before, sample)
    uncertainty = purpose_uncertainty(before, sample)
    economic = economic_contrast_support(sample, design)
    weighted = weighted_composition(sample)
    metrics, singular, vif, corr = matrix_diagnostics(design)
    distance = prep.numeric(sample.TOUR_DISTANCE_KM)
    metrics["distance_km"] = {key: float(value) for key, value in
                              distance.describe(percentiles=[.5,.95,.99]).to_dict().items()}
    metrics["weight_sum"] = float(prep.numeric(sample.W_GEW).sum())
    metrics["weight_max"] = float(prep.numeric(sample.W_GEW).max())
    metrics["households"] = int(sample.H_ID.nunique())
    metrics["initial_incomplete_distance_n"] = int(before.incomplete_tour_distance.sum())
    metrics["initial_ambiguous_purpose_n"] = int(before.uncertain_or_invalid_purpose_count.sum())
    metrics["retained_multi_purpose_n"] = int(prep.numeric(sample.MULTI_PURPOSE).eq(1).sum())
    blockers = []
    if metrics["rank"] != 21 or metrics["scaled_rank"] != 21:
        blockers.append(f"Rank {metrics['rank']}/{metrics['scaled_rank']} rather than 21/21.")
    if supports.nonzero_effective_predictor_n.eq(0).any():
        blockers.append("An active predictor has no identifying support.")
    if supports.warning.eq("ONE_SIDED_CHOICE").any():
        blockers.append("A predictor has one-sided target choice among informative contrasts.")
    if not np.isfinite(metrics["scaled_condition"]) or metrics["scaled_condition"] > 1e8:
        blockers.append("Severe scaled design conditioning requires review.")
    warnings = [f"{r.predictor}: {r.warning}" for r in supports.itertuples() if r.warning]
    warnings += [f"{r.predictor}: VIF {r.VIF:.3g} (descriptive)" for r in vif.itertuples() if r.VIF >= 5]
    warnings += [f"{name}: near-zero variance" for name in metrics["near_zero_variance_predictors"]]
    status = "BLOCKED" if blockers else "PASS_NO_BLOCKER"
    report = ["# Core-V2 pre-estimation QA", "",
              f"Gate: **{status}**. Fixed specification: 5 vehicle contrasts + 16 interactions.", "",
              f"Sample: {len(sample):,} occasions, {sample.H_ID.nunique():,} households; all have both alternatives available.",
              f"Stage-0 V2-valid: {metadata['sample']['stage0_v2_valid_n']:,} of {metadata['sample']['initial_n']:,}.",
              f"Official Main Purpose missing/invalid exclusion: {pd.read_csv(DATA/'core_v2_sample_flow.csv').set_index('metric').loc['missing_or_invalid_main_purpose','count']:,} (non-exclusive).",
              f"MULTI_PURPOSE is based on seven distinct substantive hwzweck1 purposes at non-home destinations; 2+ purposes={metrics['retained_multi_purpose_n']:,}. Repeated purposes count once. The four TOUR_PURPOSE_COUNT_* indicators and TOUR_MULTI_PURPOSE_FLAG agree with the authoritative count.",
              f"Uncertain/invalid purpose-count exclusions: {metrics['initial_ambiguous_purpose_n']:,} among initial candidates. Zero identified purposes or one with unknown non-home purpose is excluded; 2+ valid purposes remains established even if another purpose is unknown. See uncertainty and stop-comparison tables.",
              "Main Purpose selected categories are Work, Escort, Shopping. The pooled reference is Business, Education, Errand, Leisure.",
              f"Distance completeness flag requires every constituent trip distance to be valid; initial incomplete tours={metrics['initial_incomplete_distance_n']:,}. Distance mean for centering: {metadata['distance_mean_km']:.9f} km. Median={distance.median():.3f}, P95={distance.quantile(.95):.3f}, P99={distance.quantile(.99):.3f}, max={distance.max():.3f} km; no valid extreme was removed.",
              "VERY_HIGH_ECON is oek_status 5 versus valid 1–4; no ordinal linear slope. Status class is a KBA segment/year composite, not measured capacity.", "",
              "## Identification", "",
              f"Rank {metrics['rank']}/21 raw and {metrics['scaled_rank']}/21 scaled; raw condition {metrics['raw_condition']:.6g}; scaled condition {metrics['scaled_condition']:.6g}; max VIF {metrics['max_vif']:.6g}.",
              f"QR dependent candidates: {', '.join(metrics['qr_dependent_candidates']) or 'none'}.",
              f"Blockers: {'; '.join(blockers) if blockers else 'none'}.", "",
              "## Warnings", "", *([f"- {warning}" for warning in warnings] or ["- None."]), "",
              "Detailed composition, exclusion overlap, purpose uncertainty, multi-purpose versus multi-stop, economic support, distance bands, singular values, VIF and correlations are saved as CSVs. VIF and low-count flags are descriptive; no term is dropped automatically.", ""]
    checklist = ["# Core-V2 pre-estimation review checklist", "",
                 "- [x] Current shared inputs and Stage-0 V2 coding verified.",
                 "- [x] Matrix/allocation/tour joins and control alignment verified.",
                 "- [x] Official Main Purpose, purpose uncertainty, economic status and distance completeness checked.",
                 "- [x] Exactly 21 finite, label-invariant predictors reconstructed.",
                 f"- [{'x' if not blockers else ' '}] Rank, support, and numerical gate reviewed: {status}.",
                 "- [ ] Interpret coefficients as observational associations after estimation.",
                 "- [ ] Review sensitivity to the prespecified linear distance form after estimation.", ""]
    DIAG.mkdir(parents=True, exist_ok=False)
    REVIEW.mkdir(parents=True, exist_ok=False)
    for name, frame in [("context_composition", comp), ("weighted_context_composition", weighted),
                        ("sample_exclusion_overlap", overlaps),
                        ("multi_purpose_vs_multi_stop", purpose_stops),
                        ("purpose_uncertainty", uncertainty),
                        ("economic_contrast_support", economic),
                        ("predictor_support", supports),
                        ("distance_support", bands), ("singular_values", singular),
                        ("vif", vif), ("correlations", corr)]:
        frame.to_csv(DIAG/f"core_v2_{name}.csv", index=False)
    (DIAG/"core_v2_matrix_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    (REVIEW/"core_v2_pre_estimation_QA.md").write_text("\n".join(report), encoding="utf-8")
    (REVIEW/"core_v2_manual_review_checklist.md").write_text("\n".join(checklist), encoding="utf-8")
    gate = {"status": status, "blockers": blockers, "warnings": warnings,
            "prepared_input_manifest_sha256": prep.sha256(DATA/"core_v2_input_manifest.json"),
            "prepare_code_sha256": prep.sha256(PREPARE), "diagnostic_code_sha256": prep.sha256(Path(__file__)),
            "matrix_metrics": metrics,
            "diagnostic_hashes_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): prep.sha256(p)
                                          for p in sorted((*DIAG.glob("*"), *REVIEW.glob("*.md")))} }
    (REVIEW/"core_v2_diagnostic_gate.json").write_text(json.dumps(gate, indent=2)+"\n", encoding="utf-8")
    return gate


if __name__ == "__main__":
    outcome = diagnose()
    print(json.dumps({key: outcome[key] for key in ("status", "blockers", "matrix_metrics")}, indent=2))
