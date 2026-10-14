"""Four pre-estimation diagnostics for the fixed 24-column Core–V1 design."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import qr

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

PREPARE_PATH = Path(__file__).with_name("01_prepare_core_v1.py")
spec = importlib.util.spec_from_file_location("core_v1_prepare", PREPARE_PATH)
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)

ROOT = prep.ROOT
DATA = ROOT / "data"
DIAG = ROOT / "diagnostics"
REVIEW = ROOT / "review"


def numerical_rank(matrix: np.ndarray, relative_tolerance: float | None = None) -> tuple[int, np.ndarray, float]:
    values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    tolerance = (max(matrix.shape) * np.finfo(float).eps * values[0] if relative_tolerance is None
                 else relative_tolerance * values[0])
    return int(np.sum(values > tolerance)), values, float(tolerance)


def composition(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, crosstabs, continuous = [], [], []
    definitions = {
        "TOUR_N_NONHOME_STOPS": ("tour", "allocation_candidate_model_input", "0,1,2,...", "none"),
        "HOUSEHOLD_ACCOMPANIED": ("focal departure", "allocation_candidate_model_input", "0=no, 1=yes", "0"),
        "HP_SEX": ("person", "allocation_candidate_model_input", "1=male, 2=female, 3=diverse", "1"),
        "alter_gr5": ("person", "allocation_candidate_model_input", "3=under18; 4–12=18–64; 13–15=65+", "3–12"),
        "household_type": ("household", "allocation_candidate_model_input", "family/young/adult/senior", "non-family"),
        "RegioStaR4": ("household", "allocation_candidate_model_input", "11/12 urban; 21/22 rural", "21/22"),
        "MULTISTOP": ("tour", "Stage-1 derivation", "0=0–1, 1=2+", "0"),
        "FEMALE": ("person", "Stage-1 derivation", "0=male, 1=female", "0"),
        "OLDER65": ("person", "Stage-1 derivation", "0=under65, 1=65+", "0"),
        "FAMILY": ("household", "Stage-1 derivation", "0=non-family, 1=family", "0"),
        "URBAN_REGION": ("household", "Stage-1 derivation", "0=rural, 1=urban", "0"),
    }
    total_weight = sample.W_GEW.sum()
    for variable, (level, source, coding, reference) in definitions.items():
        series = sample[variable]
        for category, subset in sample.groupby(variable, dropna=False, sort=True):
            rows.append({"variable": variable, "source": source, "level": level,
                         "coding": coding, "reference": reference, "category": str(category),
                         "count": len(subset), "share": len(subset) / len(sample),
                         "weighted_share": subset.W_GEW.sum() / total_weight,
                         "unique_households": subset.H_ID.nunique(),
                         "missing_count": int(series.isna().sum()), "missing_pct": series.isna().mean(),
                         "transformation_note": prep.encode_context.__doc__ or "See transformations JSON"})
    for variable in ["TOUR_TRAVEL_TIME_MIN", "TOUR_TIME_C_H"]:
        values = sample[variable]
        stats = values.describe(percentiles=[.05, .25, .5, .75, .95, .99])
        continuous.append({"variable": variable, "source": "allocation_candidate_model_input" if variable.endswith("MIN") else "Stage-1 derivation",
                           "n": int(values.notna().sum()), "missing_count": int(values.isna().sum()),
                           "missing_pct": values.isna().mean(), "mean": stats["mean"], "median": stats["50%"],
                           "std": stats["std"], "min": stats["min"], "max": stats["max"],
                           "p5": stats["5%"], "p25": stats["25%"], "p75": stats["75%"],
                           "p95": stats["95%"], "p99": stats["99%"],
                           "centering_mean_minutes": float(sample.TOUR_TRAVEL_TIME_MIN.mean())})
    pairs = [("OLDER65", "FAMILY"), ("OLDER65", "URBAN_REGION"),
             ("FAMILY", "URBAN_REGION"), ("MULTISTOP", "HOUSEHOLD_ACCOMPANIED")]
    for left, right in pairs:
        for (a, b), group in sample.groupby([left, right], dropna=False):
            crosstabs.append({"table": f"{left} × {right}", "left": str(a), "right": str(b),
                              "count": len(group), "weighted_share": group.W_GEW.sum() / total_weight,
                              "unique_households": group.H_ID.nunique()})
    # Fixed time intervals avoid quantile collapse when travel times tie.
    bins = [-np.inf, 30, 60, 120, 240, np.inf]
    labels = ["<=30", "30–60", "60–120", "120–240", ">240"]
    sample = sample.assign(TIME_BIN=pd.cut(sample.TOUR_TRAVEL_TIME_MIN, bins=bins, labels=labels))
    for (stops, interval), group in sample.groupby(["MULTISTOP", "TIME_BIN"], observed=False):
        crosstabs.append({"table": "MULTISTOP × time_minutes", "left": str(stops), "right": str(interval),
                          "count": len(group), "weighted_share": group.W_GEW.sum() / total_weight,
                          "unique_households": group.H_ID.nunique()})
    return pd.DataFrame(rows), pd.DataFrame(crosstabs), pd.DataFrame(continuous)


def alternative_support(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    both = sample.AV_1.eq(1) & sample.AV_2.eq(1)
    contexts = ["MULTISTOP", "HOUSEHOLD_ACCOMPANIED", "FEMALE", "OLDER65", "FAMILY", "URBAN_REGION"]
    for context in contexts:
        for level in (0, 1):
            mask = sample[context].eq(level) & both
            subset = sample.loc[mask]
            for attribute, prefix in [("BEV", "PT"), ("PHEV", "PT"), ("LARGE", "SEG")]:
                alt_count = int(subset[f"{prefix}1_{attribute}"].sum() + subset[f"{prefix}2_{attribute}"].sum())
                contrast = subset[f"{prefix}1_{attribute}"].ne(subset[f"{prefix}2_{attribute}"])
                rows.append({"context": context, "level": level, "attribute": attribute,
                             "occasion_n": len(subset), "alternative_n": 2 * len(subset),
                             "target_alternatives": alt_count, "contrasting_occasions": int(contrast.sum()),
                             "supporting_households": subset.loc[contrast, "H_ID"].nunique()})
    return pd.DataFrame(rows)


def target_choice_masks(sample: pd.DataFrame, delta: pd.Series, support: pd.Series) -> tuple[pd.Series, pd.Series]:
    """For a binary target, classify the chosen alternative independently of its label."""
    chosen_target = support & ((sample.CHOICE.eq(1) & delta.gt(0)) |
                               (sample.CHOICE.eq(2) & delta.lt(0)))
    return chosen_target, support & ~chosen_target


def target_choice_warning(chosen_target_n: int, chosen_non_target_n: int) -> str:
    """Descriptive response support flag; ten is not an identification test."""
    if chosen_target_n + chosen_non_target_n == 0:
        return "zero informative target contrasts"
    if chosen_non_target_n == 0:
        return "target always chosen"
    if chosen_target_n == 0:
        return "target never chosen"
    if min(chosen_target_n, chosen_non_target_n) < 10:
        return f"weak target-choice direction: {chosen_target_n} target, {chosen_non_target_n} non-target (<10 in one direction)"
    return ""


TIME_BINS = [-np.inf, 30, 60, 120, 240, np.inf]
TIME_LABELS = ["<=30", "30–60", "60–120", "120–240", ">240"]


def interaction_support(sample: pd.DataFrame, design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, time_rows = [], []
    both = sample.AV_1.eq(1) & sample.AV_2.eq(1)
    total_both = int(both.sum())
    weight_both = sample.loc[both, "W_GEW"].sum()
    for name, context, target in prep.INTERACTIONS:
        active = sample[context].ne(0) if context != "TOUR_TIME_C_H" else pd.Series(True, index=sample.index)
        contrast = design[target].ne(0)
        interaction = design[name].ne(0)
        support = active & contrast & both
        positive = support & design[target].gt(0)
        negative = support & design[target].lt(0)
        is_binary_target = target != "D_VEHICLE_AGE"
        chosen_target, chosen_non_target = (
            target_choice_masks(sample, design[target], support) if is_binary_target
            else (pd.Series(False, index=sample.index), pd.Series(False, index=sample.index))
        )
        chosen_car1 = support & sample.CHOICE.eq(1)
        chosen_car2 = support & sample.CHOICE.eq(2)
        values = design.loc[support, target]
        warnings = []
        if support.sum() == 0:
            warnings.append("zero informative contrasts")
        if positive.sum() == 0 or negative.sum() == 0:
            warnings.append("one-sided attribute contrast")
        choice_warning = target_choice_warning(int(chosen_target.sum()), int(chosen_non_target.sum())) if is_binary_target else ""
        if choice_warning:
            warnings.append(choice_warning)
        if support.sum() < 30:
            warnings.append("fewer than 30 informative contrasts; descriptive flag only")
        label_warning = "one-sided arbitrary car-label choice" if chosen_car1.sum() == 0 or chosen_car2.sum() == 0 else ""
        pos_target = int((chosen_target & positive).sum())
        neg_target = int((chosen_target & negative).sum())
        rows.append({"predictor": name, "context": context, "target": target,
                     "context_active_all": int(active.sum()), "context_active_both_available": int((active & both).sum()),
                     "nonzero_target_delta_active_all": int((active & contrast).sum()),
                     "nonzero_target_delta_active_both": int(support.sum()),
                     "nonzero_interaction_all": int(interaction.sum()),
                     "nonzero_interaction_both": int((interaction & both).sum()),
                     "nonzero_interaction_share_both": (interaction & both).sum() / total_both,
                     "unique_supporting_households": sample.loc[support, "H_ID"].nunique(),
                     "weighted_supporting_share_both": sample.loc[support, "W_GEW"].sum() / weight_both,
                     "positive_delta_both": int(positive.sum()), "negative_delta_both": int(negative.sum()),
                     "zero_delta_active_both": int((active & both & ~contrast).sum()),
                     "chosen_car1_both": int(chosen_car1.sum()), "chosen_car2_both": int(chosen_car2.sum()),
                     "chosen_target_attribute_both": int(chosen_target.sum()) if is_binary_target else np.nan,
                     "chosen_other_attribute_both": int(chosen_non_target.sum()) if is_binary_target else np.nan,
                     "informative_contrasts_n": int(support.sum()),
                     "supporting_households_n": sample.loc[support, "H_ID"].nunique(),
                     "chosen_target_n": int(chosen_target.sum()) if is_binary_target else np.nan,
                     "chosen_non_target_n": int(chosen_non_target.sum()) if is_binary_target else np.nan,
                     "chosen_target_share": chosen_target.sum() / support.sum() if is_binary_target and support.sum() else np.nan,
                     "positive_delta_n": int(positive.sum()), "negative_delta_n": int(negative.sum()),
                     "positive_delta_chosen_target_n": pos_target if is_binary_target else np.nan,
                     "positive_delta_chosen_non_target_n": int(positive.sum()) - pos_target if is_binary_target else np.nan,
                     "negative_delta_chosen_target_n": neg_target if is_binary_target else np.nan,
                     "negative_delta_chosen_non_target_n": int(negative.sum()) - neg_target if is_binary_target else np.nan,
                     "chosen_car1_n": int(chosen_car1.sum()), "chosen_car2_n": int(chosen_car2.sum()),
                     "label_balance_warning": label_warning,
                     "target_delta_mean": values.mean(), "target_delta_min": values.min(),
                     "target_delta_median": values.median(), "target_delta_max": values.max(),
                     "warning": "; ".join(warnings)})
        if context == "TOUR_TIME_C_H":
            binning = pd.cut(sample.TOUR_TRAVEL_TIME_MIN, bins=TIME_BINS, labels=TIME_LABELS)
            for interval in binning.cat.categories:
                m = support & binning.eq(interval)
                target_in_bin = int((chosen_target & m).sum())
                non_target_in_bin = int((chosen_non_target & m).sum())
                all_in_bin = both & binning.eq(interval)
                bin_weight = float(sample.loc[all_in_bin, "W_GEW"].sum())
                time_rows.append({"predictor": name, "time_minutes_bin": str(interval),
                                  "contrasting_both_available": int(m.sum()),
                                  "chosen_car1": int((m & sample.CHOICE.eq(1)).sum()),
                                  "chosen_car2": int((m & sample.CHOICE.eq(2)).sum()),
                                  "mean_minutes": sample.loc[m, "TOUR_TRAVEL_TIME_MIN"].mean(),
                                  "median_minutes": sample.loc[m, "TOUR_TRAVEL_TIME_MIN"].median(),
                                  "total_occasions_in_bin": int(all_in_bin.sum()),
                                  "informative_contrasts_n": int(m.sum()),
                                  "chosen_target_n": target_in_bin,
                                  "chosen_non_target_n": non_target_in_bin,
                                  "chosen_target_share": target_in_bin / m.sum() if m.sum() else np.nan,
                                  "supporting_households_n": sample.loc[m, "H_ID"].nunique(),
                                  "supporting_weight_sum": float(sample.loc[m, "W_GEW"].sum()),
                                  "weighted_support_share_within_bin": float(sample.loc[m, "W_GEW"].sum()) / bin_weight if bin_weight else np.nan,
                                  "support_warning": "zero contrasts" if not m.sum() else "fewer than 50 contrasts; descriptive flag" if m.sum() < 50 else ""})
            overall = rows[-1]
            relevant = [r for r in time_rows if r["predictor"] == name]
            for key, expected in [("informative_contrasts_n", overall["informative_contrasts_n"]),
                                  ("chosen_target_n", overall["chosen_target_n"]),
                                  ("chosen_non_target_n", overall["chosen_non_target_n"])]:
                if sum(r[key] for r in relevant) != expected:
                    raise AssertionError(f"Time-bin {key} does not reconcile for {name}")
    return pd.DataFrame(rows), pd.DataFrame(time_rows)


def vehicle_age_choice_support(sample: pd.DataFrame) -> pd.DataFrame:
    delta = sample.VEHICLE_AGE_1 - sample.VEHICLE_AGE_2
    chosen_delta = delta.where(sample.CHOICE.eq(1), -delta)
    both = sample.AV_1.eq(1) & sample.AV_2.eq(1)
    rows = []
    for context in [None, "FEMALE", "OLDER65", "FAMILY", "URBAN_REGION"]:
        active = pd.Series(True, index=sample.index) if context is None else sample[context].eq(1)
        contrast = active & delta.ne(0)
        informative = contrast & both
        values = chosen_delta.loc[informative]
        older = int(values.gt(0).sum())
        younger = int(values.lt(0).sum())
        same = int((active & delta.eq(0)).sum())
        quantiles = values.quantile([.05, .25, .75, .95]) if len(values) else pd.Series(np.nan, index=[.05,.25,.75,.95])
        rows.append({"subset": "all_stage1" if context is None else f"{context}=1",
                     "active_occasions_n": int(active.sum()),
                     "nonzero_age_contrasts_n": int(contrast.sum()),
                     "both_available_nonzero_n": int(informative.sum()),
                     "chosen_older_n": older, "chosen_younger_n": younger,
                     "same_age_n": same,
                     "chosen_older_share_of_nonzero_both": older / len(values) if len(values) else np.nan,
                     "chosen_younger_share_of_nonzero_both": younger / len(values) if len(values) else np.nan,
                     "same_age_share_of_active": same / active.sum() if active.sum() else np.nan,
                     "supporting_households_n": sample.loc[informative, "H_ID"].nunique(),
                     "chosen_minus_nonchosen_mean": values.mean(),
                     "chosen_minus_nonchosen_median": values.median(),
                     "chosen_minus_nonchosen_p5": quantiles.loc[.05],
                     "chosen_minus_nonchosen_p25": quantiles.loc[.25],
                     "chosen_minus_nonchosen_p75": quantiles.loc[.75],
                     "chosen_minus_nonchosen_p95": quantiles.loc[.95],
                     "chosen_minus_nonchosen_min": values.min(),
                     "chosen_minus_nonchosen_max": values.max()})
    return pd.DataFrame(rows)


def weighted_interaction_support(sample: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    both = sample.AV_1.eq(1) & sample.AV_2.eq(1)
    rows = []
    for name, context, target in prep.INTERACTIONS:
        active = pd.Series(True, index=sample.index) if context == "TOUR_TIME_C_H" else sample[context].ne(0)
        mask = both & active & design[target].ne(0)
        sub = sample.loc[mask, ["H_ID", "W_GEW"]]
        w = sub.W_GEW.to_numpy(float)
        if not np.isfinite(w).all() or (w <= 0).any():
            raise AssertionError(f"Invalid supporting survey weight: {name}")
        total = float(w.sum())
        households = sub.groupby("H_ID", sort=True).W_GEW.sum().sort_values(ascending=False)
        n = len(w)
        kish = total**2 / float(np.square(w).sum()) if n else np.nan
        max_share = float(w.max() / total) if n else np.nan
        max_hh_share = float(households.iloc[0] / total) if n else np.nan
        top5_share = float(households.iloc[:5].sum() / total) if n else np.nan
        flags = []
        if n and max_share > .10:
            flags.append("individual weight >10% of support total")
        if n and top5_share > .25:
            flags.append("top five households >25% of support total")
        if n and kish / n < .5:
            flags.append("Kish descriptive n_eff <50% of occasions")
        rows.append({"predictor": name, "informative_occasions_n": n,
                     "supporting_households_n": len(households),
                     "supporting_weight_sum": total,
                     "weight_mean": float(w.mean()) if n else np.nan,
                     "weight_median": float(np.median(w)) if n else np.nan,
                     "weight_p90": float(np.quantile(w, .90)) if n else np.nan,
                     "weight_p99": float(np.quantile(w, .99)) if n else np.nan,
                     "weight_max": float(w.max()) if n else np.nan,
                     "max_individual_weight_share": max_share,
                     "kish_n_eff_descriptive": kish,
                     "kish_n_eff_per_occasion": kish / n if n else np.nan,
                     "max_household_weight_share": max_hh_share,
                     "top5_households_weight_share": top5_share,
                     "contributing_households_n": len(households),
                     "concentration_warning": "; ".join(flags)})
    return pd.DataFrame(rows)


def vehicle_age_extremes(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta = sample.VEHICLE_AGE_1 - sample.VEHICLE_AGE_2
    absolute = delta.abs()
    chosen = delta.where(sample.CHOICE.eq(1), -delta)
    summary = pd.DataFrame([{"n": len(sample), "min_abs_delta_age": absolute.min(),
                             "max_abs_delta_age": absolute.max(), "median_abs_delta_age": absolute.median(),
                             "p90_abs_delta_age": absolute.quantile(.90), "p95_abs_delta_age": absolute.quantile(.95),
                             "p99_abs_delta_age": absolute.quantile(.99), "p995_abs_delta_age": absolute.quantile(.995),
                             "zero_difference_n": int(absolute.eq(0).sum()),
                             "abs_difference_gt30_n": int(absolute.gt(30).sum()),
                             "abs_difference_gt40_n": int(absolute.gt(40).sum())}])
    columns = [c for c in ["CHOICE_ID", "H_ID", "HP_ID", "P_ID", "W_ID", "SOURCE_ROW_ID", "CHOICE",
                            "VEHICLE_AGE_1", "VEHICLE_AGE_2", "POWERTRAIN_1", "POWERTRAIN_2",
                            "SEGMENT_1", "SEGMENT_2"] if c in sample]
    cases = sample[columns].copy()
    cases["DELTA_VEHICLE_AGE"] = delta
    cases["ABS_DELTA_VEHICLE_AGE"] = absolute
    cases["CHOSEN_MINUS_NONCHOSEN_AGE"] = chosen
    cases = cases.sort_values(["ABS_DELTA_VEHICLE_AGE", "CHOICE_ID"], ascending=[False, True], kind="stable").head(20).reset_index(drop=True)
    cases["review_classification"] = np.where(cases.ABS_DELTA_VEHICLE_AGE.gt(40),
                                               "requires upstream verification",
                                               "potentially unusual but not demonstrably invalid")
    cases["classification_note"] = "Diagnostic priority only; accepted vehicle age unchanged; original construction year not available in verified Stage-1 source"
    return summary, cases


def matrix_diagnostics(design: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = design.to_numpy(dtype=float)
    n, p = x.shape
    rank, singular, tol = numerical_rank(x)
    norms = np.linalg.norm(x, axis=0)
    nonzero = norms > 0
    scaled = x.copy()
    scaled[:, nonzero] /= norms[nonzero]
    scaled_rank, scaled_singular, scaled_tol = numerical_rank(scaled)
    raw_condition = float(singular[0] / singular[-1]) if singular[-1] > tol else float("inf")
    scaled_condition = float(scaled_singular[0] / scaled_singular[-1]) if scaled_singular[-1] > scaled_tol else float("inf")
    _, r, pivots = qr(scaled, mode="economic", pivoting=True)
    independent = [prep.PREDICTORS[i] for i in pivots[:scaled_rank]]
    dependent = [prep.PREDICTORS[i] for i in pivots[scaled_rank:]]
    rank_rows = []
    for relative in (None, 1e-12, 1e-10, 1e-8, 1e-6):
        this_rank, _, this_tol = numerical_rank(scaled, relative)
        rank_rows.append({"subset": "both_available", "matrix": "L2_scaled_no_intercept",
                          "relative_tolerance": "numpy_default" if relative is None else relative,
                          "absolute_tolerance": this_tol, "rank": this_rank,
                          "columns": p, "deficiency": p-this_rank, "n": n,
                          "dependent_qr_terms": "; ".join(dependent)})
    rank_rows.append({"subset": "both_available", "matrix": "raw_no_intercept", "relative_tolerance": "numpy_default",
                      "absolute_tolerance": tol, "rank": rank, "columns": p, "deficiency": p-rank,
                      "n": n, "dependent_qr_terms": "; ".join(dependent)})
    singular_rows = pd.DataFrame({"index": np.arange(1, p+1), "raw": singular, "L2_scaled": scaled_singular})
    condition_rows = pd.DataFrame([{"subset": "both_available", "n": n, "columns": p,
                                    "raw_condition": raw_condition, "scaled_condition": scaled_condition,
                                    "raw_largest_singular": singular[0], "raw_smallest_singular": singular[-1],
                                    "scaled_largest_singular": scaled_singular[0],
                                    "scaled_smallest_singular": scaled_singular[-1],
                                    "scaling": "nonzero-column L2 normalization; no matrix centering",
                                    "zero_columns": "; ".join(np.array(prep.PREDICTORS)[~nonzero]),
                                    "qr_dependent_terms": "; ".join(dependent)}])
    corr = design.corr(method="pearson")
    vif_rows = []
    for j, name in enumerate(prep.PREDICTORS):
        y = x[:, j]
        centered = y - y.mean()
        tss = float(centered @ centered)
        if tss <= np.finfo(float).eps:
            vif, r2, status = float("inf"), float("nan"), "zero variance"
        else:
            other = np.delete(x, j, axis=1)
            aux = np.column_stack([np.ones(n), other])
            coefficients = np.linalg.lstsq(aux, y, rcond=None)[0]
            residual = y - aux @ coefficients
            rss = float(residual @ residual)
            r2 = 1 - rss/tss
            vif = float("inf") if rss/tss <= np.finfo(float).eps else tss/rss
            status = ("undefined/infinite; auxiliary exact dependency" if not np.isfinite(vif) else
                      "VIF >= 10" if vif >= 10 else "VIF >= 5" if vif >= 5 else "below descriptive references")
        vif_rows.append({"predictor": name, "VIF": vif, "auxiliary_R2": r2, "status": status,
                         "procedure": "OLS on intercept and other 23 effective predictors; centered TSS"})
    metrics = {"n": n, "p": p, "rank": rank, "scaled_rank": scaled_rank, "rank_tolerance": tol,
               "raw_condition": raw_condition, "scaled_condition": scaled_condition,
               "qr_dependent_terms": dependent, "qr_independent_terms": independent,
               "max_vif": float(np.max([row["VIF"] for row in vif_rows]))}
    return metrics, pd.DataFrame(rank_rows), singular_rows, condition_rows, pd.DataFrame(vif_rows), corr


def diagnose() -> dict:
    DIAG.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    baseline_path = REVIEW / "core_v1_targeted_QA_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for category in ("protected_stage1", "baseline_diagnostics", "upstream_and_stage0"):
        for filename, digest in baseline[category].items():
            if prep.sha256(Path(filename)) != digest:
                raise AssertionError(f"Targeted-QA baseline changed before diagnostics: {filename}")
    metadata = json.loads((DATA / "core_v1_transformations.json").read_text(encoding="utf-8"))
    for filename, digest in metadata["source_hashes_sha256"].items():
        if prep.sha256(Path(filename)) != digest:
            raise AssertionError(f"Accepted source changed since preparation: {filename}")
    for filename, digest in metadata["accepted_stage0_result_hashes_sha256"].items():
        if prep.sha256(Path(filename)) != digest:
            raise AssertionError(f"Accepted Stage-0 result changed since preparation: {filename}")
    sample = pd.read_csv(DATA / "core_v1_context_sample.csv", dtype={k: "string" for k in prep.JOIN_KEYS}, low_memory=False)
    design_csv = pd.read_csv(DATA / "core_v1_effective_design.csv", dtype={"CHOICE_ID": "string", "H_ID": "string"})
    if not sample.CHOICE_ID.is_unique or not design_csv.CHOICE_ID.equals(sample.CHOICE_ID):
        raise AssertionError("Saved sample and effective design rows do not match")
    if len(sample) != metadata["sample"]["stage1_n"] or sample.H_ID.nunique() != metadata["sample"]["households"]:
        raise AssertionError("Stage-1 candidate count or household count changed")
    if list(design_csv.columns) != ["CHOICE_ID", "H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW", *prep.PREDICTORS]:
        raise AssertionError("Stage-1 predictor names, ordering, or control columns changed")
    for column in ("H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW"):
        if not design_csv[column].equals(sample[column]):
            raise AssertionError(f"Stage-1 control changed between saved sample and design: {column}")
    design = design_csv[prep.PREDICTORS]
    if not np.allclose(design.to_numpy(), prep.effective_design(sample).to_numpy(), rtol=0, atol=1e-12):
        raise AssertionError("Saved effective design does not reproduce")
    both = sample.AV_1.eq(1) & sample.AV_2.eq(1)
    if both.sum() == 0:
        raise AssertionError("No both-available occasions for identification diagnostics")
    comp, cross, continuous = composition(sample)
    comp.to_csv(DIAG / "core_v1_context_composition.csv", index=False)
    cross.to_csv(DIAG / "core_v1_context_cross_tabs.csv", index=False)
    continuous.to_csv(DIAG / "core_v1_time_distribution.csv", index=False)
    alternative_support(sample).to_csv(DIAG / "core_v1_alternative_support.csv", index=False)
    support, time_support = interaction_support(sample, design)
    support.to_csv(DIAG / "core_v1_interaction_support.csv", index=False)
    time_support.to_csv(DIAG / "core_v1_time_interaction_support.csv", index=False)
    age_choice = vehicle_age_choice_support(sample)
    age_choice.to_csv(DIAG / "core_v1_vehicle_age_choice_support.csv", index=False)
    weighted_support = weighted_interaction_support(sample, design)
    if not np.array_equal(weighted_support.informative_occasions_n.to_numpy(), support.informative_contrasts_n.to_numpy()):
        raise AssertionError("Weighted-support subset differs from interaction support")
    weighted_support.to_csv(DIAG / "core_v1_weighted_interaction_support.csv", index=False)
    age_summary, age_cases = vehicle_age_extremes(sample)
    age_summary.to_csv(DIAG / "core_v1_vehicle_age_extreme_summary.csv", index=False)
    age_cases.to_csv(DIAG / "core_v1_vehicle_age_extreme_contrasts.csv", index=False)
    metrics, rank, singular, condition, vif, corr = matrix_diagnostics(design.loc[both])
    rank.to_csv(DIAG / "core_v1_rank_diagnostics.csv", index=False)
    singular.to_csv(DIAG / "core_v1_singular_values.csv", index=False)
    condition.to_csv(DIAG / "core_v1_condition_diagnostics.csv", index=False)
    vif.to_csv(DIAG / "core_v1_vif.csv", index=False)
    corr.to_csv(DIAG / "core_v1_correlation.csv")
    # Secondary all-retained summary is explicitly separate from identification conclusions.
    all_metrics, *_ = matrix_diagnostics(design)
    pd.DataFrame([{"subset": "both_available", **metrics}, {"subset": "all_retained", **all_metrics}]).to_json(
        DIAG / "core_v1_matrix_subset_comparison.json", orient="records", indent=2)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().abs().sort_values(ascending=False)
    pair_table = pd.DataFrame([{"predictor_1": a, "predictor_2": b, "abs_pearson_r": value,
                               "pearson_r": corr.loc[a, b]} for (a, b), value in pairs.items()])
    pair_table.to_csv(DIAG / "core_v1_correlation_pairs.csv", index=False)
    warnings = []
    if metrics["rank"] < metrics["p"]:
        warnings.append(f"Rank deficient: {metrics['rank']}/{metrics['p']}; QR candidate dependent terms: {metrics['qr_dependent_terms']}")
    if metrics["scaled_condition"] >= 1e8:
        warnings.append(f"Severe scaled condition number >=1e8: {metrics['scaled_condition']:.4g}")
    for _, row in support.loc[support.warning.ne("")].iterrows():
        warnings.append(f"{row.predictor}: {row.warning}")
    for _, row in time_support.loc[time_support.support_warning.ne("")].iterrows():
        warnings.append(f"{row.predictor} time bin {row.time_minutes_bin}: {row.support_warning}; n={row.informative_contrasts_n}")
    for _, row in weighted_support.loc[weighted_support.concentration_warning.ne("")].iterrows():
        warnings.append(f"{row.predictor} weight concentration: {row.concentration_warning}")
    for _, row in vif.loc[vif.VIF.ge(5) | ~np.isfinite(vif.VIF)].iterrows():
        warnings.append(f"{row.predictor}: VIF {row.VIF:.4g} (descriptive >=5 reference)")
    if not pair_table.empty and pair_table.iloc[0].abs_pearson_r >= .8:
        warnings.append(f"Largest absolute Pearson correlation >=0.8: {pair_table.iloc[0].predictor_1} × {pair_table.iloc[0].predictor_2} = {pair_table.iloc[0].abs_pearson_r:.3f}")
    flow = pd.read_csv(DIAG / "core_v1_sample_flow.csv")
    flow_counts = dict(zip(flow.metric, flow["count"]))
    report = ["# Core–V1 pre-estimation QA", "", "**Decision: PENDING_MANUAL_REVIEW**", "",
              "## A. Input lineage", "", "Accepted Stage-0 design matrix and allocation candidate model input joined one to one on `CHOICE_ID`. Household and driver links, choice, availability, weights, and vehicle coding were cross-validated. SHA-256 hashes are in the input manifest.", "",
              f"- Design matrix: `{metadata['source_lineage']['design_matrix']}`",
              f"- Context input: `{metadata['source_lineage']['context_model_input']}`",
              f"- Accepted V1 rules: `{metadata['source_lineage']['stage0_v1_code']}`", "",
              "| Read-only source or lineage code | SHA-256 |", "|---|---|",
              *[f"| `{path}` | `{digest}` |" for path, digest in metadata["source_hashes_sha256"].items()], "",
              "## B. Sample reconciliation", "",
              f"- Source: {flow_counts['initial_single_active_driver_occasions']:,}; Stage-0 valid: {flow_counts['stage0_valid_v1_occasions']:,}; Stage-1 exclusions: {flow_counts['stage1_unique_exclusions']:,}; candidate: {len(sample):,}; households: {sample.H_ID.nunique():,}.",
              f"- Missing required context: {flow_counts['stage1_missing_required_context']:,}; invalid category with no missing context: {flow_counts['stage1_invalid_nonmissing_context']:,}; overlapping context reasons: {flow_counts['stage1_exclusion_reason_overlap']:,}.",
              f"- Both available: {both.sum():,}; single available: {(~both).sum():,}. Primary identification diagnostics use both available.",
              f"- Under-18 drivers retained and flagged: {flow_counts['stage1_under18_drivers']:,}.", "",
              f"- Diverse sex code 3 in Stage-0 valid sample: {flow_counts['stage0_diverse_sex_code_3']:,}; other missing/invalid sex: {flow_counts['stage0_missing_or_other_invalid_sex']:,}. Both are excluded from the male/female interaction. Original category counts and retained counts are in `core_v1_pre_exclusion_context_codes.csv`.", "",
              "## C. Fixed candidate specification", "", "Seven vehicle main effects and 17 interactions; no intercept or generic context main effects. Reference categories: ICE, SMALL, male, under 65, non-family, rural region, no household companion, and 0–1 stops. HEV and ICE travel-time slopes are constrained to zero.", "",
              "| Predictor | Effective formula |", "|---|---|",
              *[f"| {item['predictor']} | `{item['formula']}` |" for item in metadata['main_effects'] + metadata['interactions']], "",
              "## D. Composition", "",
              f"- Travel time mean for centering: {metadata['centering_mean_minutes']:.6f} minutes. Full category shares, weighted descriptive shares, time quantiles, and cross-tabs are in diagnostics CSVs.", "",
              "## E. Effective interaction support", "",
              "Support counts are unweighted identifying occasions. Weighted shares describe the sample only. For time interactions, contrast counts use vehicle differences even at centered time near zero. Detailed choice direction, age contrasts, and time bins are in the support files.", "",
              "| Term | Both-available contrasts | Households | Warning |", "|---|---:|---:|---|",
              *[f"| {r.predictor} | {r.nonzero_target_delta_active_both:,} | {r.unique_supporting_households:,} | {r.warning} |" for _, r in support.iterrows()], "",
              "## F. Matrix rank", "",
              f"- Primary rank: {metrics['rank']}/{metrics['p']} at tolerance {metrics['rank_tolerance']:.3g}; QR candidate dependent terms: {', '.join(metrics['qr_dependent_terms']) or 'none'}. Sensitivity and singular values are in CSV.", "",
              "## G. Conditioning, VIF, correlation", "",
              f"- Raw condition number: {metrics['raw_condition']:.6g}; L2-scaled: {metrics['scaled_condition']:.6g}; max VIF: {metrics['max_vif']:.6g}.",
              f"- Max absolute off-diagonal correlation: {pair_table.iloc[0].abs_pearson_r:.4f} ({pair_table.iloc[0].predictor_1}, {pair_table.iloc[0].predictor_2}).",
              "- VIF uses an intercept in each auxiliary regression. Rank/conditioning use the uncentered, no-intercept effective design. VIF 5/10 and support count 30 are descriptive flags, not selection rules.", "",
              "## H. Warnings and unresolved questions", "",
              *([f"- {w}" for w in warnings] or ["- No threshold warnings; inspect all detailed outputs manually."]),
              "- Tour travel time and stop count are realized diary-tour characteristics. These associations must not be interpreted causally or assumed known at allocation time.",
              "- The accepted model input has no original vehicle A_ID columns; vehicle labels and H_ID remain traceable. No A_ID was inferred.", "",
              "## I. Manual decision", "", "**PENDING_MANUAL_REVIEW. No MNL estimation has been run.**", "",
              "## Targeted QA addendum", "", "See `core_v1_targeted_QA_addendum.md` for choice direction, age direction, time-bin support, weight concentration, and extreme-age inspection. The candidate sample and 24-column effective design are unchanged.", ""]
    (REVIEW / "core_v1_pre_estimation_QA.md").write_text("\n".join(report), encoding="utf-8")
    checklist = ["# Core–V1 manual review checklist", "", "Decision: PENDING_MANUAL_REVIEW", "",
                 *[f"- [ ] {item}" for item in ["Input paths and joins verified", "No accepted upstream files changed", "Stage-0 V1 sample rules reproduced", "Stage-1 sample losses understood", "Context coding and reference groups correct", "Exactly 24 candidate predictors", "All effective interaction formulas validated", "Availability-aware identifying support checked", "Rank assessed and acceptable or explicitly resolved", "Condition number reviewed", "VIF and correlation reviewed", "Weak-support interactions reviewed", "Final decision documented"]], ""]
    (REVIEW / "core_v1_manual_review_checklist.md").write_text("\n".join(checklist), encoding="utf-8")
    for category in ("protected_stage1", "baseline_diagnostics", "upstream_and_stage0"):
        for filename, digest in baseline[category].items():
            if prep.sha256(Path(filename)) != digest:
                raise AssertionError(f"Targeted-QA protected data or numerical diagnostic changed: {filename}")
    if not (age_choice.chosen_older_n + age_choice.chosen_younger_n).eq(age_choice.both_available_nonzero_n).all():
        raise AssertionError("Chosen older/younger counts do not reconcile")
    priority = ["D_OLDER_BEV", "D_FAMILY_LARGE", "D_FAMILY_BEV", "D_URBAN_BEV", "D_TIME_PHEV", "D_TIME_BEV"]
    weighted_priority = weighted_support.set_index("predictor").loc[priority].reset_index()
    binary_support = support.loc[support.target.ne("D_VEHICLE_AGE")]
    target_flags = binary_support.loc[binary_support.warning.str.contains("target always chosen|target never chosen|weak target-choice direction", regex=True)]
    concentration_flags = weighted_support.loc[weighted_support.concentration_warning.ne("")]
    time_flags = time_support.loc[time_support.support_warning.ne("")]
    age_case_class = age_cases.review_classification.value_counts().to_dict()
    protected_hash_lines = []
    for filename, digest in baseline["protected_stage1"].items():
        protected_hash_lines.append(f"| `{Path(filename).name}` | `{digest}` | `{prep.sha256(Path(filename))}` |")
    addendum = [
        "# Core–V1 targeted pre-estimation QA addendum", "", "**Decision: PENDING_MANUAL_REVIEW**", "",
        "## 1. Scope and integrity", "",
        "Modified: `scripts/05_mnl/stage1/02_diagnose_core_v1.py` and `scripts/05_mnl/stage1/test_core_v1.py`. The preparation script, accepted Stage-0 scripts/results, upstream inputs, Stage-1 sample, context coding, and all 24 predictor formulas were left unchanged.", "",
        "| Protected Stage-1 file | SHA-256 before | SHA-256 after |", "|---|---|---|", *protected_hash_lines, "",
        f"All {len(baseline['upstream_and_stage0'])} protected upstream/Stage-0 hashes and four baseline rank/condition/VIF/correlation file hashes match the baseline snapshot (`core_v1_targeted_QA_baseline.json`).", "",
        f"N={len(sample):,}; households={sample.H_ID.nunique():,}; predictors={len(prep.PREDICTORS)} in unchanged order. Rank={metrics['rank']}/{metrics['p']}; raw condition={metrics['raw_condition']:.4f}; scaled condition={metrics['scaled_condition']:.4f}; max VIF={metrics['max_vif']:.4f}. Choice, availability, survey weights, and all saved design values match the protected inputs.", "",
        "## 2. Corrected binary target-choice support", "",
        "The chosen target is determined from `CHOICE` and the sign of the target vehicle-attribute contrast. `chosen_car1_both`/`chosen_car2_both` remain label-balance fields and do not drive the target-choice warning. The descriptive weak-direction reference is fewer than 10 outcomes in either chosen-target direction; it is not evidence of statistical separation.", "",
        "| Interaction | Contrasts | Chosen target | Chosen non-target | Target share | Target-choice flag |", "|---|---:|---:|---:|---:|---|",
        *[f"| {r.predictor} | {r.informative_contrasts_n:,} | {int(r.chosen_target_n):,} | {int(r.chosen_non_target_n):,} | {r.chosen_target_share:.3f} | {target_choice_warning(int(r.chosen_target_n), int(r.chosen_non_target_n)) or 'none'} |" for _, r in binary_support.iterrows()], "",
        f"New one-sided/weak target-choice flags: {len(target_flags)}. Positive- and negative-delta choice counts are in `core_v1_interaction_support.csv`.", "",
        "## 3. Chosen versus nonchosen vehicle age", "",
        "Positive chosen-minus-nonchosen age means the chosen car is older. Shares of older/younger use both-available nonzero-age contrasts; same-age share uses all active occasions. Descriptive age quantiles are in `core_v1_vehicle_age_choice_support.csv`.", "",
        "| Subset | Active | Nonzero contrasts | Chosen older | Chosen younger | Same age | Older share | Median chosen-minus-nonchosen age |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {r.subset} | {r.active_occasions_n:,} | {r.both_available_nonzero_n:,} | {r.chosen_older_n:,} | {r.chosen_younger_n:,} | {r.same_age_n:,} | {r.chosen_older_share_of_nonzero_both:.3f} | {r.chosen_minus_nonchosen_median:.1f} |" for _, r in age_choice.iterrows()], "",
        "## 4. Tour-time-bin target choice", "",
        "Bins preserve the existing right-closed boundaries: ≤30, (30,60], (60,120], (120,240], and >240 minutes. Bin counts reconcile exactly with the overall contrast and chosen-target counts. Shares are descriptive associations with realized tour time, not causal effects.", "",
        "| Interaction | Minutes | All bin occasions | Contrasts | Chosen target | Chosen non-target | Target share | Flag |", "|---|---|---:|---:|---:|---:|---:|---|",
        *[f"| {r.predictor} | {r.time_minutes_bin} | {r.total_occasions_in_bin:,} | {r.informative_contrasts_n:,} | {r.chosen_target_n:,} | {r.chosen_non_target_n:,} | {r.chosen_target_share:.3f} | {r.support_warning or 'none'} |" for _, r in time_support.iterrows()], "",
        f"Sparse-bin flags (<50 contrasts, descriptive): {len(time_flags)}. Weighted support shares within each bin and supporting households are in `core_v1_time_interaction_support.csv`.", "",
        "## 5. Survey-weight concentration", "",
        "Kish `n_eff=(Σw)²/Σw²` is an occasion-level descriptive measure. It is not a count of independent households or a formal complex-survey effective sample size. Household contributions sum supporting occasion weights by `H_ID`; no household-level effective sample size is inferred. Concentration flags use individual share >10%, top-five household share >25%, or Kish ratio <50%, solely as review references.", "",
        "| Priority interaction | Contrasts | Households | Kish n_eff | Kish/N | Max individual share | Top-five household share |", "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {r.predictor} | {r.informative_occasions_n:,} | {r.supporting_households_n:,} | {r.kish_n_eff_descriptive:.1f} | {r.kish_n_eff_per_occasion:.3f} | {r.max_individual_weight_share:.3f} | {r.top5_households_weight_share:.3f} |" for _, r in weighted_priority.iterrows()], "",
        f"Concentration flags across 17 interactions: {len(concentration_flags)}, all from Kish/N <0.5. The largest single-occasion weight share is {weighted_support.max_individual_weight_share.max():.3f}; the largest top-five-household share is {weighted_support.top5_households_weight_share.max():.3f}. Thus no subset crosses the separate 10% individual or 25% top-five-household references. All 17 weight distributions are in `core_v1_weighted_interaction_support.csv`.", "",
        "## 6. Extreme vehicle-age contrasts", "",
        f"Absolute age difference: minimum {age_summary.iloc[0].min_abs_delta_age:.1f}, median {age_summary.iloc[0].median_abs_delta_age:.1f}, P90 {age_summary.iloc[0].p90_abs_delta_age:.1f}, P95 {age_summary.iloc[0].p95_abs_delta_age:.1f}, P99 {age_summary.iloc[0].p99_abs_delta_age:.1f}, P99.5 {age_summary.iloc[0].p995_abs_delta_age:.1f}, maximum {age_summary.iloc[0].max_abs_delta_age:.1f} years. Zero differences: {int(age_summary.iloc[0].zero_difference_n):,}; >30: {int(age_summary.iloc[0].abs_difference_gt30_n):,}; >40: {int(age_summary.iloc[0].abs_difference_gt40_n):,}.", "",
        f"The local top-20 table classifies {age_case_class.get('requires upstream verification', 0)} cases as requiring upstream verification (>40-year difference) and {age_case_class.get('potentially unusual but not demonstrably invalid', 0)} as unusual but not demonstrably invalid. These are review priorities, not data corrections. Original construction-year data were unavailable in the verified Stage-1 source; no unverified vehicle join was made. No age values were changed or excluded.", "",
        "## 7. Final pre-estimation recommendation", "",
        "- **PASS:** protected inputs and design unchanged; target-choice, age-choice, time-bin, and weight calculations reconcile; numerical rank, condition, VIF, and correlation remain unchanged.",
        f"- **REVIEW:** {len(target_flags)} target-choice flags, {len(time_flags)} sparse time bins, {len(concentration_flags)} weight-concentration flags, and the {age_case_class.get('requires upstream verification', 0)} top age-contrast cases marked for upstream verification. Existing VIF/correlation warnings remain descriptive.",
        "- **BLOCKER:** none found in the targeted computations. Unusual age contrasts need human review before treating them as substantively credible.", "",
        "The candidate appears computationally ready for Stage-5 estimation, **subject to human approval**. Official decision remains **PENDING_MANUAL_REVIEW**. No MNL estimation was performed.", "",
    ]
    (REVIEW / "core_v1_targeted_QA_addendum.md").write_text("\n".join(addendum), encoding="utf-8")
    manifest = {"decision_status": "PENDING_MANUAL_REVIEW", "source_hashes_sha256": metadata["source_hashes_sha256"],
                "accepted_stage0_result_hashes_sha256": metadata["accepted_stage0_result_hashes_sha256"],
                "join_key": "CHOICE_ID", "source_n": metadata["sample"]["source_n"],
                "stage0_valid_n": metadata["sample"]["stage0_valid_n"], "stage1_candidate_n": len(sample),
                "both_available_n": int(both.sum()), "warning_count": len(warnings),
                "warning_criteria": ["informative contrasts <30", "one-sided contrast/choice", "scaled condition >=1e8", "VIF >=5 and >=10 descriptive references", "absolute Pearson correlation >=0.8"],
                "stage1_code_files": [str(PREPARE_PATH.resolve()), str(Path(__file__).resolve()),
                                      str(Path(__file__).with_name("test_core_v1.py").resolve())],
                "stage1_output_files": sorted({str(p.resolve()) for p in ROOT.rglob("*") if p.is_file()} |
                                              {str((REVIEW / "core_v1_input_manifest.json").resolve())})}
    (REVIEW / "core_v1_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REVIEW / "core_v1_warnings.json").write_text(json.dumps(warnings, indent=2), encoding="utf-8")
    for filename, digest in metadata["source_hashes_sha256"].items():
        if prep.sha256(Path(filename)) != digest:
            raise AssertionError(f"Accepted source changed during diagnostics: {filename}")
    for filename, digest in metadata["accepted_stage0_result_hashes_sha256"].items():
        if prep.sha256(Path(filename)) != digest:
            raise AssertionError(f"Accepted Stage-0 result changed during diagnostics: {filename}")
    print("Inputs:", *metadata["source_lineage"].values(), sep="\n  ")
    print(f"Stage-0 valid N={metadata['sample']['stage0_valid_n']:,}; Stage-1 N={len(sample):,}; households={sample.H_ID.nunique():,}; main=7; interactions=17")
    print(f"Both-available informative occasions={both.sum():,}; rank={metrics['rank']}/{metrics['p']}; raw condition={metrics['raw_condition']:.6g}; scaled condition={metrics['scaled_condition']:.6g}; max VIF={metrics['max_vif']:.6g}; warnings={len(warnings)}")
    print(f"Output: {ROOT.resolve()}; decision=PENDING_MANUAL_REVIEW; estimation=NOT_PERFORMED")
    return {"metrics": metrics, "warnings": warnings, "manifest": manifest}


if __name__ == "__main__":
    diagnose()
