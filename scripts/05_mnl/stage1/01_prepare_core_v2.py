"""Prepare the fixed 21-predictor Stage-1 Core-V2 difference design.

Shared model inputs are validated and fingerprinted for this run. The Stage-0
V2 code is inspected as source text; its estimation module is never imported.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from src.thesis_pipeline.paths import (  # noqa: E402
    REPO_ROOT, OUTPUTS_DIR, SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH,
    ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, PHASE4_DIR,
)

FINAL_ROOT = OUTPUTS_DIR / "mnl" / "stage1" / "core_v2"
STAGING_ROOT = OUTPUTS_DIR / "mnl" / "stage1" / ".core_v2_replacement_staging"
ROOT = STAGING_ROOT if os.environ.get("CORE_V2_STAGING") == "1" else FINAL_ROOT
DATA = ROOT / "data"
STAGE0_CODE = REPO_ROOT / "scripts" / "05_mnl" / "01_estimate_v2_only_mnl.py"
PREVIOUS_CORE_V2_SAMPLE = FINAL_ROOT / "data" / "core_v2_context_sample.csv"
TOURS_PATH = PHASE4_DIR / "home_based_tour_features.csv"
KEYS = ("CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID")
CONTROLS = ("CHOICE", "AV_1", "AV_2", "W_GEW")
POWERTRAINS = ("ICE", "HEV", "PHEV", "BEV")
PT_FULL = (*POWERTRAINS, "OTHER", "MISSING")
STATUSES = ("LOW", "MEDIUM", "HIGH")
STATUS_FULL = (*STATUSES, "MISSING")
PURPOSES = ("WORK", "BUSINESS", "EDUCATION", "SHOPPING", "ERRAND", "LEISURE", "ESCORT", "MISSING")
PURPOSE_COUNTS = ("0", "1", "2", "3PLUS")
ECON = ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")

# One authoritative formula list for preparation, diagnostics, and estimation.
MAIN = (
    ("D_PT_HEV", "PT1_HEV", "PT2_HEV"),
    ("D_PT_PHEV", "PT1_PHEV", "PT2_PHEV"),
    ("D_PT_BEV", "PT1_BEV", "PT2_BEV"),
    ("D_STATUS_LOW", "STATUS1_LOW", "STATUS2_LOW"),
    ("D_STATUS_HIGH", "STATUS1_HIGH", "STATUS2_HIGH"),
)
INTERACTIONS = (
    ("D_DIST_BEV", "TOUR_DISTANCE_C_10KM", "D_PT_BEV"),
    ("D_DIST_PHEV", "TOUR_DISTANCE_C_10KM", "D_PT_PHEV"),
    ("D_MULTI_PURPOSE_BEV", "MULTI_PURPOSE", "D_PT_BEV"),
    ("D_MAIN_WORK_BEV", "MAIN_PURPOSE_WORK", "D_PT_BEV"),
    ("D_MAIN_WORK_STATUS_HIGH", "MAIN_PURPOSE_WORK", "D_STATUS_HIGH"),
    ("D_MAIN_ESCORT_BEV", "MAIN_PURPOSE_ESCORT", "D_PT_BEV"),
    ("D_MAIN_ESCORT_STATUS_LOW", "MAIN_PURPOSE_ESCORT", "D_STATUS_LOW"),
    ("D_MAIN_SHOPPING_BEV", "MAIN_PURPOSE_SHOPPING", "D_PT_BEV"),
    ("D_MAIN_SHOPPING_STATUS_LOW", "MAIN_PURPOSE_SHOPPING", "D_STATUS_LOW"),
    ("D_MALE_BEV", "MALE", "D_PT_BEV"),
    ("D_MALE_STATUS_HIGH", "MALE", "D_STATUS_HIGH"),
    ("D_EMPLOYED_STATUS_LOW", "EMPLOYED", "D_STATUS_LOW"),
    ("D_AGE_UNDER30_STATUS_LOW", "AGE_UNDER30", "D_STATUS_LOW"),
    ("D_VERY_HIGH_ECON_BEV", "VERY_HIGH_ECON", "D_PT_BEV"),
    ("D_VERY_HIGH_ECON_STATUS_HIGH", "VERY_HIGH_ECON", "D_STATUS_HIGH"),
    ("D_URBAN_STATUS_HIGH", "URBAN_REGION", "D_STATUS_HIGH"),
)
PREDICTORS = tuple(row[0] for row in (*MAIN, *INTERACTIONS))
assert len(MAIN) == 5 and len(INTERACTIONS) == 16 and len(set(PREDICTORS)) == 21


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def current_source_hashes() -> dict[str, str]:
    """Fingerprint only the shared inputs and code used by this Core-V2 run."""
    sources = (SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, ALLOCATION_CANDIDATE_MODEL_INPUT_PATH,
               TOURS_PATH, STAGE0_CODE, Path(__file__))
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Required Core-V2 source missing: {path}")
    return {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in sources}


def verify_current_sources(expected: dict[str, str]) -> None:
    if current_source_hashes() != expected:
        raise AssertionError("Core-V2 shared input or preparation code changed during this run.")


def verify_stage0_v2_specification() -> None:
    tree = ast.parse(STAGE0_CODE.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "DELTA_X_SPECS"
                              for target in node.targets))
    old = ast.literal_eval(assignment.value)
    if tuple((name, left, right) for name, _, left, right in old) != MAIN:
        raise AssertionError("Core-V2 contrasts differ from accepted Stage-0 V2 coding.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def exact_codes(series: pd.Series, allowed: set[int]) -> pd.Series:
    values = numeric(series)
    return values.isin(allowed) & np.isfinite(values) & values.eq(np.floor(values))


def binary_block(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].apply(numeric)
    if values.isna().any().any() or not values.isin([0, 1]).all().all():
        raise AssertionError(f"{label} contains a nonbinary indicator.")
    if not values.sum(axis=1).eq(1).all():
        raise AssertionError(f"{label} is not one-hot.")


def validate_matrix_blocks(frame: pd.DataFrame) -> None:
    if frame.CHOICE_ID.isna().any() or frame.CHOICE_ID.eq("").any() or not frame.CHOICE_ID.is_unique:
        raise AssertionError("Matrix CHOICE_ID is missing or duplicated.")
    for car in (1, 2):
        binary_block(frame, [f"PT{car}_{category}" for category in PT_FULL], f"PT{car}")
        binary_block(frame, [f"STATUS{car}_{category}" for category in STATUS_FULL], f"STATUS{car}")
    binary_block(frame, [f"MAIN_PURPOSE_{category}" for category in PURPOSES], "main purpose")
    binary_block(frame, [f"TOUR_PURPOSE_COUNT_{category}" for category in PURPOSE_COUNTS], "tour purpose count")
    binary_block(frame, [f"ECON_{category}" for category in ECON], "economic status")
    binary_block(frame, ["SEX_MALE", "SEX_FEMALE", "SEX_DIVERSE", "SEX_MISSING"], "sex")
    binary_block(frame, ["AGE_UNDER18", "AGE_UNDER30", "AGE_30_49", "AGE_50PLUS", "AGE_MISSING"], "age")


def join_sources(matrix: pd.DataFrame, candidate: pd.DataFrame, tours: pd.DataFrame) -> pd.DataFrame:
    needed = [*KEYS, *CONTROLS, "TOUR_DISTANCE_KM", "TOUR_N_NONHOME_STOPS",
              "TOUR_N_DISTINCT_PURPOSES", "TOUR_MAIN_PURPOSE",
              "HP_SEX", "erwerb", "alter_gr5", "oek_status", "RegioStaR4",
              "POWERTRAIN_1", "POWERTRAIN_2", "STATUS_1", "STATUS_2"]
    if set(needed) - set(candidate) or set((*KEYS, *CONTROLS)) - set(matrix):
        raise AssertionError("A required Core-V2 source field is unavailable.")
    if candidate.CHOICE_ID.isna().any() or not candidate.CHOICE_ID.is_unique:
        raise AssertionError("Allocation CHOICE_ID is missing or duplicated.")
    joined = matrix.merge(candidate[needed], on="CHOICE_ID", how="left",
                          validate="one_to_one", sort=False, indicator=True, suffixes=("", "_UP"))
    if len(joined) != len(matrix) or joined._merge.ne("both").any() or not joined.CHOICE_ID.equals(matrix.CHOICE_ID):
        raise AssertionError("Matrix-to-allocation join changed occasion order or coverage.")
    for field in (*KEYS[1:], *CONTROLS):
        a, b = joined[field], joined[f"{field}_UP"]
        equal = numeric(a).eq(numeric(b)) if field in CONTROLS else a.astype("string").eq(b.astype("string"))
        if not equal.all():
            raise AssertionError(f"Protected joined control differs: {field}")
    for field in ("TOUR_DISTANCE_KM", "TOUR_MAIN_PURPOSE", "erwerb"):
        a, b = numeric(joined[field]), numeric(joined[f"{field}_UP"])
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise AssertionError(f"Matrix and candidate disagree on {field}.")
    joined = joined.drop(columns=["_merge", *[f"{field}_UP" for field in (*KEYS[1:], *CONTROLS,
                                                                           "TOUR_DISTANCE_KM", "TOUR_MAIN_PURPOSE", "erwerb")]])
    tour_fields = ["TOUR_START_SOURCE_ROW_ID", "H_ID", "TOUR_CLOSED", "TOUR_DISTANCE_KM",
                   "TOUR_DISTANCE_COMPLETE_FLAG", "TOUR_N_TRIPS", "TOUR_N_NONHOME_STOPS",
                   "TOUR_N_DISTINCT_PURPOSES", "TOUR_MULTI_PURPOSE_FLAG",
                   "TOUR_PURPOSE_UNKNOWN_FLAG", "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"]
    if set(tour_fields) - set(tours):
        raise AssertionError("Required accepted tour completeness or purpose fields are unavailable.")
    if tours.TOUR_START_SOURCE_ROW_ID.duplicated().any():
        raise AssertionError("Tour start source ID is not unique.")
    tour_source = tours[tour_fields].rename(columns={
        field: f"SOURCE_{field}" for field in tour_fields if field != "TOUR_START_SOURCE_ROW_ID"})
    joined = joined.merge(tour_source,
                          left_on="SOURCE_ROW_ID", right_on="TOUR_START_SOURCE_ROW_ID",
                          how="left", validate="one_to_one", sort=False)
    if joined.SOURCE_TOUR_CLOSED.isna().any() or not joined.CHOICE_ID.equals(matrix.CHOICE_ID):
        raise AssertionError("Tour-closure join is incomplete or reordered.")
    for field in ("H_ID", "TOUR_DISTANCE_KM", "TOUR_N_NONHOME_STOPS", "TOUR_N_DISTINCT_PURPOSES"):
        left, right = joined[field], joined[f"SOURCE_{field}"]
        equal = left.astype("string").eq(right.astype("string")) if field == "H_ID" else numeric(left).eq(numeric(right))
        if not equal.all():
            raise AssertionError(f"Tour and candidate disagree on {field}.")
    joined["TOUR_CLOSED"] = joined.SOURCE_TOUR_CLOSED
    joined["TOUR_DISTANCE_COMPLETE_FLAG"] = joined.SOURCE_TOUR_DISTANCE_COMPLETE_FLAG
    joined["TOUR_PURPOSE_UNKNOWN_FLAG"] = joined.SOURCE_TOUR_PURPOSE_UNKNOWN_FLAG
    joined["TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"] = joined.SOURCE_TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG
    joined["TOUR_MULTI_PURPOSE_FLAG"] = joined.SOURCE_TOUR_MULTI_PURPOSE_FLAG
    joined["TOUR_N_TRIPS"] = joined.SOURCE_TOUR_N_TRIPS
    return joined.drop(columns=["TOUR_START_SOURCE_ROW_ID"])


def validate_coding(frame: pd.DataFrame) -> None:
    """Cross-check all selected encodings against authoritative raw codes."""
    for car in (1, 2):
        power = numeric(frame[f"POWERTRAIN_{car}"]).map({1: "ICE", 2: "ICE", 6: "ICE",
                                                          3: "HEV", 4: "PHEV", 5: "BEV", 7: "OTHER"})
        status = numeric(frame[f"STATUS_{car}"]).map({1: "LOW", 2: "MEDIUM", 3: "HIGH"})
        for category in PT_FULL:
            expected = power.eq(category) if category != "MISSING" else power.isna()
            if not numeric(frame[f"PT{car}_{category}"]).eq(expected.astype(int)).all():
                raise AssertionError(f"Powertrain coding differs from source for car {car}.")
        for category in STATUS_FULL:
            expected = status.eq(category) if category != "MISSING" else status.isna()
            if not numeric(frame[f"STATUS{car}_{category}"]).eq(expected.astype(int)).all():
                raise AssertionError(f"Status coding differs from source for car {car}.")
    raw_main = numeric(frame.TOUR_MAIN_PURPOSE)
    for code, category in enumerate(PURPOSES[:-1], 1):
        if not numeric(frame[f"MAIN_PURPOSE_{category}"]).eq(raw_main.eq(code).astype(int)).all():
            raise AssertionError("Main-purpose dummy disagrees with official raw code.")
    if not numeric(frame.MAIN_PURPOSE_MISSING).eq(raw_main.isna().astype(int)).all():
        raise AssertionError("Main-purpose missing indicator disagrees with raw code.")
    purpose_count = numeric(frame.TOUR_N_DISTINCT_PURPOSES)
    for label, test in (("0", purpose_count.eq(0)), ("1", purpose_count.eq(1)),
                        ("2", purpose_count.eq(2)), ("3PLUS", purpose_count.ge(3))):
        if not numeric(frame[f"TOUR_PURPOSE_COUNT_{label}"]).eq(test.astype(int)).all():
            raise AssertionError("Purpose-count encoding differs from raw distinct purposes.")
    indicators = [f"TOUR_HAS_{purpose}" for purpose in PURPOSES[:-1]]
    if not frame[indicators].apply(numeric).sum(axis=1).eq(purpose_count).all():
        raise AssertionError("Distinct-purpose count differs from seven broad purpose flags.")
    if not numeric(frame.TOUR_MULTI_PURPOSE_FLAG).eq(purpose_count.ge(2).astype(int)).all():
        raise AssertionError("Accepted multi-purpose flag differs from distinct-purpose count.")
    for field in ("TOUR_DISTANCE_COMPLETE_FLAG", "TOUR_PURPOSE_UNKNOWN_FLAG",
                  "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"):
        if not numeric(frame[field]).isin([0, 1]).all():
            raise AssertionError(f"Accepted tour flag has invalid coding: {field}")
    sex = numeric(frame.HP_SEX)
    for code, name in ((1, "MALE"), (2, "FEMALE"), (3, "DIVERSE")):
        if not numeric(frame[f"SEX_{name}"]).eq(sex.eq(code).astype(int)).all():
            raise AssertionError("Sex coding differs from documented HP_SEX.")
    age = numeric(frame.alter_gr5)
    age_expected = {"UNDER18": age.eq(3), "UNDER30": age.isin([4, 5]),
                    "30_49": age.isin([6, 7, 8, 9]), "50PLUS": age.isin(range(10, 16))}
    for category, expected in age_expected.items():
        if not numeric(frame[f"AGE_{category}"]).eq(expected.astype(int)).all():
            raise AssertionError("Age encoding differs from alter_gr5.")
    econ = numeric(frame.oek_status)
    for code, category in enumerate(ECON, 1):
        if not numeric(frame[f"ECON_{category}"]).eq(econ.eq(code).astype(int)).all():
            raise AssertionError("Economic-status encoding differs from oek_status.")


def context_and_reasons(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = frame.copy()
    pt_bad = out[["PT1_OTHER", "PT1_MISSING", "PT2_OTHER", "PT2_MISSING"]].apply(numeric).eq(1).any(axis=1)
    status_bad = out[["STATUS1_MISSING", "STATUS2_MISSING"]].apply(numeric).eq(1).any(axis=1)
    choice = numeric(out.CHOICE)
    av1, av2 = numeric(out.AV_1), numeric(out.AV_2)
    weights = numeric(out.W_GEW)
    distance = numeric(out.TOUR_DISTANCE_KM)
    purpose_count = numeric(out.TOUR_N_DISTINCT_PURPOSES)
    purpose_unknown = numeric(out.TOUR_PURPOSE_UNKNOWN_FLAG)
    purpose_available = numeric(out.TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG)
    count_block = out[[f"TOUR_PURPOSE_COUNT_{category}" for category in PURPOSE_COUNTS]].apply(numeric)
    count_block_valid = count_block.isin([0, 1]).all(axis=1) & count_block.sum(axis=1).eq(1)
    count_matches = (count_block["TOUR_PURPOSE_COUNT_0"].eq(purpose_count.eq(0).astype(int)) &
                     count_block["TOUR_PURPOSE_COUNT_1"].eq(purpose_count.eq(1).astype(int)) &
                     count_block["TOUR_PURPOSE_COUNT_2"].eq(purpose_count.eq(2).astype(int)) &
                     count_block["TOUR_PURPOSE_COUNT_3PLUS"].eq(purpose_count.ge(3).astype(int)))
    main = numeric(out.TOUR_MAIN_PURPOSE)
    sex = numeric(out.HP_SEX)
    employment = numeric(out.erwerb)
    age = numeric(out.alter_gr5)
    econ = numeric(out.oek_status)
    region = numeric(out.RegioStaR4)
    reasons = {
        "invalid_v2_vehicle": pt_bad | status_bad,
        "invalid_choice_or_availability": ~choice.isin([1, 2]) | av1.ne(1) | av2.ne(1),
        "invalid_weight_or_household": ~np.isfinite(weights) | weights.le(0) |
            out.H_ID.astype("string").str.strip().eq(""),
        "invalid_tour_distance": distance.isna() | ~np.isfinite(distance) | distance.lt(0),
        "incomplete_tour_distance": numeric(out.TOUR_DISTANCE_COMPLETE_FLAG).ne(1),
        "uncertain_or_invalid_purpose_count": ~exact_codes(out.TOUR_N_DISTINCT_PURPOSES, set(range(8))) |
            ~count_block_valid | ~count_matches | ~purpose_unknown.isin([0, 1]) |
            purpose_available.ne(1) | purpose_count.eq(0) |
            (purpose_count.le(1) & purpose_unknown.eq(1)),
        "missing_or_invalid_main_purpose": ~exact_codes(out.TOUR_MAIN_PURPOSE, set(range(1, 8))) |
            numeric(out.TOUR_CLOSED).ne(1),
        "invalid_sex": ~sex.isin([1, 2]),
        "invalid_employment": ~employment.isin([0, 1]),
        "invalid_age": ~age.isin(range(4, 16)),
        "invalid_economic_status": ~exact_codes(out.oek_status, set(range(1, 6))),
        "invalid_urban_region": ~region.isin([11, 12, 21, 22]),
    }
    for name, flag in reasons.items():
        out[name] = flag.fillna(True).astype(bool)
    multi_purpose = count_block["TOUR_PURPOSE_COUNT_2"] + count_block["TOUR_PURPOSE_COUNT_3PLUS"]
    out["MULTI_PURPOSE"] = multi_purpose.where(~out.uncertain_or_invalid_purpose_count).astype("Int64")
    out["MALE"] = sex.eq(1).astype(int)
    out["EMPLOYED"] = employment.eq(1).astype(int)
    out["AGE_UNDER30"] = age.isin([4, 5]).astype(int)
    out["VERY_HIGH_ECON"] = econ.eq(5).astype(int).where(exact_codes(out.oek_status, set(range(1, 6)))).astype("Int64")
    out["URBAN_REGION"] = region.isin([11, 12]).astype(int)
    return out, tuple(reasons)


def effective_design(sample: pd.DataFrame, *, swap: bool = False) -> pd.DataFrame:
    design = pd.DataFrame(index=sample.index)
    for name, left, right in MAIN:
        a, b = (right, left) if swap else (left, right)
        design[name] = numeric(sample[a]).to_numpy(float) - numeric(sample[b]).to_numpy(float)
    for name, context, target in INTERACTIONS:
        design[name] = numeric(sample[context]).to_numpy(float) * design[target].to_numpy(float)
    if list(design) != list(PREDICTORS) or not np.isfinite(design.to_numpy(float)).all():
        raise AssertionError("Core-V2 effective design is incomplete or nonfinite.")
    return design


def sample_flow(context: pd.DataFrame, reasons: tuple[str, ...], sample: pd.DataFrame) -> pd.DataFrame:
    stage0_bad = context.invalid_v2_vehicle
    all_reasons = context[list(reasons)]
    overlap = all_reasons.sum(axis=1).ge(2)
    excluded = pd.Series(False, index=context.index)
    rows = []
    def add(metric: str, count: int, note: str, *, stage0_count: int | None = None,
            overlap_count: int | None = None, unique: int | None = None,
            retained: int | None = None) -> None:
        rows.append({"metric": metric, "count": count, "denominator": len(context),
                     "share": count/len(context), "stage0_valid_count": stage0_count,
                     "overlap_with_other_reasons": overlap_count,
                     "additional_unique_exclusions": unique,
                     "retained_after_sequential": int((~excluded).sum()) if retained is None else retained,
                     "definition": note})
    add("initial_single_driver", len(context), "Accepted 208-column matrix")
    add("stage0_v2_valid", int((~stage0_bad).sum()), "Valid powertrain and status on both cars",
        retained=int((~stage0_bad).sum()))
    for name in reasons:
        flag = context[name]
        unique = int((flag & ~excluded).sum())
        excluded |= flag
        add(name, int(flag.sum()), "Non-exclusive reason; sequential unique loss also recorded",
            stage0_count=int((flag & ~stage0_bad).sum()),
            overlap_count=int((flag & overlap).sum()), unique=unique)
    add("overlap_two_or_more_reasons", int(overlap.sum()), "Initial candidates")
    add("unique_exclusions", len(context)-len(sample), "Union of all reasons")
    add("final_core_v2", len(sample), "Complete validated candidate sample")
    add("final_households", int(sample.H_ID.nunique()), "Distinct H_ID")
    add("final_both_available", int((numeric(sample.AV_1).eq(1) & numeric(sample.AV_2).eq(1)).sum()), "Primary design")
    return pd.DataFrame(rows)


def prepare() -> dict:
    if ROOT.exists():
        raise FileExistsError(f"Core-V2 output collision: {ROOT}")
    verify_stage0_v2_specification()
    before = current_source_hashes()
    matrix = pd.read_csv(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, dtype=str, keep_default_na=False)
    candidate = pd.read_csv(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, dtype=str, keep_default_na=False)
    tours = pd.read_csv(TOURS_PATH, usecols=["TOUR_START_SOURCE_ROW_ID", "H_ID", "TOUR_CLOSED",
                                             "TOUR_DISTANCE_KM", "TOUR_DISTANCE_COMPLETE_FLAG",
                                             "TOUR_N_TRIPS", "TOUR_N_NONHOME_STOPS",
                                             "TOUR_N_DISTINCT_PURPOSES", "TOUR_MULTI_PURPOSE_FLAG",
                                             "TOUR_PURPOSE_UNKNOWN_FLAG",
                                             "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"],
                        dtype=str, keep_default_na=False)
    if len(matrix) != 20931 or len(matrix.columns) != 208:
        raise AssertionError("Shared model matrix does not have the expected shape.")
    validate_matrix_blocks(matrix)
    joined = join_sources(matrix, candidate, tours)
    validate_coding(joined)
    context, reasons = context_and_reasons(joined)
    keep = ~context[list(reasons)].any(axis=1)
    sample = context.loc[keep].copy().reset_index(drop=True)
    if len(sample) != 17093:
        raise AssertionError("Core-V2 sample does not have the accepted 17,093 observations.")
    mean_distance = float(numeric(sample.TOUR_DISTANCE_KM).mean())
    sample["TOUR_DISTANCE_C_10KM"] = (numeric(sample.TOUR_DISTANCE_KM)-mean_distance)/10
    if not np.isclose(sample.TOUR_DISTANCE_C_10KM.mean(), 0, atol=1e-12):
        raise AssertionError("Distance was not centered on the final sample.")
    for field in ("MULTI_PURPOSE", "VERY_HIGH_ECON", "MALE", "EMPLOYED", "AGE_UNDER30", "URBAN_REGION"):
        if not numeric(sample[field]).isin([0, 1]).all():
            raise AssertionError(f"Invalid retained binary context: {field}")
    if not numeric(sample.TOUR_DISTANCE_COMPLETE_FLAG).eq(1).all():
        raise AssertionError("Incomplete Tour distance was retained.")
    if not sample[[f"MAIN_PURPOSE_{p}" for p in PURPOSES[:-1]]].apply(numeric).sum(axis=1).eq(1).all():
        raise AssertionError("Retained official main purpose is not one-hot among seven categories.")
    design = effective_design(sample)
    if not np.allclose(design.to_numpy(float), -effective_design(sample, swap=True).to_numpy(float), rtol=0, atol=1e-12):
        raise AssertionError("Car-label swap failed to negate the full 21-column design.")
    current_ids = set(sample.CHOICE_ID)
    flow = sample_flow(context, reasons, sample)
    previous_comparison = None
    if ROOT == STAGING_ROOT and PREVIOUS_CORE_V2_SAMPLE.is_file():
        previous = pd.read_csv(PREVIOUS_CORE_V2_SAMPLE, usecols=["CHOICE_ID"], dtype=str).CHOICE_ID
        if not previous.is_unique:
            raise AssertionError("Previous Core-V2 CHOICE_ID is duplicated.")
        previous_ids = set(previous)
        previous_comparison = {"previous_n": len(previous), "current_n": len(sample),
                               "common": len(previous_ids & current_ids),
                               "previous_only": len(previous_ids-current_ids),
                               "current_only": len(current_ids-previous_ids),
                               "previous_choice_id_sequence_sha256": hashlib.sha256(
                                   "\n".join(previous).encode("utf-8")).hexdigest()}
    effective = pd.concat([sample[["CHOICE_ID", "H_ID", *CONTROLS]].reset_index(drop=True),
                           design.reset_index(drop=True)], axis=1)
    if list(effective.columns) != ["CHOICE_ID", "H_ID", *CONTROLS, *PREDICTORS]:
        raise AssertionError("Core-V2 effective design schema is not 27 columns.")
    metadata = {
        "specification": "fixed Stage-1 Core-V2; five V2 contrasts and sixteen interactions",
        "main_effects": [{"predictor": n, "formula": f"{a} - {b}"} for n, a, b in MAIN],
        "interactions": [{"predictor": n, "formula": f"{c} * {d}"} for n, c, d in INTERACTIONS],
        "distance_mean_km": mean_distance, "distance_scale_km": 10,
        "distance_rule": "Linear (TOUR_DISTANCE_KM - final Core-V2 mean km)/10; no cap or log.",
        "purpose_count_rule": "Seven distinct valid broad hwzweck1 purpose categories at non-home destinations; repeated category counts once. MULTI_PURPOSE=1 for >=2. Zero identified or one with unknown non-home purpose is excluded; >=2 remains established despite another unknown.",
        "purpose_count_source": "TOUR_N_DISTINCT_PURPOSES; encoded by TOUR_PURPOSE_COUNT_0/1/2/3PLUS; cross-checked with TOUR_MULTI_PURPOSE_FLAG",
        "distance_completeness_rule": "TOUR_DISTANCE_COMPLETE_FLAG=1 requires every constituent trip distance valid; partial sums are excluded.",
        "purpose_rule": "Official closing-trip hwzweck2 1-7 required; Work/Escort/Shopping interact; pooled reference Business/Education/Errand/Leisure.",
        "sex_rule": "HP_SEX 1 male, 2 female; other values excluded.",
        "employment_rule": "MiD erwerb 1 employed including trainees, 0 not employed; other values excluded.",
        "age_rule": "alter_gr5 4-5 ages 18-29, 6-15 ages 30+; code 3 under 18 excluded.",
        "economic_rule": "MiD oek_status valid 1-5; VERY_HIGH_ECON=1 for 5 (sehr hoch), 0 for valid 1-4. No ordinal linear slope.",
        "urban_rule": "RegioStaR4 11/12 urban, 21/22 nonurban.",
        "status_rule": "MiD status class from KBA segment and production year; not direct vehicle capacity.",
        "sample": {"initial_n": len(matrix), "stage0_v2_valid_n": int((~context.invalid_v2_vehicle).sum()),
                   "final_n": len(sample), "households": int(sample.H_ID.nunique())},
        "previous_core_v2_comparison": previous_comparison,
        "choice_id_sequence_sha256": hashlib.sha256("\n".join(sample.CHOICE_ID).encode("utf-8")).hexdigest(),
        "purpose_count_initial": {category: int(context[f"TOUR_PURPOSE_COUNT_{category}"].astype(int).sum())
                                  for category in PURPOSE_COUNTS},
        "purpose_count_final": {category: int(sample[f"TOUR_PURPOSE_COUNT_{category}"].astype(int).sum())
                                for category in PURPOSE_COUNTS},
        "purpose_unknown_initial_n": int(numeric(context.TOUR_PURPOSE_UNKNOWN_FLAG).eq(1).sum()),
        "purpose_ambiguous_initial_n": int(context.uncertain_or_invalid_purpose_count.sum()),
        "distance_incomplete_initial_n": int(context.incomplete_tour_distance.sum()),
        "distance_incomplete_nonmissing_initial_n": int((context.incomplete_tour_distance &
                                                             numeric(context.TOUR_DISTANCE_KM).notna()).sum()),
        "economic_levels_final": {str(code): int(numeric(sample.oek_status).eq(code).sum()) for code in range(1, 6)},
        "source_hashes_sha256": before,
        "source_paths": list(before),
    }
    # Validate all accepted inputs again before the first output write.
    verify_current_sources(before)
    DATA.mkdir(parents=True, exist_ok=False)
    sample.to_csv(DATA / "core_v2_context_sample.csv", index=False)
    effective.to_csv(DATA / "core_v2_effective_design.csv", index=False)
    flow.to_csv(DATA / "core_v2_sample_flow.csv", index=False)
    (DATA / "core_v2_transformations.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    manifest = {"status": "PREPARED_PENDING_DIAGNOSTICS",
                "source_hashes_sha256": before, "sample": metadata["sample"],
                "output_hashes_sha256": {p.name: sha256(p) for p in sorted(DATA.glob("core_v2_*"))},
                "code_sha256": sha256(Path(__file__))}
    (DATA / "core_v2_input_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = prepare()
    print(json.dumps(result["sample"], indent=2))
