"""Prepare independent Stage 2 V1/V2 MNL difference designs from one upstream CSV.

This script neither imports Stage 1 nor reads its results. It does not estimate
models or screen candidate interactions. Outputs are deterministic for a fixed
input and code revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from src.thesis_pipeline.paths import (  # noqa: E402
    OUTPUTS_DIR, REPO_ROOT, SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH,
)


OUTPUT_DIR = OUTPUTS_DIR / "mnl" / "stage2" / "preparation"
IDS = ("CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID")
CONTROLS = ("CHOICE", "AV_1", "AV_2", "W_GEW")
POWERTRAIN = ("ICE", "HEV", "PHEV", "BEV", "OTHER", "MISSING")
SEGMENT = ("SMALL", "COMPACT", "MEDIUM", "LARGE", "MISSING")
STATUS = ("LOW", "MEDIUM", "HIGH", "MISSING")
SEX = ("MALE", "FEMALE", "DIVERSE", "MISSING")
AGE = ("UNDER18", "UNDER30", "30_49", "50PLUS", "MISSING")
PURPOSE = ("WORK", "BUSINESS", "EDUCATION", "SHOPPING", "ERRAND", "LEISURE", "ESCORT", "MISSING")
PURPOSE_COUNT = ("0", "1", "2", "3PLUS")
HH_TYPE = ("ADULT", "FAMILY", "SENIOR", "YOUNG")
ECON = ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")
REGION = ("METROPOLITAN_URBAN", "REGIOPOLITAN_URBAN", "RURAL_NEAR_URBAN", "PERIPHERAL_RURAL")

# Context names, sources, and definitions form the preparation feature registry.
CONTEXTS = (
    ("TOUR_TIME_C_H", "TOUR_TRAVEL_TIME_MIN", "(minutes - retained-pool mean minutes) / 60"),
    ("MULTI_PURPOSE", "TOUR_PURPOSE_COUNT_2,TOUR_PURPOSE_COUNT_3PLUS", "2+ distinct valid broad purposes"),
    ("MAIN_WORK", "MAIN_PURPOSE_WORK", "official closing-trip hwzweck2 = 1"),
    ("MAIN_ESCORT", "MAIN_PURPOSE_ESCORT", "official closing-trip hwzweck2 = 7"),
    ("MAIN_LEISURE", "MAIN_PURPOSE_LEISURE", "official closing-trip hwzweck2 = 6"),
    ("MAIN_SHOPPING", "MAIN_PURPOSE_SHOPPING", "official closing-trip hwzweck2 = 4"),
    ("HOUSEHOLD_ACCOMPANIED", "HOUSEHOLD_ACCOMPANIED", "verified household companion; 0/1"),
    ("FEMALE", "SEX_FEMALE", "female versus male"),
    ("AGE_18_29", "AGE_UNDER30", "alter_gr5 codes 4-5; ages 18-29"),
    ("AGE_50PLUS", "AGE_50PLUS", "alter_gr5 codes 10-15; ages 50+"),
    ("EMPLOYED", "erwerb", "MiD erwerb = 1 versus 0"),
    ("FAMILY", "HH_TYPE_FAMILY", "family household versus other valid types"),
    ("VERY_HIGH_ECON", "ECON_VERY_HIGH", "oek_status = 5 versus valid 1-4"),
    ("URBAN_REGION", "REGIO_METROPOLITAN_URBAN,REGIO_REGIOPOLITAN_URBAN", "RegioStaR4 11/12 versus 21/22"),
    ("FAMILY_X_ESCORT", "FAMILY,MAIN_ESCORT", "family × escort"),
    ("VERY_HIGH_ECON_X_WORK", "VERY_HIGH_ECON,MAIN_WORK", "very high economic status × work"),
    ("URBAN_X_LEISURE", "URBAN_REGION,MAIN_LEISURE", "urban region × leisure"),
)
MAIN_V1 = ("PT_HEV", "PT_PHEV", "PT_BEV", "SEG_COMPACT", "SEG_MEDIUM", "SEG_LARGE", "VEHICLE_AGE")
MAIN_V2 = ("PT_HEV", "PT_PHEV", "PT_BEV", "STATUS_LOW", "STATUS_HIGH")
INTERACT_V1 = ("PT_PHEV", "PT_BEV", "SEG_LARGE", "SEG_MEDIUM", "SEG_COMPACT", "VEHICLE_AGE")
INTERACT_V2 = ("PT_PHEV", "PT_BEV", "STATUS_LOW", "STATUS_HIGH")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column].replace("", np.nan), errors="coerce")


def block(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = values.isin([0, 1]).all(axis=1) & values.sum(axis=1).eq(1)
    return values, valid


def required_columns() -> list[str]:
    columns = [*IDS, *CONTROLS, "TOUR_TRAVEL_TIME_MIN", "HOUSEHOLD_ACCOMPANIED",
               "erwerb", "HOUSEHOLD_ACCOMPANIED_MISSING", "ERWERB_MISSING",
               "VEHICLE_AGE_1", "VEHICLE_AGE_2"]
    blocks = [("SEX", SEX), ("AGE", AGE), ("MAIN_PURPOSE", PURPOSE),
              ("TOUR_PURPOSE_COUNT", PURPOSE_COUNT), ("HH_TYPE", HH_TYPE),
              ("ECON", ECON), ("REGIO", REGION)]
    columns.extend(f"{prefix}_{category}" for prefix, categories in blocks for category in categories)
    for car in (1, 2):
        for prefix, categories in ((f"PT{car}", POWERTRAIN), (f"SEG{car}", SEGMENT),
                                   (f"STATUS{car}", STATUS)):
            columns.extend(f"{prefix}_{category}" for category in categories)
        columns.append(f"VEHICLE_AGE_{car}_MISSING")
    return columns


def eligibility(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nonexclusive exclusion reasons and validated encoded blocks."""
    encoded = pd.DataFrame(index=frame.index)
    valid_blocks = {}
    groups = [("sex", "SEX", SEX), ("age", "AGE", AGE),
              ("purpose", "MAIN_PURPOSE", PURPOSE),
              ("purpose_count", "TOUR_PURPOSE_COUNT", PURPOSE_COUNT),
              ("household", "HH_TYPE", HH_TYPE), ("economic", "ECON", ECON),
              ("region", "REGIO", REGION)]
    for car in (1, 2):
        groups.extend(((f"pt{car}", f"PT{car}", POWERTRAIN),
                       (f"segment{car}", f"SEG{car}", SEGMENT),
                       (f"status{car}", f"STATUS{car}", STATUS)))
    for name, prefix, categories in groups:
        cols = [f"{prefix}_{category}" for category in categories]
        values, valid_blocks[name] = block(frame, cols)
        encoded[cols] = values

    choice, av1, av2, weight = (numeric(frame, c) for c in CONTROLS)
    time = numeric(frame, "TOUR_TRAVEL_TIME_MIN")
    accompanied = numeric(frame, "HOUSEHOLD_ACCOMPANIED")
    employed = numeric(frame, "erwerb")
    reasons = {
        "invalid_identifier": frame[list(IDS)].apply(lambda s: s.str.strip().eq("")).any(axis=1),
        "invalid_choice_or_availability": ~choice.isin([1, 2]) | av1.ne(1) | av2.ne(1),
        "invalid_weight": ~np.isfinite(weight) | weight.le(0),
        "invalid_powertrain": ~valid_blocks["pt1"] | ~valid_blocks["pt2"] |
            encoded[["PT1_OTHER", "PT1_MISSING", "PT2_OTHER", "PT2_MISSING"]].eq(1).any(axis=1),
        "invalid_segment": ~valid_blocks["segment1"] | ~valid_blocks["segment2"] |
            encoded[["SEG1_MISSING", "SEG2_MISSING"]].eq(1).any(axis=1),
        "invalid_vehicle_age": pd.Series(False, index=frame.index),
        "invalid_status": ~valid_blocks["status1"] | ~valid_blocks["status2"] |
            encoded[["STATUS1_MISSING", "STATUS2_MISSING"]].eq(1).any(axis=1),
        "invalid_tour_time": ~np.isfinite(time) | time.lt(0),
        "invalid_purpose_count": ~valid_blocks["purpose_count"] |
            encoded["TOUR_PURPOSE_COUNT_0"].eq(1),
        "missing_or_invalid_main_purpose": ~valid_blocks["purpose"] |
            encoded["MAIN_PURPOSE_MISSING"].eq(1),
        "invalid_accompaniment": ~accompanied.isin([0, 1]) |
            ~numeric(frame, "HOUSEHOLD_ACCOMPANIED_MISSING").eq(0),
        "diverse_or_missing_sex": ~valid_blocks["sex"] |
            encoded[["SEX_DIVERSE", "SEX_MISSING"]].eq(1).any(axis=1),
        "under18_or_missing_age": ~valid_blocks["age"] |
            encoded[["AGE_UNDER18", "AGE_MISSING"]].eq(1).any(axis=1),
        "missing_or_invalid_employment": ~employed.isin([0, 1]) |
            ~numeric(frame, "ERWERB_MISSING").eq(0),
        "invalid_household_type": ~valid_blocks["household"],
        "invalid_economic_status": ~valid_blocks["economic"],
        "invalid_region": ~valid_blocks["region"],
    }
    for car in (1, 2):
        age = numeric(frame, f"VEHICLE_AGE_{car}")
        marker = numeric(frame, f"VEHICLE_AGE_{car}_MISSING")
        reasons["invalid_vehicle_age"] |= (~np.isfinite(age) | age.lt(0) |
                                             age.gt(124) | marker.ne(0))
    return pd.DataFrame({name: mask.fillna(True).astype(bool) for name, mask in reasons.items()}), encoded


def contexts(sample: pd.DataFrame, encoded: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    out = pd.DataFrame(index=sample.index)
    mean_minutes = float(numeric(sample, "TOUR_TRAVEL_TIME_MIN").mean())
    out["TOUR_TIME_C_H"] = (numeric(sample, "TOUR_TRAVEL_TIME_MIN") - mean_minutes) / 60
    out["MULTI_PURPOSE"] = encoded["TOUR_PURPOSE_COUNT_2"] + encoded["TOUR_PURPOSE_COUNT_3PLUS"]
    for label in ("WORK", "ESCORT", "LEISURE", "SHOPPING"):
        out[f"MAIN_{label}"] = encoded[f"MAIN_PURPOSE_{label}"]
    out["HOUSEHOLD_ACCOMPANIED"] = numeric(sample, "HOUSEHOLD_ACCOMPANIED")
    out["FEMALE"] = encoded["SEX_FEMALE"]
    out["AGE_18_29"] = encoded["AGE_UNDER30"]
    out["AGE_50PLUS"] = encoded["AGE_50PLUS"]
    out["EMPLOYED"] = numeric(sample, "erwerb")
    out["FAMILY"] = encoded["HH_TYPE_FAMILY"]
    out["VERY_HIGH_ECON"] = encoded["ECON_VERY_HIGH"]
    out["URBAN_REGION"] = (encoded["REGIO_METROPOLITAN_URBAN"] +
                           encoded["REGIO_REGIOPOLITAN_URBAN"])
    out["FAMILY_X_ESCORT"] = out.FAMILY * out.MAIN_ESCORT
    out["VERY_HIGH_ECON_X_WORK"] = out.VERY_HIGH_ECON * out.MAIN_WORK
    out["URBAN_X_LEISURE"] = out.URBAN_REGION * out.MAIN_LEISURE
    assert list(out) == [name for name, _, _ in CONTEXTS]
    return out, mean_minutes


def vehicle_differences(sample: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=sample.index)
    for category in ("HEV", "PHEV", "BEV"):
        out[f"D_PT_{category}"] = encoded[f"PT1_{category}"] - encoded[f"PT2_{category}"]
    for category in ("COMPACT", "MEDIUM", "LARGE"):
        out[f"D_SEG_{category}"] = encoded[f"SEG1_{category}"] - encoded[f"SEG2_{category}"]
    out["D_VEHICLE_AGE"] = numeric(sample, "VEHICLE_AGE_1") - numeric(sample, "VEHICLE_AGE_2")
    for category in ("LOW", "HIGH"):
        out[f"D_STATUS_{category}"] = encoded[f"STATUS1_{category}"] - encoded[f"STATUS2_{category}"]
    return out


def make_design(controls: pd.DataFrame, context: pd.DataFrame,
                differences: pd.DataFrame, main: tuple[str, ...],
                interacting: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    features = {}
    manifest = []
    for attribute in main:
        name = f"D_{attribute}"
        features[name] = differences[name]
        left, right = (("VEHICLE_AGE_1", "VEHICLE_AGE_2") if attribute == "VEHICLE_AGE"
                       else (f"{attribute.split('_')[0]}1_{attribute.split('_', 1)[1]}",
                             f"{attribute.split('_')[0]}2_{attribute.split('_', 1)[1]}"))
        manifest.append({"feature": name, "kind": "main", "formula": f"{left} - {right}"})
    for context_name, _, _ in CONTEXTS:
        for attribute in interacting:
            name = f"D_{context_name}__{attribute}"
            features[name] = context[context_name] * differences[f"D_{attribute}"]
            manifest.append({"feature": name, "kind": "interaction",
                             "context": context_name, "vehicle_attribute": attribute,
                             "formula": f"{context_name} * D_{attribute}"})
    result = pd.concat([controls, pd.DataFrame(features, index=controls.index)], axis=1)
    return result, manifest


def sample_flow(frame: pd.DataFrame, reasons: pd.DataFrame) -> pd.DataFrame:
    weight = numeric(frame, "W_GEW").where(lambda s: np.isfinite(s) & s.gt(0), 0)
    excluded = pd.Series(False, index=frame.index)
    rows = []
    for name, mask in (("initial", pd.Series(True, index=frame.index)), *reasons.items()):
        new = mask & ~excluded if name != "initial" else mask
        if name != "initial":
            excluded |= mask
        rows.append({"reason": name, "affected_n": int(mask.sum()),
                     "affected_weight_sum": float(weight.loc[mask].sum()),
                     "new_exclusions_n": int(new.sum()) if name != "initial" else 0,
                     "retained_after_n": int((~excluded).sum())})
    keep = ~excluded
    rows.append({"reason": "final_common_pool", "affected_n": int(keep.sum()),
                 "affected_weight_sum": float(weight.loc[keep].sum()),
                 "new_exclusions_n": 0, "retained_after_n": int(keep.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    source = SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH
    before = sha256(source)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = set(required_columns()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required upstream columns: {sorted(missing)}")
    if len(frame) != 20_931 or len(frame.columns) != 208:
        raise ValueError(f"Unexpected upstream shape: {frame.shape}")
    if frame.CHOICE_ID.eq("").any() or frame.CHOICE_ID.duplicated().any():
        raise ValueError("CHOICE_ID must be populated and unique")
    reasons, encoded = eligibility(frame)
    keep = ~reasons.any(axis=1)
    sample = frame.loc[keep].copy().reset_index(drop=True)
    encoded = encoded.loc[keep].reset_index(drop=True)
    if sample.empty:
        raise ValueError("No eligible Stage 2 occasions")
    context, mean_minutes = contexts(sample, encoded)
    differences = vehicle_differences(sample, encoded)
    controls = sample[[*IDS, *CONTROLS]].copy()
    vehicle_columns = [f"{prefix}{car}_{category}"
                       for car in (1, 2)
                       for prefix, categories in (("PT", POWERTRAIN[:-2]),
                                                  ("SEG", SEGMENT[:-1]),
                                                  ("STATUS", STATUS[:-1]))
                       for category in categories]
    vehicle_values = encoded[vehicle_columns].copy()
    for car in (1, 2):
        vehicle_values[f"VEHICLE_AGE_{car}"] = numeric(sample, f"VEHICLE_AGE_{car}")
    common = pd.concat([controls, context, vehicle_values], axis=1)
    v1, v1_features = make_design(controls, context, differences, MAIN_V1, INTERACT_V1)
    v2, v2_features = make_design(controls, context, differences, MAIN_V2, INTERACT_V2)
    if len(v1_features) != 109 or len(v2_features) != 73:
        raise AssertionError("Incomplete main or interaction feature registry")
    for design in (common, v1, v2):
        predictors = design.drop(columns=list(IDS) + list(CONTROLS))
        if not np.isfinite(predictors.to_numpy(dtype=float)).all():
            raise AssertionError("Nonfinite retained predictor")
        if not design[list(IDS) + list(CONTROLS)].equals(controls):
            raise AssertionError("Choice controls changed across designs")
    if not np.isclose(context.TOUR_TIME_C_H.mean(), 0, atol=1e-12):
        raise AssertionError("Tour time is not centered on the final common pool")
    swapped_sample = sample.copy()
    swapped_encoded = encoded.copy()
    swapped_sample["VEHICLE_AGE_1"], swapped_sample["VEHICLE_AGE_2"] = (
        sample["VEHICLE_AGE_2"], sample["VEHICLE_AGE_1"])
    for stem, categories in (("PT", POWERTRAIN), ("SEG", SEGMENT), ("STATUS", STATUS)):
        for category in categories:
            first, second = f"{stem}1_{category}", f"{stem}2_{category}"
            swapped_encoded[first], swapped_encoded[second] = encoded[second], encoded[first]
    swapped_differences = vehicle_differences(swapped_sample, swapped_encoded)
    swapped_controls = controls.copy()
    swapped_controls["CHOICE"] = 3 - numeric(controls, "CHOICE")
    for design, main_features, interaction_features in ((v1, MAIN_V1, INTERACT_V1),
                                                         (v2, MAIN_V2, INTERACT_V2)):
        swapped_design, _ = make_design(swapped_controls, context, swapped_differences,
                                        main_features, interaction_features)
        predictors = design.drop(columns=list(IDS) + list(CONTROLS)).to_numpy(float)
        swapped_predictors = swapped_design.drop(columns=list(IDS) + list(CONTROLS)).to_numpy(float)
        if not np.allclose(predictors, -swapped_predictors, rtol=0, atol=1e-12) or \
                not (3 - numeric(swapped_design, "CHOICE")).equals(numeric(design, "CHOICE")):
            raise AssertionError("Vehicle-label swap consistency failed")
    if sha256(source) != before:
        raise AssertionError("Upstream input changed during preparation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {"stage2_common_pool.csv": common, "stage2_v1_design.csv": v1,
               "stage2_v2_design.csv": v2, "stage2_sample_flow.csv": sample_flow(frame, reasons)}
    for filename, data in outputs.items():
        data.to_csv(OUTPUT_DIR / filename, index=False, float_format="%.17g")
    manifest = {
        "source": source.relative_to(REPO_ROOT).as_posix(), "source_sha256": before,
        "code_sha256": sha256(Path(__file__)), "choice_mapping": "CHOICE 1 selects vehicle 1; 2 selects vehicle 2",
        "utility": "Unlabeled alternatives; generic coefficients; no vehicle-number constant",
        "difference_rule": "vehicle 1 minus vehicle 2",
        "references": {"powertrain": "ICE", "segment_v1": "SMALL", "status_v2": "MEDIUM",
                       "driver_age_context": "30-49", "sex_context": "MALE"},
        "tour_time": {"source": "TOUR_TRAVEL_TIME_MIN", "mean_minutes": mean_minutes,
                      "formula": "(minutes - mean_minutes) / 60"},
        "contexts": [{"feature": name, "source": source_fields, "definition": definition}
                     for name, source_fields, definition in CONTEXTS],
        "v1_features": v1_features, "v2_features": v2_features,
        "limitation": "Unprovided tour quality flags are not reconstructed or used.",
    }
    (OUTPUT_DIR / "stage2_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qa = {"initial_n": len(frame), "initial_households": int(frame.H_ID.nunique()),
          "final_n": len(sample), "final_households": int(sample.H_ID.nunique()),
          "final_weight_sum": float(numeric(sample, "W_GEW").sum()),
          "contexts": len(CONTEXTS), "v1_main": len(MAIN_V1),
          "v1_interactions": len(v1_features) - len(MAIN_V1),
          "v2_main": len(MAIN_V2), "v2_interactions": len(v2_features) - len(MAIN_V2),
          "common_order_and_controls_equal": True, "label_swap_pass": True,
          "output_sha256": {filename: sha256(OUTPUT_DIR / filename) for filename in outputs},
          "source_unchanged": sha256(source) == before}
    (OUTPUT_DIR / "stage2_preparation_qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if not qa["source_unchanged"]:
        raise AssertionError("Upstream input changed during output writing")
    print(json.dumps({k: qa[k] for k in ("initial_n", "final_n", "final_households",
                                           "v1_interactions", "v2_interactions")}, indent=2))


if __name__ == "__main__":
    main()
