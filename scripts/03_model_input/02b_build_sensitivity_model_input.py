"""Build sensitivity features using accepted 07 functions and a fixed baseline schema.

No constant-variable selection or historical controlled-replacement test is run.
Only the sensitivity output directory is written. Continuous overlap comparisons
use rtol=1e-12, atol=1e-15; identifiers and discrete features remain exact.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import MODEL_INPUT_SENSITIVITY_DIR, PHASE4_SENSITIVITY_DIR
from src.thesis_pipeline.model_input import vehicle_choice as accepted
from src.thesis_pipeline.sensitivity.decision_timing import (
    read, serialized, reject, ids, overlap_equal, fingerprints,
    check_fingerprints, descriptive,
)
OUT = MODEL_INPUT_SENSITIVITY_DIR
MODEL = OUT / "mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv"
MANIFEST = OUT / "decision_timing_sensitivity_model_input_manifest.csv"
QA = OUT / "decision_timing_sensitivity_model_input_QA.csv"
BACKBONE = PHASE4_SENSITIVITY_DIR / "mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv"


def main():
    before = fingerprints()
    # The path resolver/load_inputs are read-only. Keep model_input pointing at
    # the baseline reference; never call accepted.save_outputs or accepted.main.
    paths = replace(accepted.resolve_paths(), backbone=BACKBONE)
    inputs = accepted.load_inputs(paths)
    baseline = read(paths.model_input)
    baseline_manifest = read(paths.manifest)
    backbone = inputs["backbone"]
    invariants = accepted.validate_backbone(backbone)
    accepted.canonicalize_candidate_registry(paths.candidate_registry)
    tour = accepted.select_tour_features(backbone)
    trip = accepted.select_trip_context_features(inputs["trips"])
    household = accepted.select_household_features(inputs["households"], inputs["persons"])
    person = accepted.select_person_features(inputs["persons"])
    vehicle = accepted.prepare_vehicle_features(inputs["vehicles"], set(backbone.H_ID))
    merged, merge_metrics = accepted.merge_features(backbone,tour,trip,household,person,vehicle)
    accepted.validate_missing_codes(merged)
    # Baseline schema is authoritative even if candidate variation changes.
    missing = set(baseline.columns) - set(merged.columns)
    if missing:
        raise AssertionError(f"Baseline-retained columns cannot be constructed: {sorted(missing)}")
    final = merged[list(baseline.columns)].copy()
    accepted.run_final_assertions(invariants,backbone,merged,final,vehicle)
    final_csv = serialized(final)
    added = overlap_equal(baseline,final_csv)
    ids(final_csv)
    if len(final) != len(backbone):
        raise AssertionError("Sensitivity row count changed.")
    if len(final) != 38143:
        print(f"WARNING: accepted sensitivity count differs from 38,143: {len(final):,}; verify intentional upstream change.")
    if len(baseline) != 30816:
        print(f"WARNING: baseline reference count differs from 30,816: {len(baseline):,}.")
    w = pd.to_numeric(final_csv.W_GEW,errors="coerce")
    reject(final_csv, w.isna() | ~np.isfinite(w) | w.le(0), "Invalid sensitivity weight.")
    for col in ["AV_1","AV_2"]:
        reject(final_csv, pd.to_numeric(final_csv[col],errors="coerce").ne(1), "Conceptual availability must equal 1.")
    reject(final_csv, ~pd.to_numeric(final_csv.CHOICE,errors="coerce").isin([1,2]), "Invalid CHOICE.")
    generated_manifest = serialized(accepted.build_manifest(list(baseline.columns)))
    if not generated_manifest.equals(baseline_manifest):
        print(generated_manifest.compare(baseline_manifest).head(10).to_string())
        raise AssertionError("Accepted manifest definitions/order differ from current feature catalog.")
    if baseline_manifest.final_variable.tolist() != list(final.columns):
        raise AssertionError("Manifest variable order differs from fixed model schema.")
    summary = {
        "baseline model rows":len(baseline), "sensitivity model rows":len(final),
        "baseline-overlap rows":len(baseline), "new sensitivity rows":len(added),
        "baseline households":baseline.H_ID.nunique(), "sensitivity households":final.H_ID.nunique(),
        "baseline persons":baseline.HP_ID.nunique(), "sensitivity persons":final.HP_ID.nunique(),
        "added households":len(set(final.H_ID)-set(baseline.H_ID)),
        "added persons":len(set(final.HP_ID)-set(baseline.HP_ID)),
        "missing W_GEW":int(w.isna().sum()), "non-positive W_GEW":int(w.le(0).sum()),
        "minimum W_GEW":w.min(), "maximum W_GEW":w.max(), "mean W_GEW":w.mean(), "median W_GEW":w.median(),
    }
    rows = [dict(section="summary",metric=k,value=v) for k,v in summary.items()]
    rows.extend(dict(section="source merge",metric=k,value=v) for k,v in merge_metrics.items())
    for sample,frame in [("baseline",baseline),("sensitivity",final_csv),("extension",added)]:
        for column in baseline.columns:
            if column in accepted.ESTIMATION_COLUMNS: continue
            missing = frame[column].eq("")
            rows.append(dict(section="missingness",sample=sample,variable=column,
                non_missing_count=int((~missing).sum()),missing_count=int(missing.sum()),
                missing_share=float(missing.mean())))
        for col in ["CHOICE","AV_1","AV_2"]:
            for category,count in frame[col].value_counts().sort_index().items():
                rows.append(dict(section=sample,metric=f"{col} distribution",category=category,value=count))
    descriptive(rows,"extension",added)
    check_fingerprints(before)
    OUT.mkdir(parents=True,exist_ok=True)
    final.to_csv(MODEL,index=False)
    # Byte-for-byte reproduction preserves all accepted substantive definitions.
    MANIFEST.write_bytes(paths.manifest.read_bytes())
    pd.DataFrame(rows).to_csv(QA,index=False)
    overlap_equal(baseline,read(MODEL))
    check_fingerprints(before)
    print("\nDECISION-TIMING SENSITIVITY DOWNSTREAM PIPELINE")
    for k,v in summary.items(): print(f"{k}: {v}")
    print("baseline overlap feature equality: PASS (exact discrete; continuous rtol=1e-12, atol=1e-15)")
    phase4 = read(BACKBONE,["TOUR_MATCH_FOUND"])
    print(f"tour matching: {phase4.TOUR_MATCH_FOUND.eq('1').sum()} / {len(final)}")
    print("CHOICE distribution:"); print(final.CHOICE.value_counts().sort_index().to_string())
    for col in ["AV_1","AV_2"]: print(f"{col} == 1: {pd.to_numeric(final[col]).eq(1).sum()}")
    print("OUTPUTS:")
    for path in [MODEL,MANIFEST,QA]: print(path)
    print("Baseline reconstruction, Phase 4 outputs, and baseline model-input outputs were not modified.")
    print("Accepted sensitivity Phase 3 outputs were not modified.")

if __name__ == "__main__":
    main()
