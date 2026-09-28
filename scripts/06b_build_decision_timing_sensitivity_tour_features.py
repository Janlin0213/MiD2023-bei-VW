"""Attach accepted, sample-independent tours; never reconstruct tours."""
import pandas as pd
from decision_timing_sensitivity_utils import (
    ROOT, read, serialized, reject, ids, overlap_equal, fingerprints,
    check_fingerprints, load_script,
)
RECON = ROOT / "data_processed/reconstruction"
OUT = RECON / "phase4_sensitivity"
OUTPUT = OUT / "mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv"

def main():
    before = fingerprints()
    accepted = load_script("06_build_home_based_tour_features.py")
    wide = read(RECON / "phase3_sensitivity/mnl_vehicle_choice_wide_decision_timing_sensitivity_base.csv")
    baseline = read(RECON / "phase4/mnl_vehicle_choice_wide_tour_features.csv")
    tours = read(RECON / "phase4/home_based_tour_features.csv")
    trips = read(RECON / "phase1/trips_home_chain_enriched.csv", ["SOURCE_ROW_ID", "CHAIN_ELIGIBLE", "HOME_ORIGIN"])
    ids(wide)
    # Only the accepted merge is reused; no chain preparation or reconstruction.
    output = accepted.merge_to_strict_wide(wide, trips, tours)
    reject(output, output.TOUR_MATCH_FOUND.ne(1), "Every sensitivity choice must match an accepted tour start.")
    output = serialized(output)
    if list(output.columns) != list(baseline.columns):
        raise AssertionError("Accepted Phase 4 schema differs from merged schema.")
    if not output[wide.columns].equals(wide):
        raise AssertionError("Sensitivity Phase 3 backbone values or row order changed.")
    added = overlap_equal(baseline, output)
    if added.empty:
        raise AssertionError("Baseline must be a proper subset of sensitivity choices.")
    metrics = {
        "baseline Phase 4 choices":len(baseline), "sensitivity Phase 4 choices":len(output),
        "overlapping baseline choices":len(baseline), "new sensitivity choices":len(added),
        "sensitivity households":output.H_ID.nunique(), "baseline households":baseline.H_ID.nunique(),
        "sensitivity tour matches":len(output), "sensitivity tour match rate":1.0,
        "extension closed tours":int(pd.to_numeric(added.TOUR_CLOSED).eq(1).sum()),
        "extension open tours":int(pd.to_numeric(added.TOUR_OPEN_END_FLAG).eq(1).sum()),
    }
    check_fingerprints(before)
    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT,index=False)
    check_fingerprints(before)
    print("DECISION-TIMING SENSITIVITY PHASE 4")
    for k,v in metrics.items(): print(f"{k}: {v}")
    print("baseline overlap feature equality: PASS")
    print(OUTPUT)
    print("Baseline reconstruction, Phase 4 outputs, and baseline model-input outputs were not modified.")

if __name__ == "__main__":
    main()
