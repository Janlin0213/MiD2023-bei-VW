"""Compare raw tour-purpose candidates across the saved analysis populations."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

import pandas as pd

from src.thesis_pipeline.paths import MODEL_INPUT_DIR, PHASE4_DIR, PHASE1_DIR, OUTPUTS_DIR
from src.thesis_pipeline.model_input import purpose_qa as qa
from src.thesis_pipeline.model_input.tour_features import VEHICLE_QA_COLUMNS, vehicle_continuity


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> None:
    tours = read(PHASE4_DIR / "home_based_tour_features.csv")
    membership = read(PHASE4_DIR / "tour_trip_membership.csv")
    fleet = read(PHASE1_DIR / "trips_home_chain_enriched.csv")[
        ["SOURCE_ROW_ID", "W_WAUTO", "FLEET_CAR_SET"]]
    membership = membership.merge(fleet, on="SOURCE_ROW_ID", how="left",
                                  validate="one_to_one", sort=False)
    if membership[["W_WAUTO", "FLEET_CAR_SET"]].isna().any().any():
        raise AssertionError("Tour member missing from the accepted Phase 1 source.")
    continuity, detail = vehicle_continuity(membership)
    check = tours[["TOUR_ID", *VEHICLE_QA_COLUMNS]].merge(
        continuity, on="TOUR_ID", validate="one_to_one", suffixes=("", "_computed"))
    if len(check) != len(tours) or any(
        not qa.numeric(check[column]).equals(qa.numeric(check[f"{column}_computed"]))
        for column in VEHICLE_QA_COLUMNS
    ):
        raise AssertionError("Saved Tour continuity fields differ from membership evidence.")
    strict = read(MODEL_INPUT_DIR / "mnl_vehicle_choice_model_input.csv")
    candidate = read(MODEL_INPUT_DIR / "allocation_candidate_model_input.csv")
    single = read(MODEL_INPUT_DIR / "single_driver_mnl_design_matrix.csv")
    populations = {
        "all reconstructed tours": tours,
        "closed reconstructed tours": tours.loc[qa.numeric(tours.TOUR_CLOSED).eq(1)],
        "non-closed reconstructed tours": tours.loc[qa.numeric(tours.TOUR_CLOSED).ne(1)],
        "strict canonical choice occasions": qa.attach_tour_diagnostics(strict, tours),
        "allocation-candidate choice occasions": qa.attach_tour_diagnostics(candidate, tours),
        "retained single-driver choice occasions": qa.attach_tour_diagnostics(single, tours),
    }
    saved_core = OUTPUTS_DIR / "mnl" / "stage1" / "core_v1" / "data" / "core_v1_context_sample.csv"
    if saved_core.is_file():
        core_ids = pd.read_csv(saved_core, usecols=["CHOICE_ID"], dtype=str)["CHOICE_ID"]
        if core_ids.duplicated().any() or not core_ids.isin(single.CHOICE_ID).all():
            raise AssertionError("Saved Core-V1 sample is not a unique subset of the current single-driver input.")
        core = single.set_index("CHOICE_ID").loc[core_ids].reset_index()
        populations["saved Core-V1 choice occasions (read-only)"] = qa.attach_tour_diagnostics(core, tours)
    rows: list[dict] = []
    tables = []
    for name, population in populations.items():
        rows.extend(qa.summarize_population(population, name))
        tables.append(qa.disagreement_crosstab(population, name))
    single_diag = populations["retained single-driver choice occasions"]
    closing_max = qa.numeric(single_diag.TOUR_LONGEST_IS_CLOSING_TRIP_FLAG).eq(1)
    tied_max = qa.numeric(single_diag.TOUR_LONGEST_DURATION_TIE_FLAG).eq(1)
    structural = {
        "single-driver: max set includes closing trip": single_diag.loc[closing_max],
        "single-driver: unique longest is closing trip": single_diag.loc[closing_max & ~tied_max],
        "single-driver: tied max includes closing trip": single_diag.loc[closing_max & tied_max],
        "single-driver: other return-home trip": single_diag.loc[qa.numeric(single_diag.TOUR_LONGEST_RETURN_HOME_FLAG).eq(1) & qa.numeric(single_diag.TOUR_LONGEST_IS_CLOSING_TRIP_FLAG).ne(1)],
        "single-driver: longest not return-home": single_diag.loc[qa.numeric(single_diag.TOUR_LONGEST_RETURN_HOME_FLAG).ne(1)],
        "single-driver: closed tour": single_diag.loc[qa.numeric(single_diag.TOUR_CLOSED).eq(1)],
    }
    for name, population in structural.items():
        rows.extend(qa.summarize_population(population, name))
        tables.append(qa.disagreement_crosstab(population, name))
    rows.extend(qa.main_scope_rows(tours, membership))
    for name in ("strict canonical choice occasions", "allocation-candidate choice occasions",
                 "retained single-driver choice occasions",
                 *(["saved Core-V1 choice occasions (read-only)"] if saved_core.is_file() else [])):
        rows.extend(qa.vehicle_choice_rows(populations[name], detail, name))
    summary = pd.DataFrame(rows)
    crosstab = pd.concat(tables, ignore_index=True)
    summary.to_csv(MODEL_INPUT_DIR / "purpose_comparison_QA.csv", index=False)
    crosstab.to_csv(MODEL_INPUT_DIR / "purpose_disagreement_crosstab.csv", index=False)
    print(summary.loc[summary.section.isin(["coverage", "pairwise"])].to_string(index=False))
    print("Saved purpose QA and semantic cross-tab.")


if __name__ == "__main__":
    main()
