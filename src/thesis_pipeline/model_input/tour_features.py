"""Reusable implementation of accepted home-based-tour feature construction.

Tours are reconstructed from the complete, chain-eligible Phase 1 person chain,
not from the household vehicle-event timeline.  Tour attributes are observed ex
post from the reporting-day chain and should be interpreted in later modelling
as proxies for mobility requirements that may have been anticipated at the
home-origin vehicle choice.  They support association and behavioural
interpretation, not a causal claim that the realised tour mechanically caused
the observed vehicle choice.

This module reads accepted Phase 1 and Phase 3 outputs and writes only new
downstream Phase 4 files.  It does not change sample construction or estimate a
choice model.  Immediate-leg and broader-tour attributes are deliberately both
retained for later model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import PHASE1_DIR, PHASE3_DIR, PHASE4_DIR

TRIPS_PATH = PHASE1_DIR / "trips_home_chain_enriched.csv"
WIDE_BASE_PATH = PHASE3_DIR / "mnl_vehicle_choice_wide_base.csv"

OUT_DIR = PHASE4_DIR
MEMBERSHIP_PATH = OUT_DIR / "tour_trip_membership.csv"
TOUR_FEATURES_PATH = OUT_DIR / "home_based_tour_features.csv"
WIDE_FEATURES_PATH = OUT_DIR / "mnl_vehicle_choice_wide_tour_features.csv"
QA_PATH = OUT_DIR / "tour_feature_QA_summary.csv"

DISTANCE_COLUMN = "wegkm_imp"
TRAVEL_TIME_COLUMN = "wegmin_imp2"
DISTANCE_MIN_KM = 0.01
DISTANCE_MAX_KM = 950.0
TRAVEL_TIME_MIN_MIN = 1.0
TRAVEL_TIME_MAX_MIN = 480.0
VALID_BROAD_PURPOSES = set(range(1, 8))
PURPOSE_INDICATORS = {
    1: "TOUR_HAS_WORK",
    2: "TOUR_HAS_BUSINESS",
    3: "TOUR_HAS_EDUCATION",
    4: "TOUR_HAS_SHOPPING",
    5: "TOUR_HAS_ERRAND",
    6: "TOUR_HAS_LEISURE",
    7: "TOUR_HAS_ESCORT",
}

DESTINATION_SPATIAL_SPECS = {
    "XMStadt_ZO": {
        "clean": "_XMSTADT_ZO",
        "nonhome": "_NONHOME_XMSTADT_ZO",
        "feature": "TOUR_WORST_XMSTADT_ZO",
        "valid_codes": set(range(1, 7)),
        "aggregation": "max",
    },
    "quali_opnv_zo": {
        "clean": "_QUALI_OPNV_ZO",
        "nonhome": "_NONHOME_QUALI_OPNV_ZO",
        "feature": "TOUR_WORST_QUALI_OPNV_ZO",
        "valid_codes": set(range(1, 5)),
        "aggregation": "min",
    },
}

# W_ZWECK 11--16 are documented detailed refinements of the summarised `zweck`
# categories.  These are reported but are not unexpected disagreements.
EXPECTED_PURPOSE_REFINEMENTS = {
    (11, 3),
    (12, 3),
    (13, 6),
    (14, 7),
    (15, 7),
    (16, 7),
}

ORIGINAL_INVARIANT_COLUMNS = [
    "CHOICE_ID",
    "SOURCE_ROW_ID",
    "CHOICE",
    "AV_1",
    "AV_2",
    "W_GEW",
]


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    unanchored_before_first_home_departure: int
    unanchored_after_first_home_departure: int
    expected_w_zweck_zweck_refinements: int
    broad_purpose_source: str | None
    official_main_purpose_available: bool


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required accepted input not found: {path}")
    frame = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def nonblank(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().ne("")


def make_fallback_person_key(frame: pd.DataFrame) -> pd.Series:
    hp = frame["HP_ID"].astype("string").str.strip() if "HP_ID" in frame.columns else pd.Series("", index=frame.index)
    fallback = "HHP|" + frame["H_ID"].astype("string") + "|" + frame["P_ID"].astype("string")
    return pd.Series(np.where(hp.ne(""), "HP|" + hp, fallback), index=frame.index, dtype="string")


def canonicalize_purpose(frame: pd.DataFrame) -> tuple[pd.Series, int]:
    if "W_ZWECK" not in frame.columns and "zweck" not in frame.columns:
        raise KeyError("Phase 1 trip data contains neither W_ZWECK nor accepted fallback zweck.")

    primary = numeric(frame["W_ZWECK"]) if "W_ZWECK" in frame.columns else pd.Series(np.nan, index=frame.index)
    fallback = numeric(frame["zweck"]) if "zweck" in frame.columns else pd.Series(np.nan, index=frame.index)
    both = primary.notna() & fallback.notna()
    disagreement = both & primary.ne(fallback)
    pairs = pd.DataFrame({"W_ZWECK": primary[disagreement], "zweck": fallback[disagreement]}).astype(int)
    expected = pairs.apply(lambda row: (int(row.W_ZWECK), int(row.zweck)) in EXPECTED_PURPOSE_REFINEMENTS, axis=1)
    unexpected_pairs = pairs.loc[~expected]
    if not unexpected_pairs.empty:
        print("\nUNEXPECTED W_ZWECK / zweck DISAGREEMENTS")
        print(unexpected_pairs.value_counts().head(20).to_string())
        raise ValueError("W_ZWECK and zweck disagree outside the documented detailed-purpose refinements.")

    canonical = primary.mask(primary.isna(), fallback)
    return canonical.round().astype("Int64"), int(expected.sum())


def canonicalize_trip_attributes(trips: pd.DataFrame) -> tuple[pd.DataFrame, ReconstructionDiagnostics]:
    required = [
        "SOURCE_ROW_ID",
        "H_ID",
        "P_ID",
        "W_ID",
        "CHAIN_ELIGIBLE",
        "W_RBW",
        "CHAIN_SEQUENCE_NUM",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "START_MIN",
        "ARRIVAL_MIN",
        DISTANCE_COLUMN,
        TRAVEL_TIME_COLUMN,
        *DESTINATION_SPATIAL_SPECS,
    ]
    require_columns(trips, required, "Phase 1")
    out = trips.copy()
    if "HP_ID" not in out.columns:
        out["HP_ID"] = ""

    if "PERSON_KEY" in out.columns:
        missing_person_key = ~nonblank(out["PERSON_KEY"])
        if missing_person_key.any():
            print(out.loc[missing_person_key, ["SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID"]].head(10).to_string(index=False))
            raise ValueError("Accepted PERSON_KEY exists but is missing on Phase 1 rows.")
    else:
        out["PERSON_KEY"] = make_fallback_person_key(out)

    if ~nonblank(out["SOURCE_ROW_ID"]).all() or not out["SOURCE_ROW_ID"].is_unique:
        raise ValueError("SOURCE_ROW_ID must be non-missing and unique in the accepted Phase 1 source.")

    out["_SOURCE_ROW_NUM"] = numeric(out["SOURCE_ROW_ID"])
    out["_CHAIN_ELIGIBLE"] = numeric(out["CHAIN_ELIGIBLE"])
    out["_W_RBW"] = numeric(out["W_RBW"])
    out["_CHAIN_SEQUENCE"] = numeric(out["CHAIN_SEQUENCE_NUM"])
    out["_W_ID_SORT"] = numeric(out["W_ID_NUM"] if "W_ID_NUM" in out.columns else out["W_ID"])
    out["_HOME_ORIGIN"] = numeric(out["HOME_ORIGIN"]).where(numeric(out["HOME_ORIGIN"]).isin([0, 1]))
    out["_HOME_DESTINATION"] = numeric(out["HOME_DESTINATION"]).where(numeric(out["HOME_DESTINATION"]).isin([0, 1]))
    out["_START_MIN"] = numeric(out["START_MIN"])
    out["_ARRIVAL_MIN"] = numeric(out["ARRIVAL_MIN"])
    out["_PKW_FMF"] = numeric(out["pkw_fmf"]) if "pkw_fmf" in out.columns else np.nan
    for source, spec in DESTINATION_SPATIAL_SPECS.items():
        raw_spatial = numeric(out[source])
        out[spec["clean"]] = raw_spatial.where(raw_spatial.isin(spec["valid_codes"])).astype("Int64")

    invalid_chain_flag = ~out["_CHAIN_ELIGIBLE"].isin([0, 1])
    if invalid_chain_flag.any():
        raise ValueError("CHAIN_ELIGIBLE contains missing or non-binary values.")
    rbw_entering_chain = out["_CHAIN_ELIGIBLE"].eq(1) & out["_W_RBW"].eq(1)
    if rbw_entering_chain.any():
        print(out.loc[rbw_entering_chain, ["SOURCE_ROW_ID", "PERSON_KEY", "W_ID", "W_RBW", "CHAIN_ELIGIBLE"]].head(10).to_string(index=False))
        raise ValueError("rbW rows must never enter tour reconstruction.")

    eligible = out["_CHAIN_ELIGIBLE"].eq(1)
    invalid_order = eligible & (out["_CHAIN_SEQUENCE"].isna() | out["_CHAIN_SEQUENCE"].lt(1))
    if invalid_order.any():
        raise ValueError("Chain-eligible Phase 1 rows require a valid CHAIN_SEQUENCE_NUM.")
    duplicate_order = out.loc[eligible].duplicated(["PERSON_KEY", "_CHAIN_SEQUENCE"], keep=False)
    if duplicate_order.any():
        print(out.loc[eligible].loc[duplicate_order, ["PERSON_KEY", "SOURCE_ROW_ID", "W_ID", "CHAIN_SEQUENCE_NUM"]].head(10).to_string(index=False))
        raise ValueError("CHAIN_SEQUENCE_NUM must be unique within each accepted person chain.")

    out["CANONICAL_PURPOSE"], refinement_count = canonicalize_purpose(out)

    raw_distance = numeric(out[DISTANCE_COLUMN])
    distance_valid = raw_distance.notna() & np.isfinite(raw_distance) & raw_distance.between(DISTANCE_MIN_KM, DISTANCE_MAX_KM)
    out["TRIP_DISTANCE_KM"] = raw_distance.where(distance_valid)
    out["_DISTANCE_VALID"] = distance_valid.astype(int)

    raw_travel_time = numeric(out[TRAVEL_TIME_COLUMN])
    travel_valid = (
        raw_travel_time.notna()
        & np.isfinite(raw_travel_time)
        & raw_travel_time.between(TRAVEL_TIME_MIN_MIN, TRAVEL_TIME_MAX_MIN)
    )
    out["TRIP_TRAVEL_TIME_MIN"] = raw_travel_time.where(travel_valid)
    out["_TRAVEL_TIME_VALID"] = travel_valid.astype(int)

    if "hwzweck1" in out.columns:
        broad_source = "hwzweck1"
    elif "hwzweck2" in out.columns:
        broad_source = "hwzweck2"
    else:
        broad_source = None
    out["BROAD_PURPOSE"] = numeric(out[broad_source]).round().astype("Int64") if broad_source else pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["_MAIN_PURPOSE"] = numeric(out["hwzweck2"]).round().astype("Int64") if "hwzweck2" in out.columns else pd.Series(pd.NA, index=out.index, dtype="Int64")

    for column, label in [("BROAD_PURPOSE", broad_source), ("_MAIN_PURPOSE", "hwzweck2")]:
        if label is None:
            continue
        values = numeric(out.loc[eligible, column])
        unexpected = values.notna() & ~values.isin([*VALID_BROAD_PURPOSES, 99])
        if unexpected.any():
            raise ValueError(f"{label} contains values outside the documented 1--7 and 99 codes.")

    start_flag_valid = numeric(out["invalid_start_time_flag"]).fillna(0).eq(0) if "invalid_start_time_flag" in out.columns else True
    arrival_flag_valid = numeric(out["invalid_arrival_time_flag"]).fillna(0).eq(0) if "invalid_arrival_time_flag" in out.columns else True
    out["_START_TIME_VALID"] = (start_flag_valid & out["_START_MIN"].notna() & out["_START_MIN"].between(0, 1439)).astype(int)
    out["_ARRIVAL_TIME_VALID"] = (arrival_flag_valid & out["_ARRIVAL_MIN"].notna() & out["_ARRIVAL_MIN"].between(0, 2879)).astype(int)

    diagnostics = ReconstructionDiagnostics(
        unanchored_before_first_home_departure=0,
        unanchored_after_first_home_departure=0,
        expected_w_zweck_zweck_refinements=refinement_count,
        broad_purpose_source=broad_source,
        official_main_purpose_available="hwzweck2" in out.columns,
    )
    return out, diagnostics


def reconstruct_home_based_tours(
    prepared: pd.DataFrame,
    initial_diagnostics: ReconstructionDiagnostics,
) -> tuple[pd.DataFrame, pd.DataFrame, ReconstructionDiagnostics]:
    eligible = prepared.loc[prepared["_CHAIN_ELIGIBLE"].eq(1)].copy()
    eligible = eligible.sort_values(
        ["PERSON_KEY", "_CHAIN_SEQUENCE", "_W_ID_SORT", "_SOURCE_ROW_NUM", "SOURCE_ROW_ID"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    n_rows = len(eligible)
    tour_index = np.full(n_rows, -1, dtype=np.int64)
    start_flag = np.zeros(n_rows, dtype=np.int8)
    end_flag = np.zeros(n_rows, dtype=np.int8)
    closing_flag = np.zeros(n_rows, dtype=np.int8)
    unanchored_pre_flag = np.zeros(n_rows, dtype=np.int8)
    unanchored_other_flag = np.zeros(n_rows, dtype=np.int8)
    tours: list[dict[str, object]] = []

    person_values = eligible["PERSON_KEY"].astype(str).to_numpy()
    home_origins = eligible["_HOME_ORIGIN"].to_numpy()
    home_destinations = eligible["_HOME_DESTINATION"].to_numpy()
    current_person: str | None = None
    active_positions: list[int] = []
    active_sequence = 0
    next_sequence = 0
    seen_home_departure = False
    unanchored_pre_count = 0
    unanchored_other_count = 0

    def finish_active(closed: int, open_end: int, inferred_boundary: int) -> None:
        nonlocal active_positions
        if not active_positions:
            return
        index_value = len(tours)
        positions = np.asarray(active_positions, dtype=np.int64)
        tour_index[positions] = index_value
        start_flag[positions[0]] = 1
        end_flag[positions[-1]] = 1
        closing_flag[positions[-1]] = int(closed == 1)
        person_key = str(person_values[positions[0]])
        tours.append(
            {
                "_TOUR_INDEX": index_value,
                "TOUR_ID": f"{person_key}|TOUR|{active_sequence}",
                "TOUR_SEQUENCE_NUM": active_sequence,
                "PERSON_KEY": person_key,
                "TOUR_CLOSED": int(closed),
                "TOUR_OPEN_END_FLAG": int(open_end),
                "TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG": int(inferred_boundary),
            }
        )
        active_positions = []

    for position in range(n_rows):
        person = str(person_values[position])
        if current_person != person:
            finish_active(closed=0, open_end=1, inferred_boundary=0)
            current_person = person
            next_sequence = 0
            seen_home_departure = False

        origin_home = home_origins[position] == 1
        destination_home = home_destinations[position] == 1

        if active_positions and origin_home:
            # A second known home departure cannot belong to the still-open outing.
            finish_active(closed=0, open_end=0, inferred_boundary=1)

        if not active_positions:
            if origin_home:
                next_sequence += 1
                active_sequence = next_sequence
                active_positions = [position]
                seen_home_departure = True
            else:
                if not seen_home_departure:
                    unanchored_pre_flag[position] = 1
                    unanchored_pre_count += 1
                else:
                    unanchored_other_flag[position] = 1
                    unanchored_other_count += 1
                continue
        else:
            active_positions.append(position)

        if destination_home:
            finish_active(closed=1, open_end=0, inferred_boundary=0)

    finish_active(closed=0, open_end=1, inferred_boundary=0)

    eligible["UNANCHORED_BEFORE_FIRST_HOME_DEPARTURE_FLAG"] = unanchored_pre_flag
    eligible["UNANCHORED_OUTSIDE_ACTIVE_TOUR_FLAG"] = unanchored_other_flag
    eligible["_TOUR_INDEX"] = tour_index
    eligible["TOUR_START_TRIP_FLAG"] = start_flag
    eligible["TOUR_END_TRIP_FLAG"] = end_flag
    eligible["TOUR_CLOSING_TRIP_FLAG"] = closing_flag

    tour_meta = pd.DataFrame(tours)
    if tour_meta.empty:
        raise ValueError("No anchored home-based tours were reconstructed from the accepted Phase 1 chain.")

    membership = eligible.loc[eligible["_TOUR_INDEX"].ge(0)].copy()
    membership = membership.merge(tour_meta, on=["_TOUR_INDEX", "PERSON_KEY"], how="left", validate="many_to_one")
    diagnostics = ReconstructionDiagnostics(
        unanchored_before_first_home_departure=unanchored_pre_count,
        unanchored_after_first_home_departure=unanchored_other_count,
        expected_w_zweck_zweck_refinements=initial_diagnostics.expected_w_zweck_zweck_refinements,
        broad_purpose_source=initial_diagnostics.broad_purpose_source,
        official_main_purpose_available=initial_diagnostics.official_main_purpose_available,
    )
    return membership, tour_meta, diagnostics


def build_tour_features(membership: pd.DataFrame, tour_meta: pd.DataFrame) -> pd.DataFrame:
    work = membership.copy()
    work["_NONHOME_STOP"] = work["_HOME_DESTINATION"].eq(0).fillna(False).astype(int)
    for spec in DESTINATION_SPATIAL_SPECS.values():
        work[spec["nonhome"]] = work[spec["clean"]].where(work["_NONHOME_STOP"].eq(1))
    work["_UNKNOWN_HOME_ORIGIN"] = work["_HOME_ORIGIN"].isna().astype(int)
    work["_UNKNOWN_HOME_DESTINATION"] = work["_HOME_DESTINATION"].isna().astype(int)
    broad_valid = numeric(work["BROAD_PURPOSE"]).isin(VALID_BROAD_PURPOSES)
    work["_UNKNOWN_NONHOME_PURPOSE"] = (work["_NONHOME_STOP"].eq(1) & ~broad_valid.fillna(False)).astype(int)
    for code, column in PURPOSE_INDICATORS.items():
        work[column] = (
            work["_NONHOME_STOP"].eq(1) & numeric(work["BROAD_PURPOSE"]).eq(code).fillna(False)
        ).astype(int)

    grouped = work.groupby("TOUR_ID", sort=False)
    aggregate = grouped.agg(
        TOUR_N_TRIPS=("SOURCE_ROW_ID", "size"),
        TOUR_N_NONHOME_STOPS=("_NONHOME_STOP", "sum"),
        TOUR_VALID_DISTANCE_TRIPS=("_DISTANCE_VALID", "sum"),
        TOUR_VALID_TRAVEL_TIME_TRIPS=("_TRAVEL_TIME_VALID", "sum"),
        TOUR_CONTAINS_UNKNOWN_HOME_ORIGIN_FLAG=("_UNKNOWN_HOME_ORIGIN", "max"),
        TOUR_CONTAINS_UNKNOWN_HOME_DESTINATION_FLAG=("_UNKNOWN_HOME_DESTINATION", "max"),
        _ANY_UNKNOWN_NONHOME_PURPOSE=("_UNKNOWN_NONHOME_PURPOSE", "max"),
        **{column: (column, "max") for column in PURPOSE_INDICATORS.values()},
        **{
            spec["feature"]: (spec["nonhome"], spec["aggregation"])
            for spec in DESTINATION_SPATIAL_SPECS.values()
        },
    ).reset_index()
    distance = grouped["TRIP_DISTANCE_KM"].sum(min_count=1).rename("TOUR_DISTANCE_KM").reset_index()
    travel_time = grouped["TRIP_TRAVEL_TIME_MIN"].sum(min_count=1).rename("TOUR_TRAVEL_TIME_MIN").reset_index()
    aggregate = aggregate.merge(distance, on="TOUR_ID", validate="one_to_one")
    aggregate = aggregate.merge(travel_time, on="TOUR_ID", validate="one_to_one")
    aggregate["TOUR_DISTANCE_COMPLETE_FLAG"] = aggregate["TOUR_VALID_DISTANCE_TRIPS"].eq(aggregate["TOUR_N_TRIPS"]).astype(int)
    aggregate["TOUR_TRAVEL_TIME_COMPLETE_FLAG"] = aggregate["TOUR_VALID_TRAVEL_TIME_TRIPS"].eq(aggregate["TOUR_N_TRIPS"]).astype(int)
    aggregate["TOUR_MULTI_STOP_FLAG"] = aggregate["TOUR_N_NONHOME_STOPS"].gt(1).astype(int)
    aggregate["TOUR_N_DISTINCT_PURPOSES"] = aggregate[list(PURPOSE_INDICATORS.values())].sum(axis=1).astype(int)
    aggregate["TOUR_MULTI_PURPOSE_FLAG"] = aggregate["TOUR_N_DISTINCT_PURPOSES"].gt(1).astype(int)
    aggregate["TOUR_HAS_OTHER_PURPOSE"] = (
        work.assign(
            _OTHER=(
                work["_NONHOME_STOP"].eq(1)
                & numeric(work["CANONICAL_PURPOSE"]).eq(10).fillna(False)
            ).astype(int)
        )
        .groupby("TOUR_ID", sort=False)["_OTHER"]
        .max()
        .reindex(aggregate["TOUR_ID"])
        .to_numpy()
    )
    aggregate["TOUR_PURPOSE_UNKNOWN_FLAG"] = (
        aggregate["_ANY_UNKNOWN_NONHOME_PURPOSE"].eq(1) | aggregate["TOUR_N_NONHOME_STOPS"].eq(0)
    ).astype(int)
    aggregate["TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"] = int(membership["BROAD_PURPOSE"].notna().any())

    start = work.loc[work["TOUR_START_TRIP_FLAG"].eq(1)].copy()
    end = work.loc[work["TOUR_END_TRIP_FLAG"].eq(1)].copy()
    start = start.rename(
        columns={
            "SOURCE_ROW_ID": "TOUR_START_SOURCE_ROW_ID",
            "W_ID": "TOUR_START_W_ID",
            "CHAIN_SEQUENCE_NUM": "TOUR_START_CHAIN_SEQUENCE_NUM",
            "_START_MIN": "TOUR_START_MIN",
            "TRIP_DISTANCE_KM": "LEG_DISTANCE_KM",
            "TRIP_TRAVEL_TIME_MIN": "LEG_TRAVEL_TIME_MIN",
            "CANONICAL_PURPOSE": "LEG_PURPOSE",
            "_START_TIME_VALID": "_TOUR_START_TIME_VALID",
        }
    )
    start_columns = [
        "TOUR_ID",
        "PERSON_KEY",
        "H_ID",
        "HP_ID",
        "P_ID",
        "TOUR_START_SOURCE_ROW_ID",
        "TOUR_START_W_ID",
        "TOUR_START_CHAIN_SEQUENCE_NUM",
        "TOUR_START_MIN",
        "LEG_DISTANCE_KM",
        "LEG_TRAVEL_TIME_MIN",
        "LEG_PURPOSE",
        "_TOUR_START_TIME_VALID",
    ]
    end = end.rename(
        columns={
            "SOURCE_ROW_ID": "TOUR_END_SOURCE_ROW_ID",
            "W_ID": "TOUR_END_W_ID",
            "CHAIN_SEQUENCE_NUM": "TOUR_END_CHAIN_SEQUENCE_NUM",
            "_ARRIVAL_MIN": "TOUR_END_ARRIVAL_MIN",
            "_ARRIVAL_TIME_VALID": "_TOUR_END_ARRIVAL_TIME_VALID",
            "_MAIN_PURPOSE": "_TOUR_END_MAIN_PURPOSE",
        }
    )
    end_columns = [
        "TOUR_ID",
        "TOUR_END_SOURCE_ROW_ID",
        "TOUR_END_W_ID",
        "TOUR_END_CHAIN_SEQUENCE_NUM",
        "TOUR_END_ARRIVAL_MIN",
        "_TOUR_END_ARRIVAL_TIME_VALID",
        "_TOUR_END_MAIN_PURPOSE",
    ]

    tours = tour_meta.drop(columns="_TOUR_INDEX").merge(start[start_columns], on=["TOUR_ID", "PERSON_KEY"], validate="one_to_one")
    tours = tours.merge(end[end_columns], on="TOUR_ID", validate="one_to_one")
    tours = tours.merge(aggregate, on="TOUR_ID", validate="one_to_one")
    tours["TOUR_FIRST_PURPOSE"] = tours["LEG_PURPOSE"].astype("Int64")

    elapsed_valid = (
        tours["TOUR_CLOSED"].eq(1)
        & tours["_TOUR_START_TIME_VALID"].eq(1)
        & tours["_TOUR_END_ARRIVAL_TIME_VALID"].eq(1)
        & numeric(tours["TOUR_END_ARRIVAL_MIN"]).ge(numeric(tours["TOUR_START_MIN"]))
    )
    tours["TOUR_ELAPSED_TIME_MIN"] = (
        numeric(tours["TOUR_END_ARRIVAL_MIN"]) - numeric(tours["TOUR_START_MIN"])
    ).where(elapsed_valid)
    tours["TOUR_ELAPSED_TIME_VALID_FLAG"] = elapsed_valid.astype(int)

    main_valid = (
        tours["TOUR_CLOSED"].eq(1)
        & numeric(tours["_TOUR_END_MAIN_PURPOSE"]).isin(VALID_BROAD_PURPOSES)
    )
    tours["TOUR_MAIN_PURPOSE"] = numeric(tours["_TOUR_END_MAIN_PURPOSE"]).where(main_valid).round().astype("Int64")
    tours["TOUR_MAIN_PURPOSE_DERIVABLE_FLAG"] = main_valid.astype(int)

    columns = [
        "TOUR_ID",
        "TOUR_SEQUENCE_NUM",
        "PERSON_KEY",
        "H_ID",
        "HP_ID",
        "P_ID",
        "TOUR_START_SOURCE_ROW_ID",
        "TOUR_START_W_ID",
        "TOUR_START_CHAIN_SEQUENCE_NUM",
        "TOUR_END_SOURCE_ROW_ID",
        "TOUR_END_W_ID",
        "TOUR_END_CHAIN_SEQUENCE_NUM",
        "TOUR_START_MIN",
        "TOUR_END_ARRIVAL_MIN",
        "TOUR_CLOSED",
        "TOUR_OPEN_END_FLAG",
        "TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG",
        "TOUR_CONTAINS_UNKNOWN_HOME_ORIGIN_FLAG",
        "TOUR_CONTAINS_UNKNOWN_HOME_DESTINATION_FLAG",
        "LEG_DISTANCE_KM",
        "LEG_TRAVEL_TIME_MIN",
        "LEG_PURPOSE",
        "TOUR_DISTANCE_KM",
        "TOUR_TRAVEL_TIME_MIN",
        "TOUR_ELAPSED_TIME_MIN",
        "TOUR_N_TRIPS",
        "TOUR_N_NONHOME_STOPS",
        *[spec["feature"] for spec in DESTINATION_SPATIAL_SPECS.values()],
        "TOUR_MULTI_STOP_FLAG",
        "TOUR_FIRST_PURPOSE",
        "TOUR_MAIN_PURPOSE",
        *PURPOSE_INDICATORS.values(),
        "TOUR_HAS_OTHER_PURPOSE",
        "TOUR_N_DISTINCT_PURPOSES",
        "TOUR_MULTI_PURPOSE_FLAG",
        "TOUR_DISTANCE_COMPLETE_FLAG",
        "TOUR_TRAVEL_TIME_COMPLETE_FLAG",
        "TOUR_ELAPSED_TIME_VALID_FLAG",
        "TOUR_PURPOSE_UNKNOWN_FLAG",
        "TOUR_MAIN_PURPOSE_DERIVABLE_FLAG",
        "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG",
    ]
    return tours[columns]


def select_membership_output(membership: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "TOUR_ID",
        "TOUR_SEQUENCE_NUM",
        "PERSON_KEY",
        "H_ID",
        "HP_ID",
        "P_ID",
        "SOURCE_ROW_ID",
        "W_ID",
        "CHAIN_SEQUENCE_NUM",
        "CHAIN_ELIGIBLE",
        "W_RBW",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "CANONICAL_PURPOSE",
        "BROAD_PURPOSE",
        "TRIP_DISTANCE_KM",
        "TRIP_TRAVEL_TIME_MIN",
        "START_MIN",
        "ARRIVAL_MIN",
        *( ["pkw_fmf"] if "pkw_fmf" in membership.columns else [] ),
        "TOUR_START_TRIP_FLAG",
        "TOUR_END_TRIP_FLAG",
        "TOUR_CLOSING_TRIP_FLAG",
        "TOUR_CLOSED",
        "TOUR_OPEN_END_FLAG",
        "TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG",
    ]
    return membership[columns].copy()


def assert_tour_integrity(prepared: pd.DataFrame, membership: pd.DataFrame, tours: pd.DataFrame) -> None:
    eligible = prepared.loc[prepared["_CHAIN_ELIGIBLE"].eq(1)]
    if not membership["_CHAIN_ELIGIBLE"].eq(1).all():
        raise AssertionError("Only CHAIN_ELIGIBLE == 1 rows may enter tour membership.")
    if membership["_W_RBW"].eq(1).any():
        raise AssertionError("rbW rows entered tour membership.")
    if not membership["SOURCE_ROW_ID"].is_unique:
        raise AssertionError("A chain-eligible trip was assigned to more than one tour.")
    if not tours["TOUR_ID"].is_unique:
        raise AssertionError("TOUR_ID is not unique.")
    if set(membership["SOURCE_ROW_ID"]) - set(eligible["SOURCE_ROW_ID"]):
        raise AssertionError("Tour membership contains rows outside the accepted chain-eligible source.")

    start_counts = membership.groupby("TOUR_ID", sort=False)["TOUR_START_TRIP_FLAG"].sum()
    if not start_counts.eq(1).all():
        raise AssertionError("Every tour must have exactly one start trip.")
    starts = membership.loc[membership["TOUR_START_TRIP_FLAG"].eq(1)]
    if not starts["_HOME_ORIGIN"].eq(1).all():
        raise AssertionError("Every tour start must have HOME_ORIGIN == 1.")

    ordered = tours.sort_values(["PERSON_KEY", "TOUR_SEQUENCE_NUM"], kind="mergesort")
    sequence_ok = ordered.groupby("PERSON_KEY", sort=False)["TOUR_SEQUENCE_NUM"].apply(
        lambda values: list(values.astype(int)) == list(range(1, len(values) + 1))
    )
    if not sequence_ok.all():
        raise AssertionError("Tour sequence numbers are not deterministic consecutive integers within person.")

    closed_membership = membership.loc[membership["TOUR_CLOSED"].eq(1)]
    closed_end = closed_membership.loc[closed_membership["TOUR_CLOSING_TRIP_FLAG"].eq(1)]
    if not closed_end["_HOME_DESTINATION"].eq(1).all():
        raise AssertionError("A closed tour lacks an explicit HOME_DESTINATION == 1 closing trip.")
    earlier_home = closed_membership["_HOME_DESTINATION"].eq(1) & closed_membership["TOUR_CLOSING_TRIP_FLAG"].eq(0)
    if earlier_home.any():
        raise AssertionError("A closed tour contains an earlier return-home trip before its closing record.")

    open_tours = tours.loc[tours["TOUR_CLOSED"].eq(0)]
    if open_tours["TOUR_ELAPSED_TIME_MIN"].notna().any():
        raise AssertionError("Open or inferred-boundary tours must not receive fabricated elapsed duration.")
    if tours.loc[tours["TOUR_ELAPSED_TIME_VALID_FLAG"].eq(1), "TOUR_CLOSED"].ne(1).any():
        raise AssertionError("Elapsed duration may only be valid for closed tours.")

    comparable_distance = tours["LEG_DISTANCE_KM"].notna() & tours["TOUR_DISTANCE_KM"].notna()
    bad_distance = comparable_distance & (tours["TOUR_DISTANCE_KM"] + 1e-9 < tours["LEG_DISTANCE_KM"])
    comparable_time = tours["LEG_TRAVEL_TIME_MIN"].notna() & tours["TOUR_TRAVEL_TIME_MIN"].notna()
    bad_time = comparable_time & (tours["TOUR_TRAVEL_TIME_MIN"] + 1e-9 < tours["LEG_TRAVEL_TIME_MIN"])
    if bad_distance.any() or bad_time.any():
        raise AssertionError("Tour totals are smaller than their immediate-leg component.")
    if tours["TOUR_N_TRIPS"].lt(1).any():
        raise AssertionError("Every reconstructed tour must contain at least one observed trip.")

    nonhome = membership["_HOME_DESTINATION"].eq(0).fillna(False)
    indexed_tours = tours.set_index("TOUR_ID")
    for spec in DESTINATION_SPATIAL_SPECS.values():
        expected = (
            membership[spec["clean"]]
            .where(nonhome)
            .groupby(membership["TOUR_ID"], sort=False)
            .agg(spec["aggregation"])
            .reindex(tours["TOUR_ID"])
            .reset_index(drop=True)
        )
        actual = indexed_tours[spec["feature"]].reindex(tours["TOUR_ID"]).reset_index(drop=True)
        try:
            pd.testing.assert_series_equal(expected, actual, check_names=False, check_dtype=False)
        except AssertionError as error:
            raise AssertionError(f"Incorrect worst-observed aggregation for {spec['feature']}.") from error


def merge_to_strict_wide(
    wide: pd.DataFrame,
    prepared: pd.DataFrame,
    tours: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(wide, ORIGINAL_INVARIANT_COLUMNS, "strict wide")
    if not wide["CHOICE_ID"].is_unique or ~nonblank(wide["CHOICE_ID"]).all():
        raise ValueError("Input strict wide CHOICE_ID must be non-missing and unique.")
    if not wide["SOURCE_ROW_ID"].is_unique or ~nonblank(wide["SOURCE_ROW_ID"]).all():
        raise ValueError("Input strict wide SOURCE_ROW_ID must be non-missing and unique.")

    source_check = wide[["CHOICE_ID", "SOURCE_ROW_ID"]].merge(
        prepared[["SOURCE_ROW_ID", "CHAIN_ELIGIBLE", "HOME_ORIGIN"]],
        on="SOURCE_ROW_ID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    structural_bad = (
        source_check["_merge"].ne("both")
        | numeric(source_check["CHAIN_ELIGIBLE"]).ne(1)
        | numeric(source_check["HOME_ORIGIN"]).ne(1)
    )
    if structural_bad.any():
        print("\nSTRUCTURAL STRICT-TO-PHASE-1 CONTRADICTIONS")
        print(source_check.loc[structural_bad].head(20).to_string(index=False))
        raise ValueError("A strict wide source trip is missing, not chain eligible, or not HOME_ORIGIN == 1.")

    merge_columns = [column for column in tours.columns if column not in {"H_ID", "HP_ID", "P_ID"}]
    if not tours["TOUR_START_SOURCE_ROW_ID"].is_unique:
        raise ValueError("TOUR_START_SOURCE_ROW_ID must be unique before the strict merge.")
    original = wide.copy()
    left = wide.copy()
    left["_STRICT_ROW_ORDER"] = np.arange(len(left), dtype=np.int64)
    enriched = left.merge(
        tours[merge_columns],
        left_on="SOURCE_ROW_ID",
        right_on="TOUR_START_SOURCE_ROW_ID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    enriched["TOUR_MATCH_FOUND"] = enriched["_merge"].eq("both").astype(int)
    enriched = enriched.sort_values("_STRICT_ROW_ORDER", kind="mergesort").drop(columns=["_STRICT_ROW_ORDER", "_merge"]).reset_index(drop=True)

    if len(enriched) != len(original):
        raise AssertionError("Strict wide row count changed during tour-feature merge.")
    if not enriched["CHOICE_ID"].is_unique or enriched["CHOICE_ID"].nunique() != original["CHOICE_ID"].nunique():
        raise AssertionError("Strict CHOICE_ID uniqueness changed during tour-feature merge.")
    for column in original.columns:
        if not enriched[column].equals(original[column]):
            raise AssertionError(f"Canonical strict wide column changed during merge: {column}")
    matched = enriched["TOUR_MATCH_FOUND"].eq(1)
    if not enriched.loc[matched, "SOURCE_ROW_ID"].equals(enriched.loc[matched, "TOUR_START_SOURCE_ROW_ID"]):
        raise AssertionError("Matched SOURCE_ROW_ID differs from TOUR_START_SOURCE_ROW_ID.")
    unmatched = enriched.loc[~matched, ["CHOICE_ID", "SOURCE_ROW_ID"]]
    if not unmatched.empty:
        print("\nSTRICT ROWS WITHOUT A RECONSTRUCTED TOUR (retained with TOUR_MATCH_FOUND=0)")
        print(unmatched.head(20).to_string(index=False))
    return enriched


def qa_row(section: str, metric: str, count: float | int, denominator: int | None, description: str) -> dict[str, object]:
    share: float | str = "" if denominator is None else (float(count) / denominator if denominator else 0.0)
    return {
        "section": section,
        "metric": metric,
        "count": count,
        "share": share,
        "denominator_description": description,
    }


def destination_spatial_tour_diagnostics(
    membership: pd.DataFrame,
    tours: pd.DataFrame,
) -> pd.DataFrame:
    """Derive destination coverage from accepted member trips without changing their order."""
    ordered = membership.sort_values(
        ["PERSON_KEY", "_CHAIN_SEQUENCE", "_W_ID_SORT", "_SOURCE_ROW_NUM", "SOURCE_ROW_ID"],
        kind="mergesort",
        na_position="last",
    ).copy()
    ordered["_NONHOME_STOP"] = ordered["_HOME_DESTINATION"].eq(0).fillna(False).astype(int)
    nonhome = ordered.loc[ordered["_NONHOME_STOP"].eq(1)].copy()

    diagnostics = tours[["TOUR_ID"]].copy()
    tour_ids = diagnostics["TOUR_ID"]
    observed = nonhome.groupby("TOUR_ID", sort=False).size()
    diagnostics["_N_NONHOME_DESTINATIONS"] = tour_ids.map(observed).fillna(0).astype(int)

    first_nonhome = nonhome.drop_duplicates("TOUR_ID", keep="first").set_index("TOUR_ID")
    for source, spec in DESTINATION_SPATIAL_SPECS.items():
        prefix = f"_{source.upper()}"
        valid_counts = nonhome[spec["clean"]].notna().groupby(nonhome["TOUR_ID"], sort=False).sum()
        diagnostics[f"{prefix}_N_VALID"] = tour_ids.map(valid_counts).fillna(0).astype(int)
        diagnostics[f"{prefix}_ZERO_NONHOME"] = diagnostics["_N_NONHOME_DESTINATIONS"].eq(0)
        diagnostics[f"{prefix}_AT_LEAST_VALID"] = diagnostics[f"{prefix}_N_VALID"].gt(0)
        diagnostics[f"{prefix}_ALL_VALID"] = (
            diagnostics["_N_NONHOME_DESTINATIONS"].gt(0)
            & diagnostics[f"{prefix}_N_VALID"].eq(diagnostics["_N_NONHOME_DESTINATIONS"])
        )
        diagnostics[f"{prefix}_PARTIAL"] = (
            diagnostics[f"{prefix}_N_VALID"].gt(0)
            & diagnostics[f"{prefix}_N_VALID"].lt(diagnostics["_N_NONHOME_DESTINATIONS"])
        )
        diagnostics[f"{prefix}_ALL_MISSING"] = (
            diagnostics["_N_NONHOME_DESTINATIONS"].gt(0)
            & diagnostics[f"{prefix}_N_VALID"].eq(0)
        )
        first_valid = first_nonhome[spec["clean"]].notna()
        diagnostics[f"{prefix}_FIRST_VALID"] = tour_ids.map(first_valid).eq(True)
        diagnostics[f"{prefix}_FIRST_MISSING"] = (
            diagnostics["_N_NONHOME_DESTINATIONS"].gt(0)
            & ~diagnostics[f"{prefix}_FIRST_VALID"]
        )
        diagnostics[f"{prefix}_RESCUED"] = (
            diagnostics[f"{prefix}_FIRST_MISSING"]
            & diagnostics[f"{prefix}_AT_LEAST_VALID"]
        )

    return diagnostics


def build_qa_summary(
    prepared: pd.DataFrame,
    membership: pd.DataFrame,
    tours: pd.DataFrame,
    wide_input: pd.DataFrame,
    wide_output: pd.DataFrame,
    diagnostics: ReconstructionDiagnostics,
) -> pd.DataFrame:
    eligible = prepared.loc[prepared["_CHAIN_ELIGIBLE"].eq(1)]
    matched = wide_output.loc[wide_output["TOUR_MATCH_FOUND"].eq(1)]
    tour_den = len(tours)
    strict_den = len(wide_output)
    rows: list[dict[str, object]] = []

    rows.extend(
        [
            qa_row("INPUT", "Phase 1 retained rows", len(prepared), len(prepared), "all accepted Phase 1 retained rows"),
            qa_row("INPUT", "chain-eligible rows", len(eligible), len(prepared), "all accepted Phase 1 retained rows"),
            qa_row("INPUT", "unique chain-eligible persons", eligible["PERSON_KEY"].nunique(), eligible["PERSON_KEY"].nunique(), "persons with chain-eligible trips"),
            qa_row("INPUT", "strict MNL wide rows", len(wide_input), len(wide_input), "accepted strict wide rows"),
            qa_row("INPUT", "documented W_ZWECK/zweck detailed refinements", diagnostics.expected_w_zweck_zweck_refinements, len(eligible), "chain-eligible trips"),
        ]
    )
    rows.extend(
        [
            qa_row("TOUR RECONSTRUCTION", "home-based tours", tour_den, tour_den, "reconstructed anchored tours"),
            qa_row("TOUR RECONSTRUCTION", "persons with at least one reconstructed tour", tours["PERSON_KEY"].nunique(), eligible["PERSON_KEY"].nunique(), "chain-eligible persons"),
            qa_row("TOUR RECONSTRUCTION", "closed tours", int(tours["TOUR_CLOSED"].sum()), tour_den, "reconstructed tours"),
            qa_row("TOUR RECONSTRUCTION", "open/end-censored tours", int(tours["TOUR_OPEN_END_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("TOUR RECONSTRUCTION", "tours closed through explicit HOME_DESTINATION", int(tours["TOUR_CLOSED"].sum()), tour_den, "reconstructed tours"),
            qa_row("TOUR RECONSTRUCTION", "tours terminated by next HOME_ORIGIN", int(tours["TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("TOUR RECONSTRUCTION", "unanchored trips before first home departure", diagnostics.unanchored_before_first_home_departure, len(eligible), "chain-eligible trips"),
            qa_row("TOUR RECONSTRUCTION", "unanchored trips after first home departure outside active tour", diagnostics.unanchored_after_first_home_departure, len(eligible), "chain-eligible trips"),
            qa_row("TOUR RECONSTRUCTION", "tours containing unknown home-origin status", int(tours["TOUR_CONTAINS_UNKNOWN_HOME_ORIGIN_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("TOUR RECONSTRUCTION", "tours containing unknown home-destination status", int(tours["TOUR_CONTAINS_UNKNOWN_HOME_DESTINATION_FLAG"].sum()), tour_den, "reconstructed tours"),
        ]
    )
    for label, mask in [
        ("1-trip tours", tours["TOUR_N_TRIPS"].eq(1)),
        ("1-trip closed explicit home-to-home tours", tours["TOUR_N_TRIPS"].eq(1) & tours["TOUR_CLOSED"].eq(1)),
        ("1-trip open/end-censored tours", tours["TOUR_N_TRIPS"].eq(1) & tours["TOUR_OPEN_END_FLAG"].eq(1)),
        ("2-trip tours", tours["TOUR_N_TRIPS"].eq(2)),
        ("3-trip tours", tours["TOUR_N_TRIPS"].eq(3)),
        ("4+ trip tours", tours["TOUR_N_TRIPS"].ge(4)),
    ]:
        rows.append(qa_row("TOUR SIZE", label, int(mask.sum()), tour_den, "reconstructed tours"))
    rows.extend(
        [
            qa_row("TOUR SIZE", "mean observed trips per tour", float(tours["TOUR_N_TRIPS"].mean()), None, "reconstructed tours"),
            qa_row("TOUR SIZE", "median observed trips per tour", float(tours["TOUR_N_TRIPS"].median()), None, "reconstructed tours"),
            qa_row("TOUR SIZE", "mean non-home stops", float(tours["TOUR_N_NONHOME_STOPS"].mean()), None, "reconstructed tours"),
            qa_row("TOUR SIZE", "median non-home stops", float(tours["TOUR_N_NONHOME_STOPS"].median()), None, "reconstructed tours"),
        ]
    )
    rows.extend(
        [
            qa_row("ATTRIBUTE COMPLETENESS", "complete distance tours", int(tours["TOUR_DISTANCE_COMPLETE_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "incomplete distance tours", int(tours["TOUR_DISTANCE_COMPLETE_FLAG"].eq(0).sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "complete travel-time tours", int(tours["TOUR_TRAVEL_TIME_COMPLETE_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "incomplete travel-time tours", int(tours["TOUR_TRAVEL_TIME_COMPLETE_FLAG"].eq(0).sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "valid elapsed-time closed tours", int(tours["TOUR_ELAPSED_TIME_VALID_FLAG"].sum()), int(tours["TOUR_CLOSED"].sum()), "closed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "tours with unknown purpose information", int(tours["TOUR_PURPOSE_UNKNOWN_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "tours where official main purpose was derivable", int(tours["TOUR_MAIN_PURPOSE_DERIVABLE_FLAG"].sum()), tour_den, "reconstructed tours"),
            qa_row("ATTRIBUTE COMPLETENESS", "tours where main purpose was not derivable", int(tours["TOUR_MAIN_PURPOSE_DERIVABLE_FLAG"].eq(0).sum()), tour_den, "reconstructed tours"),
        ]
    )
    rows.extend(
        [
            qa_row("STRICT MNL MERGE", "strict input rows", len(wide_input), strict_den, "accepted strict wide rows"),
            qa_row("STRICT MNL MERGE", "matched strict rows", len(matched), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "unmatched strict rows", int(wide_output["TOUR_MATCH_FOUND"].eq(0).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict closed-tour rows", int(matched["TOUR_CLOSED"].eq(1).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict open-tour rows", int(matched["TOUR_OPEN_END_FLAG"].eq(1).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict inferred-boundary rows", int(matched["TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG"].eq(1).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict rows with complete tour distance", int(matched["TOUR_DISTANCE_COMPLETE_FLAG"].eq(1).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict rows with complete tour travel time", int(matched["TOUR_TRAVEL_TIME_COMPLETE_FLAG"].eq(1).sum()), strict_den, "strict output rows"),
            qa_row("STRICT MNL MERGE", "strict rows with valid elapsed time", int(matched["TOUR_ELAPSED_TIME_VALID_FLAG"].eq(1).sum()), strict_den, "strict output rows"),
        ]
    )

    readiness = [
        ("rows with valid immediate-leg distance", matched["LEG_DISTANCE_KM"].notna()),
        ("rows with valid tour distance", matched["TOUR_DISTANCE_KM"].notna()),
        ("rows with both leg and tour distance", matched["LEG_DISTANCE_KM"].notna() & matched["TOUR_DISTANCE_KM"].notna()),
        ("rows with valid immediate-leg travel time", matched["LEG_TRAVEL_TIME_MIN"].notna()),
        ("rows with valid tour travel time", matched["TOUR_TRAVEL_TIME_MIN"].notna()),
        ("rows with both leg and tour travel time", matched["LEG_TRAVEL_TIME_MIN"].notna() & matched["TOUR_TRAVEL_TIME_MIN"].notna()),
        ("rows with first purpose", matched["TOUR_FIRST_PURPOSE"].notna()),
        ("rows with broader tour-purpose indicators", matched["TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG"].eq(1)),
    ]
    rows.extend(qa_row("MODEL-COMPARISON READINESS", label, int(mask.sum()), strict_den, "strict output rows") for label, mask in readiness)

    destination_rows: list[dict[str, object]] = []
    destination_diagnostics = destination_spatial_tour_diagnostics(membership, tours)
    strict_destination = matched[["CHOICE_ID", "TOUR_ID"]].merge(
        destination_diagnostics,
        on="TOUR_ID",
        how="left",
        validate="one_to_one",
    )
    if len(strict_destination) != len(matched) or strict_destination["_N_NONHOME_DESTINATIONS"].isna().any():
        raise AssertionError("Destination-spatial QA did not preserve all strict matched rows.")

    matched_den = len(strict_destination)
    for source, spec in DESTINATION_SPATIAL_SPECS.items():
        prefix = f"_{source.upper()}"
        with_nonhome = strict_destination["_N_NONHOME_DESTINATIONS"].gt(0)
        first_missing = strict_destination[f"{prefix}_FIRST_MISSING"]
        rescued = strict_destination[f"{prefix}_RESCUED"]
        feature_valid = strict_destination[f"{prefix}_AT_LEAST_VALID"]
        nonhome_den = int(with_nonhome.sum())
        first_missing_den = int(first_missing.sum())
        feature = spec["feature"]
        destination_rows.extend(
            [
                qa_row("DESTINATION SPATIAL QA", f"{source}: strict matched tours/rows", matched_den, matched_den, "strict rows with TOUR_MATCH_FOUND == 1"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: strict rows with zero non-home destinations", int((~with_nonhome).sum()), matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: strict rows with at least one non-home destination", nonhome_den, matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: first non-home destination valid", int(strict_destination[f"{prefix}_FIRST_VALID"].sum()), nonhome_den, "strict matched rows with at least one non-home destination"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: first non-home destination missing", first_missing_den, nonhome_den, "strict matched rows with at least one non-home destination"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: {feature} valid", int(feature_valid.sum()), matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: {feature} missing", int((~feature_valid).sum()), matched_den, "strict matched rows, including zero-non-home-destination rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: rescued by tour aggregation", int(rescued.sum()), matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: rescued among first-destination-missing", int(rescued.sum()), first_missing_den, "strict matched rows whose first non-home destination is missing"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: all non-home destinations missing", int(strict_destination[f"{prefix}_ALL_MISSING"].sum()), matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: complete destination coverage", int(strict_destination[f"{prefix}_ALL_VALID"].sum()), matched_den, "strict matched rows"),
                qa_row("DESTINATION SPATIAL QA", f"{source}: partial destination coverage", int(strict_destination[f"{prefix}_PARTIAL"].sum()), matched_den, "strict matched rows"),
            ]
        )

    for metric, column in [
        ("mean observed tour distance km", "TOUR_DISTANCE_KM"),
        ("median observed tour distance km", "TOUR_DISTANCE_KM"),
        ("mean observed tour travel time min", "TOUR_TRAVEL_TIME_MIN"),
        ("median observed tour travel time min", "TOUR_TRAVEL_TIME_MIN"),
        ("mean valid closed-tour elapsed time min", "TOUR_ELAPSED_TIME_MIN"),
        ("median valid closed-tour elapsed time min", "TOUR_ELAPSED_TIME_MIN"),
    ]:
        value = float(tours[column].mean()) if metric.startswith("mean") else float(tours[column].median())
        rows.append(qa_row("DESCRIPTIVE TOUR ATTRIBUTES", metric, value, None, "tours with a valid observed value"))
    rows.extend(destination_rows)
    # Object dtype keeps true counts as integers while allowing descriptive
    # means/medians to share the requested compact `count` field.
    return pd.DataFrame(rows, dtype=object)


def print_tour_examples(membership: pd.DataFrame, tours: pd.DataFrame, wide_output: pd.DataFrame) -> None:
    # MiD pkw_fmf == 706 is the documented "trip without car use" code.
    noncar_tours = set(membership.loc[membership["_PKW_FMF"].eq(706), "TOUR_ID"])
    strict_tours = set(wide_output.loc[wide_output["TOUR_MATCH_FOUND"].eq(1), "TOUR_ID"].astype(str))
    selectors = [
        ("normal HOME -> activity -> HOME", tours["TOUR_CLOSED"].eq(1) & tours["TOUR_N_TRIPS"].eq(2) & tours["TOUR_N_NONHOME_STOPS"].eq(1)),
        ("multi-stop tour", tours["TOUR_CLOSED"].eq(1) & tours["TOUR_N_NONHOME_STOPS"].ge(2)),
        ("multi-purpose tour", tours["TOUR_N_DISTINCT_PURPOSES"].ge(3) if tours["TOUR_N_DISTINCT_PURPOSES"].ge(3).any() else tours["TOUR_MULTI_PURPOSE_FLAG"].eq(1)),
        ("tour containing non-car trips", tours["TOUR_ID"].isin(noncar_tours)),
        ("open/end-censored tour", tours["TOUR_OPEN_END_FLAG"].eq(1)),
        ("tour terminated by next HOME_ORIGIN", tours["TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG"].eq(1)),
        ("tour containing unknown home status", tours["TOUR_CONTAINS_UNKNOWN_HOME_ORIGIN_FLAG"].eq(1) | tours["TOUR_CONTAINS_UNKNOWN_HOME_DESTINATION_FLAG"].eq(1)),
        ("strict MNL occasion matched by SOURCE_ROW_ID", tours["TOUR_ID"].isin(strict_tours)),
    ]
    display_columns = [
        "PERSON_KEY",
        "TOUR_ID",
        "SOURCE_ROW_ID",
        "W_ID",
        "CHAIN_SEQUENCE_NUM",
        "HOME_ORIGIN",
        "HOME_DESTINATION",
        "START_MIN",
        "ARRIVAL_MIN",
        "CANONICAL_PURPOSE",
        "BROAD_PURPOSE",
        "TRIP_DISTANCE_KM",
        "TRIP_TRAVEL_TIME_MIN",
        *( ["pkw_fmf"] if "pkw_fmf" in membership.columns else [] ),
        "TOUR_START_TRIP_FLAG",
        "TOUR_END_TRIP_FLAG",
        "TOUR_CLOSING_TRIP_FLAG",
    ]
    chosen: set[str] = set()
    print("\nMANUAL TOUR INSPECTION EXAMPLES")
    for label, mask in selectors:
        candidates = tours.loc[mask & ~tours["TOUR_ID"].isin(chosen), "TOUR_ID"]
        if candidates.empty:
            candidates = tours.loc[mask, "TOUR_ID"]
        if candidates.empty:
            print(f"\n{label}: no example available")
            continue
        tour_id = str(candidates.iloc[0])
        chosen.add(tour_id)
        chain = membership.loc[membership["TOUR_ID"].eq(tour_id), display_columns]
        print(f"\n--- {label}: {tour_id} ---")
        with pd.option_context("display.max_columns", None, "display.max_rows", 20, "display.width", 240):
            print(chain.to_string(index=False))


def print_final_summary(
    prepared: pd.DataFrame,
    tours: pd.DataFrame,
    wide_input: pd.DataFrame,
    wide_output: pd.DataFrame,
    qa: pd.DataFrame,
) -> None:
    eligible = prepared.loc[prepared["_CHAIN_ELIGIBLE"].eq(1)]
    matched = wide_output["TOUR_MATCH_FOUND"].eq(1)
    print("\nHOME-BASED TOUR FEATURE ENGINEERING")
    print(f"\nPhase 1 chain-eligible trips: {len(eligible):,}")
    print(f"persons: {eligible['PERSON_KEY'].nunique():,}")
    print(f"\nhome-based tours: {len(tours):,}")
    print(f"closed tours: {int(tours['TOUR_CLOSED'].sum()):,}")
    print(f"open tours: {int(tours['TOUR_OPEN_END_FLAG'].sum()):,}")
    print(f"inferred-boundary tours: {int(tours['TOUR_BOUNDARY_INFERRED_FROM_NEXT_HOME_DEPARTURE_FLAG'].sum()):,}")
    print(f"\nstrict MNL input rows: {len(wide_input):,}")
    print(f"strict rows matched to tours: {int(matched.sum()):,}")
    print(f"strict rows unmatched: {int((~matched).sum()):,}")
    print(f"match rate: {matched.mean():.6%}")
    print(f"\nstrict closed-tour rows: {int(wide_output.loc[matched, 'TOUR_CLOSED'].eq(1).sum()):,}")
    print(f"strict open-tour rows: {int(wide_output.loc[matched, 'TOUR_OPEN_END_FLAG'].eq(1).sum()):,}")
    print(f"\ndistance complete: {int(wide_output.loc[matched, 'TOUR_DISTANCE_COMPLETE_FLAG'].eq(1).sum()):,}")
    print(f"travel-time complete: {int(wide_output.loc[matched, 'TOUR_TRAVEL_TIME_COMPLETE_FLAG'].eq(1).sum()):,}")
    print(f"elapsed-time valid: {int(wide_output.loc[matched, 'TOUR_ELAPSED_TIME_VALID_FLAG'].eq(1).sum()):,}")
    print(f"\ninput wide rows: {len(wide_input):,}")
    print(f"output wide rows: {len(wide_output):,}")
    print(f"unique input CHOICE_ID: {wide_input['CHOICE_ID'].nunique():,}")
    print(f"unique output CHOICE_ID: {wide_output['CHOICE_ID'].nunique():,}")
    print("\nDESTINATION SPATIAL QA")
    spatial_qa = qa.loc[qa["section"].eq("DESTINATION SPATIAL QA")].set_index("metric")
    for source, spec in DESTINATION_SPATIAL_SPECS.items():
        first_missing = spatial_qa.loc[f"{source}: first non-home destination missing"]
        worst_missing = spatial_qa.loc[f"{source}: {spec['feature']} missing"]
        rescued = spatial_qa.loc[f"{source}: rescued by tour aggregation"]
        rescued_among_missing = spatial_qa.loc[f"{source}: rescued among first-destination-missing"]
        print(f"{source}:")
        print(f"  first-destination missing share: {float(first_missing['share']):.6%}")
        print(f"  worst-observed missing share: {float(worst_missing['share']):.6%}")
        print(f"  rescued using later observed tour destinations: {int(rescued['count']):,}")
        print(f"  rescued among first-destination-missing: {float(rescued_among_missing['share']):.6%}")
    print("\nOUTPUTS:")
    for path in [MEMBERSHIP_PATH, TOUR_FEATURES_PATH, WIDE_FEATURES_PATH, QA_PATH]:
        print(path)
    print("\nAccepted Phase 1–3 reconstruction and strict sample logic were not modified.")


def main() -> None:
    print("HOME-BASED TOUR FEATURE PATHS")
    print(f"Phase 1 trip-chain input: {TRIPS_PATH.resolve()}")
    print(f"Phase 3 strict wide input: {WIDE_BASE_PATH.resolve()}")
    print(f"tour-trip membership output: {MEMBERSHIP_PATH.resolve()}")
    print(f"tour features output: {TOUR_FEATURES_PATH.resolve()}")
    print(f"feature-enriched strict wide output: {WIDE_FEATURES_PATH.resolve()}")
    print(f"QA summary output: {QA_PATH.resolve()}")

    trips_raw = read_csv_strings(TRIPS_PATH)
    wide_input = read_csv_strings(WIDE_BASE_PATH)
    prepared, initial_diagnostics = canonicalize_trip_attributes(trips_raw)
    membership, tour_meta, diagnostics = reconstruct_home_based_tours(prepared, initial_diagnostics)
    reconstructed_tour_ids = tour_meta["TOUR_ID"].copy()
    reconstructed_membership_source_ids = membership["SOURCE_ROW_ID"].copy()
    tours = build_tour_features(membership, tour_meta)

    assert_tour_integrity(prepared, membership, tours)
    membership_output = select_membership_output(membership)
    wide_output = merge_to_strict_wide(wide_input, prepared, tours)
    qa = build_qa_summary(prepared, membership, tours, wide_input, wide_output, diagnostics)

    # Validate the strict sample and all derived tables before creating any output.
    if len(wide_output) != len(wide_input) or wide_output["CHOICE_ID"].nunique() != wide_input["CHOICE_ID"].nunique():
        raise AssertionError("Accepted strict sample size changed before output writing.")
    if not membership_output["SOURCE_ROW_ID"].is_unique:
        raise AssertionError("Trip-to-tour membership output contains duplicate SOURCE_ROW_ID values.")
    if len(tours) != tours["TOUR_ID"].nunique():
        raise AssertionError("Tour feature output does not contain one row per TOUR_ID.")
    if len(tours) != len(tour_meta) or set(tours["TOUR_ID"]) != set(reconstructed_tour_ids):
        raise AssertionError("Destination-spatial feature construction changed the reconstructed tour universe.")
    if not membership_output["SOURCE_ROW_ID"].equals(reconstructed_membership_source_ids.reset_index(drop=True)):
        raise AssertionError("Destination-spatial feature construction changed the membership SOURCE_ROW_ID universe or order.")
    if not wide_output[["CHOICE_ID", "SOURCE_ROW_ID"]].equals(wide_input[["CHOICE_ID", "SOURCE_ROW_ID"]]):
        raise AssertionError("Destination-spatial feature construction changed strict SOURCE_ROW_ID matching or row order.")
    if any(spec["clean"] in membership_output.columns for spec in DESTINATION_SPATIAL_SPECS.values()):
        raise AssertionError("Internal destination-spatial working columns leaked into membership output.")

    single_trip = tours.loc[tours["TOUR_N_TRIPS"].eq(1)]
    single_closed = single_trip["TOUR_CLOSED"].eq(1)
    single_open = single_trip["TOUR_OPEN_END_FLAG"].eq(1)
    print(f"\nSingle-trip observed tours requiring later sensitivity review: {len(single_trip):,}")
    print(f"  closed explicit home-to-home records: {int(single_closed.sum()):,}")
    print(f"  open/end-censored records: {int(single_open.sum()):,}")
    print(f"Documented W_ZWECK/zweck detailed refinements retained via W_ZWECK: {diagnostics.expected_w_zweck_zweck_refinements:,}")
    print(f"Broad purpose source: {diagnostics.broad_purpose_source or 'unavailable'}")
    print(
        "TOUR_MAIN_PURPOSE: closing-trip hwzweck2 for closed tours only"
        if diagnostics.official_main_purpose_available
        else "TOUR_MAIN_PURPOSE: deferred; no validated hwzweck2 source was available"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    membership_output.to_csv(MEMBERSHIP_PATH, index=False)
    tours.to_csv(TOUR_FEATURES_PATH, index=False)
    wide_output.to_csv(WIDE_FEATURES_PATH, index=False)
    qa.to_csv(QA_PATH, index=False)

    print_tour_examples(membership, tours, wide_output)
    print_final_summary(prepared, tours, wide_input, wide_output, qa)


if __name__ == "__main__":
    main()
