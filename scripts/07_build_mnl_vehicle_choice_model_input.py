"""Build the clean one-row-per-choice input for the vehicle-choice MNL.

The accepted Phase 1--3 reconstruction and the accepted home-based-tour output
are authoritative inputs.  This downstream step selects canonical explanatory
features, merges exact departure/household/person/vehicle attributes, pivots
vehicle attributes for alternatives 1 and 2, and writes model, manifest and QA
CSVs without changing or re-filtering the strict choice universe.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_household_type import build_household_type  # noqa: E402


@dataclass(frozen=True)
class Paths:
    backbone: Path
    trips: Path
    households: Path
    persons: Path
    vehicles: Path
    candidate_registry: Path
    output_dir: Path
    model_input: Path
    manifest: Path
    qa: Path


ESTIMATION_COLUMNS = [
    "CHOICE_ID",
    "SOURCE_ROW_ID",
    "H_ID",
    "HP_ID",
    "P_ID",
    "W_ID",
    "CHOICE",
    "AV_1",
    "AV_2",
    "W_GEW",
]

TOUR_CONTEXT_COLUMNS = [
    "START_MIN",
    "DAY_TYPE",
    "saison",
    "TOUR_DISTANCE_KM",
    "TOUR_TRAVEL_TIME_MIN",
    "TOUR_N_NONHOME_STOPS",
    "TOUR_N_DISTINCT_PURPOSES",
    "TOUR_FIRST_PURPOSE",
    "TOUR_HAS_WORK",
    "TOUR_HAS_BUSINESS",
    "TOUR_HAS_EDUCATION",
    "TOUR_HAS_SHOPPING",
    "TOUR_HAS_ERRAND",
    "TOUR_HAS_LEISURE",
    "TOUR_HAS_ESCORT",
    "TOUR_HAS_OTHER_PURPOSE",
    "W_ANZBEGL",
    "HOUSEHOLD_ACCOMPANIED",
    "P_STWETTER",
]

HOUSEHOLD_COLUMNS = [
    "household_type",
    "hhgr_gr",
    "H_MIETE",
    "oek_status",
    "H_CS",
    "H_ANZPED",
    "H_ANZRAD",
    "RegioStaR4",
    "XMStadt",
    "quali_opnv",
    "min_bab",
    "min_ozmz",
]

PERSON_COLUMNS = [
    "HP_SEX",
    "alter_gr5",
    "erwerb",
    "P_VPED",
    "P_VRAD",
    "carsharing",
    "mobein",
]

VEHICLE_COLUMNS = [
    "POWERTRAIN_1",
    "POWERTRAIN_2",
    "SEGMENT_1",
    "SEGMENT_2",
    "VEHICLE_AGE_1",
    "VEHICLE_AGE_2",
    "HOLDER_1",
    "HOLDER_2",
    "STATUS_1",
    "STATUS_2",
]

ADDITIVE_VEHICLE_COLUMNS = ["HOLDER_1", "HOLDER_2", "STATUS_1", "STATUS_2"]

PREFERRED_FINAL_COLUMNS = [
    *ESTIMATION_COLUMNS,
    *TOUR_CONTEXT_COLUMNS,
    *HOUSEHOLD_COLUMNS,
    *PERSON_COLUMNS,
    *VEHICLE_COLUMNS,
]

BACKBONE_TOUR_SOURCE_COLUMNS = [
    "START_MIN",
    "TOUR_DISTANCE_KM",
    "TOUR_TRAVEL_TIME_MIN",
    "TOUR_N_NONHOME_STOPS",
    "TOUR_N_DISTINCT_PURPOSES",
    "TOUR_FIRST_PURPOSE",
    "TOUR_HAS_WORK",
    "TOUR_HAS_BUSINESS",
    "TOUR_HAS_EDUCATION",
    "TOUR_HAS_SHOPPING",
    "TOUR_HAS_ERRAND",
    "TOUR_HAS_LEISURE",
    "TOUR_HAS_ESCORT",
    "TOUR_HAS_OTHER_PURPOSE",
]

TRIP_SOURCE_COLUMNS = [
    "SOURCE_ROW_ID",
    "ST_WOTAG",
    "feiertag",
    "saison",
    "ST_JAHR",
    "W_ANZBEGL",
    "W_BEGL_HH",
    "P_STWETTER",
]

HOUSEHOLD_SOURCE_COLUMNS = [
    "H_ID",
    "hhgr_gr",
    "H_MIETE",
    "oek_status",
    "H_CS",
    "H_ANZPED",
    "H_ANZRAD",
    "A_LADEN",
    "RegioStaR4",
    "XMStadt",
    "quali_opnv",
    "min_bab",
    "min_ozmz",
]

PERSON_SOURCE_COLUMNS = [
    "HP_ID",
    "H_ID",
    "P_ID",
    "HP_SEX",
    "alter_gr5",
    "P_FSJAHR",
    "erwerb",
    "P_ARB_ENTF",
    "hoff1",
    "P_VPED",
    "P_VRAD",
    "hind_opnv_anz",
    "carsharing",
    "mobein",
]

VEHICLE_SOURCE_COLUMNS = [
    "H_ID",
    "A_ID",
    "A_ANTRIEB",
    "A_HALTER",
    "A_BAUJ",
    "ST_JAHR",
    "A_KW",
    "seg_kba_gr",
    "status",
    "A_STELL",
]

FORBIDDEN_EXPLICIT = {
    "TOUR_OPEN_END_FLAG",
    "TOUR_CLOSED",
    "TOUR_ELAPSED_TIME_MIN",
    "TOUR_MAIN_PURPOSE",
    "W_ZWECK",
    "zweck",
    "hwzweck1",
    "hwzweck2",
    "wegkm_imp",
    "wegkm_imp_gr",
    "wegmin_imp2",
    "wegmin_imp2_gr",
    "W_SZ",
    "sz_gr1",
    "sz_gr2",
    "W_WAUTO",
    "CHOSEN_A_ID",
    "pkw_fmf",
    "alternative_A_ID",
    "chosen",
}


def resolve_paths() -> Paths:
    output_dir = ROOT / "data_processed" / "model_input"
    paths = Paths(
        backbone=ROOT / "data_processed/reconstruction/phase4/mnl_vehicle_choice_wide_tour_features.csv",
        trips=ROOT / "data_processed/reconstruction/phase1/trips_home_chain_enriched.csv",
        households=ROOT / "data_processed/selected_raw/hh_selected_raw.csv",
        persons=ROOT / "data_processed/selected_raw/persons_selected_raw.csv",
        vehicles=ROOT / "data_processed/selected_raw/cars_selected_raw.csv",
        candidate_registry=ROOT / "metadata/mnl_attribute_candidates.py",
        output_dir=output_dir,
        model_input=output_dir / "mnl_vehicle_choice_model_input.csv",
        manifest=output_dir / "mnl_vehicle_choice_model_input_manifest.csv",
        qa=output_dir / "mnl_vehicle_choice_model_input_QA.csv",
    )
    required = [
        paths.backbone,
        paths.trips,
        paths.households,
        paths.persons,
        paths.vehicles,
        paths.candidate_registry,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required input(s) not found: {missing}")

    print("MNL VEHICLE-CHOICE MODEL INPUT PATHS")
    print(f"strict tour-feature backbone: {paths.backbone.resolve()}")
    print(f"exact trip/departure source: {paths.trips.resolve()}")
    print(f"household source: {paths.households.resolve()}")
    print(f"person source: {paths.persons.resolve()}")
    print(f"vehicle source: {paths.vehicles.resolve()}")
    print(f"candidate registry: {paths.candidate_registry.resolve()}")
    print(f"model input output: {paths.model_input.resolve()}")
    print(f"variable manifest output: {paths.manifest.resolve()}")
    print(f"QA output: {paths.qa.resolve()}")
    return paths


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if usecols is not None:
        missing = [column for column in usecols if column not in header]
        if missing:
            raise KeyError(f"{path.name} is missing required column(s): {missing}")
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def load_inputs(paths: Paths) -> dict[str, pd.DataFrame]:
    if not paths.model_input.exists():
        raise FileNotFoundError(
            "Strictly additive HOLDER/STATUS update requires the accepted existing model input: "
            f"{paths.model_input}"
        )
    backbone_columns = list(dict.fromkeys([*ESTIMATION_COLUMNS, *BACKBONE_TOUR_SOURCE_COLUMNS]))
    household_header = pd.read_csv(paths.households, nrows=0).columns.tolist()
    household_columns = [*HOUSEHOLD_SOURCE_COLUMNS]
    if "household_type" in household_header:
        household_columns.append("household_type")
    return {
        "backbone": read_csv_strings(paths.backbone, backbone_columns),
        "trips": read_csv_strings(paths.trips, TRIP_SOURCE_COLUMNS),
        "households": read_csv_strings(paths.households, household_columns),
        "persons": read_csv_strings(paths.persons, PERSON_SOURCE_COLUMNS),
        "vehicles": read_csv_strings(paths.vehicles, VEHICLE_SOURCE_COLUMNS),
        "previous_model": read_csv_strings(paths.model_input),
    }


def require_nonblank(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    bad = pd.Series(False, index=frame.index)
    for column in columns:
        bad |= frame[column].astype("string").str.strip().eq("")
    if bad.any():
        print(f"\n{label.upper()} ROWS WITH BLANK REQUIRED KEYS")
        print(frame.loc[bad, list(columns)].head(10).to_string(index=False))
        raise ValueError(f"{label} has blank required key values.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def valid_integer(series: pd.Series, valid_values: Iterable[int]) -> pd.Series:
    values = numeric(series)
    result = values.where(values.isin(set(valid_values))).round().astype("Int64")
    return result


def valid_range_integer(series: pd.Series, lower: int, upper: int) -> pd.Series:
    values = numeric(series)
    return values.where(values.between(lower, upper)).round().astype("Int64")


def validate_backbone(backbone: pd.DataFrame) -> pd.DataFrame:
    require_nonblank(
        backbone,
        ["CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID"],
        "strict backbone",
    )
    if not backbone["CHOICE_ID"].is_unique:
        raise ValueError("Strict backbone CHOICE_ID must be unique.")
    if not backbone["SOURCE_ROW_ID"].is_unique:
        raise ValueError("Strict backbone SOURCE_ROW_ID must be unique.")

    choice = numeric(backbone["CHOICE"])
    av_1 = numeric(backbone["AV_1"])
    av_2 = numeric(backbone["AV_2"])
    weight = numeric(backbone["W_GEW"])
    start = numeric(backbone["START_MIN"])
    invalid = (
        ~choice.isin([1, 2])
        | ~av_1.isin([0, 1])
        | ~av_2.isin([0, 1])
        | av_1.ne(1)
        | av_2.ne(1)
        | weight.isna()
        | weight.le(0)
        | start.isna()
    )
    if invalid.any():
        print("\nINVALID STRICT BACKBONE ROWS")
        print(
            backbone.loc[
                invalid,
                ["CHOICE_ID", "CHOICE", "AV_1", "AV_2", "W_GEW", "START_MIN"],
            ].head(10).to_string(index=False)
        )
        raise ValueError("Authoritative strict backbone failed structural validation; no rows were filtered.")
    return backbone[ESTIMATION_COLUMNS + ["START_MIN"]].copy()


def _flatten_registry(value: object) -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for child in value.values():
            flattened.extend(_flatten_registry(child))
        return flattened
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def canonicalize_candidate_registry(path: Path) -> list[dict[str, str]]:
    spec = importlib.util.spec_from_file_location("mnl_attribute_candidates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate registry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = getattr(module, "MNL_ATTRIBUTE_CANDIDATES")
    flat = _flatten_registry(registry)
    if len(flat) != len(set(flat)):
        duplicates = sorted({name for name in flat if flat.count(name) > 1})
        raise ValueError(f"Canonical candidate registry contains duplicate names: {duplicates}")
    forbidden_registry = {
        "W_ZWECK",
        "zweck",
        "wegkm_imp",
        "wegmin_imp2",
        "ST_WOTAG",
        "feiertag",
        "RegioStaR2",
        "RegioStaR7",
        "RegioStaR17",
        "MSIndex",
        "antrieb",
        "A_BAUJ",
        "bauj_gr",
        "A_KW",
        "kw_gr",
        "anzbegl",
        "anzpers",
        "carsharing_diff",
        "hind_opnv",
        "P_ARB_ENTF",
        "arb_entf_gr2",
    }
    present_forbidden = sorted(forbidden_registry.intersection(flat))
    if present_forbidden:
        raise ValueError(f"Candidate registry still contains non-canonical alternatives: {present_forbidden}")

    return [
        {"final": "TOUR_DISTANCE_KM", "selected": "TOUR_DISTANCE_KM", "dropped": "wegkm_imp, wegkm_imp_gr, LEG_DISTANCE_KM", "reason": "accepted tour demand replaces immediate-trip distance"},
        {"final": "TOUR_TRAVEL_TIME_MIN", "selected": "TOUR_TRAVEL_TIME_MIN", "dropped": "wegmin_imp2, wegmin_imp2_gr, LEG_TRAVEL_TIME_MIN", "reason": "accepted tour demand replaces immediate-trip duration"},
        {"final": "TOUR_N_NONHOME_STOPS", "selected": "TOUR_N_NONHOME_STOPS", "dropped": "TOUR_N_TRIPS, TOUR_MULTI_STOP_FLAG", "reason": "direct compact tour-complexity count"},
        {"final": "TOUR_N_DISTINCT_PURPOSES", "selected": "TOUR_N_DISTINCT_PURPOSES", "dropped": "TOUR_MULTI_PURPOSE_FLAG", "reason": "purpose-complexity count contains the binary transformation"},
        {"final": "TOUR_FIRST_PURPOSE", "selected": "TOUR_FIRST_PURPOSE", "dropped": "W_ZWECK, zweck, hwzweck1, hwzweck2, LEG_PURPOSE, TOUR_MAIN_PURPOSE", "reason": "home-origin tour-start purpose is complete for the strict baseline"},
        {"final": "START_MIN", "selected": "START_MIN", "dropped": "W_SZ, sz_gr1, sz_gr2", "reason": "accepted continuous departure time"},
        {"final": "DAY_TYPE", "selected": "ST_WOTAG + feiertag", "dropped": "ST_WOTAG, feiertag", "reason": "single documented day context with holiday precedence"},
        {"final": "saison", "selected": "saison", "dropped": "ST_MONAT, ST_JAHR", "reason": "single documented seasonality construct"},
        {"final": "W_ANZBEGL", "selected": "W_ANZBEGL", "dropped": "anzbegl, anzpers", "reason": "direct accompaniment count"},
        {"final": "household_type", "selected": "household_type", "dropped": "H_GR, HP_ALTER_1, HP_ALTER_2, HP_ALTER_3, HP_ALTER_4, HP_ALTER_5, HP_ALTER_6", "reason": "accepted thesis life-stage classification"},
        {"final": "oek_status", "selected": "oek_status", "dropped": "hheink_gr2, aq_eink_gr", "reason": "official analytical economic-status measure"},
        {"final": "H_ANZPED + H_ANZRAD", "selected": "H_ANZPED + H_ANZRAD", "dropped": "anzpedrad", "reason": "separate direct resource counts preserve pedelec/bicycle distinction"},
        {"final": "RegioStaR4", "selected": "RegioStaR4", "dropped": "RegioStaR2, RegioStaR7, RegioStaR17", "reason": "sole baseline RegioStaR resolution"},
        {"final": "XMStadt", "selected": "XMStadt", "dropped": "MSIndex", "reason": "direct minute-city interpretation; strict-sample Spearman association is about -0.91"},
        {"final": "quali_opnv", "selected": "quali_opnv", "dropped": "quali_nv, bus28, tram28, bahn28", "reason": "interpretable composite PT-quality measure"},
        {"final": "alter_gr5", "selected": "alter_gr5", "dropped": "HP_ALTER, alter_gr6", "reason": "single interpretable categorical age structure"},
        {"final": "erwerb", "selected": "erwerb", "dropped": "P_TAET, taet, taet_diff, P_BKAT", "reason": "single official broad employment indicator"},
        {"final": "P_VPED + P_VRAD", "selected": "P_VPED + P_VRAD", "dropped": "vpedrad", "reason": "separate direct availability preserves pedelec/bicycle distinction"},
        {"final": "carsharing", "selected": "carsharing", "dropped": "carsharing_diff", "reason": "direct binary membership representation"},
        {"final": "POWERTRAIN_1/2", "selected": "A_ANTRIEB", "dropped": "antrieb", "reason": "direct official powertrain preserves hybrid, PHEV and BEV categories"},
        {"final": "VEHICLE_AGE_1/2", "selected": "ST_JAHR - A_BAUJ", "dropped": "A_BAUJ, bauj_gr", "reason": "continuous alternative-specific age"},
        {"final": "SEGMENT_1/2", "selected": "seg_kba_gr", "dropped": "seg_kba", "reason": "validated grouped segment representation"},
        {"final": "HOLDER_1/2", "selected": "A_HALTER", "dropped": "none", "reason": "optional alternative-specific holder type with documented non-substantive codes set missing"},
        {"final": "STATUS_1/2", "selected": "status", "dropped": "none", "reason": "optional analytical vehicle representation derived from KBA segment and construction year; do not automatically combine with SEGMENT + VEHICLE_AGE"},
        {"final": "baseline route/OD", "selected": "none", "dropped": "auto_dist, auto_dauer, rad_dist, rad_dauer, opnv_dist, opnv_dauer_*", "reason": "deferred extension overlaps tour scale"},
        {"final": "baseline trip spatial", "selected": "household/home context", "dropped": "*_SO, *_ZO, RegioStaRGem7, RegioStaRGem5", "reason": "home-origin baseline uses household spatial context"},
    ]


def select_tour_features(backbone: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=backbone.index)
    out["START_MIN"] = backbone["START_MIN"]
    distance = numeric(backbone["TOUR_DISTANCE_KM"])
    duration = numeric(backbone["TOUR_TRAVEL_TIME_MIN"])
    if distance.isna().any() or distance.lt(0).any():
        raise ValueError("Accepted TOUR_DISTANCE_KM contains missing or negative values.")
    if duration.isna().any() or duration.lt(0).any():
        raise ValueError("Accepted TOUR_TRAVEL_TIME_MIN contains missing or negative values.")
    out["TOUR_DISTANCE_KM"] = distance
    out["TOUR_TRAVEL_TIME_MIN"] = duration
    out["TOUR_N_NONHOME_STOPS"] = valid_range_integer(backbone["TOUR_N_NONHOME_STOPS"], 0, 100)
    out["TOUR_N_DISTINCT_PURPOSES"] = valid_range_integer(backbone["TOUR_N_DISTINCT_PURPOSES"], 0, 7)
    out["TOUR_FIRST_PURPOSE"] = valid_range_integer(backbone["TOUR_FIRST_PURPOSE"], 1, 16)
    for column in [
        "TOUR_HAS_WORK",
        "TOUR_HAS_BUSINESS",
        "TOUR_HAS_EDUCATION",
        "TOUR_HAS_SHOPPING",
        "TOUR_HAS_ERRAND",
        "TOUR_HAS_LEISURE",
        "TOUR_HAS_ESCORT",
        "TOUR_HAS_OTHER_PURPOSE",
    ]:
        out[column] = valid_integer(backbone[column], [0, 1])
    return out


def select_trip_context_features(trips: pd.DataFrame) -> pd.DataFrame:
    if not trips["SOURCE_ROW_ID"].is_unique:
        raise ValueError("Trip SOURCE_ROW_ID must be unique before the exact merge.")
    weekday = valid_integer(trips["ST_WOTAG"], range(1, 8))
    holiday = valid_integer(trips["feiertag"], [0, 1])
    day_type = pd.Series(pd.NA, index=trips.index, dtype="string")
    day_type.loc[holiday.eq(1)] = "HOLIDAY"
    day_type.loc[holiday.eq(0) & weekday.isin([6, 7])] = "WEEKEND"
    day_type.loc[holiday.eq(0) & weekday.isin([1, 2, 3, 4, 5])] = "WEEKDAY"

    out = pd.DataFrame({"SOURCE_ROW_ID": trips["SOURCE_ROW_ID"]})
    out["DAY_TYPE"] = day_type
    out["saison"] = valid_integer(trips["saison"], [1, 2, 3, 4])
    out["W_ANZBEGL"] = valid_range_integer(trips["W_ANZBEGL"], 0, 25)
    out["HOUSEHOLD_ACCOMPANIED"] = numeric(trips["W_BEGL_HH"]).map({1: 1, 2: 0, 716: 0}).astype("Int64")
    out["P_STWETTER"] = valid_integer(trips["P_STWETTER"], [1, 2, 3, 4, 5, 6])
    out["_REPORT_YEAR"] = valid_integer(trips["ST_JAHR"], [2023, 2024])
    out["_TRIP_KEY_MATCH"] = 1
    return out


def select_household_features(households: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    if not households["H_ID"].is_unique:
        raise ValueError("Household H_ID must be unique before the merge.")
    source = households.copy()
    if "household_type" not in source.columns:
        # Reuse the accepted thesis classifier without modifying selected/raw data.
        derived = build_household_type(persons[["H_ID", "alter_gr5"]])
        source = source.merge(derived, on="H_ID", how="left", validate="one_to_one")

    out = pd.DataFrame({"H_ID": source["H_ID"]})
    out["household_type"] = source["household_type"].astype("string").replace({"": pd.NA, "unknown": pd.NA})
    out["hhgr_gr"] = valid_integer(source["hhgr_gr"], [1, 2, 3, 4, 5])
    out["H_MIETE"] = valid_integer(source["H_MIETE"], [1, 2, 3])
    out["oek_status"] = valid_integer(source["oek_status"], [1, 2, 3, 4, 5])
    out["H_CS"] = numeric(source["H_CS"]).map({1: 1, 2: 0}).astype("Int64")
    out["H_ANZPED"] = valid_range_integer(source["H_ANZPED"], 0, 10)
    out["H_ANZRAD"] = valid_range_integer(source["H_ANZRAD"], 0, 10)
    out["RegioStaR4"] = valid_integer(source["RegioStaR4"], [11, 12, 21, 22])
    out["XMStadt"] = valid_integer(source["XMStadt"], [1, 2, 3, 4, 5, 6])
    out["quali_opnv"] = valid_integer(source["quali_opnv"], [1, 2, 3, 4])
    out["min_bab"] = valid_integer(source["min_bab"], [1, 2, 3, 4, 5])
    out["min_ozmz"] = valid_integer(source["min_ozmz"], [1, 2, 3, 4, 5])
    out["_OMIT_HOME_CHARGING"] = valid_integer(source["A_LADEN"], [1, 2, 3, 4])
    out["_HOUSEHOLD_KEY_MATCH"] = 1
    return out


def select_person_features(persons: pd.DataFrame) -> pd.DataFrame:
    if not persons["HP_ID"].is_unique:
        raise ValueError("Person HP_ID must be unique before the merge.")
    if persons.duplicated(["H_ID", "P_ID"], keep=False).any():
        raise ValueError("Person H_ID + P_ID must also be unique in the selected source.")
    out = pd.DataFrame(
        {
            "HP_ID": persons["HP_ID"],
            "_PERSON_H_ID": persons["H_ID"],
            "_PERSON_P_ID": persons["P_ID"],
        }
    )
    out["HP_SEX"] = valid_integer(persons["HP_SEX"], [1, 2, 3])
    out["alter_gr5"] = valid_range_integer(persons["alter_gr5"], 1, 15)
    out["erwerb"] = valid_integer(persons["erwerb"], [0, 1])
    out["P_VPED"] = numeric(persons["P_VPED"]).map({1: 1, 2: 0}).astype("Int64")
    out["P_VRAD"] = numeric(persons["P_VRAD"]).map({1: 1, 2: 0}).astype("Int64")
    out["carsharing"] = valid_integer(persons["carsharing"], [0, 1])
    out["mobein"] = valid_integer(persons["mobein"], [0, 1])
    out["_OMIT_LICENSE_YEAR"] = valid_range_integer(persons["P_FSJAHR"], 1956, 2024)
    out["_OMIT_P_ARB_ENTF"] = numeric(persons["P_ARB_ENTF"]).where(numeric(persons["P_ARB_ENTF"]).between(0, 200))
    out["_OMIT_HOFF1"] = valid_integer(persons["hoff1"], [0, 1])
    out["_OMIT_HIND_OPNV_ANZ"] = valid_range_integer(persons["hind_opnv_anz"], 0, 5)
    out["_PERSON_KEY_MATCH"] = 1
    return out


def prepare_vehicle_features(vehicles: pd.DataFrame, strict_households: set[str]) -> pd.DataFrame:
    if vehicles.duplicated(["H_ID", "A_ID"], keep=False).any():
        raise ValueError("Vehicle H_ID + A_ID must be unique before pivoting.")
    source = vehicles.loc[
        vehicles["H_ID"].isin(strict_households) & vehicles["A_ID"].isin(["1", "2"])
    ].copy()
    source["_A_ID_NUM"] = valid_integer(source["A_ID"], [1, 2])

    expected = pd.MultiIndex.from_product(
        [sorted(strict_households), [1, 2]], names=["H_ID", "_A_ID_NUM"]
    )
    actual = pd.MultiIndex.from_frame(source[["H_ID", "_A_ID_NUM"]])
    missing_pairs = expected.difference(actual)
    if len(missing_pairs):
        print("\nSTRICT HOUSEHOLD VEHICLE PAIRS MISSING FROM CARS SOURCE")
        print(list(missing_pairs[:10]))
        raise ValueError("Every strict household must resolve A_ID 1 and A_ID 2; no choice rows were dropped.")

    source["POWERTRAIN"] = valid_integer(source["A_ANTRIEB"], [1, 2, 3, 4, 5, 6, 7])
    source["SEGMENT"] = valid_integer(source["seg_kba_gr"], [1, 2, 3, 4])
    reporting_year = valid_integer(source["ST_JAHR"], [2023, 2024])
    build_year = valid_range_integer(source["A_BAUJ"], 1900, 2024)
    age = reporting_year - build_year
    if age.dropna().lt(0).any() or age.dropna().gt(124).any():
        bad = source.loc[age.notna() & (~age.between(0, 124)), ["H_ID", "A_ID", "ST_JAHR", "A_BAUJ"]]
        print("\nIMPLAUSIBLE DOCUMENTED VEHICLE AGE DERIVATIONS")
        print(bad.head(10).to_string(index=False))
        raise ValueError("Vehicle age must be non-negative and within the documented 1900--2024 range.")
    source["VEHICLE_AGE"] = age.astype("Int64")
    source["HOLDER"] = valid_integer(source["A_HALTER"], [1, 2, 3])
    source["STATUS"] = valid_integer(source["status"], [1, 2, 3])
    source["_OMIT_ENGINE_POWER_KW"] = numeric(source["A_KW"]).where(numeric(source["A_KW"]).between(15, 600))
    source["_OMIT_PARKING"] = valid_integer(source["A_STELL"], [1, 2, 3, 4, 5])
    source["_VEHICLE_KEY_MATCH"] = 1

    features = [
        "POWERTRAIN",
        "SEGMENT",
        "VEHICLE_AGE",
        "HOLDER",
        "STATUS",
        "_OMIT_ENGINE_POWER_KW",
        "_OMIT_PARKING",
        "_VEHICLE_KEY_MATCH",
    ]
    wide = pd.DataFrame({"H_ID": sorted(strict_households)})
    for feature in features:
        pivot = source.pivot(index="H_ID", columns="_A_ID_NUM", values=feature)
        pivot = pivot.reindex(columns=[1, 2])
        pivot.columns = [f"{feature}_{alternative}" for alternative in [1, 2]]
        wide = wide.merge(pivot.reset_index(), on="H_ID", how="left", validate="one_to_one")
    return wide


def merge_features(
    backbone: pd.DataFrame,
    tour: pd.DataFrame,
    trip: pd.DataFrame,
    household: pd.DataFrame,
    person: pd.DataFrame,
    vehicle: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = backbone[ESTIMATION_COLUMNS + ["START_MIN"]].copy()
    merged["_STRICT_ROW_ORDER"] = np.arange(len(merged), dtype=np.int64)
    for column in tour.columns:
        if column != "START_MIN":
            merged[column] = tour[column].to_numpy()

    merged = merged.merge(trip, on="SOURCE_ROW_ID", how="left", validate="one_to_one", sort=False)
    merged = merged.merge(household, on="H_ID", how="left", validate="many_to_one", sort=False)
    merged = merged.merge(person, on="HP_ID", how="left", validate="many_to_one", sort=False)
    merged = merged.merge(vehicle, on="H_ID", how="left", validate="many_to_one", sort=False)
    merged = merged.sort_values("_STRICT_ROW_ORDER", kind="mergesort").reset_index(drop=True)

    person_match = merged["_PERSON_KEY_MATCH"].eq(1)
    person_id_mismatch = person_match & (
        merged["H_ID"].ne(merged["_PERSON_H_ID"])
        | merged["P_ID"].ne(merged["_PERSON_P_ID"])
    )
    if person_id_mismatch.any():
        print("\nHP_ID PERSON MAPPING CONTRADICTIONS")
        print(
            merged.loc[
                person_id_mismatch,
                ["CHOICE_ID", "H_ID", "P_ID", "HP_ID", "_PERSON_H_ID", "_PERSON_P_ID"],
            ].head(10).to_string(index=False)
        )
        raise ValueError("HP_ID mapped to a different H_ID/P_ID than the accepted backbone.")

    experience = merged["_REPORT_YEAR"] - merged["_OMIT_LICENSE_YEAR"]
    merged["_OMIT_DRIVING_EXPERIENCE_YEARS"] = experience.where(experience.between(0, 80))

    metrics = {
        "source_row_unmatched": int(merged["_TRIP_KEY_MATCH"].ne(1).sum()),
        "household_unmatched": int(merged["_HOUSEHOLD_KEY_MATCH"].ne(1).sum()),
        "person_unmatched": int(merged["_PERSON_KEY_MATCH"].ne(1).sum()),
        "vehicle_1_unmatched": int(merged["_VEHICLE_KEY_MATCH_1"].ne(1).sum()),
        "vehicle_2_unmatched": int(merged["_VEHICLE_KEY_MATCH_2"].ne(1).sum()),
    }
    return merged, metrics


def validate_missing_codes(frame: pd.DataFrame) -> None:
    allowed: dict[str, set[object]] = {
        "DAY_TYPE": {"WEEKDAY", "WEEKEND", "HOLIDAY"},
        "saison": {1, 2, 3, 4},
        "TOUR_HAS_WORK": {0, 1},
        "TOUR_HAS_BUSINESS": {0, 1},
        "TOUR_HAS_EDUCATION": {0, 1},
        "TOUR_HAS_SHOPPING": {0, 1},
        "TOUR_HAS_ERRAND": {0, 1},
        "TOUR_HAS_LEISURE": {0, 1},
        "TOUR_HAS_ESCORT": {0, 1},
        "TOUR_HAS_OTHER_PURPOSE": {0, 1},
        "HOUSEHOLD_ACCOMPANIED": {0, 1},
        "P_STWETTER": {1, 2, 3, 4, 5, 6},
        "hhgr_gr": {1, 2, 3, 4, 5},
        "H_MIETE": {1, 2, 3},
        "oek_status": {1, 2, 3, 4, 5},
        "H_CS": {0, 1},
        "RegioStaR4": {11, 12, 21, 22},
        "XMStadt": {1, 2, 3, 4, 5, 6},
        "quali_opnv": {1, 2, 3, 4},
        "min_bab": {1, 2, 3, 4, 5},
        "min_ozmz": {1, 2, 3, 4, 5},
        "HP_SEX": {1, 2, 3},
        "erwerb": {0, 1},
        "P_VPED": {0, 1},
        "P_VRAD": {0, 1},
        "carsharing": {0, 1},
        "mobein": {0, 1},
        "POWERTRAIN_1": {1, 2, 3, 4, 5, 6, 7},
        "POWERTRAIN_2": {1, 2, 3, 4, 5, 6, 7},
        "SEGMENT_1": {1, 2, 3, 4},
        "SEGMENT_2": {1, 2, 3, 4},
        "HOLDER_1": {1, 2, 3},
        "HOLDER_2": {1, 2, 3},
        "STATUS_1": {1, 2, 3},
        "STATUS_2": {1, 2, 3},
    }
    for column, valid in allowed.items():
        unexpected = set(frame[column].dropna().unique()).difference(valid)
        if unexpected:
            raise ValueError(f"{column} contains undocumented valid values after cleaning: {sorted(unexpected)}")


def drop_constant_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    explanatory = [column for column in PREFERRED_FINAL_COLUMNS if column not in ESTIMATION_COLUMNS]
    excluded: list[str] = []
    retained: list[str] = []
    for column in explanatory:
        if frame[column].nunique(dropna=True) <= 1:
            excluded.append(column)
        else:
            retained.append(column)
    final_columns = [*ESTIMATION_COLUMNS, *retained]
    return final_columns, excluded


def validate_additive_update(previous: pd.DataFrame, final: pd.DataFrame) -> dict[str, object]:
    existing_columns = [
        column for column in previous.columns if column not in ADDITIVE_VEHICLE_COLUMNS
    ]
    expected_columns = [*existing_columns, *ADDITIVE_VEHICLE_COLUMNS]
    if list(final.columns) != expected_columns:
        raise AssertionError(
            "Strict additive column order failed: expected all accepted columns unchanged "
            "followed by HOLDER_1/2 and STATUS_1/2."
        )
    if len(final) != len(previous):
        raise AssertionError("Strict additive update changed the accepted row count.")
    for column in existing_columns:
        before = previous[column].astype("string").fillna("").reset_index(drop=True)
        after = final[column].astype("string").fillna("").reset_index(drop=True)
        if not after.equals(before):
            mismatch = after.ne(before)
            first_position = int(mismatch[mismatch].index[0])
            raise AssertionError(
                f"Strict additive update changed accepted column {column} at row "
                f"position {first_position}."
            )
    if previous["CHOICE_ID"].nunique() != final["CHOICE_ID"].nunique():
        raise AssertionError("Strict additive update changed the unique CHOICE_ID count.")
    if previous["H_ID"].nunique() != final["H_ID"].nunique():
        raise AssertionError("Strict additive update changed the household count.")
    if previous["HP_ID"].nunique() != final["HP_ID"].nunique():
        raise AssertionError("Strict additive update changed the person count.")
    before_choice_distribution = previous["CHOICE"].value_counts(dropna=False).sort_index()
    after_choice_distribution = final["CHOICE"].astype("string").value_counts(dropna=False).sort_index()
    if not before_choice_distribution.equals(after_choice_distribution):
        raise AssertionError("Strict additive update changed the CHOICE distribution.")
    return {
        "rows_before": len(previous),
        "rows_after": len(final),
        "columns_before": len(existing_columns),
        "columns_after": len(final.columns),
        "unique_choice_id_before": previous["CHOICE_ID"].nunique(),
        "unique_choice_id_after": final["CHOICE_ID"].nunique(),
        "existing_column_count": len(existing_columns),
        "existing_column_equality": "PASS",
    }


def _manifest_catalog() -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}

    def add(
        name: str,
        level: str,
        source_file: str,
        source_variable: str,
        transformation: str,
        role: str,
        valid: str,
        missing: str,
        reason: str,
        dropped: str = "",
    ) -> None:
        catalog[name] = {
            "final_variable": name,
            "level": level,
            "source_file": source_file,
            "source_variable": source_variable,
            "transformation": transformation,
            "role": role,
            "valid_value_definition": valid,
            "missing_value_handling": missing,
            "reason_selected": reason,
            "alternatives_dropped": dropped,
        }

    backbone = "data_processed/reconstruction/phase4/mnl_vehicle_choice_wide_tour_features.csv"
    trips = "data_processed/reconstruction/phase1/trips_home_chain_enriched.csv"
    households = "data_processed/selected_raw/hh_selected_raw.csv"
    persons = "data_processed/selected_raw/persons_selected_raw.csv"
    vehicles = "data_processed/selected_raw/cars_selected_raw.csv"
    for name, role in [
        ("CHOICE_ID", "unique choice occasion identifier"),
        ("SOURCE_ROW_ID", "exact source-trip merge key"),
        ("H_ID", "household identifier"),
        ("HP_ID", "accepted unique person identifier"),
        ("P_ID", "within-household person identifier"),
        ("W_ID", "within-person trip identifier"),
        ("CHOICE", "observed alternative outcome"),
        ("AV_1", "alternative 1 availability"),
        ("AV_2", "alternative 2 availability"),
        ("W_GEW", "trip/choice survey weight"),
    ]:
        add(name, "estimation", backbone, name, "copied unchanged", role, "accepted strict-backbone value", "none introduced", "required estimation structure")

    add("START_MIN", "departure", backbone, "START_MIN", "copied unchanged", "departure time", "numeric minute of reporting day", "none in accepted backbone", "canonical continuous departure time", "W_SZ; sz_gr1; sz_gr2")
    add("DAY_TYPE", "departure", trips, "ST_WOTAG + feiertag", "HOLIDAY if feiertag=1; else WEEKEND for ST_WOTAG 6/7; else WEEKDAY for 1--5", "day context", "WEEKDAY, WEEKEND, HOLIDAY", "undocumented source codes become blank", "single documented day representation with holiday precedence", "ST_WOTAG; feiertag")
    add("saison", "departure", trips, "saison", "documented codes retained", "seasonal context", "1 winter; 2 spring; 3 summer; 4 autumn", "other codes become blank", "official seasonal context", "ST_MONAT; ST_JAHR")
    add("TOUR_DISTANCE_KM", "tour", backbone, "TOUR_DISTANCE_KM", "numeric accepted tour total", "tour demand", "non-negative kilometres", "none in accepted backbone", "canonical demand magnitude", "wegkm_imp; wegkm_imp_gr; LEG_DISTANCE_KM")
    add("TOUR_TRAVEL_TIME_MIN", "tour", backbone, "TOUR_TRAVEL_TIME_MIN", "numeric accepted tour total", "tour demand", "non-negative minutes", "none in accepted backbone", "canonical demand duration", "wegmin_imp2; wegmin_imp2_gr; LEG_TRAVEL_TIME_MIN")
    add("TOUR_N_NONHOME_STOPS", "tour", backbone, "TOUR_N_NONHOME_STOPS", "documented non-negative count", "tour complexity", "integer 0 or greater", "undocumented values become blank", "compact stop-complexity measure", "TOUR_N_TRIPS; TOUR_MULTI_STOP_FLAG")
    add("TOUR_N_DISTINCT_PURPOSES", "tour", backbone, "TOUR_N_DISTINCT_PURPOSES", "documented count of broad purpose indicators", "purpose complexity", "integer 0--7", "undocumented values become blank", "compact purpose-complexity measure", "TOUR_MULTI_PURPOSE_FLAG")
    add("TOUR_FIRST_PURPOSE", "tour", backbone, "TOUR_FIRST_PURPOSE", "documented purpose codes retained", "tour-start purpose", "MiD purpose codes 1--16", "99/undocumented values become blank", "home-origin departure purpose is complete baseline concept", "W_ZWECK; zweck; hwzweck1; hwzweck2; LEG_PURPOSE; TOUR_MAIN_PURPOSE")
    for name in ["TOUR_HAS_WORK", "TOUR_HAS_BUSINESS", "TOUR_HAS_EDUCATION", "TOUR_HAS_SHOPPING", "TOUR_HAS_ERRAND", "TOUR_HAS_LEISURE", "TOUR_HAS_ESCORT", "TOUR_HAS_OTHER_PURPOSE"]:
        add(name, "tour", backbone, name, "documented binary retained", "observed tour purpose composition", "0 no; 1 yes", "other codes become blank", "substantive behavioural composition feature")
    add("W_ANZBEGL", "departure", trips, "W_ANZBEGL", "retain documented direct count", "departure accompaniment", "integer 0--25 (25=25 or more)", "99 and 701 become blank", "direct count at vehicle-selection moment", "anzbegl; anzpers")
    add("HOUSEHOLD_ACCOMPANIED", "departure", trips, "W_BEGL_HH", "1->1; 2 or documented no-accompaniment 716->0", "household accompaniment", "0 no household companion; 1 household companion", "9/202/701/717/995 become blank", "distinct from total accompaniment count")
    add("P_STWETTER", "departure", trips, "P_STWETTER", "documented weather codes retained", "weather context", "1 sunny; 2 lightly cloudy; 3 changeable; 4 overcast; 5 rainy; 6 snow", "9 and proxy code 210 become blank", "single reporting-day weather representation")

    add("household_type", "household", f"{households}; {persons}", "household_type or accepted aggregation of alter_gr5", "reuse src/build_household_type.py in memory when selected household file lacks the stored field", "household life stage", "family_household; young_household; adult_household; senior_household", "unknown becomes blank; rows retained", "validated thesis construct", "H_GR; HP_ALTER_1--6")
    add("hhgr_gr", "household", households, "hhgr_gr", "documented codes retained", "household size", "1,2,3,4,5 persons or more", "other codes become blank", "distinct size construct from household_type", "H_GR")
    add("H_MIETE", "household", households, "H_MIETE", "documented codes retained", "housing tenure", "1 rent; 2 ownership; 3 other", "9 and 309 become blank", "interpretable housing context")
    add("oek_status", "household", households, "oek_status", "documented codes retained", "economic status", "1 very low through 5 very high", "undocumented codes become blank", "official analytical socioeconomic measure", "hheink_gr2; aq_eink_gr")
    add("H_CS", "household", households, "H_CS", "1 yes -> 1; 2 no -> 0", "household mobility resource", "0 no; 1 yes", "9 becomes blank", "direct household carsharing membership")
    add("H_ANZPED", "household", households, "H_ANZPED", "direct documented count", "household pedelec resources", "integer 0--10 (10=10 or more)", "99 becomes blank", "separate direct resource count", "anzpedrad")
    add("H_ANZRAD", "household", households, "H_ANZRAD", "direct documented count", "household bicycle resources", "integer 0--10 (10=10 or more)", "99 becomes blank", "separate direct resource count", "anzpedrad")
    add("RegioStaR4", "household", households, "RegioStaR4", "documented codes retained", "broad spatial context", "11, 12, 21, 22", "undocumented codes become blank", "sole baseline RegioStaR resolution", "RegioStaR2; RegioStaR7; RegioStaR17")
    add("XMStadt", "household", households, "XMStadt", "documented minute-city categories retained", "local urban structure", "1 five-minute through 6 sixty-minute/other", "95 becomes blank", "more directly interpretable than highly overlapping MSIndex", "MSIndex")
    add("quali_opnv", "household", households, "quali_opnv", "documented ordinal codes retained", "public transport quality", "1 very poor through 4 very good", "95 becomes blank", "composite baseline PT accessibility", "quali_nv; bus28; tram28; bahn28")
    add("min_bab", "household", households, "min_bab", "documented ordinal codes retained", "motorway accessibility", "1 under 10 minutes through 5 40+ minutes", "95 becomes blank", "distinct road-accessibility construct")
    add("min_ozmz", "household", households, "min_ozmz", "documented ordinal codes retained", "central-place accessibility", "1 under 10 minutes through 5 40+ minutes", "95 becomes blank", "distinct central-place construct")

    add("HP_SEX", "person", persons, "HP_SEX", "documented codes retained", "sex", "1 male; 2 female; 3 diverse", "9 becomes blank", "official completed sex variable")
    add("alter_gr5", "person", persons, "alter_gr5", "documented grouped age retained", "age", "ordinal categories 1--15", "undocumented codes become blank", "single thesis age representation", "HP_ALTER; alter_gr6")
    add("erwerb", "person", persons, "erwerb", "documented binary retained", "employment status", "0 no; 1 yes including apprentice", "9 becomes blank", "single broad employment representation", "P_TAET; taet; taet_diff; P_BKAT")
    add("P_VPED", "person", persons, "P_VPED", "1 yes -> 1; 2 no -> 0", "personal pedelec availability", "0 no; 1 yes", "9/402 become blank", "direct personal resource measure", "vpedrad")
    add("P_VRAD", "person", persons, "P_VRAD", "1 yes -> 1; 2 no -> 0", "personal bicycle availability", "0 no; 1 yes", "9 becomes blank", "direct personal resource measure", "vpedrad")
    add("carsharing", "person", persons, "carsharing", "documented binary retained", "personal carsharing membership", "0 no; 1 yes", "9/202/206/403 become blank", "single direct membership representation", "carsharing_diff")
    add("mobein", "person", persons, "mobein", "documented binary retained", "mobility limitation", "0 no; 1 yes", "9/403/418 become blank", "documented mobility limitation")

    for alternative in [1, 2]:
        add(f"POWERTRAIN_{alternative}", "vehicle alternative", vehicles, "A_ANTRIEB", f"select H_ID + A_ID {alternative}; retain documented codes", f"alternative {alternative} powertrain", "1 petrol; 2 diesel; 3 hybrid; 4 PHEV; 5 BEV; 6 gas; 7 other", "94/99 become blank", "preserves relevant powertrain distinctions", "antrieb")
        add(f"SEGMENT_{alternative}", "vehicle alternative", vehicles, "seg_kba_gr", f"select H_ID + A_ID {alternative}; retain documented grouped segment", f"alternative {alternative} segment", "1 small; 2 compact; 3 medium; 4 large", "95 becomes blank", "validated modelling segment grouping", "seg_kba")
        add(f"VEHICLE_AGE_{alternative}", "vehicle alternative", vehicles, "ST_JAHR - A_BAUJ", f"select H_ID + A_ID {alternative}; reporting year minus valid build year", f"alternative {alternative} vehicle age", "integer 0--124 under documented 1900--2024 range", "A_BAUJ=9999 or invalid year becomes blank", "continuous age is more interpretable than build-year groups", "A_BAUJ; bauj_gr")
        add(f"HOLDER_{alternative}", "vehicle alternative", vehicles, "A_HALTER", f"pivot by H_ID + A_ID; select alternative {alternative}; retain documented codes", "alternative-specific vehicle holder type", "1 private; 2 company; 3 other", "9/101/202 become blank; rows retained", "optional alternative-specific holder feature")
        add(f"STATUS_{alternative}", "vehicle alternative", vehicles, "status", f"pivot by H_ID + A_ID; select alternative {alternative}; retain documented codes", "alternative-specific vehicle status class", "1 simple; 2 medium; 3 high", "95 becomes blank; rows retained", "STATUS is derived from KBA segment and construction year and should be treated as an alternative vehicle representation rather than automatically combined with SEGMENT + VEHICLE_AGE in the same MNL specification.")
    return catalog


def build_manifest(final_columns: list[str]) -> pd.DataFrame:
    catalog = _manifest_catalog()
    missing_specs = [column for column in final_columns if column not in catalog]
    if missing_specs:
        raise KeyError(f"Manifest catalog is missing final column(s): {missing_specs}")
    columns = [
        "final_variable",
        "level",
        "source_file",
        "source_variable",
        "transformation",
        "role",
        "valid_value_definition",
        "missing_value_handling",
        "reason_selected",
        "alternatives_dropped",
    ]
    return pd.DataFrame([catalog[column] for column in final_columns], columns=columns)


QA_COLUMNS = [
    "section",
    "metric",
    "variable",
    "value",
    "denominator",
    "share",
    "non_missing_count",
    "missing_count",
    "missing_share",
    "unique_valid_values",
    "min_value",
    "max_value",
    "modal_value",
    "constant_indicator",
    "details",
]


def qa_row(section: str, metric: str, **values: object) -> dict[str, object]:
    row = {column: "" for column in QA_COLUMNS}
    row.update({"section": section, "metric": metric})
    row.update(values)
    return row


def feature_profile_row(section: str, variable: str, series: pd.Series, details: str = "") -> dict[str, object]:
    missing_count = int(series.isna().sum())
    non_missing = series.dropna()
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    numeric_complete = len(non_missing) > 0 and int(numeric_values.notna().sum()) == len(non_missing)
    modes = non_missing.mode(dropna=True)
    return qa_row(
        section,
        "feature profile" if section == "FEATURE_QA" else "omitted candidate profile",
        variable=variable,
        non_missing_count=len(non_missing),
        missing_count=missing_count,
        missing_share=(missing_count / len(series)) if len(series) else 0.0,
        unique_valid_values=int(non_missing.nunique(dropna=True)),
        min_value=(numeric_values.min() if numeric_complete else ""),
        max_value=(numeric_values.max() if numeric_complete else ""),
        modal_value=(modes.iloc[0] if len(modes) else ""),
        constant_indicator=int(non_missing.nunique(dropna=True) <= 1),
        details=details,
    )


def build_QA_summary(
    inputs: dict[str, pd.DataFrame],
    merged: pd.DataFrame,
    final: pd.DataFrame,
    final_columns: list[str],
    merge_metrics: dict[str, int],
    canonicalization: list[dict[str, str]],
    constant_exclusions: list[str],
    additive_comparison: dict[str, object],
) -> pd.DataFrame:
    backbone = inputs["backbone"]
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            qa_row("SAMPLE_PRESERVATION", "input rows", value=len(backbone)),
            qa_row("SAMPLE_PRESERVATION", "output rows", value=len(final)),
            qa_row("SAMPLE_PRESERVATION", "input unique CHOICE_ID", value=backbone["CHOICE_ID"].nunique()),
            qa_row("SAMPLE_PRESERVATION", "output unique CHOICE_ID", value=final["CHOICE_ID"].nunique()),
            qa_row("SAMPLE_PRESERVATION", "households", value=final["H_ID"].nunique()),
            qa_row("SAMPLE_PRESERVATION", "persons", value=final["HP_ID"].nunique()),
        ]
    )
    for value, count in numeric(final["CHOICE"]).value_counts().sort_index().items():
        rows.append(qa_row("SAMPLE_PRESERVATION", "CHOICE distribution", variable=str(int(value)), value=int(count), denominator=len(final), share=int(count) / len(final)))

    duplicate_metrics = {
        "duplicate SOURCE_ROW_ID rows in trip source": int(inputs["trips"].duplicated(["SOURCE_ROW_ID"], keep=False).sum()),
        "duplicate H_ID rows in household source": int(inputs["households"].duplicated(["H_ID"], keep=False).sum()),
        "duplicate HP_ID rows in person source": int(inputs["persons"].duplicated(["HP_ID"], keep=False).sum()),
        "duplicate H_ID + P_ID rows in person source": int(inputs["persons"].duplicated(["H_ID", "P_ID"], keep=False).sum()),
        "duplicate H_ID + A_ID rows in vehicle source": int(inputs["vehicles"].duplicated(["H_ID", "A_ID"], keep=False).sum()),
    }
    for metric, value in duplicate_metrics.items():
        rows.append(qa_row("MERGE_QA", metric, value=value))
    rows.extend(
        [
            qa_row("MERGE_QA", "unmatched SOURCE_ROW_ID rows", value=merge_metrics["source_row_unmatched"], denominator=len(final), share=merge_metrics["source_row_unmatched"] / len(final)),
            qa_row("MERGE_QA", "unmatched household rows", value=merge_metrics["household_unmatched"], denominator=len(final), share=merge_metrics["household_unmatched"] / len(final)),
            qa_row("MERGE_QA", "unmatched person rows", value=merge_metrics["person_unmatched"], denominator=len(final), share=merge_metrics["person_unmatched"] / len(final)),
            qa_row("MERGE_QA", "unmatched vehicle-1 rows", value=merge_metrics["vehicle_1_unmatched"], denominator=len(final), share=merge_metrics["vehicle_1_unmatched"] / len(final)),
            qa_row("MERGE_QA", "unmatched vehicle-2 rows", value=merge_metrics["vehicle_2_unmatched"], denominator=len(final), share=merge_metrics["vehicle_2_unmatched"] / len(final)),
            qa_row("MERGE_QA", "many-to-many merge attempts", value=0, details="all merges use explicit one_to_one or many_to_one validation"),
        ]
    )

    choice = numeric(final["CHOICE"])
    weight = numeric(final["W_GEW"])
    rows.extend(
        [
            qa_row("STRUCTURAL_QA", "missing CHOICE", value=int(choice.isna().sum())),
            qa_row("STRUCTURAL_QA", "invalid CHOICE", value=int((choice.notna() & ~choice.isin([1, 2])).sum())),
            qa_row("STRUCTURAL_QA", "missing W_GEW", value=int(weight.isna().sum())),
            qa_row("STRUCTURAL_QA", "non-positive W_GEW", value=int(weight.le(0).fillna(False).sum())),
        ]
    )
    for column in ["AV_1", "AV_2"]:
        for value, count in numeric(final[column]).value_counts().sort_index().items():
            rows.append(qa_row("STRUCTURAL_QA", f"{column} distribution", variable=str(int(value)), value=int(count), denominator=len(final), share=int(count) / len(final)))

    rows.extend(
        [
            qa_row("ADDITIVE_UPDATE", "rows before", value=additive_comparison["rows_before"]),
            qa_row("ADDITIVE_UPDATE", "rows after", value=additive_comparison["rows_after"]),
            qa_row("ADDITIVE_UPDATE", "columns before", value=additive_comparison["columns_before"]),
            qa_row("ADDITIVE_UPDATE", "columns after", value=additive_comparison["columns_after"]),
            qa_row("ADDITIVE_UPDATE", "unique CHOICE_ID before", value=additive_comparison["unique_choice_id_before"]),
            qa_row("ADDITIVE_UPDATE", "unique CHOICE_ID after", value=additive_comparison["unique_choice_id_after"]),
            qa_row(
                "ADDITIVE_UPDATE",
                "existing-column equality check",
                variable=str(additive_comparison["existing_column_count"]),
                value=additive_comparison["existing_column_equality"],
                details="Every pre-existing column compared row-for-row in its accepted order.",
            ),
        ]
    )

    for feature in ["HOLDER", "STATUS"]:
        for alternative in [1, 2]:
            column = f"{feature}_{alternative}"
            valid_count = int(final[column].notna().sum())
            missing_count = int(final[column].isna().sum())
            rows.extend(
                [
                    qa_row(
                        "ALTERNATIVE_VEHICLE_QA",
                        "valid count",
                        variable=column,
                        value=valid_count,
                        denominator=len(final),
                        share=valid_count / len(final),
                    ),
                    qa_row(
                        "ALTERNATIVE_VEHICLE_QA",
                        "missing count",
                        variable=column,
                        value=missing_count,
                        denominator=len(final),
                        share=missing_count / len(final),
                    ),
                ]
            )
            valid_values = numeric(final[column]).dropna()
            for category in [1, 2, 3]:
                count = int(valid_values.eq(category).sum())
                rows.append(
                    qa_row(
                        "ALTERNATIVE_VEHICLE_QA",
                        "valid category distribution",
                        variable=column,
                        value=category,
                        denominator=valid_count,
                        share=(count / valid_count) if valid_count else 0.0,
                        non_missing_count=count,
                        details="Share denominator is the alternative's valid values.",
                    )
                )
        pair_complete = final[[f"{feature}_1", f"{feature}_2"]].notna().all(axis=1)
        rows.append(
            qa_row(
                "ALTERNATIVE_VEHICLE_QA",
                "pair complete count",
                variable=feature,
                value=int(pair_complete.sum()),
                denominator=len(final),
                share=float(pair_complete.mean()),
                details=f"Both {feature}_1 and {feature}_2 are valid.",
            )
        )

    explanatory = [column for column in final_columns if column not in ESTIMATION_COLUMNS]
    rows.extend(feature_profile_row("FEATURE_QA", column, final[column]) for column in explanatory)

    for decision in canonicalization:
        rows.append(
            qa_row(
                "CANONICALIZATION",
                "canonical representation",
                variable=decision["final"],
                value=decision["selected"],
                details=f"Dropped: {decision['dropped']}. Reason: {decision['reason']}",
            )
        )

    omitted = {
        "HOME_CHARGING": (merged["_OMIT_HOME_CHARGING"], "structurally unavailable for most strict rows; household A_LADEN valid codes 1--4 only"),
        "DRIVING_EXPERIENCE_YEARS": (merged["_OMIT_DRIVING_EXPERIENCE_YEARS"], "valid only when documented P_FSJAHR and reporting year permit a non-negative derivation"),
        "P_ARB_ENTF": (merged["_OMIT_P_ARB_ENTF"], "structurally unavailable for most strict rows after documented work-distance design codes are removed"),
        "hoff1": (merged["_OMIT_HOFF1"], "structurally unavailable for most strict rows after module/non-employed/design codes are removed"),
        "hind_opnv_anz": (merged["_OMIT_HIND_OPNV_ANZ"], "structurally unavailable for most strict rows after module/design codes are removed"),
        "ENGINE_POWER_KW_1": (merged["_OMIT_ENGINE_POWER_KW_1"], "vehicle module/PAPI design codes leave most rows unavailable"),
        "ENGINE_POWER_KW_2": (merged["_OMIT_ENGINE_POWER_KW_2"], "vehicle module/PAPI design codes leave most rows unavailable"),
        "PARKING_1": (merged["_OMIT_PARKING_1"], "vehicle module/PAPI design codes leave most rows unavailable"),
        "PARKING_2": (merged["_OMIT_PARKING_2"], "vehicle module/PAPI design codes leave most rows unavailable"),
    }
    rows.extend(feature_profile_row("OMITTED_FEATURE", name, series, reason) for name, (series, reason) in omitted.items())
    for column in constant_exclusions:
        rows.append(qa_row("OMITTED_FEATURE", "constant candidate excluded", variable=column, details="zero or one unique non-missing value in strict sample"))
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def run_final_assertions(
    backbone: pd.DataFrame,
    merged: pd.DataFrame,
    final: pd.DataFrame,
    vehicle_wide: pd.DataFrame,
) -> None:
    if len(final) != len(backbone):
        raise AssertionError("Output row count differs from the authoritative backbone.")
    if not final["CHOICE_ID"].is_unique:
        raise AssertionError("Final CHOICE_ID is not unique.")
    if set(final["CHOICE_ID"]) != set(backbone["CHOICE_ID"]):
        raise AssertionError("Final and input CHOICE_ID universes differ.")
    for column in [*ESTIMATION_COLUMNS, "START_MIN"]:
        final_values = final[column].astype("string").reset_index(drop=True)
        input_values = backbone[column].astype("string").reset_index(drop=True)
        if not final_values.equals(input_values):
            raise AssertionError(f"Accepted backbone column changed: {column}")

    forbidden = []
    for column in final.columns:
        if column.startswith(("LEG_", "EXCL_", "VALID_")):
            forbidden.append(column)
        if column.endswith(("_MATCH_FOUND", "_COMPLETE_FLAG", "_VALID_FLAG", "_UNKNOWN_FLAG", "_CONFLICT_FLAG", "_UNRESOLVED_FLAG")):
            forbidden.append(column)
        if column in FORBIDDEN_EXPLICIT:
            forbidden.append(column)
    if forbidden:
        raise AssertionError(f"Forbidden modelling columns remain: {sorted(set(forbidden))}")
    regiostar = [column for column in final.columns if column.startswith("RegioStaR")]
    if regiostar != ["RegioStaR4"]:
        raise AssertionError(f"Final input must retain only RegioStaR4, found: {regiostar}")

    source_vehicle = vehicle_wide.set_index("H_ID")
    for alternative in [1, 2]:
        for feature in ["POWERTRAIN", "SEGMENT", "VEHICLE_AGE", "HOLDER", "STATUS"]:
            column = f"{feature}_{alternative}"
            expected = merged["H_ID"].map(source_vehicle[column])
            observed_values = final[column].astype("string").reset_index(drop=True)
            expected_values = expected.astype("string").reset_index(drop=True)
            if not observed_values.equals(expected_values):
                raise AssertionError(f"Alternative-specific vehicle feature was swapped or changed: {column}")


def save_outputs(paths: Paths, final: pd.DataFrame, manifest: pd.DataFrame, qa: pd.DataFrame) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(paths.model_input, index=False)
    manifest.to_csv(paths.manifest, index=False)
    qa.to_csv(paths.qa, index=False)


def print_vehicle_examples(final: pd.DataFrame) -> None:
    columns = [
        "H_ID",
        "CHOICE",
        "POWERTRAIN_1",
        "POWERTRAIN_2",
        "SEGMENT_1",
        "SEGMENT_2",
        "VEHICLE_AGE_1",
        "VEHICLE_AGE_2",
        "HOLDER_1",
        "HOLDER_2",
        "STATUS_1",
        "STATUS_2",
    ]
    examples = final.loc[
        final["POWERTRAIN_1"].ne(final["POWERTRAIN_2"])
        | final["SEGMENT_1"].ne(final["SEGMENT_2"])
        | final["HOLDER_1"].ne(final["HOLDER_2"])
        | final["STATUS_1"].ne(final["STATUS_2"]),
        columns,
    ].drop_duplicates("H_ID").head(5)
    if examples.empty:
        examples = final[columns].drop_duplicates("H_ID").head(5)
    print("\nALTERNATIVE-SPECIFIC VEHICLE EXAMPLES")
    print(examples.to_string(index=False))


def print_final_summary(
    paths: Paths,
    final: pd.DataFrame,
    additive_comparison: dict[str, object],
) -> None:
    print("\nADDITIVE HOLDER / STATUS UPDATE")
    print(f"\nrows before: {additive_comparison['rows_before']:,}")
    print(f"rows after: {additive_comparison['rows_after']:,}")
    print(f"unique choices before: {additive_comparison['unique_choice_id_before']:,}")
    print(f"unique choices after: {additive_comparison['unique_choice_id_after']:,}")
    print(f"\ncolumns before: {additive_comparison['columns_before']:,}")
    print(f"columns after: {additive_comparison['columns_after']:,}")
    for feature in ["HOLDER", "STATUS"]:
        print(f"\n{feature}:")
        valid_1 = int(final[f"{feature}_1"].notna().sum())
        valid_2 = int(final[f"{feature}_2"].notna().sum())
        pair_complete = final[[f"{feature}_1", f"{feature}_2"]].notna().all(axis=1)
        print(f"vehicle 1 valid: {valid_1:,}")
        print(f"vehicle 2 valid: {valid_2:,}")
        print(f"both vehicles valid: {int(pair_complete.sum()):,}")
        print(f"both-valid share: {pair_complete.mean():.2%}")
    print(f"\nExisting-column equality check: {additive_comparison['existing_column_equality']}")
    print("\nOUTPUTS UPDATED:")
    print(paths.model_input.resolve())
    print(paths.manifest.resolve())
    print(paths.qa.resolve())
    print(
        "\nNo accepted reconstruction, tour logic, sample-selection logic, or existing "
        "model-input feature was changed. Only HOLDER_1/2 and STATUS_1/2 were added."
    )


def main() -> None:
    paths = resolve_paths()
    inputs = load_inputs(paths)
    backbone_invariants = validate_backbone(inputs["backbone"])
    canonicalization = canonicalize_candidate_registry(paths.candidate_registry)

    tour = select_tour_features(inputs["backbone"])
    trip = select_trip_context_features(inputs["trips"])
    household = select_household_features(inputs["households"], inputs["persons"])
    person = select_person_features(inputs["persons"])
    strict_households = set(inputs["backbone"]["H_ID"])
    vehicle = prepare_vehicle_features(inputs["vehicles"], strict_households)

    merged, merge_metrics = merge_features(
        inputs["backbone"], tour, trip, household, person, vehicle
    )
    validate_missing_codes(merged)
    final_columns, constant_exclusions = drop_constant_features(merged)
    final = merged[final_columns].copy()
    additive_comparison = validate_additive_update(inputs["previous_model"], final)
    manifest = build_manifest(final_columns)
    qa = build_QA_summary(
        inputs,
        merged,
        final,
        final_columns,
        merge_metrics,
        canonicalization,
        constant_exclusions,
        additive_comparison,
    )
    run_final_assertions(backbone_invariants, merged, final, vehicle)
    save_outputs(paths, final, manifest, qa)
    print_vehicle_examples(final)
    print_final_summary(
        paths,
        final,
        additive_comparison,
    )


if __name__ == "__main__":
    main()
