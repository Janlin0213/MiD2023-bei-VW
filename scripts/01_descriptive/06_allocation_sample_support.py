"""Describe allocation-candidate support and potential joint-model complexity.

This script is read-only with respect to reconstruction.  It compares the
accepted full-diary-day bipartite classification with strict, sensitivity, and
unified allocation-candidate samples; summarizes repeated allocation support;
and reports the unconstrained 2**K configuration upper bound without creating
joint alternatives or estimating a model.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (
    ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
    ALLOCATION_CANDIDATE_OCCASIONS_PATH,
    ALLOCATION_CANDIDATE_QA_PATH,
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR,
    ALLOCATION_SAMPLE_SUPPORT_OUTPUT_DIR,
    BIPARTITE_DATA_DIR,
    BIPARTITE_OUTPUT_DIR,
    PHASE3_DIR,
)


FULL_DAY_CELLS_PATH = BIPARTITE_DATA_DIR / "household_mapping_cells.csv"
FULL_DAY_CELL_DISTRIBUTION_PATH = (
    BIPARTITE_OUTPUT_DIR / "household_mapping_cell_distribution.csv"
)
CLASSIFIED_MASTER_PATH = PHASE3_DIR / "vehicle_choice_occasions_classified.csv"

SAMPLE_COMPARISON_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR / "allocation_sample_comparison.csv"
)
BRANCH_SUPPORT_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR / "allocation_modelling_branch_support.csv"
)
MULTI_DRIVER_SUPPORT_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR / "multi_driver_joint_support.csv"
)
OCCASION_COUNT_DISTRIBUTION_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR
    / "multi_driver_occasion_count_distribution.csv"
)
JOINT_COMPLEXITY_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR
    / "joint_alternative_complexity_diagnostic.csv"
)
CELL_ATTRITION_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR / "full_to_candidate_cell_attrition.csv"
)
CELL_TRANSITION_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR
    / "full_to_candidate_cell_transition_matrix.csv"
)
CELL_EXCLUSION_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR
    / "full_cell_candidate_exclusion_diagnostics.csv"
)

CANDIDATE_CELL_DISTRIBUTION_FIGURE_PATH = (
    ALLOCATION_SAMPLE_SUPPORT_OUTPUT_DIR
    / "allocation_candidate_cell_distribution.png"
)

CSV_OUTPUT_PATHS = [
    SAMPLE_COMPARISON_PATH,
    BRANCH_SUPPORT_PATH,
    MULTI_DRIVER_SUPPORT_PATH,
    OCCASION_COUNT_DISTRIBUTION_PATH,
    JOINT_COMPLEXITY_PATH,
    CELL_ATTRITION_PATH,
    CELL_TRANSITION_PATH,
    CELL_EXCLUSION_PATH,
]
FIGURE_OUTPUT_PATHS = [
    CANDIDATE_CELL_DISTRIBUTION_FIGURE_PATH,
]

REFERENCE_COUNTS = {
    "full-day classifiable households": 29_457,
    "strict occasions": 30_816,
    "strict households": 23_321,
    "sensitivity occasions": 38_143,
    "sensitivity households": 23_759,
    "candidate occasions": 38_779,
    "candidate households": 23_919,
    "candidate single-active-driver households": 16_986,
    "candidate multi-active-driver households": 6_933,
}

CELL_LABELS = {
    1: "Cell 1: single driver, one vehicle observed",
    2: "Cell 2: single driver, multiple vehicles",
    3: "Cell 3: multiple drivers, observed one-to-one matching",
    4: "Cell 4: multiple drivers, flexible or mixed mapping",
}
COMPLEXITY_NOTE = (
    "2^K is only the unconstrained theoretical upper bound. "
    "Overlapping-tour constraints, vehicle-state consistency, and other "
    "feasibility rules would reduce the actual feasible set."
)

REQUIRED_OCCASION_COLUMNS = [
    "H_ID",
    "DRIVER_KEY",
    "CHOICE_ID",
    "CHOSEN_A_ID",
    "CHOSEN_A_ID_CANONICAL",
    "START_MIN",
    "ARRIVAL_MIN",
    "ALLOCATION_STATE_CLASS",
    "simultaneous_home_departure_flag",
    "IN_STRICT_BASELINE",
    "IN_DECISION_TIMING_SENSITIVITY",
]
REQUIRED_HOUSEHOLD_COLUMNS = [
    "H_ID",
    "N_CANDIDATE_OCCASIONS",
    "N_ACTIVE_DRIVERS",
    "N_USED_VEHICLES",
    "MEAN_OCCASIONS_PER_ACTIVE_DRIVER",
    "MAX_OCCASIONS_PER_DRIVER",
    "N_DRIVERS_WITH_2PLUS_OCCASIONS",
    "AT_LEAST_ONE_REPEATED_DRIVER",
    "AT_LEAST_TWO_REPEATED_DRIVERS",
    "HAS_SIMULTANEOUS_DEPARTURE",
    "N_SIMULTANEOUS_OCCASIONS",
    "SINGLE_ACTIVE_DRIVER_HH",
    "MULTI_ACTIVE_DRIVER_HH",
    "CANDIDATE_CELL_ID",
    "CANDIDATE_CELL_LABEL",
]
HOUSEHOLD_NUMERIC_COLUMNS = [
    column for column in REQUIRED_HOUSEHOLD_COLUMNS if column not in {"H_ID", "CANDIDATE_CELL_LABEL"}
]


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
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


def require_columns(
    columns: Iterable[str], required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def flag_equals_one(series: pd.Series) -> pd.Series:
    return numeric(series).fillna(0).eq(1)


def assert_no_outputs_exist(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output file(s):\n{joined}")


def empirical_quantile(values: pd.Series, quantile: float) -> int:
    if values.empty:
        return 0
    return int(values.quantile(quantile, interpolation="higher"))


def qa_metric_count(qa: pd.DataFrame, metric: str) -> int:
    require_columns(qa.columns, ["metric", "count"], "candidate QA")
    rows = qa.loc[qa["metric"].eq(metric), "count"]
    if len(rows) != 1:
        raise ValueError(f"Expected one candidate QA row for metric {metric!r}; found {len(rows)}.")
    value = numeric(rows).iloc[0]
    if pd.isna(value) or not float(value).is_integer():
        raise ValueError(f"Candidate QA metric {metric!r} is not an integer count.")
    return int(value)


def prepare_inputs(
    occasions: pd.DataFrame,
    households: pd.DataFrame,
    full_cells: pd.DataFrame,
    full_distribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(occasions.columns, REQUIRED_OCCASION_COLUMNS, "candidate occasion")
    require_columns(households.columns, REQUIRED_HOUSEHOLD_COLUMNS, "candidate household")
    require_columns(
        full_cells.columns,
        [
            "H_ID",
            "N_OBSERVED_DRIVERS",
            "N_USED_VEHICLES",
            "MAX_DRIVER_DEGREE",
            "MAX_VEHICLE_DEGREE",
            "CELL_ID",
            "CELL_LABEL",
        ],
        "full-diary-day household Cell",
    )
    require_columns(
        full_distribution.columns,
        ["CELL_ID", "CELL_LABEL", "N_HOUSEHOLDS", "SHARE"],
        "full-diary-day Cell distribution",
    )

    occasions = occasions.copy()
    for column in [
        "simultaneous_home_departure_flag",
        "IN_STRICT_BASELINE",
        "IN_DECISION_TIMING_SENSITIVITY",
    ]:
        values = numeric(occasions[column])
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"Candidate occasion column {column} must be populated and binary.")
        occasions[column] = values.astype(int)
    if occasions["CHOICE_ID"].eq("").any() or occasions["CHOICE_ID"].duplicated().any():
        raise ValueError("Candidate CHOICE_ID must be non-missing and unique.")
    if occasions["H_ID"].eq("").any() or occasions["DRIVER_KEY"].eq("").any():
        raise ValueError("Candidate H_ID and DRIVER_KEY must be populated.")

    households = households.copy()
    for column in HOUSEHOLD_NUMERIC_COLUMNS:
        values = numeric(households[column])
        if values.isna().any():
            raise ValueError(f"Candidate household column {column} must be numeric and populated.")
        households[column] = values
    if households["H_ID"].eq("").any() or households["H_ID"].duplicated().any():
        raise ValueError("Candidate household H_ID must be non-missing and unique.")

    full_cells = full_cells.copy()
    for column in [
        "N_OBSERVED_DRIVERS",
        "N_USED_VEHICLES",
        "MAX_DRIVER_DEGREE",
        "MAX_VEHICLE_DEGREE",
        "CELL_ID",
    ]:
        values = numeric(full_cells[column])
        if values.isna().any():
            raise ValueError(f"Full-diary-day column {column} must be numeric and populated.")
        full_cells[column] = values.astype(int)
    if full_cells["H_ID"].eq("").any() or full_cells["H_ID"].duplicated().any():
        raise ValueError("Full-diary-day H_ID must be non-missing and unique.")

    full_distribution = full_distribution.copy()
    for column in ["CELL_ID", "N_HOUSEHOLDS", "SHARE"]:
        values = numeric(full_distribution[column])
        if values.isna().any():
            raise ValueError(f"Full-diary-day distribution column {column} must be numeric.")
        full_distribution[column] = values
    full_distribution["CELL_ID"] = full_distribution["CELL_ID"].astype(int)
    full_distribution["N_HOUSEHOLDS"] = full_distribution["N_HOUSEHOLDS"].astype(int)
    return occasions, households, full_cells, full_distribution


def build_sample_observed_households(occasions: pd.DataFrame) -> pd.DataFrame:
    if occasions.empty:
        return pd.DataFrame()
    driver_counts = (
        occasions.groupby(["H_ID", "DRIVER_KEY"], sort=False)
        .size()
        .rename("N_OCCASIONS_FOR_DRIVER")
        .reset_index()
    )
    repeated = (
        driver_counts.groupby("H_ID", as_index=False)
        .agg(
            N_DRIVERS_WITH_2PLUS_OCCASIONS=(
                "N_OCCASIONS_FOR_DRIVER", lambda values: int(values.ge(2).sum())
            ),
            MAX_OCCASIONS_PER_DRIVER=("N_OCCASIONS_FOR_DRIVER", "max"),
        )
    )
    households = (
        occasions.groupby("H_ID", as_index=False, sort=False)
        .agg(
            N_ANALYTICAL_OCCASIONS=("CHOICE_ID", "size"),
            N_ACTIVE_DRIVERS=("DRIVER_KEY", "nunique"),
            N_USED_VEHICLES=("CHOSEN_A_ID_CANONICAL", "nunique"),
            N_SIMULTANEOUS_OCCASIONS=("simultaneous_home_departure_flag", "sum"),
        )
        .merge(repeated, on="H_ID", how="left", validate="one_to_one")
    )

    edges = occasions[
        ["H_ID", "DRIVER_KEY", "CHOSEN_A_ID_CANONICAL"]
    ].drop_duplicates()
    driver_degree = (
        edges.groupby(["H_ID", "DRIVER_KEY"])["CHOSEN_A_ID_CANONICAL"]
        .nunique()
        .rename("DRIVER_DEGREE")
        .reset_index()
    )
    vehicle_degree = (
        edges.groupby(["H_ID", "CHOSEN_A_ID_CANONICAL"])["DRIVER_KEY"]
        .nunique()
        .rename("VEHICLE_DEGREE")
        .reset_index()
    )
    degrees = (
        driver_degree.groupby("H_ID", as_index=False)["DRIVER_DEGREE"]
        .max()
        .rename(columns={"DRIVER_DEGREE": "MAX_DRIVER_DEGREE"})
        .merge(
            vehicle_degree.groupby("H_ID", as_index=False)["VEHICLE_DEGREE"]
            .max()
            .rename(columns={"VEHICLE_DEGREE": "MAX_VEHICLE_DEGREE"}),
            on="H_ID",
            how="outer",
            validate="one_to_one",
        )
    )
    households = households.merge(degrees, on="H_ID", how="left", validate="one_to_one")
    conditions = [
        households["N_ACTIVE_DRIVERS"].eq(1) & households["MAX_DRIVER_DEGREE"].eq(1),
        households["N_ACTIVE_DRIVERS"].eq(1) & households["MAX_DRIVER_DEGREE"].ge(2),
        households["N_ACTIVE_DRIVERS"].ge(2)
        & households["MAX_DRIVER_DEGREE"].eq(1)
        & households["MAX_VEHICLE_DEGREE"].eq(1),
        households["N_ACTIVE_DRIVERS"].ge(2)
        & (
            households["MAX_DRIVER_DEGREE"].ge(2)
            | households["MAX_VEHICLE_DEGREE"].ge(2)
        ),
    ]
    membership_count = sum(condition.astype(int) for condition in conditions)
    if not membership_count.eq(1).all():
        raise AssertionError("Every sample-observed household must belong to exactly one Cell.")
    households["SAMPLE_OBSERVED_CELL_ID"] = pd.NA
    for cell_id, condition in enumerate(conditions, start=1):
        households.loc[condition, "SAMPLE_OBSERVED_CELL_ID"] = cell_id
    households["SAMPLE_OBSERVED_CELL_ID"] = households[
        "SAMPLE_OBSERVED_CELL_ID"
    ].astype(int)
    households["AT_LEAST_ONE_REPEATED_DRIVER"] = households[
        "N_DRIVERS_WITH_2PLUS_OCCASIONS"
    ].ge(1).astype(int)
    households["AT_LEAST_TWO_REPEATED_DRIVERS"] = households[
        "N_DRIVERS_WITH_2PLUS_OCCASIONS"
    ].ge(2).astype(int)
    households["HAS_SIMULTANEOUS_DEPARTURE"] = households[
        "N_SIMULTANEOUS_OCCASIONS"
    ].gt(0).astype(int)
    return households.sort_values("H_ID", kind="mergesort").reset_index(drop=True)


def sample_summary_from_observed(
    occasions: pd.DataFrame, households: pd.DataFrame
) -> dict[str, object]:
    total_households = len(households)
    occasion_counts = households["N_ANALYTICAL_OCCASIONS"]
    result: dict[str, object] = {
        "Households": total_households,
        "Analytical allocation occasions": len(occasions),
        "Mean analytical occasions per household": float(occasion_counts.mean()),
        "Median analytical occasions per household": float(occasion_counts.median()),
        "Households with >=2 analytical occasions": int(occasion_counts.ge(2).sum()),
        "Households with >=2 active drivers": int(households["N_ACTIVE_DRIVERS"].ge(2).sum()),
        "Households with >=1 driver having >=2 analytical occasions": int(households["AT_LEAST_ONE_REPEATED_DRIVER"].sum()),
        "Households with >=2 drivers each having >=2 analytical occasions": int(households["AT_LEAST_TWO_REPEATED_DRIVERS"].sum()),
        "Single-active-driver households": int(households["N_ACTIVE_DRIVERS"].eq(1).sum()),
        "Multi-active-driver households": int(households["N_ACTIVE_DRIVERS"].ge(2).sum()),
        "Households with simultaneous candidate allocation": int(households["HAS_SIMULTANEOUS_DEPARTURE"].sum()),
        "Simultaneous allocation occasions": int(households["N_SIMULTANEOUS_OCCASIONS"].sum()),
    }
    for cell_id in range(1, 5):
        count = int(households["SAMPLE_OBSERVED_CELL_ID"].eq(cell_id).sum())
        result[f"Cell {cell_id} households"] = count
        result[f"Cell {cell_id} share"] = count / total_households if total_households else 0.0
    return result


def candidate_summary(
    occasions: pd.DataFrame, households: pd.DataFrame
) -> dict[str, object]:
    total_households = len(households)
    counts = households["N_CANDIDATE_OCCASIONS"]
    result: dict[str, object] = {
        "Households": total_households,
        "Analytical allocation occasions": len(occasions),
        "Mean analytical occasions per household": float(counts.mean()),
        "Median analytical occasions per household": float(counts.median()),
        "Households with >=2 analytical occasions": int(counts.ge(2).sum()),
        "Households with >=2 active drivers": int(households["N_ACTIVE_DRIVERS"].ge(2).sum()),
        "Households with >=1 driver having >=2 analytical occasions": int(households["AT_LEAST_ONE_REPEATED_DRIVER"].sum()),
        "Households with >=2 drivers each having >=2 analytical occasions": int(households["AT_LEAST_TWO_REPEATED_DRIVERS"].sum()),
        "Single-active-driver households": int(households["SINGLE_ACTIVE_DRIVER_HH"].sum()),
        "Multi-active-driver households": int(households["MULTI_ACTIVE_DRIVER_HH"].sum()),
        "Households with simultaneous candidate allocation": int(households["HAS_SIMULTANEOUS_DEPARTURE"].sum()),
        "Simultaneous allocation occasions": int(households["N_SIMULTANEOUS_OCCASIONS"].sum()),
    }
    for cell_id in range(1, 5):
        count = int(households["CANDIDATE_CELL_ID"].eq(cell_id).sum())
        result[f"Cell {cell_id} households"] = count
        result[f"Cell {cell_id} share"] = count / total_households if total_households else 0.0
    return result


def full_day_summary(full_cells: pd.DataFrame) -> dict[str, object]:
    not_applicable = "NA"
    total = len(full_cells)
    result: dict[str, object] = {
        "Households": total,
        "Analytical allocation occasions": not_applicable,
        "Mean analytical occasions per household": not_applicable,
        "Median analytical occasions per household": not_applicable,
        "Households with >=2 analytical occasions": not_applicable,
        "Households with >=2 active drivers": int(full_cells["N_OBSERVED_DRIVERS"].ge(2).sum()),
        "Households with >=1 driver having >=2 analytical occasions": not_applicable,
        "Households with >=2 drivers each having >=2 analytical occasions": not_applicable,
        "Single-active-driver households": int(full_cells["N_OBSERVED_DRIVERS"].eq(1).sum()),
        "Multi-active-driver households": int(full_cells["N_OBSERVED_DRIVERS"].ge(2).sum()),
        "Households with simultaneous candidate allocation": not_applicable,
        "Simultaneous allocation occasions": not_applicable,
    }
    for cell_id in range(1, 5):
        count = int(full_cells["CELL_ID"].eq(cell_id).sum())
        result[f"Cell {cell_id} households"] = count
        result[f"Cell {cell_id} share"] = count / total if total else 0.0
    return result


def build_sample_comparison(
    occasions: pd.DataFrame,
    candidate_households: pd.DataFrame,
    full_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict_occasions = occasions.loc[occasions["IN_STRICT_BASELINE"].eq(1)].copy()
    sensitivity_occasions = occasions.loc[
        occasions["IN_DECISION_TIMING_SENSITIVITY"].eq(1)
    ].copy()
    strict_households = build_sample_observed_households(strict_occasions)
    sensitivity_households = build_sample_observed_households(sensitivity_occasions)
    candidate_observed = build_sample_observed_households(occasions)

    authoritative = candidate_households.set_index("H_ID")["CANDIDATE_CELL_ID"].astype(int)
    recomputed = candidate_observed.set_index("H_ID")["SAMPLE_OBSERVED_CELL_ID"].astype(int)
    if not authoritative.sort_index().equals(recomputed.sort_index()):
        raise AssertionError("Recomputed candidate Cells disagree with the authoritative candidate household file.")

    summaries = {
        "FULL_DIARY_DAY": full_day_summary(full_cells),
        "STRICT_BASELINE": sample_summary_from_observed(strict_occasions, strict_households),
        "SENSITIVITY": sample_summary_from_observed(sensitivity_occasions, sensitivity_households),
        "ALLOCATION_CANDIDATE": candidate_summary(occasions, candidate_households),
    }
    measures = [
        "Households",
        "Analytical allocation occasions",
        "Mean analytical occasions per household",
        "Median analytical occasions per household",
        "Households with >=2 analytical occasions",
        "Households with >=2 active drivers",
        "Households with >=1 driver having >=2 analytical occasions",
        "Households with >=2 drivers each having >=2 analytical occasions",
        "Single-active-driver households",
        "Multi-active-driver households",
        "Cell 1 households",
        "Cell 2 households",
        "Cell 3 households",
        "Cell 4 households",
        "Cell 1 share",
        "Cell 2 share",
        "Cell 3 share",
        "Cell 4 share",
        "Households with simultaneous candidate allocation",
        "Simultaneous allocation occasions",
    ]
    rows = [
        {
            "MEASURE": measure,
            **{universe: summary[measure] for universe, summary in summaries.items()},
        }
        for measure in measures
    ]
    rows.append(
        {
            "MEASURE": "Cell classification basis",
            "FULL_DIARY_DAY": "authoritative observed full diary-day mapping",
            "STRICT_BASELINE": "sample-observed mapping within strict occasions",
            "SENSITIVITY": "sample-observed mapping within sensitivity occasions",
            "ALLOCATION_CANDIDATE": "authoritative unified allocation-candidate mapping",
        }
    )
    return pd.DataFrame(rows), strict_households, sensitivity_households


def branch_row(
    sample: str,
    branch: str,
    occasions: pd.DataFrame,
    branch_map: pd.Series,
    branch_value: int,
) -> dict[str, object]:
    selected = occasions.loc[occasions["H_ID"].map(branch_map).eq(branch_value)]
    counts = selected.groupby("H_ID").size()
    return {
        "SAMPLE": sample,
        "CANDIDATE_BRANCH": branch,
        "BRANCH_DEFINITION": "N_ACTIVE_DRIVERS == 1" if branch_value == 0 else "N_ACTIVE_DRIVERS >= 2",
        "N_HOUSEHOLDS": len(counts),
        "N_OCCASIONS": len(selected),
        "MEAN_OCCASIONS_PER_HH": float(counts.mean()) if len(counts) else 0.0,
        "MEDIAN_OCCASIONS_PER_HH": float(counts.median()) if len(counts) else 0.0,
        "P75_OCCASIONS_PER_HH": empirical_quantile(counts, 0.75),
        "P90_OCCASIONS_PER_HH": empirical_quantile(counts, 0.90),
        "P95_OCCASIONS_PER_HH": empirical_quantile(counts, 0.95),
        "MAX_OCCASIONS_PER_HH": int(counts.max()) if len(counts) else 0,
    }


def build_branch_support(
    occasions: pd.DataFrame, candidate_households: pd.DataFrame
) -> pd.DataFrame:
    branch_map = candidate_households.set_index("H_ID")["MULTI_ACTIVE_DRIVER_HH"].astype(int)
    if occasions["H_ID"].map(branch_map).isna().any():
        raise AssertionError("Every candidate occasion must map to a unified candidate branch.")
    samples = {
        "STRICT_BASELINE": occasions.loc[occasions["IN_STRICT_BASELINE"].eq(1)],
        "SENSITIVITY": occasions.loc[occasions["IN_DECISION_TIMING_SENSITIVITY"].eq(1)],
        "ALLOCATION_CANDIDATE": occasions,
    }
    rows = []
    for sample_name, sample in samples.items():
        rows.append(
            branch_row(
                sample_name,
                "SINGLE_ACTIVE_DRIVER",
                sample,
                branch_map,
                0,
            )
        )
        rows.append(
            branch_row(
                sample_name,
                "MULTI_ACTIVE_DRIVER",
                sample,
                branch_map,
                1,
            )
        )
    return pd.DataFrame(rows)


def build_multi_driver_support(
    candidate_households: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    multi = candidate_households.loc[
        candidate_households["MULTI_ACTIVE_DRIVER_HH"].eq(1)
    ].copy()
    counts = multi["N_CANDIDATE_OCCASIONS"].astype(int)
    total_households = len(multi)
    metrics = [
        ("N_MULTI_DRIVER_HOUSEHOLDS", total_households, "candidate-defined multi-active-driver household-days"),
        ("N_MULTI_DRIVER_OCCASIONS", int(counts.sum()), "candidate allocation occasions"),
        ("MEAN_OCCASIONS_PER_HH", float(counts.mean()), "allocation occasions per multi-driver household-day"),
        ("MEDIAN_OCCASIONS_PER_HH", float(counts.median()), "allocation occasions per multi-driver household-day"),
        ("P75_OCCASIONS_PER_HH", empirical_quantile(counts, 0.75), "empirical higher quantile"),
        ("P90_OCCASIONS_PER_HH", empirical_quantile(counts, 0.90), "empirical higher quantile"),
        ("P95_OCCASIONS_PER_HH", empirical_quantile(counts, 0.95), "empirical higher quantile"),
        ("MAX_OCCASIONS_PER_HH", int(counts.max()), "maximum observed K"),
        ("MEAN_ACTIVE_DRIVERS_PER_HH", float(multi["N_ACTIVE_DRIVERS"].mean()), "observed active drivers in candidate"),
        ("MEDIAN_ACTIVE_DRIVERS_PER_HH", float(multi["N_ACTIVE_DRIVERS"].median()), "observed active drivers in candidate"),
        ("MEAN_OCCASIONS_PER_ACTIVE_DRIVER", float(multi["MEAN_OCCASIONS_PER_ACTIVE_DRIVER"].mean()), "household-level mean, averaged across households"),
        ("MEDIAN_OCCASIONS_PER_ACTIVE_DRIVER", float(multi["MEAN_OCCASIONS_PER_ACTIVE_DRIVER"].median()), "median household-level mean"),
    ]
    one_repeated = int(multi["AT_LEAST_ONE_REPEATED_DRIVER"].sum())
    two_repeated = int(multi["AT_LEAST_TWO_REPEATED_DRIVERS"].sum())
    simultaneous = int(multi["HAS_SIMULTANEOUS_DEPARTURE"].sum())
    cell3 = int(multi["CANDIDATE_CELL_ID"].eq(3).sum())
    cell4 = int(multi["CANDIDATE_CELL_ID"].eq(4).sum())
    metrics.extend(
        [
            ("HH_WITH_AT_LEAST_ONE_REPEATED_DRIVER", one_repeated, "empirical repeated allocation opportunities only"),
            ("SHARE_WITH_AT_LEAST_ONE_REPEATED_DRIVER", one_repeated / total_households, "share of multi-driver candidate households"),
            ("HH_WITH_AT_LEAST_TWO_REPEATED_DRIVERS", two_repeated, "empirical repeated allocation opportunities only"),
            ("SHARE_WITH_AT_LEAST_TWO_REPEATED_DRIVERS", two_repeated / total_households, "share of multi-driver candidate households"),
            ("HH_WITH_SIMULTANEOUS_DEPARTURE", simultaneous, "multi-driver candidate households"),
            ("SHARE_WITH_SIMULTANEOUS_DEPARTURE", simultaneous / total_households, "share of multi-driver candidate households"),
            ("CELL3_HOUSEHOLDS", cell3, "candidate-level observed mapping Cell 3"),
            ("CELL3_SHARE", cell3 / total_households, "share of multi-driver candidate households"),
            ("CELL4_HOUSEHOLDS", cell4, "candidate-level observed mapping Cell 4"),
            ("CELL4_SHARE", cell4 / total_households, "share of multi-driver candidate households"),
            ("HH_WITH_2_OCCASIONS", int(counts.eq(2).sum()), "exact K"),
            ("HH_WITH_3_OCCASIONS", int(counts.eq(3).sum()), "exact K"),
            ("HH_WITH_4_OCCASIONS", int(counts.eq(4).sum()), "exact K"),
            ("HH_WITH_5PLUS_OCCASIONS", int(counts.ge(5).sum()), "K >= 5"),
        ]
    )
    support = pd.DataFrame(metrics, columns=["METRIC", "VALUE", "DESCRIPTION"])

    distribution = counts.value_counts().sort_index().rename_axis("K_OCCASIONS").reset_index(name="N_HOUSEHOLDS")
    distribution["SHARE_MULTI_DRIVER_HH"] = distribution["N_HOUSEHOLDS"] / total_households
    distribution["CUMULATIVE_SHARE"] = distribution["SHARE_MULTI_DRIVER_HH"].cumsum()
    return support, distribution


def build_joint_complexity(distribution: pd.DataFrame) -> pd.DataFrame:
    total = int(distribution["N_HOUSEHOLDS"].sum())
    expanded = pd.Series(
        np.repeat(
            distribution["K_OCCASIONS"].astype(int).to_numpy(),
            distribution["N_HOUSEHOLDS"].astype(int).to_numpy(),
        )
    )
    rows: list[dict[str, object]] = []
    for row in distribution.itertuples(index=False):
        k = int(row.K_OCCASIONS)
        rows.append(
            {
                "ROW_TYPE": "EMPIRICAL_DISTRIBUTION",
                "STATISTIC": "",
                "K_OCCASIONS": k,
                "N_HOUSEHOLDS": int(row.N_HOUSEHOLDS),
                "SHARE_MULTI_DRIVER_HH": float(row.N_HOUSEHOLDS / total),
                "RAW_CONFIGURATION_UPPER_BOUND": 2**k,
                "INTERPRETATION": COMPLEXITY_NOTE,
            }
        )
    summary_specs = [
        ("MEDIAN_K", 0.50),
        ("P75_K", 0.75),
        ("P90_K", 0.90),
        ("P95_K", 0.95),
    ]
    for label, quantile in summary_specs:
        k = empirical_quantile(expanded, quantile)
        rows.append(
            {
                "ROW_TYPE": "EMPIRICAL_QUANTILE",
                "STATISTIC": label,
                "K_OCCASIONS": k,
                "N_HOUSEHOLDS": pd.NA,
                "SHARE_MULTI_DRIVER_HH": pd.NA,
                "RAW_CONFIGURATION_UPPER_BOUND": 2**k,
                "INTERPRETATION": COMPLEXITY_NOTE,
            }
        )
    max_k = int(expanded.max())
    rows.append(
        {
            "ROW_TYPE": "EMPIRICAL_QUANTILE",
            "STATISTIC": "MAX_K",
            "K_OCCASIONS": max_k,
            "N_HOUSEHOLDS": pd.NA,
            "SHARE_MULTI_DRIVER_HH": pd.NA,
            "RAW_CONFIGURATION_UPPER_BOUND": 2**max_k,
            "INTERPRETATION": COMPLEXITY_NOTE,
        }
    )
    return pd.DataFrame(rows)


def build_cell_attrition_and_transition(
    full_cells: pd.DataFrame, candidate_households: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_cell = candidate_households[
        ["H_ID", "CANDIDATE_CELL_ID", "CANDIDATE_CELL_LABEL"]
    ].copy()
    merged = full_cells[["H_ID", "CELL_ID", "CELL_LABEL"]].merge(
        candidate_cell,
        on="H_ID",
        how="left",
        validate="one_to_one",
    )
    attrition_rows = []
    transition_rows = []
    for cell_id in range(1, 5):
        group = merged.loc[merged["CELL_ID"].eq(cell_id)]
        entered = group["CANDIDATE_CELL_ID"].notna()
        same = group["CANDIDATE_CELL_ID"].eq(cell_id)
        n_full = len(group)
        n_enter = int(entered.sum())
        n_same = int(same.sum())
        n_changed = int((entered & ~same).sum())
        attrition_rows.append(
            {
                "FULL_CELL_ID": cell_id,
                "FULL_CELL_LABEL": CELL_LABELS[cell_id],
                "N_FULL_HOUSEHOLDS": n_full,
                "N_ENTER_CANDIDATE": n_enter,
                "CANDIDATE_RETENTION_SHARE": n_enter / n_full if n_full else 0.0,
                "N_NOT_IN_CANDIDATE": n_full - n_enter,
                "N_SAME_CANDIDATE_CELL": n_same,
                "SHARE_SAME_CANDIDATE_CELL": n_same / n_enter if n_enter else 0.0,
                "N_CHANGED_CANDIDATE_CELL": n_changed,
                "SHARE_CHANGED_CANDIDATE_CELL": n_changed / n_enter if n_enter else 0.0,
            }
        )
        transition: dict[str, object] = {
            "FULL_CELL": f"FULL_CELL_{cell_id}",
            "FULL_CELL_LABEL": CELL_LABELS[cell_id],
            "N_FULL_HOUSEHOLDS": n_full,
        }
        for candidate_cell_id in range(1, 5):
            count = int(group["CANDIDATE_CELL_ID"].eq(candidate_cell_id).sum())
            transition[f"CANDIDATE_CELL_{candidate_cell_id}_COUNT"] = count
            transition[f"CANDIDATE_CELL_{candidate_cell_id}_ROW_SHARE"] = count / n_full if n_full else 0.0
        not_in = int(group["CANDIDATE_CELL_ID"].isna().sum())
        transition["NOT_IN_CANDIDATE_COUNT"] = not_in
        transition["NOT_IN_CANDIDATE_ROW_SHARE"] = not_in / n_full if n_full else 0.0
        transition_rows.append(transition)
    return pd.DataFrame(attrition_rows), pd.DataFrame(transition_rows), merged


def build_exclusion_diagnostics(
    full_cells: pd.DataFrame,
    candidate_households: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame | None:
    category_flags = {
        "INVALID_OR_UNKNOWN_SNAPSHOT": ["EXCL_INVALID_SNAPSHOT", "EXCL_UNKNOWN_SNAPSHOT"],
        "UNRESOLVED_HOUSEHOLD_TIME_HISTORY": ["EXCL_HOUSEHOLD_TIME_UNRESOLVED"],
        "HOUSEHOLD_TIMELINE_CONFLICT": ["EXCL_HOUSEHOLD_TIMELINE_CONFLICT"],
        "SAME_VEHICLE_OVERLAP": ["EXCL_VEHICLE_OVERLAP"],
        "LOCATION_TRANSITION_CONFLICT": ["EXCL_LOCATION_TRANSITION_CONFLICT"],
        "ZERO_DURATION_HISTORY": ["EXCL_ZERO_DURATION_HISTORY"],
        "OTHER_EXPLICIT_ALLOCATION_CANDIDATE_EXCLUSION": [
            "EXCL_INCOMPLETE_FLEET",
            "EXCL_UNMATCHED_VEHICLE",
            "EXCL_DUPLICATE_IDENTIFIER",
            "EXCL_INCOMPLETE_CHOICE_TRIP_TIME",
            "EXCL_AVAILABILITY_CONFLICT",
        ],
    }
    required = ["H_ID", *sorted({flag for flags in category_flags.values() for flag in flags})]
    missing = sorted(set(required) - set(master.columns))
    if missing:
        print(
            "\nNOTE: Skipping full_cell_candidate_exclusion_diagnostics.csv because "
            f"the classified master lacks explicit columns: {missing}"
        )
        return None

    occasion_flags = pd.DataFrame({"H_ID": master["H_ID"]})
    for category, flags in category_flags.items():
        active = pd.Series(False, index=master.index)
        for flag in flags:
            active |= flag_equals_one(master[flag])
        occasion_flags[category] = active.astype(int)
    household_flags = occasion_flags.groupby("H_ID", as_index=False).max()
    joined = full_cells[["H_ID", "CELL_ID"]].merge(
        household_flags,
        on="H_ID",
        how="left",
        validate="one_to_one",
    )
    for category in category_flags:
        joined[category] = numeric(joined[category]).fillna(0).astype(int)
    master_households = set(master["H_ID"])
    candidate_ids = set(candidate_households["H_ID"])
    joined["NO_CLASSIFIED_HOME_ORIGIN_OCCASION"] = (~joined["H_ID"].isin(master_households)).astype(int)
    joined["NO_VALID_CANDIDATE_OCCASION_REMAINING"] = (~joined["H_ID"].isin(candidate_ids)).astype(int)

    rows = []
    for cell_id in range(1, 5):
        group = joined.loc[joined["CELL_ID"].eq(cell_id)]
        row: dict[str, object] = {
            "FULL_CELL_ID": cell_id,
            "FULL_CELL_LABEL": CELL_LABELS[cell_id],
            "N_FULL_HOUSEHOLDS": len(group),
            "N_NO_VALID_CANDIDATE_OCCASION_REMAINING": int(group["NO_VALID_CANDIDATE_OCCASION_REMAINING"].sum()),
            "N_NO_CLASSIFIED_HOME_ORIGIN_OCCASION": int(group["NO_CLASSIFIED_HOME_ORIGIN_OCCASION"].sum()),
        }
        for category in category_flags:
            row[f"N_{category}"] = int(group[category].sum())
        row["COUNTING_NOTE"] = "Counts are non-exclusive; one household may appear in multiple diagnostic columns."
        rows.append(row)
    return pd.DataFrame(rows)


def validate_results(
    occasions: pd.DataFrame,
    households: pd.DataFrame,
    candidate_qa: pd.DataFrame,
    full_cells: pd.DataFrame,
    full_distribution: pd.DataFrame,
    strict_households: pd.DataFrame,
    sensitivity_households: pd.DataFrame,
    branch_support: pd.DataFrame,
    multi_support: pd.DataFrame,
    occasion_distribution: pd.DataFrame,
    attrition: pd.DataFrame,
    transition: pd.DataFrame,
    transition_source: pd.DataFrame,
) -> None:
    assert occasions["H_ID"].nunique() == len(households), (
        "Candidate household count must equal unique candidate occasion households."
    )
    assert int(households["N_CANDIDATE_OCCASIONS"].sum()) == len(occasions), (
        "Candidate occasion count must equal the authoritative household aggregation."
    )
    assert int(households["CANDIDATE_CELL_ID"].isin([1, 2, 3, 4]).sum()) == len(households), (
        "Candidate Cell IDs must be mutually exhaustive."
    )
    assert sum(int(households["CANDIDATE_CELL_ID"].eq(cell_id).sum()) for cell_id in range(1, 5)) == len(households)
    assert int(households["CANDIDATE_CELL_ID"].isin([1, 2]).sum()) == int(households["SINGLE_ACTIVE_DRIVER_HH"].sum())
    assert int(households["CANDIDATE_CELL_ID"].isin([3, 4]).sum()) == int(households["MULTI_ACTIVE_DRIVER_HH"].sum())

    strict_count = int(occasions["IN_STRICT_BASELINE"].sum())
    sensitivity_count = int(occasions["IN_DECISION_TIMING_SENSITIVITY"].sum())
    assert strict_count == qa_metric_count(candidate_qa, "candidate occasions also in strict baseline")
    assert sensitivity_count == qa_metric_count(candidate_qa, "candidate occasions also in sensitivity sample")
    assert strict_count == len(occasions.loc[occasions["IN_STRICT_BASELINE"].eq(1)])
    assert sensitivity_count == len(occasions.loc[occasions["IN_DECISION_TIMING_SENSITIVITY"].eq(1)])
    assert len(strict_households) == occasions.loc[occasions["IN_STRICT_BASELINE"].eq(1), "H_ID"].nunique()
    assert len(sensitivity_households) == occasions.loc[occasions["IN_DECISION_TIMING_SENSITIVITY"].eq(1), "H_ID"].nunique()

    expected_full = full_cells["CELL_ID"].value_counts().sort_index()
    reported_full = full_distribution.set_index("CELL_ID")["N_HOUSEHOLDS"].astype(int).sort_index()
    assert expected_full.to_dict() == reported_full.to_dict(), (
        "Full-diary-day Cell counts must agree with the authoritative distribution output."
    )
    assert int(transition["N_FULL_HOUSEHOLDS"].sum()) == len(full_cells)
    count_columns = [f"CANDIDATE_CELL_{cell_id}_COUNT" for cell_id in range(1, 5)] + ["NOT_IN_CANDIDATE_COUNT"]
    assert transition[count_columns].sum(axis=1).eq(transition["N_FULL_HOUSEHOLDS"]).all(), (
        "Each transition row must reconcile to its full-diary-day Cell total."
    )
    candidate_in_full = transition_source["CANDIDATE_CELL_ID"].notna().sum()
    assert candidate_in_full == len(households), (
        "Every candidate household must exist in the full-day classified population."
    )
    for cell_id in range(1, 5):
        column_total = int(transition[f"CANDIDATE_CELL_{cell_id}_COUNT"].sum())
        candidate_total = int(households["CANDIDATE_CELL_ID"].eq(cell_id).sum())
        assert column_total == candidate_total, (
            f"Candidate Cell {cell_id} transition column does not reconcile."
        )
    assert int(attrition["N_FULL_HOUSEHOLDS"].sum()) == len(full_cells)

    multi_households = int(households["MULTI_ACTIVE_DRIVER_HH"].sum())
    reported_multi = int(
        numeric(
            multi_support.loc[
                multi_support["METRIC"].eq("N_MULTI_DRIVER_HOUSEHOLDS"), "VALUE"
            ]
        ).iloc[0]
    )
    assert reported_multi == multi_households
    assert int(occasion_distribution["N_HOUSEHOLDS"].sum()) == multi_households
    assert math.isclose(float(occasion_distribution["SHARE_MULTI_DRIVER_HH"].sum()), 1.0, abs_tol=1e-12)
    assert math.isclose(float(occasion_distribution["CUMULATIVE_SHARE"].iloc[-1]), 1.0, abs_tol=1e-12)

    candidate_branch_rows = branch_support.loc[branch_support["SAMPLE"].eq("ALLOCATION_CANDIDATE")]
    assert int(candidate_branch_rows["N_HOUSEHOLDS"].sum()) == len(households)
    assert int(candidate_branch_rows["N_OCCASIONS"].sum()) == len(occasions)


def warn_reference_drift(actual: dict[str, int]) -> None:
    mismatches = [
        (label, actual[label], reference)
        for label, reference in REFERENCE_COUNTS.items()
        if actual[label] != reference
    ]
    if mismatches:
        print("\nWARNING: REFERENCE COUNTS CHANGED")
        for label, observed, reference in mismatches:
            print(f"- {label}: observed {observed:,}; reference only {reference:,}")


FIGURE_DPI = 200
PRIMARY_BLUE = "#4C78A8"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#2A2A2A"
MUTED_TEXT_COLOR = "#5A5A5A"


def style_figure_axis(ax: plt.Axes) -> None:
    """Apply the shared thesis figure treatment without changing chart content."""
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A")
    ax.spines["bottom"].set_color("#8A8A8A")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=9, length=3)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def add_figure_footnote(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.012,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED_TEXT_COLOR,
        style="italic",
    )


def save_candidate_cell_distribution_figure(comparison: pd.DataFrame) -> None:
    candidate_values = comparison.set_index("MEASURE")["ALLOCATION_CANDIDATE"]
    denominator = int(float(candidate_values.loc["Households"]))
    cell_ids = list(range(1, 5))
    counts = [
        int(float(candidate_values.loc[f"Cell {cell_id} households"]))
        for cell_id in cell_ids
    ]
    shares = [count / denominator for count in counts]
    labels = [
        "Cell 1 — Single driver, one vehicle",
        "Cell 2 — Single driver, multiple vehicles",
        "Cell 3 — Multiple drivers, one-to-one matching",
        "Cell 4 — Multiple drivers, flexible/mixed mapping",
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    fig.patch.set_facecolor("#FFFFFF")
    bars = ax.barh(
        labels,
        shares,
        color=PRIMARY_BLUE,
        edgecolor="#FFFFFF",
        linewidth=0.6,
        height=0.68,
    )
    ax.invert_yaxis()
    ax.set_title(
        "Observed driver–vehicle mapping patterns in the\n"
        "allocation-analysis candidate",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=12,
    )
    ax.set_xlabel("Share of candidate households", fontsize=10)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    xmax = min(1.0, max(0.80, max(shares) + 0.18))
    ax.set_xlim(0, xmax)
    style_figure_axis(ax)
    ax.grid(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)

    for bar, share, count in zip(bars, shares, counts):
        ax.text(
            min(share + 0.012, xmax - 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1%}  (n={count:,})",
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    add_figure_footnote(
        fig,
        f"Denominator: candidate households (N = {denominator:,}).",
    )
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.17, top=0.80)
    fig.savefig(
        CANDIDATE_CELL_DISTRIBUTION_FIGURE_PATH,
        dpi=FIGURE_DPI,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def print_summary(
    occasions: pd.DataFrame,
    households: pd.DataFrame,
    full_cells: pd.DataFrame,
    strict_households: pd.DataFrame,
    sensitivity_households: pd.DataFrame,
    branch_support: pd.DataFrame,
    multi_support: pd.DataFrame,
    complexity: pd.DataFrame,
    attrition: pd.DataFrame,
    transition: pd.DataFrame,
) -> None:
    print("\nA. SAMPLE COMPARISON")
    print(f"full-day classifiable households: {len(full_cells):,}")
    print(
        "strict households / occasions: "
        f"{len(strict_households):,} / {int(occasions['IN_STRICT_BASELINE'].sum()):,}"
    )
    print(
        "sensitivity households / occasions: "
        f"{len(sensitivity_households):,} / "
        f"{int(occasions['IN_DECISION_TIMING_SENSITIVITY'].sum()):,}"
    )
    print(f"allocation-candidate households / occasions: {len(households):,} / {len(occasions):,}")

    candidate_branch = branch_support.loc[
        branch_support["SAMPLE"].eq("ALLOCATION_CANDIDATE")
    ].set_index("CANDIDATE_BRANCH")
    print("\nB. CANDIDATE BRANCHES")
    for branch in ["SINGLE_ACTIVE_DRIVER", "MULTI_ACTIVE_DRIVER"]:
        row = candidate_branch.loc[branch]
        print(
            f"{branch}: {int(row.N_HOUSEHOLDS):,} households; "
            f"{int(row.N_OCCASIONS):,} occasions"
        )

    support = multi_support.set_index("METRIC")["VALUE"].map(float)
    repeated_one = int(support["HH_WITH_AT_LEAST_ONE_REPEATED_DRIVER"])
    repeated_two = int(support["HH_WITH_AT_LEAST_TWO_REPEATED_DRIVERS"])
    print("\nC. REPEATED INFORMATION IN MULTI-DRIVER HOUSEHOLDS")
    print(
        f">=1 repeated driver: {repeated_one:,} "
        f"({support['SHARE_WITH_AT_LEAST_ONE_REPEATED_DRIVER']:.2%})"
    )
    print(
        f">=2 repeated drivers: {repeated_two:,} "
        f"({support['SHARE_WITH_AT_LEAST_TWO_REPEATED_DRIVERS']:.2%})"
    )

    complexity_summary = complexity.loc[
        complexity["ROW_TYPE"].eq("EMPIRICAL_QUANTILE")
    ].set_index("STATISTIC")
    print("\nD. JOINT COMPLEXITY DIAGNOSTIC")
    for statistic in ["MEDIAN_K", "P75_K", "P90_K", "P95_K", "MAX_K"]:
        row = complexity_summary.loc[statistic]
        print(
            f"{statistic}: K={int(row.K_OCCASIONS)}; "
            f"2^K upper bound={int(row.RAW_CONFIGURATION_UPPER_BOUND):,}"
        )
    print("2^K is an unconstrained upper bound only; no joint alternatives were enumerated.")

    print("\nE. CELL STRUCTURE")
    for cell_id in range(1, 5):
        full_share = full_cells["CELL_ID"].eq(cell_id).mean()
        candidate_share = households["CANDIDATE_CELL_ID"].eq(cell_id).mean()
        retention = float(
            attrition.loc[
                attrition["FULL_CELL_ID"].eq(cell_id), "CANDIDATE_RETENTION_SHARE"
            ].iloc[0]
        )
        print(
            f"Cell {cell_id}: full-day {full_share:.2%}; candidate {candidate_share:.2%}; "
            f"retention {retention:.2%}"
        )
    full4 = transition.loc[transition["FULL_CELL"].eq("FULL_CELL_4")].iloc[0]
    print(
        "Full-day Cell 4 transition: "
        f"{int(full4.CANDIDATE_CELL_4_COUNT):,} remain Cell 4; "
        f"{int(full4.CANDIDATE_CELL_1_COUNT):,} become Cell 1; "
        f"{int(full4.CANDIDATE_CELL_2_COUNT):,} become Cell 2; "
        f"{int(full4.CANDIDATE_CELL_3_COUNT):,} become Cell 3; "
        f"{int(full4.NOT_IN_CANDIDATE_COUNT):,} do not enter the candidate."
    )

    print("\nF. SIMULTANEOUS SUPPORT")
    print(
        "candidate households with simultaneous departures: "
        f"{int(households['HAS_SIMULTANEOUS_DEPARTURE'].sum()):,}"
    )
    print(
        "candidate simultaneous allocation occasions: "
        f"{int(households['N_SIMULTANEOUS_OCCASIONS'].sum()):,}"
    )


def main() -> None:
    started = perf_counter()
    assert_no_outputs_exist([*CSV_OUTPUT_PATHS, *FIGURE_OUTPUT_PATHS])

    occasions_raw = read_csv_strings(ALLOCATION_CANDIDATE_OCCASIONS_PATH)
    households_raw = read_csv_strings(ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH)
    candidate_qa = read_csv_strings(ALLOCATION_CANDIDATE_QA_PATH)
    full_cells_raw = read_csv_strings(FULL_DAY_CELLS_PATH)
    full_distribution_raw = read_csv_strings(FULL_DAY_CELL_DISTRIBUTION_PATH)
    master = read_csv_strings(CLASSIFIED_MASTER_PATH)
    occasions, households, full_cells, full_distribution = prepare_inputs(
        occasions_raw,
        households_raw,
        full_cells_raw,
        full_distribution_raw,
    )

    comparison, strict_households, sensitivity_households = build_sample_comparison(
        occasions,
        households,
        full_cells,
    )
    branch_support = build_branch_support(occasions, households)
    multi_support, occasion_distribution = build_multi_driver_support(households)
    complexity = build_joint_complexity(occasion_distribution)
    attrition, transition, transition_source = build_cell_attrition_and_transition(
        full_cells,
        households,
    )
    exclusion_diagnostics = build_exclusion_diagnostics(
        full_cells,
        households,
        master,
    )

    validate_results(
        occasions,
        households,
        candidate_qa,
        full_cells,
        full_distribution,
        strict_households,
        sensitivity_households,
        branch_support,
        multi_support,
        occasion_distribution,
        attrition,
        transition,
        transition_source,
    )

    actual_counts = {
        "full-day classifiable households": len(full_cells),
        "strict occasions": int(occasions["IN_STRICT_BASELINE"].sum()),
        "strict households": len(strict_households),
        "sensitivity occasions": int(occasions["IN_DECISION_TIMING_SENSITIVITY"].sum()),
        "sensitivity households": len(sensitivity_households),
        "candidate occasions": len(occasions),
        "candidate households": len(households),
        "candidate single-active-driver households": int(households["SINGLE_ACTIVE_DRIVER_HH"].sum()),
        "candidate multi-active-driver households": int(households["MULTI_ACTIVE_DRIVER_HH"].sum()),
    }
    warn_reference_drift(actual_counts)

    ALLOCATION_SAMPLE_SUPPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ALLOCATION_SAMPLE_SUPPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(SAMPLE_COMPARISON_PATH, index=False)
    branch_support.to_csv(BRANCH_SUPPORT_PATH, index=False)
    multi_support.to_csv(MULTI_DRIVER_SUPPORT_PATH, index=False)
    occasion_distribution.to_csv(OCCASION_COUNT_DISTRIBUTION_PATH, index=False)
    complexity.to_csv(JOINT_COMPLEXITY_PATH, index=False)
    attrition.to_csv(CELL_ATTRITION_PATH, index=False)
    transition.to_csv(CELL_TRANSITION_PATH, index=False)
    if exclusion_diagnostics is not None:
        exclusion_diagnostics.to_csv(CELL_EXCLUSION_PATH, index=False)

    save_candidate_cell_distribution_figure(comparison)

    print("INPUTS")
    for path in [
        ALLOCATION_CANDIDATE_OCCASIONS_PATH,
        ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH,
        ALLOCATION_CANDIDATE_QA_PATH,
        FULL_DAY_CELLS_PATH,
        FULL_DAY_CELL_DISTRIBUTION_PATH,
        CLASSIFIED_MASTER_PATH,
    ]:
        print(path)
    print_summary(
        occasions,
        households,
        full_cells,
        strict_households,
        sensitivity_households,
        branch_support,
        multi_support,
        complexity,
        attrition,
        transition,
    )
    print("\nOUTPUTS")
    for path in [*CSV_OUTPUT_PATHS, *FIGURE_OUTPUT_PATHS]:
        if path.exists():
            print(path)
    print(f"\nCompleted in {perf_counter() - started:.2f}s.")


if __name__ == "__main__":
    main()
