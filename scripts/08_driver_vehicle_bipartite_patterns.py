from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

EVENTS_PATH = DATA_PROCESSED / "vehicle_events_v2.csv"
TRIPS_PATH = DATA_PROCESSED / "trips_home_chain_enriched.csv"
CARS_PATH = DATA_PROCESSED / "cars_selected_raw.csv"
PERSONS_PATH = DATA_PROCESSED / "persons_selected_raw.csv"

EDGE_PATH = DATA_PROCESSED / "driver_vehicle_binary_edges.csv"
DRIVER_DEGREE_PATH = DATA_PROCESSED / "driver_degrees.csv"
VEHICLE_DEGREE_PATH = DATA_PROCESSED / "vehicle_degrees.csv"
HOUSEHOLD_CELL_PATH = DATA_PROCESSED / "household_mapping_cells.csv"

DRIVER_DISTRIBUTION_PATH = RESULTS / "driver_degree_distribution.csv"
VEHICLE_DISTRIBUTION_PATH = RESULTS / "vehicle_degree_distribution.csv"
HOUSEHOLD_CELL_DISTRIBUTION_PATH = RESULTS / "household_mapping_cell_distribution.csv"

DRIVER_FIGURE_PATH = FIGURES / "driver_degree_distribution.png"
VEHICLE_FIGURE_PATH = FIGURES / "vehicle_degree_distribution.png"
HOUSEHOLD_CELL_FIGURE_PATH = FIGURES / "household_mapping_cell_distribution.png"
HOUSEHOLD_COVERAGE_FIGURE_PATH = FIGURES / "household_observation_coverage.png"

ORIGINAL_OUTPUT_PATHS = [
    EDGE_PATH,
    DRIVER_DEGREE_PATH,
    VEHICLE_DEGREE_PATH,
    HOUSEHOLD_CELL_PATH,
    DRIVER_DISTRIBUTION_PATH,
    VEHICLE_DISTRIBUTION_PATH,
    HOUSEHOLD_CELL_DISTRIBUTION_PATH,
    DRIVER_FIGURE_PATH,
    VEHICLE_FIGURE_PATH,
]
NEW_FIGURE_PATHS = [
    HOUSEHOLD_CELL_FIGURE_PATH,
    HOUSEHOLD_COVERAGE_FIGURE_PATH,
]
OUTPUT_PATHS = ORIGINAL_OUTPUT_PATHS + NEW_FIGURE_PATHS

VALID_A_IDS = {1, 2, 3}
EXPECTED_DEPARTURE_ROWS = 139_546
EXPECTED_DEPARTURE_TRIPS = 139_546
EXPECTED_ELIGIBLE_HOUSEHOLDS = 56_793
EXPECTED_CLASSIFIABLE_HOUSEHOLDS = 29_457
EXPECTED_HOUSEHOLDS_WITHOUT_EDGE = 27_336

CELL_LABELS = {
    1: "Cell 1: single driver, one vehicle observed",
    2: "Cell 2: single driver, multiple vehicles",
    3: "Cell 3: multiple drivers, observed one-to-one matching",
    4: "Cell 4: multiple drivers, flexible or mixed mapping",
}
CELL_DISPLAY_LABELS = {
    1: "Cell 1 — Single driver, one vehicle",
    2: "Cell 2 — Single driver, multiple vehicles",
    3: "Cell 3 — Multiple drivers, one-to-one matching",
    4: "Cell 4 — Multiple drivers, flexible/mixed mapping",
}


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def csv_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def flag_equals_one(series: pd.Series) -> pd.Series:
    return numeric(series).fillna(0).eq(1)


def canonical_a_id(series: pd.Series) -> pd.Series:
    values = numeric(series)
    whole_number = values.notna() & values.mod(1).eq(0)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[whole_number] = values.loc[whole_number].astype("Int64").astype("string")
    return out


def make_driver_key(df: pd.DataFrame) -> pd.Series:
    hp = df["HP_ID"] if "HP_ID" in df.columns else pd.Series("", index=df.index, dtype="string")
    pid = df["P_ID"] if "P_ID" in df.columns else pd.Series("", index=df.index, dtype="string")
    hp = hp.astype("string").str.strip()
    pid = pid.astype("string").str.strip()
    h_id = df["H_ID"].astype("string").str.strip()

    key = hp.copy()
    fallback_ok = key.eq("") & h_id.ne("") & pid.ne("")
    key.loc[fallback_ok] = h_id.loc[fallback_ok] + "__P_ID_" + pid.loc[fallback_ok]
    key.loc[key.eq("")] = pd.NA
    return key


def choose_p_fs_pkw(values: pd.Series) -> str:
    text = values.astype("string").str.strip()
    if numeric(text).eq(1).any():
        return "1"
    nonmissing = text[text.ne("")]
    if nonmissing.empty:
        return ""
    return str(nonmissing.iloc[0])


def assert_no_outputs_exist(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output file(s):\n{joined}")


def output_state() -> str:
    existing_original = [path for path in ORIGINAL_OUTPUT_PATHS if path.exists()]
    if len(existing_original) == len(ORIGINAL_OUTPUT_PATHS):
        return "original_outputs_exist"
    if not existing_original:
        return "fresh"

    missing = [path for path in ORIGINAL_OUTPUT_PATHS if not path.exists()]
    joined_existing = "\n".join(str(path) for path in existing_original)
    joined_missing = "\n".join(str(path) for path in missing)
    raise FileExistsError(
        "Found a partial original-output set. Refusing to overwrite or regenerate outputs.\n"
        f"Existing original outputs:\n{joined_existing}\n"
        f"Missing original outputs:\n{joined_missing}"
    )


def ensure_output_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def eligible_households_from_trips() -> pd.Index:
    required = ["H_ID", "complete_fleet_baseline_flag", "topcoded_multicar_household_flag"]
    require_columns(csv_columns(TRIPS_PATH), required, "trip")
    trips = read_csv_strings(TRIPS_PATH, usecols=required)
    eligible = trips.loc[
        flag_equals_one(trips["complete_fleet_baseline_flag"])
        & ~flag_equals_one(trips["topcoded_multicar_household_flag"]),
        "H_ID",
    ]
    eligible = eligible[eligible.ne("")]
    return pd.Index(sorted(eligible.drop_duplicates()))


def vehicle_roster(eligible_hids: pd.Index) -> pd.DataFrame:
    required = ["H_ID", "A_ID"]
    require_columns(csv_columns(CARS_PATH), required, "vehicle")
    cars = read_csv_strings(CARS_PATH, usecols=required)
    cars["A_ID"] = canonical_a_id(cars["A_ID"])
    vehicles = (
        cars.loc[cars["H_ID"].isin(eligible_hids) & cars["A_ID"].isin({"1", "2", "3"}), ["H_ID", "A_ID"]]
        .drop_duplicates(["H_ID", "A_ID"])
        .sort_values(["H_ID", "A_ID"])
        .reset_index(drop=True)
    )

    vehicle_counts = vehicles.groupby("H_ID", sort=False).size()
    bad_hids = vehicle_counts[vehicle_counts.ne(2)].index.tolist()
    missing_hids = sorted(set(eligible_hids) - set(vehicle_counts.index))
    if bad_hids or missing_hids:
        examples = (bad_hids + missing_hids)[:10]
        raise AssertionError(
            "Every complete-fleet household must have exactly two recorded vehicle nodes. "
            f"Problem household examples: {examples}"
        )
    assert vehicle_counts.eq(2).all(), "Every complete-fleet household must have exactly two recorded vehicles."
    return vehicles


def observed_binary_edges(eligible_hids: pd.Index, vehicles: pd.DataFrame) -> pd.DataFrame:
    required = [
        "TRIP_ID",
        "H_ID",
        "HP_ID",
        "P_ID",
        "CHOSEN_A_ID",
        "event_type",
        "complete_fleet_baseline_flag",
    ]
    require_columns(csv_columns(EVENTS_PATH), required, "vehicle event")
    events = read_csv_strings(EVENTS_PATH, usecols=required)
    departures = events.loc[events["event_type"].eq("DEPARTURE")].copy()

    departure_rows = len(departures)
    departure_trips = departures["TRIP_ID"].nunique()
    if departure_rows != EXPECTED_DEPARTURE_ROWS or departure_trips != EXPECTED_DEPARTURE_TRIPS:
        print(
            "WARNING: DEPARTURE reference counts differ from expected "
            f"({EXPECTED_DEPARTURE_ROWS:,} rows, {EXPECTED_DEPARTURE_TRIPS:,} unique TRIP_ID). "
            f"Found {departure_rows:,} rows and {departure_trips:,} unique TRIP_ID."
        )
    else:
        print(
            "Validated DEPARTURE reference counts: "
            f"{departure_rows:,} rows and {departure_trips:,} unique TRIP_ID."
        )

    assert departures["event_type"].eq("DEPARTURE").all(), "Only DEPARTURE rows may be used to construct edges."

    source = departures.loc[
        departures["H_ID"].isin(eligible_hids) & flag_equals_one(departures["complete_fleet_baseline_flag"])
    ].copy()
    source["DRIVER_KEY"] = make_driver_key(source)
    source["A_ID"] = canonical_a_id(source["CHOSEN_A_ID"])

    missing_driver = source["DRIVER_KEY"].isna().sum()
    missing_vehicle = source["A_ID"].isna().sum()
    if missing_driver or missing_vehicle:
        print(
            "WARNING: Dropping unidentifiable departure rows before edge aggregation: "
            f"{missing_driver:,} without DRIVER_KEY, {missing_vehicle:,} without valid CHOSEN_A_ID."
        )
    source = source.loc[source["DRIVER_KEY"].notna() & source["A_ID"].isin({"1", "2", "3"})].copy()

    before_roster = len(source)
    source = source.merge(vehicles, on=["H_ID", "A_ID"], how="inner")
    dropped_not_roster = before_roster - len(source)
    if dropped_not_roster:
        print(
            "WARNING: Dropping departure rows whose CHOSEN_A_ID is not in the recorded household vehicle roster: "
            f"{dropped_not_roster:,}."
        )

    assert source["event_type"].eq("DEPARTURE").all(), "Only DEPARTURE rows may be retained before aggregation."
    assert source["TRIP_ID"].ne("").all(), "Retained departure rows must have non-missing TRIP_ID."
    duplicate_trip_ids = source["TRIP_ID"].duplicated(keep=False)
    assert not duplicate_trip_ids.any(), "Each retained TRIP_ID must appear once before edge aggregation."

    edges = (
        source[["H_ID", "DRIVER_KEY", "A_ID"]]
        .drop_duplicates()
        .sort_values(["H_ID", "DRIVER_KEY", "A_ID"])
        .reset_index(drop=True)
    )
    assert not edges.duplicated(["H_ID", "DRIVER_KEY", "A_ID"]).any(), (
        "Binary edge table contains duplicate H_ID + DRIVER_KEY + A_ID combinations."
    )
    return edges


def driver_roster_and_degrees(eligible_hids: pd.Index, edges: pd.DataFrame) -> pd.DataFrame:
    person_columns = csv_columns(PERSONS_PATH)
    require_columns(person_columns, ["H_ID", "P_FS_PKW"], "person")
    if "HP_ID" not in person_columns and "P_ID" not in person_columns:
        raise KeyError("Person file must contain HP_ID and/or P_ID to build DRIVER_KEY.")

    usecols = ["H_ID", "P_FS_PKW"] + [col for col in ["HP_ID", "P_ID"] if col in person_columns]
    persons = read_csv_strings(PERSONS_PATH, usecols=usecols)
    persons = persons.loc[persons["H_ID"].isin(eligible_hids)].copy()
    persons["DRIVER_KEY"] = make_driver_key(persons)

    missing_licensed_key = (flag_equals_one(persons["P_FS_PKW"]) & persons["DRIVER_KEY"].isna()).sum()
    if missing_licensed_key:
        print(
            "WARNING: Licensed person rows without HP_ID/P_ID cannot enter the driver roster: "
            f"{missing_licensed_key:,}."
        )
    persons = persons.loc[persons["DRIVER_KEY"].notna()].copy()

    person_nodes = (
        persons.groupby(["H_ID", "DRIVER_KEY"], as_index=False)
        .agg(P_FS_PKW=("P_FS_PKW", choose_p_fs_pkw))
        .sort_values(["H_ID", "DRIVER_KEY"])
    )
    licensed_nodes = person_nodes.loc[flag_equals_one(person_nodes["P_FS_PKW"]), ["H_ID", "DRIVER_KEY", "P_FS_PKW"]]

    observed_nodes = edges[["H_ID", "DRIVER_KEY"]].drop_duplicates()
    observed_nodes = observed_nodes.merge(person_nodes, on=["H_ID", "DRIVER_KEY"], how="left")
    observed_nodes["P_FS_PKW"] = observed_nodes["P_FS_PKW"].fillna("")

    roster = (
        pd.concat([licensed_nodes, observed_nodes], ignore_index=True)
        .drop_duplicates(["H_ID", "DRIVER_KEY"])
        .sort_values(["H_ID", "DRIVER_KEY"])
        .reset_index(drop=True)
    )

    degree = edges.groupby(["H_ID", "DRIVER_KEY"], as_index=False)["A_ID"].nunique()
    degree = degree.rename(columns={"A_ID": "DRIVER_DEGREE"})
    roster = roster.merge(degree, on=["H_ID", "DRIVER_KEY"], how="left")
    roster["DRIVER_DEGREE"] = roster["DRIVER_DEGREE"].fillna(0).astype(int)

    observed_degree = roster.merge(observed_nodes[["H_ID", "DRIVER_KEY"]], on=["H_ID", "DRIVER_KEY"], how="inner")
    assert observed_degree["DRIVER_DEGREE"].ge(1).all(), "Every observed driver must have DRIVER_DEGREE >= 1."
    assert set(roster["DRIVER_DEGREE"].unique()).issubset({0, 1, 2}), (
        "DRIVER_DEGREE must be in {0, 1, 2} in the complete two-car sample."
    )
    return roster[["H_ID", "DRIVER_KEY", "P_FS_PKW", "DRIVER_DEGREE"]]


def vehicle_degrees(vehicles: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    degree = edges.groupby(["H_ID", "A_ID"], as_index=False)["DRIVER_KEY"].nunique()
    degree = degree.rename(columns={"DRIVER_KEY": "VEHICLE_DEGREE"})
    out = vehicles.merge(degree, on=["H_ID", "A_ID"], how="left")
    out["VEHICLE_DEGREE"] = out["VEHICLE_DEGREE"].fillna(0).astype(int)
    return out[["H_ID", "A_ID", "VEHICLE_DEGREE"]].sort_values(["H_ID", "A_ID"]).reset_index(drop=True)


def household_mapping_cells(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "H_ID",
                "N_OBSERVED_DRIVERS",
                "N_USED_VEHICLES",
                "MAX_DRIVER_DEGREE",
                "MAX_VEHICLE_DEGREE",
                "CELL_ID",
                "CELL_LABEL",
            ]
        )

    driver_degree = edges.groupby(["H_ID", "DRIVER_KEY"], as_index=False)["A_ID"].nunique()
    driver_degree = driver_degree.rename(columns={"A_ID": "DRIVER_DEGREE"})
    vehicle_degree = edges.groupby(["H_ID", "A_ID"], as_index=False)["DRIVER_KEY"].nunique()
    vehicle_degree = vehicle_degree.rename(columns={"DRIVER_KEY": "VEHICLE_DEGREE"})

    stats = (
        edges.groupby("H_ID", as_index=False)
        .agg(N_OBSERVED_DRIVERS=("DRIVER_KEY", "nunique"), N_USED_VEHICLES=("A_ID", "nunique"))
        .merge(driver_degree.groupby("H_ID", as_index=False)["DRIVER_DEGREE"].max(), on="H_ID", how="left")
        .merge(vehicle_degree.groupby("H_ID", as_index=False)["VEHICLE_DEGREE"].max(), on="H_ID", how="left")
        .rename(columns={"DRIVER_DEGREE": "MAX_DRIVER_DEGREE", "VEHICLE_DEGREE": "MAX_VEHICLE_DEGREE"})
    )

    conditions = [
        stats["N_OBSERVED_DRIVERS"].eq(1) & stats["MAX_DRIVER_DEGREE"].eq(1),
        stats["N_OBSERVED_DRIVERS"].eq(1) & stats["MAX_DRIVER_DEGREE"].ge(2),
        stats["N_OBSERVED_DRIVERS"].ge(2)
        & stats["MAX_DRIVER_DEGREE"].eq(1)
        & stats["MAX_VEHICLE_DEGREE"].eq(1),
        stats["N_OBSERVED_DRIVERS"].ge(2)
        & (stats["MAX_DRIVER_DEGREE"].ge(2) | stats["MAX_VEHICLE_DEGREE"].ge(2)),
    ]
    membership_count = sum(condition.astype(int) for condition in conditions)
    assert membership_count.eq(1).all(), "Every classifiable household must belong to exactly one cell."

    stats["CELL_ID"] = pd.NA
    for cell_id, condition in enumerate(conditions, start=1):
        stats.loc[condition, "CELL_ID"] = cell_id
    stats["CELL_ID"] = stats["CELL_ID"].astype(int)
    stats["CELL_LABEL"] = stats["CELL_ID"].map(CELL_LABELS)

    cell3 = stats["CELL_ID"].eq(3)
    assert (
        stats.loc[cell3, "MAX_DRIVER_DEGREE"].eq(1).all()
        and stats.loc[cell3, "MAX_VEHICLE_DEGREE"].eq(1).all()
    ), "Cell 3 households must have both maximum degrees equal to 1."
    cell4 = stats["CELL_ID"].eq(4)
    assert (
        stats.loc[cell4, "MAX_DRIVER_DEGREE"].gt(1)
        | stats.loc[cell4, "MAX_VEHICLE_DEGREE"].gt(1)
    ).all(), "Cell 4 households must have at least one maximum degree greater than 1."

    return stats[
        [
            "H_ID",
            "N_OBSERVED_DRIVERS",
            "N_USED_VEHICLES",
            "MAX_DRIVER_DEGREE",
            "MAX_VEHICLE_DEGREE",
            "CELL_ID",
            "CELL_LABEL",
        ]
    ].sort_values("H_ID").reset_index(drop=True)


def driver_degree_distribution(driver_degrees: pd.DataFrame) -> pd.DataFrame:
    total = len(driver_degrees)
    rows = []
    for degree in [0, 1, 2]:
        count = int(driver_degrees["DRIVER_DEGREE"].eq(degree).sum())
        rows.append({"DRIVER_DEGREE": degree, "N_DRIVERS": count, "SHARE": count / total if total else 0.0})
    return pd.DataFrame(rows)


def vehicle_degree_distribution(vehicle_degree_table: pd.DataFrame) -> pd.DataFrame:
    total = len(vehicle_degree_table)
    rows = []
    groups = [
        ("0", vehicle_degree_table["VEHICLE_DEGREE"].eq(0)),
        ("1", vehicle_degree_table["VEHICLE_DEGREE"].eq(1)),
        ("2+", vehicle_degree_table["VEHICLE_DEGREE"].ge(2)),
    ]
    for label, mask in groups:
        count = int(mask.sum())
        rows.append(
            {"VEHICLE_DEGREE_GROUP": label, "N_VEHICLES": count, "SHARE": count / total if total else 0.0}
        )
    return pd.DataFrame(rows)


def household_cell_distribution(cells: pd.DataFrame) -> pd.DataFrame:
    total = len(cells)
    rows = []
    for cell_id, label in CELL_LABELS.items():
        count = int(cells["CELL_ID"].eq(cell_id).sum()) if total else 0
        rows.append(
            {
                "CELL_ID": cell_id,
                "CELL_LABEL": label,
                "N_HOUSEHOLDS": count,
                "SHARE": count / total if total else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    assert abs(out["SHARE"].sum() - 1.0) < 1e-12, "Cell 1-4 household shares must sum to 1."
    return out


def save_bar_chart(labels: list[str], shares: pd.Series, title: str, path: Path) -> None:
    percentages = shares.astype(float) * 100
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labels, percentages, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Sample percentage")
    ax.set_ylim(0, max(100, float(percentages.max()) * 1.15 if len(percentages) else 100))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_household_cell_distribution(cells: pd.DataFrame, cell_dist: pd.DataFrame) -> int:
    require_columns(
        cells.columns,
        [
            "H_ID",
            "N_OBSERVED_DRIVERS",
            "N_USED_VEHICLES",
            "MAX_DRIVER_DEGREE",
            "MAX_VEHICLE_DEGREE",
            "CELL_ID",
            "CELL_LABEL",
        ],
        "household mapping cell",
    )
    require_columns(
        cell_dist.columns,
        ["CELL_ID", "CELL_LABEL", "N_HOUSEHOLDS", "SHARE"],
        "household mapping cell distribution",
    )

    dist = cell_dist.copy()
    dist["CELL_ID"] = numeric(dist["CELL_ID"]).astype("Int64")
    dist["N_HOUSEHOLDS"] = numeric(dist["N_HOUSEHOLDS"]).astype("Int64")
    dist["SHARE"] = numeric(dist["SHARE"])
    if dist[["CELL_ID", "N_HOUSEHOLDS", "SHARE"]].isna().any().any():
        raise AssertionError("Household cell distribution contains non-numeric CELL_ID, N_HOUSEHOLDS, or SHARE.")

    cell_ids = sorted(dist["CELL_ID"].astype(int).tolist())
    assert len(dist) == 4, "Exactly four cell categories must be present."
    assert cell_ids == [1, 2, 3, 4], "Cell IDs must be exactly 1, 2, 3, and 4."

    cells_check = cells.copy()
    cells_check["CELL_ID"] = numeric(cells_check["CELL_ID"]).astype("Int64")
    if cells_check["CELL_ID"].isna().any():
        raise AssertionError("Household mapping cells contain non-numeric CELL_ID values.")
    assert sorted(cells_check["CELL_ID"].drop_duplicates().astype(int).tolist()) == [1, 2, 3, 4], (
        "Household mapping cells must contain Cell IDs 1, 2, 3, and 4."
    )

    counts_from_cells = cells_check["CELL_ID"].astype(int).value_counts().sort_index()
    counts_from_dist = dist.set_index("CELL_ID")["N_HOUSEHOLDS"].astype(int).sort_index()
    assert counts_from_cells.to_dict() == counts_from_dist.to_dict(), (
        "Household cell distribution counts must match household_mapping_cells.csv."
    )

    classifiable_households = int(dist["N_HOUSEHOLDS"].sum())
    assert classifiable_households == len(cells), (
        "The four household counts must sum to the number of classifiable households."
    )
    assert abs(float(dist["SHARE"].sum()) - 1.0) < 1e-12, (
        "The four household-cell shares must sum to 1."
    )
    return classifiable_households


def validate_coverage_totals(
    eligible_households: int, classifiable_households: int, households_without_edge: int
) -> pd.DataFrame:
    assert classifiable_households + households_without_edge == eligible_households, (
        "Classifiable households plus households without an observed edge must equal all eligible households."
    )
    rows = [
        {
            "label": "Classifiable household",
            "count": classifiable_households,
            "share": classifiable_households / eligible_households if eligible_households else 0.0,
        },
        {
            "label": "No identifiable observed edge",
            "count": households_without_edge,
            "share": households_without_edge / eligible_households if eligible_households else 0.0,
        },
    ]
    coverage = pd.DataFrame(rows)
    assert abs(float(coverage["share"].sum()) - 1.0) < 1e-12, "The two coverage shares must sum to 1."
    assert int(coverage.loc[coverage["label"].eq("Classifiable household"), "count"].iloc[0]) == classifiable_households, (
        "Classifiable-household count must match the Figure 1 denominator."
    )
    warn_reference_totals(eligible_households, classifiable_households, households_without_edge)
    return coverage


def warn_reference_totals(
    eligible_households: int, classifiable_households: int, households_without_edge: int
) -> None:
    mismatches = []
    if eligible_households != EXPECTED_ELIGIBLE_HOUSEHOLDS:
        mismatches.append(
            f"eligible complete-fleet households expected {EXPECTED_ELIGIBLE_HOUSEHOLDS:,}, "
            f"found {eligible_households:,}"
        )
    if classifiable_households != EXPECTED_CLASSIFIABLE_HOUSEHOLDS:
        mismatches.append(
            f"classifiable households expected {EXPECTED_CLASSIFIABLE_HOUSEHOLDS:,}, "
            f"found {classifiable_households:,}"
        )
    if households_without_edge != EXPECTED_HOUSEHOLDS_WITHOUT_EDGE:
        mismatches.append(
            f"households without an identifiable observed edge expected {EXPECTED_HOUSEHOLDS_WITHOUT_EDGE:,}, "
            f"found {households_without_edge:,}"
        )
    if mismatches:
        print("WARNING: Household coverage reference totals differ from expected:")
        for mismatch in mismatches:
            print(f"- {mismatch}")


def save_household_cell_figure(cell_dist: pd.DataFrame, classifiable_households: int) -> None:
    if HOUSEHOLD_CELL_FIGURE_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {HOUSEHOLD_CELL_FIGURE_PATH}")

    plot_df = cell_dist.copy()
    plot_df["CELL_ID"] = numeric(plot_df["CELL_ID"]).astype(int)
    plot_df["N_HOUSEHOLDS"] = numeric(plot_df["N_HOUSEHOLDS"]).astype(int)
    plot_df["SHARE"] = numeric(plot_df["SHARE"]).astype(float)
    plot_df = plot_df.sort_values("CELL_ID")
    labels = [CELL_DISPLAY_LABELS[cell_id] for cell_id in plot_df["CELL_ID"]]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    bars = ax.barh(labels, plot_df["SHARE"])
    ax.invert_yaxis()
    ax.set_title(
        "Observed driver–vehicle mapping patterns among classifiable\n"
        "complete two-car households"
    )
    ax.set_xlabel("Share of classifiable households")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="x", linewidth=0.8)
    ax.set_axisbelow(True)
    xmax = min(1.0, max(0.75, float(plot_df["SHARE"].max()) + 0.18))
    ax.set_xlim(0, xmax)

    for bar, share, count in zip(bars, plot_df["SHARE"], plot_df["N_HOUSEHOLDS"]):
        label = f"{share * 100:.1f}%  (n={count:,})"
        ax.text(
            min(float(share) + 0.01, xmax - 0.01),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.text(
        0.01,
        0.02,
        "Denominator: complete-fleet households with at least one identifiable "
        f"driver–vehicle edge (N = {classifiable_households:,})",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.38, right=0.98, bottom=0.18, top=0.84)
    fig.savefig(HOUSEHOLD_CELL_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_household_coverage_figure(coverage: pd.DataFrame) -> None:
    if HOUSEHOLD_COVERAGE_FIGURE_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {HOUSEHOLD_COVERAGE_FIGURE_PATH}")

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    left = 0.0
    for _, row in coverage.iterrows():
        share = float(row["share"])
        count = int(row["count"])
        label = str(row["label"])
        ax.barh([""], [share], left=left, label=label)

        text = f"{share * 100:.1f}%\nn={count:,}"
        midpoint = left + share / 2
        if share >= 0.14:
            ax.text(midpoint, 0, text, ha="center", va="center", fontsize=10)
        else:
            x_text = min(left + share + 0.04, 0.98)
            ax.annotate(
                text,
                xy=(left + share, 0),
                xytext=(x_text, 0.24),
                ha="left",
                va="center",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "linewidth": 0.8},
            )
        left += share

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Share of eligible complete-fleet households")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Observation coverage among complete two-car households")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False)
    fig.text(
        0.06,
        0.04,
        "An unclassifiable household is not necessarily a household without car use; it only lacks "
        "an identifiable recorded-household\n"
        "driver–vehicle edge on the diary day.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.46, top=0.78)
    fig.savefig(HOUSEHOLD_COVERAGE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def create_household_figures(
    cells: pd.DataFrame,
    cell_dist: pd.DataFrame,
    eligible_household_count: int,
) -> tuple[int, int]:
    classifiable_households = validate_household_cell_distribution(cells, cell_dist)
    households_without_edge = eligible_household_count - classifiable_households
    coverage = validate_coverage_totals(
        eligible_household_count, classifiable_households, households_without_edge
    )
    save_household_cell_figure(cell_dist, classifiable_households)
    save_household_coverage_figure(coverage)
    return classifiable_households, households_without_edge


def write_outputs(
    edges: pd.DataFrame,
    drivers: pd.DataFrame,
    vehicles: pd.DataFrame,
    cells: pd.DataFrame,
    driver_dist: pd.DataFrame,
    vehicle_dist: pd.DataFrame,
    cell_dist: pd.DataFrame,
    eligible_household_count: int,
) -> None:
    edges.to_csv(EDGE_PATH, index=False)
    drivers.to_csv(DRIVER_DEGREE_PATH, index=False)
    vehicles.to_csv(VEHICLE_DEGREE_PATH, index=False)
    cells.to_csv(HOUSEHOLD_CELL_PATH, index=False)
    driver_dist.to_csv(DRIVER_DISTRIBUTION_PATH, index=False)
    vehicle_dist.to_csv(VEHICLE_DISTRIBUTION_PATH, index=False)
    cell_dist.to_csv(HOUSEHOLD_CELL_DISTRIBUTION_PATH, index=False)

    save_bar_chart(
        driver_dist["DRIVER_DEGREE"].astype(str).tolist(),
        driver_dist["SHARE"],
        "Driver-degree distribution",
        DRIVER_FIGURE_PATH,
    )
    save_bar_chart(
        vehicle_dist["VEHICLE_DEGREE_GROUP"].astype(str).tolist(),
        vehicle_dist["SHARE"],
        "Vehicle-degree distribution",
        VEHICLE_FIGURE_PATH,
    )
    create_household_figures(cells, cell_dist, eligible_household_count)


def print_distribution(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def create_new_figures_from_existing_outputs(started: float) -> None:
    assert_no_outputs_exist(NEW_FIGURE_PATHS)
    ensure_output_dirs()

    print("EXISTING OUTPUT MODE")
    print("Original CSV and PNG outputs already exist; reading them to create only the two new figures.")
    print(f"Using household cells: {HOUSEHOLD_CELL_PATH.relative_to(ROOT)}")
    print(f"Using household cell distribution: {HOUSEHOLD_CELL_DISTRIBUTION_PATH.relative_to(ROOT)}")

    cells = read_csv_strings(HOUSEHOLD_CELL_PATH)
    cell_dist = read_csv_strings(HOUSEHOLD_CELL_DISTRIBUTION_PATH)
    eligible_household_count = len(eligible_households_from_trips())
    classifiable_households, _households_without_edge = create_household_figures(
        cells, cell_dist, eligible_household_count
    )

    print("\nFINAL SUMMARY")
    print("Created:")
    print(f"{HOUSEHOLD_CELL_FIGURE_PATH.relative_to(ROOT)}")
    print(f"{HOUSEHOLD_COVERAGE_FIGURE_PATH.relative_to(ROOT)}")
    print(f"Four-cell denominator: {classifiable_households:,}")
    print(f"Observation-coverage denominator: {eligible_household_count:,}")
    print(f"\nWrote new figures in {perf_counter() - started:.1f} seconds.")


def main() -> None:
    started = perf_counter()
    state = output_state()
    ensure_output_dirs()
    if state == "original_outputs_exist":
        create_new_figures_from_existing_outputs(started)
        return

    assert_no_outputs_exist(OUTPUT_PATHS)

    print("PERSON-LEVEL INPUT")
    print(f"Using accepted person-level file: {PERSONS_PATH.relative_to(ROOT)}")

    eligible_hids = eligible_households_from_trips()
    vehicles = vehicle_roster(eligible_hids)
    edges = observed_binary_edges(eligible_hids, vehicles)
    drivers = driver_roster_and_degrees(eligible_hids, edges)
    vehicle_degree_table = vehicle_degrees(vehicles, edges)
    cells = household_mapping_cells(edges)

    households_with_edges = pd.Index(edges["H_ID"].drop_duplicates())
    households_without_edges = len(set(eligible_hids) - set(households_with_edges))

    driver_dist = driver_degree_distribution(drivers)
    vehicle_dist = vehicle_degree_distribution(vehicle_degree_table)
    cell_dist = household_cell_distribution(cells)

    assert not edges.duplicated(["H_ID", "DRIVER_KEY", "A_ID"]).any(), (
        "Binary edge table contains duplicate H_ID + DRIVER_KEY + A_ID combinations."
    )
    assert set(drivers["DRIVER_DEGREE"].unique()).issubset({0, 1, 2}), (
        "DRIVER_DEGREE must be in {0, 1, 2} in the complete two-car sample."
    )
    assert abs(cell_dist["SHARE"].sum() - 1.0) < 1e-12, "Cell 1-4 household shares must sum to 1."

    write_outputs(
        edges,
        drivers,
        vehicle_degree_table,
        cells,
        driver_dist,
        vehicle_dist,
        cell_dist,
        len(eligible_hids),
    )

    print("\nFINAL SUMMARY")
    print(f"eligible_complete_fleet_households: {len(eligible_hids):,}")
    print(f"complete_fleet_households_with_observed_driver_vehicle_edge: {len(households_with_edges):,}")
    print(
        "complete_fleet_households_without_observed_driver_vehicle_use: "
        f"{households_without_edges:,}"
    )
    print(f"driver_nodes: {len(drivers):,}")
    print(f"vehicle_nodes: {len(vehicle_degree_table):,}")
    print(f"unique_binary_edges: {len(edges):,}")
    print_distribution("HOUSEHOLD MAPPING CELL DISTRIBUTION", cell_dist)
    print_distribution("DRIVER-DEGREE DISTRIBUTION", driver_dist)
    print_distribution("VEHICLE-DEGREE DISTRIBUTION", vehicle_dist)
    print("\nCreated:")
    print(f"{HOUSEHOLD_CELL_FIGURE_PATH.relative_to(ROOT)}")
    print(f"{HOUSEHOLD_COVERAGE_FIGURE_PATH.relative_to(ROOT)}")
    print(f"Four-cell denominator: {len(households_with_edges):,}")
    print(f"Observation-coverage denominator: {len(eligible_hids):,}")
    print(f"\nWrote outputs in {perf_counter() - started:.1f} seconds.")


if __name__ == "__main__":
    main()
