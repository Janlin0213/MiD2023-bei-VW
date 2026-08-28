"""Plot weighted discrete distributions for Theme 1 licensed-driver resources.

This plotting-only script reads the canonical household backbone to calculate
H_GEW-weighted shares at observed outcome levels. Weighted means and HC3 95%
confidence intervals are read from the existing validated summary tables.
The script does not rebuild upstream data or fit any statistical model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BACKBONE_PATH = (
    ROOT
    / "data_processed"
    / "eda"
    / "statistical_test"
    / "theme1_household_backbone.csv"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "eda"
    / "statistical_tests"
    / "theme1_single_vs_multicar"
    / "01_licensed_driver_resources"
)

LICENSED_SUMMARY_PATH = OUTPUT_DIR / "licensed_drivers_weighted_summary_household_type.csv"
EXACT_RATIO_SUMMARY_PATH = (
    OUTPUT_DIR / "driver_car_ratio_exact_weighted_summary_household_type.csv"
)
FIGURE_STEM = OUTPUT_DIR / "licensed_driver_resources_by_household_type_discrete"
PNG_PATH = FIGURE_STEM.with_suffix(".png")
PDF_PATH = FIGURE_STEM.with_suffix(".pdf")

HOUSEHOLD_TYPE_ORDER = [
    "family_household",
    "young_household",
    "adult_household",
    "senior_household",
]
HOUSEHOLD_TYPE_LABELS = {
    "family_household": "Family",
    "young_household": "Young",
    "adult_household": "Adult",
    "senior_household": "Senior",
}
CAR_ORDER = ["single_car", "multi_car"]
CAR_LABELS = {"single_car": "Single-car", "multi_car": "Multi-car"}
CAR_COLORS = {"single_car": "#4C78A8", "multi_car": "#F58518"}
CAR_OFFSETS = {"single_car": -0.18, "multi_car": 0.18}

# Scatter marker area is expressed in points squared. A small visibility floor
# keeps rare observed levels legible while retaining one common scale in both panels.
DOT_AREA_SCALE = 760.0
DOT_AREA_MIN = 10.0

BACKBONE_COLUMNS = [
    "H_ID",
    "H_GEW",
    "H_ANZAUTO",
    "car_ownership_group",
    "household_type",
    "n_licensed_drivers",
    "license_information_complete_flag",
    "driver_car_ratio_exact",
    "driver_car_ratio_exact_flag",
]
SUMMARY_COLUMNS = [
    "household_type",
    "car_ownership_group",
    "outcome",
    "unweighted_n",
    "weight_sum",
    "weighted_mean",
    "robust_standard_error",
    "ci95_low",
    "ci95_high",
    "analysis_scope",
]


def require_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise KeyError(f"{source} is missing required column(s): {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_weight_mask(weight: pd.Series) -> pd.Series:
    values = numeric(weight)
    return values.notna() & np.isfinite(values) & values.gt(0)


def read_backbone() -> pd.DataFrame:
    if not BACKBONE_PATH.exists():
        raise FileNotFoundError(f"Canonical household backbone not found: {BACKBONE_PATH}")
    header = pd.read_csv(BACKBONE_PATH, nrows=0).columns.tolist()
    require_columns(header, BACKBONE_COLUMNS, "household backbone")
    frame = pd.read_csv(
        BACKBONE_PATH,
        usecols=BACKBONE_COLUMNS,
        dtype={"H_ID": "string"},
        low_memory=False,
    )
    if frame["H_ID"].isna().any() or frame["H_ID"].duplicated().any():
        raise RuntimeError("Canonical backbone must contain one row per nonmissing H_ID.")
    for column in [
        "H_GEW",
        "H_ANZAUTO",
        "n_licensed_drivers",
        "license_information_complete_flag",
        "driver_car_ratio_exact",
        "driver_car_ratio_exact_flag",
    ]:
        frame[column] = numeric(frame[column])
    return frame


def read_summary(path: Path, expected_outcome: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Existing weighted summary not found: {path}. "
            "Run 02_licensed_driver_resources.py before this plotting script."
        )
    summary = pd.read_csv(path)
    require_columns(summary.columns, SUMMARY_COLUMNS, path.name)
    for column in [
        "unweighted_n",
        "weight_sum",
        "weighted_mean",
        "robust_standard_error",
        "ci95_low",
        "ci95_high",
    ]:
        summary[column] = numeric(summary[column])
    unexpected_outcomes = set(summary["outcome"].dropna()).difference([expected_outcome])
    if unexpected_outcomes:
        raise RuntimeError(f"Unexpected outcome values in {path.name}: {sorted(unexpected_outcomes)}")
    return summary


def build_plotting_samples(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = (
        frame["car_ownership_group"].isin(CAR_ORDER)
        & frame["license_information_complete_flag"].eq(1)
        & valid_weight_mask(frame["H_GEW"])
    )

    licensed_values = numeric(frame["n_licensed_drivers"])
    panel_a = frame.loc[
        common & licensed_values.notna() & np.isfinite(licensed_values)
    ].copy()

    ratio_values = numeric(frame["driver_car_ratio_exact"])
    panel_b = frame.loc[
        common
        & frame["driver_car_ratio_exact_flag"].eq(1)
        & frame["H_ANZAUTO"].isin([1, 2])
        & ratio_values.notna()
        & np.isfinite(ratio_values)
    ].copy()

    for panel_name, sample in [("Panel A", panel_a), ("Panel B", panel_b)]:
        unexpected_types = set(sample["household_type"].dropna()).difference(
            HOUSEHOLD_TYPE_ORDER
        )
        if unexpected_types:
            raise RuntimeError(
                f"{panel_name} contains household types outside the fixed plotting order: "
                f"{sorted(unexpected_types)}"
            )
        if sample["household_type"].isna().any():
            raise RuntimeError(f"{panel_name} contains missing household_type values.")

    if panel_b["H_ANZAUTO"].gt(2).any():
        raise AssertionError("Panel B must exclude every 3+ car household.")
    if not panel_b.loc[
        panel_b["car_ownership_group"].eq("single_car"), "H_ANZAUTO"
    ].eq(1).all():
        raise AssertionError("Panel B Single-car observations must have exactly one car.")
    if not panel_b.loc[
        panel_b["car_ownership_group"].eq("multi_car"), "H_ANZAUTO"
    ].eq(2).all():
        raise AssertionError("Panel B Multi-car observations must have exactly two cars.")
    return panel_a, panel_b


def validate_summary_against_sample(
    sample: pd.DataFrame,
    summary: pd.DataFrame,
    outcome: str,
    source_name: str,
) -> pd.DataFrame:
    """Validate and return the eight ordered summary rows used by the figure."""
    rows: list[pd.Series] = []
    for household_type in HOUSEHOLD_TYPE_ORDER:
        for car_group in CAR_ORDER:
            cell = sample.loc[
                sample["household_type"].eq(household_type)
                & sample["car_ownership_group"].eq(car_group)
            ]
            result_row = summary.loc[
                summary["household_type"].eq(household_type)
                & summary["car_ownership_group"].eq(car_group)
            ]
            if cell.empty or len(result_row) != 1:
                raise RuntimeError(
                    f"Expected one nonempty cell and one summary row for "
                    f"{household_type} x {car_group} in {source_name}."
                )
            row = result_row.iloc[0]
            if int(row["unweighted_n"]) != len(cell):
                raise AssertionError(
                    f"Sample n does not match {source_name} for "
                    f"{household_type} x {car_group}."
                )
            weights = numeric(cell["H_GEW"]).to_numpy(dtype=float)
            values = numeric(cell[outcome]).to_numpy(dtype=float)
            direct_weight_sum = float(weights.sum())
            direct_mean = float(np.average(values, weights=weights))
            if not np.isclose(
                direct_weight_sum,
                float(row["weight_sum"]),
                rtol=1e-8,
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Weight sum does not match {source_name} for "
                    f"{household_type} x {car_group}."
                )
            if not np.isclose(
                direct_mean,
                float(row["weighted_mean"]),
                rtol=1e-8,
                atol=1e-9,
            ):
                raise AssertionError(
                    f"Weighted mean does not match {source_name} for "
                    f"{household_type} x {car_group}: {direct_mean} vs "
                    f"{row['weighted_mean']}."
                )
            mean = float(row["weighted_mean"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])
            if not (np.isfinite([mean, ci_low, ci_high]).all() and ci_low <= mean <= ci_high):
                raise AssertionError(
                    f"Invalid existing mean/CI in {source_name} for "
                    f"{household_type} x {car_group}."
                )
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def weighted_discrete_distribution(cell: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Return the H_GEW-weighted share at each observed outcome value."""
    if cell.empty:
        raise RuntimeError(f"Empty plotting cell for {outcome}.")
    distribution = (
        cell.assign(
            _outcome=numeric(cell[outcome]),
            _weight=numeric(cell["H_GEW"]),
        )
        .groupby("_outcome", sort=True, as_index=False)
        .agg(weight_sum=("_weight", "sum"), unweighted_n=("H_ID", "size"))
        .rename(columns={"_outcome": "outcome_value"})
    )
    total_weight = float(distribution["weight_sum"].sum())
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise RuntimeError(f"Invalid cell weight total for {outcome}.")
    distribution["weighted_share"] = distribution["weight_sum"] / total_weight

    observed = np.sort(numeric(cell[outcome]).unique().astype(float))
    plotted = distribution["outcome_value"].to_numpy(dtype=float)
    if not np.array_equal(observed, plotted):
        raise AssertionError(f"Plotted values do not equal observed values for {outcome}.")
    if not np.isclose(distribution["weighted_share"].sum(), 1.0, atol=1e-12):
        raise AssertionError(f"Weighted shares do not sum to one for {outcome}.")
    if (distribution["weighted_share"] <= 0).any():
        raise AssertionError(f"Weighted shares must be positive for {outcome}.")
    return distribution


def ordered_summary_row(
    summary: pd.DataFrame,
    household_type: str,
    car_group: str,
) -> pd.Series:
    row = summary.loc[
        summary["household_type"].eq(household_type)
        & summary["car_ownership_group"].eq(car_group)
    ]
    if len(row) != 1:
        raise AssertionError(
            f"Expected one summary row for {household_type} x {car_group}."
        )
    return row.iloc[0]


def plot_panel(
    ax: plt.Axes,
    sample: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    outcome: str,
    panel_label: str,
    title: str,
    y_label: str,
) -> None:
    base_positions = np.arange(len(HOUSEHOLD_TYPE_ORDER), dtype=float)

    for household_index, household_type in enumerate(HOUSEHOLD_TYPE_ORDER):
        for car_group in CAR_ORDER:
            cell = sample.loc[
                sample["household_type"].eq(household_type)
                & sample["car_ownership_group"].eq(car_group)
            ]
            distribution = weighted_discrete_distribution(cell, outcome)
            center = float(base_positions[household_index] + CAR_OFFSETS[car_group])
            sizes = np.maximum(
                DOT_AREA_MIN,
                DOT_AREA_SCALE * distribution["weighted_share"].to_numpy(dtype=float),
            )
            ax.scatter(
                np.full(len(distribution), center),
                distribution["outcome_value"].to_numpy(dtype=float),
                s=sizes,
                color=CAR_COLORS[car_group],
                alpha=0.34,
                edgecolors=CAR_COLORS[car_group],
                linewidths=0.65,
                zorder=2,
            )

    for car_group in CAR_ORDER:
        positions = base_positions + CAR_OFFSETS[car_group]
        rows = [
            ordered_summary_row(summary, household_type, car_group)
            for household_type in HOUSEHOLD_TYPE_ORDER
        ]
        means = np.array([float(row["weighted_mean"]) for row in rows])
        ci_low = np.array([float(row["ci95_low"]) for row in rows])
        ci_high = np.array([float(row["ci95_high"]) for row in rows])
        errors = np.vstack([means - ci_low, ci_high - means])
        color = CAR_COLORS[car_group]
        ax.plot(
            positions,
            means,
            color=color,
            linewidth=1.25,
            alpha=0.95,
            zorder=4,
        )
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="D",
            markersize=7.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            color=color,
            ecolor=color,
            elinewidth=1.45,
            capsize=3.4,
            capthick=1.25,
            zorder=5,
        )

    maximum = float(numeric(sample[outcome]).max())
    ax.set_title(f"{panel_label}  {title}", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xticks(base_positions, [HOUSEHOLD_TYPE_LABELS[x] for x in HOUSEHOLD_TYPE_ORDER])
    ax.set_xlim(-0.58, len(HOUSEHOLD_TYPE_ORDER) - 0.42)
    ax.set_ylim(-0.22, maximum + 0.32)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)
    if outcome == "n_licensed_drivers":
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))


def create_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    licensed_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.9), constrained_layout=False)

    plot_panel(
        axes[0],
        panel_a,
        licensed_summary,
        outcome="n_licensed_drivers",
        panel_label="A",
        title="Licensed drivers by household type",
        y_label="Number of licensed drivers",
    )
    plot_panel(
        axes[1],
        panel_b,
        ratio_summary,
        outcome="driver_car_ratio_exact",
        panel_label="B",
        title="Licensed drivers per car by household type",
        y_label="Licensed drivers per car",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CAR_COLORS[car_group],
            linewidth=1.25,
            marker="D",
            markersize=7.0,
            markerfacecolor=CAR_COLORS[car_group],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=CAR_LABELS[car_group],
        )
        for car_group in CAR_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    note = (
        "Dot area encodes the H_GEW-weighted share at each observed outcome value "
        "(small visibility floor); diamonds show H_GEW-weighted means and error bars "
        "show 95% HC3 CIs.\n"
        "Lines connect weighted means across household types. All observed outcome "
        "levels are shown; no tail collapsing.\n"
        "Panel A: Multi = 2+ cars. Panel B: Multi = exactly 2 cars; 3+ households excluded."
    )
    fig.text(0.065, 0.025, note, ha="left", va="bottom", fontsize=8.2, color="#333333")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.21, wspace=0.22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=220, facecolor="white")
    fig.savefig(PDF_PATH, facecolor="white")
    plt.close(fig)


def validate_all_distributions(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> None:
    for sample, outcome in [
        (panel_a, "n_licensed_drivers"),
        (panel_b, "driver_car_ratio_exact"),
    ]:
        for household_type in HOUSEHOLD_TYPE_ORDER:
            for car_group in CAR_ORDER:
                cell = sample.loc[
                    sample["household_type"].eq(household_type)
                    & sample["car_ownership_group"].eq(car_group)
                ]
                weighted_discrete_distribution(cell, outcome)


def main() -> None:
    backbone = read_backbone()
    panel_a, panel_b = build_plotting_samples(backbone)
    licensed_summary = read_summary(LICENSED_SUMMARY_PATH, "n_licensed_drivers")
    ratio_summary = read_summary(EXACT_RATIO_SUMMARY_PATH, "driver_car_ratio_exact")

    licensed_summary = validate_summary_against_sample(
        panel_a,
        licensed_summary,
        "n_licensed_drivers",
        LICENSED_SUMMARY_PATH.name,
    )
    ratio_summary = validate_summary_against_sample(
        panel_b,
        ratio_summary,
        "driver_car_ratio_exact",
        EXACT_RATIO_SUMMARY_PATH.name,
    )
    validate_all_distributions(panel_a, panel_b)
    create_figure(panel_a, panel_b, licensed_summary, ratio_summary)

    if not PNG_PATH.exists() or PNG_PATH.stat().st_size == 0:
        raise RuntimeError(f"PNG figure was not created correctly: {PNG_PATH}")
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        raise RuntimeError(f"PDF figure was not created correctly: {PDF_PATH}")

    print("LICENSED-DRIVER HOUSEHOLD-TYPE DISCRETE DISTRIBUTIONS")
    print(f"Panel A households: {len(panel_a):,}")
    print(f"Panel B households: {len(panel_b):,}")
    print(f"Panel B 3+ car households: {panel_b['H_ANZAUTO'].gt(2).sum():,}")
    print("Panel A tail collapsing: no")
    print("Weighted shares: H_GEW within each household-type x car-group cell")
    print("Means/CIs: validated against existing household-type summary tables")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
