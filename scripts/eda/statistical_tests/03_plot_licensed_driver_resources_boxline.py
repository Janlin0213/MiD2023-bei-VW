"""Create the combined household-type distribution figure for Theme 1.

This is a plotting-only script. It reads the canonical household backbone for
full-sample weighted box statistics and visual scatter subsamples, and reads
the existing household-type summary tables for weighted means and HC3 95%
confidence intervals. It does not rebuild upstream data or fit any model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
EXACT_RATIO_SUMMARY_PATH = OUTPUT_DIR / "driver_car_ratio_exact_weighted_summary_household_type.csv"

FIGURE_STEM = OUTPUT_DIR / "licensed_driver_resources_by_household_type_boxline"
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
CAR_OFFSETS = {"single_car": -0.19, "multi_car": 0.19}

SCATTER_RANDOM_SEED = 20260818
SCATTER_MAX_PER_CELL = 350
SCATTER_JITTER_HALF_WIDTH = 0.085
BOX_WIDTH = 0.30

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


def weighted_quantile(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    q: float,
) -> float:
    """Return the first sorted value whose normalized cumulative weight reaches q.

    This inverse weighted-empirical-CDF definition matches the weighted-quantile
    convention already used by the Theme 1 vehicle-mileage scripts. It is local
    here so this plotting-only script does not import or execute another analysis.
    """
    if not 0 <= q <= 1:
        raise ValueError("q must lie in [0, 1].")
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return np.nan
    order = np.argsort(x[valid], kind="mergesort")
    sorted_x = x[valid][order]
    sorted_w = w[valid][order]
    cumulative = np.cumsum(sorted_w) / sorted_w.sum()
    index = min(np.searchsorted(cumulative, q, side="left"), len(sorted_x) - 1)
    return float(sorted_x[index])


def validate_weighted_quantile() -> None:
    values = np.array([20.0, 0.0, 10.0])
    weights = np.array([1.0, 1.0, 2.0])
    assert weighted_quantile(values, weights, 0.25) == 0.0
    assert weighted_quantile(values, weights, 0.50) == 10.0
    assert weighted_quantile(values, weights, 0.90) == 20.0


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
    valid_weight = valid_weight_mask(frame["H_GEW"])
    common = (
        frame["car_ownership_group"].isin(CAR_ORDER)
        & frame["license_information_complete_flag"].eq(1)
        & valid_weight
    )

    licensed_valid = numeric(frame["n_licensed_drivers"])
    panel_a = frame.loc[
        common & licensed_valid.notna() & np.isfinite(licensed_valid)
    ].copy()

    ratio_valid = numeric(frame["driver_car_ratio_exact"])
    panel_b = frame.loc[
        common
        & frame["driver_car_ratio_exact_flag"].eq(1)
        & frame["H_ANZAUTO"].isin([1, 2])
        & ratio_valid.notna()
        & np.isfinite(ratio_valid)
    ].copy()

    for panel_name, sample in [("Panel A", panel_a), ("Panel B", panel_b)]:
        unexpected_types = set(sample["household_type"].dropna()).difference(HOUSEHOLD_TYPE_ORDER)
        if unexpected_types:
            raise RuntimeError(
                f"{panel_name} contains household types outside the fixed plotting order: "
                f"{sorted(unexpected_types)}"
            )
        if sample["household_type"].isna().any():
            raise RuntimeError(f"{panel_name} contains missing household_type values.")

    if panel_b["H_ANZAUTO"].eq(3).any():
        raise AssertionError("Panel B must exclude every 3+ household.")
    if not panel_b.loc[panel_b["car_ownership_group"].eq("single_car"), "H_ANZAUTO"].eq(1).all():
        raise AssertionError("Panel B Single-car observations must have exactly one car.")
    if not panel_b.loc[panel_b["car_ownership_group"].eq("multi_car"), "H_ANZAUTO"].eq(2).all():
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
            if len(cell) == 0 or len(result_row) != 1:
                raise RuntimeError(
                    f"Expected one nonempty cell and one summary row for "
                    f"{household_type} × {car_group} in {source_name}."
                )
            row = result_row.iloc[0]
            if int(row["unweighted_n"]) != len(cell):
                raise AssertionError(
                    f"Sample n does not match {source_name} for {household_type} × {car_group}."
                )
            direct_mean = float(
                np.average(
                    numeric(cell[outcome]).to_numpy(dtype=float),
                    weights=numeric(cell["H_GEW"]).to_numpy(dtype=float),
                )
            )
            if not np.isclose(
                direct_mean,
                float(row["weighted_mean"]),
                rtol=1e-8,
                atol=1e-9,
            ):
                raise AssertionError(
                    f"Weighted mean does not match {source_name} for "
                    f"{household_type} × {car_group}: {direct_mean} vs "
                    f"{row['weighted_mean']}."
                )
            mean = float(row["weighted_mean"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])
            if not (np.isfinite([mean, ci_low, ci_high]).all() and ci_low <= mean <= ci_high):
                raise AssertionError(
                    f"Invalid existing mean/CI in {source_name} for "
                    f"{household_type} × {car_group}."
                )
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def weighted_box_statistics(cell: pd.DataFrame, outcome: str) -> dict[str, float | list[float]]:
    """Return weighted Q1/median/Q3 and observed 1.5-IQR whiskers.

    Quantiles use the full analytical cell and H_GEW. Whiskers are the most
    extreme observed outcome values inside the weighted-quartile fences.
    """
    values = numeric(cell[outcome]).to_numpy(dtype=float)
    weights = numeric(cell["H_GEW"]).to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.all() or len(values) == 0:
        raise RuntimeError(f"Invalid values/weights in plotting cell for {outcome}.")

    q1 = weighted_quantile(values, weights, 0.25)
    median = weighted_quantile(values, weights, 0.50)
    q3 = weighted_quantile(values, weights, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    if len(inside) == 0:
        raise AssertionError("No observations fall inside the weighted 1.5-IQR fences.")
    stats: dict[str, float | list[float]] = {
        "q1": q1,
        "med": median,
        "q3": q3,
        "whislo": float(inside.min()),
        "whishi": float(inside.max()),
        "fliers": [],
    }
    if not (stats["whislo"] <= q1 <= median <= q3 <= stats["whishi"]):
        raise AssertionError(f"Invalid weighted box ordering for {outcome}: {stats}")
    return stats


def visual_subsample(
    cell: pd.DataFrame,
    outcome: str,
    center: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, jittered points for distribution illustration only."""
    n_display = min(SCATTER_MAX_PER_CELL, len(cell))
    indices = rng.choice(len(cell), size=n_display, replace=False)
    displayed = numeric(cell.iloc[indices][outcome]).to_numpy(dtype=float)
    jitter = rng.uniform(
        -SCATTER_JITTER_HALF_WIDTH,
        SCATTER_JITTER_HALF_WIDTH,
        size=n_display,
    )
    return np.full(n_display, center) + jitter, displayed


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
        raise AssertionError(f"Expected one ordered summary row for {household_type} × {car_group}.")
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
    rng: np.random.Generator,
) -> None:
    base_positions = np.arange(len(HOUSEHOLD_TYPE_ORDER), dtype=float)

    # Boxes use full-sample H_GEW-weighted quartiles. Scatter points are
    # visually subsampled; statistical summaries use the full analytical sample.
    for household_index, household_type in enumerate(HOUSEHOLD_TYPE_ORDER):
        for car_group in CAR_ORDER:
            cell = sample.loc[
                sample["household_type"].eq(household_type)
                & sample["car_ownership_group"].eq(car_group)
            ]
            center = float(base_positions[household_index] + CAR_OFFSETS[car_group])
            color = CAR_COLORS[car_group]
            box_stats = weighted_box_statistics(cell, outcome)
            ax.bxp(
                [box_stats],
                positions=[center],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": to_rgba(color, 0.22),
                    "edgecolor": color,
                    "linewidth": 1.15,
                    "zorder": 1,
                },
                medianprops={"color": "#222222", "linewidth": 1.55, "zorder": 3},
                whiskerprops={"color": color, "linewidth": 1.0, "zorder": 1},
                capprops={"color": color, "linewidth": 1.0, "zorder": 1},
            )
            scatter_x, scatter_y = visual_subsample(cell, outcome, center, rng)
            ax.scatter(
                scatter_x,
                scatter_y,
                s=8,
                color=color,
                alpha=0.11,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

    # Thin lines connect only the existing full-sample weighted means.
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
            linewidth=1.15,
            alpha=0.90,
            zorder=3.5,
        )
        ax.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="D",
            markersize=6.4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.85,
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=3.2,
            capthick=1.2,
            zorder=5,
        )

    ax.set_title(f"{panel_label}  {title}", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xticks(base_positions, [HOUSEHOLD_TYPE_LABELS[x] for x in HOUSEHOLD_TYPE_ORDER])
    ax.set_xlim(-0.62, len(HOUSEHOLD_TYPE_ORDER) - 0.38)
    ax.set_ylim(bottom=-0.12)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if outcome == "n_licensed_drivers":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.5))


def create_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    licensed_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(SCATTER_RANDOM_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.9), constrained_layout=False)

    plot_panel(
        axes[0],
        panel_a,
        licensed_summary,
        outcome="n_licensed_drivers",
        panel_label="A",
        title="Licensed drivers by household type",
        y_label="Number of licensed drivers",
        rng=rng,
    )
    plot_panel(
        axes[1],
        panel_b,
        ratio_summary,
        outcome="driver_car_ratio_exact",
        panel_label="B",
        title="Licensed drivers per car by household type",
        y_label="Licensed drivers per car",
        rng=rng,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CAR_COLORS[car_group],
            linewidth=1.15,
            marker="D",
            markersize=6.4,
            markerfacecolor=CAR_COLORS[car_group],
            markeredgecolor="white",
            markeredgewidth=0.85,
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
        "Boxes show H_GEW-weighted Q1, median, and Q3; whiskers use 1.5×IQR. "
        "Diamonds/lines show H_GEW-weighted means; error bars show 95% HC3 CIs.\n"
        f"Scatter points are a deterministic visual subsample (≤{SCATTER_MAX_PER_CELL} "
        f"per cell; seed {SCATTER_RANDOM_SEED}); box statistics, means, and CIs use "
        "the full analytical samples.\n"
        "Panel A: Multi = 2+ cars. Panel B: Multi = exactly 2 cars; 3+ households excluded."
    )
    fig.text(0.065, 0.025, note, ha="left", va="bottom", fontsize=8.2, color="#333333")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.21, wspace=0.22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=220, facecolor="white")
    fig.savefig(PDF_PATH, facecolor="white")
    plt.close(fig)


def main() -> None:
    validate_weighted_quantile()
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

    # Recheck every plotted box immediately before rendering: its internal
    # line is the weighted median, never the weighted mean.
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
                stats = weighted_box_statistics(cell, outcome)
                expected_median = weighted_quantile(cell[outcome], cell["H_GEW"], 0.50)
                if stats["med"] != expected_median:
                    raise AssertionError("Box center line is not the weighted median.")

    create_figure(panel_a, panel_b, licensed_summary, ratio_summary)
    if not PNG_PATH.exists() or PNG_PATH.stat().st_size == 0:
        raise RuntimeError(f"PNG figure was not created correctly: {PNG_PATH}")
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        raise RuntimeError(f"PDF figure was not created correctly: {PDF_PATH}")

    print("LICENSED-DRIVER HOUSEHOLD-TYPE BOXPLOT")
    print(f"Panel A households: {len(panel_a):,}")
    print(f"Panel B households: {len(panel_b):,}")
    print(f"Panel B 3+ households: {panel_b['H_ANZAUTO'].eq(3).sum():,}")
    print("Boxes: full-sample H_GEW-weighted Q1/median/Q3 with 1.5-IQR whiskers")
    print(
        f"Scatter: deterministic unweighted visual subsample, up to "
        f"{SCATTER_MAX_PER_CELL} per cell, seed {SCATTER_RANDOM_SEED}"
    )
    print("Means/CIs: validated against existing household-type summary tables")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
