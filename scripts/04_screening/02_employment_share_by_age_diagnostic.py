"""Diagnose employment shares across ordered age groups in the MNL sample.

This is a read-only downstream diagnostic for the person-level Spearman
association between ``alter_gr5`` and ``erwerb`` reported by script 08.  It
does not recompute or replace the mixed-association screening workflow.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import CODEBOOK_PATH, MODEL_INPUT_DIR

MODEL_DIR = MODEL_INPUT_DIR
MODEL_INPUT_PATH = MODEL_DIR / "mnl_vehicle_choice_model_input.csv"
SCREENING_PAIRWISE_PATH = MODEL_DIR / "mixed_association_pairwise.csv"
CODEPLAN_PATH = CODEBOOK_PATH
OUTPUT_DIR = MODEL_DIR / "diagnose"
TABLE_PATH = OUTPUT_DIR / "employment_share_by_age_group.csv"
FIGURE_PATH = OUTPUT_DIR / "employment_share_by_age_group.png"

PERSON_METADATA: dict[str, str] = {
    "HP_SEX": "nominal",
    "alter_gr5": "ordinal",
    "erwerb": "binary",
    "P_VPED": "binary",
    "P_VRAD": "binary",
    "carsharing": "binary",
    "mobein": "binary",
}

RHO_TOLERANCE = 1e-12


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_read_only_files() -> dict[Path, str]:
    paths = [MODEL_INPUT_PATH, SCREENING_PAIRWISE_PATH, CODEPLAN_PATH]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required existing input/metadata file(s) not found: "
            + ", ".join(str(path) for path in missing)
        )
    return {path: file_sha256(path) for path in paths}


def assert_read_only_files_unchanged(snapshots: dict[Path, str]) -> None:
    changed = [path for path, digest in snapshots.items() if file_sha256(path) != digest]
    if changed:
        raise AssertionError(
            "A read-only canonical input, screening result, or metadata file changed "
            f"during the diagnostic: {', '.join(str(path) for path in changed)}"
        )


def nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def read_canonical_input() -> pd.DataFrame:
    return pd.read_csv(
        MODEL_INPUT_PATH,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )


def validate_canonical_input(canonical: pd.DataFrame) -> None:
    required = ["CHOICE_ID", "H_ID", "HP_ID", *PERSON_METADATA]
    missing = [column for column in required if column not in canonical.columns]
    if missing:
        raise ValueError(f"Canonical model input is missing required columns: {missing}")
    if not nonblank(canonical["CHOICE_ID"]).all():
        raise ValueError("Canonical CHOICE_ID must be non-missing.")
    if not canonical["CHOICE_ID"].is_unique:
        raise ValueError("Canonical CHOICE_ID must be unique.")
    for column in ["H_ID", "HP_ID"]:
        if not nonblank(canonical[column]).all():
            raise ValueError(f"Canonical {column} must be non-missing.")


def build_person_table(canonical: pd.DataFrame) -> pd.DataFrame:
    """Reproduce script 08's invariance checks and one-row-per-HP_ID table."""
    contradictory_variables: list[str] = []
    for variable in PERSON_METADATA:
        per_id = canonical.groupby("HP_ID", sort=False)[variable].nunique(dropna=False)
        if per_id.gt(1).any():
            contradictory_variables.append(variable)

    hp_household_counts = canonical.groupby("HP_ID", sort=False)["H_ID"].nunique(
        dropna=False
    )
    mapping_bad = hp_household_counts.loc[hp_household_counts.ne(1)]
    if contradictory_variables or not mapping_bad.empty:
        raise ValueError(
            "Person-level invariance failed; no first/last deduplication was performed. "
            f"Contradictory variables: {contradictory_variables}; "
            f"HP_ID -> H_ID contradictions: {len(mapping_bad)}."
        )

    columns = ["HP_ID", "H_ID", *PERSON_METADATA]
    persons = canonical[columns].drop_duplicates("HP_ID").reset_index(drop=True)
    if not persons["HP_ID"].is_unique:
        raise AssertionError("Duplicated person IDs remain after person-level construction.")
    return persons


def numeric_working(
    series: pd.Series,
    variable: str,
    declared_type: str,
) -> pd.Series:
    """Match script 08's numeric conversion and invalid-value treatment exactly."""
    text = series.astype("string")
    blank = text.isna() | text.str.strip().eq("")
    values = pd.to_numeric(text.where(~blank), errors="coerce")
    invalid = ~blank & (values.isna() | ~np.isfinite(values))
    if declared_type in {"count", "ordinal", "binary"}:
        invalid |= values.notna() & values.ne(values.round())
    if invalid.any():
        print(f"\nNON-NUMERIC OR NON-INTEGER CANONICAL VALUES: {variable}")
        print(series.loc[invalid].value_counts(dropna=False).head(10).to_string())
        raise ValueError(
            f"Non-blank canonical values for {variable} cannot be converted as "
            f"declared {declared_type}."
        )
    return values.astype(float)


def build_pairwise_valid_sample(persons: pd.DataFrame) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "HP_ID": persons["HP_ID"],
            "alter_gr5": numeric_working(persons["alter_gr5"], "alter_gr5", "ordinal"),
            "erwerb": numeric_working(persons["erwerb"], "erwerb", "binary"),
        }
    )
    complete = working["alter_gr5"].notna() & working["erwerb"].notna()
    pairwise = working.loc[complete].copy().reset_index(drop=True)

    if not pairwise["HP_ID"].is_unique:
        raise AssertionError("Pairwise-valid person IDs are not unique.")
    if not pairwise["erwerb"].isin([0.0, 1.0]).all():
        invalid = sorted(pairwise.loc[~pairwise["erwerb"].isin([0.0, 1.0]), "erwerb"].unique())
        raise AssertionError(
            "Only canonical 0/1 erwerb values may enter the denominator; "
            f"found {invalid}."
        )
    return pairwise


def read_age_group_labels() -> dict[int, str]:
    """Read the authoritative labels from the existing MiD code-plan workbook."""
    codeplan = pd.read_excel(CODEPLAN_PATH, sheet_name="Personen", header=1, dtype=str)
    required = ["Variable", "Wert", "Wertelabel"]
    missing = [column for column in required if column not in codeplan.columns]
    if missing:
        raise ValueError(f"MiD code plan is missing expected columns: {missing}")

    variable = codeplan["Variable"].ffill()
    rows = codeplan.loc[variable.eq("alter_gr5"), ["Wert", "Wertelabel"]].copy()
    rows["age_group_code"] = pd.to_numeric(rows["Wert"], errors="raise").astype(int)
    rows["age_group_label"] = rows["Wertelabel"].astype("string").str.strip()
    if rows.empty or rows["age_group_code"].duplicated().any():
        raise AssertionError("MiD code plan has missing or duplicated alter_gr5 codes.")
    if rows["age_group_label"].isna().any() or rows["age_group_label"].eq("").any():
        raise AssertionError("MiD code plan has a blank alter_gr5 value label.")
    return dict(zip(rows["age_group_code"], rows["age_group_label"], strict=True))


def spearman_coefficient(x: pd.Series, y: pd.Series) -> float:
    """Use the same average-rank plus Pearson implementation as script 08."""
    ranked_x = stats.rankdata(x.to_numpy(dtype=float), method="average")
    ranked_y = stats.rankdata(y.to_numpy(dtype=float), method="average")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def read_reference_screening_result() -> tuple[float, int]:
    screening = pd.read_csv(SCREENING_PAIRWISE_PATH)
    pair = screening.loc[
        screening["level"].eq("person")
        & (
            screening["variable_1"].eq("alter_gr5")
            & screening["variable_2"].eq("erwerb")
            | screening["variable_1"].eq("erwerb")
            & screening["variable_2"].eq("alter_gr5")
        )
    ]
    if len(pair) != 1:
        raise AssertionError(
            "Expected exactly one person-level alter_gr5 <-> erwerb row in the "
            f"existing screening result, found {len(pair)}."
        )
    row = pair.iloc[0]
    if row["method"] != "Spearman" or row["status"] != "ok":
        raise AssertionError(
            "The existing person-level alter_gr5 <-> erwerb result is not an "
            "evaluable Spearman association."
        )
    return float(row["association"]), int(row["pairwise_n"])


def build_diagnostic_table(
    pairwise: pd.DataFrame,
    age_labels: dict[int, str],
) -> pd.DataFrame:
    observed_codes = pairwise["alter_gr5"].astype(int)
    if not np.array_equal(pairwise["alter_gr5"].to_numpy(), observed_codes.to_numpy()):
        raise AssertionError("alter_gr5 contains a non-integer code after screening conversion.")
    missing_labels = sorted(set(observed_codes) - set(age_labels))
    if missing_labels:
        raise AssertionError(
            "Observed alter_gr5 code(s) lack authoritative code-plan labels: "
            f"{missing_labels}."
        )

    grouped = (
        pairwise.assign(age_group_code=observed_codes)
        .groupby("age_group_code", sort=True)["erwerb"]
        .agg(
            valid_person_n="size",
            employed_n=lambda values: int(values.eq(1.0).sum()),
            non_employed_n=lambda values: int(values.eq(0.0).sum()),
        )
        .reset_index()
    )
    grouped.insert(
        1,
        "age_group_label",
        grouped["age_group_code"].map(age_labels),
    )
    grouped["employed_share"] = grouped["employed_n"] / grouped["valid_person_n"]
    grouped["non_employment_share"] = (
        grouped["non_employed_n"] / grouped["valid_person_n"]
    )
    grouped["employment_percentage"] = 100.0 * grouped["employed_share"]

    if not grouped["age_group_code"].is_monotonic_increasing:
        raise AssertionError("Age groups are not ordered numerically.")
    if grouped["age_group_code"].duplicated().any():
        raise AssertionError("Age-group codes are not unique after grouping.")
    if not grouped["employed_share"].between(0.0, 1.0).all():
        raise AssertionError("At least one employment share lies outside [0, 1].")
    if not grouped["non_employment_share"].between(0.0, 1.0).all():
        raise AssertionError("At least one non-employment share lies outside [0, 1].")
    if not (
        grouped["employed_n"] + grouped["non_employed_n"]
    ).eq(grouped["valid_person_n"]).all():
        raise AssertionError(
            "Employed plus non-employed counts do not equal valid N in every age group."
        )
    if int(grouped["valid_person_n"].sum()) != len(pairwise):
        raise AssertionError("Total grouped N does not equal pairwise-valid N.")
    return grouped


def create_figure(table: pd.DataFrame, overall_share: float) -> None:
    positions = np.arange(len(table))
    percentages = table["employment_percentage"].to_numpy(dtype=float)
    tick_labels = [
        f"{label}\nn={valid_n:,}"
        for label, valid_n in zip(
            table["age_group_label"], table["valid_person_n"], strict=True
        )
    ]

    figure, axis = plt.subplots(figsize=(14.5, 7.8))
    axis.plot(
        positions,
        percentages,
        color="#236192",
        linewidth=2.0,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )
    axis.axhline(
        100.0 * overall_share,
        color="#68737D",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Overall employment share: {overall_share:.1%}",
        zorder=1,
    )
    for position, percentage in zip(positions, percentages, strict=True):
        offset = -16 if percentage >= 94 else 8
        axis.annotate(
            f"{percentage:.1f}%",
            (position, percentage),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=8,
            color="#17324D",
        )

    axis.set_xticks(positions, labels=tick_labels, rotation=38, ha="right")
    axis.set_xlim(-0.45, len(table) - 0.55)
    axis.set_ylim(0.0, 100.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_ylabel("Employment share")
    axis.set_xlabel("Ordered age group (valid person N shown below each label)")
    axis.grid(axis="y", color="#D7DCE0", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False)
    figure.suptitle(
        "Employment share by age group in the MNL analysis sample",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    axis.set_title(
        "Diagnostic for Spearman association between age group and employment",
        fontsize=10.5,
        color="#4F5962",
        pad=12,
    )
    figure.text(
        0.012,
        0.012,
        "Note: `erwerb`: 1 = employed, 0 = not employed. Descriptive diagnostic; no trend line fitted.",
        fontsize=9,
        color="#4F5962",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.30)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def print_table(table: pd.DataFrame) -> None:
    print("\nEMPLOYMENT SHARE BY AGE GROUP")
    print(
        table.to_string(
            index=False,
            formatters={
                "employed_share": lambda value: f"{value:.4f}",
                "non_employment_share": lambda value: f"{value:.4f}",
                "employment_percentage": lambda value: f"{value:.1f}%",
            },
        )
    )


def main() -> None:
    read_only_snapshots = snapshot_read_only_files()
    canonical = read_canonical_input()
    validate_canonical_input(canonical)
    persons = build_person_table(canonical)
    pairwise = build_pairwise_valid_sample(persons)
    age_labels = read_age_group_labels()

    reference_rho, reference_pairwise_n = read_reference_screening_result()
    reproduced_rho = spearman_coefficient(pairwise["alter_gr5"], pairwise["erwerb"])
    if len(pairwise) != reference_pairwise_n:
        raise AssertionError(
            "Reconstructed pairwise-valid N does not match script 08: "
            f"diagnostic N={len(pairwise):,}, script 08 N={reference_pairwise_n:,}."
        )
    if not np.isclose(
        reproduced_rho,
        reference_rho,
        rtol=0.0,
        atol=RHO_TOLERANCE,
    ):
        raise AssertionError(
            "Reconstructed Spearman rho does not match script 08 within the "
            f"required tolerance ({RHO_TOLERANCE:g}): diagnostic "
            f"rho={reproduced_rho:.16f}, script 08 rho={reference_rho:.16f}. "
            "The reconstructed sample or coding is inconsistent; no diagnostic "
            "outputs were written."
        )

    table = build_diagnostic_table(pairwise, age_labels)
    total_valid_n = len(pairwise)
    overall_share = float(pairwise["erwerb"].mean())
    if not 0.0 <= overall_share <= 1.0:
        raise AssertionError("Overall employment share lies outside [0, 1].")

    for path in (TABLE_PATH, FIGURE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    if TABLE_PATH.parent != OUTPUT_DIR or FIGURE_PATH.parent != OUTPUT_DIR:
        raise AssertionError("A diagnostic output path lies outside the diagnose directory.")
    table.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    create_figure(table, overall_share)
    assert_read_only_files_unchanged(read_only_snapshots)

    highest = table.loc[table["employed_share"].idxmax()]
    lowest = table.loc[table["employed_share"].idxmin()]
    print_table(table)
    print("\nQA SUMMARY")
    print(f"- total valid persons: {total_valid_n:,}")
    print(f"- number of age groups: {len(table):,}")
    print(f"- overall employment share: {overall_share:.2%}")
    print(f"- reproduced Spearman rho: {reproduced_rho:.12f}")
    print(
        "- highest employment share: "
        f"{highest['age_group_label']} (code {int(highest['age_group_code'])}, "
        f"{highest['employed_share']:.2%})"
    )
    print(
        "- lowest employment share: "
        f"{lowest['age_group_label']} (code {int(lowest['age_group_code'])}, "
        f"{lowest['employed_share']:.2%})"
    )
    print(f"- CSV: {TABLE_PATH.resolve()}")
    print(f"- PNG: {FIGURE_PATH.resolve()}")
    print(
        "\nThe age-specific employment shares show the empirical pattern "
        "underlying the overall negative rank association."
    )


if __name__ == "__main__":
    main()
