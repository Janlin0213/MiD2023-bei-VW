"""Screen canonical MNL predictors for hierarchical mixed-type association.

This is a read-only downstream diagnostic.  It neither ranks predictors
against CHOICE nor modifies the accepted model input or selects variables.
Household, person, and tour/choice-occasion predictors are screened at their
natural analysis units using metadata-driven association measures.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data_processed" / "model_input"
MODEL_INPUT_PATH = MODEL_DIR / "mnl_vehicle_choice_model_input.csv"
PAIRWISE_PATH = MODEL_DIR / "mixed_association_pairwise.csv"
CROSS_LEVEL_PATH = MODEL_DIR / "mixed_association_cross_level.csv"
QA_PATH = MODEL_DIR / "mixed_association_QA.csv"
HOUSEHOLD_HEATMAP_PATH = MODEL_DIR / "mixed_association_household_heatmap.png"
PERSON_HEATMAP_PATH = MODEL_DIR / "mixed_association_person_heatmap.png"
TOUR_HEATMAP_PATH = MODEL_DIR / "mixed_association_tour_heatmap.png"

OUTPUT_PATHS = [
    PAIRWISE_PATH,
    CROSS_LEVEL_PATH,
    QA_PATH,
    HOUSEHOLD_HEATMAP_PATH,
    PERSON_HEATMAP_PATH,
    TOUR_HEATMAP_PATH,
]

NEW_DESTINATION_COLUMNS = [
    "TOUR_WORST_XMSTADT_ZO",
    "TOUR_WORST_QUALI_OPNV_ZO",
]
OBSOLETE_DESTINATION_COLUMNS = ["XMStadt_ZO", "quali_opnv_zo"]

VARIABLE_METADATA: dict[str, dict[str, str]] = {
    "household": {
        "household_type": "nominal",
        "H_MIETE": "nominal",
        "RegioStaR4": "nominal",
        "hhgr_gr": "ordinal",
        "oek_status": "ordinal",
        "XMStadt": "ordinal",
        "quali_opnv": "ordinal",
        "min_bab": "ordinal",
        "min_ozmz": "ordinal",
        "H_CS": "binary",
        "H_ANZPED": "count",
        "H_ANZRAD": "count",
    },
    "person": {
        "HP_SEX": "nominal",
        "alter_gr5": "ordinal",
        "erwerb": "binary",
        "P_VPED": "binary",
        "P_VRAD": "binary",
        "carsharing": "binary",
        "mobein": "binary",
    },
    "tour": {
        "TOUR_DISTANCE_KM": "continuous",
        "TOUR_TRAVEL_TIME_MIN": "continuous",
        "TOUR_N_NONHOME_STOPS": "count",
        "TOUR_N_DISTINCT_PURPOSES": "count",
        "W_ANZBEGL": "count",
        "TOUR_WORST_XMSTADT_ZO": "ordinal",
        "TOUR_WORST_QUALI_OPNV_ZO": "ordinal",
        "TOUR_HAS_WORK": "binary",
        "TOUR_HAS_BUSINESS": "binary",
        "TOUR_HAS_EDUCATION": "binary",
        "TOUR_HAS_SHOPPING": "binary",
        "TOUR_HAS_ERRAND": "binary",
        "TOUR_HAS_LEISURE": "binary",
        "TOUR_HAS_ESCORT": "binary",
        "TOUR_HAS_OTHER_PURPOSE": "binary",
        "HOUSEHOLD_ACCOMPANIED": "binary",
        "DAY_TYPE": "nominal",
        "saison": "nominal",
        "TOUR_FIRST_PURPOSE": "nominal",
        "P_STWETTER": "nominal",
    },
}

LEVEL_ID = {"household": "H_ID", "person": "HP_ID", "tour": "CHOICE_ID"}
LEVEL_TITLES = {
    "household": "Household-level mixed-type pairwise association strength",
    "person": "Person-level mixed-type pairwise association strength",
    "tour": "Tour-level mixed-type pairwise association strength",
}
HEATMAP_PATHS = {
    "household": HOUSEHOLD_HEATMAP_PATH,
    "person": PERSON_HEATMAP_PATH,
    "tour": TOUR_HEATMAP_PATH,
}

PAIRWISE_COLUMNS = [
    "level",
    "variable_1",
    "variable_2",
    "type_1",
    "type_2",
    "method",
    "association",
    "association_strength",
    "directional",
    "pairwise_n",
    "level_n",
    "coverage",
    "status",
    "review_flag",
]
CROSS_LEVEL_COLUMNS = [*PAIRWISE_COLUMNS, "analysis_unit", "conceptual_reason"]
QA_COLUMNS = [
    "section",
    "metric",
    "level",
    "variable",
    "value",
    "denominator",
    "share",
    "non_missing_n",
    "missing_n",
    "missing_share",
    "unique_valid_values",
    "details",
]

NUMERIC_TYPES = {"continuous", "count", "ordinal", "binary"}
ORDERED_TYPES = {"continuous", "count", "ordinal", "binary"}

CROSS_LEVEL_SPECS = [
    {
        "level": "household_person",
        "variable_1": "H_ANZPED",
        "type_1": "count",
        "variable_2": "P_VPED",
        "type_2": "binary",
        "analysis_unit": "unique HP_ID",
        "conceptual_reason": "household pedelec stock vs personal pedelec availability",
    },
    {
        "level": "household_person",
        "variable_1": "H_ANZRAD",
        "type_1": "count",
        "variable_2": "P_VRAD",
        "type_2": "binary",
        "analysis_unit": "unique HP_ID",
        "conceptual_reason": "household bicycle stock vs personal bicycle availability",
    },
    {
        "level": "household_person",
        "variable_1": "H_CS",
        "type_1": "binary",
        "variable_2": "carsharing",
        "type_2": "binary",
        "analysis_unit": "unique HP_ID",
        "conceptual_reason": "household vs personal carsharing membership",
    },
    {
        "level": "residential_tour",
        "variable_1": "XMStadt",
        "type_1": "ordinal",
        "variable_2": "TOUR_WORST_XMSTADT_ZO",
        "type_2": "ordinal",
        "analysis_unit": "unique CHOICE_ID",
        "conceptual_reason": "residential vs worst-observed non-home destination minute-city context",
    },
    {
        "level": "residential_tour",
        "variable_1": "quali_opnv",
        "type_1": "ordinal",
        "variable_2": "TOUR_WORST_QUALI_OPNV_ZO",
        "type_2": "ordinal",
        "analysis_unit": "unique CHOICE_ID",
        "conceptual_reason": "residential vs worst-observed non-home destination PT quality",
    },
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_canonical_input() -> pd.DataFrame:
    if not MODEL_INPUT_PATH.exists():
        raise FileNotFoundError(f"Accepted canonical model input not found: {MODEL_INPUT_PATH}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Required existing output directory not found: {MODEL_DIR}")
    return pd.read_csv(
        MODEL_INPUT_PATH,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )


def screened_columns() -> list[str]:
    return list(
        dict.fromkeys(
            variable
            for metadata in VARIABLE_METADATA.values()
            for variable in metadata
        )
    )


def nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def validate_input(canonical: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = ["CHOICE_ID", "H_ID", "HP_ID", *screened_columns()]
    missing = [column for column in required if column not in canonical.columns]
    obsolete_present = [
        column for column in OBSOLETE_DESTINATION_COLUMNS if column in canonical.columns
    ]
    missing_new = [column for column in NEW_DESTINATION_COLUMNS if column not in canonical.columns]
    if missing or missing_new or obsolete_present:
        raise ValueError(
            "Accepted 06 -> 07 pipeline must be completed before mixed-association screening. "
            f"Missing required columns: {sorted(set([*missing, *missing_new]))}; "
            f"obsolete destination columns still present: {obsolete_present}."
        )
    if not nonblank(canonical["CHOICE_ID"]).all():
        raise ValueError("Canonical CHOICE_ID must be non-missing.")
    if not canonical["CHOICE_ID"].is_unique:
        raise ValueError("Canonical CHOICE_ID must be unique.")
    for column in ["H_ID", "HP_ID"]:
        if not nonblank(canonical[column]).all():
            raise ValueError(f"Canonical {column} must be non-missing.")
    return missing, obsolete_present


def numeric_working(series: pd.Series, variable: str, declared_type: str) -> pd.Series:
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
            f"Non-blank canonical values for {variable} cannot be converted as declared {declared_type}."
        )
    return values.astype(float)


def working_series(series: pd.Series, variable: str, declared_type: str) -> pd.Series:
    if declared_type in NUMERIC_TYPES:
        return numeric_working(series, variable, declared_type)
    text = series.astype("string")
    return text.mask(text.isna() | text.str.strip().eq(""))


def prepare_working_table(
    frame: pd.DataFrame,
    metadata: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            variable: working_series(frame[variable], variable, declared_type)
            for variable, declared_type in metadata.items()
        },
        index=frame.index,
    )


def contradiction_counts(
    canonical: pd.DataFrame,
    id_column: str,
    variables: Iterable[str],
    label: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for variable in variables:
        per_id = canonical.groupby(id_column, sort=False)[variable].nunique(dropna=False)
        contradictory_ids = per_id.loc[per_id.gt(1)].index
        counts[variable] = len(contradictory_ids)
        if len(contradictory_ids):
            sample = canonical.loc[
                canonical[id_column].isin(contradictory_ids[:5]),
                [id_column, variable],
            ].drop_duplicates()
            print(f"\n{label.upper()} INVARIANCE CONTRADICTION: {variable}")
            print(sample.head(15).to_string(index=False))
    return counts


def build_natural_level_tables(
    canonical: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]], int]:
    household_contradictions = contradiction_counts(
        canonical,
        "H_ID",
        VARIABLE_METADATA["household"],
        "household",
    )
    person_contradictions = contradiction_counts(
        canonical,
        "HP_ID",
        VARIABLE_METADATA["person"],
        "person",
    )
    hp_household_counts = canonical.groupby("HP_ID", sort=False)["H_ID"].nunique(dropna=False)
    mapping_bad = hp_household_counts.loc[hp_household_counts.ne(1)]
    if not mapping_bad.empty:
        print("\nHP_ID -> H_ID MAPPING CONTRADICTIONS")
        print(
            canonical.loc[
                canonical["HP_ID"].isin(mapping_bad.index[:10]),
                ["HP_ID", "H_ID"],
            ].drop_duplicates().to_string(index=False)
        )

    contradiction_total = sum(household_contradictions.values()) + sum(
        person_contradictions.values()
    )
    if contradiction_total or len(mapping_bad):
        raise ValueError(
            "Natural-level invariance failed; no first/last deduplication was performed."
        )

    household_columns = ["H_ID", *VARIABLE_METADATA["household"]]
    person_columns = ["HP_ID", "H_ID", *VARIABLE_METADATA["person"]]
    tour_columns = ["CHOICE_ID", *VARIABLE_METADATA["tour"]]
    tables = {
        "household": canonical[household_columns].drop_duplicates("H_ID").reset_index(drop=True),
        "person": canonical[person_columns].drop_duplicates("HP_ID").reset_index(drop=True),
        "tour": canonical[tour_columns].copy().reset_index(drop=True),
    }
    return (
        tables,
        {
            "household": household_contradictions,
            "person": person_contradictions,
        },
        len(mapping_bad),
    )


def expected_method(type_1: str, type_2: str) -> tuple[str, int]:
    if type_1 == "binary" and type_2 == "binary":
        return "Phi", 1
    if type_1 != "nominal" and type_2 != "nominal":
        return "Spearman", 1
    other_type = type_2 if type_1 == "nominal" else type_1
    if other_type in {"continuous", "count"}:
        return "Eta", 0
    return "Cramers_V", 0


def phi_coefficient(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])


def spearman_coefficient(x: pd.Series, y: pd.Series) -> float:
    ranked_x = stats.rankdata(x.to_numpy(dtype=float), method="average")
    ranked_y = stats.rankdata(y.to_numpy(dtype=float), method="average")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def bias_corrected_cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    n = int(contingency.to_numpy().sum())
    if contingency.shape[0] < 2 or contingency.shape[1] < 2 or n <= 1:
        return np.nan
    observed = contingency.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    if np.any(expected <= 0):
        return np.nan
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    phi2 = chi2 / n
    rows, columns = contingency.shape
    phi2_corrected = max(
        0.0,
        phi2 - ((columns - 1) * (rows - 1)) / (n - 1),
    )
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return np.nan
    return float(np.sqrt(phi2_corrected / denominator))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    numeric_values = values.to_numpy(dtype=float)
    overall_mean = float(np.mean(numeric_values))
    denominator = float(np.sum((numeric_values - overall_mean) ** 2))
    if denominator <= 0:
        return np.nan
    grouped = pd.DataFrame(
        {"category": categories.astype("string"), "value": numeric_values}
    ).groupby("category", sort=True)["value"]
    numerator = float(
        sum(len(group) * (float(group.mean()) - overall_mean) ** 2 for _, group in grouped)
    )
    return float(np.sqrt(numerator / denominator))


def dispatch_association(
    x: pd.Series,
    y: pd.Series,
    type_1: str,
    type_2: str,
) -> dict[str, object]:
    method, directional = expected_method(type_1, type_2)
    complete = x.notna() & y.notna()
    pair_x = x.loc[complete]
    pair_y = y.loc[complete]
    pairwise_n = int(complete.sum())
    if pair_x.nunique(dropna=True) < 2 or pair_y.nunique(dropna=True) < 2:
        return {
            "method": method,
            "association": np.nan,
            "association_strength": np.nan,
            "directional": directional,
            "pairwise_n": pairwise_n,
            "status": "insufficient_variation",
        }

    try:
        if method == "Phi":
            association = phi_coefficient(pair_x, pair_y)
        elif method == "Spearman":
            association = spearman_coefficient(pair_x, pair_y)
        elif method == "Cramers_V":
            association = bias_corrected_cramers_v(pair_x, pair_y)
        else:
            if type_1 == "nominal":
                association = correlation_ratio(pair_x, pair_y)
            else:
                association = correlation_ratio(pair_y, pair_x)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        association = np.nan

    if not np.isfinite(association):
        return {
            "method": method,
            "association": np.nan,
            "association_strength": np.nan,
            "directional": directional,
            "pairwise_n": pairwise_n,
            "status": "insufficient_variation",
        }
    if directional:
        association = float(np.clip(association, -1.0, 1.0))
        strength = abs(association)
    else:
        association = float(np.clip(association, 0.0, 1.0))
        strength = association
    return {
        "method": method,
        "association": association,
        "association_strength": strength,
        "directional": directional,
        "pairwise_n": pairwise_n,
        "status": "ok",
    }


def review_flag(strength: float) -> str:
    if pd.isna(strength):
        return "NOT_EVALUABLE"
    if strength >= 0.80:
        return "STRONG_REVIEW"
    if strength >= 0.60:
        return "MODERATE_REVIEW"
    return "NONE"


def pair_record(
    level: str,
    variable_1: str,
    variable_2: str,
    type_1: str,
    type_2: str,
    table: pd.DataFrame,
    level_n: int,
) -> dict[str, object]:
    result = dispatch_association(
        table[variable_1], table[variable_2], type_1, type_2
    )
    coverage = result["pairwise_n"] / level_n if level_n else 0.0
    return {
        "level": level,
        "variable_1": variable_1,
        "variable_2": variable_2,
        "type_1": type_1,
        "type_2": type_2,
        **result,
        "level_n": level_n,
        "coverage": coverage,
        "review_flag": review_flag(result["association_strength"]),
    }


def build_within_level_associations(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    working_tables: dict[str, pd.DataFrame] = {}
    for level, metadata in VARIABLE_METADATA.items():
        working = prepare_working_table(tables[level], metadata)
        working_tables[level] = working
        for variable_1, variable_2 in combinations(metadata, 2):
            records.append(
                pair_record(
                    level,
                    variable_1,
                    variable_2,
                    metadata[variable_1],
                    metadata[variable_2],
                    working,
                    len(tables[level]),
                )
            )
    pairwise = pd.DataFrame(records, columns=PAIRWISE_COLUMNS)
    level_order = pd.Categorical(pairwise["level"], ["household", "person", "tour"], ordered=True)
    pairwise = (
        pairwise.assign(_LEVEL_ORDER=level_order)
        .sort_values(
            ["_LEVEL_ORDER", "association_strength", "variable_1", "variable_2"],
            ascending=[True, False, True, True],
            na_position="last",
            kind="mergesort",
        )
        .drop(columns="_LEVEL_ORDER")
        .reset_index(drop=True)
    )
    return pairwise, working_tables


def build_cross_level_associations(canonical: pd.DataFrame) -> pd.DataFrame:
    person_columns = [
        "HP_ID",
        "H_ANZPED",
        "P_VPED",
        "H_ANZRAD",
        "P_VRAD",
        "H_CS",
        "carsharing",
    ]
    person_source = canonical[person_columns].drop_duplicates("HP_ID").reset_index(drop=True)
    tour_columns = [
        "CHOICE_ID",
        "XMStadt",
        "TOUR_WORST_XMSTADT_ZO",
        "quali_opnv",
        "TOUR_WORST_QUALI_OPNV_ZO",
    ]
    tour_source = canonical[tour_columns].copy().reset_index(drop=True)

    records: list[dict[str, object]] = []
    for spec in CROSS_LEVEL_SPECS:
        source = person_source if spec["analysis_unit"] == "unique HP_ID" else tour_source
        metadata = {
            spec["variable_1"]: spec["type_1"],
            spec["variable_2"]: spec["type_2"],
        }
        working = prepare_working_table(source, metadata)
        record = pair_record(
            spec["level"],
            spec["variable_1"],
            spec["variable_2"],
            spec["type_1"],
            spec["type_2"],
            working,
            len(source),
        )
        record["analysis_unit"] = spec["analysis_unit"]
        record["conceptual_reason"] = spec["conceptual_reason"]
        records.append(record)
    return pd.DataFrame(records, columns=CROSS_LEVEL_COLUMNS)


def compact_signed(value: float) -> str:
    return f"{value:+.2f}".replace("+0.", "+.").replace("-0.", "-.")


def compact_unsigned(value: float) -> str:
    return f"{value:.2f}".replace("0.", ".")


def heatmap_annotation(method: str, association: float) -> str:
    if pd.isna(association):
        return "NA"
    if method == "Spearman":
        return f"rho={compact_signed(float(association))}"
    if method == "Phi":
        return f"phi={compact_signed(float(association))}"
    if method == "Cramers_V":
        return f"V={compact_unsigned(float(association))}"
    return f"eta={compact_unsigned(float(association))}"


def create_heatmap(
    level: str,
    pairwise: pd.DataFrame,
    metadata: dict[str, str],
    path: Path,
) -> None:
    variables = list(metadata)
    n_variables = len(variables)
    strengths = np.full((n_variables, n_variables), np.nan, dtype=float)
    annotations = np.full((n_variables, n_variables), "", dtype=object)
    np.fill_diagonal(strengths, 1.0)
    for index in range(n_variables):
        annotations[index, index] = "1.00"

    positions = {variable: index for index, variable in enumerate(variables)}
    for row in pairwise.loc[pairwise["level"].eq(level)].itertuples(index=False):
        first = positions[row.variable_1]
        second = positions[row.variable_2]
        display_row, display_column = max(first, second), min(first, second)
        strengths[display_row, display_column] = row.association_strength
        annotations[display_row, display_column] = heatmap_annotation(
            row.method, row.association
        )

    upper_triangle = np.triu(np.ones_like(strengths, dtype=bool), k=1)
    masked = np.ma.array(strengths, mask=upper_triangle | np.isnan(strengths))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    figure_size = (
        max(8.0, 3.5 + 0.68 * n_variables),
        max(7.0, 3.0 + 0.62 * n_variables),
    )
    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    axis.set_xticks(range(n_variables), labels=variables, rotation=55, ha="right")
    axis.set_yticks(range(n_variables), labels=variables)
    tick_size = 8 if n_variables >= 16 else 9
    axis.tick_params(axis="both", labelsize=tick_size)
    axis.set_title(LEVEL_TITLES[level], pad=16)

    for row_index in range(n_variables):
        for column_index in range(row_index + 1):
            label = annotations[row_index, column_index]
            if label == "":
                label = "NA"
            strength = strengths[row_index, column_index]
            text_colour = "white" if np.isfinite(strength) and strength >= 0.62 else "black"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=6.5 if n_variables >= 16 else 8,
                color=text_colour,
            )
    colour_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colour_bar.set_label("Association strength (0–1)")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def qa_row(
    section: str,
    metric: str,
    *,
    level: str = "",
    variable: str = "",
    value: object = "",
    denominator: object = "",
    share: object = "",
    non_missing_n: object = "",
    missing_n: object = "",
    missing_share: object = "",
    unique_valid_values: object = "",
    details: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "level": level,
        "variable": variable,
        "value": value,
        "denominator": denominator,
        "share": share,
        "non_missing_n": non_missing_n,
        "missing_n": missing_n,
        "missing_share": missing_share,
        "unique_valid_values": unique_valid_values,
        "details": details,
    }


def build_qa(
    canonical: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    working_tables: dict[str, pd.DataFrame],
    pairwise: pd.DataFrame,
    contradiction_summary: dict[str, dict[str, int]],
    mapping_contradictions: int,
    missing_required: list[str],
    obsolete_present: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        qa_row("INPUT", "input model rows", value=len(canonical)),
        qa_row("INPUT", "unique CHOICE_ID", value=canonical["CHOICE_ID"].nunique()),
        qa_row("INPUT", "unique H_ID", value=canonical["H_ID"].nunique()),
        qa_row("INPUT", "unique HP_ID", value=canonical["HP_ID"].nunique()),
        qa_row(
            "INPUT",
            "CHOICE_ID duplicate count",
            value=int(canonical["CHOICE_ID"].duplicated(keep=False).sum()),
        ),
        qa_row(
            "INPUT",
            "required screened columns missing",
            value=len(missing_required),
            details=", ".join(missing_required),
        ),
        qa_row(
            "INPUT",
            "obsolete destination columns present",
            value=len(obsolete_present),
            details=", ".join(obsolete_present),
        ),
    ]
    for level in ["household", "person", "tour"]:
        rows.append(
            qa_row(
                "LEVEL_TABLES",
                f"{level} analysis rows",
                level=level,
                value=len(tables[level]),
            )
        )

    rows.extend(
        [
            qa_row(
                "INVARIANCE",
                "household variables with within-H_ID contradictions",
                level="household",
                value=sum(count > 0 for count in contradiction_summary["household"].values()),
            ),
            qa_row(
                "INVARIANCE",
                "person variables with within-HP_ID contradictions",
                level="person",
                value=sum(count > 0 for count in contradiction_summary["person"].values()),
            ),
            qa_row(
                "INVARIANCE",
                "HP_ID -> H_ID mapping contradictions",
                level="person",
                value=mapping_contradictions,
            ),
        ]
    )

    for level, metadata in VARIABLE_METADATA.items():
        level_n = len(tables[level])
        for variable in metadata:
            series = working_tables[level][variable]
            non_missing_count = int(series.notna().sum())
            missing_count = level_n - non_missing_count
            rows.append(
                qa_row(
                    "VARIABLE_MISSINGNESS",
                    "variable missingness",
                    level=level,
                    variable=variable,
                    denominator=level_n,
                    non_missing_n=non_missing_count,
                    missing_n=missing_count,
                    missing_share=missing_count / level_n if level_n else 0.0,
                    unique_valid_values=int(series.nunique(dropna=True)),
                )
            )

    for level in ["household", "person", "tour"]:
        level_pairs = pairwise.loc[pairwise["level"].eq(level)]
        evaluable = level_pairs["association"].notna()
        coverage = level_pairs["coverage"]
        coverage_metrics = [
            ("number of tested pairs", len(level_pairs)),
            ("evaluable pairs", int(evaluable.sum())),
            ("non-evaluable pairs", int((~evaluable).sum())),
            ("minimum coverage", float(coverage.min())),
            ("median coverage", float(coverage.median())),
            ("pairs with coverage < 0.50", int(coverage.lt(0.50).sum())),
            ("pairs with coverage < 0.80", int(coverage.lt(0.80).sum())),
        ]
        rows.extend(
            qa_row("PAIRWISE_COVERAGE", metric, level=level, value=value)
            for metric, value in coverage_metrics
        )
        for flag in ["STRONG_REVIEW", "MODERATE_REVIEW", "NONE", "NOT_EVALUABLE"]:
            rows.append(
                qa_row(
                    "REVIEW_FLAGS",
                    f"{flag} count",
                    level=level,
                    value=int(level_pairs["review_flag"].eq(flag).sum()),
                )
            )
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def assert_method_dispatch(frame: pd.DataFrame) -> None:
    incorrect = frame.apply(
        lambda row: (row["method"], int(row["directional"]))
        != expected_method(row["type_1"], row["type_2"]),
        axis=1,
    )
    if incorrect.any():
        raise AssertionError("At least one pair used a method inconsistent with declared metadata.")


def assert_statistical_bounds(frame: pd.DataFrame) -> None:
    evaluable = frame["association"].notna()
    if not frame.loc[evaluable, "association_strength"].between(0, 1).all():
        raise AssertionError("Evaluable association strength lies outside [0, 1].")
    directional = frame["method"].isin(["Phi", "Spearman"]) & evaluable
    if not frame.loc[directional, "association"].between(-1, 1).all():
        raise AssertionError("Phi or Spearman association lies outside [-1, 1].")
    unsigned = frame["method"].isin(["Cramers_V", "Eta"]) & evaluable
    if not frame.loc[unsigned, "association"].between(0, 1).all():
        raise AssertionError("Cramer's V or Eta association lies outside [0, 1].")
    if not frame["coverage"].between(0, 1).all():
        raise AssertionError("Pairwise coverage lies outside [0, 1].")


def run_final_assertions(
    canonical: pd.DataFrame,
    canonical_snapshot: pd.DataFrame,
    input_hash: str,
    tables: dict[str, pd.DataFrame],
    contradiction_summary: dict[str, dict[str, int]],
    mapping_contradictions: int,
    pairwise: pd.DataFrame,
    cross_level: pd.DataFrame,
) -> None:
    if not canonical.equals(canonical_snapshot) or file_sha256(MODEL_INPUT_PATH) != input_hash:
        raise AssertionError("Canonical model input was modified during screening.")
    if not canonical["CHOICE_ID"].is_unique:
        raise AssertionError("CHOICE_ID uniqueness changed during screening.")
    if not tables["household"]["H_ID"].is_unique:
        raise AssertionError("Household natural-level table is not one row per H_ID.")
    if not tables["person"]["HP_ID"].is_unique:
        raise AssertionError("Person natural-level table is not one row per HP_ID.")
    if not tables["tour"]["CHOICE_ID"].is_unique:
        raise AssertionError("Tour table is not one row per CHOICE_ID.")
    if len(tables["tour"]) != len(canonical):
        raise AssertionError("A canonical input row was filtered from the tour table.")
    if any(count for values in contradiction_summary.values() for count in values.values()):
        raise AssertionError("Household/person invariance contradiction remains.")
    if mapping_contradictions:
        raise AssertionError("HP_ID maps to more than one H_ID.")
    assert_method_dispatch(pairwise)
    assert_method_dispatch(cross_level)
    assert_statistical_bounds(pairwise)
    assert_statistical_bounds(cross_level)
    for level, metadata in VARIABLE_METADATA.items():
        expected_pairs = len(metadata) * (len(metadata) - 1) // 2
        actual_pairs = int(pairwise["level"].eq(level).sum())
        if actual_pairs != expected_pairs:
            raise AssertionError(
                f"{level} pair count is {actual_pairs}, expected {expected_pairs}."
            )
    if len(cross_level) != 5:
        raise AssertionError("Selected cross-level output must contain exactly five pairs.")


def save_outputs(
    pairwise: pd.DataFrame,
    cross_level: pd.DataFrame,
    qa: pd.DataFrame,
    input_hash: str,
) -> None:
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    cross_level.to_csv(CROSS_LEVEL_PATH, index=False)
    qa.to_csv(QA_PATH, index=False)
    if file_sha256(MODEL_INPUT_PATH) != input_hash:
        raise AssertionError("Canonical model input changed while diagnostic outputs were written.")


def print_summary(
    canonical: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    pairwise: pd.DataFrame,
    cross_level: pd.DataFrame,
) -> None:
    print("HIERARCHICAL MIXED-ASSOCIATION SCREENING")
    print("\nInput:")
    print(f"- rows: {len(canonical):,}")
    print(f"- households: {canonical['H_ID'].nunique():,}")
    print(f"- persons: {canonical['HP_ID'].nunique():,}")
    print(f"- choices: {canonical['CHOICE_ID'].nunique():,}")
    print("\nNatural-level tables:")
    for level in ["household", "person", "tour"]:
        print(f"- {level} rows: {len(tables[level]):,}")
    print("\nStrong-review pairs:")
    for level in ["household", "person", "tour"]:
        count = int(
            pairwise.loc[pairwise["level"].eq(level), "review_flag"]
            .eq("STRONG_REVIEW")
            .sum()
        )
        print(f"- {level}: {count:,}")

    for level in ["household", "person", "tour"]:
        strongest = pairwise.loc[
            pairwise["level"].eq(level) & pairwise["association"].notna()
        ].head(3)
        print(f"\nStrongest evaluable {level} pairs:")
        for row in strongest.itertuples(index=False):
            print(
                f"- {row.variable_1} <-> {row.variable_2}: {row.method}, "
                f"association={row.association:+.4f}, N={row.pairwise_n:,}, "
                f"coverage={row.coverage:.2%}"
            )

    print("\nSelected cross-level associations:")
    for row in cross_level.itertuples(index=False):
        association = "NA" if pd.isna(row.association) else f"{row.association:+.4f}"
        print(
            f"- {row.variable_1} <-> {row.variable_2}: {row.method}, "
            f"association={association}, N={row.pairwise_n:,}, coverage={row.coverage:.2%}"
        )
    print("\nOutputs:")
    for path in OUTPUT_PATHS:
        print(f"- {path.resolve()}")


def main() -> None:
    canonical = read_canonical_input()
    input_hash = file_sha256(MODEL_INPUT_PATH)
    canonical_snapshot = canonical.copy(deep=True)
    missing_required, obsolete_present = validate_input(canonical)
    tables, contradiction_summary, mapping_contradictions = build_natural_level_tables(
        canonical
    )
    pairwise, working_tables = build_within_level_associations(tables)
    cross_level = build_cross_level_associations(canonical)
    qa = build_qa(
        canonical,
        tables,
        working_tables,
        pairwise,
        contradiction_summary,
        mapping_contradictions,
        missing_required,
        obsolete_present,
    )
    run_final_assertions(
        canonical,
        canonical_snapshot,
        input_hash,
        tables,
        contradiction_summary,
        mapping_contradictions,
        pairwise,
        cross_level,
    )
    for level in ["household", "person", "tour"]:
        create_heatmap(
            level,
            pairwise,
            VARIABLE_METADATA[level],
            HEATMAP_PATHS[level],
        )
    save_outputs(pairwise, cross_level, qa, input_hash)
    print_summary(canonical, tables, pairwise, cross_level)


if __name__ == "__main__":
    main()
