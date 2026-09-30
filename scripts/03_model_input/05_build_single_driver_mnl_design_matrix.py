"""Build the encoded feature superset for the single-driver MNL branch.

The allocation-candidate model input is the authoritative row universe.  This
script selects its authoritative single-active-driver branch, applies the
first-pass encodings recorded in ``TRANSFORMATION_REGISTRY``, and writes one
wide design matrix plus QA and registry-derived documentation.  It does not
estimate a model, create interactions, or apply a complete-case restriction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (
    ALLOCATION_CANDIDATE_MODEL_INPUT_PATH,
    MODEL_INPUT_DIR,
    SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH,
    SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH,
    SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH,
)


DICTIONARY_NAME = "MiD2023_analytical_variable_dictionary.xlsx"
DICTIONARY_SHEET = "Slide-ready table"
STRICT_MODEL_INPUT_PATH = MODEL_INPUT_DIR / "mnl_vehicle_choice_model_input.csv"
SENSITIVITY_MODEL_INPUT_PATH = (
    MODEL_INPUT_DIR
    / "sensitivity"
    / "mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv"
)

REFERENCE_HOUSEHOLDS = 16_986
REFERENCE_OCCASIONS = 20_931

IDENTIFIER_COLUMNS = ["CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID"]
CHOICE_CONTROL_COLUMNS = ["CHOICE", "AV_1", "AV_2", "W_GEW"]
BRANCH_COLUMNS = ["SINGLE_ACTIVE_DRIVER_HH", "MULTI_ACTIVE_DRIVER_HH"]
CONTROL_COLUMNS = [*IDENTIFIER_COLUMNS, *CHOICE_CONTROL_COLUMNS, *BRANCH_COLUMNS]


@dataclass(frozen=True)
class CategoryRule:
    """One output indicator and the source values assigned to it."""

    output: str
    label: str
    values: tuple[object, ...] = ()
    minimum: int | None = None


@dataclass(frozen=True)
class TransformSpec:
    """Executable and documentable definition of one source transformation."""

    key: str
    attribute: str
    source_fields: tuple[str, ...]
    block: str
    variable_type: str
    recommended_treatment: str
    kind: str
    output_columns: tuple[str, ...]
    implemented_transformation: str
    missing_treatment: str
    use: str
    recommended_reference: str
    notes: str
    categories: tuple[CategoryRule, ...] = ()
    key_type: str = "integer"
    missing_column: str = ""
    require_positive: bool = False
    grouping_qa: bool = False
    symmetry_group: str = ""
    symmetry_signature: str = ""


def _categorical(
    *,
    key: str,
    attribute: str,
    source: str,
    block: str,
    recommended: str,
    categories: Sequence[CategoryRule],
    implementation: str,
    missing: str,
    use: str,
    reference: str,
    notes: str = "",
    key_type: str = "integer",
    missing_column: str = "",
    grouping_qa: bool = False,
    symmetry_group: str = "",
    symmetry_signature: str = "",
) -> TransformSpec:
    outputs = [rule.output for rule in categories]
    if missing_column:
        outputs.append(missing_column)
    return TransformSpec(
        key=key,
        attribute=attribute,
        source_fields=(source,),
        block=block,
        variable_type="Categorical",
        recommended_treatment=recommended,
        kind="categorical",
        output_columns=tuple(outputs),
        implemented_transformation=implementation,
        missing_treatment=missing,
        use=use,
        recommended_reference=reference,
        notes=notes,
        categories=tuple(categories),
        key_type=key_type,
        missing_column=missing_column,
        grouping_qa=grouping_qa,
        symmetry_group=symmetry_group,
        symmetry_signature=symmetry_signature,
    )


def _binary(
    *,
    key: str,
    attribute: str,
    source: str,
    block: str,
    use: str,
    notes: str = "",
    missing_column: str = "",
) -> TransformSpec:
    outputs = [source]
    if missing_column:
        outputs.append(missing_column)
    missing = (
        f"Retain NA in {source} and set {missing_column}=1."
        if missing_column
        else "No missing values are expected; stop if missingness appears."
    )
    return TransformSpec(
        key=key,
        attribute=attribute,
        source_fields=(source,),
        block=block,
        variable_type="Binary",
        recommended_treatment="Retain observed binary 0/1 coding.",
        kind="binary",
        output_columns=tuple(outputs),
        implemented_transformation=f"Validate and retain {source} as 0/1.",
        missing_treatment=missing,
        use=use,
        recommended_reference="No / 0",
        notes=notes,
        missing_column=missing_column,
    )


def _continuous_log(
    *,
    key: str,
    attribute: str,
    source: str,
    log_output: str,
    block: str,
    use: str,
    notes: str,
) -> TransformSpec:
    return TransformSpec(
        key=key,
        attribute=attribute,
        source_fields=(source,),
        block=block,
        variable_type="Continuous",
        recommended_treatment="Retain raw value and test a log/flexible functional form.",
        kind="continuous_log",
        output_columns=(source, log_output),
        implemented_transformation=(
            f"Retain {source}; create {log_output}=ln({source}) for positive values."
        ),
        missing_treatment="No missing or non-positive values are permitted for this required log candidate.",
        use=use,
        recommended_reference="Not applicable",
        notes=notes,
        require_positive=True,
    )


def _continuous_age(
    *,
    key: str,
    attribute: str,
    source: str,
    missing_column: str,
    symmetry_signature: str,
) -> TransformSpec:
    return TransformSpec(
        key=key,
        attribute=attribute,
        source_fields=(source,),
        block="Vehicle attributes",
        variable_type="Continuous",
        recommended_treatment=(
            "Keep continuous years; inspect outliers and functional form. Do not cap or spline by default."
        ),
        kind="continuous",
        output_columns=(source, missing_column),
        implemented_transformation=(
            f"Retain {source} in years without capping/winsorizing; add {missing_column}."
        ),
        missing_treatment=f"Retain NA in {source} and set {missing_column}=1; do not impute.",
        use="Baseline candidate",
        recommended_reference="Not applicable",
        notes=(
            "Centering, splines, and robust caps remain sensitivity/functional-form decisions."
        ),
        missing_column=missing_column,
        symmetry_group="vehicle_age",
        symmetry_signature=symmetry_signature,
    )


def build_transformation_registry() -> tuple[TransformSpec, ...]:
    """Return the sole executable/documentation source for all encodings."""

    tour = "Tour & departure context"
    household = "Household & spatial context"
    person = "Person / driver context"
    vehicle = "Vehicle attributes"

    def rules(items: Sequence[tuple[str, str, Sequence[object]]]) -> tuple[CategoryRule, ...]:
        return tuple(CategoryRule(output, label, tuple(values)) for output, label, values in items)

    registry: list[TransformSpec] = [
        _categorical(
            key="day_type",
            attribute="Day type",
            source="DAY_TYPE",
            block=tour,
            recommended="Keep three categories for stratified heterogeneity analysis, not as a baseline utility predictor.",
            categories=rules(
                [
                    ("DAYTYPE_HOLIDAY", "Holiday", ("HOLIDAY",)),
                    ("DAYTYPE_WEEKDAY", "Weekday", ("WEEKDAY",)),
                    ("DAYTYPE_WEEKEND", "Weekend", ("WEEKEND",)),
                ]
            ),
            implementation="Create the full Holiday/Weekday/Weekend indicator set.",
            missing="No missing category is expected; stop if missingness appears.",
            use="Stratification only",
            reference="WEEKDAY",
            key_type="text",
        ),
        _categorical(
            key="season",
            attribute="Season",
            source="saison",
            block=tour,
            recommended="Treat codes 1–4 as four categorical seasons; never as a linear predictor.",
            categories=rules(
                [
                    ("SEASON_WINTER", "Winter", (1,)),
                    ("SEASON_SPRING", "Spring", (2,)),
                    ("SEASON_SUMMER", "Summer", (3,)),
                    ("SEASON_AUTUMN", "Autumn", (4,)),
                ]
            ),
            implementation="Create four season indicators.",
            missing="No missing category is expected; stop if missingness appears.",
            use="Baseline candidate",
            reference="WINTER",
        ),
        _continuous_log(
            key="tour_distance",
            attribute="Tour distance",
            source="TOUR_DISTANCE_KM",
            log_output="LN_TOUR_DISTANCE_KM",
            block=tour,
            use="Baseline candidate / alternative functional form",
            notes=(
                "Raw and log forms are alternative functional-form candidates; use only the predefined selected form in a specific model."
            ),
        ),
        _continuous_log(
            key="tour_travel_time",
            attribute="Tour travel time",
            source="TOUR_TRAVEL_TIME_MIN",
            log_output="LN_TOUR_TRAVEL_TIME_MIN",
            block=tour,
            use="Alternative specification / redundancy check",
            notes=(
                "Distance and travel time should not automatically enter together until redundancy/conditioning diagnostics support doing so."
            ),
        ),
        _categorical(
            key="tour_stops",
            attribute="Number of non-home stops",
            source="TOUR_N_NONHOME_STOPS",
            block=tour,
            recommended="Group as 0 / 1 / 2 / 3+; retain and interpret the rare 0-stop category.",
            categories=(
                CategoryRule("TOUR_STOPS_0", "0", (0,)),
                CategoryRule("TOUR_STOPS_1", "1", (1,)),
                CategoryRule("TOUR_STOPS_2", "2", (2,)),
                CategoryRule("TOUR_STOPS_3PLUS", "3+", minimum=3),
            ),
            implementation="Create mutually exclusive indicators for 0, 1, 2, and 3+ stops.",
            missing="No missing count is expected; stop if missingness appears.",
            use="Baseline candidate",
            reference="1 stop",
            notes="The 0-stop category is rare/special and requires interpretation.",
            grouping_qa=True,
        ),
        _categorical(
            key="tour_purpose_count",
            attribute="Number of distinct purposes",
            source="TOUR_N_DISTINCT_PURPOSES",
            block=tour,
            recommended="Group as 0 / 1 / 2 / 3+; do not delete 0-purpose rows.",
            categories=(
                CategoryRule("TOUR_PURPOSE_COUNT_0", "0", (0,)),
                CategoryRule("TOUR_PURPOSE_COUNT_1", "1", (1,)),
                CategoryRule("TOUR_PURPOSE_COUNT_2", "2", (2,)),
                CategoryRule("TOUR_PURPOSE_COUNT_3PLUS", "3+", minimum=3),
            ),
            implementation="Create mutually exclusive indicators for 0, 1, 2, and 3+ distinct purposes.",
            missing="No missing count is expected; stop if missingness appears.",
            use="Baseline candidate",
            reference="1 purpose",
            grouping_qa=True,
        ),
        _categorical(
            key="first_purpose",
            attribute="First tour purpose",
            source="TOUR_FIRST_PURPOSE",
            block=tour,
            recommended="Treat codes 1–10 categorically; retain sparse Home and Return categories.",
            categories=rules(
                [
                    ("FIRST_PURPOSE_WORK", "Work", (1,)),
                    ("FIRST_PURPOSE_BUSINESS", "Business", (2,)),
                    ("FIRST_PURPOSE_EDUCATION", "Education", (3,)),
                    ("FIRST_PURPOSE_SHOPPING", "Shopping", (4,)),
                    ("FIRST_PURPOSE_ERRAND", "Private errand", (5,)),
                    ("FIRST_PURPOSE_ESCORT", "Escort", (6,)),
                    ("FIRST_PURPOSE_LEISURE", "Leisure", (7,)),
                    ("FIRST_PURPOSE_HOME", "Home", (8,)),
                    ("FIRST_PURPOSE_RETURN", "Return", (9,)),
                    ("FIRST_PURPOSE_OTHER", "Other", (10,)),
                ]
            ),
            implementation="Create the full ten-category first-purpose indicator block.",
            missing="Do not impute; encode missing values with FIRST_PURPOSE_MISSING.",
            use="Alternative specification / redundancy check",
            reference="WORK",
            notes=(
                "Home and Return are retained as sparse categories. This block is an alternative representation to TOUR_HAS_*; do not include both full blocks automatically."
            ),
            missing_column="FIRST_PURPOSE_MISSING",
            grouping_qa=True,
        ),
    ]

    purpose_binaries = [
        ("tour_has_work", "Contains work purpose", "TOUR_HAS_WORK", ""),
        ("tour_has_business", "Contains business purpose", "TOUR_HAS_BUSINESS", ""),
        (
            "tour_has_education",
            "Contains education purpose",
            "TOUR_HAS_EDUCATION",
            "Sparse positive support; review before estimation.",
        ),
        ("tour_has_shopping", "Contains shopping purpose", "TOUR_HAS_SHOPPING", ""),
        ("tour_has_errand", "Contains errand purpose", "TOUR_HAS_ERRAND", ""),
        ("tour_has_leisure", "Contains leisure purpose", "TOUR_HAS_LEISURE", ""),
        ("tour_has_escort", "Contains escort purpose", "TOUR_HAS_ESCORT", ""),
        (
            "tour_has_other",
            "Contains other purpose",
            "TOUR_HAS_OTHER_PURPOSE",
            "",
        ),
    ]
    registry.extend(
        _binary(
            key=key,
            attribute=attribute,
            source=source,
            block=tour,
            use="Alternative specification / redundancy check",
            notes=(
                "Alternative purpose representation to TOUR_FIRST_PURPOSE. " + note
            ).strip(),
        )
        for key, attribute, source, note in purpose_binaries
    )
    registry.extend(
        [
            _categorical(
                key="accompanying_count",
                attribute="Number of accompanying persons",
                source="W_ANZBEGL",
                block=tour,
                recommended="Group as 0 / 1 / 2 / 3+ rather than using the long-tailed count linearly.",
                categories=(
                    CategoryRule("ACCOMP_COUNT_0", "0", (0,)),
                    CategoryRule("ACCOMP_COUNT_1", "1", (1,)),
                    CategoryRule("ACCOMP_COUNT_2", "2", (2,)),
                    CategoryRule("ACCOMP_COUNT_3PLUS", "3+", minimum=3),
                ),
                implementation="Create mutually exclusive indicators for 0, 1, 2, and 3+ accompanying persons.",
                missing="Do not impute; encode missing values with ACCOMP_COUNT_MISSING.",
                use="Baseline candidate",
                reference="0",
                missing_column="ACCOMP_COUNT_MISSING",
                grouping_qa=True,
            ),
            _binary(
                key="household_accompanied",
                attribute="Accompanied by household member(s)",
                source="HOUSEHOLD_ACCOMPANIED",
                block=tour,
                use="Baseline candidate",
                missing_column="HOUSEHOLD_ACCOMPANIED_MISSING",
            ),
            _categorical(
                key="weather",
                attribute="Weather",
                source="P_STWETTER",
                block=tour,
                recommended="Treat codes 1–6 as six categorical weather states.",
                categories=rules(
                    [
                        ("WEATHER_SUNNY", "Sunny", (1,)),
                        ("WEATHER_FAIR", "Fair", (2,)),
                        ("WEATHER_CHANGEABLE", "Changeable", (3,)),
                        ("WEATHER_OVERCAST", "Overcast", (4,)),
                        ("WEATHER_RAINY", "Rainy", (5,)),
                        ("WEATHER_SNOW", "Snow", (6,)),
                    ]
                ),
                implementation="Create six weather indicators plus WEATHER_MISSING.",
                missing="Do not impute a weather state; encode missing values with WEATHER_MISSING.",
                use="Baseline candidate",
                reference="SUNNY",
                missing_column="WEATHER_MISSING",
            ),
            _categorical(
                key="destination_xmstadt",
                attribute="Worst destination X-minute-city category",
                source="TOUR_WORST_XMSTADT_ZO",
                block=tour,
                recommended="Treat codes 1–6 categorically as a destination-context extension.",
                categories=rules(
                    [
                        ("DEST_XMSTADT_5MIN", "5-minute city", (1,)),
                        ("DEST_XMSTADT_10MIN", "10-minute city", (2,)),
                        ("DEST_XMSTADT_15MIN", "15-minute city", (3,)),
                        ("DEST_XMSTADT_20MIN", "20-minute city", (4,)),
                        ("DEST_XMSTADT_30MIN", "30-minute city", (5,)),
                        ("DEST_XMSTADT_60MIN_OTHER", "60-minute city / other", (6,)),
                    ]
                ),
                implementation="Create six destination-accessibility indicators plus DEST_XMSTADT_MISSING.",
                missing="Do not impute; encode missing values with DEST_XMSTADT_MISSING.",
                use="Destination-context extension only",
                reference="10MIN",
                notes="Complete-case use of this extension changes the estimable sample.",
                missing_column="DEST_XMSTADT_MISSING",
            ),
            _categorical(
                key="destination_pt_quality",
                attribute="Worst destination public-transport quality",
                source="TOUR_WORST_QUALI_OPNV_ZO",
                block=tour,
                recommended="Treat codes 1–4 categorically as a destination-context extension.",
                categories=rules(
                    [
                        ("DEST_PT_QUALITY_VERY_POOR", "Very poor", (1,)),
                        ("DEST_PT_QUALITY_POOR", "Poor", (2,)),
                        ("DEST_PT_QUALITY_GOOD", "Good", (3,)),
                        ("DEST_PT_QUALITY_VERY_GOOD", "Very good", (4,)),
                    ]
                ),
                implementation="Create four destination PT-quality indicators plus DEST_PT_QUALITY_MISSING.",
                missing="Do not impute; encode missing values with DEST_PT_QUALITY_MISSING.",
                use="Destination-context extension only",
                reference="POOR",
                notes="Complete-case use of this extension changes the estimable sample.",
                missing_column="DEST_PT_QUALITY_MISSING",
            ),
            _categorical(
                key="household_type",
                attribute="Household type",
                source="household_type",
                block=household,
                recommended="Use four derived household life-stage categories.",
                categories=rules(
                    [
                        ("HH_TYPE_ADULT", "Adult household", ("adult_household",)),
                        ("HH_TYPE_FAMILY", "Family household", ("family_household",)),
                        ("HH_TYPE_SENIOR", "Senior household", ("senior_household",)),
                        ("HH_TYPE_YOUNG", "Young household", ("young_household",)),
                    ]
                ),
                implementation="Create four household-type indicators.",
                missing="No missing category is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="ADULT",
                notes="Household type is derived from member ages and overlaps conceptually with household size and person age.",
                key_type="text",
            ),
            _categorical(
                key="household_size",
                attribute="Household size group",
                source="hhgr_gr",
                block=household,
                recommended="Treat codes 1 / 2 / 3 / 4 / 5+ categorically.",
                categories=rules(
                    [
                        ("HH_SIZE_1", "1 person", (1,)),
                        ("HH_SIZE_2", "2 persons", (2,)),
                        ("HH_SIZE_3", "3 persons", (3,)),
                        ("HH_SIZE_4", "4 persons", (4,)),
                        ("HH_SIZE_5PLUS", "5+ persons", (5,)),
                    ]
                ),
                implementation="Create five household-size indicators.",
                missing="No missing category is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="2-person household",
            ),
            _categorical(
                key="tenure",
                attribute="Housing tenure",
                source="H_MIETE",
                block=household,
                recommended="Treat Rent / Owner / Other categorically.",
                categories=rules(
                    [
                        ("TENURE_RENT", "Rent", (1,)),
                        ("TENURE_OWNER", "Owner", (2,)),
                        ("TENURE_OTHER", "Other", (3,)),
                    ]
                ),
                implementation="Create three tenure indicators plus TENURE_MISSING.",
                missing="Do not impute; encode missing values with TENURE_MISSING.",
                use="Baseline candidate",
                reference="OWNER",
                missing_column="TENURE_MISSING",
            ),
            _categorical(
                key="economic_status",
                attribute="Economic status",
                source="oek_status",
                block=household,
                recommended="Use five categorical indicators; do not impose a linear monotonic score at this stage.",
                categories=rules(
                    [
                        ("ECON_VERY_LOW", "Very low", (1,)),
                        ("ECON_LOW", "Low", (2,)),
                        ("ECON_MEDIUM", "Medium", (3,)),
                        ("ECON_HIGH", "High", (4,)),
                        ("ECON_VERY_HIGH", "Very high", (5,)),
                    ]
                ),
                implementation="Create five economic-status indicators.",
                missing="No missing category is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="MEDIUM",
            ),
            _binary(
                key="household_carsharing",
                attribute="Household carsharing resource",
                source="H_CS",
                block=household,
                use="Baseline candidate",
                notes="Potential overlap with individual carsharing membership.",
                missing_column="H_CS_MISSING",
            ),
            _categorical(
                key="household_pedelecs",
                attribute="Number of pedelecs in household",
                source="H_ANZPED",
                block=household,
                recommended="Group as 0 / 1 / 2+ rather than using the count linearly.",
                categories=(
                    CategoryRule("HH_PEDELEC_0", "0", (0,)),
                    CategoryRule("HH_PEDELEC_1", "1", (1,)),
                    CategoryRule("HH_PEDELEC_2PLUS", "2+", minimum=2),
                ),
                implementation="Create mutually exclusive indicators for 0, 1, and 2+ household pedelecs.",
                missing="No missing count is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="0",
                notes="Alternative resource representation to personal pedelec availability.",
                grouping_qa=True,
            ),
            _categorical(
                key="household_bicycles",
                attribute="Number of bicycles in household",
                source="H_ANZRAD",
                block=household,
                recommended="Group as 0 / 1 / 2 / 3+ rather than using the count linearly.",
                categories=(
                    CategoryRule("HH_BICYCLE_0", "0", (0,)),
                    CategoryRule("HH_BICYCLE_1", "1", (1,)),
                    CategoryRule("HH_BICYCLE_2", "2", (2,)),
                    CategoryRule("HH_BICYCLE_3PLUS", "3+", minimum=3),
                ),
                implementation="Create mutually exclusive indicators for 0, 1, 2, and 3+ household bicycles.",
                missing="No missing count is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="0",
                notes="Alternative resource representation to personal bicycle availability.",
                grouping_qa=True,
            ),
            _categorical(
                key="regional_type",
                attribute="Regional settlement type",
                source="RegioStaR4",
                block=household,
                recommended="Treat codes 11 / 12 / 21 / 22 as four settlement types, never numerically.",
                categories=rules(
                    [
                        ("REGIO_METROPOLITAN_URBAN", "Metropolitan urban", (11,)),
                        ("REGIO_REGIOPOLITAN_URBAN", "Regiopolitan urban", (12,)),
                        ("REGIO_RURAL_NEAR_URBAN", "Rural near urban", (21,)),
                        ("REGIO_PERIPHERAL_RURAL", "Peripheral rural", (22,)),
                    ]
                ),
                implementation="Create four regional-settlement indicators.",
                missing="No missing category is expected; stop if missingness appears.",
                use="Baseline candidate",
                reference="METROPOLITAN_URBAN",
            ),
            _categorical(
                key="residential_xmstadt",
                attribute="Residential X-minute-city category",
                source="XMStadt",
                block=household,
                recommended="Treat codes 1–6 categorically.",
                categories=rules(
                    [
                        ("RES_XMSTADT_5MIN", "5-minute city", (1,)),
                        ("RES_XMSTADT_10MIN", "10-minute city", (2,)),
                        ("RES_XMSTADT_15MIN", "15-minute city", (3,)),
                        ("RES_XMSTADT_20MIN", "20-minute city", (4,)),
                        ("RES_XMSTADT_30MIN", "30-minute city", (5,)),
                        ("RES_XMSTADT_60MIN_OTHER", "60-minute city / other", (6,)),
                    ]
                ),
                implementation="Create six residential-accessibility indicators plus RES_XMSTADT_MISSING.",
                missing="Do not impute; encode missing values with RES_XMSTADT_MISSING.",
                use="Baseline candidate",
                reference="10MIN",
                missing_column="RES_XMSTADT_MISSING",
            ),
            _categorical(
                key="residential_pt_quality",
                attribute="Residential public-transport quality",
                source="quali_opnv",
                block=household,
                recommended="Treat codes 1–4 categorically.",
                categories=rules(
                    [
                        ("RES_PT_QUALITY_VERY_POOR", "Very poor", (1,)),
                        ("RES_PT_QUALITY_POOR", "Poor", (2,)),
                        ("RES_PT_QUALITY_GOOD", "Good", (3,)),
                        ("RES_PT_QUALITY_VERY_GOOD", "Very good", (4,)),
                    ]
                ),
                implementation="Create four residential PT-quality indicators plus RES_PT_QUALITY_MISSING.",
                missing="Do not impute; encode missing values with RES_PT_QUALITY_MISSING.",
                use="Baseline candidate",
                reference="POOR",
                missing_column="RES_PT_QUALITY_MISSING",
            ),
            _categorical(
                key="motorway_access",
                attribute="Motorway accessibility",
                source="min_bab",
                block=household,
                recommended="Collapse the sparse 30–<40 and 40+ source categories into 30+.",
                categories=rules(
                    [
                        ("MOTORWAY_LT10", "<10 min", (1,)),
                        ("MOTORWAY_10_20", "10–<20 min", (2,)),
                        ("MOTORWAY_20_30", "20–<30 min", (3,)),
                        ("MOTORWAY_30PLUS", "30+ min", (4, 5)),
                    ]
                ),
                implementation="Create <10, 10–<20, 20–<30, and 30+ indicators; combine original codes 4 and 5.",
                missing="Do not impute; encode missing values with MOTORWAY_ACCESS_MISSING.",
                use="Baseline candidate",
                reference="LT10",
                missing_column="MOTORWAY_ACCESS_MISSING",
                grouping_qa=True,
            ),
            _categorical(
                key="central_access",
                attribute="Central-place accessibility",
                source="min_ozmz",
                block=household,
                recommended="Collapse the very sparse upper tail into 20+.",
                categories=rules(
                    [
                        ("CENTRAL_LT10", "<10 min", (1,)),
                        ("CENTRAL_10_20", "10–<20 min", (2,)),
                        ("CENTRAL_20PLUS", "20+ min", (3, 4, 5)),
                    ]
                ),
                implementation="Create <10, 10–<20, and 20+ indicators; combine original codes 3, 4, and 5.",
                missing="Do not impute; encode missing values with CENTRAL_ACCESS_MISSING.",
                use="Baseline candidate",
                reference="LT10",
                missing_column="CENTRAL_ACCESS_MISSING",
                grouping_qa=True,
            ),
            _categorical(
                key="sex",
                attribute="Gender",
                source="HP_SEX",
                block=person,
                recommended="Treat Male / Female / Diverse categorically; retain the extremely sparse Diverse category.",
                categories=rules(
                    [
                        ("SEX_MALE", "Male", (1,)),
                        ("SEX_FEMALE", "Female", (2,)),
                        ("SEX_DIVERSE", "Diverse", (3,)),
                    ]
                ),
                implementation="Create Male/Female/Diverse indicators plus SEX_MISSING.",
                missing="Do not impute or recode; encode missing values with SEX_MISSING.",
                use="Baseline candidate",
                reference="MALE",
                notes="Diverse is extremely sparse and requires an explicit estimation-stage decision.",
                missing_column="SEX_MISSING",
                grouping_qa=True,
            ),
            _categorical(
                key="age_band",
                attribute="Age group",
                source="alter_gr5",
                block=person,
                recommended="Regroup narrow codes 3–15 into broad age bands; never use the source code linearly.",
                categories=rules(
                    [
                        ("AGE_UNDER18", "Under 18", (3,)),
                        ("AGE_18_34", "Age 18–34", (4, 5, 6)),
                        ("AGE_35_49", "Age 35–49", (7, 8, 9)),
                        ("AGE_50_64", "Age 50–64", (10, 11, 12)),
                        ("AGE_65PLUS", "Age 65+", (13, 14, 15)),
                    ]
                ),
                implementation="Map code 3 to Under 18; 4–6 to 18–34; 7–9 to 35–49; 10–12 to 50–64; 13–15 to 65+.",
                missing="Do not impute; encode missing values with AGE_MISSING.",
                use="Baseline candidate",
                reference="AGE_35_49",
                notes="Under 18 is rare and requires support review.",
                missing_column="AGE_MISSING",
                grouping_qa=True,
            ),
            _binary(
                key="employment",
                attribute="Employment status",
                source="erwerb",
                block=person,
                use="Baseline candidate",
                missing_column="ERWERB_MISSING",
            ),
            _binary(
                key="personal_pedelec",
                attribute="Personal pedelec availability",
                source="P_VPED",
                block=person,
                use="Baseline candidate",
                notes="Alternative resource representation to H_ANZPED.",
                missing_column="P_VPED_MISSING",
            ),
            _binary(
                key="personal_bicycle",
                attribute="Personal bicycle availability",
                source="P_VRAD",
                block=person,
                use="Baseline candidate",
                notes="Alternative resource representation to H_ANZRAD.",
                missing_column="P_VRAD_MISSING",
            ),
            _binary(
                key="carsharing",
                attribute="Carsharing membership",
                source="carsharing",
                block=person,
                use="Baseline candidate",
                notes="Missingness reflects design/nonresponse and must not be recoded to No.",
                missing_column="CARSHARING_MISSING",
            ),
            _binary(
                key="mobility_limitation",
                attribute="Mobility limitation",
                source="mobein",
                block=person,
                use="Baseline candidate",
                notes="Missingness reflects health-module coverage and must not be recoded to No.",
                missing_column="MOBILITY_LIMITATION_MISSING",
            ),
        ]
    )

    powertrain_rules_1 = rules(
        [
            ("PT1_ICE", "ICE", (1, 2, 6)),
            ("PT1_HEV", "HEV", (3,)),
            ("PT1_PHEV", "PHEV", (4,)),
            ("PT1_BEV", "BEV", (5,)),
            ("PT1_OTHER", "Other", (7,)),
        ]
    )
    powertrain_rules_2 = rules(
        [
            ("PT2_ICE", "ICE", (1, 2, 6)),
            ("PT2_HEV", "HEV", (3,)),
            ("PT2_PHEV", "PHEV", (4,)),
            ("PT2_BEV", "BEV", (5,)),
            ("PT2_OTHER", "Other", (7,)),
        ]
    )
    segment_rules_1 = rules(
        [
            ("SEG1_SMALL", "Small", (1,)),
            ("SEG1_COMPACT", "Compact", (2,)),
            ("SEG1_MEDIUM", "Medium", (3,)),
            ("SEG1_LARGE", "Large", (4,)),
        ]
    )
    segment_rules_2 = rules(
        [
            ("SEG2_SMALL", "Small", (1,)),
            ("SEG2_COMPACT", "Compact", (2,)),
            ("SEG2_MEDIUM", "Medium", (3,)),
            ("SEG2_LARGE", "Large", (4,)),
        ]
    )
    holder_rules_1 = rules(
        [
            ("HOLDER1_PRIVATE", "Private", (1,)),
            ("HOLDER1_COMPANY", "Company", (2,)),
            ("HOLDER1_OTHER", "Other", (3,)),
        ]
    )
    holder_rules_2 = rules(
        [
            ("HOLDER2_PRIVATE", "Private", (1,)),
            ("HOLDER2_COMPANY", "Company", (2,)),
            ("HOLDER2_OTHER", "Other", (3,)),
        ]
    )
    status_rules_1 = rules(
        [
            ("STATUS1_LOW", "Low", (1,)),
            ("STATUS1_MEDIUM", "Medium", (2,)),
            ("STATUS1_HIGH", "High", (3,)),
        ]
    )
    status_rules_2 = rules(
        [
            ("STATUS2_LOW", "Low", (1,)),
            ("STATUS2_MEDIUM", "Medium", (2,)),
            ("STATUS2_HIGH", "High", (3,)),
        ]
    )
    registry.extend(
        [
            _categorical(
                key="powertrain_1",
                attribute="Powertrain — Car 1",
                source="POWERTRAIN_1",
                block=vehicle,
                recommended="Regroup as ICE / HEV / PHEV / BEV / Other.",
                categories=powertrain_rules_1,
                implementation="Map Gasoline, Diesel, and Gas (1/2/6) to ICE; retain HEV, PHEV, BEV, and Other separately.",
                missing="Do not impute; encode missing values with PT1_MISSING.",
                use="Baseline candidate",
                reference="ICE",
                notes="Other is retained, not deleted.",
                missing_column="PT1_MISSING",
                grouping_qa=True,
                symmetry_group="powertrain",
                symmetry_signature="ICE=1,2,6;HEV=3;PHEV=4;BEV=5;OTHER=7;MISSING",
            ),
            _categorical(
                key="powertrain_2",
                attribute="Powertrain — Car 2",
                source="POWERTRAIN_2",
                block=vehicle,
                recommended="Regroup as ICE / HEV / PHEV / BEV / Other.",
                categories=powertrain_rules_2,
                implementation="Map Gasoline, Diesel, and Gas (1/2/6) to ICE; retain HEV, PHEV, BEV, and Other separately.",
                missing="Do not impute; encode missing values with PT2_MISSING.",
                use="Baseline candidate",
                reference="ICE",
                notes="Other is retained, not deleted.",
                missing_column="PT2_MISSING",
                grouping_qa=True,
                symmetry_group="powertrain",
                symmetry_signature="ICE=1,2,6;HEV=3;PHEV=4;BEV=5;OTHER=7;MISSING",
            ),
            _categorical(
                key="segment_1",
                attribute="Vehicle segment — Car 1",
                source="SEGMENT_1",
                block=vehicle,
                recommended="Treat Small / Compact / Medium / Large categorically.",
                categories=segment_rules_1,
                implementation="Create four segment indicators plus SEG1_MISSING.",
                missing="Do not impute; encode missing values with SEG1_MISSING.",
                use="Baseline candidate",
                reference="SMALL",
                missing_column="SEG1_MISSING",
                symmetry_group="segment",
                symmetry_signature="SMALL=1;COMPACT=2;MEDIUM=3;LARGE=4;MISSING",
            ),
            _categorical(
                key="segment_2",
                attribute="Vehicle segment — Car 2",
                source="SEGMENT_2",
                block=vehicle,
                recommended="Treat Small / Compact / Medium / Large categorically.",
                categories=segment_rules_2,
                implementation="Create four segment indicators plus SEG2_MISSING.",
                missing="Do not impute; encode missing values with SEG2_MISSING.",
                use="Baseline candidate",
                reference="SMALL",
                missing_column="SEG2_MISSING",
                symmetry_group="segment",
                symmetry_signature="SMALL=1;COMPACT=2;MEDIUM=3;LARGE=4;MISSING",
            ),
            _continuous_age(
                key="vehicle_age_1",
                attribute="Vehicle age — Car 1",
                source="VEHICLE_AGE_1",
                missing_column="VEHICLE_AGE_1_MISSING",
                symmetry_signature="CONTINUOUS_YEARS;NO_CAP;MISSING",
            ),
            _continuous_age(
                key="vehicle_age_2",
                attribute="Vehicle age — Car 2",
                source="VEHICLE_AGE_2",
                missing_column="VEHICLE_AGE_2_MISSING",
                symmetry_signature="CONTINUOUS_YEARS;NO_CAP;MISSING",
            ),
            _categorical(
                key="holder_1",
                attribute="Holder type — Car 1",
                source="HOLDER_1",
                block=vehicle,
                recommended="Treat Private / Company / Other categorically, but not as a baseline because coverage is poor.",
                categories=holder_rules_1,
                implementation="Create Private/Company/Other indicators plus HOLDER1_MISSING.",
                missing="Retain all rows and encode missing values with HOLDER1_MISSING; do not impute.",
                use="Extension only / not baseline",
                reference="PRIVATE",
                notes=">50% missingness makes holder unsuitable for the baseline unless a later specification intentionally accepts restricted coverage or explicit missingness treatment.",
                missing_column="HOLDER1_MISSING",
                symmetry_group="holder",
                symmetry_signature="PRIVATE=1;COMPANY=2;OTHER=3;MISSING",
            ),
            _categorical(
                key="holder_2",
                attribute="Holder type — Car 2",
                source="HOLDER_2",
                block=vehicle,
                recommended="Treat Private / Company / Other categorically, but not as a baseline because coverage is poor.",
                categories=holder_rules_2,
                implementation="Create Private/Company/Other indicators plus HOLDER2_MISSING.",
                missing="Retain all rows and encode missing values with HOLDER2_MISSING; do not impute.",
                use="Extension only / not baseline",
                reference="PRIVATE",
                notes=">50% missingness makes holder unsuitable for the baseline unless a later specification intentionally accepts restricted coverage or explicit missingness treatment.",
                missing_column="HOLDER2_MISSING",
                symmetry_group="holder",
                symmetry_signature="PRIVATE=1;COMPANY=2;OTHER=3;MISSING",
            ),
            _categorical(
                key="status_1",
                attribute="Vehicle status — Car 1",
                source="STATUS_1",
                block=vehicle,
                recommended="Treat Low / Medium / High categorically as an alternative vehicle representation.",
                categories=status_rules_1,
                implementation="Create Low/Medium/High indicators plus STATUS1_MISSING.",
                missing="Do not impute; encode missing values with STATUS1_MISSING.",
                use="Alternative specification / redundancy check",
                reference="LOW",
                notes="STATUS is an alternative to Segment + Vehicle Age; do not automatically combine all three representations.",
                missing_column="STATUS1_MISSING",
                symmetry_group="status",
                symmetry_signature="LOW=1;MEDIUM=2;HIGH=3;MISSING",
            ),
            _categorical(
                key="status_2",
                attribute="Vehicle status — Car 2",
                source="STATUS_2",
                block=vehicle,
                recommended="Treat Low / Medium / High categorically as an alternative vehicle representation.",
                categories=status_rules_2,
                implementation="Create Low/Medium/High indicators plus STATUS2_MISSING.",
                missing="Do not impute; encode missing values with STATUS2_MISSING.",
                use="Alternative specification / redundancy check",
                reference="LOW",
                notes="STATUS is an alternative to Segment + Vehicle Age; do not automatically combine all three representations.",
                missing_column="STATUS2_MISSING",
                symmetry_group="status",
                symmetry_signature="LOW=1;MEDIUM=2;HIGH=3;MISSING",
            ),
        ]
    )
    return tuple(registry)


TRANSFORMATION_REGISTRY = build_transformation_registry()


def read_strings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def numeric(series: pd.Series) -> pd.Series:
    stripped = series.astype("string").str.strip().replace("", pd.NA)
    return pd.to_numeric(stripped, errors="coerce")


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")


def assert_outputs_absent() -> None:
    targets = [
        SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH,
        SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH,
        SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH,
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing single-driver design-matrix output(s): "
            f"{existing}"
        )


def file_hashes(paths: Iterable[Path]) -> dict[Path, str]:
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def registry_fingerprint(registry: Sequence[TransformSpec]) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in registry],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_columns(registry: Sequence[TransformSpec]) -> list[str]:
    columns = [column for spec in registry for column in spec.output_columns]
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise AssertionError(f"Transformation registry has duplicate output columns: {duplicates}")
    return columns


def validate_registry(registry: Sequence[TransformSpec]) -> None:
    keys = [spec.key for spec in registry]
    if len(keys) != len(set(keys)):
        raise AssertionError("Transformation registry keys must be unique.")
    feature_columns(registry)

    symmetry: dict[str, list[str]] = {}
    for spec in registry:
        if spec.symmetry_group:
            symmetry.setdefault(spec.symmetry_group, []).append(spec.symmetry_signature)
    for group, signatures in symmetry.items():
        if len(signatures) != 2 or len(set(signatures)) != 1:
            raise AssertionError(
                f"Vehicle symmetry registry mismatch for {group}: {signatures}"
            )


def select_single_driver_branch(source: pd.DataFrame) -> pd.DataFrame:
    required_sources = [field for spec in TRANSFORMATION_REGISTRY for field in spec.source_fields]
    require_columns(source.columns, [*CONTROL_COLUMNS, *required_sources], "candidate input")
    if source["CHOICE_ID"].eq("").any() or not source["CHOICE_ID"].is_unique:
        raise ValueError("Candidate CHOICE_ID must be populated and unique.")

    single_flag = numeric(source["SINGLE_ACTIVE_DRIVER_HH"])
    multi_flag = numeric(source["MULTI_ACTIVE_DRIVER_HH"])
    if single_flag.isna().any() or multi_flag.isna().any():
        raise ValueError("Authoritative branch flags must be populated.")
    if not single_flag.isin([0, 1]).all() or not multi_flag.isin([0, 1]).all():
        raise ValueError("Authoritative branch flags must be binary.")
    if (single_flag + multi_flag).ne(1).any():
        raise ValueError("Every candidate row must belong to exactly one modelling branch.")

    retained = source.loc[single_flag.eq(1)].copy().reset_index(drop=True)
    if numeric(retained["MULTI_ACTIVE_DRIVER_HH"]).ne(0).any():
        raise AssertionError("A retained single-driver row is flagged as multi-driver.")
    return retained


def _source_keys(spec: TransformSpec, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    stripped = series.astype("string").str.strip()
    missing = stripped.eq("") | stripped.isna()
    if spec.key_type == "text":
        keys = stripped.mask(missing, pd.NA)
        invalid_format = pd.Series(False, index=series.index)
        return keys, missing, invalid_format

    values = pd.to_numeric(stripped.mask(missing, pd.NA), errors="coerce")
    invalid_format = (~missing) & (values.isna() | values.mod(1).ne(0))
    keys = values.round().astype("Int64")
    return keys, missing, invalid_format


def apply_categorical(
    frame: pd.DataFrame,
    spec: TransformSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = spec.source_fields[0]
    keys, missing, invalid_format = _source_keys(spec, frame[source])
    encoded = pd.DataFrame(index=frame.index)
    category_counts: dict[str, int] = {}
    for rule in spec.categories:
        if rule.minimum is not None:
            mask = keys.ge(rule.minimum).fillna(False)
        else:
            mask = keys.isin(rule.values).fillna(False)
        encoded[rule.output] = mask.astype("int8")
        category_counts[rule.output] = int(mask.sum())

    category_sum = encoded[[rule.output for rule in spec.categories]].sum(axis=1)
    unmapped = (~missing) & (~invalid_format) & category_sum.eq(0)
    overlap = category_sum.gt(1)
    if spec.missing_column:
        encoded[spec.missing_column] = missing.astype("int8")
        full_sum = encoded[list(spec.output_columns)].sum(axis=1)
        row_sum_failure = full_sum.ne(1)
        category_counts[spec.missing_column] = int(missing.sum())
    else:
        row_sum_failure = category_sum.ne(1)

    invalid_or_unmapped = invalid_format | unmapped
    state = {
        "source_nonmissing": int((~missing).sum()),
        "source_missing": int(missing.sum()),
        "encoded_valid_rows": int((~row_sum_failure).sum()),
        "invalid_unmapped": int(invalid_or_unmapped.sum()),
        "overlap_rows": int(overlap.sum()),
        "row_sum_failures": int(row_sum_failure.sum()),
        "category_counts": category_counts,
    }
    if state["invalid_unmapped"] or state["overlap_rows"] or state["row_sum_failures"]:
        examples = frame.loc[
            invalid_or_unmapped | overlap | row_sum_failure, ["CHOICE_ID", source]
        ].head(10)
        if not examples.empty:
            print(f"\nENCODING FAILURE EXAMPLES: {source}")
            print(examples.to_string(index=False))
        raise ValueError(
            f"Categorical encoding failed for {source}: "
            f"invalid/unmapped={state['invalid_unmapped']}, overlap={state['overlap_rows']}, "
            f"row-sum failures={state['row_sum_failures']}."
        )
    return encoded[list(spec.output_columns)], state


def apply_binary(
    frame: pd.DataFrame,
    spec: TransformSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = spec.source_fields[0]
    stripped = frame[source].astype("string").str.strip()
    missing = stripped.eq("") | stripped.isna()
    values = pd.to_numeric(stripped.mask(missing, pd.NA), errors="coerce")
    invalid = (~missing) & (values.isna() | ~values.isin([0, 1]))
    unsupported_missing = missing & (not bool(spec.missing_column))
    if invalid.any() or unsupported_missing.any():
        print(
            frame.loc[invalid | unsupported_missing, ["CHOICE_ID", source]]
            .head(10)
            .to_string(index=False)
        )
        raise ValueError(
            f"Binary validation failed for {source}: invalid={int(invalid.sum())}, "
            f"missing without indicator={int(unsupported_missing.sum())}."
        )
    encoded = pd.DataFrame(index=frame.index)
    encoded[source] = values.astype("Int64")
    if spec.missing_column:
        encoded[spec.missing_column] = missing.astype("int8")
    state = {
        "source_nonmissing": int((~missing).sum()),
        "source_missing": int(missing.sum()),
        "invalid_unmapped": int(invalid.sum()),
        "row_sum_failures": 0,
        "category_counts": {
            f"{source}=0": int(values.eq(0).sum()),
            f"{source}=1": int(values.eq(1).sum()),
            **(
                {spec.missing_column: int(missing.sum())}
                if spec.missing_column
                else {}
            ),
        },
    }
    return encoded[list(spec.output_columns)], state


def continuous_summary(values: pd.Series) -> dict[str, object]:
    valid = values.dropna()
    return {
        "missing": int(values.isna().sum()),
        "min": float(valid.min()),
        "median": float(valid.median()),
        "p95": float(valid.quantile(0.95)),
        "p99": float(valid.quantile(0.99)),
        "max": float(valid.max()),
    }


def apply_continuous(
    frame: pd.DataFrame,
    spec: TransformSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = spec.source_fields[0]
    stripped = frame[source].astype("string").str.strip()
    missing = stripped.eq("") | stripped.isna()
    values = pd.to_numeric(stripped.mask(missing, pd.NA), errors="coerce")
    invalid = (~missing) & values.isna()
    if invalid.any():
        raise ValueError(f"Continuous field {source} contains non-numeric populated values.")
    if spec.require_positive and (missing.any() or values.dropna().le(0).any()):
        raise ValueError(
            f"{source} must be complete and positive for its log transformation; "
            f"missing={int(missing.sum())}, non-positive={int(values.dropna().le(0).sum())}."
        )

    encoded = pd.DataFrame(index=frame.index)
    encoded[source] = values.astype(float)
    log_error = 0.0
    if spec.kind == "continuous_log":
        log_output = spec.output_columns[1]
        encoded[log_output] = np.log(values.astype(float))
        log_error = float(
            np.nanmax(np.abs(encoded[log_output].to_numpy() - np.log(values.to_numpy())))
        )
    elif spec.missing_column:
        encoded[spec.missing_column] = missing.astype("int8")
    state = {
        "source_nonmissing": int((~missing).sum()),
        "source_missing": int(missing.sum()),
        "invalid_unmapped": int(invalid.sum()),
        "row_sum_failures": 0,
        "category_counts": {},
        "continuous": continuous_summary(values),
        "log_max_abs_error": log_error,
    }
    return encoded[list(spec.output_columns)], state


def apply_registry(
    frame: pd.DataFrame,
    registry: Sequence[TransformSpec],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    pieces: list[pd.DataFrame] = []
    states: dict[str, dict[str, object]] = {}
    for spec in registry:
        if spec.kind == "categorical":
            encoded, state = apply_categorical(frame, spec)
        elif spec.kind == "binary":
            encoded, state = apply_binary(frame, spec)
        elif spec.kind in {"continuous", "continuous_log"}:
            encoded, state = apply_continuous(frame, spec)
        else:
            raise ValueError(f"Unsupported transformation kind in registry: {spec.kind}")
        pieces.append(encoded)
        states[spec.key] = state
    features = pd.concat(pieces, axis=1)
    expected = feature_columns(registry)
    if list(features.columns) != expected:
        raise AssertionError("Encoded feature order differs from the transformation registry.")
    return features, states


def validate_and_build(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    validate_registry(TRANSFORMATION_REGISTRY)
    retained = select_single_driver_branch(source)

    choice = numeric(retained["CHOICE"])
    av_1 = numeric(retained["AV_1"])
    av_2 = numeric(retained["AV_2"])
    if not choice.isin([1, 2]).all():
        raise ValueError("Retained CHOICE must be restricted to alternatives 1 and 2.")
    if av_1.ne(1).any() or av_2.ne(1).any():
        raise ValueError("Retained AV_1 and AV_2 must both equal one.")
    if retained["CHOICE_ID"].eq("").any() or not retained["CHOICE_ID"].is_unique:
        raise ValueError("Retained CHOICE_ID must be populated and unique.")

    features, states = apply_registry(retained, TRANSFORMATION_REGISTRY)
    repeat_features, repeat_states = apply_registry(retained, TRANSFORMATION_REGISTRY)
    if not features.equals(repeat_features) or states != repeat_states:
        raise AssertionError("Transformation registry is not deterministic.")

    final = pd.concat([retained[CONTROL_COLUMNS].copy(), features], axis=1)
    expected_ids = set(
        source.loc[
            numeric(source["SINGLE_ACTIVE_DRIVER_HH"]).eq(1), "CHOICE_ID"
        ]
    )
    if len(final) != len(retained):
        raise AssertionError("Encoding changed the authoritative branch row count.")
    if set(final["CHOICE_ID"]) != expected_ids:
        raise AssertionError("Output CHOICE_ID set differs from the authoritative branch.")
    if not final["CHOICE_ID"].is_unique:
        raise AssertionError("Output CHOICE_ID is not unique.")
    if numeric(final["MULTI_ACTIVE_DRIVER_HH"]).ne(0).any():
        raise AssertionError("Output contains multi-driver rows.")
    if not final[CONTROL_COLUMNS].equals(retained[CONTROL_COLUMNS]):
        raise AssertionError("An identifier/control value or its source order changed.")
    if list(final.columns) != [*CONTROL_COLUMNS, *feature_columns(TRANSFORMATION_REGISTRY)]:
        raise AssertionError("Final column order differs from the documented registry order.")
    return retained, final, states


def add_qa(
    rows: list[dict[str, object]],
    section: str,
    metric: str,
    value: object,
    description: str,
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "description": description,
        }
    )


def build_qa(
    source: pd.DataFrame,
    retained: pd.DataFrame,
    final: pd.DataFrame,
    states: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    add_qa(rows, "SAMPLE", "source candidate rows", len(source), "authoritative allocation-candidate rows")
    add_qa(rows, "SAMPLE", "retained single-driver rows", len(retained), "SINGLE_ACTIVE_DRIVER_HH == 1")
    add_qa(rows, "SAMPLE", "unique CHOICE_ID", retained["CHOICE_ID"].nunique(), "retained choice occasions")
    add_qa(rows, "SAMPLE", "unique households", retained["H_ID"].nunique(), "retained H_ID")
    add_qa(
        rows,
        "SAMPLE",
        "excluded because multi-driver",
        int(numeric(source["MULTI_ACTIVE_DRIVER_HH"]).eq(1).sum()),
        "authoritative candidate rows not in this branch",
    )
    add_qa(rows, "SAMPLE", "duplicate CHOICE_ID", int(retained["CHOICE_ID"].duplicated().sum()), "retained rows")

    add_qa(rows, "BRANCH", "SINGLE_ACTIVE_DRIVER_HH != 1", int(numeric(retained["SINGLE_ACTIVE_DRIVER_HH"]).ne(1).sum()), "must be zero")
    add_qa(rows, "BRANCH", "MULTI_ACTIVE_DRIVER_HH != 0", int(numeric(retained["MULTI_ACTIVE_DRIVER_HH"]).ne(0).sum()), "must be zero")

    choice = numeric(retained["CHOICE"])
    add_qa(rows, "CHOICE STRUCTURE", "CHOICE 1", int(choice.eq(1).sum()), "chosen conceptual Car 1")
    add_qa(rows, "CHOICE STRUCTURE", "CHOICE 2", int(choice.eq(2).sum()), "chosen conceptual Car 2")
    add_qa(
        rows,
        "CHOICE STRUCTURE",
        "invalid CHOICE",
        int((~choice.isin([1, 2])).sum()),
        "must be zero",
    )
    add_qa(rows, "CHOICE STRUCTURE", "AV_1 != 1", int(numeric(retained["AV_1"]).ne(1).sum()), "must be zero")
    add_qa(rows, "CHOICE STRUCTURE", "AV_2 != 1", int(numeric(retained["AV_2"]).ne(1).sum()), "must be zero")

    for spec in TRANSFORMATION_REGISTRY:
        state = states[spec.key]
        if spec.kind == "categorical":
            add_qa(rows, "ENCODING VALIDATION", f"{spec.key}: source non-missing rows", state["source_nonmissing"], spec.source_fields[0])
            add_qa(rows, "ENCODING VALIDATION", f"{spec.key}: encoded valid rows", state["encoded_valid_rows"], "full-indicator block")
            add_qa(rows, "ENCODING VALIDATION", f"{spec.key}: invalid/unmapped source codes", state["invalid_unmapped"], "must be zero")
            add_qa(rows, "ENCODING VALIDATION", f"{spec.key}: category row-sum failures", state["row_sum_failures"], "must be zero")
        elif spec.kind == "binary":
            add_qa(rows, "ENCODING VALIDATION", f"{spec.key}: invalid binary values", state["invalid_unmapped"], spec.source_fields[0])

    analytical_sources: list[str] = []
    for spec in TRANSFORMATION_REGISTRY:
        for source_field in spec.source_fields:
            if source_field not in analytical_sources:
                analytical_sources.append(source_field)
    for field in analytical_sources:
        missing = int(retained[field].astype("string").str.strip().eq("").sum())
        add_qa(rows, "MISSINGNESS", field, missing, "blank/NA source values in retained branch")

    for spec in TRANSFORMATION_REGISTRY:
        if spec.grouping_qa:
            for category, count in states[spec.key]["category_counts"].items():
                add_qa(rows, "GROUPING QA", f"{spec.key}: {category}", count, spec.implemented_transformation)

    for group in ["powertrain", "segment", "vehicle_age", "holder", "status"]:
        signatures = [
            spec.symmetry_signature
            for spec in TRANSFORMATION_REGISTRY
            if spec.symmetry_group == group
        ]
        add_qa(
            rows,
            "VEHICLE SYMMETRY",
            f"{group}: identical Car 1/Car 2 mapping",
            int(len(signatures) == 2 and len(set(signatures)) == 1),
            signatures[0] if signatures else "missing registry mapping",
        )

    for field in [
        "TOUR_DISTANCE_KM",
        "TOUR_TRAVEL_TIME_MIN",
        "VEHICLE_AGE_1",
        "VEHICLE_AGE_2",
    ]:
        spec = next(item for item in TRANSFORMATION_REGISTRY if field in item.source_fields)
        state = states[spec.key]
        for metric, value in state["continuous"].items():
            add_qa(rows, "CONTINUOUS QA", f"{field}: {metric}", value, "retained single-driver branch")
        if spec.kind == "continuous_log":
            add_qa(rows, "CONTINUOUS QA", f"{field}: log max absolute error", state["log_max_abs_error"], "must be zero within floating-point arithmetic")

    unexpected = sum(
        int(state.get("invalid_unmapped", 0)) + int(state.get("row_sum_failures", 0))
        for state in states.values()
    )
    add_qa(rows, "OUTPUT", "final rows", len(final), "single-driver design matrix")
    add_qa(rows, "OUTPUT", "final columns", len(final.columns), "controls plus encoded feature superset")
    add_qa(rows, "OUTPUT", "encoded features", len(feature_columns(TRANSFORMATION_REGISTRY)), "registry-defined feature columns")
    add_qa(rows, "OUTPUT", "unique CHOICE_ID", final["CHOICE_ID"].nunique(), "must equal final rows")
    add_qa(rows, "OUTPUT", "rows with any unexpected encoding failure", unexpected, "must be zero")
    add_qa(rows, "OUTPUT", "global complete-case filtering performed", 0, "no optional missingness caused row deletion")
    add_qa(rows, "OUTPUT", "model estimation performed", 0, "feature construction only")
    add_qa(rows, "REFERENCE", "single-driver occasions", len(final), f"warning target only: {REFERENCE_OCCASIONS:,}")
    add_qa(rows, "REFERENCE", "single-driver households", final["H_ID"].nunique(), f"warning target only: {REFERENCE_HOUSEHOLDS:,}")
    add_qa(rows, "REGISTRY", "transformation entries", len(TRANSFORMATION_REGISTRY), "execution and Markdown share this registry")
    add_qa(rows, "REGISTRY", "registry SHA-256", registry_fingerprint(TRANSFORMATION_REGISTRY), "documentation/execution fingerprint")
    return pd.DataFrame(rows, columns=["section", "metric", "value", "description"])


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def registry_table(specs: Sequence[TransformSpec]) -> str:
    lines = [
        "| Attribute | Source field | Recommended treatment | Implemented transformation | Output columns | Recommended reference | Candidate use / caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in specs:
        caveat = spec.use + (f". {spec.notes}" if spec.notes else "")
        values = [
            spec.attribute,
            ", ".join(spec.source_fields),
            spec.recommended_treatment,
            spec.implemented_transformation + " " + spec.missing_treatment,
            "<br>".join(spec.output_columns),
            spec.recommended_reference,
            caveat,
        ]
        lines.append("| " + " | ".join(markdown_escape(value) for value in values) + " |")
    return "\n".join(lines)


def generate_markdown(
    retained: pd.DataFrame,
    qa: pd.DataFrame,
    states: dict[str, dict[str, object]],
) -> str:
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    fingerprint = registry_fingerprint(TRANSFORMATION_REGISTRY)
    blocks = {
        block: [spec for spec in TRANSFORMATION_REGISTRY if spec.block == block]
        for block in [
            "Tour & departure context",
            "Household & spatial context",
            "Person / driver context",
            "Vehicle attributes",
        ]
    }
    missing_rows = []
    seen: set[str] = set()
    for spec in TRANSFORMATION_REGISTRY:
        for field in spec.source_fields:
            if field not in seen:
                seen.add(field)
                count = int(retained[field].astype("string").str.strip().eq("").sum())
                missing_rows.append(f"| {field} | {count:,} | {count / len(retained):.1%} |")

    validation_failures = int(
        qa.loc[
            (qa["section"] == "OUTPUT")
            & (qa["metric"] == "rows with any unexpected encoding failure"),
            "value",
        ].iloc[0]
    )
    lines = [
        "# Single-driver MNL design-matrix encoding specification",
        "",
        "## 1. Source sample",
        "",
        f"- Authoritative input: `{ALLOCATION_CANDIDATE_MODEL_INPUT_PATH.as_posix()}`",
        "- Authoritative filter: `SINGLE_ACTIVE_DRIVER_HH == 1`; retained rows are also required to have `MULTI_ACTIVE_DRIVER_HH == 0`.",
        f"- Result: {retained['H_ID'].nunique():,} households and {len(retained):,} choice occasions.",
        f"- Generated: {generated}.",
        f"- Human-readable coding reference: `{DICTIONARY_NAME}`, worksheet `{DICTIONARY_SHEET}`; the workbook recommendations are modelling proposals operationalized here.",
        "- This artifact is a reusable encoded feature superset. No model was estimated.",
        f"- Registry fingerprint (SHA-256): `{fingerprint}`.",
        "",
        "## 2. General encoding rules",
        "",
        "- Categorical features retain a full, mutually exclusive indicator set. No reference category is dropped here; recommended references are recorded for later specification code.",
        "- Missing optional attributes do not remove rows. Categorical missingness is explicit, continuous missing values remain NA with an indicator where prescribed, and binary missing values remain NA with an explicit indicator where observed.",
        "- No raw categorical code is retained as a numeric predictor. Raw continuous and validated binary features remain available where specified.",
        "- No global complete-case restriction, interaction construction, stepwise selection, or branch-specific specification selection is performed.",
        "",
        "> These context variables are encoded here for reproducible downstream use. In the unlabelled vehicle-choice MNL, a generic alternative-invariant main effect cancels within a choice occasion. Context variables must enter through an identified alternative-specific structure and/or interactions with alternative-varying vehicle attributes according to the predefined specification.",
        "",
        "## 3. Tour & departure context",
        "",
        registry_table(blocks["Tour & departure context"]),
        "",
        "## 4. Household & spatial context",
        "",
        registry_table(blocks["Household & spatial context"]),
        "",
        "## 5. Person / driver context",
        "",
        registry_table(blocks["Person / driver context"]),
        "",
        "## 6. Vehicle attributes",
        "",
        registry_table(blocks["Vehicle attributes"]),
        "",
        "## 7. Alternative specification blocks",
        "",
        "- Purpose representation: use either the `TOUR_FIRST_PURPOSE` indicator block or the eight `TOUR_HAS_*` indicators as a predefined specification. Do not automatically stack both complete blocks.",
        "- V1 vehicle representation: Powertrain + Segment + Vehicle Age.",
        "- V2 vehicle representation: Powertrain + Status.",
        "- `STATUS_1/2` is an alternative composite representation to `SEGMENT_1/2 + VEHICLE_AGE_1/2`; do not automatically combine all of them.",
        "- Destination context is an extension because destination-field coverage is incomplete; any complete-case version must define its own restricted sample at estimation time.",
        "- Holder type is an extension because more than half of its source values are missing.",
        "- Household bicycle/pedelec counts and personal availability indicators overlap. Later specifications should choose representations based on behavioural meaning and multicollinearity/screening, not include all merely because they are available.",
        "",
        "## 8. Variables requiring later estimation-stage decisions",
        "",
        "- Tour distance: raw versus log functional form.",
        "- Tour travel time: raw versus log functional form.",
        "- Distance versus travel-time redundancy and conditioning before joint use.",
        "- Sparse first-purpose Home and Return categories.",
        "- Sparse education-purpose support.",
        "- Extremely sparse `HP_SEX` Diverse category.",
        "- Rare `AGE_UNDER18` category.",
        "- Vehicle-age centering, spline, or robust cap as sensitivity choices only.",
        "- Optional holder restriction or explicit holder-missingness treatment.",
        "- Destination-context coverage and any resulting estimation-sample restriction.",
        "",
        "## 9. Validation summary",
        "",
        f"- Rows / unique `CHOICE_ID`: {len(retained):,} / {retained['CHOICE_ID'].nunique():,}.",
        f"- Households: {retained['H_ID'].nunique():,}.",
        f"- Encoded feature columns: {len(feature_columns(TRANSFORMATION_REGISTRY)):,}.",
        f"- Unexpected encoding failures: {validation_failures:,}.",
        "- Every categorical block passed its full-indicator row-sum validation, every binary field passed 0/1 validation, both availability flags equal one, and Car 1/Car 2 vehicle mappings are symmetric.",
        "- Distance and travel time were positive for all retained rows and their log transformations passed exact recomputation checks.",
        "",
        "### Source-field missingness",
        "",
        "| Source field | Missing N | Missing share |",
        "|---|---:|---:|",
        *missing_rows,
        "",
        "### Continuous-variable summaries",
        "",
        "| Source field | Missing | Min | Median | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field in [
        "TOUR_DISTANCE_KM",
        "TOUR_TRAVEL_TIME_MIN",
        "VEHICLE_AGE_1",
        "VEHICLE_AGE_2",
    ]:
        spec = next(item for item in TRANSFORMATION_REGISTRY if field in item.source_fields)
        summary = states[spec.key]["continuous"]
        lines.append(
            f"| {field} | {summary['missing']:,} | {summary['min']:.6g} | "
            f"{summary['median']:.6g} | {summary['p95']:.6g} | "
            f"{summary['p99']:.6g} | {summary['max']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_markdown_registry(markdown: str) -> None:
    if registry_fingerprint(TRANSFORMATION_REGISTRY) not in markdown:
        raise AssertionError("Markdown does not carry the execution-registry fingerprint.")
    for spec in TRANSFORMATION_REGISTRY:
        if spec.attribute not in markdown:
            raise AssertionError(f"Markdown omitted registry attribute: {spec.attribute}")
        for column in spec.output_columns:
            if column not in markdown:
                raise AssertionError(f"Markdown omitted registry output column: {column}")


def warn_reference_drift(retained: pd.DataFrame) -> None:
    observed = {
        "single-driver households": retained["H_ID"].nunique(),
        "single-driver occasions": len(retained),
    }
    references = {
        "single-driver households": REFERENCE_HOUSEHOLDS,
        "single-driver occasions": REFERENCE_OCCASIONS,
    }
    mismatches = [
        (label, observed[label], references[label])
        for label in references
        if observed[label] != references[label]
    ]
    if mismatches:
        print("\nWARNING: REFERENCE COUNTS CHANGED")
        for label, actual, reference in mismatches:
            print(f"- {label}: observed {actual:,}; warning target only {reference:,}")


def write_outputs(final: pd.DataFrame, qa: pd.DataFrame, markdown: str) -> None:
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        final.to_csv(handle, index=False)
    with SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        qa.to_csv(handle, index=False)
    with SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH.open("x", encoding="utf-8") as handle:
        handle.write(markdown)

    saved = read_strings(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH)
    if list(saved.columns) != list(final.columns) or len(saved) != len(final):
        raise AssertionError("Saved design matrix failed shape/schema read-back validation.")
    if not saved["CHOICE_ID"].equals(final["CHOICE_ID"].astype(str)):
        raise AssertionError("Saved design matrix changed CHOICE_ID order during serialization.")
    saved_qa = read_strings(SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH)
    if list(saved_qa.columns) != ["section", "metric", "value", "description"]:
        raise AssertionError("Saved QA schema changed during serialization.")
    if SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH.read_text(encoding="utf-8") != markdown:
        raise AssertionError("Saved Markdown differs from the generated registry documentation.")


def print_summary(
    retained: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    print("\nSINGLE-DRIVER MNL DESIGN MATRIX")
    print(f"source candidate rows: {len(read_strings(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH)):,}")
    print(f"retained households / occasions: {retained['H_ID'].nunique():,} / {len(retained):,}")
    print(f"final shape: {len(final):,} rows x {len(final.columns):,} columns")
    print(f"encoded features: {len(feature_columns(TRANSFORMATION_REGISTRY)):,}")
    print("unexpected source codes: none")
    print("major missingness:")
    for field in [
        "TOUR_WORST_XMSTADT_ZO",
        "TOUR_WORST_QUALI_OPNV_ZO",
        "carsharing",
        "mobein",
        "HOLDER_1",
        "HOLDER_2",
    ]:
        missing = int(retained[field].astype("string").str.strip().eq("").sum())
        print(f"- {field}: {missing:,} ({missing / len(retained):.1%})")
    print("OUTPUTS:")
    print(SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH.resolve())
    print(SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH.resolve())
    print(SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH.resolve())
    print(
        "No model estimation, interaction construction, joint alternatives, or "
        "global complete-case filtering was performed."
    )


def main() -> None:
    assert_outputs_absent()
    required_inputs = [ALLOCATION_CANDIDATE_MODEL_INPUT_PATH]
    missing_inputs = [path for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Required accepted input(s) not found: {missing_inputs}")

    protected_paths = [
        path
        for path in [
            ALLOCATION_CANDIDATE_MODEL_INPUT_PATH,
            STRICT_MODEL_INPUT_PATH,
            SENSITIVITY_MODEL_INPUT_PATH,
        ]
        if path.exists()
    ]
    protected_before = file_hashes(protected_paths)
    source = read_strings(ALLOCATION_CANDIDATE_MODEL_INPUT_PATH)
    retained, final, states = validate_and_build(source)
    warn_reference_drift(retained)
    qa = build_qa(source, retained, final, states)
    markdown = generate_markdown(retained, qa, states)
    validate_markdown_registry(markdown)

    if file_hashes(protected_paths) != protected_before:
        raise AssertionError("An accepted candidate/strict/sensitivity input changed before output writing.")
    write_outputs(final, qa, markdown)
    if file_hashes(protected_paths) != protected_before:
        raise AssertionError("An accepted candidate/strict/sensitivity input changed during execution.")
    print_summary(retained, final)


if __name__ == "__main__":
    main()
