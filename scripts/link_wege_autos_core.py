from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

HH_PATH = DATA_PROCESSED / "hh_selected_raw.csv"
PERSONS_PATH = DATA_PROCESSED / "persons_selected_raw.csv"
TRIPS_PATH = DATA_PROCESSED / "trips_selected_raw.csv"
CARS_PATH = DATA_PROCESSED / "cars_selected_raw.csv"

OUT_WIDE_PATH = DATA_PROCESSED / "02_trip_vehicle_wide.csv"
OUT_FUNNEL_PATH = DATA_PROCESSED / "02_mnl_estimation_funnel.csv"
OUT_MD_PATH = DATA_PROCESSED / "02_trial_model_variables.md"

HH_COLS = [
    "H_ID",
    "H_ANZAUTO",
    "hhgr_gr",
    "oek_status",
    "RegioStaR7",
]
PERSON_COLS = [
    "H_ID",
    "P_ID",
    "HP_SEX",
    "alter_gr5",
    "Taet",
]
TRIP_COLS = [
    "H_ID",
    "P_ID",
    "W_ID",
    "W_WAUTO",
    "pkw_fmf",
    "zweck",
    "wegkm_imp",
    "wegmin_imp2",
    "ST_WOTAG",
    "ST_MONAT",
]
CAR_COLS = [
    "H_ID",
    "A_ID",
    "A_ANTRIEB",
    "A_BAUJ",
    "seg_kba_gr",
    "status",
]

ALT_IDS = ["1", "2", "3"]
CAR_ATTRS = ["A_ANTRIEB", "A_BAUJ", "seg_kba_gr", "status"]
CAR_ATTR_WIDE_NAMES = {
    "A_ANTRIEB": "antrieb",
    "A_BAUJ": "A_BAUJ",
    "seg_kba_gr": "seg_kba_gr",
    "status": "status",
}

COLUMN_ALIASES = {
    # The selected raw export in this workspace uses lower-case taet. Keep the
    # trial model column as requested while resolving this case-only variant.
    "Taet": ["Taet", "taet"],
}

DESIGN_MISSING_CODES = {
    "",
    "9",
    "94",
    "95",
    "99",
    "101",
    "202",
    "206",
    "402",
    "403",
    "701",
    "706",
    "708",
    "995",
}
NUMERIC_MISSING_CODES = DESIGN_MISSING_CODES | {
    "9994",
    "9995",
    "9999",
    "99994",
    "99995",
    "99999",
    "999994",
    "999995",
    "999999",
    "70701",
}


def resolve_selected_raw_path(path: Path) -> Path:
    if path.exists():
        return path

    raise FileNotFoundError(
        f"Required selected raw input not found: {path}. "
        "Expected the selected raw files without a 01_ prefix in data_processed."
    )


def strip_all_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def resolve_columns(path: Path, requested_cols: list[str]) -> tuple[list[str], dict[str, str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    header_set = set(header)
    usecols: list[str] = []
    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for requested in requested_cols:
        candidates = COLUMN_ALIASES.get(requested, [requested])
        actual = next((candidate for candidate in candidates if candidate in header_set), None)
        if actual is None:
            missing.append(requested)
            continue
        usecols.append(actual)
        if actual != requested:
            rename_map[actual] = requested
            print(f"Resolved requested column {requested!r} from source column {actual!r}.")

    if missing:
        raise ValueError(
            f"{path.name} is missing required trial-model column(s): {missing}. "
            "This script does not silently ignore missing selected variables."
        )

    return usecols, rename_map


def read_selected_raw(
    path: Path,
    cols: list[str],
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    resolved_path = resolve_selected_raw_path(path)
    usecols, rename_map = resolve_columns(resolved_path, cols)
    df = reader(resolved_path, usecols=usecols, dtype=str, keep_default_na=False)
    df = df.rename(columns=rename_map)
    return strip_all_string_columns(df[cols].copy())


def present(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.notna() & values.ne("") & values.str.lower().ne("nan")


def in_codes(series: pd.Series, valid_codes: set[str]) -> pd.Series:
    values = series.astype("string").str.strip()
    return present(values) & values.isin(valid_codes)


def not_special_code(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return present(values) & ~values.isin(DESIGN_MISSING_CODES)


def valid_int_range(series: pd.Series, low: int, high: int) -> pd.Series:
    values = series.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    return present(values) & ~values.isin(NUMERIC_MISSING_CODES) & numeric.between(low, high)


def valid_positive_number(series: pd.Series, max_value: float | None = None) -> pd.Series:
    values = series.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    mask = present(values) & ~values.isin(NUMERIC_MISSING_CODES) & numeric.gt(0)
    if max_value is not None:
        mask &= numeric.le(max_value)
    return mask


def valid_car_attribute_mask(cars: pd.DataFrame) -> pd.Series:
    return (
        in_codes(cars["A_ANTRIEB"], {"1", "2", "3", "4", "5", "6", "7"})
        & valid_int_range(cars["A_BAUJ"], 1886, 2023)
        & in_codes(cars["seg_kba_gr"], {"1", "2", "3", "4"})
        & in_codes(cars["status"], {"1", "2", "3"})
    )


def valid_trip_model_mask(df: pd.DataFrame) -> pd.Series:
    return (
        in_codes(df["W_WAUTO"], set(ALT_IDS))
        & in_codes(df["pkw_fmf"], {"1"})
        & in_codes(df["zweck"], {str(value) for value in range(1, 11)})
        & valid_positive_number(df["wegkm_imp"], max_value=1000)
        & valid_positive_number(df["wegmin_imp2"], max_value=1440)
        & valid_int_range(df["ST_WOTAG"], 1, 7)
        & valid_int_range(df["ST_MONAT"], 1, 12)
    )


def valid_hh_model_mask(df: pd.DataFrame) -> pd.Series:
    return (
        valid_int_range(df["H_ANZAUTO"], 0, 3)
        & in_codes(df["hhgr_gr"], {"1", "2", "3", "4", "5"})
        & in_codes(df["oek_status"], {"1", "2", "3", "4", "5"})
        & in_codes(df["RegioStaR7"], {"71", "72", "73", "74", "75", "76", "77"})
    )


def valid_person_model_mask(df: pd.DataFrame) -> pd.Series:
    return (
        in_codes(df["HP_SEX"], {"1", "2", "3"})
        & valid_int_range(df["alter_gr5"], 1, 15)
        & in_codes(df["Taet"], {"1", "2", "3", "4", "5"})
    )


def ensure_unique_key(df: pd.DataFrame, key_cols: list[str], name: str) -> None:
    duplicates = df.duplicated(key_cols, keep=False)
    if duplicates.any():
        example = df.loc[duplicates, key_cols].head(10)
        raise ValueError(f"{name} has duplicate keys {key_cols}; examples:\n{example}")


def print_removed(previous_n: int, current_n: int, label: str) -> None:
    removed = previous_n - current_n
    print(f"{label}: kept {current_n:,} rows; removed {removed:,}.")


def add_funnel_step(
    funnel_rows: list[dict[str, object]],
    step: str,
    df: pd.DataFrame,
) -> None:
    n_persons = pd.NA
    if {"H_ID", "P_ID"}.issubset(df.columns):
        n_persons = df[["H_ID", "P_ID"]].drop_duplicates().shape[0]

    funnel_rows.append(
        {
            "step": step,
            "n_households": df["H_ID"].nunique() if "H_ID" in df.columns else pd.NA,
            "n_persons": n_persons,
            "n_trip_rows": len(df),
        }
    )


def add_vehicle_wide_columns(base: pd.DataFrame, model_usable_cars: pd.DataFrame) -> pd.DataFrame:
    wide = base.copy()
    for alt_id in ALT_IDS:
        alt = model_usable_cars.loc[
            model_usable_cars["A_ID"].eq(alt_id),
            ["H_ID", *CAR_ATTRS],
        ].copy()
        rename = {
            attr: f"{CAR_ATTR_WIDE_NAMES[attr]}_{alt_id}"
            for attr in CAR_ATTRS
        }
        alt = alt.rename(columns=rename)
        wide = wide.merge(alt, on="H_ID", how="left", validate="many_to_one")
        attr_cols = list(rename.values())
        wide[f"av_{alt_id}"] = wide[attr_cols].notna().all(axis=1).astype("int64")

    return wide


def chosen_alternative_available(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for alt_id in ALT_IDS:
        mask |= df["choice"].eq(alt_id) & df[f"av_{alt_id}"].eq(1)
    return mask


def assert_available_attrs_complete(df: pd.DataFrame) -> None:
    problems: dict[str, int] = {}
    for alt_id in ALT_IDS:
        attr_cols = [
            f"{CAR_ATTR_WIDE_NAMES[attr]}_{alt_id}"
            for attr in CAR_ATTRS
        ]
        missing_available = df[f"av_{alt_id}"].eq(1) & df[attr_cols].isna().any(axis=1)
        if missing_available.any():
            problems[alt_id] = int(missing_available.sum())

    if problems:
        raise ValueError(
            "Available alternatives have missing selected vehicle attributes: "
            f"{problems}"
        )


def coerce_final_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    int_cols = [
        "choice",
        "av_1",
        "av_2",
        "av_3",
        "n_valid_alternatives",
        "W_WAUTO",
        "pkw_fmf",
        "zweck",
        "wegmin_imp2",
        "ST_WOTAG",
        "ST_MONAT",
        "H_ANZAUTO",
        "hhgr_gr",
        "oek_status",
        "RegioStaR7",
        "HP_SEX",
        "alter_gr5",
        "Taet",
    ]
    for alt_id in ALT_IDS:
        int_cols.extend(
            [
                f"antrieb_{alt_id}",
                f"A_BAUJ_{alt_id}",
                f"seg_kba_gr_{alt_id}",
                f"status_{alt_id}",
            ]
        )

    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["wegkm_imp"] = pd.to_numeric(out["wegkm_imp"], errors="coerce")
    return out


def write_trial_markdown(
    path: Path,
    final_wide: pd.DataFrame,
    funnel: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    choice_dist = final_wide["choice"].value_counts().sort_index()
    alt_dist = final_wide["n_valid_alternatives"].value_counts().sort_index()

    md = f"""# Trial trip-level vehicle-allocation MNL sample

This script builds a wide-form, trip-level vehicle choice estimation sample from the selected raw files in `data_processed`.

## Inputs

- Household: `{resolve_selected_raw_path(HH_PATH).name}`
- Persons: `{resolve_selected_raw_path(PERSONS_PATH).name}`
- Trips: `{resolve_selected_raw_path(TRIPS_PATH).name}`
- Cars: `{resolve_selected_raw_path(CARS_PATH).name}`

## Trial variables

Trip variables: `{", ".join(TRIP_COLS)}`

Household variables: `{", ".join(HH_COLS)}`

Person variables: `{", ".join(PERSON_COLS)}`

Vehicle variables: `{", ".join(CAR_COLS)}`

No holder, charging, parking, home office, attitude, satisfaction, health, mobility limitation, or other module-heavy variables are used.

## Sample unit and choice set

Each row is one trip-level vehicle choice occasion. Trips are restricted to Pkw driver trips with `pkw_fmf == 1`, a chosen household vehicle ID `W_WAUTO` in `1, 2, 3`, and a chosen car that can be matched to `Autos` by `(H_ID, W_WAUTO) = (H_ID, A_ID)`.

Households are kept only when they have at least two model-usable recorded household vehicles among `A_ID` 1, 2, and 3. A vehicle is model-usable only when all selected vehicle variables are valid. The sample does not require a household to have at least two linked driver trips and does not require the household to have used at least two different cars on the Stichtag.

## Validity rules

- ID and code columns are stripped as strings before linkage.
- `W_WAUTO` and `A_ID` are valid only for `1`, `2`, `3`.
- `pkw_fmf` is valid only for driver value `1`.
- Distance and duration are converted to numeric and must be positive; special missing codes such as `99994`, `99995`, `99999`, and `70701` are invalid.
- `RegioStaR7` must be 71-77 and `oek_status` must be 1-5.
- Categorical trial variables exclude MiD special/design-missing codes including `9`, `99`, `94`, `95`, `101`, `202`, `206`, `402`, `403`, `701`, `706`, and `708` where applicable.
- Available alternatives (`av_k == 1`) have complete selected vehicle attributes; unavailable alternative attributes are left blank in the raw wide file.

## Diagnostics not used as filters

- Households with at least two final linked driver trips: {diagnostics["households_with_at_least_two_final_trips"]:,}
- Households using at least two different cars in final rows: {diagnostics["households_using_at_least_two_cars"]:,}

## Final sample QA

- Trip rows: {len(final_wide):,}
- Households: {final_wide["H_ID"].nunique():,}
- Persons/drivers: {final_wide[["H_ID", "P_ID"]].drop_duplicates().shape[0]:,}

Choice distribution:

{choice_dist.to_string()}

`n_valid_alternatives` distribution:

{alt_dist.to_string()}

## Funnel

{dataframe_to_markdown(funnel)}
"""
    path.write_text(md, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    rows = [[format_markdown_cell(value) for value in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def build_trip_vehicle_mnl_sample(
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh = read_selected_raw(HH_PATH, HH_COLS, reader=reader)
    persons = read_selected_raw(PERSONS_PATH, PERSON_COLS, reader=reader)
    trips = read_selected_raw(TRIPS_PATH, TRIP_COLS, reader=reader)
    cars = read_selected_raw(CARS_PATH, CAR_COLS, reader=reader)

    ensure_unique_key(hh, ["H_ID"], "Household selected raw")
    ensure_unique_key(persons, ["H_ID", "P_ID"], "Person selected raw")
    ensure_unique_key(cars, ["H_ID", "A_ID"], "Cars selected raw")

    funnel_rows: list[dict[str, object]] = []
    add_funnel_step(funnel_rows, "1. all trip rows", trips)

    trips_with_vehicle_choice = trips[in_codes(trips["W_WAUTO"], set(ALT_IDS))].copy()
    print_removed(len(trips), len(trips_with_vehicle_choice), "+ W_WAUTO in {1,2,3}")
    add_funnel_step(funnel_rows, "2. + W_WAUTO in {1,2,3}", trips_with_vehicle_choice)

    driver_trips = trips_with_vehicle_choice[in_codes(trips_with_vehicle_choice["pkw_fmf"], {"1"})].copy()
    print_removed(len(trips_with_vehicle_choice), len(driver_trips), "+ Pkw driver trip")
    add_funnel_step(funnel_rows, "3. + Pkw driver trip", driver_trips)

    # Required construction: start from household cars with A_ID 1/2/3.
    valid_cars = cars[cars["A_ID"].astype(str).str.strip().isin(["1", "2", "3"])].copy()
    print_removed(len(cars), len(valid_cars), "Cars: A_ID in {1,2,3}")

    car_attr_mask = valid_car_attribute_mask(valid_cars)
    model_usable_cars = valid_cars[car_attr_mask].copy()
    print_removed(len(valid_cars), len(model_usable_cars), "Cars: selected vehicle attributes valid")

    n_alt = (
        model_usable_cars
        .groupby("H_ID")["A_ID"]
        .nunique()
        .rename("n_valid_alternatives")
    )
    mnl_hh_ids = n_alt[n_alt >= 2].index

    matched_choice_ids = valid_cars[["H_ID", "A_ID"]].drop_duplicates()
    matched_choice = driver_trips.merge(
        matched_choice_ids,
        left_on=["H_ID", "W_WAUTO"],
        right_on=["H_ID", "A_ID"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["A_ID"])
    print_removed(len(driver_trips), len(matched_choice), "+ chosen car matched in Autos")
    add_funnel_step(funnel_rows, "4. + chosen car matched in Autos", matched_choice)

    matched_choice = matched_choice.merge(
        n_alt.reset_index(),
        on="H_ID",
        how="left",
        validate="many_to_one",
    )
    two_alt_hh = matched_choice[matched_choice["H_ID"].isin(mnl_hh_ids)].copy()
    print_removed(
        len(matched_choice),
        len(two_alt_hh),
        "+ household has at least 2 model-usable alternatives in Autos",
    )
    add_funnel_step(
        funnel_rows,
        "5. + household has at least 2 model-usable alternatives in Autos",
        two_alt_hh,
    )

    merged = two_alt_hh.merge(hh, on="H_ID", how="left", validate="many_to_one")
    merged = merged.merge(persons, on=["H_ID", "P_ID"], how="left", validate="many_to_one")
    merge_available_cols = [
        col
        for col in [*HH_COLS, *PERSON_COLS]
        if col not in {"H_ID", "P_ID"}
    ]
    merged_available = merged[merged[merge_available_cols].apply(present).all(axis=1)].copy()
    print_removed(len(two_alt_hh), len(merged_available), "+ merged HH/person variables available")
    add_funnel_step(
        funnel_rows,
        "6. + merged HH/person variables available",
        merged_available,
    )

    wide = add_vehicle_wide_columns(merged_available, model_usable_cars)
    wide["choice"] = wide["W_WAUTO"]
    final_valid_mask = (
        valid_trip_model_mask(wide)
        & valid_hh_model_mask(wide)
        & valid_person_model_mask(wide)
        & chosen_alternative_available(wide)
        & pd.to_numeric(wide["n_valid_alternatives"], errors="coerce").ge(2)
    )
    final_wide = wide[final_valid_mask].copy()
    print_removed(len(wide), len(final_wide), "+ all selected trial-model variables valid")

    final_wide = coerce_final_types(final_wide)
    assert_available_attrs_complete(final_wide)

    output_cols = [
        "H_ID",
        "P_ID",
        "W_ID",
        "choice",
        "n_valid_alternatives",
        "av_1",
        "av_2",
        "av_3",
        "W_WAUTO",
        "pkw_fmf",
        "zweck",
        "wegkm_imp",
        "wegmin_imp2",
        "ST_WOTAG",
        "ST_MONAT",
        "H_ANZAUTO",
        "hhgr_gr",
        "oek_status",
        "RegioStaR7",
        "HP_SEX",
        "alter_gr5",
        "Taet",
    ]
    for alt_id in ALT_IDS:
        output_cols.extend(
            [
                f"antrieb_{alt_id}",
                f"A_BAUJ_{alt_id}",
                f"seg_kba_gr_{alt_id}",
                f"status_{alt_id}",
            ]
        )
    final_wide = final_wide[output_cols].sort_values(["H_ID", "P_ID", "W_ID"]).reset_index(drop=True)

    add_funnel_step(
        funnel_rows,
        "7. + all selected trial-model variables valid",
        final_wide,
    )
    add_funnel_step(funnel_rows, "8. final wide MNL sample", final_wide)
    funnel = pd.DataFrame(funnel_rows)

    diagnostics = {
        "households_with_at_least_two_final_trips": int(
            final_wide.groupby("H_ID").size().ge(2).sum()
        ),
        "households_using_at_least_two_cars": int(
            final_wide.groupby("H_ID")["choice"].nunique().ge(2).sum()
        ),
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    final_wide.to_csv(OUT_WIDE_PATH, index=False)
    funnel.to_csv(OUT_FUNNEL_PATH, index=False)
    write_trial_markdown(OUT_MD_PATH, final_wide, funnel, diagnostics)

    return final_wide, funnel


def main() -> None:
    final_wide, _ = build_trip_vehicle_mnl_sample()

    print("\nQA summary")
    print(f"Final trip rows: {len(final_wide):,}")
    print(f"Final households: {final_wide['H_ID'].nunique():,}")
    print(f"Final unique persons/drivers: {final_wide[['H_ID', 'P_ID']].drop_duplicates().shape[0]:,}")
    print("\nChoice distribution:")
    print(final_wide["choice"].value_counts().sort_index())
    print("\nn_valid_alternatives distribution:")
    print(final_wide["n_valid_alternatives"].value_counts().sort_index())
    print("\nSaved file paths:")
    print(f"- {OUT_WIDE_PATH}")
    print(f"- {OUT_FUNNEL_PATH}")
    print(f"- {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
