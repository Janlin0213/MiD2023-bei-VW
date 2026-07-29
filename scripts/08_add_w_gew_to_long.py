from __future__ import annotations

import os
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

LONG_PATH = DATA_PROCESSED / "mnl_vehicle_choice_long_base.csv"
TRIPS_PATH = DATA_PROCESSED / "trips_home_chain_enriched.csv"
ARCHIVE_DIR = DATA_PROCESSED / "archive"
BACKUP_PATH = ARCHIVE_DIR / "mnl_vehicle_choice_long_base_unweighted.csv"
TMP_PATH = DATA_PROCESSED / "mnl_vehicle_choice_long_base.csv.tmp"
QA_PATH = DATA_PROCESSED / "phase3_weight_merge_QA.csv"

EXPECTED_ROWS = 61_632
EXPECTED_CHOICES = 30_816
EXPECTED_ALTERNATIVES = {1, 2}


def read_csv_strings(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def parse_alt_set(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).apply(lambda s: set(numeric(s).dropna().astype(int)))


def assert_long_structure(long: pd.DataFrame, label: str) -> None:
    assert len(long) == EXPECTED_ROWS, f"{label}: expected {EXPECTED_ROWS} rows, found {len(long)}."
    assert long["CHOICE_ID"].nunique() == EXPECTED_CHOICES, (
        f"{label}: expected {EXPECTED_CHOICES} unique CHOICE_ID values, found {long['CHOICE_ID'].nunique()}."
    )
    row_counts = long.groupby("CHOICE_ID", sort=False).size()
    assert row_counts.eq(2).all(), f"{label}: every CHOICE_ID must have exactly two alternative rows."
    chosen_sum = long.groupby("CHOICE_ID", sort=False)["chosen"].apply(lambda s: numeric(s).sum())
    assert chosen_sum.eq(1).all(), f"{label}: every CHOICE_ID must have exactly one chosen row."
    alternative_sets = long.set_index("CHOICE_ID").groupby(level=0)["alternative_A_ID"].apply(
        lambda s: set(numeric(s).dropna().astype(int))
    )
    assert alternative_sets.eq(EXPECTED_ALTERNATIVES).all(), (
        f"{label}: every CHOICE_ID must have alternative_A_ID set exactly {EXPECTED_ALTERNATIVES}."
    )
    assert not numeric(long["alternative_A_ID"]).eq(3).any(), f"{label}: alternative_A_ID == 3 found."
    assert not long.duplicated(["CHOICE_ID", "alternative_A_ID"]).any(), (
        f"{label}: duplicate CHOICE_ID + alternative_A_ID rows found."
    )
    assert long["SOURCE_ROW_ID"].nunique() == EXPECTED_CHOICES, (
        f"{label}: expected {EXPECTED_CHOICES} unique SOURCE_ROW_ID values."
    )


def weight_stats(values: pd.Series) -> dict[str, float | int]:
    stats = {
        "missing W_GEW values": int(values.isna().sum()),
        "zero W_GEW values": int(values.eq(0).sum()),
        "negative W_GEW values": int(values.lt(0).sum()),
        "minimum W_GEW": float(values.min()),
        "maximum W_GEW": float(values.max()),
        "mean W_GEW": float(values.mean()),
        "median W_GEW": float(values.median()),
        "p01 W_GEW": float(values.quantile(0.01)),
        "p05 W_GEW": float(values.quantile(0.05)),
        "p95 W_GEW": float(values.quantile(0.95)),
        "p99 W_GEW": float(values.quantile(0.99)),
    }
    return stats


def validate_weight_source(weights: pd.DataFrame) -> pd.Series:
    assert "SOURCE_ROW_ID" in weights.columns, "SOURCE_ROW_ID missing from enriched trip file."
    assert "W_GEW" in weights.columns, "W_GEW missing from enriched trip file."
    assert weights["SOURCE_ROW_ID"].ne("").all(), "SOURCE_ROW_ID must be non-missing in enriched trip file."
    assert not weights["SOURCE_ROW_ID"].duplicated(keep=False).any(), (
        "SOURCE_ROW_ID must be unique in enriched trip file."
    )
    w_gew = numeric(weights["W_GEW"])
    bad = weights.loc[weights["W_GEW"].ne("") & w_gew.isna(), ["SOURCE_ROW_ID", "W_GEW"]]
    if not bad.empty:
        print("\nNON-NUMERIC W_GEW VALUES")
        print(bad.to_string(index=False))
    assert bad.empty, "W_GEW contains non-numeric values."
    return w_gew


def validate_merged(before: pd.DataFrame, merged: pd.DataFrame) -> pd.Series:
    assert_long_structure(merged, "after merge")
    assert set(before["CHOICE_ID"]) == set(merged["CHOICE_ID"]), "No CHOICE_ID values may be lost."
    assert set(before["SOURCE_ROW_ID"]) == set(merged["SOURCE_ROW_ID"]), "No SOURCE_ROW_ID values may be lost."
    merged_w = numeric(merged["W_GEW"])
    assert not merged_w.isna().any(), "W_GEW must be non-missing for every long-format row."
    inconsistent = merged.groupby("CHOICE_ID", sort=False)["W_GEW"].nunique(dropna=False)
    assert inconsistent.eq(1).all(), "W_GEW must have exactly one unique value within every CHOICE_ID."
    before_chosen = before["CHOSEN_A_ID"].value_counts().sort_index()
    after_chosen = merged["CHOSEN_A_ID"].value_counts().sort_index()
    assert before_chosen.equals(after_chosen), "CHOSEN_A_ID distribution changed during merge."
    return merged.groupby("CHOICE_ID", sort=False)["W_GEW"].first().pipe(numeric)


def qa_rows(before: pd.DataFrame, merged: pd.DataFrame, choice_weights: pd.Series) -> list[dict[str, object]]:
    inconsistent = merged.groupby("CHOICE_ID", sort=False)["W_GEW"].nunique(dropna=False).ne(1)
    rows = [
        {"metric": "long rows before merge", "value": len(before)},
        {"metric": "long rows after merge", "value": len(merged)},
        {"metric": "unique CHOICE_ID before merge", "value": before["CHOICE_ID"].nunique()},
        {"metric": "unique CHOICE_ID after merge", "value": merged["CHOICE_ID"].nunique()},
        {"metric": "missing W_GEW rows", "value": int(numeric(merged["W_GEW"]).isna().sum())},
        {"metric": "CHOICE_ID with inconsistent W_GEW", "value": int(inconsistent.sum())},
        {"metric": "zero W_GEW choices", "value": int(choice_weights.eq(0).sum())},
        {"metric": "negative W_GEW choices", "value": int(choice_weights.lt(0).sum())},
        {"metric": "minimum W_GEW", "value": float(choice_weights.min())},
        {"metric": "maximum W_GEW", "value": float(choice_weights.max())},
        {"metric": "mean W_GEW at choice level", "value": float(choice_weights.mean())},
        {"metric": "median W_GEW at choice level", "value": float(choice_weights.median())},
        {"metric": "p01 W_GEW", "value": float(choice_weights.quantile(0.01))},
        {"metric": "p05 W_GEW", "value": float(choice_weights.quantile(0.05))},
        {"metric": "p95 W_GEW", "value": float(choice_weights.quantile(0.95))},
        {"metric": "p99 W_GEW", "value": float(choice_weights.quantile(0.99))},
    ]
    return rows


def main() -> None:
    t0 = perf_counter()
    long = read_csv_strings(LONG_PATH)
    weights = read_csv_strings(TRIPS_PATH, usecols=["SOURCE_ROW_ID", "W_GEW"])

    if "W_GEW" in long.columns:
        raise ValueError("Canonical long file already contains W_GEW; refusing to create a duplicate merge.")

    assert_long_structure(long, "before merge")
    source_w = validate_weight_source(weights)
    source_stats = weight_stats(source_w)

    merged = long.merge(weights[["SOURCE_ROW_ID", "W_GEW"]], on="SOURCE_ROW_ID", how="left", validate="many_to_one")
    choice_weights = validate_merged(long, merged)
    choice_stats = weight_stats(choice_weights)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if BACKUP_PATH.exists():
        raise FileExistsError(f"Backup already exists; refusing to overwrite: {BACKUP_PATH}")
    shutil.copy2(LONG_PATH, BACKUP_PATH)

    merged.to_csv(TMP_PATH, index=False)
    reread = read_csv_strings(TMP_PATH)
    validate_merged(long, reread)
    pd.DataFrame(qa_rows(long, reread, choice_weights)).to_csv(QA_PATH, index=False)
    os.replace(TMP_PATH, LONG_PATH)

    print("\nW_GEW SOURCE DIAGNOSTICS")
    for metric, value in source_stats.items():
        print(f"{metric}: {value}")
    print("\nCHOICE-LEVEL W_GEW DIAGNOSTICS")
    for metric, value in choice_stats.items():
        print(f"{metric}: {value}")
    print("\nIMPORTANT ESTIMATION NOTE")
    print("W_GEW is repeated across the two alternative rows in long format.")
    print("Choice-level diagnostics use one unique row per CHOICE_ID to avoid double-counting.")
    print("\nOUTPUT PATHS")
    print(f"backup path: {BACKUP_PATH}")
    print(f"final canonical long-file path: {LONG_PATH}")
    print(f"QA-summary path: {QA_PATH}")
    print("\nFINAL COUNTS")
    print(f"final row count: {len(reread):,}")
    print(f"final CHOICE_ID count: {reread['CHOICE_ID'].nunique():,}")
    print(f"missing W_GEW rows: {int(numeric(reread['W_GEW']).isna().sum()):,}")
    print(
        "CHOICE_ID values with inconsistent W_GEW: "
        f"{int(reread.groupby('CHOICE_ID', sort=False)['W_GEW'].nunique(dropna=False).ne(1).sum()):,}"
    )
    print(f"elapsed seconds: {perf_counter() - t0:.2f}")


if __name__ == "__main__":
    main()
