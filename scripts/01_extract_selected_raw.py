from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "data_processed"
SELECTED_LIST_PATH = ROOT / "metadata" / "selected raw list.py"

ID_COLUMNS = {"H_ID", "HP_ID", "P_ID", "W_ID", "A_ID"}


@dataclass(frozen=True)
class ExtractSpec:
    label: str
    source_name: str
    selected_cols_attr: str
    output_name: str

    @property
    def source_path(self) -> Path:
        return RAW_DIR / self.source_name

    @property
    def output_path(self) -> Path:
        return OUT_DIR / self.output_name


SPECS = [
    ExtractSpec("Haushalte", "MiD2023_Haushalte.csv", "HH_SELECTED_COLS", "hh_selected_raw.csv"),
    ExtractSpec("Personen", "MiD2023_Personen.csv", "PERSON_SELECTED_COLS", "persons_selected_raw.csv"),
    ExtractSpec("Wege", "MiD2023_Wege.csv", "WEGE_SELECTED_COLS", "trips_selected_raw.csv"),
    ExtractSpec("Autos", "MiD2023_Autos.csv", "AUTOS_SELECTED_COLS", "cars_selected_raw.csv"),
]


def load_selected_list(path: Path = SELECTED_LIST_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location("selected_raw_list", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load selected raw list: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_selected_raw(path: Path, selected_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing_cols = [col for col in selected_cols if col not in header]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"{path.name} is missing selected column(s): {missing}")

    df = pd.read_csv(
        path,
        usecols=selected_cols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    df = df[selected_cols]

    for col in df.columns:
        df[col] = df[col].str.strip()

    for col in ID_COLUMNS.intersection(df.columns):
        df[col] = df[col].astype("string")

    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.eq("").sum()
    summary = pd.DataFrame(
        {
            "column": missing.index,
            "missing": missing.to_numpy(),
            "missing_pct": (missing.to_numpy() / len(df) * 100) if len(df) else 0,
        }
    )
    return summary.loc[summary["missing"] > 0].sort_values(["missing", "column"], ascending=[False, True])


def print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"\n[{label}]")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print("columns:")
    print(", ".join(df.columns))

    summary = missing_summary(df)
    print("missing summary:")
    if summary.empty:
        print("  no blank values in selected columns")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(summary.to_string(index=False, formatters={"missing_pct": "{:.2f}".format}))


def extract_one(spec: ExtractSpec, selected_cols: list[str]) -> None:
    df = read_selected_raw(spec.source_path, selected_cols)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec.output_path, index=False)
    print_summary(spec.label, df)
    print(f"saved: {spec.output_path}")


def main() -> None:
    selected = load_selected_list()
    for spec in SPECS:
        selected_cols = list(getattr(selected, spec.selected_cols_attr))
        extract_one(spec, selected_cols)


if __name__ == "__main__":
    main()
