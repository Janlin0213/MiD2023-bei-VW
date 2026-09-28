"""Read-only baseline comparisons and QA shared by downstream sensitivity scripts."""
from pathlib import Path
import hashlib
import importlib.util
import io
import sys
import pandas as pd
import numpy as np

# Explicit continuous-variable allowlist: discrete features always compare exactly.
CONTINUOUS_COLUMNS = {"W_GEW", "TOUR_DISTANCE_KM", "TOUR_TRAVEL_TIME_MIN",
                      "TOUR_ELAPSED_TIME_MIN", "LEG_DISTANCE_KM", "LEG_TRAVEL_TIME_MIN"}
RTOL = 1e-12
ATOL = 1e-15

ROOT = Path(__file__).resolve().parents[1]

def load_script(filename):
    name = "accepted_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def read(path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)

def serialized(frame):
    return read(io.StringIO(frame.to_csv(index=False)))

def reject(frame, bad, message):
    if bad.fillna(True).any():
        print(frame.loc[bad.fillna(True)].head(10).to_string(index=False))
        raise AssertionError(message)

def ids(frame):
    for col in ["CHOICE_ID", "SOURCE_ROW_ID"]:
        reject(frame, frame[col].str.strip().eq("") | frame[col].duplicated(False), f"Invalid {col}")

def overlap_equal(baseline, sensitivity, diagnostic_path=None):
    ids(baseline); ids(sensitivity)
    if list(baseline.columns) != list(sensitivity.columns):
        raise AssertionError("Baseline and sensitivity ordered schemas differ.")
    reject(baseline, ~baseline.CHOICE_ID.isin(sensitivity.CHOICE_ID), "Baseline choices missing.")
    old = baseline.set_index("CHOICE_ID")
    new = sensitivity.set_index("CHOICE_ID").loc[old.index]
    differences = old.ne(new)
    for column in CONTINUOUS_COLUMNS.intersection(old.columns):
        # Identically blank explanatory values are missing in both datasets.
        # Weights must always be populated, finite and positive.
        x = pd.to_numeric(old[column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(new[column], errors="coerce").to_numpy(dtype=float)
        same = np.isfinite(x) & np.isfinite(y) & np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=False)
        if column != "W_GEW":
            same |= old[column].eq("").to_numpy() & new[column].eq("").to_numpy()
        else:
            same &= (x > 0) & (y > 0)
        differences[column] = ~same
    weight_bad = differences["W_GEW"]
    if weight_bad.any():
        path = diagnostic_path or (ROOT / "data_processed/reconstruction/phase4_sensitivity/decision_timing_weight_precision_diagnostic.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"SOURCE_ROW_ID": old.loc[weight_bad, "SOURCE_ROW_ID"],
                      "baseline_W_GEW": old.loc[weight_bad, "W_GEW"],
                      "sensitivity_W_GEW": new.loc[weight_bad, "W_GEW"],
                      "rtol": RTOL, "atol": ATOL}).to_csv(path, index_label="CHOICE_ID")
        print(f"Weight tolerance failures: {int(weight_bad.sum())}; diagnostic: {path}")
    else:
        print("W_GEW baseline-overlap numerical equality: PASS\ntolerance failures: 0")
    if differences.any().any():
        samples = []
        for column in old.columns:
            for key in old.index[differences[column]][:10]:
                samples.append(dict(CHOICE_ID=key, column=column, old_value=old.at[key,column], new_value=new.at[key,column]))
            if len(samples) >= 10: break
        print(pd.DataFrame(samples[:10]).to_string(index=False))
        raise AssertionError("Baseline overlap differs: exact discrete comparison; continuous rtol=1e-12, atol=1e-15.")
    return sensitivity.loc[~sensitivity.CHOICE_ID.isin(baseline.CHOICE_ID)].copy()

def fingerprints():
    paths = list((ROOT / "data_processed/reconstruction").glob("phase[1-4]/*"))
    paths += list((ROOT / "data_processed/reconstruction/phase3_sensitivity").glob("*"))
    paths += list((ROOT / "data_processed/model_input").glob("*.csv"))
    paths += [ROOT / "scripts" / name for name in ["04_phase3_strict_baseline.py", "06_build_home_based_tour_features.py", "07_build_mnl_vehicle_choice_model_input.py"]]
    result = {}
    for path in paths:
        if path.is_file():
            with path.open("rb") as source:
                result[path] = hashlib.file_digest(source, "sha256").hexdigest()
    return result

def check_fingerprints(before):
    if fingerprints() != before:
        raise AssertionError("An accepted baseline input/output changed during execution.")

def descriptive(rows, sample, frame):
    for column in ["TOUR_DISTANCE_KM", "TOUR_TRAVEL_TIME_MIN", "TOUR_N_NONHOME_STOPS"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        for stat in ["mean", "median"]:
            rows.append(dict(section=sample, metric=f"{column} {stat}", value=getattr(values, stat)()))
    for category, count in frame["TOUR_FIRST_PURPOSE"].value_counts(dropna=False).sort_index().items():
        rows.append(dict(section=sample, metric="TOUR_FIRST_PURPOSE distribution", category=category, value=count))
