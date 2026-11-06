"""Prepare the Stage-1 Core–V1 candidate sample; never estimate a model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

for ancestor in Path(__file__).resolve().parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break

from src.thesis_pipeline.paths import (  # noqa: E402
    ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, OUTPUTS_DIR,
    SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, REPO_ROOT,
)

ROOT = OUTPUTS_DIR / "mnl" / "stage1" / "core_v1"
V1_SCRIPT = Path(__file__).resolve().parents[1] / "02_estimate_v1_only_mnl.py"
LINEAGE_CODE = [
    REPO_ROOT / "scripts" / "03_model_input" / "05_build_single_driver_mnl_design_matrix.py",
    REPO_ROOT / "src" / "thesis_pipeline" / "model_input" / "vehicle_choice.py",
    REPO_ROOT / "src" / "thesis_pipeline" / "model_input" / "tour_features.py",
    REPO_ROOT / "src" / "thesis_pipeline" / "features" / "household_type.py",
]
JOIN_KEYS = ["CHOICE_ID", "SOURCE_ROW_ID", "H_ID", "HP_ID", "P_ID", "W_ID"]
CONTROLS = ["CHOICE", "AV_1", "AV_2", "W_GEW"]
CONTEXT_ORIGINAL = ["TOUR_TRAVEL_TIME_MIN", "TOUR_N_NONHOME_STOPS",
                    "HOUSEHOLD_ACCOMPANIED", "HP_SEX", "alter_gr5",
                    "household_type", "RegioStaR4"]
VEHICLE_ORIGINAL = [f"{x}_{i}" for x in ("POWERTRAIN", "SEGMENT", "VEHICLE_AGE") for i in (1, 2)]
CONTEXT_DERIVED = ["TOUR_TIME_C_H", "MULTISTOP", "FEMALE", "AGE_UNDER30", "AGE_50PLUS",
                   "FAMILY", "URBAN_REGION"]
AGE_COLUMNS = ["AGE_UNDER18", "AGE_UNDER30", "AGE_30_49", "AGE_50PLUS", "AGE_MISSING"]
MAIN = (
    ("D_PT_HEV", "PT1_HEV", "PT2_HEV"),
    ("D_PT_PHEV", "PT1_PHEV", "PT2_PHEV"),
    ("D_PT_BEV", "PT1_BEV", "PT2_BEV"),
    ("D_SEG_COMPACT", "SEG1_COMPACT", "SEG2_COMPACT"),
    ("D_SEG_MEDIUM", "SEG1_MEDIUM", "SEG2_MEDIUM"),
    ("D_SEG_LARGE", "SEG1_LARGE", "SEG2_LARGE"),
    ("D_VEHICLE_AGE", "VEHICLE_AGE_1", "VEHICLE_AGE_2"),
)
INTERACTIONS = (
    ("D_TIME_PHEV", "TOUR_TIME_C_H", "D_PT_PHEV"),
    ("D_TIME_BEV", "TOUR_TIME_C_H", "D_PT_BEV"),
    ("D_MULTISTOP_LARGE", "MULTISTOP", "D_SEG_LARGE"),
    ("D_MULTISTOP_BEV", "MULTISTOP", "D_PT_BEV"),
    ("D_HHACCOMP_LARGE", "HOUSEHOLD_ACCOMPANIED", "D_SEG_LARGE"),
    ("D_HHACCOMP_BEV", "HOUSEHOLD_ACCOMPANIED", "D_PT_BEV"),
    ("D_FEMALE_BEV", "FEMALE", "D_PT_BEV"),
    ("D_FEMALE_LARGE", "FEMALE", "D_SEG_LARGE"),
    ("D_FEMALE_VEHAGE", "FEMALE", "D_VEHICLE_AGE"),
    ("D_FAMILY_BEV", "FAMILY", "D_PT_BEV"),
    ("D_FAMILY_LARGE", "FAMILY", "D_SEG_LARGE"),
    ("D_FAMILY_VEHAGE", "FAMILY", "D_VEHICLE_AGE"),
    ("D_URBAN_BEV", "URBAN_REGION", "D_PT_BEV"),
    ("D_URBAN_LARGE", "URBAN_REGION", "D_SEG_LARGE"),
    ("D_URBAN_VEHAGE", "URBAN_REGION", "D_VEHICLE_AGE"),
    ("D_AGE_UNDER30_BEV", "AGE_UNDER30", "D_PT_BEV"),
    ("D_AGE_UNDER30_VEHAGE", "AGE_UNDER30", "D_VEHICLE_AGE"),
    ("D_AGE_50PLUS_BEV", "AGE_50PLUS", "D_PT_BEV"),
    ("D_AGE_50PLUS_VEHAGE", "AGE_50PLUS", "D_VEHICLE_AGE"),
)
PREDICTORS = [x[0] for x in MAIN] + [x[0] for x in INTERACTIONS]
assert len(MAIN) == 7 and len(INTERACTIONS) == 19 and len(set(PREDICTORS)) == 26


def repo_relative(path: Path) -> str:
    """Store protected paths relative to the checkout for portable snapshots."""
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def protected_path(filename: str) -> Path:
    """Read current relative manifests and historical absolute-path manifests."""
    path = Path(filename)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_v1():
    """Import accepted filtering and validation functions without calling main/estimate."""
    spec = importlib.util.spec_from_file_location("accepted_v1_for_stage1", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [row[0] for row in module.DELTA_X_SPECS] == [row[0] for row in MAIN]
    assert [(row[2], row[3]) for row in module.DELTA_X_SPECS] == [(row[1], row[2]) for row in MAIN]
    return module


def encode_context(frame: pd.DataFrame, mean_minutes: float | None = None) -> tuple[pd.DataFrame, float]:
    out = frame.copy()
    missing_age_columns = set(AGE_COLUMNS) - set(out.columns)
    if missing_age_columns:
        raise AssertionError(f"Encoded Script-05 age columns missing: {sorted(missing_age_columns)}")
    t = pd.to_numeric(out["TOUR_TRAVEL_TIME_MIN"], errors="coerce")
    stops = pd.to_numeric(out["TOUR_N_NONHOME_STOPS"], errors="coerce")
    accompanied = pd.to_numeric(out["HOUSEHOLD_ACCOMPANIED"], errors="coerce")
    sex = pd.to_numeric(out["HP_SEX"], errors="coerce")
    age = pd.to_numeric(out["alter_gr5"], errors="coerce")
    expected_age = {
        "AGE_UNDER18": age.eq(3),
        "AGE_UNDER30": age.isin([4, 5]),
        "AGE_30_49": age.isin([6, 7, 8, 9]),
        "AGE_50PLUS": age.isin([10, 11, 12, 13, 14, 15]),
        "AGE_MISSING": ~age.isin(range(3, 16)),
    }
    for column, expected in expected_age.items():
        actual = pd.to_numeric(out[column], errors="coerce")
        if actual.isna().any() or not actual.eq(expected.fillna(False).astype(int)).all():
            raise AssertionError(f"Script-05 encoded {column} disagrees with original alter_gr5")
    region = pd.to_numeric(out["RegioStaR4"], errors="coerce")
    hh = out["household_type"].astype("string")
    valid = {
        "invalid_tour_time": t.isna() | ~np.isfinite(t) | t.lt(0),
        "invalid_stops": stops.isna() | ~np.isfinite(stops) | stops.lt(0) | stops.ne(np.floor(stops)),
        "invalid_accompaniment": ~accompanied.isin([0, 1]),
        "invalid_sex_including_diverse": ~sex.isin([1, 2]),
        "invalid_age": ~age.isin(range(3, 16)),
        "under18_driver": age.eq(3).fillna(False),
        "invalid_household_type": ~hh.isin(["family_household", "young_household", "adult_household", "senior_household"]),
        "invalid_regiostar4": ~region.isin([11, 12, 21, 22]),
    }
    for name, flag in valid.items():
        out[name] = flag.fillna(True).astype(bool)
    if mean_minutes is None:
        accepted = ~out[list(valid)].any(axis=1)
        mean_minutes = float(t.loc[accepted].mean())
    out["TOUR_TIME_C_H"] = (t - mean_minutes) / 60
    out["MULTISTOP"] = pd.Series(np.where(stops.notna(), stops.ge(2).astype(float), np.nan), index=out.index)
    out["FEMALE"] = sex.map({1: 0, 2: 1})
    out["FAMILY"] = hh.map({"family_household": 1, "young_household": 0,
                             "adult_household": 0, "senior_household": 0})
    out["URBAN_REGION"] = region.map({11: 1, 12: 1, 21: 0, 22: 0})
    return out, mean_minutes


def effective_design(sample: pd.DataFrame, *, swap: bool = False) -> pd.DataFrame:
    design = pd.DataFrame(index=sample.index)
    for name, left, right in MAIN:
        a, b = (right, left) if swap else (left, right)
        design[name] = pd.to_numeric(sample[a], errors="raise").to_numpy(float) - pd.to_numeric(sample[b], errors="raise").to_numpy(float)
    for name, context, target in INTERACTIONS:
        design[name] = sample[context].to_numpy(float) * design[target].to_numpy(float)
    assert list(design.columns) == PREDICTORS
    assert np.isfinite(design.to_numpy()).all()
    return design


def prepare() -> dict:
    inputs = [SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH, ALLOCATION_CANDIDATE_MODEL_INPUT_PATH, V1_SCRIPT, *LINEAGE_CODE]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required verified input missing: {path}")
    before = {repo_relative(p): sha256(p) for p in inputs}
    accepted_results = sorted((OUTPUTS_DIR / "mnl" / "v1_only").glob("*")) + sorted((OUTPUTS_DIR / "mnl" / "v2_only").glob("*"))
    accepted_results = [p for p in accepted_results if p.is_file()]
    result_hashes = {repo_relative(p): sha256(p) for p in accepted_results}
    design_source = pd.read_csv(inputs[0], dtype={k: "string" for k in JOIN_KEYS}, low_memory=False)
    upstream = pd.read_csv(inputs[1], dtype={k: "string" for k in JOIN_KEYS}, low_memory=False)
    needed = JOIN_KEYS + CONTROLS + CONTEXT_ORIGINAL + VEHICLE_ORIGINAL
    missing = set(needed) - set(upstream.columns)
    if missing:
        raise AssertionError(f"Verified context source missing columns: {sorted(missing)}")
    if upstream.CHOICE_ID.isna().any() or not upstream.CHOICE_ID.is_unique:
        raise AssertionError("Context source CHOICE_ID missing or duplicated")
    v1 = load_v1()
    v1.validate_source(design_source)
    stage0, flow0 = v1.build_v1_sample(design_source)
    joined = stage0.merge(upstream[needed], on="CHOICE_ID", how="left", validate="one_to_one", indicator=True, suffixes=("", "_UP"))
    if len(joined) != len(stage0) or not joined.CHOICE_ID.is_unique or joined._merge.ne("both").any():
        raise AssertionError("Stage-0-to-context CHOICE_ID mapping is not complete and one to one")
    for column in JOIN_KEYS[1:] + CONTROLS:
        left, right = joined[column], joined[f"{column}_UP"]
        if column == "W_GEW":
            equal = np.isclose(pd.to_numeric(left), pd.to_numeric(right), rtol=0, atol=0)
        elif column in CONTROLS:
            equal = pd.to_numeric(left).eq(pd.to_numeric(right))
        else:
            equal = left.astype("string").eq(right.astype("string"))
        if not bool(np.all(equal)):
            raise AssertionError(f"Untrusted join: accepted {column} differs from upstream")
    joined = joined.drop(columns=["_merge"] + [f"{c}_UP" for c in JOIN_KEYS[1:] + CONTROLS])
    for column in ("TOUR_TRAVEL_TIME_MIN", "HOUSEHOLD_ACCOMPANIED"):
        left = pd.to_numeric(joined[column], errors="coerce")
        right = pd.to_numeric(joined[f"{column}_UP"], errors="coerce")
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.all():
            raise AssertionError(f"Accepted encoded source and upstream context differ: {column}")
    joined = joined.drop(columns=["TOUR_TRAVEL_TIME_MIN_UP", "HOUSEHOLD_ACCOMPANIED_UP"])
    # Compare original numeric vehicle codes with the accepted encoded blocks.
    for car in (1, 2):
        for source, prefix, code_map in [
            ("POWERTRAIN", "PT", {1: "ICE", 2: "ICE", 6: "ICE", 3: "HEV", 4: "PHEV", 5: "BEV"}),
            ("SEGMENT", "SEG", {1: "SMALL", 2: "COMPACT", 3: "MEDIUM", 4: "LARGE"}),
        ]:
            actual = pd.to_numeric(joined[f"{source}_{car}"], errors="coerce").map(code_map).astype("string")
            decoded = v1.dummy_category(joined, f"{prefix}{car}", v1.POWERTRAINS if prefix == "PT" else v1.SEGMENTS).astype("string")
            if not actual.eq(decoded).all():
                raise AssertionError(f"Vehicle coding mismatch after join: {source}_{car}")
        if not np.allclose(pd.to_numeric(joined[f"VEHICLE_AGE_{car}"]), joined[f"VEHICLE_AGE_{car}_UP"], rtol=0, atol=0):
            raise AssertionError(f"Vehicle age mismatch after join: car {car}")
    joined = joined.drop(columns=["VEHICLE_AGE_1_UP", "VEHICLE_AGE_2_UP"])
    context, mean_minutes = encode_context(joined)
    reasons = [c for c in context.columns if c.startswith("invalid_")] + ["under18_driver"]
    excluded = context[reasons].any(axis=1)
    missing_context = context[CONTEXT_ORIGINAL].isna().any(axis=1)
    sample = context.loc[~excluded].copy().sort_values("CHOICE_ID", kind="stable").reset_index(drop=True)
    if sample.empty:
        raise AssertionError("No complete Stage-1 context occasions")
    if sample[CONTEXT_DERIVED].isna().any().any():
        raise AssertionError("Complete-case context has missing derived values")
    if not np.allclose(sample.TOUR_TIME_C_H.mean(), 0, atol=1e-12):
        raise AssertionError(f"Travel time not centered on final candidate sample: sample_mean={sample.TOUR_TIME_C_H.mean()}, raw_mean={sample.TOUR_TRAVEL_TIME_MIN.mean()}, centering_mean={mean_minutes}, N={len(sample)}")
    matrix = effective_design(sample)
    if not np.allclose(matrix.to_numpy(), -effective_design(sample, swap=True).to_numpy()):
        raise AssertionError("Car-label swap did not negate all predictors")
    for name, left, right in MAIN:
        if not np.array_equal(matrix[name], sample[left].to_numpy(float) - sample[right].to_numpy(float)):
            raise AssertionError(f"Stage-0 delta mismatch: {name}")
    for name, context_name, target in INTERACTIONS:
        if not np.array_equal(matrix[name], sample[context_name].to_numpy(float) * matrix[target].to_numpy(float)):
            raise AssertionError(f"Interaction mismatch: {name}")
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(data_dir / "core_v1_context_sample.csv", index=False)
    effective = pd.concat([sample[["CHOICE_ID", "H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW"]], matrix], axis=1)
    effective.to_csv(data_dir / "core_v1_effective_design.csv", index=False)
    flow = flow0.copy()
    additional = [
        ("stage0_valid_v1_occasions", len(stage0), "Accepted build_v1_sample and validate_retained_sample"),
        ("stage0_diverse_sex_code_3", int(pd.to_numeric(context.HP_SEX, errors="coerce").eq(3).sum()), "Reported separately; excluded from male/female interaction"),
        ("stage0_missing_or_other_invalid_sex", int((context.invalid_sex_including_diverse & ~pd.to_numeric(context.HP_SEX, errors="coerce").eq(3)).sum()), "Reported separately; excluded from male/female interaction"),
        ("stage1_missing_required_context", int(missing_context.sum()), "Any missing original required context field; non-exclusive with coding reasons"),
        ("stage1_invalid_nonmissing_context", int((excluded & ~missing_context).sum()), "Invalid category among occasions with no missing original context"),
        *[(f"stage0_missing_{field}", int(context[field].isna().sum()), "Missing original context field before Stage-1 exclusions") for field in CONTEXT_ORIGINAL],
        *[(name, int(context[name].sum()), "Non-exclusive Stage-1 context exclusion") for name in reasons],
        ("stage1_exclusion_reason_overlap", int(context[reasons].sum(axis=1).gt(1).sum()), "At least two context reasons"),
        ("stage1_unique_exclusions", int(excluded.sum()), "Union of context reasons"),
        ("stage1_candidate_occasions", len(sample), "Complete valid context"),
        ("stage1_unique_households", sample.H_ID.nunique(), "Distinct H_ID"),
        ("stage1_both_available", int(sample.AV_1.eq(1).mul(sample.AV_2.eq(1)).sum()), "Primary informative subset"),
        ("stage1_single_available", int(sample.AV_1.ne(sample.AV_2).sum()), "Deterministic occasions"),
        ("stage0_under18_drivers_excluded_at_stage1", int(context.under18_driver.sum()), "Encoded AGE_UNDER18=1; excluded only from Stage-1"),
    ]
    flow = pd.concat([flow, pd.DataFrame([{"metric": m, "count": n, "share_of_initial": n / len(design_source), "notes": note} for m, n, note in additional])], ignore_index=True)
    diag_dir = ROOT / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    flow.to_csv(diag_dir / "core_v1_sample_flow.csv", index=False)
    pre_rows = []
    for field in CONTEXT_ORIGINAL:
        if field == "TOUR_TRAVEL_TIME_MIN":
            continue
        for value, group in context.groupby(field, dropna=False):
            pre_rows.append({"variable": field, "source": "allocation_candidate_model_input",
                             "category": str(value), "stage0_valid_count": len(group),
                             "stage0_valid_share": len(group) / len(context),
                             "stage1_retained_count": int((~excluded).loc[group.index].sum()),
                             "stage0_unique_households": group.H_ID.nunique()})
    pd.DataFrame(pre_rows).to_csv(diag_dir / "core_v1_pre_exclusion_context_codes.csv", index=False)
    metadata = {
        "decision_status": "PENDING_MANUAL_REVIEW", "centering_mean_minutes": mean_minutes,
        "source_lineage": {"design_matrix": repo_relative(inputs[0]), "context_model_input": repo_relative(inputs[1]), "stage0_v1_code": repo_relative(inputs[2])},
        "join": {"key": "CHOICE_ID", "validated_controls": JOIN_KEYS[1:] + CONTROLS + VEHICLE_ORIGINAL},
        "source_hashes_sha256": before,
        "accepted_stage0_result_hashes_sha256": result_hashes,
        "lineage_code_paths": [repo_relative(p) for p in LINEAGE_CODE],
        "context_rules": {"MULTISTOP": "0 for 0–1 non-home stops, 1 for 2+", "FEMALE": "HP_SEX 2 vs 1; 3/missing excluded", "AGE_UNDER30": "Script-05 encoded alter_gr5 codes 4–5; codes 6–9 reference; code 3 excluded", "AGE_50PLUS": "Script-05 encoded alter_gr5 codes 10–15; codes 6–9 reference", "FAMILY": "family_household vs three other valid types", "URBAN_REGION": "RegioStaR4 11/12 vs 21/22", "HOUSEHOLD_ACCOMPANIED": "Verified focal-person driving trips within reconstructed tour; 0/1, unresolved missing", "TOUR_TIME_C_H": "(TOUR_TRAVEL_TIME_MIN - centering_mean_minutes)/60"},
        "main_effects": [{"predictor": n, "formula": f"{a} - {b}"} for n, a, b in MAIN],
        "interactions": [{"predictor": n, "formula": f"{c} * {d}"} for n, c, d in INTERACTIONS],
        "sample": {"source_n": len(design_source), "stage0_valid_n": len(stage0), "stage1_n": len(sample), "households": int(sample.H_ID.nunique())},
        "vehicle_id_note": "Accepted design and allocation model input contain H_ID and vehicle labels 1/2, but no original A_ID columns; no vehicle ID was inferred.",
    }
    (data_dir / "core_v1_transformations.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    after = {repo_relative(p): sha256(p) for p in inputs}
    if before != after:
        raise AssertionError("An accepted input changed during preparation")
    if result_hashes != {repo_relative(p): sha256(p) for p in accepted_results}:
        raise AssertionError("Accepted Stage-0 results changed during preparation")
    return metadata


if __name__ == "__main__":
    result = prepare()
    print(f"Stage-1 prepared: Stage-0 N={result['sample']['stage0_valid_n']}; candidate N={result['sample']['stage1_n']}; H={result['sample']['households']}; 7 main + 19 interactions; output={ROOT.resolve()}")
