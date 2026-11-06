"""Build new Core-V1 protection manifests after each validated stage.

Run prepared -> diagnose -> diagnosed -> diagnose -> estimation. This command
never replaces an existing baseline and never estimates a model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("core_v1_prepare_for_baseline", HERE / "01_prepare_core_v1.py")
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)

ROOT = prep.ROOT
DATA = ROOT / "data"
DIAG = ROOT / "diagnostics"
REVIEW = ROOT / "review"
QA_BASELINE = REVIEW / "core_v1_targeted_QA_baseline.json"
EST_BASELINE = ROOT / "estimation" / "review" / "core_v1_stage5_protected_baseline.json"
NUMERICAL_DIAGNOSTICS = (
    "core_v1_rank_diagnostics.csv", "core_v1_condition_diagnostics.csv",
    "core_v1_vif.csv", "core_v1_correlation.csv",
)


def hashes(paths: list[Path]) -> dict[str, str]:
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    return {prep.repo_relative(path): prep.sha256(path) for path in paths}


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace protected baseline: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validated_metadata() -> dict:
    metadata = json.loads((DATA / "core_v1_transformations.json").read_text(encoding="utf-8"))
    for section in ("source_hashes_sha256", "accepted_stage0_result_hashes_sha256"):
        for filename, expected in metadata[section].items():
            if prep.sha256(prep.protected_path(filename)) != expected:
                raise AssertionError(f"Prepared source changed: {filename}")
    import pandas as pd
    import numpy as np
    sample = pd.read_csv(DATA / "core_v1_context_sample.csv", dtype={k: "string" for k in prep.JOIN_KEYS}, low_memory=False)
    effective = pd.read_csv(DATA / "core_v1_effective_design.csv", dtype={"CHOICE_ID": "string", "H_ID": "string"})
    if len(sample) != metadata["sample"]["stage1_n"] or not sample.CHOICE_ID.is_unique:
        raise AssertionError("Prepared sample identifiers/count changed")
    if not sample.AV_1.eq(1).all() or not sample.AV_2.eq(1).all() or not sample.CHOICE.isin([1, 2]).all():
        raise AssertionError("Prepared choice/availability invalid")
    weights = sample.W_GEW.to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise AssertionError("Prepared observation weights invalid")
    if list(effective) != ["CHOICE_ID", "H_ID", "CHOICE", "AV_1", "AV_2", "W_GEW", *prep.PREDICTORS]:
        raise AssertionError("Prepared predictor schema changed")
    if not effective.CHOICE_ID.equals(sample.CHOICE_ID):
        raise AssertionError("Prepared row order changed")
    rebuilt = prep.effective_design(sample)
    if not np.allclose(rebuilt.to_numpy(float), effective[prep.PREDICTORS].to_numpy(float), rtol=0, atol=1e-12):
        raise AssertionError("Prepared design does not reconstruct")
    if np.linalg.matrix_rank(rebuilt.to_numpy(float)) != len(prep.PREDICTORS):
        raise AssertionError("Prepared design is not full rank")
    return metadata


def prepared() -> None:
    metadata = validated_metadata()
    if QA_BASELINE.exists():
        raise FileExistsError(QA_BASELINE)
    stage1 = hashes([DATA / "core_v1_context_sample.csv", DATA / "core_v1_effective_design.csv",
                     DATA / "core_v1_transformations.json"])
    upstream = dict(metadata["source_hashes_sha256"])
    upstream.update(metadata["accepted_stage0_result_hashes_sha256"])
    write_new(QA_BASELINE, {"schema_version": 2, "status": "PREPARED_PENDING_DIAGNOSTICS",
                            "protected_stage1": stage1, "baseline_diagnostics": {},
                            "upstream_and_stage0": upstream})


def diagnosed() -> None:
    validated_metadata()
    baseline = json.loads(QA_BASELINE.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 2 or baseline["baseline_diagnostics"]:
        raise AssertionError("Expected an unsealed new Core-V1 QA baseline")
    for section in ("protected_stage1", "upstream_and_stage0"):
        for filename, expected in baseline[section].items():
            if prep.sha256(prep.protected_path(filename)) != expected:
                raise AssertionError(f"Protected file changed: {filename}")
    import pandas as pd
    rank = pd.read_csv(DIAG / "core_v1_rank_diagnostics.csv")
    selected = rank.loc[rank.matrix.eq("raw_no_intercept")]
    if len(selected) != 1 or int(selected.iloc[0]["rank"]) != len(prep.PREDICTORS):
        raise AssertionError("Pre-estimation rank did not pass")
    baseline["baseline_diagnostics"] = hashes([DIAG / name for name in NUMERICAL_DIAGNOSTICS])
    baseline["status"] = "DIAGNOSTICS_PASSED"
    QA_BASELINE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def estimation() -> None:
    validated_metadata()
    qa = json.loads(QA_BASELINE.read_text(encoding="utf-8"))
    if qa.get("status") != "DIAGNOSTICS_PASSED":
        raise AssertionError("Sealed diagnostic baseline required")
    for section in ("protected_stage1", "baseline_diagnostics", "upstream_and_stage0"):
        for filename, expected in qa[section].items():
            if prep.sha256(prep.protected_path(filename)) != expected:
                raise AssertionError(f"Protected file changed: {filename}")
    manifest = json.loads((REVIEW / "core_v1_input_manifest.json").read_text(encoding="utf-8"))
    if manifest["decision_status"] != "PENDING_MANUAL_REVIEW" or manifest["stage1_candidate_n"] < 1:
        raise AssertionError("Diagnostic input manifest is not ready")
    if EST_BASELINE.exists():
        raise FileExistsError(EST_BASELINE)
    stage1_files = sorted(path for path in ROOT.rglob("*") if path.is_file() and "estimation" not in path.parts)
    stage1_files += [HERE / name for name in ("00_build_core_v1_baselines.py", "01_prepare_core_v1.py",
                                               "02_diagnose_core_v1.py", "test_core_v1.py")]
    write_new(EST_BASELINE, {"schema_version": 2, "status": "DIAGNOSTICS_PASSED_PENDING_ESTIMATION",
                             "stage1_files_and_code": hashes(stage1_files),
                             "upstream_and_stage0": qa["upstream_and_stage0"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepared", "diagnosed", "estimation"))
    args = parser.parse_args()
    {"prepared": prepared, "diagnosed": diagnosed, "estimation": estimation}[args.phase]()
    print(f"Core-V1 baseline phase {args.phase} completed at {ROOT}")
