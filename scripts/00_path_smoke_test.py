"""Read-only smoke test for centralized repository paths and module imports."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import (  # noqa: E402
    MODEL_INPUT_DIR,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    PHASE3_SENSITIVITY_DIR,
    PHASE4_DIR,
    REPO_ROOT,
    SELECTED_RAW_DIR,
    is_within_repo,
)


ACCEPTED_FILES = [
    SELECTED_RAW_DIR / "hh_selected_raw.csv",
    SELECTED_RAW_DIR / "persons_selected_raw.csv",
    SELECTED_RAW_DIR / "trips_selected_raw.csv",
    SELECTED_RAW_DIR / "cars_selected_raw.csv",
    PHASE1_DIR / "trips_home_chain_enriched.csv",
    PHASE2_DIR / "vehicle_events_v2.csv",
    PHASE2_DIR / "vehicle_choice_occasions_all_v2.csv",
    PHASE3_DIR / "mnl_vehicle_choice_wide_base.csv",
    PHASE3_DIR / "vehicle_choice_occasions_classified.csv",
    PHASE3_SENSITIVITY_DIR
    / "mnl_vehicle_choice_wide_decision_timing_sensitivity_base.csv",
    PHASE4_DIR / "mnl_vehicle_choice_wide_tour_features.csv",
    MODEL_INPUT_DIR / "mnl_vehicle_choice_model_input.csv",
    MODEL_INPUT_DIR / "mnl_vehicle_choice_model_input_manifest.csv",
]


def module_files() -> list[Path]:
    files = []
    for directory in (
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
        REPO_ROOT / "config",
        REPO_ROOT / "metadata",
    ):
        files.extend(directory.rglob("*.py"))
    return sorted(
        path
        for path in files
        if "__pycache__" not in path.parts and path.resolve() != Path(__file__).resolve()
    )


def module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT)
    if path.name == "__init__.py":
        relative = relative.parent
    else:
        relative = relative.with_suffix("")
    if relative.parts[0] == "src":
        return ".".join(relative.parts)
    safe_stem = "".join(character if character.isalnum() else "_" for character in path.stem)
    return f"_path_smoke_{safe_stem}_{abs(hash(relative.as_posix()))}"


def import_file(path: Path):
    name = module_name(path)
    if path.is_relative_to(REPO_ROOT / "src"):
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.dont_write_bytecode = True
    if not (REPO_ROOT / ".git").exists():
        raise AssertionError(f"Resolved root does not contain .git: {REPO_ROOT}")

    missing = [path for path in ACCEPTED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Accepted files do not resolve: {missing}")

    imported = []
    declared_paths = []
    for path in module_files():
        module = import_file(path)
        imported.append(path)
        for name, value in vars(module).items():
            if isinstance(value, Path):
                declared_paths.append((path, name, value.expanduser().resolve()))

    outside = [item for item in declared_paths if not is_within_repo(item[2])]
    if outside:
        formatted = [f"{source}:{name} -> {value}" for source, name, value in outside]
        raise AssertionError("Declared paths outside repository: " + "; ".join(formatted))

    print(f"repository root: {REPO_ROOT}")
    print(f"imported Python modules: {len(imported)}")
    print(f"resolved declared Path constants: {len(declared_paths)}")
    print(f"accepted existing files checked: {len(ACCEPTED_FILES)}")
    print("outside-repository declared paths: 0")
    print("PATH SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
