"""Central, location-independent repository paths.

Scripts import paths from this module instead of deriving the repository root
or physical layout from their own nesting depth.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT_ENV = "THESIS_REPO_ROOT"


def _walk_for_marker(start: Path) -> Path | None:
    """Return the nearest ancestor containing the repository marker."""

    resolved = start.expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root from an override or an upward marker walk."""

    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(
                f"{REPO_ROOT_ENV} does not identify an existing directory: {root}"
            )
        return root

    search_starts = []
    if start is not None:
        search_starts.append(start)
    search_starts.extend((Path.cwd(), Path(__file__).resolve()))
    for search_start in search_starts:
        root = _walk_for_marker(search_start)
        if root is not None:
            return root
    raise RuntimeError(
        f"Could not locate the repository root. Set {REPO_ROOT_ENV} or run "
        "inside a checkout containing .git."
    )


REPO_ROOT = find_repo_root()
ROOT = REPO_ROOT

DATA_RAW_DIR = REPO_ROOT / "data_raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data_processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
METADATA_DIR = REPO_ROOT / "metadata"
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_PREPARATION_SCRIPTS_DIR = SCRIPTS_DIR / "00_data_preparation"
DESCRIPTIVE_SCRIPTS_DIR = SCRIPTS_DIR / "01_descriptive"
CHOICE_RECONSTRUCTION_SCRIPTS_DIR = SCRIPTS_DIR / "02_choice_reconstruction"
MODEL_INPUT_SCRIPTS_DIR = SCRIPTS_DIR / "03_model_input"
SCREENING_SCRIPTS_DIR = SCRIPTS_DIR / "04_screening"
MNL_SCRIPTS_DIR = SCRIPTS_DIR / "05_mnl"
SRC_DIR = REPO_ROOT / "src"

RAW_HOUSEHOLDS_PATH = DATA_RAW_DIR / "MiD2023_Haushalte.csv"
RAW_PERSONS_PATH = DATA_RAW_DIR / "MiD2023_Personen.csv"
RAW_TRIPS_PATH = DATA_RAW_DIR / "MiD2023_Wege.csv"
RAW_CARS_PATH = DATA_RAW_DIR / "MiD2023_Autos.csv"
RAW_MID_FILES = {
    "Haushalte": RAW_HOUSEHOLDS_PATH,
    "Personen": RAW_PERSONS_PATH,
    "Wege": RAW_TRIPS_PATH,
    "Autos": RAW_CARS_PATH,
}

SELECTED_RAW_DIR = DATA_PROCESSED_DIR / "selected_raw"
RECONSTRUCTION_DIR = DATA_PROCESSED_DIR / "reconstruction"
PHASE1_DIR = RECONSTRUCTION_DIR / "phase1"
PHASE2_DIR = RECONSTRUCTION_DIR / "phase2"
PHASE3_DIR = RECONSTRUCTION_DIR / "phase3"
PHASE3_SENSITIVITY_DIR = RECONSTRUCTION_DIR / "phase3_sensitivity"
ALLOCATION_CANDIDATE_DIR = RECONSTRUCTION_DIR / "allocation_candidate"
PHASE4_DIR = RECONSTRUCTION_DIR / "phase4"
PHASE4_SENSITIVITY_DIR = RECONSTRUCTION_DIR / "phase4_sensitivity"

ALLOCATION_CANDIDATE_OCCASIONS_PATH = (
    ALLOCATION_CANDIDATE_DIR / "allocation_analysis_candidate_occasions.csv"
)
ALLOCATION_CANDIDATE_HOUSEHOLDS_PATH = (
    ALLOCATION_CANDIDATE_DIR / "allocation_analysis_candidate_households.csv"
)
ALLOCATION_CANDIDATE_QA_PATH = (
    ALLOCATION_CANDIDATE_DIR / "allocation_analysis_candidate_QA.csv"
)

MODEL_INPUT_DIR = DATA_PROCESSED_DIR / "model_input"
MODEL_INPUT_SENSITIVITY_DIR = MODEL_INPUT_DIR / "sensitivity"
ALLOCATION_CANDIDATE_MODEL_INPUT_PATH = (
    MODEL_INPUT_DIR / "allocation_candidate_model_input.csv"
)
ALLOCATION_CANDIDATE_MODEL_INPUT_QA_PATH = (
    MODEL_INPUT_DIR / "allocation_candidate_model_input_QA.csv"
)
SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH = (
    MODEL_INPUT_DIR / "single_driver_mnl_design_matrix.csv"
)
SINGLE_DRIVER_MNL_DESIGN_MATRIX_QA_PATH = (
    MODEL_INPUT_DIR / "single_driver_mnl_design_matrix_QA.csv"
)
SINGLE_DRIVER_MNL_ENCODING_SPEC_PATH = (
    MODEL_INPUT_DIR / "single_driver_mnl_encoding_spec.md"
)
SCREENING_DIR = MODEL_INPUT_DIR

EDA_PROCESSED_DIR = DATA_PROCESSED_DIR / "eda"
THEME1_BACKBONE_DIR = EDA_PROCESSED_DIR / "statistical_test"
BIPARTITE_DATA_DIR = EDA_PROCESSED_DIR / "bipartite"
ALLOCATION_SAMPLE_SUPPORT_DATA_DIR = EDA_PROCESSED_DIR / "allocation_sample_support"

DESCRIPTIVE_OUTPUT_DIR = OUTPUTS_DIR / "descriptive"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
BIPARTITE_OUTPUT_DIR = EDA_OUTPUT_DIR / "bipartite"
ALLOCATION_SAMPLE_SUPPORT_OUTPUT_DIR = EDA_OUTPUT_DIR / "allocation_sample_support"
THEME1_OUTPUT_DIR = (
    EDA_OUTPUT_DIR / "statistical_tests" / "theme1_single_vs_multicar"
)

CODEBOOK_PATH = (
    REFERENCE_DATA_DIR / "codebooks" / "MiD2023_Codepläne_B1_Standard_v1.1.xlsx"
)
SELECTED_RAW_COLUMNS_PATH = CONFIG_DIR / "selected_raw_columns.py"
MNL_ATTRIBUTE_CANDIDATES_PATH = CONFIG_DIR / "mnl_attribute_candidates.py"


def is_within_repo(path: Path) -> bool:
    """Return whether a resolved path is contained by the repository root."""

    try:
        path.expanduser().resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True
