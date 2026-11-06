# MiD 2023 vehicle-choice thesis pipeline

This repository contains the research code for a Master's thesis using the
German mobility study *Mobilität in Deutschland 2023* (MiD 2023) to study
vehicle choice and intra-household vehicle allocation in multi-car households.
It supports data preparation, descriptive analysis, choice-occasion
reconstruction, model-input construction, and diagnostic screening.

This is a thesis research-code repository, not an official Volkswagen software
product. Licensed MiD microdata and processed record-level data are not
distributed through GitHub.

## Quick start

```bash
git clone https://github.com/Janlin0213/MiD2023-bei-VW.git
cd MiD2023-bei-VW
uv sync
```

MiD data are not included; authorized collaborators must provide the private
data layer described below before running the full analytical workflow.

## Repository layout

The public, version-controlled project is organized as follows:

```text
.
├── config/                 Analysis configuration and selected-variable lists
├── notebooks/              Shareable preparation, descriptive, and legacy checks
├── scripts/                Workflow-facing executable entry points
│   ├── 00_data_preparation/
│   ├── 01_descriptive/
│   ├── 02_choice_reconstruction/
│   ├── 03_model_input/
│   ├── 04_screening/
│   └── 05_mnl/
├── run-pipeline.ps1        Windows pipeline launcher
├── src/                    Reusable Python implementation
│   └── thesis_pipeline/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

| Path | Purpose |
|---|---|
| `config/` | Selected MiD columns and candidate MNL attributes used by the pipeline. |
| `notebooks/` | Notebooks that are safe to share, grouped by preparation, descriptive analysis, and legacy diagnostics. |
| `scripts/` | Stage-organized commands for running the analytical workflow. |
| `src/thesis_pipeline/` | Reusable path, feature, model-input, and sensitivity implementations. |
| `.python-version` | Pins the project interpreter to Python 3.12.13. |
| `pyproject.toml` | Declares the Python project and its direct dependencies. |
| `uv.lock` | Locks resolved dependency versions for reproducible installation. |
| `.gitignore` | Keeps environments, restricted data, processed data, and generated outputs out of Git. |
| `README.md` | Public project overview and operating instructions. |

## Local/private runtime directories

The public clone contains code, configuration, environment metadata, and
shareable notebooks. Licensed data, processed data, generated artifacts, and
private reference material are supplied or created locally:

```text
data_raw/        Licensed MiD source tables
data/reference/  Private reference material and codebooks
data_processed/  Selected extracts, intermediates, and record-level outputs
outputs/         Generated tables, figures, diagnostics, and model artifacts
.venv/           Locally recreated Python environment
```

These paths are intentionally not distributed through GitHub. Git/GitHub
version-controls the public code layer; authorized private or VW storage holds
the data and analytical-artifact layer.

### Expected private input layout

The current active code expects these exact repository-relative source files:

```text
data_raw/
├── MiD2023_Haushalte.csv
├── MiD2023_Personen.csv
├── MiD2023_Wege.csv
└── MiD2023_Autos.csv

data/reference/
└── codebooks/
    └── MiD2023_Codepläne_B1_Standard_v1.1.xlsx
```

Authorized users must supply `data_raw/` and the private codebook. For a
from-scratch run, the pipeline creates `data_processed/`, while analytical
scripts write generated results to `outputs/`.

## Analytical workflow

Descriptive analysis and choice reconstruction branch from prepared data.
Association screening is a diagnostic branch from model inputs; it is not a
prerequisite for MNL estimation.

```text
                         ┌── 01_descriptive
                         │
00_data_preparation ─────┤
                         │
                         └── 02_choice_reconstruction
                                      │
                                      ▼
                                03_model_input
                                  ├── 04_screening
                                  └── 05_mnl (Stage-0, Core-V1, Core-V2)
```

### 00 — Data preparation

| Entry point | Purpose |
|---|---|
| `scripts/00_data_preparation/01_extract_selected_raw.py` | Extracts configured columns from the four authorized MiD source tables into consistently typed selected-raw files. |

### 01 — Descriptive analysis

| Entry point | Purpose |
|---|---|
| `scripts/01_descriptive/01_build_household_backbone.py` | Builds the reusable household-level backbone, including licensed-driver and household-type information. |
| `scripts/01_descriptive/02_licensed_driver_resources.py` | Compares licensed-driver resources and driver/car ratios between single- and multi-car households. |
| `scripts/01_descriptive/03_habitual_car_use_orientation.py` | Examines weighted habitual car-use orientation by car ownership, spatial context, and household type. |
| `scripts/01_descriptive/04_reference_day_mobility_intensity.py` | Compares weighted reference-day trip, distance, and car-driver mobility intensity. |
| `scripts/01_descriptive/05_driver_vehicle_bipartite_patterns.py` | Constructs and summarizes observed driver–vehicle usage links, degrees, and household mapping patterns. |
| `scripts/01_descriptive/06_allocation_sample_support.py` | Reports allocation-candidate support and joint-choice complexity after reconstruction. |

### 02 — Choice-occasion reconstruction

| Entry point | Purpose |
|---|---|
| `scripts/02_choice_reconstruction/01_phase1_home_chain.py` | Orders trips chronologically and reconstructs person-level home/away chains and recorded household fleets. |
| `scripts/02_choice_reconstruction/01b_phase1_diagnostics.py` | Reports read-only diagnostics for Phase-1 timing, home-location, sequence, and fleet flags. |
| `scripts/02_choice_reconstruction/02_phase2_vehicle_states.py` | Reconstructs household vehicle HOME/AWAY states and candidate vehicle-choice occasions. |
| `scripts/02_choice_reconstruction/03_phase3_strict_baseline.py` | Applies the strict baseline sample rules and writes classified and wide-format choice occasions. |
| `scripts/02_choice_reconstruction/04_decision_timing_sensitivity.py` | Builds the broader decision-timing sensitivity sample without overwriting the strict baseline. |
| `scripts/02_choice_reconstruction/05_allocation_analysis_candidate.py` | Builds the allocation-candidate occasion and household universe. |

### 03 — Model-input construction

| Entry point | Purpose |
|---|---|
| `scripts/03_model_input/01_build_home_based_tour_features.py` | Runs the accepted reusable construction of home-based-tour features. |
| `scripts/03_model_input/01b_build_sensitivity_tour_features.py` | Attaches the accepted tour features to the decision-timing sensitivity sample. |
| `scripts/03_model_input/02_build_mnl_vehicle_choice_model_input.py` | Runs the accepted construction of the baseline wide MNL model-input table. |
| `scripts/03_model_input/02b_build_sensitivity_model_input.py` | Builds the sensitivity model input using the fixed baseline feature schema. |
| `scripts/03_model_input/04_build_allocation_candidate_model_input.py` | Joins the allocation candidate to model attributes and tour features. |
| `scripts/03_model_input/05_build_single_driver_mnl_design_matrix.py` | Encodes the shared model-ready input for Stage-0 and both core models. |
| `scripts/03_model_input/06_build_purpose_comparison_qa.py` | Compares purpose definitions across the rebuilt analysis populations after Core-V1 preparation. |

### 04 — Association screening

| Entry point | Purpose |
|---|---|
| `scripts/04_screening/01_hierarchical_mixed_association_screening.py` | Screens candidate predictors at their natural household, person, and choice-occasion levels without selecting a final model. |
| `scripts/04_screening/02_employment_share_by_age_diagnostic.py` | Diagnoses the person-level relationship between ordered age group and employment status in the MNL sample. |

### 05 — MNL estimation

`scripts/05_mnl/01_estimate_v2_only_mnl.py` and
`scripts/05_mnl/02_estimate_v1_only_mnl.py` run the Stage-0 models. Core-V1
and Core-V2 each have preparation, diagnostics, and estimation entry points in
`scripts/05_mnl/stage1/`. Core-V1 also uses
`00_build_core_v1_baselines.py` to seal its current-run input and diagnostic
checks. These entry points run in the order specified by `run-pipeline.ps1`.

## Scripts and reusable implementation

Files under `scripts/` are workflow-facing executable or orchestration entry
points. Reusable logic shared across workflows lives under
`src/thesis_pipeline/`:

| Module | Reusable responsibility |
|---|---|
| `paths.py` | Centralized, location-independent repository paths. |
| `features/household_type.py` | Household-type feature construction from person age groups. |
| `model_input/tour_features.py` | Accepted home-based-tour feature construction. |
| `model_input/vehicle_choice.py` | Accepted vehicle-choice model-input construction. |
| `sensitivity/decision_timing.py` | Shared decision-timing sensitivity comparisons and QA. |

Legacy compatibility aliases are retained where needed; new development should
use the stage-organized entry points under `scripts/` and reusable modules under
`src/thesis_pipeline/`.

## Python environment

The project uses Python 3.12.13 and [uv](https://docs.astral.sh/uv/). The
`uv sync` command in the quick start creates the local `.venv/` from
`pyproject.toml` and `uv.lock`.

For notebook support, install the locked optional group:

```bash
uv sync --group notebooks
```

The environment is ignored by Git and is not stored in the repository.

## Supported data workflows

### From-scratch workflow

1. Clone the repository.
2. Run `uv sync`.
3. Supply the authorized raw and reference inputs shown above.
4. Run `./run-pipeline.ps1 -Mode full -Overwrite` in PowerShell.
5. The pipeline creates `data_processed/` and analytical scripts create
   `outputs/` locally.

### Existing private snapshot workflow

1. Clone the repository.
2. Run `uv sync`.
3. Restore the matching `data_processed/` and/or `outputs/` snapshot from
   approved VW or other private storage.
4. Preserve the repository-relative paths.
5. Where possible, use the snapshot associated with the corresponding Git
   revision.

Important thesis snapshots may be archived privately with a date and code
revision marker. GitHub does not supply private inputs, snapshots, or outputs.

## Validation and execution

The `scripts/00_path_smoke_test.py` path smoke test imports all active Python
modules, checks declared paths, and verifies that 13 accepted private analytical
files exist. A code-only fresh clone therefore will not pass the full smoke
test. For a from-scratch workflow, first supply the authorized raw/reference
inputs and run the required stages until the expected processed analytical
files exist. Alternatively, restore a compatible private analytical snapshot.
Then run:

```bash
uv run python scripts/00_path_smoke_test.py
```

The smoke test is read-only and does not run the analytical pipeline. To run a
workflow entry point from the repository root, use:

```bash
uv run python <path-to-stage-script>
```

### Run the maintained pipeline in PowerShell

From the repository root, use the existing `.venv`:

```powershell
.\run-pipeline.ps1 -Mode models -Overwrite
.\run-pipeline.ps1 -Mode full -Overwrite
.\run-pipeline.ps1 -Mode full -Plan
```

`models` reuses the current model-ready inputs and runs Stage-0 V2, Stage-0 V1,
Core-V1, and Core-V2. It is the routine model-development command. `full`
starts with the four authorized raw MiD2023 tables, rebuilds selected raw data,
choice occasions, tour and model inputs, descriptive outputs, screening, and
the same models. Both commands print each stage and stop on failure. The
`-Plan` lists the selected stages and existing generated outputs without
running a stage. `-Overwrite` is required when those outputs already exist.
The launcher holds protected results only while their replacement stage runs;
it commits each successful stage separately. If a stage fails, it restores
that stage's protected results and writes `outputs/.pipeline-interrupted.json`.
Resume the failed stage and remaining suffix with the same mode:

```powershell
.\run-pipeline.ps1 -Mode full -Overwrite -Resume -Plan
.\run-pipeline.ps1 -Mode full -Overwrite -Resume
```

Resume checks the completed stage order, expected output presence and model
manifest status, and SHA-256 hashes of source code, direct inputs, and
generated files. It refuses to skip stages if these differ from the failure
checkpoint. To deliberately recompute from the beginning after an
interruption, use `-Restart -Overwrite` with the same mode. A hard crash that
leaves a temporary hold or staging directory requires inspection before
rerunning. The launcher never removes the original raw MiD files or the
historical E001 file.

The full run includes the six maintained descriptive scripts, allocation
candidate support, two screening scripts, and current purpose QA. The
historical `03_model_input/07_verify_purpose_extension.py` compares old
snapshots supplied separately and is not part of current production. The
retired decision-timing comparison and frozen historical Core-V2 baseline are
not run. Tests are separate from the launcher; Core-V1 estimation runs its
own required numerical test gate.

## Data availability

MiD source and microdata are not distributed with this repository. Access must
be obtained through the appropriate authorized channel; this GitHub repository
is a research-code resource and should not be treated as a data-distribution
source.
