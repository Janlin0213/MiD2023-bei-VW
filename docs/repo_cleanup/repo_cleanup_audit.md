# Repository cleanup audit

Audit date: 2026-09-28  
Mode: safety-first, documentation-only

## Scope and safeguards

This pass inspected repository-controlled code, notebooks, data, outputs, and local generated clutter. It did not delete, move, rename, overwrite, regenerate, or modify any pre-existing file, and it did not execute the analytical pipeline. The only new files are the three requested audit documents under `docs/`.

The file-level inventory covers 239 non-cache project files: 7 raw CSVs, 44 processed analytical files, 3 metadata/configuration files, 4 notebooks, 156 output artifacts, 23 scripts, `src/build_household_type.py`, and `.gitignore`. Third-party/environment contents under `.venv/`, `.codex/python-packages/`, and `notebooks/.python-packages/` are summarized as generated environment trees rather than listed package-by-package. Git internals are outside the analytical audit.

Classification codes used in the mapping:

| Code | Meaning |
|---|---|
| A | Source code or executable notebook/configuration code |
| B | Raw/input/reference data |
| C | Intermediate analytical dataset |
| D | Canonical/final analytical output |
| E | Diagnostic or QA output |
| F | Presentation/thesis artifact |
| G | Cache or temporary/generated environment file |
| H | Obsolete, legacy, backup, or duplicate candidate; preservation required |
| I | Unknown / cannot safely classify |

The 239-file inventory contains 31 A, 8 B, 28 C, 73 D, 31 E, 35 F, 31 H, and 2 I classifications. Generated cache/environment trees are summarized separately as G (or G/I where process ownership could not be verified).

## Current-state summary

The active analytical backbone is coherent but its physical layout does not follow the thesis workflow. The main dependency chain is:

`data_raw` → selected raw extracts → chronological trip chains → household vehicle states → strict/sensitivity choice occasions → tour features → model input → association screening/diagnostics.

The descriptive branch starts from selected raw extracts and a household backbone, then produces licensed-driver, habitual-use, annual-mileage, reference-day-mobility, and driver–vehicle bipartite outputs.

Important findings:

- The repository has no current MNL specification/estimation/diagnostic script or result set. The implemented workflow stops at model-input construction and association screening. `outputs/mnl_check.xlsx` is not an MNL estimation result.
- Stage numbers are misleading. `scripts/05_driver_vehicle_bipartite_patterns.py` belongs to descriptive analysis, while scripts 06–10 mix reconstruction, model-input, screening, and sensitivity diagnostics.
- Intermediate datasets, final tables, diagnostics, and figures are split between `data_processed/` and `outputs/`. Screening figures and QA tables currently live beside the canonical model input.
- The raw/processed/output trees are ignored by Git. Reproducibility therefore depends on local files whose exact inventory and provenance are not versioned.
- No environment specification such as `pyproject.toml`, `requirements.txt`, or a repository-level lockfile was found. The local `.venv/` should not be removed until dependencies are captured.
- Several scripts derive the project root from their current file depth. Moving them into stage subdirectories without changing that logic would redirect every input and output path.
- Several fresh-run directory assumptions exist: Phase 2, Phase 3, screening, and the annual-mileage composition diagnostic write into directories they do not create themselves.
- `src/build_household_type.py`, when run as a CLI, overwrites `data_processed/selected_raw/hh_selected_raw.csv` in place. The same file is initially produced by extraction. This is valid current behavior but weak provenance because an “extract” can later be mutated.

## Dependency overview by thesis stage

| Thesis stage | Current producers | Principal current outputs |
|---|---|---|
| 00 data preparation | `scripts/01_extract_selected_raw.py`; optionally `src/build_household_type.py` | `data_processed/selected_raw/*.csv` |
| 01 descriptive: household backbone | `notebooks/RQ1.ipynb`; `scripts/eda/statistical_tests/01_build_theme1_household_backbone.py` | `outputs/descriptive/*`; `data_processed/eda/statistical_test/theme1_household_backbone*.csv` |
| 01 descriptive: licensed-driver resources | `02_licensed_driver_resources.py` plus two plot scripts | Current ratio outputs plus legacy `backup/` outputs |
| 01 descriptive: habitual car use | `03_habitual_car_use_orientation.py` | `outputs/eda/.../02_habitual_car_use_orientation/*` |
| 01 descriptive: annual vehicle mileage | `04_annual_vehicle_mileage.py`; `04b_...composition_diagnostic.py` | `outputs/eda/.../03_annual_vehicle_mileage/*` |
| 01 descriptive: reference-day mobility | `05_reference_day_mobility_intensity.py` | `outputs/eda/.../04_reference_day_mobility_intensity/*` |
| 01 descriptive: bipartite usage | `scripts/05_driver_vehicle_bipartite_patterns.py` | `data_processed/eda/bipartite/*`; `outputs/eda/bipartite/*` |
| 02 choice reconstruction, Phase 1 | `scripts/02_phase1_home_chain.py` | enriched trip chain and QA summary |
| 02 choice reconstruction, Phase 2 | `scripts/03_phase2_vehicle_states_v2.py` | vehicle events, all occasions, QA |
| 02 choice reconstruction, Phase 3 | `scripts/04_phase3_strict_baseline.py` | classified occasions, strict wide base, funnel, availability QA |
| 02 decision-timing sensitivity | `04b_...`, `06b_...`, `07b_...`, `10_...` | broader occasions, sensitivity tour/model inputs, comparison tables |
| 03 model-input construction | `06_build_home_based_tour_features.py`; `07_build_mnl_vehicle_choice_model_input.py` | tour tables, canonical model input, manifest, QA |
| 04 screening | `08_hierarchical_mixed_association_screening.py`; `09_employment_share_by_age_diagnostic.py` | pairwise/cross-level tables, heatmaps, QA, employment diagnostic |
| 05 MNL | No implementation found | None found |

## Definite generated clutter

The following are safe cleanup candidates because they are interpreter caches, not analytical artifacts. They were not removed.

- Six project cache directories: `metadata/__pycache__/`, `notebooks/__pycache__/`, `notebooks/.python-packages/__pycache__/`, `scripts/__pycache__/`, `scripts/eda/statistical_tests/__pycache__/`, and `src/__pycache__/`.
- Nineteen project `.pyc` files outside third-party package trees, totaling 720,950 bytes.
- Two especially stale bytecode files have no matching current source filename: `scripts/__pycache__/07_phase3_strict_baseline.cpython-312.pyc` and `scripts/__pycache__/08_driver_vehicle_bipartite_patterns.cpython-312.pyc`.
- No `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, editor swap file, editor backup, or notebook checkpoint was found in the project-controlled tree.
- `tmp/tmpu9hrg6vp` and some local package-cache paths could not be enumerated because access was denied. They look generated, but should only be cleaned after the owning process is closed and the exact target is rechecked.

The `.venv/` tree is generated, but is **not** marked safe to delete in this audit because the repository does not contain a complete environment specification from which it can be reliably recreated.

## Likely redundant, legacy, or superseded candidates

These are category H candidates, not approved deletions:

- `outputs/mnl_check.xlsx` is produced by `notebooks/small_check.ipynb`. No consumer was found. It is exactly value-equal to `data_processed/reconstruction/phase3/mnl_vehicle_choice_wide_base.csv`: same 30,816 rows, 14 columns, order, values, and `CHOICE_ID`s. It remains preserved because spreadsheet artifacts require explicit review.
- `data_processed/archive/mnl_vehicle_choice_long_base_unweighted.csv` is an unweighted long-format legacy artifact. No current producer or consumer was found.
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/backup/**` is a historical output set. No byte-identical duplicates exist under `outputs/`, but many backup filenames represent an older licensed-driver analysis that the current main script no longer produces.
- `notebooks/check_LinkageOutput.ipynb` reads missing legacy files `data_processed/02_mnl_estimation_funnel.csv` and `data_processed/02_trip_vehicle_wide.csv`.
- `notebooks/small_check.ipynb` reads missing legacy paths directly under `data_processed/` rather than the current `reconstruction/phase3/` paths.
- `notebooks/CodeBook_helper.ipynb` correctly reads raw/codebook inputs, but its selected-raw helper still points to now-missing files directly under `data_processed/` rather than `data_processed/selected_raw/`.
- `RESULT_REVIEW_driver_car_ratio.md`, `habitual_car_use_orientation_results_review.md`, and the licensed-driver backup review Markdown have no scripted producer. They may be manually curated thesis notes and must be preserved.

Hashing all 156 files under `outputs/` found no byte-identical file pairs. Similar names, PNG/PDF pairs, and backup/main variants are therefore not assumed to be duplicates.

## Uncertain files that must not be deleted

- All CSV, XLSX, PNG, PDF, notebook, and Markdown analytical artifacts, including those with no current consumer.
- The three raw subfiles not referenced by current code: `MiD2023_Etappen.csv`, `MiD2023_Reisen.csv`, and `MiD2023_Tagesreisen.csv`. Lack of a current reference is not evidence that they are disposable.
- All legacy/backup artifacts above until the thesis author confirms whether they document earlier specifications or feed external thesis writing.
- `.venv/`, `.codex/`, `.python-packages/`, and the access-denied `tmp/` content until reproducibility and process ownership are resolved.
- Any output not referenced by Python may still be included manually in the thesis or presentation.

## Major structural and provenance problems

1. **Stage and location mismatch.** Descriptive artifacts are split across `outputs/descriptive`, `outputs/eda`, and `data_processed/eda`; model-input screening outputs are stored in `data_processed/model_input`.
2. **Root discovery depends on script depth.** Root scripts use `Path(__file__).resolve().parents[1]`; EDA scripts use `parents[3]`. Both break when files are moved to the proposed stage folders.
3. **Notebook paths depend on the launch directory.** `RQ1.ipynb` uses `Path('..').resolve()` and is only correct when the kernel CWD is `notebooks/`. Other notebooks use a two-case CWD heuristic and retain legacy data locations.
4. **Local imports depend on current placement.** Three sensitivity scripts import `decision_timing_sensitivity_utils` as a sibling. The utility dynamically loads exact filenames from `ROOT/scripts`. The model builder mutates `sys.path` before importing `src.build_household_type`.
5. **Directory creation is inconsistent.** Some scripts create all parents, while others assume upstream directories already exist. This makes a clean rebuild order-sensitive beyond data dependencies.
6. **Self-read/overwrite behavior weakens provenance.** The model-input builder reads a prior model input if present before writing the canonical path; the household-type CLI overwrites a selected-raw extract; several analyses read outputs they created earlier in the same run.
7. **Legacy paths are still executable documentation.** Old notebooks can fail or, if matching old paths reappear, write ambiguous outputs.
8. **No explicit Stage 05.** MNL estimation and diagnostics are absent, so the repository cannot yet mirror the complete thesis workflow.

## Recommended target structure

```text
config/
  selected_raw_columns.py
  mnl_attribute_candidates.py
data/
  raw/
  reference/codebooks/
scripts/
  00_data_preparation/
  01_descriptive/
  02_choice_reconstruction/
  03_model_input/
  04_screening/
  05_mnl/
src/thesis_pipeline/
  paths.py
  features/
  sensitivity/
outputs/
  00_data_preparation/
  01_descriptive/
  02_choice_reconstruction/
  03_model_input/
  04_screening/
  05_mnl/
archive/
  legacy/
notebooks/
  00_data_preparation/
  01_descriptive/
  99_legacy_diagnostics/
docs/
  analysis_reviews/
  repo_cleanup/
```

Within each output stage, use `intermediate/`, `final/`, `diagnostics/`, and `figures/` only where needed. This preserves the workflow order without pretending that every table is a final thesis result.

## Recommended cleanup order

1. Freeze the current state: record file hashes for raw inputs and all accepted canonical outputs; capture the Python environment; identify thesis documents that link to output paths.
2. Add a shared `src/thesis_pipeline/paths.py` and refactor all scripts/notebooks to use it while files remain in place.
3. Add smoke tests that resolve every declared input/output path and import every local module without running analysis.
4. Fix fresh-run directory creation and separate in-place mutation (`build_household_type`) from immutable extraction outputs.
5. Move source code stage-by-stage, updating imports and dynamic-loader references first.
6. Move outputs stage-by-stage, starting with terminal artifacts and then intermediates in dependency order; verify hashes after every move.
7. Quarantine H candidates under `archive/legacy/`; do not delete them until thesis references and provenance are signed off.
8. Remove only proven generated clutter (`__pycache__`, `.pyc`, tool caches) after active Python/Jupyter processes are closed.
9. Add the Stage 05 MNL implementation and result layout when its specification is available.

## Phase-2 action plan

Phase 2 should first introduce centralized paths and import-safe shared modules without changing analytical logic. Then add path/import smoke tests, capture hashes and the environment, and migrate one stage at a time in dependency order. After each stage, compare file counts, schemas, row counts, and hashes where byte stability is expected. Legacy/backup artifacts should be moved only into a clearly labelled archive, never deleted without explicit thesis-owner approval.
