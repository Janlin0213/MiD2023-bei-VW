# Phase 2 repository stabilization report

Date: 2026-09-28  
Scope: conservative path/import/output-directory stabilization and proven cache cleanup

## Outcome

Phase 2 completed without moving or renaming analytical scripts, data, outputs, or notebooks. No analytical pipeline stage was rerun. No CSV, XLSX, PNG, PDF, Markdown analysis artifact, or notebook was deleted. The accepted datasets listed below retain their exact pre-change SHA-256 hashes.

The changes are limited to repository path discovery, shared-import plumbing, output-parent creation, a read-only smoke test, documentation, and deletion of interpreter bytecode caches.

## Centralized path design

`src/thesis_pipeline/paths.py` is now the single registry for the current physical layout. It resolves the repository root in this order:

1. use `THESIS_REPO_ROOT` when set and pointing to an existing directory;
2. walk upward from an explicit start, the current working directory, and the path module itself until `.git` is found;
3. fail clearly if neither mechanism resolves a root.

The registry intentionally retains the current directories: `data_raw`, `data_processed`, `outputs`, `metadata`, `scripts`, and `src`. It exposes named directories for selected raw data, reconstruction Phases 1–4, decision-timing sensitivity, model input, screening, descriptive outputs, bipartite outputs, Theme-1 outputs, and the codebook/configuration files.

Directly executed scripts use a small `.git` marker walk only to make the repository root importable, then obtain all repository paths from `src.thesis_pipeline.paths`. No script calculates the repository root with a fixed `parents[n]` depth anymore.

## Files added

- `src/thesis_pipeline/__init__.py`
- `src/thesis_pipeline/paths.py`
- `src/thesis_pipeline/sensitivity/__init__.py`
- `src/thesis_pipeline/sensitivity/decision_timing.py`
- `scripts/00_path_smoke_test.py`
- `docs/repo_cleanup/phase2_stabilization_report.md`

## Existing source files modified

- `scripts/01_extract_selected_raw.py`
- `scripts/02_phase1_diagnostics.py`
- `scripts/02_phase1_home_chain.py`
- `scripts/03_phase2_vehicle_states_v2.py`
- `scripts/04_phase3_strict_baseline.py`
- `scripts/04b_phase3_decision_timing_sensitivity.py`
- `scripts/05_driver_vehicle_bipartite_patterns.py`
- `scripts/06_build_home_based_tour_features.py`
- `scripts/06b_build_decision_timing_sensitivity_tour_features.py`
- `scripts/07_build_mnl_vehicle_choice_model_input.py`
- `scripts/07b_build_mnl_vehicle_choice_model_input_decision_timing_sensitivity.py`
- `scripts/08_hierarchical_mixed_association_screening.py`
- `scripts/09_employment_share_by_age_diagnostic.py`
- `scripts/10_decision_timing_sensitivity_sample_comparison.py`
- `scripts/decision_timing_sensitivity_utils.py`
- `scripts/eda/statistical_tests/01_build_theme1_household_backbone.py`
- `scripts/eda/statistical_tests/02_licensed_driver_resources.py`
- `scripts/eda/statistical_tests/03_habitual_car_use_orientation.py`
- `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_boxline.py`
- `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_discrete.py`
- `scripts/eda/statistical_tests/04_annual_vehicle_mileage.py`
- `scripts/eda/statistical_tests/04b_annual_vehicle_mileage_composition_diagnostic.py`
- `scripts/eda/statistical_tests/05_reference_day_mobility_intensity.py`
- `src/build_household_type.py`
- `docs/path_dependency_report.md`

All Python edits above are path/import/directory-creation plumbing. `src/build_household_type.py` received only central path use and parent-directory creation before its existing write; household-type definitions and calculations were not changed.

## Scripts migrated to the registry

All 23 files under `scripts/` that existed before Phase 2 and `src/build_household_type.py` now import repository locations from `src.thesis_pipeline.paths`. The two metadata Python files contain configuration data and do not calculate repository paths. A repository-wide search found zero remaining `Path(__file__).resolve().parents[n]` expressions.

The three decision-timing sensitivity consumers now import shared helpers from `src.thesis_pipeline.sensitivity.decision_timing` rather than a fragile sibling import:

- `06b_build_decision_timing_sensitivity_tour_features.py`
- `07b_build_mnl_vehicle_choice_model_input_decision_timing_sensitivity.py`
- `10_decision_timing_sensitivity_sample_comparison.py`

The original `scripts/decision_timing_sensitivity_utils.py` remains in place for compatibility and now uses the centralized registry. It is not deleted or renamed in this pass.

## Output-directory creation fixes

The following previously identified risks now create required parent directories before writing:

- Phase 2 vehicle events, occasions, and QA outputs;
- Phase 3 classified occasions, funnel, wide baseline, and QA outputs;
- bipartite processed-data and presentation-output directories;
- association-screening tables, QA, and heatmaps;
- annual-mileage composition diagnostic tables and Markdown;
- employment-by-age diagnostic table and figure;
- the existing household-type CLI write path.

Other writers already created their output directory and were left unchanged apart from registry use.

## Deleted interpreter caches

Only project-owned `__pycache__` directories were removed. No package environment or analytical artifact was touched.

| Deleted directory | Files | Bytes |
|---|---:|---:|
| `metadata/__pycache__/` | 2 | 5,064 |
| `notebooks/__pycache__/` | 1 | 28,424 |
| `scripts/__pycache__/` | 10 | 417,782 |
| `scripts/eda/statistical_tests/__pycache__/` | 5 | 262,072 |
| `src/__pycache__/` | 1 | 7,608 |
| **Total** | **19** | **720,950** |

The deleted set included stale bytecode with no matching current source: `notebooks/__pycache__/link_wege_autos_core.cpython-312.pyc`, `scripts/__pycache__/07_phase3_strict_baseline.cpython-312.pyc`, and `scripts/__pycache__/08_driver_vehicle_bipartite_patterns.cpython-312.pyc`.

Post-cleanup verification found zero project `__pycache__` directories and zero project `.pyc` files outside the explicitly preserved notebook package environment.

## Before/after hash verification

All hashes were recorded before source changes and recomputed after the refactor and cache cleanup.

| Accepted file | SHA-256 before | SHA-256 after | Result |
|---|---|---|---|
| `data_processed/selected_raw/hh_selected_raw.csv` | `97b6ec4784a539a00929443b2d4d0710cd07d11954064a88908211d172c1313f` | same | PASS |
| `data_processed/selected_raw/persons_selected_raw.csv` | `698fbab5c2cd35398131885649c0700415bccbae01176b440f488af68a3058bc` | same | PASS |
| `data_processed/selected_raw/trips_selected_raw.csv` | `3ca4357094082da5ec60e03ee4a3bb36bbf271049e9cbf2781b2bb229b63a455` | same | PASS |
| `data_processed/selected_raw/cars_selected_raw.csv` | `58ef900ca894a0aea7688bfcf8a951d834d389e44c116ec8c4868259608291a9` | same | PASS |
| `data_processed/reconstruction/phase1/trips_home_chain_enriched.csv` | `d999ff20a641c8bc40306363cfe677c326f6e6d7acd30bfd5fe888a9679c6fa7` | same | PASS |
| `data_processed/reconstruction/phase2/vehicle_events_v2.csv` | `f19dca8422beab49624feb7f4d5f48a019280dea8223610600d4bd1b115003d8` | same | PASS |
| `data_processed/reconstruction/phase2/vehicle_choice_occasions_all_v2.csv` | `16e0d8b0d162d315a842de3e8dfd7b5bdde8abe4e1732d63be77366be1a2d1bf` | same | PASS |
| `data_processed/reconstruction/phase3/mnl_vehicle_choice_wide_base.csv` | `0a34fa61a170758f5291cb3ba76e2d8eefb363b0650c2d15930e2aeff24a651d` | same | PASS |
| `data_processed/reconstruction/phase3/vehicle_choice_occasions_classified.csv` | `fa3d2a35aeeef0dbc3d9db6e4de7974547ce68882eee2c0d6d4bd731250e54aa` | same | PASS |
| `data_processed/reconstruction/phase3_sensitivity/mnl_vehicle_choice_wide_decision_timing_sensitivity_base.csv` | `dbe3a4a47cf943420e221ab1d07a53526008c061d2e748034377e312692e766c` | same | PASS |
| `data_processed/reconstruction/phase4/mnl_vehicle_choice_wide_tour_features.csv` | `9a462669edfbc77b9131ddedd1fc5f3cace0e06a4048966f3628c6ced56f1066` | same | PASS |
| `data_processed/model_input/mnl_vehicle_choice_model_input.csv` | `c3a9d4b4c3925f79d370d6ba881ff58f414c088eddd89af62efc00ca01f0fddc` | same | PASS |
| `data_processed/model_input/mnl_vehicle_choice_model_input_manifest.csv` | `30f678188142b488fadd787163bee69bbdd45374c36168f4d70b99558c82ec85` | same | PASS |

Hash mismatches: **0**.

## Path-only verification

`scripts/00_path_smoke_test.py` is read-only and sets no output-generating code in motion. With bytecode writing disabled it:

- imported all 30 Python modules other than the smoke-test module itself;
- resolved 262 module-level `Path` constants;
- verified all declared `Path` constants are inside the repository;
- verified the 13 accepted files above exist at their unchanged physical paths;
- passed when launched from the repository root;
- passed when launched from the repository's parent directory;
- confirmed the `THESIS_REPO_ROOT` override resolves correctly.

No accepted analytical output was overwritten for testing.

## Remaining fragile imports and behavior left unchanged

The shared sensitivity module still dynamically loads these accepted scripts by exact filename:

- `scripts/06_build_home_based_tour_features.py`
- `scripts/07_build_mnl_vehicle_choice_model_input.py`

Replacing those loads with normal imports would require extracting or packaging analytical functions from the accepted scripts. That change could exceed path-only plumbing, so it was intentionally deferred. The loader now obtains `SCRIPTS_DIR` centrally, which removes script-depth dependence but not filename dependence.

The selected-column and MNL-candidate registries are also loaded dynamically because their current filenames/modules are not conventional importable package names. Their paths are now centralized; normalizing their names is deferred to structural Phase 3.

## Notebook paths intentionally unresolved

No notebook JSON, analysis cell, or stored output was modified.

- `notebooks/RQ1.ipynb` still relies on `Path('..').resolve()` and therefore on its kernel working directory.
- `notebooks/CodeBook_helper.ipynb` still points selected-raw helpers to files directly under `data_processed/` instead of `data_processed/selected_raw/`.
- `notebooks/check_LinkageOutput.ipynb` still references missing legacy files `data_processed/02_mnl_estimation_funnel.csv` and `data_processed/02_trip_vehicle_wide.csv`.
- `notebooks/small_check.ipynb` still references legacy processed paths directly under `data_processed/`.

These are documented rather than repaired because doing so safely requires notebook-by-notebook intent review.

## Licensed-driver backup dependency

Both licensed-driver plotting scripts declare two summary inputs in the active output directory:

- `licensed_drivers_weighted_summary_household_type.csv` — **missing from the active directory**, present only at `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/backup/licensed_drivers_weighted_summary_household_type.csv` (SHA-256 `c344e6d596127884bf90f016b9788412b18df151cc63a185b59e498f8a1c3deb`).
- `driver_car_ratio_exact_weighted_summary_household_type.csv` — present in the active directory; no file with that exact name exists in `backup/`.

Therefore the backup directory cannot be archived or deleted. The plotting scripts remain non-runnable from their configured paths until provenance is reviewed and the missing licensed-driver summary is deliberately restored or regenerated by an approved workflow. No file was copied from backup in this pass.

## Preserved intentionally

- All raw, processed, intermediate, final, diagnostic, figure, spreadsheet, PDF, Markdown, and notebook artifacts.
- `outputs/mnl_check.xlsx`.
- The licensed-driver `backup/` tree and `data_processed/archive/`.
- `.venv/`, `.codex/`, `notebooks/.python-packages/`, and `tmp/`.
- Notebook analytical results and legacy notebook paths.
- Household-type analytical definitions and values.

## Recommended Phase 3 actions

1. Review and fix notebook path initialization without clearing stored results.
2. Resolve the licensed-driver summary provenance and make the active plotting inputs self-contained before archiving backup outputs.
3. Package the accepted Phase-4 and model-input reusable functions so the two exact-filename dynamic loads can become normal imports.
4. Capture a reproducible environment specification before considering `.venv/` cleanup.
5. Add automated tests for the path registry, direct script launch from arbitrary working directories, and declared writer parents.
6. Only after those gates pass, execute the stage-by-stage physical reorganization proposed in the Phase-1 mapping.

## Phase 2 readiness decision (superseded by Phase 2B below)

**READY FOR PHASE 3? NO.** Central path plumbing and cache cleanup are complete, but the licensed-driver backup dependency, stale notebook paths, and two exact-filename dynamic loaders should be resolved before physical restructuring.

## Phase 2B — pre-move stabilization

Date: 2026-09-28  
Scope: resolve the remaining relocation blockers without moving analytical entry points, changing calculations, or regenerating accepted outputs

### RESOLVED BLOCKERS

#### Licensed-driver summary provenance

No current Python script reads a file from the licensed-driver `backup/` directory. The two current consumers are:

- `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_boxline.py`;
- `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_discrete.py`.

Both consumers read the normal active output path for:

- `licensed_drivers_weighted_summary_household_type.csv`;
- `driver_car_ratio_exact_weighted_summary_household_type.csv`.

The ratio summary is already present in the active directory. The licensed-driver summary was absent there and existed only as the preserved historical file `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/backup/licensed_drivers_weighted_summary_household_type.csv`.

The logical active producer is `scripts/eda/statistical_tests/02_licensed_driver_resources.py`. Its existing `analysis_sample` and `weighted_summary` functions reproduce the preserved backup file byte-for-byte when called for `n_licensed_drivers`, the household-type/car-ownership grouping, and the documented licensed-driver scope. The producer now writes only that missing summary to the normal active output directory. Existing ratio samples, models, QA rows, calculations, and outputs are unchanged. The pipeline was not executed, so the active CSV was not materialized or treated as an accepted regenerated output in this pass.

The backup directory remains preserved and is no longer an active code dependency.

#### Sensitivity imports and loaders

The accepted Phase-4 and model-input implementations are now importable package modules:

- `src/thesis_pipeline/model_input/tour_features.py`;
- `src/thesis_pipeline/model_input/vehicle_choice.py`.

The existing script paths remain as thin compatibility entry points. Baseline and sensitivity workflows import the same packaged function objects. Scripts 06b and 07b no longer call `load_script`, and the shared sensitivity module no longer contains a filename loader. A repository-wide Python search found no dependency on the exact filenames `06_build_home_based_tour_features.py` or `07_build_mnl_vehicle_choice_model_input.py`. Identity checks confirmed that the compatibility entry points expose the same `main` and reusable function objects as the package modules.

#### Notebook path initialization

Only initialization/path source lines were updated in:

- `notebooks/RQ1.ipynb`;
- `notebooks/CodeBook_helper.ipynb`;
- `notebooks/check_LinkageOutput.ipynb`;
- `notebooks/small_check.ipynb`.

Each notebook now bootstraps from `THESIS_REPO_ROOT` or an upward `.git` search and then imports named directories from `src.thesis_pipeline.paths`. `CodeBook_helper.ipynb` now resolves selected raw files through `SELECTED_RAW_DIR`; `small_check.ipynb` resolves its current selected-raw and Phase-3 files through the registry. Stored outputs were not cleared or regenerated. All notebook JSON parsed and every code cell compiled after the edit.

### REMAINING BLOCKERS

None of the remaining known issues blocks physical relocation of active scripts. Repository-root discovery, package/local imports, sensitivity imports, and active licensed-driver dependencies are relocation-safe.

`notebooks/check_LinkageOutput.ipynb` remains a legacy diagnostic whose two requested inputs do not exist: `data_processed/02_mnl_estimation_funnel.csv` and `data_processed/02_trip_vehicle_wide.csv`. They were not recreated or redirected because their intended historical contracts cannot be inferred safely. This notebook is not an active pipeline dependency.

### ACCEPTED DOCUMENTED RISKS

- Running `src/build_household_type.py` as a CLI still writes `hh_selected_raw.csv` in place. Its import/path handling is centralized, but the mutation contract is intentionally unchanged.
- The licensed-driver backup directory is preserved for historical/thesis review even though active code no longer reads it.
- The newly supported active licensed-driver summary is intentionally not generated during this stabilization pass; normal producer-before-plotter execution will create it.
- Notebook output cells may retain historical absolute path text. Only executable path initialization was stabilized.

### Phase 2B verification

- Path smoke test: PASS — 33 modules imported, 277 declared `Path` constants resolved, 13 accepted files found, and zero declared paths outside the repository.
- Compatibility entry-point identity checks: PASS for the Phase-4 merge and all model-input functions used by sensitivity workflows.
- Exact-filename sensitivity-loader search: zero matches.
- Active Python `backup/` reference search: zero matches.
- Notebook JSON and code-cell compilation: PASS for all four notebooks.
- Accepted-file SHA-256 comparison: all 13 hashes remain identical to the Phase-2 baseline; mismatches: 0.
- `git diff --check`: PASS.

### READY FOR PHASE 3? YES

The four relocation gates are satisfied: repository-root resolution is location-independent, local imports are package-based, sensitivity workflows no longer load scripts by filename, and active plotting code no longer relies on a backup directory. Phase 3 may proceed as a mechanical relocation with the existing hash and smoke-test gates retained.
