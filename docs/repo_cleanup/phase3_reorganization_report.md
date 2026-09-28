# Stage 3 repository reorganization report

Date: 2026-09-28  
Mode: conservative mechanical physical reorganization

## 1. Objective and constraints

Stage 3 reorganized source, configuration, reference, notebook, and cleanup-document entry points so that their physical layout follows the thesis workflow. It changed where those files live, not analytical behavior.

No analytical pipeline stage was executed. No analytical dataset, table, spreadsheet, figure, PDF, or stored notebook output was regenerated or overwritten. The 13 accepted analytical files stayed at their Phase-2B paths and retained their SHA-256 hashes. The licensed-driver backup, legacy data, local environments, package caches, and temporary content were preserved.

The Phase-1 mapping also proposed moving every raw, processed, and output artifact. That part was intentionally deferred because Stage 3 requires the licensed-driver backup to remain, repeatedly identifies the accepted files at their current paths, and external/manual thesis references to figures and tables are still unknown. Moving those artifacts would therefore have exceeded a conservative mechanical source reorganization.

## 2. Exact before/after mapping

### Configuration and reference input

| Before | After |
|---|---|
| `metadata/selected raw list.py` | `config/selected_raw_columns.py` |
| `metadata/mnl_attribute_candidates.py` | `config/mnl_attribute_candidates.py` |
| `metadata/MiD2023_Codepläne_B1_Standard_v1.1.xlsx` | `data/reference/codebooks/MiD2023_Codepläne_B1_Standard_v1.1.xlsx` |

### Data preparation and choice reconstruction

| Before | After |
|---|---|
| `scripts/01_extract_selected_raw.py` | `scripts/00_data_preparation/01_extract_selected_raw.py` |
| `scripts/02_phase1_home_chain.py` | `scripts/02_choice_reconstruction/01_phase1_home_chain.py` |
| `scripts/02_phase1_diagnostics.py` | `scripts/02_choice_reconstruction/01b_phase1_diagnostics.py` |
| `scripts/03_phase2_vehicle_states_v2.py` | `scripts/02_choice_reconstruction/02_phase2_vehicle_states.py` |
| `scripts/04_phase3_strict_baseline.py` | `scripts/02_choice_reconstruction/03_phase3_strict_baseline.py` |
| `scripts/04b_phase3_decision_timing_sensitivity.py` | `scripts/02_choice_reconstruction/04_decision_timing_sensitivity.py` |

### Model input and screening

| Before | After |
|---|---|
| `scripts/06b_build_decision_timing_sensitivity_tour_features.py` | `scripts/03_model_input/01b_build_sensitivity_tour_features.py` |
| `scripts/07b_build_mnl_vehicle_choice_model_input_decision_timing_sensitivity.py` | `scripts/03_model_input/02b_build_sensitivity_model_input.py` |
| `scripts/10_decision_timing_sensitivity_sample_comparison.py` | `scripts/03_model_input/03_decision_timing_sample_comparison.py` |
| `scripts/08_hierarchical_mixed_association_screening.py` | `scripts/04_screening/01_hierarchical_mixed_association_screening.py` |
| `scripts/09_employment_share_by_age_diagnostic.py` | `scripts/04_screening/02_employment_share_by_age_diagnostic.py` |

Two workflow-facing entry points were added at the Phase-1 target paths while retaining the established compatibility entry points:

- `scripts/03_model_input/01_build_home_based_tour_features.py`;
- `scripts/03_model_input/02_build_mnl_vehicle_choice_model_input.py`.

### Descriptive analysis

| Before | After |
|---|---|
| `scripts/eda/statistical_tests/01_build_theme1_household_backbone.py` | `scripts/01_descriptive/01_build_household_backbone.py` |
| `scripts/eda/statistical_tests/02_licensed_driver_resources.py` | `scripts/01_descriptive/02_licensed_driver_resources.py` |
| `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_boxline.py` | `scripts/01_descriptive/02b_plot_licensed_driver_resources_boxline.py` |
| `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_discrete.py` | `scripts/01_descriptive/02c_plot_licensed_driver_resources_discrete.py` |
| `scripts/eda/statistical_tests/03_habitual_car_use_orientation.py` | `scripts/01_descriptive/03_habitual_car_use_orientation.py` |
| `scripts/eda/statistical_tests/04_annual_vehicle_mileage.py` | `scripts/01_descriptive/04_annual_vehicle_mileage.py` |
| `scripts/eda/statistical_tests/04b_annual_vehicle_mileage_composition_diagnostic.py` | `scripts/01_descriptive/04b_annual_vehicle_mileage_composition_diagnostic.py` |
| `scripts/eda/statistical_tests/05_reference_day_mobility_intensity.py` | `scripts/01_descriptive/05_reference_day_mobility_intensity.py` |
| `scripts/05_driver_vehicle_bipartite_patterns.py` | `scripts/01_descriptive/06_driver_vehicle_bipartite_patterns.py` |

### Reusable household feature

| Before | After |
|---|---|
| `src/build_household_type.py` implementation | `src/thesis_pipeline/features/household_type.py` implementation |

`src/build_household_type.py` was recreated as a compatibility import/CLI wrapper. Its CLI still calls the same `main` function and retains the documented in-place write to `hh_selected_raw.csv`.

### Notebooks

| Before | After |
|---|---|
| `notebooks/CodeBook_helper.ipynb` | `notebooks/00_data_preparation/CodeBook_helper.ipynb` |
| `notebooks/RQ1.ipynb` | `notebooks/01_descriptive/RQ1.ipynb` |
| `notebooks/check_LinkageOutput.ipynb` | `notebooks/99_legacy_diagnostics/check_LinkageOutput.ipynb` |
| `notebooks/small_check.ipynb` | `notebooks/99_legacy_diagnostics/small_check.ipynb` |

### Cleanup documentation

| Before | After |
|---|---|
| `docs/repo_cleanup_audit.md` | `docs/repo_cleanup/repo_cleanup_audit.md` |
| `docs/repo_file_mapping.md` | `docs/repo_cleanup/repo_file_mapping.md` |
| `docs/path_dependency_report.md` | `docs/repo_cleanup/path_dependency_report.md` |

## 3. Files moved

Thirty-one existing files were relocated in six verified batches: three configuration/reference files, six preparation/reconstruction scripts, five model-input/screening scripts, nine descriptive scripts, the household implementation plus four notebooks, and three audit documents. Every moved notebook and the moved household implementation retained its exact pre-move content hash.

## 4. Files intentionally not moved

- All files under `data_raw/`, `data_processed/`, and `outputs/`.
- All 13 accepted analytical files and their current parent directories.
- The licensed-driver `backup/` directory.
- `data_processed/archive/` and `outputs/mnl_check.xlsx`.
- `.venv/`, `.codex/`, notebook package environments, and `tmp/`.
- `scripts/00_path_smoke_test.py`, which remains a repository-maintenance entry point.
- `scripts/05_mnl/` contains no implementation because no MNL estimator or diagnostic stage exists yet.
- Existing package implementations under `src/thesis_pipeline/model_input/` and `src/thesis_pipeline/sensitivity/`.

## 5. Compatibility entry points retained

- `scripts/06_build_home_based_tour_features.py`;
- `scripts/07_build_mnl_vehicle_choice_model_input.py`;
- `scripts/decision_timing_sensitivity_utils.py`;
- `src/build_household_type.py`.

The first two remain functional aliases to the packaged model-input implementations. The sensitivity utility remains a compatibility re-export. The household wrapper exposes the same `build_household_type` and `main` function objects as the canonical feature module.

## 6. Central path-registry changes

`src/thesis_pipeline/paths.py` now declares the target stage directories and adds `CONFIG_DIR`, `DATA_DIR`, and `REFERENCE_DATA_DIR`. Only three existing resource constants changed physical locations:

- `SELECTED_RAW_COLUMNS_PATH` → `config/selected_raw_columns.py`;
- `MNL_ATTRIBUTE_CANDIDATES_PATH` → `config/mnl_attribute_candidates.py`;
- `CODEBOOK_PATH` → `data/reference/codebooks/MiD2023_Codepläne_B1_Standard_v1.1.xlsx`.

All raw, processed, reconstruction, model-input, screening, and analytical-output constants continue to point to their existing locations. The smoke test now discovers Python configuration modules under `config/`.

The decision-timing Phase-3 sensitivity fingerprint now follows `CHOICE_RECONSTRUCTION_SCRIPTS_DIR / "03_phase3_strict_baseline.py"`. No fixed-depth root calculations or filename-based sensitivity loaders were introduced.

## 7. Notebook path-reference changes

Notebook cells, outputs, execution counts, and metadata were not altered for relocation. Before the notebook move, the CodeBook helper's missing-file instruction was mechanically updated from `scripts/01_extract_selected_raw.py` to `scripts/00_data_preparation/01_extract_selected_raw.py`.

All notebooks retain the Phase-2B `THESIS_REPO_ROOT`/upward-`.git` bootstrap and imports from `src.thesis_pipeline.paths`. Their moved files retained exact hashes, proving that the relocation itself did not alter notebook content.

## 8. Historical and legacy items preserved

- `notebooks/99_legacy_diagnostics/check_LinkageOutput.ipynb` still references the genuinely absent `data_processed/02_mnl_estimation_funnel.csv` and `data_processed/02_trip_vehicle_wide.csv`; neither file was recreated or redirected.
- The full licensed-driver backup is preserved at its original location.
- Phase-1 and Phase-2/2B reports remain unchanged as historical audit records apart from their mechanical document relocation where applicable.
- Persisted absolute Windows paths in historical notebook output text remain unchanged.

## 9. Verification results

Every relocation batch passed before the next batch began.

| Gate | Final result |
|---|---|
| Path smoke test | PASS |
| Project modules imported | 37 |
| Declared `Path` constants resolved | 291 |
| Declared paths outside repository | 0 |
| Python source compilation | PASS |
| Notebook JSON parsing and code compilation | PASS: 4 notebooks, 36 code cells |
| Fixed-depth `parents[n]` root expressions | 0 |
| Filename-based sensitivity loaders | 0 |
| Active Python backup dependencies | 0 |
| Moved notebook/household content hash mismatches | 0 |
| Accepted analytical hash mismatches | 0 |
| `git diff --check` | PASS |
| Analytical pipeline stages executed | 0 |

The smoke test also passed when launched from the repository parent directory, confirming that the relocated module tree does not depend on the caller's working directory.

## 10. Accepted-file SHA-256 status

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

## 11. Remaining risks and technical debt

- The analytical artifact trees still use the pre-Stage-3 `data_raw/`, `data_processed/`, and `outputs/eda/` layout. Moving them requires a separate decision about external thesis links and is not merely an internal import change.
- The household-type compatibility CLI intentionally mutates `hh_selected_raw.csv` in place.
- The legacy linkage notebook remains non-runnable without its absent historical inputs.
- No reproducible environment manifest exists yet; `.venv/` remains necessary local state.
- No Stage-05 MNL specification, estimation, or diagnostic implementation exists.
- The working tree contains the cumulative uncommitted Phase-2, Phase-2B, and Stage-3 changes; a reviewed checkpoint commit is advisable before further cleanup.

## 12. Later archival/deletion candidates

No analytical file is approved for deletion by Stage 3. Candidates requiring later manual/provenance review remain:

- `outputs/mnl_check.xlsx`;
- `data_processed/archive/mnl_vehicle_choice_long_base_unweighted.csv`;
- the licensed-driver backup tree, after thesis references are confirmed;
- historical review Markdown files, after confirming how they are cited;
- additional empty obsolete directories, if later created by tools; Stage 3 already removed the verified-empty `metadata/` and `scripts/eda/` directories;
- local environments/caches only after a reproducible environment specification exists.

## 13. READY FOR STAGE 4? YES

The active source/orchestration tree, configuration paths, reusable package modules, notebooks, and cleanup documentation now follow the accepted workflow structure without changing analytical content. Stage 4 may begin only as a separately authorized pass. It should first resolve external thesis references before moving or archiving analytical artifacts and should not infer or invent the absent MNL stage.
