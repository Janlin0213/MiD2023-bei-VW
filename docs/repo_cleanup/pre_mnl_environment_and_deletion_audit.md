# Pre-MNL environment and deletion audit

Date: 2026-09-28

Pass: conservative pre-MNL maintenance

Scope: reproducible environment metadata and a read-only deletion audit

## Scope and safeguards

This pass did not run an analytical producer, regenerate an accepted output,
change analytical code, or delete a script, dataset, figure, table, report, or
backup. The four deletion candidates were inspected as source and searched
across active Python, notebooks, cleanup documentation, and the available
presentation/thesis artifact types. The repository contains no `.docx`,
`.pptx`, `.tex`, or `.qmd` thesis artifact against which figure citations could
be conclusively checked; that is why non-core candidates remain conditional.

## Reproducible environment

### State before this pass

- Tooling: uv 0.11.21 with a uv-created repository-local `.venv`.
- Interpreter: CPython 3.12.13.
- No `pyproject.toml`, `uv.lock`, `.python-version`, root `README.md`,
  requirements file, Poetry file, Pipfile, or Conda environment file existed.
- `.gitignore` already covered `.venv/`, `__pycache__/`, `*.py[cod]`, pytest,
  mypy, and Ruff caches.

### Metadata added

- `pyproject.toml`: non-package uv project, Python 3.12, direct runtime
  dependencies, and a notebook dependency group.
- `uv.lock`: complete lock of direct and transitive packages.
- `.python-version`: `3.12.13`.
- `README.md`: setup commands, smoke-test command, local `.venv` behavior, and
  the licensed MiD-data boundary.
- `.gitignore`: retained environment/cache and licensed data exclusions, but
  removed the blanket exclusion of canonical `outputs/` content.

No source package is ignored. Raw and record-level paths (`data/`, `data_raw/`,
and `data_processed/`) remain excluded because they may contain licensed MiD
material. The canonical aggregate `outputs/` tree is explicitly unignored,
including files that otherwise match generic CSV/XLSX patterns. This makes
analytical outputs visible for explicit case-by-case review; it does not stage
or commit them.

### Declared direct dependencies

| Dependency | Declared version | Evidence |
|---|---:|---|
| Python | 3.12.13 | Existing uv `.venv`; pinned by `.python-version`, constrained to 3.12 in `pyproject.toml` |
| numpy | 2.4.6 | Imported by active scripts and notebooks |
| pandas | 2.3.3 | Imported throughout active scripts and notebooks |
| matplotlib | 3.11.0 | Imported by active descriptive/diagnostic scripts and RQ1 notebook |
| scipy | 1.18.0 | Imported by active statistical scripts |
| statsmodels | 0.14.6 | Imported by active descriptive/statistical scripts; two modules explicitly assert this version |
| openpyxl | 3.1.5 | Required by active `pandas.read_excel` calls and notebook Excel I/O |
| ipython | 9.14.1 | Notebook-only direct import (`IPython.display`) |
| ipykernel | 7.3.0 | Notebook execution kernel; notebook dependency group |

Biogeme 3.3.3 happened to be installed in the old local `.venv`, but no active
script or notebook imports it. It is therefore intentionally absent from the
project metadata. Stage-05 should add the selected MNL package only when the
implemented modeling code makes it an active dependency.

Import classification used for the dependency decision:

- Standard library: `argparse`, `collections`, `dataclasses`, `gc`, `hashlib`,
  `importlib`, `io`, `itertools`, `math`, `os`, `pathlib`, `sys`, `textwrap`,
  `time`, `types`, `typing`, and `warnings`.
- Direct runtime dependencies: NumPy, pandas, Matplotlib, SciPy, statsmodels,
  and openpyxl (the Excel engine required by active pandas Excel calls).
- Notebook-only dependencies: IPython and ipykernel in the `notebooks` group.
- Development-only dependencies: none justified by the active repository.
- Local imports: `src.thesis_pipeline`, `config`, and compatibility `paths`
  imports are repository code, not external packages.

### Git-ignore status

- `.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` are ignored.
- The canonical `outputs/` tree is explicitly visible for case-by-case review.
- Licensed/raw and record-level analytical data directories remain ignored.
- No active source directory or Python-source pattern is ignored.

### Setup commands

Standard active-code environment and read-only validation:

```powershell
uv sync
uv run python scripts/00_path_smoke_test.py
```

Notebook tools are optional:

```powershell
uv sync --group notebooks
```

## Deletion audit method

For each candidate, the audit inspected all declared paths, `read_csv` calls,
calculation functions, write/save calls, command-line modes, and local imports.
It then searched the repository for the script basename and each distinctive
output basename. "No downstream reference" means no scripted consumer or
notebook reference was found; it does not prove that a file is absent from an
external thesis document or manually curated workflow.

## Deletion decision table

| Script | Role | Inputs | Exclusive outputs | Shared outputs | Active downstream consumers | Notebook/doc references | Safe to delete script now? | Safe to delete exclusive outputs now? | Reason | Required action before deletion |
|---|---|---|---|---|---|---|---|---|---|---|
| `scripts/01_descriptive/02b_plot_licensed_driver_resources_boxline.py` | Optional standalone visualization; calculates weighted quartiles/whiskers and validates saved means/CIs, but fits no accepted model | Household backbone; `licensed_drivers_weighted_summary_household_type.csv`; `driver_car_ratio_exact_weighted_summary_household_type.csv` | `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_boxline.png` and `.pdf` (both currently absent) | None | None found | No notebook reference; cleanup docs mention the script and protected same-named backup PNG; no in-repository thesis/presentation artifact exists | **CONDITIONAL** | **CONDITIONAL** | Reproducible from untouched core producer outputs, but external/manual figure use cannot be disproved | Search the actual thesis/manuscript outside this repository for the exact stem; approve only the active paths, never the backup PNG |
| `scripts/01_descriptive/02c_plot_licensed_driver_resources_discrete.py` | Optional standalone visualization; derives H_GEW-weighted discrete shares and validates saved means/CIs | Household backbone; same two core-producer summaries | `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_discrete.png` and `.pdf` (both currently absent) | None | None found | No notebook reference; cleanup docs mention the script and protected same-named backup PNG/PDF; no in-repository thesis/presentation artifact exists | **CONDITIONAL** | **CONDITIONAL** | Same as `02b`; its presentation variant is distinct but not consumed programmatically | Search the actual thesis/manuscript for the exact stem; approve only active paths, never either backup figure |
| `scripts/01_descriptive/04_annual_vehicle_mileage.py` | Core substantive analytical producer: weighted descriptives, household-clustered GEE, sensitivities, QA, publication tables, and result figures | `cars_selected_raw.csv`; Theme-1 household backbone; saved summary/model tables for its figure-only modes | Exact inventory below; every write except the three shared CSVs | `07_model_overall.csv`, `13_exact_two_sensitivity_overall.csv`, `20_log1p_sensitivity_comparison.csv` | `04b_annual_vehicle_mileage_composition_diagnostic.py` consumes the three shared CSVs | No notebook consumer; cleanup mappings/docs identify outputs as accepted/possible thesis artifacts | **NO** | **NO** | It remains the only reproducible producer of the annual-mileage analysis and has an active diagnostic consumer | Retain script and outputs; no deletion action is justified |
| `scripts/01_descriptive/04b_annual_vehicle_mileage_composition_diagnostic.py` | Terminal diagnostic extension: common-sample household-type/powertrain composition and additive nested GEE checks | `cars_selected_raw.csv`; household backbone; the three shared `04` outputs | `.../03_annual_vehicle_mileage/25_composition_diagnostic_hh_type.csv`; `26_composition_diagnostic_powertrain.csv`; `27_composition_diagnostic_nested_models.csv`; `28_composition_diagnostic_QA.csv`; `annual_vehicle_mileage_composition_diagnostic.md` (all present) | None among its writes | None found | No notebook consumer; cleanup docs record possible thesis/manual use | **CONDITIONAL** | **CONDITIONAL** | It is derived from `04` but analytically distinct, and its report may support thesis interpretation | Review its conclusions against the actual thesis methods/results; authorize the five exact outputs individually if the diagnostic is retired |

All abbreviated `.../03_annual_vehicle_mileage/` entries in the final row begin
with `outputs/eda/statistical_tests/theme1_single_vs_multicar/`.

## Exact candidate output files

The following paths are the complete set that may be considered in a later,
explicitly authorized deletion pass. Their status remains **CONDITIONAL** now:

- `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_boxline.png`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_boxline.pdf`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_discrete.png`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/01_licensed_driver_resources/licensed_driver_resources_by_household_type_discrete.pdf`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/03_annual_vehicle_mileage/25_composition_diagnostic_hh_type.csv`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/03_annual_vehicle_mileage/26_composition_diagnostic_powertrain.csv`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/03_annual_vehicle_mileage/27_composition_diagnostic_nested_models.csv`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/03_annual_vehicle_mileage/28_composition_diagnostic_QA.csv`
- `outputs/eda/statistical_tests/theme1_single_vs_multicar/03_annual_vehicle_mileage/annual_vehicle_mileage_composition_diagnostic.md`

The first four active paths are absent. Same-named files below the licensed-
driver backup are deliberately not in this list. The final five paths exist.
No output of the retained main annual-mileage producer is approved for deletion.

### Licensed-driver plot helpers

The active directory does not currently contain either helper's PNG/PDF pair.
The protected `backup/` directory does contain historical same-named figures.
Those files have unknown historical/manual provenance and are **not** included
in any deletion recommendation.

Both helpers depend on summaries produced by the untouched active producer
`scripts/01_descriptive/02_licensed_driver_resources.py`:

- `licensed_drivers_weighted_summary_household_type.csv`
- `driver_car_ratio_exact_weighted_summary_household_type.csv`

### Main annual-mileage producer (`04`)

Declared CSV/metadata outputs:

- `00_A_JAHRESFL_observed_QA.csv`
- `01_weighted_summary_overall.csv`
- `02_weighted_distribution_overall.csv`
- `03_weighted_summary_spatial.csv`
- `04_weighted_distribution_spatial.csv`
- `05_weighted_summary_household_type.csv`
- `06_weighted_distribution_household_type.csv`
- `07_model_overall.csv`
- `08_model_spatial.csv`
- `09_spatial_multicar_contrasts.csv`
- `10_model_household_type.csv`
- `11_household_type_multicar_contrasts.csv`
- `12_joint_wald_tests.csv`
- `13_exact_two_sensitivity_overall.csv`
- `14_exact_two_sensitivity_spatial.csv`
- `15_exact_two_sensitivity_household_type.csv`
- `16_exact_two_sensitivity_comparison.csv`
- `17_log1p_sensitivity_overall.csv`
- `18_log1p_sensitivity_spatial.csv`
- `19_log1p_sensitivity_household_type.csv`
- `20_log1p_sensitivity_comparison.csv`
- `21_car_count_detail_summary.csv`
- `22_sample_QA.csv`
- `23_weight_QA.csv`
- `24_fleet_record_QA.csv`
- `25_optional_exact_two_complete_mileage.csv` (conditional write only when the
  incomplete-coverage threshold is exceeded; not currently present)
- `publication_table_overall.csv`
- `publication_table_spatial.csv`
- `publication_table_household_type.csv`
- `analysis_metadata.csv`

Declared figures (each `.png` and `.pdf`):

- `annual_vehicle_mileage_distribution_overall`
- `annual_vehicle_mileage_spatial`
- `annual_vehicle_mileage_household_type`
- `annual_vehicle_mileage_mean_ci_overall`
- `annual_vehicle_mileage_mean_ci_household_type`
- `annual_vehicle_mileage_mean_ci_spatial`
- `annual_vehicle_mileage_box_jitter_mean_overall` (`--figures-only`)
- `annual_vehicle_mileage_box_jitter_mean_household_type` (`--figures-only`)
- `annual_vehicle_mileage_box_jitter_mean_spatial` (`--figures-only`)

The directory currently contains the accepted CSVs through `24`, the three
publication tables, metadata, the PNG variants, and the three mean/CI PDFs.
Absence of some other declared PDF variants is not evidence that the producer
or its existing outputs are obsolete.

### Composition diagnostic (`04b`)

The script reads but hash-protects every existing file in the annual-mileage
output directory other than its own five outputs. It constructs one common
vehicle sample, tabulates household-type and powertrain composition, fits
additive diagnostic GEE specifications, compares them with the completed full
model, writes four QA/result CSVs, and writes a narrative Markdown diagnostic.
This function is complementary to `04`, not duplicated by it.

## Items explicitly not approved for deletion

- `scripts/01_descriptive/02_licensed_driver_resources.py` and every summary
  CSV it owns.
- `scripts/01_descriptive/04_annual_vehicle_mileage.py` and every output in its
  declared inventory.
- The four audited scripts and all their outputs during this pass, including
  the three scripts with a conditional future recommendation.
- The licensed-driver `backup/` directory and every file below it.
- Any of the 13 accepted analytical files protected by the hash verification.
- Any file whose use in the external thesis/manuscript remains unknown.

## Script integrity

The four audited scripts were not edited. Their SHA-256 hashes at the end of
the pass are expected to remain:

| Script | SHA-256 |
|---|---|
| `02b_plot_licensed_driver_resources_boxline.py` | `414eb27d9ad1d28f3d02304838817081d4d8688491363ca35ac9f5dd41752f70` |
| `02c_plot_licensed_driver_resources_discrete.py` | `ba46051370e9f4af7cc3a38e0dcaf1a3ebe28e629b08e74d0064213c1cee11d8` |
| `04_annual_vehicle_mileage.py` | `0772b48e5ae268eabc14d40e5eaf1a036a3cadd736553533d5b0566de6050248` |
| `04b_annual_vehicle_mileage_composition_diagnostic.py` | `f983c7cee2261005f5252b87140e2d30da9f7725307fd385fcc3012f9ebc30c8` |

## Verification

All checks were read-only with respect to analytical inputs and outputs.

| Check | Result |
|---|---|
| Existing path smoke test | **PASS** — 37 active Python modules imported; 291 declared `Path` constants resolved; 13 accepted files found; zero declared paths outside the repository |
| Clean locked runtime environment | **PASS** — `uv run --isolated --frozen` created a new environment from `uv.lock` (19 packages) and the full path smoke test passed |
| Clean locked notebook environment | **PASS** — the isolated `notebooks` dependency group recreated (45 packages); IPython 9.14.1 and ipykernel 7.3.0 imported |
| Lock consistency | **PASS** — `uv lock --check` resolved 51 locked packages without changing the lock |
| Python compile | **PASS** — 38 Python files compiled in memory |
| Notebook parse/compile | **PASS** — 4 notebooks parsed as JSON and 36 code cells compiled in memory |
| Accepted-file integrity | **PASS** — all 13 previously recorded SHA-256 hashes match exactly |
| Candidate-script integrity | **PASS** — all four hashes in the script-integrity table match exactly |
| Intended path containment | **PASS** — no declared active path resolves outside the repository |
| Git whitespace check | **PASS** — `git diff --check` |
| Local environment boundary | **PASS** — `.venv` resolves below the repository root, is ignored, and contains no tracked file |
| Analytical-output visibility | **PASS** — canonical `outputs/` files are not blanket-ignored; 155 existing local output files are now visible as untracked, and none was staged or committed |
| Analytical output regeneration | **NONE** — only import, path, hash, in-memory compile, and isolated environment checks ran; no analytical `main()` or notebook cell was executed |

The four audited scripts, all candidate outputs, and the licensed-driver backup
directory were not modified.

## Recommended next action

1. Keep `04_annual_vehicle_mileage.py` and all of its accepted outputs.
2. Before deleting either licensed-driver visualization helper, search the
   actual thesis manuscript/project outside this repository for the exact
   figure names and decide whether those presentation variants remain useful.
3. Review the `04b` composition findings against the thesis methods/results;
   retain them unless the diagnostic is explicitly judged out of scope.
4. If a later pass authorizes deletion, delete only the individually approved
   script and the exact active outputs listed above; never infer deletion of a
   backup or similarly named file.
5. Start Stage-05 MNL in a separate pass and add its modeling dependency only
   when the implementation choice is made.

## Ready to start Stage-05 MNL?

**YES.** The repository now has a reproducible locked runtime, an optional
locked notebook environment, a documented licensed-data boundary, and a
completed conservative deletion audit. This verdict authorizes starting a
separate Stage-05 design/implementation pass; it does **not** authorize any of
the conditional deletions above. The chosen MNL library should be added only
when Stage-05 implementation establishes it as an active dependency.
