# Path and dependency report

Audit date: 2026-09-28  
Status: updated after Phase 2 stabilization

## Phase 2B status update

Phase 2B resolves the remaining active relocation dependencies:

- The licensed-driver plotters read only the active output directory. No active Python code references `backup/`. The active producer, `02_licensed_driver_resources.py`, now generates the previously missing `licensed_drivers_weighted_summary_household_type.csv` with its existing sample and weighted-summary functions; an in-memory check reproduced the preserved backup CSV byte-for-byte. The backup remains preserved but is no longer an active dependency.
- The exact-filename sensitivity loaders are removed. Accepted Phase-4 and model-input implementations live in `src/thesis_pipeline/model_input/`, their original script paths are compatibility entry points, and baseline/sensitivity consumers share the same packaged function objects.
- All four notebooks bootstrap the repository through `THESIS_REPO_ROOT` or an upward `.git` search and then import paths from `src.thesis_pipeline.paths`. `CodeBook_helper.ipynb` and `small_check.ipynb` now use the current selected-raw/reconstruction directories.
- `check_LinkageOutput.ipynb` remains a documented legacy diagnostic because `data_processed/02_mnl_estimation_funnel.csv` and `data_processed/02_trip_vehicle_wide.csv` are genuinely absent. It is not an active pipeline dependency.
- The household-type CLI's in-place write to `hh_selected_raw.csv` remains an accepted documented risk and does not block relocation.

Current pre-move decision: **READY FOR PHASE 3 — YES**. Physical restructuring was not performed in Phase 2B.

## Phase 2 status update

The fixed-depth root dependency identified in the original audit is resolved for all active Python source files. `src/thesis_pipeline/paths.py` now resolves the root via `THESIS_REPO_ROOT` or an upward `.git` marker walk, and all active scripts import named current-layout paths from that registry. A repository-wide search finds no remaining `Path(__file__).resolve().parents[n]` root calculation.

The sibling imports from `decision_timing_sensitivity_utils.py` and the exact-filename dynamic loaders are resolved. Scripts 06b, 07b, and 10 use packaged modules under `src.thesis_pipeline`; the original script paths remain relocation-safe compatibility entry points.

Previously identified output-parent gaps are resolved for Phase 2, Phase 3, bipartite processed outputs, association screening, annual-mileage composition diagnostics, and the employment diagnostic. Notebook CWD paths and the licensed-driver backup dependency are resolved; only the explicitly documented legacy linkage notebook inputs remain absent. See `docs/repo_cleanup/phase2_stabilization_report.md` for verification evidence.

## Executive path-risk assessment

Every executable Python file avoids a literal absolute Windows input/output path. Active Python and notebook root discovery is independent of script nesting depth. Physical moves remain deferred only because this pass intentionally stops before Phase 3, not because of an active path/import/backup blocker.

Persisted output cells in `notebooks/RQ1.ipynb` contain absolute Windows paths from the machine on which the notebook was run. Those are notebook display history, not active path constants, but they leak machine-specific context and should be cleared or regenerated only after paths are fixed. No active notebook code cell contains a literal drive-letter path.

The only active `../` path computation is `PROJECT_ROOT = Path('..').resolve()` in `notebooks/RQ1.ipynb`; it depends on kernel CWD. The text `../src/build_household_type.py` also appears in an error message.

## Implemented centralized path design

Phase 2 created `src/thesis_pipeline/paths.py` and migrated active scripts to its named paths. The module locates the repository root from the optional `THESIS_REPO_ROOT` override or by walking upward to `.git`; it does not infer the root from a script's directory depth.

The implemented approach follows this interface pattern:

```python
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    reference: Path
    outputs: Path

def find_project_root(start: Path | None = None) -> Path:
    override = os.environ.get("THESIS_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Repository root not found")
```

Active scripts now default to the central registry, and identified writer-parent gaps create their directories before writing. Optional per-script CLI path overrides and a full producer-by-producer read/write dry run remain possible future hardening; the Phase 2 smoke test resolves declared paths without executing analysis.

## Original Python-file dependency inventory

The table below records the Phase-1 baseline and is retained as historical evidence. Statements about fixed `parents[n]` root calculations and the listed writer-directory gaps have been superseded by the Phase-2 status update above. “Move risk” otherwise remains relevant until the remaining import/notebook dependencies are addressed.

| Python file | Reads | Writes | Local imports / downstream consumers | Path and mkdir behavior | Move risk |
|---|---|---|---|---|---|
| `metadata/selected raw list.py` | None | None | Dynamically loaded by `scripts/01_extract_selected_raw.py` | No path logic | Filename contains spaces; move requires updating dynamic path |
| `metadata/mnl_attribute_candidates.py` | None | None | Dynamically loaded by `scripts/07_build_mnl_vehicle_choice_model_input.py` | No path logic | Move requires updating registry path |
| `scripts/01_extract_selected_raw.py` | Four raw CSVs: Haushalte, Personen, Wege, Autos; selected-column registry | Four `data_processed/selected_raw/*.csv` files | Dynamic local module load; outputs feed nearly all stages | `ROOT=parents[1]`; creates `OUT_DIR` with parents | Root and registry path break after nesting |
| `scripts/02_phase1_diagnostics.py` | Phase-1 enriched trips; selected cars | Console only | No local imports; diagnostic terminal consumer | `ROOT=parents[1]`; no mkdir | Root breaks after nesting |
| `scripts/02_phase1_home_chain.py` | Selected trips/cars; conditionally reads prior enriched output for stability checks | Phase-1 enriched trips and QA | Outputs feed Phases 2–4, bipartite, model input, sensitivity | `ROOT=parents[1]`; creates parent of trip output | Root breaks; QA shares created parent safely |
| `scripts/03_phase2_vehicle_states_v2.py` | Phase-1 trips; selected cars | Phase-2 events, occasions, QA | Events feed bipartite; occasions/QA feed strict baseline | `ROOT=parents[1]`; **does not create Phase-2 directory** | Root breaks; fresh target directory causes write failure |
| `scripts/04_phase3_strict_baseline.py` | Phase-2 occasions/QA; Phase-1 trips/QA; cars | Classified occasions, funnel, strict wide base, availability QA | Feeds sensitivity Phase 3, tour features, legacy check workbook | `ROOT=parents[1]`; **does not create Phase-3 directory** | Root breaks; fresh target directory causes write failure |
| `scripts/04b_phase3_decision_timing_sensitivity.py` | Strict classified occasions, strict wide base, Phase-1 trips | Sensitivity rich occasions, wide base, QA | Feeds sensitivity tour features and sample comparison | `ROOT=parents[1]`; creates output dir with parents | Root breaks after nesting |
| `scripts/05_driver_vehicle_bipartite_patterns.py` | Phase-2 events, Phase-1 trips, selected cars/persons; rereads own tables for plots | Four processed bipartite tables, three output distributions, five figures | Terminal descriptive branch | `ROOT=parents[1]`; creates only `outputs/eda/bipartite`, **not** `data_processed/eda/bipartite` | Root breaks; fresh processed-data directory causes write failure |
| `scripts/06_build_home_based_tour_features.py` | Phase-1 trips; strict wide base | Membership, tour features, wide tour features, QA | Wide/tours feed model-input and sensitivity Phase 4 | `ROOT=parents[1]`; creates Phase-4 dir with parents | Root breaks after nesting |
| `scripts/06b_build_decision_timing_sensitivity_tour_features.py` | Sensitivity wide base; baseline wide tour features; baseline tours; Phase-1 trips | Sensitivity wide tour features | Imports sibling utility; dynamically loads exact accepted script filename; feeds sensitivity model input | Utility supplies root; creates output dir | Sibling import and `ROOT/scripts/<filename>` loader both break |
| `scripts/07_build_mnl_vehicle_choice_model_input.py` | Wide tour backbone, Phase-1 trips, selected household/person/car files, candidate registry, and prior model input if present | Canonical model input, manifest, QA | Imports `src.build_household_type`; outputs feed screening, diagnostics, sensitivity | `ROOT=parents[1]`; injects root into `sys.path`; creates output dir | Root, `sys.path`, local import, registry path break |
| `scripts/07b_build_mnl_vehicle_choice_model_input_decision_timing_sensitivity.py` | Sensitivity wide tour features; all baseline inputs resolved by script 07; baseline model and manifest | Sensitivity model, byte-copied manifest, QA | Imports sibling utility; dynamically loads script 07; feeds comparison | Utility root; creates output dir | Sibling import and dynamic exact filename break |
| `scripts/08_hierarchical_mixed_association_screening.py` | Canonical model input | Pairwise/cross-level tables, QA, three heatmaps | Pairwise table consumed by script 09 | `ROOT=parents[1]`; **does not create model directory** | Root breaks; fresh directory causes write failure |
| `scripts/09_employment_share_by_age_diagnostic.py` | Canonical model input, pairwise screening table, codebook XLSX | Employment table and figure | Terminal screening diagnostic | `ROOT=parents[1]`; creates only final child with `parents=False` | Root breaks; assumes parent model directory exists |
| `scripts/10_decision_timing_sensitivity_sample_comparison.py` | Baseline/sensitivity model inputs, sensitivity tour features, sensitivity rich occasions | Full and key comparison CSVs | Imports sibling utility; terminal sensitivity diagnostic | Utility root; creates output dir with parents | Sibling import breaks after separation |
| `scripts/decision_timing_sensitivity_utils.py` | Arbitrary CSVs via helper; fingerprints phases 1–4/model inputs; dynamically loads baseline scripts | Conditional weight-precision diagnostic CSV | Imported by scripts 06b, 07b, and 10 | `ROOT=parents[1]`; loader requires `ROOT/scripts/<exact filename>`; conditional writer creates parents | Must become package module; dynamic loader paths break after any move |
| `scripts/eda/statistical_tests/01_build_theme1_household_backbone.py` | Selected household/person CSVs | Household backbone and backbone QA | Backbone feeds all Theme-1 analyses | `ROOT=parents[3]`; creates output dir with parents | New `scripts/01_descriptive` depth requires different parent index |
| `scripts/eda/statistical_tests/02_licensed_driver_resources.py` | Household backbone | Ratio summaries/models/contrasts/Wald tables, sample QA, three figures | Some summaries are expected by plot scripts | `ROOT=parents[3]`; creates output dir | Root breaks after move |
| `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_boxline.py` | Household backbone plus two summary CSVs | Boxline PNG/PDF | Terminal figure producer | `ROOT=parents[3]`; creates output dir | Root breaks; currently required summary files exist only in `backup/`, not the configured main directory |
| `scripts/eda/statistical_tests/03_plot_licensed_driver_resources_discrete.py` | Household backbone plus two summary CSVs | Discrete PNG/PDF | Terminal figure producer | `ROOT=parents[3]`; creates output dir | Same root and missing-configured-input risk as boxline script |
| `scripts/eda/statistical_tests/03_habitual_car_use_orientation.py` | Selected persons; household backbone; rereads own output distributions | Tables, QA, predicted probabilities, publication tables, PNG/PDF figures | Terminal descriptive branch | `ROOT=parents[3]`; creates output dir | Root breaks after move |
| `scripts/eda/statistical_tests/04_annual_vehicle_mileage.py` | Selected cars; household backbone; rereads own summaries/model outputs for figures | Main mileage tables, QA, sensitivity tables, publication tables, figures | Three outputs consumed by composition diagnostic | `ROOT=parents[3]`; creates output dir | Root breaks after move |
| `scripts/eda/statistical_tests/04b_annual_vehicle_mileage_composition_diagnostic.py` | Selected cars, household backbone, three mileage result CSVs | Four diagnostic CSVs and one Markdown report | Terminal descriptive diagnostic | `ROOT=parents[3]`; **does not create output directory** | Root breaks; must run after main mileage script/directory exists |
| `scripts/eda/statistical_tests/05_reference_day_mobility_intensity.py` | Selected persons/trips; household backbone; rereads three own CSVs for validation | Summary, contrasts, two QA CSVs, metadata, two PNGs | Terminal descriptive branch | `ROOT=parents[3]`; creates output dir | Root breaks after move |
| `src/build_household_type.py` | Selected household/person CSVs | **Overwrites the household CSV in place** when run as CLI | `build_household_type` imported by model-input script; notebook tells user to run CLI | `ROOT=parents[1]`; CLI arguments can override paths | Package move breaks defaults/import; in-place mutation should be separated from extraction |

## Notebook path inventory

| Notebook | Reads | Writes | Risks |
|---|---|---|---|
| `notebooks/CodeBook_helper.ipynb` | Codebook XLSX and four raw files; configured selected-raw paths | None | CWD heuristic is acceptable only from repo root or `notebooks`; selected-raw paths are stale and missing the `selected_raw/` directory |
| `notebooks/RQ1.ipynb` | Selected household CSV (other selected datasets are declared but not loaded by default) | Nine descriptive CSV/PNG artifacts | `Path('..').resolve()` breaks when launched from repo root or another CWD; `mkdir(exist_ok=True)` assumes `outputs/` exists; persisted outputs contain absolute Windows paths |
| `notebooks/check_LinkageOutput.ipynb` | Two missing legacy processed CSVs | None | Entire input contract is stale; classify as legacy diagnostic pending author review |
| `notebooks/small_check.ipynb` | Raw Wege/Autos and missing legacy processed paths | `outputs/mnl_check.xlsx` | Generated workbook is proven equal to current strict baseline, but configured source paths are stale |

## Paths requiring updates before restructuring

1. **Resolved:** fixed-depth Python root calculations were replaced by the shared registry.
2. **Resolved:** notebook root/CWD heuristics were replaced by the shared environment/marker discovery pattern and central named paths.
3. **Resolved for relocation:** sensitivity sibling imports and accepted helper imports are package imports; `src.build_household_type` remains stable and unmoved to avoid touching its analytical contract.
4. **Resolved:** the two accepted-script exact-filename dynamic loads were replaced by normal package imports.
5. **Deferred to structural Phase 3:** update selected-column and candidate-registry locations only if `metadata/` is reorganized.
6. **Deferred to structural Phase 3:** raw paths remain `data_raw/` by design in this pass.
7. **Deferred to structural Phase 3:** outputs remain in their current physical locations.
8. **Resolved for current files:** selected-raw and Phase-3 notebook paths use the registry; the linkage notebook's genuinely missing legacy inputs remain documented and unrecreated.
9. **Resolved:** identified Python writer-parent gaps now create their directories.
10. **Accepted documented risk:** the household-type CLI mutates `hh_selected_raw.csv` in place; this is explicit and does not block relocation.
11. **Open:** preserve compatibility aliases or update thesis links before moving figures/tables; external references remain unknown.

## Pre-move verification gates

- A path-only test imports all modules and asserts that every declared input exists.
- A dry run prints the complete read/write set and rejects writes outside the repository output root.
- A clean-directory test verifies each producer creates its own output parents.
- Raw inputs and accepted canonical outputs have recorded SHA-256 hashes.
- Moved files retain hashes; regenerated files are compared by schema, row count, keys, and analytical invariants.
- Notebooks are tested from both repository root and their own directory.
- Stage 05 remains empty and documented until actual MNL code/results are supplied; do not infer an estimator from `mnl_check.xlsx`.
