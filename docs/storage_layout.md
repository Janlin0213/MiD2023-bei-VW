# Git repository and private analytical storage

This project uses one logical repository layout with two different ownership
layers. Paths in active code are repository-relative; no personal Windows or
VW Drive path is hard-coded.

## Git-controlled layer

Git version-controls the shareable and reproducible project definition:

```text
config/
docs/
notebooks/
scripts/
src/
.gitignore
.python-version
pyproject.toml
uv.lock
README.md
```

Safe notebooks may be tracked, but their code must not embed restricted data or
personal absolute paths. Git does not provide licensed MiD data, record-level
derivatives, routine generated outputs, or private model artifacts.

## Private authorized-storage layer

Users with authorized MiD access must provide these logical directories at the
repository root:

```text
data_raw/        Licensed source files
data_processed/  Selected extracts, intermediates, and accepted analytical data
outputs/         Generated tables, figures, diagnostics, and model artifacts
```

Some workflows also use local reference material under `data/reference/`.
These directories are ignored by Git. Their expected filenames and subfolders
are defined centrally in `src/thesis_pipeline/paths.py` and by the producing
scripts. They may be ordinary local folders, approved mounts, or synchronized
folders, provided they appear at these repository-relative locations.

If the checkout itself is inside an approved VW-synced location, the ignored
data and output directories may be backed up or synchronized by that service.
That does not change the responsibility boundary: Git manages source and
version history; VW Drive or other approved private storage preserves
restricted data and analytical artifacts.

## Environments and reproducibility

The local `.venv` is disposable and must not be copied or committed. Recreate
it from `pyproject.toml` and `uv.lock` with:

```powershell
uv sync
```

Generated outputs should be regenerated from versioned code when feasible.
Important thesis snapshots may additionally be archived in approved private
storage with a date, code revision, or other version marker so their provenance
can be reconstructed without placing the artifacts in Git.
