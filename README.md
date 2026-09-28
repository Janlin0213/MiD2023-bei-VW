# MiD thesis analytical pipeline

This repository contains the source code and project-local metadata for the
master's-thesis analytical workflow. The licensed MiD raw data and generated
analytical data remain separate, local inputs; they are not fetched by the
environment setup and are intentionally excluded from Git.

## Clone and recreate the Python environment

Install [uv](https://docs.astral.sh/uv/), then clone the repository and run:

```powershell
git clone <repository-url>
cd <repository-directory>
uv sync
```

The project is pinned to Python 3.12.13. `uv sync` creates or updates the
repository-local `.venv` from `pyproject.toml` and `uv.lock`; `.venv` is local
and ignored by Git. To run the Jupyter notebooks, include their optional tools:

```powershell
uv sync --group notebooks
```

## Data boundary

MiD source files are licensed inputs and are not supplied by this repository.
Obtain authorized data separately and provide the repository-relative private
directories described in [docs/storage_layout.md](docs/storage_layout.md).
Cloning the Git repository alone does not provide `data_raw/`,
`data_processed/`, or `outputs/` content.

After the authorized private data is available, run the read-only path check:

```powershell
uv run python scripts/00_path_smoke_test.py
```

The smoke test imports active modules, validates centralized paths, checks the
accepted analytical files, and verifies that declared paths stay inside the
repository. It does not regenerate analytical outputs.

## Workflow

Run scripts from the repository root and in analytical order:

1. `scripts/00_data_preparation/`
2. `scripts/01_descriptive/`
3. `scripts/02_choice_reconstruction/`
4. `scripts/03_model_input/`
5. `scripts/04_screening/`
6. `scripts/05_mnl/` when Stage-05 modeling is implemented

Each stage expects its documented private inputs to exist. Do not assume that
running a later stage will download or recreate missing licensed source data.
