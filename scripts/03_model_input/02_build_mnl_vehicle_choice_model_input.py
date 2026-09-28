"""Stage entry point for accepted MNL model-input construction."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.model_input.vehicle_choice import main  # noqa: E402


if __name__ == "__main__":
    main()
