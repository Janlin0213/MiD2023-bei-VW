"""Compatibility imports for the packaged decision-timing sensitivity helpers."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.sensitivity.decision_timing import *  # noqa: E402,F403
