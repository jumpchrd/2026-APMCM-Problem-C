"""Shared configuration for 2026 APMCM Problem C modeling scripts."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def resolve_data_dir() -> Path:
    """Resolve the Problem C CSV directory from env, local data/, or workspace."""
    env_dir = os.environ.get("APMCM_C_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    local_data = REPO_ROOT / "data"
    if (local_data / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv").exists():
        return local_data

    workspace_data = (
        REPO_ROOT.parent
        / "2026 APMCM Problems B&C&D"
        / "2026 APMCM Problems B&C&D"
        / "2026 APMCM Problems B&C&D"
        / "2026 APMCM Problems"
        / "2026 APMCM Problem C"
        / "date"
    )
    if workspace_data.exists():
        return workspace_data

    raise FileNotFoundError(
        "Could not locate Problem C CSV files. Place them in ./data or set APMCM_C_DATA_DIR."
    )
