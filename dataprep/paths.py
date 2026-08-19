"""Default read/write paths for pipeline data (override with BIODIVERSITY_DATA_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BIODIVERSITY_DATA_ROOT", "/mnt/f/biodiversity"))

EBIRD_ROOT = DATA_ROOT / "ebird"
EBIRD_PARQUET = DATA_ROOT / "ebirdpolars"
GBIF_ROOT = DATA_ROOT / "gbif"
PARAM_CSV_DIR = DATA_ROOT / "param_csvs"
INFERENCE_RASTERS = DATA_ROOT / "rasters" / "inference"
MODEL_PREP = DATA_ROOT / "modelprep"
LOGS_DIR = DATA_ROOT / "logs"

# MaxEnt writes to DATA_ROOT / "ppp_paramsoutput" / <species>
MODEL_OUTPUT_ROOT = DATA_ROOT

# BioAPI deploy pack (scp to server)
DEPLOY_API_ROOT = Path(os.environ.get("DEPLOY_API_ROOT", "/mnt/f/deployapi"))

# Optional national EBD monolith (manifest-filtered extract; not per-species custom downloads)
NATIONAL_EBD_TSV = Path(
    os.environ.get("EBIRD_NATIONAL_TSV", "/mnt/e/backupfrompc/ebd_US_relSep-2025.txt")
)

# Aliases used by orchestration scripts
DEFAULT_EBD_ROOT = EBIRD_ROOT
DEFAULT_EBIRD_PARQUET = EBIRD_PARQUET
DEFAULT_PARAM_CSV = PARAM_CSV_DIR
