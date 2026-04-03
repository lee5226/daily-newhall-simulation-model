from pathlib import Path

# Project root: daily-newhall-model/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_REFERENCE_DIR = PROJECT_ROOT / "data_reference"

# Raw / source data locations
USCRN_RAW_DIR = PROJECT_ROOT / "uscrn_daily01"
GSSURGO_GDB = PROJECT_ROOT / "gSSURGO_CONUS.gdb"

# Reference metadata
USCRN_METADATA_TXT = DATA_REFERENCE_DIR / "crn-stations.txt"