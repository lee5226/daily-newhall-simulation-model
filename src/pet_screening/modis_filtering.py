import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
SCREEN_DIR = OUTPUTS_DIR / str(YEAR) / "pet_screening"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = SCREEN_DIR / "MODIS_dailyPET.csv"
OUT_CSV = SCREEN_DIR / "MODIS_ETPET_STRICT_8day.csv"

FILL_VALUES = {32762.0, 32766.0}
VALUE_COLS = ["MOD16A2GF_061_ET_500m", "MOD16A2GF_061_PET_500m"]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    for col in VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin(FILL_VALUES))

    mask = (
        (df["MOD16A2GF_061_ET_QC_500m_MODLAND"] == "0b0")
        & (df["MOD16A2GF_061_ET_QC_500m_DeadDetector"] == "0b0")
        & (df["MOD16A2GF_061_ET_QC_500m_CloudState"] == "0b00")
        & (df["MOD16A2GF_061_ET_QC_500m_SCF_QC"].isin(["0b000", "0b001"]))
    )

    out = df.loc[
        mask,
        [
            "ID",
            "Latitude",
            "Longitude",
            "Date",
            "MODIS_Tile",
            "MOD16A2GF_061_ET_500m",
            "MOD16A2GF_061_PET_500m",
        ],
    ].rename(
        columns={
            "MOD16A2GF_061_ET_500m": "ET_mm8d",
            "MOD16A2GF_061_PET_500m": "PET_mm8d",
        }
    )

    dates = pd.to_datetime(out["Date"])
    out["key8"] = (
        dates.dt.year.astype(str)
        + "-"
        + (((dates.dt.dayofyear - 1) // 8 + 1).astype(str).str.zfill(2))
    )

    out.to_csv(OUT_CSV, index=False)
    print(f"{len(df)} -> {len(out)} rows saved to {OUT_CSV.name}")


if __name__ == "__main__":
    main()