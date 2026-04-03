import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR, USCRN_RAW_DIR


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)
INPUT_ROOT = USCRN_RAW_DIR
STATION_CSV = BASE / f"aws_by_station_final_{YEAR}.csv"

OUTPUT_DIR = BASE / "daily_climate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPINUP_OUTPUT_DIR = BASE / "daily_climate_spinup"
SPINUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = [
    "WBANNO",
    "LST_DATE",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "T_DAILY_MAX",
    "T_DAILY_MIN",
    "T_DAILY_MEAN",
    "T_DAILY_AVG",
    "P_DAILY_CALC",
    "SOLARAD_DAILY",
    "SUR_TEMP_DAILY_TYPE",
    "SUR_TEMP_DAILY_MAX",
    "SUR_TEMP_DAILY_MIN",
    "SUR_TEMP_DAILY_AVG",
    "RH_DAILY_MAX",
    "RH_DAILY_MIN",
    "RH_DAILY_AVG",
    "SOIL_MOISTURE_5_DAILY",
    "SOIL_MOISTURE_10_DAILY",
    "SOIL_MOISTURE_20_DAILY",
    "SOIL_MOISTURE_50_DAILY",
    "SOIL_MOISTURE_100_DAILY",
    "SOIL_TEMP_5_DAILY",
    "SOIL_TEMP_10_DAILY",
    "SOIL_TEMP_20_DAILY",
    "SOIL_TEMP_50_DAILY",
    "SOIL_TEMP_100_DAILY",
]

NUM_COLS = ["precipitation", "tmin", "tmax", "tavg", "solar"]


def extract_one_year(station_name: str, station_id: str, year: int, out_kind: str = "main") -> bool:
    """
    Extract one year of daily climate data for a station.

    out_kind:
        - "spinup_y1": previous year climate
        - "main": target year climate
    """
    input_dir = INPUT_ROOT / str(year)
    pattern = f"CRND0103-{year}-*{station_name}.txt"
    matches = list(input_dir.glob(pattern)) if input_dir.exists() else []

    if not matches:
        tag = f" {out_kind}" if out_kind != "main" else ""
        print(f"Missing raw file: {station_name} ({year}){tag}")
        return False

    raw_path = matches[0]

    try:
        df_raw = pd.read_csv(
            raw_path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=HEADERS,
            dtype=str,
            engine="python",
        )

        df = df_raw[
            [
                "WBANNO",
                "LST_DATE",
                "T_DAILY_MIN",
                "T_DAILY_MAX",
                "T_DAILY_MEAN",
                "P_DAILY_CALC",
                "SOLARAD_DAILY",
            ]
        ].copy()

        df["station"] = station_name
        df["station_id"] = station_id
        df["date"] = pd.to_datetime(df["LST_DATE"], format="%Y%m%d", errors="coerce")

        df = df.rename(
            columns={
                "T_DAILY_MIN": "tmin",
                "T_DAILY_MAX": "tmax",
                "T_DAILY_MEAN": "tavg",
                "P_DAILY_CALC": "precipitation",
                "SOLARAD_DAILY": "solar",
            }
        )[
            [
                "station",
                "station_id",
                "date",
                "precipitation",
                "tmin",
                "tmax",
                "tavg",
                "solar",
            ]
        ]

        df = df.dropna(subset=["date", "tavg", "precipitation"])
        df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

        sid5 = str(station_id).zfill(5)
        if out_kind == "main":
            out_path = OUTPUT_DIR / f"daily_climate_{YEAR}_{sid5}.csv"
        elif out_kind == "spinup_y1":
            out_path = SPINUP_OUTPUT_DIR / f"daily_climate_{YEAR}_{sid5}_spinup_y1.csv"
        else:
            raise ValueError(f"Unknown out_kind: {out_kind}")

        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path.name}")
        return True

    except Exception as e:
        print(f"Failed: {station_name} ({year}) {out_kind}: {e}")
        return False


def main() -> None:
    station_df = pd.read_csv(STATION_CSV)
    station_df["station_id"] = station_df["station_id"].astype(str).str.zfill(5)

    for _, row in station_df.iterrows():
        station_name = row["station"]
        station_id = row["station_id"]
        real_year = int(row["year"])

        extract_one_year(station_name, station_id, real_year - 1, out_kind="spinup_y1")
        extract_one_year(station_name, station_id, real_year, out_kind="main")


if __name__ == "__main__":
    main()