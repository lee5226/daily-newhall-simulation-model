"""Extract raw daily climate for the single-year P/T-QC station set.

This script intentionally does not fill precipitation or temperature gaps.
It writes one complete calendar-year file for the selected evaluation year and
one for the preceding warm-up year. Missing/invalid climate values remain NaN
and are handled later in ``src.dnsm.climate`` using the final strict rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR, USCRN_RAW_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"
OVERWRITE_EXISTING = True

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
STATION_CSV = SOIL_DIR / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
OUTPUT_DIR = BASE_DIR / "daily_climate_ptqc"
WARMUP_OUTPUT_DIR = BASE_DIR / "daily_climate_warmup_ptqc"
INPUT_ROOT = USCRN_RAW_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

NUMERIC_COLUMNS = ["precipitation", "tmin", "tmax", "tavg_raw", "solar"]


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def output_path(station_id: str, out_kind: str) -> Path:
    sid5 = normalize_station_id(station_id)
    if out_kind == "evaluation":
        return OUTPUT_DIR / f"daily_climate_{YEAR}_{sid5}.csv"
    if out_kind == "warmup":
        return WARMUP_OUTPUT_DIR / f"daily_climate_{YEAR}_{sid5}_warmup_y1.csv"
    raise ValueError(f"Unknown out_kind: {out_kind}")


def read_raw_year(station_name: str, station_id: str, source_year: int) -> pd.DataFrame:
    input_dir = INPUT_ROOT / str(source_year)
    pattern = f"CRND0103-{source_year}-*{station_name}.txt"
    matches = sorted(input_dir.glob(pattern)) if input_dir.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Missing raw file: station={station_name}, source_year={source_year}, "
            f"pattern={pattern}"
        )

    raw_path = matches[0]
    raw = pd.read_csv(
        raw_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=HEADERS,
        dtype=str,
        engine="python",
    )

    climate = raw[
        [
            "LST_DATE",
            "T_DAILY_MIN",
            "T_DAILY_MAX",
            "T_DAILY_MEAN",
            "P_DAILY_CALC",
            "SOLARAD_DAILY",
        ]
    ].copy()
    climate = climate.rename(
        columns={
            "T_DAILY_MIN": "tmin",
            "T_DAILY_MAX": "tmax",
            "T_DAILY_MEAN": "tavg_raw",
            "P_DAILY_CALC": "precipitation",
            "SOLARAD_DAILY": "solar",
        }
    )

    climate["date"] = pd.to_datetime(
        climate.pop("LST_DATE"), format="%Y%m%d", errors="coerce"
    )
    if climate["date"].isna().any():
        n_bad = int(climate["date"].isna().sum())
        raise RuntimeError(f"Invalid dates in {raw_path.name}: {n_bad}")

    if climate["date"].duplicated().any():
        dup = climate.loc[climate["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Duplicate dates in {raw_path.name}: {dup}")

    climate = climate.loc[climate["date"].dt.year == source_year].copy()
    climate[NUMERIC_COLUMNS] = climate[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Convert known USCRN sentinel values to NaN, but do not otherwise fill data.
    climate[NUMERIC_COLUMNS] = climate[NUMERIC_COLUMNS].replace(
        [-9999.0, -99.0], np.nan
    )

    expected = pd.date_range(
        f"{source_year}-01-01", f"{source_year}-12-31", freq="D", name="date"
    )
    climate = climate.set_index("date").reindex(expected).reset_index()

    climate.insert(0, "station_id", normalize_station_id(station_id))
    climate.insert(0, "station", station_name)
    return climate[
        [
            "station",
            "station_id",
            "date",
            "precipitation",
            "tmin",
            "tmax",
            "tavg_raw",
            "solar",
        ]
    ]


def extract_one_year(
    station_name: str,
    station_id: str,
    source_year: int,
    out_kind: str,
) -> tuple[bool, str]:
    destination = output_path(station_id, out_kind)
    if destination.exists() and not OVERWRITE_EXISTING:
        print(f"Exists, skipped: {destination.name}")
        return True, "existing"

    try:
        climate = read_raw_year(station_name, station_id, source_year)
        climate.to_csv(destination, index=False)
        print(
            f"Saved: {destination.name} (rows={len(climate)}, "
            f"P_missing={int(climate['precipitation'].isna().sum())}, "
            f"Tmin_missing={int(climate['tmin'].isna().sum())}, "
            f"Tmax_missing={int(climate['tmax'].isna().sum())})"
        )
        return True, "created"
    except Exception as error:
        message = f"Failed: {station_name} ({source_year}) {out_kind}: {error}"
        print(message)
        return False, message


def main() -> None:
    if not STATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing final-QC station list: {STATION_CSV}. "
            "Run the single-year 04 soil-QC script first."
        )

    stations = pd.read_csv(STATION_CSV, dtype={"station_id": "string"})
    stations["station_id"] = stations["station_id"].map(normalize_station_id)
    stations["year"] = pd.to_numeric(stations["year"], errors="raise").astype(int)

    failures: list[str] = []
    created = 0
    existing = 0

    for row in stations.itertuples(index=False):
        tasks = [
            (int(row.year) - 1, "warmup"),
            (int(row.year), "evaluation"),
        ]
        for source_year, out_kind in tasks:
            success, status = extract_one_year(
                row.station, row.station_id, source_year, out_kind
            )
            if success and status == "created":
                created += 1
            elif success and status == "existing":
                existing += 1
            else:
                failures.append(status)

    if failures:
        raise RuntimeError(
            f"Climate extraction failed for {len(failures)} files: "
            + "; ".join(failures)
        )

    print(f"Final single-year station-years: {len(stations)}")
    print(f"Climate files created: {created}")
    print(f"Existing climate files preserved: {existing}")
    print("No P/T gap filling was performed in this extractor.")


if __name__ == "__main__":
    main()
