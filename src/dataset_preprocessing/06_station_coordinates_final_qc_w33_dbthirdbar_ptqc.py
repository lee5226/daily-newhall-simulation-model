"""Build coordinates for stations retained by the single-year P/T-QC + final soil QC."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"
# SOIL_TAG = "final_qc_fc10_33_dbthirdbar_ptqc"

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
STATION_CSV = SOIL_DIR / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
OUTPUT_CSV = SOIL_DIR / f"coordinates_final_{YEAR}_{SOIL_TAG}.csv"


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def main() -> None:
    if not STATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing final-QC station list: {STATION_CSV}. "
            "Run 04_build_soil_params_by_station_final_qc_w33.py first."
        )

    stations = pd.read_csv(STATION_CSV, dtype={"station_id": "string"})
    stations["station_id"] = stations["station_id"].map(normalize_station_id)
    stations["year"] = pd.to_numeric(stations["year"], errors="raise").astype(int)

    records: list[dict] = []
    failures: list[tuple[str, int, str]] = []
    yearly_coordinates: dict[int, pd.DataFrame] = {}

    for row in stations.itertuples(index=False):
        station_id = normalize_station_id(row.station_id)
        year = int(row.year)
        coordinate_source = OUTPUTS_DIR / str(year) / f"uscrn_{year}_vwc_stats.csv"

        try:
            if year not in yearly_coordinates:
                if not coordinate_source.exists():
                    raise FileNotFoundError(f"Missing coordinate source: {coordinate_source}")
                coordinates = pd.read_csv(
                    coordinate_source,
                    usecols=["station_id", "lat", "lon", "elev_m"],
                    dtype={"station_id": "string"},
                )
                coordinates["station_id"] = coordinates["station_id"].map(
                    normalize_station_id
                )
                yearly_coordinates[year] = coordinates.drop_duplicates("station_id")

            match = yearly_coordinates[year]
            match = match[match["station_id"] == station_id]
            if match.empty:
                raise RuntimeError(
                    f"Station not found in {coordinate_source.name}: {station_id}"
                )

            info = match.iloc[0]
            records.append(
                {
                    "station": row.station,
                    "station_id": station_id,
                    "year": year,
                    "latitude": float(info["lat"]),
                    "longitude": float(info["lon"]),
                    "elev_m": float(info["elev_m"]) if pd.notna(info["elev_m"]) else pd.NA,
                }
            )
        except Exception as error:
            failures.append((station_id, year, str(error)))
            print(f"{station_id}-{year}: {error}")

    if failures:
        detail = "; ".join(
            f"{station_id}-{year}: {message}"
            for station_id, year, message in failures
        )
        raise RuntimeError(
            f"Coordinate extraction failed for {len(failures)} station-years: {detail}"
        )

    result = pd.DataFrame(records)
    result["nsHemisphere"] = result["latitude"].map(
        lambda value: "N" if value >= 0 else "S"
    )
    result["ewHemisphere"] = result["longitude"].map(
        lambda value: "E" if value >= 0 else "W"
    )
    result = result[
        [
            "station",
            "station_id",
            "year",
            "latitude",
            "longitude",
            "elev_m",
            "nsHemisphere",
            "ewHemisphere",
        ]
    ]

    if len(result) != len(stations):
        raise RuntimeError(
            f"Coordinate row count mismatch: expected {len(stations)}, got {len(result)}"
        )

    SOIL_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} (rows={len(result)})")


if __name__ == "__main__":
    main()
