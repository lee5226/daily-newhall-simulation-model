"""Build monthly classical-NSM climate inputs for the final P/T-QC station set.

The extracted daily climate files preserve missing values. This script uses
``src.dnsm.climate.load_climate_pair`` so the monthly NSM receives the same
strictly processed evaluation-year precipitation and temperature forcing used
by the primary daily D-NSM analysis. The preceding warm-up year is used only
for joint P/T validation and for temperature interpolation across the year
boundary; only the selected evaluation year is aggregated to monthly inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnsm.climate import load_climate_pair
from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG

STATION_CSV = SOIL_DIR / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
DAILY_DIR = BASE_DIR / "daily_climate_ptqc"
WARMUP_DAILY_DIR = BASE_DIR / "daily_climate_warmup_ptqc"
MONTHLY_DIR = BASE_DIR / "monthly_climate_ptqc"
MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def build_monthly_from_daily(station_id: str, evaluation_year: int) -> None:
    """Build one station's 12-month NSM climate input from strict PT-QC daily data."""
    sid5 = normalize_station_id(station_id)
    warmup_year = int(evaluation_year) - 1

    daily_path = DAILY_DIR / f"daily_climate_{YEAR}_{sid5}.csv"
    warmup_path = WARMUP_DAILY_DIR / f"daily_climate_{YEAR}_{sid5}_warmup_y1.csv"

    if not daily_path.exists():
        raise FileNotFoundError(f"Missing evaluation daily climate file: {daily_path}")
    if not warmup_path.exists():
        raise FileNotFoundError(f"Missing warm-up daily climate file: {warmup_path}")

    _, daily = load_climate_pair(
        warmup_filepath=warmup_path,
        evaluation_filepath=daily_path,
        warmup_year=warmup_year,
        evaluation_year=int(evaluation_year),
    )

    if daily.empty:
        raise RuntimeError(f"No processed evaluation climate records for {sid5}")

    if not daily["date"].dt.year.eq(int(evaluation_year)).all():
        years = sorted(daily["date"].dt.year.unique().tolist())
        raise RuntimeError(
            f"Processed evaluation file for {sid5} contains unexpected years: {years}"
        )

    required = ["precipitation", "tavg"]
    if daily[required].isna().any().any():
        raise RuntimeError(
            f"Missing processed precipitation or tavg remains for {sid5}"
        )

    daily = daily.copy()
    daily["month"] = daily["date"].dt.month

    monthly = (
        daily.groupby("month", as_index=False)
        .agg(
            PPT=("precipitation", "sum"),
            TAVG=("tavg", "mean"),
            valid_days=("date", "count"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    expected_months = list(range(1, 13))
    actual_months = monthly["month"].astype(int).tolist()
    if actual_months != expected_months:
        raise RuntimeError(
            f"Incomplete monthly climate for {sid5}: months={actual_months}"
        )

    expected_days = (
        pd.Series(
            pd.date_range(
                f"{evaluation_year}-01-01",
                f"{evaluation_year}-12-31",
                freq="D",
            )
        )
        .dt.month.value_counts()
        .sort_index()
        .to_numpy()
    )
    if not (monthly["valid_days"].to_numpy() == expected_days).all():
        raise RuntimeError(
            f"Monthly day counts are incomplete for {sid5}: "
            f"observed={monthly['valid_days'].tolist()}, expected={expected_days.tolist()}"
        )

    if monthly[["PPT", "TAVG"]].isna().any().any():
        raise RuntimeError(f"Missing monthly PPT or TAVG values for {sid5}")

    monthly.insert(0, "evaluation_year", int(evaluation_year))
    output_path = MONTHLY_DIR / f"monthly_climate_{YEAR}_{sid5}.csv"
    monthly.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main() -> None:
    if not STATION_CSV.exists():
        raise FileNotFoundError(f"Missing final P/T-QC station file: {STATION_CSV}")

    stations = pd.read_csv(STATION_CSV, dtype={"station_id": "string"})
    required_station = {"station_id", "year"}
    missing_station = sorted(required_station.difference(stations.columns))
    if missing_station:
        raise RuntimeError(
            f"Missing station columns in {STATION_CSV.name}: {missing_station}"
        )

    stations["station_id"] = stations["station_id"].map(normalize_station_id)
    stations["year"] = pd.to_numeric(stations["year"], errors="raise").astype(int)
    stations = stations.drop_duplicates(["station_id", "year"])

    failures: list[tuple[str, int, str]] = []
    for row in stations.itertuples(index=False):
        try:
            build_monthly_from_daily(row.station_id, int(row.year))
        except Exception as error:
            failures.append((row.station_id, int(row.year), str(error)))
            print(f"{row.station_id}-{int(row.year)}: {error}")

    if failures:
        detail = "; ".join(
            f"{station_id}-{year}: {message}"
            for station_id, year, message in failures
        )
        raise RuntimeError(
            f"Monthly PT-QC climate generation failed for {len(failures)} station-years: "
            f"{detail}"
        )

    print(f"Completed monthly PT-QC climate generation: {len(stations)} stations")
    print(f"Saved monthly inputs to: {MONTHLY_DIR}")


if __name__ == "__main__":
    main()
