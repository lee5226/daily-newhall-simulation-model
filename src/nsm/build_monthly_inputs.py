import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
BASE_IN = OUTPUTS_DIR / str(YEAR)

DAILY_DIR = BASE_IN / "daily_climate"
MONTHLY_DIR = BASE_IN / "monthly_climate"
MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


def build_monthly_from_daily(station_id: int | str) -> None:
    """
    Build monthly climate input from a daily climate file for one station.
    """
    sid5 = str(station_id).zfill(5)
    daily_path = DAILY_DIR / f"daily_climate_{YEAR}_{sid5}.csv"

    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    df_month = (
        df.groupby("month")
        .agg(
            PPT=("precipitation", "sum"),
            TAVG=("tavg", "mean"),
        )
        .reset_index()
    )

    out_path = MONTHLY_DIR / f"monthly_climate_{YEAR}_{sid5}.csv"
    df_month.to_csv(out_path, index=False)

    print(f"Saved monthly climate: {sid5}")


def main() -> None:
    coord_path = BASE_IN / f"coordinates_final_{YEAR}.csv"
    coord_df = pd.read_csv(coord_path)

    station_ids = (
        coord_df["station_id"]
        .astype(str)
        .str.lstrip("0")
        .astype(int)
        .unique()
        .tolist()
    )

    for sid in station_ids:
        build_monthly_from_daily(sid)


if __name__ == "__main__":
    main()