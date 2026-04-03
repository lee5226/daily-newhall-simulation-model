import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
OUT_DIR = OUTPUTS_DIR / str(YEAR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AWS_FILE = OUT_DIR / f"aws_by_station_final_{YEAR}.csv"
VWC_DIR = OUTPUTS_DIR


def main() -> None:
    aws_df = pd.read_csv(AWS_FILE)
    aws_df["station_id"] = aws_df["station_id"].astype(str).str.zfill(5)

    records = []

    for _, row in aws_df.iterrows():
        sid = row["station_id"]
        yr = row["year"]
        vwc_path = VWC_DIR / str(yr) / f"uscrn_{yr}_vwc_stats.csv"

        if not vwc_path.exists():
            print(f"Missing file: {vwc_path.name}")
            continue

        try:
            vwc_df = pd.read_csv(vwc_path, usecols=["station_id", "lat", "lon", "elev_m"])
            vwc_df["station_id"] = vwc_df["station_id"].astype(str).str.zfill(5)

            info = vwc_df[vwc_df["station_id"] == sid].drop_duplicates()
            if len(info):
                lat = info.iloc[0]["lat"]
                lon = info.iloc[0]["lon"]
                elev_m = info.iloc[0]["elev_m"]

                records.append(
                    {
                        "station": row["station"],
                        "station_id": sid,
                        "year": yr,
                        "latitude": lat,
                        "longitude": lon,
                        "elev_m": elev_m,
                    }
                )
        except Exception as e:
            print(f"Failed: {vwc_path.name}: {e}")

    coords_df = pd.DataFrame(records)
    coords_df["nsHemisphere"] = coords_df["latitude"].apply(lambda x: "N" if x >= 0 else "S")
    coords_df["ewHemisphere"] = coords_df["longitude"].apply(lambda x: "E" if x >= 0 else "W")

    cols = [
        "station",
        "station_id",
        "year",
        "latitude",
        "longitude",
        "elev_m",
        "nsHemisphere",
        "ewHemisphere",
    ]
    coords_df = coords_df[cols]
    coords_df.to_csv(OUT_DIR / f"coordinates_final_{YEAR}.csv", index=False)

    print(f"Saved: coordinates_final_{YEAR}.csv")


if __name__ == "__main__":
    main()