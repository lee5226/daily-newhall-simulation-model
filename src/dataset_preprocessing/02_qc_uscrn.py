import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR, USCRN_RAW_DIR, USCRN_METADATA_TXT


YEAR = 2016
DATA_DIR = USCRN_RAW_DIR / str(YEAR)
OUT_DIR = OUTPUTS_DIR / str(YEAR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_MISS_RATE = 0.15
MAX_CONSEC_GAP = 20
VWC_RANGE = (0.0, 0.6)


def load_metadata(fpath: str) -> pd.DataFrame:
    """
    Load station metadata including WBAN, state, name, and elevation.
    """
    colspecs = [
        (9, 14),
        (15, 17),
        (18, 52),
        (122, 128),
    ]
    meta = pd.read_fwf(
        fpath,
        colspecs=colspecs,
        names=["wban", "state", "name", "elev_m"],
        skiprows=2,
        dtype={"wban": str, "state": str, "name": str, "elev_m": float},
    )
    meta["wban"] = meta["wban"].str.zfill(5)
    meta["name"] = meta["name"].str.strip().str.replace(" ", "_")
    return meta


def load_station(fp: Path) -> pd.DataFrame:
    """
    Read selected columns from a USCRN daily01 file.
    """
    df = pd.read_fwf(fp, header=None)
    df = df.rename(
        columns={
            1: "DATE",
            3: "LON",
            4: "LAT",
            18: "SM_5",
            22: "SM_100",
        }
    )
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    df.replace([-9999.0, -99.000], np.nan, inplace=True)
    return df[["DATE", "LAT", "LON", "SM_5", "SM_100"]]


def first_data_wban(fp: Path) -> str:
    """
    Extract the WBAN ID from the first valid data line in the file.
    """
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit():
                return line.split()[0].zfill(5)
    raise ValueError(f"WBAN not found in {fp}")


def qc_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the VWC range filter and set out-of-range values to NaN.
    """
    df.loc[~df["SM_5"].between(*VWC_RANGE), "SM_5"] = np.nan
    df.loc[~df["SM_100"].between(*VWC_RANGE), "SM_100"] = np.nan
    return df


def gap_stats(series: pd.Series) -> tuple[float, int]:
    """
    Calculate missing rate and maximum consecutive gap.
    """
    miss_rate = series.isna().mean()
    consec = series.isna().astype(int)
    max_gap = (
        consec.groupby((consec.shift() != consec).cumsum())
        .transform("size")
        .mul(consec)
        .max()
    )
    return miss_rate, int(max_gap)


def save_qc_summary(
    data_dir: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict, dict, list]:
    """
    Evaluate QC for all station files and save a summary table.
    """
    records = []
    station_coords = {}
    station_wbans = {}

    for fp in data_dir.glob(f"CRND0103-{YEAR}-*.txt"):
        station = fp.stem.split("-")[-1]
        raw = load_station(fp)

        station_coords[station] = (raw["LAT"].iloc[0], raw["LON"].iloc[0])
        station_wbans[station] = first_data_wban(fp)

        df = qc_filter(raw)
        miss5, gap5 = gap_stats(df["SM_5"])
        miss100, gap100 = gap_stats(df["SM_100"])

        pass_flag = not (
            miss5 > MAX_MISS_RATE
            or gap5 > MAX_CONSEC_GAP
            or miss100 > MAX_MISS_RATE
            or gap100 > MAX_CONSEC_GAP
        )

        records.append(
            {
                "station": station,
                "station_id": station_wbans[station],
                "miss_rate_5cm": round(miss5, 3),
                "max_gap_5cm": gap5,
                "miss_rate_100cm": round(miss100, 3),
                "max_gap_100cm": gap100,
                "pass_flag": pass_flag,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("station")
    summary_df.to_csv(out_dir / f"uscrn_{YEAR}_qc_summary.csv", index=False)

    print(f"QC summary saved: {out_dir / f'uscrn_{YEAR}_qc_summary.csv'}")

    pass_list = summary_df.query("pass_flag")["station"].tolist()
    return summary_df, station_coords, station_wbans, pass_list


def save_obs_stats(
    data_dir: Path,
    out_dir: Path,
    pass_list: list,
    station_coords: dict,
    station_wbans: dict,
    meta: pd.DataFrame,
) -> None:
    """
    Save observational VWC statistics for passing stations.
    """
    all_stats = []

    for station in pass_list:
        df = qc_filter(load_station(data_dir / f"CRND0103-{YEAR}-{station}.txt"))

        p5_5, p95_5 = df["SM_5"].quantile([0.05, 0.95])
        p5_100, p95_100 = df["SM_100"].quantile([0.05, 0.95])
        lat, lon = station_coords[station]
        wban = station_wbans[station].zfill(5)

        all_stats.append(
            [
                station,
                wban,
                lat,
                lon,
                p5_5,
                p5_100,
                p95_5,
                p95_100,
            ]
        )

    stats_df = pd.DataFrame(
        all_stats,
        columns=[
            "station",
            "station_id",
            "lat",
            "lon",
            "P5_5cm",
            "P5_100cm",
            "P95_5cm",
            "P95_100cm",
        ],
    )

    meta_elev = meta[["wban", "elev_m"]].rename(columns={"wban": "station_id"})
    stats_df = stats_df.merge(meta_elev, on="station_id", how="left")
    stats_df.to_csv(out_dir / f"uscrn_{YEAR}_vwc_stats.csv", index=False)

    print(f"Observational stats saved: {out_dir / f'uscrn_{YEAR}_vwc_stats.csv'}")


def save_vwc_per_station(
    data_dir: Path,
    out_dir: Path,
    pass_list: list,
    station_wbans: dict,
) -> None:
    """
    Save cleaned VWC time series for each passing station.
    """
    per_dir = out_dir / "per_station_sm"
    per_dir.mkdir(exist_ok=True)

    for station in pass_list:
        df = qc_filter(load_station(data_dir / f"CRND0103-{YEAR}-{station}.txt"))
        df = df.rename(columns={"DATE": "date", "SM_5": "sm5", "SM_100": "sm100"})

        station_id = station_wbans[station]
        out_fn = per_dir / f"obs_sm_{YEAR}_{station_id}.csv"
        df.to_csv(out_fn, index=False)

        print(f"Saved VWC for {station} (ID {station_id}): {out_fn}")


def main() -> None:
    meta = load_metadata(USCRN_METADATA_TXT)

    _, station_coords, station_wbans, pass_list = save_qc_summary(DATA_DIR, OUT_DIR)
    save_obs_stats(DATA_DIR, OUT_DIR, pass_list, station_coords, station_wbans, meta)
    save_vwc_per_station(DATA_DIR, OUT_DIR, pass_list, station_wbans)


if __name__ == "__main__":
    main()