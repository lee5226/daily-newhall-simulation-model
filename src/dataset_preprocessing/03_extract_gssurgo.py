import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import GSSURGO_GDB, OUTPUTS_DIR


STN_LIST_CSV = OUTPUTS_DIR / "selected_station_year.csv"
GDB = GSSURGO_GDB
OUT_FILE = OUTPUTS_DIR / "station_mukey_map.parquet"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_mukey(point_5070, buffer_m: float = 50):
    """
    Return the first MUKEY intersecting the buffered station point.
    """
    mask_geom = point_5070.buffer(buffer_m)
    subset = gpd.read_file(
        GDB,
        layer="MUPOLYGON",
        engine="pyogrio",
        mask=mask_geom,
        columns=["MUKEY", "geometry"],
        use_arrow=True,
    )

    if subset.empty:
        return None

    hit = subset[subset.intersects(point_5070)]
    if hit.empty:
        return None

    return hit.iloc[0]["MUKEY"]


def main() -> None:
    sel_df = pd.read_csv(STN_LIST_CSV)
    sel_df["station_id"] = sel_df["station_id"].astype(str).str.zfill(5)

    coord_frames = []

    for yr in sel_df["year"].unique():
        stats = OUTPUTS_DIR / str(yr) / f"uscrn_{yr}_vwc_stats.csv"
        if not stats.exists():
            print(f"Missing coordinate file: {stats.name}")
            continue

        df = pd.read_csv(stats, usecols=["station_id", "station", "lat", "lon"])
        df["station_id"] = df["station_id"].astype(str).str.zfill(5)

        ids_this_year = sel_df.loc[sel_df["year"] == yr, "station_id"]
        coord_frames.append(df[df["station_id"].isin(ids_this_year)])

    if not coord_frames:
        raise RuntimeError("No station coordinates were found for the selected station-years.")

    stations = pd.concat(coord_frames, ignore_index=True).drop_duplicates("station_id")

    gdf_sta = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(5070)

    gdf_sta["mukey"] = gdf_sta.geometry.apply(get_mukey)

    cols = ["station", "station_id", "lat", "lon", "mukey"]
    gdf_sta[cols].to_parquet(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()