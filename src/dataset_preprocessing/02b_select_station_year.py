import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEARS = list(range(2010, 2017))
OUT_PATH = OUTPUTS_DIR / "selected_station_year.csv"


frames = []

for yr in YEARS:
    f = OUTPUTS_DIR / str(yr) / f"uscrn_{yr}_qc_summary.csv"
    if not f.exists():
        print(f"Missing file: {f}")
        continue

    df = pd.read_csv(f)

    required_cols = {
        "station",
        "station_id",
        "miss_rate_5cm",
        "max_gap_5cm",
        "miss_rate_100cm",
        "max_gap_100cm",
        "pass_flag",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{f.name} missing columns: {missing}")

    df["year"] = yr
    df["station_id"] = df["station_id"].astype(str).str.zfill(5)
    df["pass_flag"] = df["pass_flag"].astype(bool)
    df["miss_sum"] = df["miss_rate_5cm"] + df["miss_rate_100cm"]
    df["gap_sum"] = df["max_gap_5cm"] + df["max_gap_100cm"]

    frames.append(df)

if not frames:
    raise RuntimeError("No yearly QC summary files were found.")

qc_all = pd.concat(frames, ignore_index=True)
qc_pass = qc_all.loc[qc_all["pass_flag"]].copy()

if qc_pass.empty:
    raise RuntimeError("No station-year records passed QC.")

qc_pass = qc_pass.sort_values(
    by=["station_id", "miss_sum", "gap_sum", "year"],
    ascending=[True, True, True, True],
)

selected = qc_pass.drop_duplicates(subset=["station_id"], keep="first").copy()
selected = selected[
    ["station", "station_id", "year", "miss_sum", "gap_sum"]
].sort_values(["station_id", "year"])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
print(f"Selected stations: {len(selected)}")