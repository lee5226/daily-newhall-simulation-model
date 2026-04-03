import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
PET_TAG = "pet_PP"
RESULT_GLOB = "daily_results_*.csv"
MODEL_ROOT = OUTPUTS_DIR / str(YEAR) / "daily_newhall" / PET_TAG
OUT_DIR = OUTPUTS_DIR / str(YEAR) / "daily_PET" / PET_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_key8(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="coerce")
    return (
        dates.dt.year.astype("Int64").astype(str)
        + "-"
        + (((dates.dt.dayofyear - 1) // 8 + 1).astype("Int64").astype(str).str.zfill(2))
    )


def find_result_file(station_dir: Path) -> Path | None:
    files = sorted(station_dir.glob(RESULT_GLOB))
    if not files:
        return None
    if len(files) > 1:
        raise RuntimeError(
            f"Multiple result files found in {station_dir}. "
            f"Refine RESULT_GLOB. Files: {[p.name for p in files]}"
        )
    return files[0]


def select_pet_column(df: pd.DataFrame) -> str:
    if "PET" in df.columns:
        return "PET"
    if "PET_pre" in df.columns:
        return "PET_pre"
    raise KeyError("Expected 'PET' or 'PET_pre' column in D-NSM output.")


def convert_one(result_path: Path, station_id: str) -> Path:
    df = pd.read_csv(result_path)
    if "date" not in df.columns:
        raise KeyError(f"Missing 'date' column in {result_path.name}")

    pet_col = select_pet_column(df)
    out = df[["date", pet_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[pet_col] = pd.to_numeric(out[pet_col], errors="coerce")
    out = out.dropna(subset=["date", pet_col])

    out["key8"] = build_key8(out["date"])
    out = (
        out.groupby("key8", as_index=False)[pet_col]
        .sum()
        .rename(columns={pet_col: "PET_mm8d"})
    )
    out.insert(0, "ID", str(station_id))

    out_path = OUT_DIR / f"PET_8day_{station_id}.csv"
    out[["ID", "key8", "PET_mm8d"]].to_csv(out_path, index=False)
    return out_path


def main() -> None:
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"Missing model output directory: {MODEL_ROOT}")

    station_dirs = sorted([p for p in MODEL_ROOT.iterdir() if p.is_dir()])
    if not station_dirs:
        raise FileNotFoundError(f"No station directories found in {MODEL_ROOT}")

    written = 0
    for station_dir in station_dirs:
        station_id = station_dir.name
        result_file = find_result_file(station_dir)
        if result_file is None:
            print(f"Skipping {station_id}: no file matching {RESULT_GLOB}")
            continue

        out_path = convert_one(result_file, station_id)
        print(f"Saved: {out_path}")
        written += 1

    if written == 0:
        raise FileNotFoundError(
            "No PET_8day files were created. Check MODEL_ROOT and RESULT_GLOB."
        )


if __name__ == "__main__":
    main()
