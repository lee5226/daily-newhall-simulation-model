import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)
SCREEN_DIR = BASE / "pet_screening"

MODIS_CSV = SCREEN_DIR / "MODIS_ETPET_STRICT_8day.csv"
DAILY_PET_ROOT = BASE / "daily_PET"
OUT_DIR = SCREEN_DIR / "pet_batch_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_modis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ID", "key8", "PET_mm8d"])
    df = df.rename(columns={"PET_mm8d": "PET_mm8d_MODIS"})
    df["ID"] = df["ID"].astype(str)
    df["key8"] = df["key8"].astype(str)
    return df


def load_our_folder(folder: Path) -> pd.DataFrame | None:
    paths = sorted(folder.glob("PET_8day_*.csv"))
    if not paths:
        return None

    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=["ID", "key8", "PET_mm8d"])
        df["ID"] = df["ID"].astype(str)
        df["key8"] = df["key8"].astype(str)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.groupby(["ID", "key8"], as_index=False)
        .agg(PET_mm8d=("PET_mm8d", "mean"))
        .rename(columns={"PET_mm8d": "PET_mm8d_OUR"})
    )
    return df


def metrics(modis, ours) -> dict:
    """
    modis: MODIS PET values
    ours: model PET values
    """
    modis = np.asarray(modis, float)
    ours = np.asarray(ours, float)

    ok = np.isfinite(modis) & np.isfinite(ours)
    n = int(ok.sum())

    if n < 6:
        return {
            "R": np.nan,
            "R2": np.nan,
            "NSE": np.nan,
            "ubRMSE": np.nan,
            "MAE": np.nan,
            "bias": np.nan,
            "n": n,
        }

    modis = modis[ok]
    ours = ours[ok]

    R = float(np.corrcoef(ours, modis)[0, 1])
    R2 = R**2

    bias = float(np.mean(ours - modis))

    sse = np.sum((ours - modis) ** 2)
    sst = np.sum((modis - modis.mean()) ** 2)
    NSE = float(1.0 - sse / sst) if sst > 0 else np.nan

    ubRMSE = float(np.sqrt(np.mean(((ours - ours.mean()) - (modis - modis.mean())) ** 2)))
    MAE = float(np.mean(np.abs(ours - modis)))

    return {
        "R": R,
        "R2": R2,
        "NSE": NSE,
        "ubRMSE": ubRMSE,
        "MAE": MAE,
        "bias": bias,
        "n": n,
    }


def run_one(version_folder: Path, mod: pd.DataFrame) -> dict | None:
    tag = version_folder.name
    our = load_our_folder(version_folder)

    if our is None:
        print(f"Skipping empty folder: {tag}")
        return None

    dfc = mod.merge(our, on=["ID", "key8"], how="inner")
    dfc["valid"] = np.isfinite(dfc["PET_mm8d_MODIS"]) & np.isfinite(dfc["PET_mm8d_OUR"])

    counts = (
        dfc.groupby("ID")["valid"]
        .sum()
        .reset_index()
        .rename(columns={"valid": "n_valid"})
    )
    counts.to_csv(OUT_DIR / f"pair_counts_{tag}.csv", index=False)

    by_station = (
        dfc.groupby("ID")
        .apply(lambda g: pd.Series(metrics(g["PET_mm8d_MODIS"], g["PET_mm8d_OUR"])))
        .reset_index()
    )
    by_station.to_csv(OUT_DIR / f"metrics_by_station_{tag}.csv", index=False)

    overall = by_station[["R", "R2", "NSE", "ubRMSE", "MAE", "bias"]].median(numeric_only=True)

    return {
        "PET_version": tag,
        "stations": int((~by_station["R"].isna()).sum()),
        "pairs": int(dfc["valid"].sum()),
        "R": float(overall.get("R", np.nan)),
        "R2": float(overall.get("R2", np.nan)),
        "NSE": float(overall.get("NSE", np.nan)),
        "ubRMSE": float(overall.get("ubRMSE", np.nan)),
        "MAE": float(overall.get("MAE", np.nan)),
        "bias": float(overall.get("bias", np.nan)),
    }


def main() -> None:
    mod = load_modis(MODIS_CSV)
    versions = sorted([p for p in DAILY_PET_ROOT.glob("pet_*") if p.is_dir()])

    summary = []
    for version_folder in versions:
        result = run_one(version_folder, mod)
        if result is not None:
            summary.append(result)

    if not summary:
        print("No results.")
        return

    df_sum = (
        pd.DataFrame(summary)
        .sort_values(
            ["NSE", "R2", "ubRMSE", "MAE", "R"],
            ascending=[False, False, True, True, False],
        )
    )
    df_sum.to_csv(OUT_DIR / "summary_all_versions.csv", index=False)
    print(df_sum)


if __name__ == "__main__":
    main()