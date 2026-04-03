import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    from src.utils.paths import OUTPUTS_DIR
    from src.utils.utils_obs_bulk import add_bulk_columns
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.paths import OUTPUTS_DIR
    from src.utils.utils_obs_bulk import add_bulk_columns


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)

STATION_CSV = BASE / f"aws_by_station_final_{YEAR}.csv"
PET_TAG = "pet_PP"
RUN_IDS = ["PP_spinup_layers_ptf_df_dual"]

PREFER_5CM_SURF = False
OBS_ROOT = OUTPUTS_DIR


def calc_metrics(obs: pd.DataFrame, mod: pd.DataFrame, col_obs: str, col_mod: str) -> pd.Series:
    df = pd.merge(
        obs[["date", col_obs]],
        mod[["date", col_mod]],
        on="date",
        how="inner",
    ).dropna()

    if len(df) == 0:
        return pd.Series(
            {
                "pearson_r": np.nan,
                "bias_signed": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "ubrmse": np.nan,
                "n": 0,
            }
        )

    try:
        r = pearsonr(df[col_obs], df[col_mod])[0]
    except Exception:
        r = np.nan

    err = df[col_mod] - df[col_obs]
    bias_signed = err.mean()
    mae = np.abs(err).mean()
    rmse = np.sqrt((err**2).mean())
    ubrmse = np.sqrt(np.mean((err - bias_signed) ** 2))

    return pd.Series(
        {
            "pearson_r": r,
            "bias_signed": bias_signed,
            "mae": mae,
            "rmse": rmse,
            "ubrmse": ubrmse,
            "n": len(df),
        }
    )


def get_model_path(model_dir: Path, station: str, run_id: str) -> Path:
    filename = f"daily_results_{station}_{run_id}.csv"
    return model_dir / str(station) / filename


def get_obs_path(year: int, station_obs: str) -> Path:
    return OBS_ROOT / str(year) / "per_station_sm" / f"obs_sm_{year}_{station_obs}.csv"


def load_station_info() -> pd.DataFrame:
    if not STATION_CSV.exists():
        raise FileNotFoundError(f"Missing station file: {STATION_CSV}")

    station_info = pd.read_csv(STATION_CSV, usecols=["station_id", "year"]).drop_duplicates()
    station_info["station_id"] = station_info["station_id"].astype(str).str.zfill(5)
    station_info["year"] = station_info["year"].astype(int)
    return station_info


def main() -> None:
    station_info = load_station_info()
    model_dir = BASE / "daily_newhall" / PET_TAG

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    out_dir = BASE / "metrics" / "vwc"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_station = []

    for _, row in station_info.iterrows():
        station_obs = row["station_id"]
        station_model = row["station_id"].lstrip("0")
        year = int(row["year"])

        obs_path = get_obs_path(year, station_obs)
        if not obs_path.exists():
            continue

        obs = pd.read_csv(obs_path, parse_dates=["date"])
        obs = add_bulk_columns(obs, prefer_5cm=PREFER_5CM_SURF)

        for run_id in RUN_IDS:
            model_path = get_model_path(model_dir, station_model, run_id)
            if not model_path.exists():
                continue

            mod = pd.read_csv(model_path, parse_dates=["date"])

            res_surface = calc_metrics(obs, mod, "sm_surf_bulk", "VWC_0_12_5cm")
            rows_station.append(
                {
                    "station_id": station_model,
                    "year": year,
                    "version": run_id,
                    "depth": "surface",
                    **res_surface,
                }
            )

            res_rootzone = calc_metrics(obs, mod, "sm_rz_bulk", "VWC_0_100cm")
            rows_station.append(
                {
                    "station_id": station_model,
                    "year": year,
                    "version": run_id,
                    "depth": "rootzone",
                    **res_rootzone,
                }
            )

    if not rows_station:
        raise FileNotFoundError(
            "No matching observation/model files were found. "
            "Check outputs/{YEAR}/daily_newhall/pet_PP/ and outputs/{YEAR}/per_station_sm/."
        )

    df_station = pd.DataFrame(rows_station)

    for col in ["pearson_r", "bias_signed", "mae", "rmse", "ubrmse"]:
        if col in df_station.columns:
            df_station[col] = df_station[col].round(3)

    station_out = out_dir / "station_metrics_long.csv"
    df_station.to_csv(station_out, index=False, encoding="utf-8-sig")

    q25 = lambda x: x.quantile(0.25)
    q75 = lambda x: x.quantile(0.75)

    summary = (
        df_station.groupby(["version", "depth"])
        .agg(
            pearson_r_median=("pearson_r", "median"),
            pearson_r_p25=("pearson_r", q25),
            pearson_r_p75=("pearson_r", q75),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_sd=("pearson_r", "std"),
            bias_mean=("bias_signed", "mean"),
            bias_median=("bias_signed", "median"),
            bias_sd=("bias_signed", "std"),
            mae_median=("mae", "median"),
            mae_p25=("mae", q25),
            mae_p75=("mae", q75),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_median=("rmse", "median"),
            rmse_p25=("rmse", q25),
            rmse_p75=("rmse", q75),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            ubrmse_median=("ubrmse", "median"),
            ubrmse_p25=("ubrmse", q25),
            ubrmse_p75=("ubrmse", q75),
            ubrmse_mean=("ubrmse", "mean"),
            ubrmse_sd=("ubrmse", "std"),
            n=("pearson_r", "count"),
        )
        .reset_index()
    )

    for metric in ["pearson_r", "mae", "rmse", "ubrmse"]:
        summary[f"{metric}_iqr"] = summary[f"{metric}_p75"] - summary[f"{metric}_p25"]

    round_cols = [c for c in summary.columns if c not in ["version", "depth", "n"]]
    summary[round_cols] = summary[round_cols].round(3)

    summary_out = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"Saved: {station_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()