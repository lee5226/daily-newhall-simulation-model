"""Evaluate strict-P/T-QC single-year D-NSM VWC against USCRN observations.

Reported metrics are Pearson's r, unbiased RMSE, signed bias, modified KGE
(Kling et al., 2012), KGE beta, KGE gamma, and the paired-day count. Inputs
are read from the ptqc soil/run directories, and outputs are written to
a run-specific metrics subdirectory so earlier analyses are not overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.utils.utils_obs_bulk import add_bulk_columns


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"
RUN_ID = "PP_warmup_ptqc"
THETA_TAG = f"layers_ptf_df_{SOIL_TAG}"
MODEL_VERSION = f"{RUN_ID}_{THETA_TAG}_dual"

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
MODEL_DIR = BASE_DIR / "daily_newhall" / SOIL_TAG
STATION_CSV = SOIL_DIR / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
# Keep this analysis fully separate from earlier soil/run versions.
OUT_DIR = BASE_DIR / "metrics" / "vwc" / SOIL_TAG / RUN_ID

PREFER_5CM_SURFACE = False
ZERO_TOL = 1.0e-12
SEASON_ORDER = ["ALL", "DJF", "MAM", "JJA", "SON"]
DEPTH_SPECS = [
    ("surface", "sm_surf_bulk", "VWC_0_12_5cm"),
    ("rootzone", "sm_rz_bulk", "VWC_0_100cm"),
]
METRIC_COLUMNS = [
    "pearson_r",
    "ubrmse",
    "bias_signed",
    "kge",
    "kge_beta",
    "kge_gamma",
]


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def empty_metrics() -> dict[str, float | int]:
    return {
        "pearson_r": np.nan,
        "ubrmse": np.nan,
        "bias_signed": np.nan,
        "kge": np.nan,
        "kge_beta": np.nan,
        "kge_gamma": np.nan,
        "n": 0,
    }


def calculate_metrics(
    observed: pd.DataFrame,
    modeled: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
) -> dict[str, float | int]:
    """Calculate all metrics from the same date-matched valid pairs."""
    paired = observed[["date", observed_column]].merge(
        modeled[["date", modeled_column]], on="date", how="inner"
    )
    paired = paired.dropna(subset=[observed_column, modeled_column])
    if paired.empty:
        return empty_metrics()

    obs = paired[observed_column].astype(float).to_numpy()
    mod = paired[modeled_column].astype(float).to_numpy()
    n = len(paired)

    obs_sd = float(np.std(obs, ddof=1)) if n >= 2 else np.nan
    mod_sd = float(np.std(mod, ddof=1)) if n >= 2 else np.nan
    obs_constant = n < 2 or np.isclose(obs_sd, 0.0, atol=ZERO_TOL, rtol=0.0)
    mod_constant = n < 2 or np.isclose(mod_sd, 0.0, atol=ZERO_TOL, rtol=0.0)

    if obs_constant or mod_constant:
        correlation = np.nan
    else:
        correlation = float(pearsonr(obs, mod)[0])

    error = mod - obs
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))

    obs_mean = float(np.mean(obs))
    mod_mean = float(np.mean(mod))
    if (
        not np.isfinite(correlation)
        or obs_constant
        or mod_constant
        or np.isclose(obs_mean, 0.0, atol=ZERO_TOL, rtol=0.0)
        or np.isclose(mod_mean, 0.0, atol=ZERO_TOL, rtol=0.0)
    ):
        beta = np.nan
        gamma = np.nan
        kge = np.nan
    else:
        beta = mod_mean / obs_mean
        obs_cv = obs_sd / obs_mean
        mod_cv = mod_sd / mod_mean
        gamma = mod_cv / obs_cv
        kge = float(
            1.0
            - np.sqrt(
                (correlation - 1.0) ** 2
                + (beta - 1.0) ** 2
                + (gamma - 1.0) ** 2
            )
        )

    return {
        "pearson_r": correlation,
        "ubrmse": ubrmse,
        "bias_signed": bias,
        "kge": kge,
        "kge_beta": beta,
        "kge_gamma": gamma,
        "n": n,
    }


def load_station_info() -> pd.DataFrame:
    if not STATION_CSV.exists():
        raise FileNotFoundError(f"Missing final-QC station file: {STATION_CSV}")
    stations = pd.read_csv(STATION_CSV, dtype={"station_id": "string"})
    required = {"station_id", "year"}
    missing = sorted(required.difference(stations.columns))
    if missing:
        raise RuntimeError(f"Missing station columns: {missing}")
    stations["station_id"] = stations["station_id"].map(normalize_station_id)
    stations["year"] = pd.to_numeric(stations["year"], errors="raise").astype(int)
    return stations.drop_duplicates(["station_id", "year"])


def observation_path(year: int, station_id: str) -> Path:
    return (
        OUTPUTS_DIR
        / str(year)
        / "per_station_sm"
        / f"obs_sm_{year}_{station_id}.csv"
    )


def model_path(station_id: str) -> Path:
    station_folder = station_id.lstrip("0") or "0"
    return (
        MODEL_DIR
        / station_folder
        / f"daily_results_{station_folder}_{MODEL_VERSION}.csv"
    )


def load_observations(year: int, station_id: str) -> pd.DataFrame:
    path = observation_path(year, station_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing observation file: {path}")
    observed = pd.read_csv(path, parse_dates=["date"])
    observed = add_bulk_columns(observed, prefer_5cm=PREFER_5CM_SURFACE)
    observed = observed[observed["date"].dt.year == year].copy()
    if observed.empty:
        raise RuntimeError(f"No observations for {station_id} in {year}")
    observed["season"] = observed["date"].dt.month.map(season_from_month)
    return observed


def load_model(year: int, station_id: str) -> pd.DataFrame:
    path = model_path(station_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing final-QC model file: {path}")
    modeled = pd.read_csv(path, parse_dates=["date"])
    modeled = modeled[modeled["date"].dt.year == year].copy()
    if modeled.empty:
        raise RuntimeError(f"No modeled values for {station_id} in {year}")
    modeled["season"] = modeled["date"].dt.month.map(season_from_month)
    return modeled


def append_metrics(
    records: list[dict],
    station_id: str,
    year: int,
    season: str,
    depth: str,
    metrics: dict[str, float | int],
) -> None:
    records.append(
        {
            "station_id": station_id,
            "year": year,
            "version": MODEL_VERSION,
            "soil_tag": SOIL_TAG,
            "season": season,
            "depth": depth,
            **metrics,
        }
    )


def summarize(station_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (season, depth), group in station_metrics.groupby(
        ["season", "depth"], sort=False
    ):
        record = {
            "version": MODEL_VERSION,
            "soil_tag": SOIL_TAG,
            "season": season,
            "depth": depth,
            "station_count": int(group["station_id"].nunique()),
            "station_count_pairs": int((group["n"] > 0).sum()),
            "station_count_r": int(group["pearson_r"].notna().sum()),
            "station_count_kge": int(group["kge"].notna().sum()),
            "paired_n_min": int(group["n"].min()),
            "paired_n_median": float(group["n"].median()),
            "paired_n_max": int(group["n"].max()),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_median"] = values.median()
            record[f"{metric}_p25"] = values.quantile(0.25)
            record[f"{metric}_p75"] = values.quantile(0.75)
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_sd"] = values.std()
        records.append(record)

    result = pd.DataFrame(records)
    result["season"] = pd.Categorical(
        result["season"], categories=SEASON_ORDER, ordered=True
    )
    result["depth"] = pd.Categorical(
        result["depth"], categories=["surface", "rootzone"], ordered=True
    )
    return result.sort_values(["season", "depth"]).reset_index(drop=True)


def main() -> None:
    stations = load_station_info()
    records: list[dict] = []
    failures: list[tuple[str, int, str]] = []

    for station in stations.itertuples(index=False):
        station_id = normalize_station_id(station.station_id)
        year = int(station.year)
        try:
            observed = load_observations(year, station_id)
            modeled = load_model(year, station_id)

            for depth, observed_column, modeled_column in DEPTH_SPECS:
                append_metrics(
                    records,
                    station_id,
                    year,
                    "ALL",
                    depth,
                    calculate_metrics(
                        observed, modeled, observed_column, modeled_column
                    ),
                )
                for season in SEASON_ORDER[1:]:
                    append_metrics(
                        records,
                        station_id,
                        year,
                        season,
                        depth,
                        calculate_metrics(
                            observed[observed["season"] == season],
                            modeled[modeled["season"] == season],
                            observed_column,
                            modeled_column,
                        ),
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
            f"VWC evaluation failed for {len(failures)} station-years: {detail}"
        )
    if not records:
        raise RuntimeError("No VWC evaluation records were produced.")

    station_metrics = pd.DataFrame(records)
    station_metrics = station_metrics[
        [
            "station_id",
            "year",
            "version",
            "soil_tag",
            "season",
            "depth",
            *METRIC_COLUMNS,
            "n",
        ]
    ]
    summary = summarize(station_metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{YEAR}_{MODEL_VERSION}"
    station_csv = OUT_DIR / f"station_metrics_long_{base_name}.csv"
    summary_csv = OUT_DIR / f"summary_metrics_{base_name}.csv"
    workbook = OUT_DIR / f"vwc_metrics_{base_name}.xlsx"

    station_metrics.to_csv(station_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for season in SEASON_ORDER:
            table = station_metrics[station_metrics["season"] == season].copy()
            table.to_excel(writer, sheet_name=season, index=False)

    print(f"Evaluated final station-years: {len(stations)}")
    print(f"Saved: {station_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()

