"""Evaluate final D-NSM alpha_fast sensitivity runs against USCRN VWC.

Metrics match the primary final_qc_w33_dbthirdbar_ptqc VWC evaluation: Pearson's
r, unbiased RMSE, signed bias, modified KGE (Kling et al., 2012), KGE beta,
KGE gamma, and paired-day count. Only all-year metrics are calculated because
this script supports the Appendix C parameter sensitivity check.
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
THETA_TAG = f"layers_ptf_df_{SOIL_TAG}"

ALPHA_FAST_VALUES = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45
]
FINAL_ALPHA_FAST = 0.35

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
SENSITIVITY_DIR = (
    BASE_DIR / "alpha_fast_sensitivity" / SOIL_TAG
)
MODEL_ROOT = SENSITIVITY_DIR / "daily_newhall"
OUT_DIR = SENSITIVITY_DIR / "metrics"

STATION_CSV = (
    SOIL_DIR
    / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
)

PRIMARY_RUN_ID = "PP_warmup_ptqc"
PRIMARY_MODEL_VERSION = f"{PRIMARY_RUN_ID}_{THETA_TAG}_dual"
FINAL_STATION_METRICS_CSV = (
    BASE_DIR
    / "metrics"
    / "vwc"
    / SOIL_TAG
    / PRIMARY_RUN_ID
    / f"station_metrics_long_{YEAR}_{PRIMARY_MODEL_VERSION}.csv"
)

PREFER_5CM_SURFACE = False
ZERO_TOL = 1.0e-12
CONSISTENCY_TOL = 1.0e-10

DEPTH_SPECS = [
    ("surface", "sm_surf_bulk", "VWC_0_12_5cm"),
    ("rootzone", "sm_rz_bulk", "VWC_0_100cm"),
]
DEPTH_ORDER = ["surface", "rootzone"]
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


def alpha_tag(alpha_fast: float) -> str:
    return f"alpha_{alpha_fast:.2f}".replace(".", "p")


def run_id_for_alpha(alpha_fast: float) -> str:
    return f"PP_warmup_ptqc_alpha_fast_{alpha_fast:.2f}".replace(".", "p")


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
    """Calculate final VWC metrics from identical date-matched pairs."""
    paired = observed[
        ["date", observed_column]
    ].merge(
        modeled[["date", modeled_column]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    paired = paired.dropna(
        subset=[observed_column, modeled_column]
    )
    if paired.empty:
        return empty_metrics()

    obs = paired[observed_column].astype(float).to_numpy()
    mod = paired[modeled_column].astype(float).to_numpy()
    n = len(paired)

    obs_sd = (
        float(np.std(obs, ddof=1)) if n >= 2 else np.nan
    )
    mod_sd = (
        float(np.std(mod, ddof=1)) if n >= 2 else np.nan
    )
    obs_constant = (
        n < 2
        or np.isclose(
            obs_sd, 0.0, atol=ZERO_TOL, rtol=0.0
        )
    )
    mod_constant = (
        n < 2
        or np.isclose(
            mod_sd, 0.0, atol=ZERO_TOL, rtol=0.0
        )
    )

    if obs_constant or mod_constant:
        correlation = np.nan
    else:
        correlation = float(pearsonr(obs, mod)[0])

    error = mod - obs
    bias = float(np.mean(error))
    ubrmse = float(
        np.sqrt(np.mean((error - bias) ** 2))
    )

    obs_mean = float(np.mean(obs))
    mod_mean = float(np.mean(mod))
    if (
        not np.isfinite(correlation)
        or obs_constant
        or mod_constant
        or np.isclose(
            obs_mean, 0.0, atol=ZERO_TOL, rtol=0.0
        )
        or np.isclose(
            mod_mean, 0.0, atol=ZERO_TOL, rtol=0.0
        )
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
        raise FileNotFoundError(
            f"Missing final-QC station file: {STATION_CSV}"
        )
    stations = pd.read_csv(
        STATION_CSV,
        dtype={"station_id": "string"},
    )
    required = {"station_id", "year"}
    missing = sorted(required.difference(stations.columns))
    if missing:
        raise RuntimeError(
            f"Missing station columns: {missing}"
        )
    stations["station_id"] = stations[
        "station_id"
    ].map(normalize_station_id)
    stations["year"] = pd.to_numeric(
        stations["year"], errors="raise"
    ).astype(int)
    return stations.drop_duplicates(
        ["station_id", "year"]
    )


def observation_path(year: int, station_id: str) -> Path:
    return (
        OUTPUTS_DIR
        / str(year)
        / "per_station_sm"
        / f"obs_sm_{year}_{station_id}.csv"
    )


def model_path(station_id: str, alpha_fast: float) -> Path:
    station_folder = station_id.lstrip("0") or "0"
    return (
        MODEL_ROOT
        / alpha_tag(alpha_fast)
        / station_folder
        / (
            f"daily_results_{station_folder}_{alpha_tag(alpha_fast)}.csv"
        )
    )


def load_observations(
    year: int,
    station_id: str,
) -> pd.DataFrame:
    path = observation_path(year, station_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing observation file: {path}"
        )
    observed = pd.read_csv(path, parse_dates=["date"])
    observed = add_bulk_columns(
        observed,
        prefer_5cm=PREFER_5CM_SURFACE,
    )
    observed["date"] = observed["date"].dt.normalize()
    observed = observed[
        observed["date"].dt.year == year
    ].copy()
    if observed.empty:
        raise RuntimeError(
            f"No observations for {station_id} in {year}"
        )
    return observed


def load_model(
    year: int,
    station_id: str,
    alpha_fast: float,
) -> pd.DataFrame:
    path = model_path(station_id, alpha_fast)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing alpha sensitivity model file: {path}"
        )
    modeled = pd.read_csv(path, parse_dates=["date"])
    modeled["date"] = modeled["date"].dt.normalize()
    modeled = modeled[
        modeled["date"].dt.year == year
    ].copy()
    if modeled.empty:
        raise RuntimeError(
            f"No modeled values for {station_id} in {year}, "
            f"alpha_fast={alpha_fast:.2f}"
        )
    return modeled


def summarize(
    station_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []

    for (alpha_fast, depth), group in station_metrics.groupby(
        ["alpha_fast", "depth"], sort=True
    ):
        record = {
            "soil_tag": SOIL_TAG,
            "alpha_fast": float(alpha_fast),
            "alpha_tag": alpha_tag(float(alpha_fast)),
            "is_final_alpha": bool(
                np.isclose(alpha_fast, FINAL_ALPHA_FAST)
            ),
            "depth": depth,
            "station_count": int(
                group["station_id"].nunique()
            ),
            "station_count_pairs": int(
                (group["n"] > 0).sum()
            ),
            "station_count_r": int(
                group["pearson_r"].notna().sum()
            ),
            "station_count_kge": int(
                group["kge"].notna().sum()
            ),
            "paired_n_min": int(group["n"].min()),
            "paired_n_median": float(
                group["n"].median()
            ),
            "paired_n_max": int(group["n"].max()),
        }

        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(
                group[metric], errors="coerce"
            ).dropna()
            record[f"{metric}_median"] = values.median()
            record[f"{metric}_p25"] = values.quantile(0.25)
            record[f"{metric}_p75"] = values.quantile(0.75)
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_sd"] = values.std()

        records.append(record)

    summary = pd.DataFrame(records)
    summary["depth"] = pd.Categorical(
        summary["depth"],
        categories=DEPTH_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(
        ["alpha_fast", "depth"]
    ).reset_index(drop=True)

    for depth in DEPTH_ORDER:
        depth_mask = summary["depth"] == depth
        final_rows = summary[
            depth_mask
            & np.isclose(
                summary["alpha_fast"], FINAL_ALPHA_FAST
            )
        ]
        if len(final_rows) != 1:
            raise RuntimeError(
                f"Expected one final-alpha summary row for {depth}, "
                f"found {len(final_rows)}"
            )
        final_row = final_rows.iloc[0]
        for metric in METRIC_COLUMNS:
            column = f"{metric}_median"
            delta_column = (
                f"{metric}_median_delta_from_"
                f"{alpha_tag(FINAL_ALPHA_FAST)}"
            )
            summary.loc[depth_mask, delta_column] = (
                summary.loc[depth_mask, column]
                - float(final_row[column])
            )

    return summary


def make_wide_summary(
    summary_long: pd.DataFrame,
) -> pd.DataFrame:
    value_columns = [
        f"{metric}_median" for metric in METRIC_COLUMNS
    ]
    wide = summary_long.pivot(
        index=[
            "soil_tag",
            "alpha_fast",
            "alpha_tag",
            "is_final_alpha",
        ],
        columns="depth",
        values=value_columns,
    )
    wide.columns = [
        f"{depth}_{metric}"
        for metric, depth in wide.columns
    ]
    return wide.reset_index().sort_values(
        "alpha_fast"
    ).reset_index(drop=True)


def make_primary_consistency_check(
    sensitivity_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the selected final-alpha run with the primary final metrics."""
    columns = [
        "depth",
        "metric",
        "n_common",
        "median_abs_difference",
        "max_abs_difference",
        "within_tolerance",
    ]
    if not FINAL_STATION_METRICS_CSV.exists():
        print(
            "Primary final station-metrics file was not found; "
            f"skipping alpha={FINAL_ALPHA_FAST:.2f} consistency check: "
            f"{FINAL_STATION_METRICS_CSV}"
        )
        return pd.DataFrame(columns=columns)

    primary = pd.read_csv(
        FINAL_STATION_METRICS_CSV,
        dtype={"station_id": "string"},
    )
    primary["station_id"] = primary[
        "station_id"
    ].map(normalize_station_id)
    if "season" in primary.columns:
        primary = primary[
            primary["season"] == "ALL"
        ].copy()

    alpha_final = sensitivity_metrics[
        np.isclose(
            sensitivity_metrics["alpha_fast"],
            FINAL_ALPHA_FAST,
        )
    ].copy()

    merge_keys = ["station_id", "year", "depth"]
    merged = alpha_final.merge(
        primary[merge_keys + METRIC_COLUMNS + ["n"]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_sensitivity", "_primary"),
    )
    if len(merged) != len(alpha_final):
        raise RuntimeError(
            f"Alpha={FINAL_ALPHA_FAST:.2f} sensitivity and primary final "
            "metrics do not "
            f"contain identical station-depth keys: sensitivity="
            f"{len(alpha_final)}, primary_common={len(merged)}"
        )

    records: list[dict] = []
    for depth in DEPTH_ORDER:
        depth_data = merged[merged["depth"] == depth]
        for metric in [*METRIC_COLUMNS, "n"]:
            sensitivity_values = pd.to_numeric(
                depth_data[f"{metric}_sensitivity"],
                errors="coerce",
            )
            primary_values = pd.to_numeric(
                depth_data[f"{metric}_primary"],
                errors="coerce",
            )
            valid = (
                sensitivity_values.notna()
                & primary_values.notna()
            )
            differences = (
                sensitivity_values[valid]
                - primary_values[valid]
            ).abs()
            records.append(
                {
                    "depth": depth,
                    "metric": metric,
                    "n_common": int(valid.sum()),
                    "median_abs_difference": (
                        differences.median()
                        if not differences.empty
                        else np.nan
                    ),
                    "max_abs_difference": (
                        differences.max()
                        if not differences.empty
                        else np.nan
                    ),
                    "within_tolerance": (
                        bool(
                            differences.max()
                            <= CONSISTENCY_TOL
                        )
                        if not differences.empty
                        else False
                    ),
                }
            )

    return pd.DataFrame(records, columns=columns)


def main() -> None:
    stations = load_station_info()
    records: list[dict] = []
    failures: list[tuple[str, int, float, str]] = []

    for station in stations.itertuples(index=False):
        station_id = normalize_station_id(station.station_id)
        year = int(station.year)

        try:
            observed = load_observations(year, station_id)
        except Exception as error:
            for alpha_fast in ALPHA_FAST_VALUES:
                failures.append(
                    (station_id, year, alpha_fast, str(error))
                )
            print(f"{station_id}-{year}: {error}")
            continue

        for alpha_fast in ALPHA_FAST_VALUES:
            try:
                modeled = load_model(
                    year, station_id, alpha_fast
                )
                for (
                    depth,
                    observed_column,
                    modeled_column,
                ) in DEPTH_SPECS:
                    metrics = calculate_metrics(
                        observed,
                        modeled,
                        observed_column,
                        modeled_column,
                    )
                    records.append(
                        {
                            "station_id": station_id,
                            "year": year,
                            "alpha_fast": alpha_fast,
                            "alpha_tag": alpha_tag(alpha_fast),
                            "run_id": run_id_for_alpha(
                                alpha_fast
                            ),
                            "soil_tag": SOIL_TAG,
                            "depth": depth,
                            **metrics,
                        }
                    )
            except Exception as error:
                failures.append(
                    (station_id, year, alpha_fast, str(error))
                )
                print(
                    f"{station_id}-{year}, "
                    f"alpha_fast={alpha_fast:.2f}: {error}"
                )

    if failures:
        detail = "; ".join(
            f"{station_id}-{year}, alpha={alpha_fast:.2f}: "
            f"{message}"
            for station_id, year, alpha_fast, message in failures
        )
        raise RuntimeError(
            f"Alpha-fast evaluation failed for "
            f"{len(failures)} cases: {detail}"
        )
    if not records:
        raise RuntimeError(
            "No alpha-fast sensitivity metrics were produced."
        )

    station_metrics = pd.DataFrame(records)
    station_metrics = station_metrics[
        [
            "station_id",
            "year",
            "alpha_fast",
            "alpha_tag",
            "run_id",
            "soil_tag",
            "depth",
            *METRIC_COLUMNS,
            "n",
        ]
    ]

    expected = (
        len(stations)
        * len(ALPHA_FAST_VALUES)
        * len(DEPTH_SPECS)
    )
    if len(station_metrics) != expected:
        raise RuntimeError(
            f"Unexpected metric row count: produced="
            f"{len(station_metrics)}, expected={expected}"
        )

    summary_long = summarize(station_metrics)
    summary_wide = make_wide_summary(summary_long)
    consistency = make_primary_consistency_check(
        station_metrics
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{YEAR}_{SOIL_TAG}"

    station_path = (
        OUT_DIR
        / f"alpha_fast_station_metrics_{base_name}.csv"
    )
    summary_long_path = (
        OUT_DIR
        / f"alpha_fast_summary_long_{base_name}.csv"
    )
    summary_wide_path = (
        OUT_DIR
        / f"alpha_fast_summary_wide_{base_name}.csv"
    )
    consistency_path = (
        OUT_DIR
        / (
            f"{alpha_tag(FINAL_ALPHA_FAST)}_primary_consistency_"
            f"{base_name}.csv"
        )
    )
    workbook_path = (
        OUT_DIR
        / f"alpha_fast_sensitivity_{base_name}.xlsx"
    )

    station_metrics.to_csv(
        station_path, index=False, encoding="utf-8-sig"
    )
    summary_long.to_csv(
        summary_long_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_wide.to_csv(
        summary_wide_path,
        index=False,
        encoding="utf-8-sig",
    )
    consistency.to_csv(
        consistency_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(workbook_path) as writer:
        summary_long.to_excel(
            writer, sheet_name="summary_long", index=False
        )
        summary_wide.to_excel(
            writer, sheet_name="summary_wide", index=False
        )
        consistency.to_excel(
            writer,
            sheet_name=f"{alpha_tag(FINAL_ALPHA_FAST)}_check",
            index=False,
        )
        station_metrics.to_excel(
            writer, sheet_name="station_metrics", index=False
        )

    print(
        f"Evaluated alpha-fast sensitivity: {len(stations)} stations x "
        f"{len(ALPHA_FAST_VALUES)} alpha values x "
        f"{len(DEPTH_SPECS)} depths"
    )
    print(f"Saved: {station_path}")
    print(f"Saved: {summary_long_path}")
    print(f"Saved: {summary_wide_path}")
    print(f"Saved: {consistency_path}")
    print(f"Saved: {workbook_path}")

    if not consistency.empty:
        failed_checks = consistency[
            ~consistency["within_tolerance"]
        ]
        if failed_checks.empty:
            print(
                f"Alpha={FINAL_ALPHA_FAST:.2f} sensitivity metrics match "
                "the primary "
                f"final metrics within {CONSISTENCY_TOL:g}."
            )
        else:
            print(
                f"WARNING: Alpha={FINAL_ALPHA_FAST:.2f} sensitivity "
                "results do not "
                "fully match the primary final metrics. Inspect: "
                f"{consistency_path}"
            )


if __name__ == "__main__":
    main()
