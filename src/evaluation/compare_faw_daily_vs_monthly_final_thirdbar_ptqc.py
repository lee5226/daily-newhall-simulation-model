"""Evaluate observed and modeled fractional available water (FAW).

Daily D-NSM FAW is evaluated at the daily time step and on exact calendar
month-end dates. Classical NSM monthly FAW is compared with D-NSM using the
same observed month-end targets and identical station-month samples.

Reported metrics are Pearson's r, unbiased RMSE, signed bias, and the paired
sample count. Results are written to a dedicated PT-QC FAW metrics
directory so earlier final_qc_w33_dbthirdbar results are not overwritten.
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

STATION_CSV = (
    SOIL_DIR
    / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
)
THETA_PATH = (
    SOIL_DIR
    / f"theta_by_station_final_{YEAR}_{SOIL_TAG}.csv"
)

# PT-QC monthly NSM results are stored under the same soil tag.
MONTHLY_DIR = BASE_DIR / "monthly_newhall" / SOIL_TAG

OBS_ROOT = OUTPUTS_DIR
OUT_DIR = BASE_DIR / "metrics" / "faw" / SOIL_TAG / RUN_ID

PREFER_5CM_SURFACE = False
ZERO_TOL = 1.0e-12

VERSION_DAILY = "D-NSM_daily"
VERSION_DNSM_MONTHLY = "D-NSM_EOM"
VERSION_NSM_MONTHLY = "NSM_monthly"

VERSION_ORDER = [
    VERSION_DAILY,
    VERSION_DNSM_MONTHLY,
    VERSION_NSM_MONTHLY,
]
DEPTH_ORDER = ["surface", "rootzone"]

METRIC_COLUMNS = [
    "pearson_r",
    "ubrmse",
    "bias_signed",
]

DEPTH_SPECS = [
    {
        "depth": "surface",
        "observed_vwc": "sm_surf_bulk",
        "modeled_vwc": "VWC_0_12_5cm",
        "observed_faw": "FAW_surf_obs",
        "modeled_faw": "FAW_surf_mod",
        "monthly_faw": "FAW_0_12_5cm",
    },
    {
        "depth": "rootzone",
        "observed_vwc": "sm_rz_bulk",
        "modeled_vwc": "VWC_0_100cm",
        "observed_faw": "FAW_rz_obs",
        "modeled_faw": "FAW_rz_mod",
        "monthly_faw": "FAW_0_100cm",
    },
]


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def empty_metrics() -> dict[str, float | int]:
    return {
        "pearson_r": np.nan,
        "ubrmse": np.nan,
        "bias_signed": np.nan,
        "n": 0,
    }


def calculate_paired_metrics(
    paired: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
) -> dict[str, float | int]:
    """Calculate metrics from an already paired data frame."""
    paired = paired.dropna(
        subset=[observed_column, modeled_column]
    )

    if paired.empty:
        return empty_metrics()

    observed = (
        paired[observed_column]
        .astype(float)
        .to_numpy()
    )
    modeled = (
        paired[modeled_column]
        .astype(float)
        .to_numpy()
    )
    n_pairs = len(paired)

    observed_sd = (
        float(np.std(observed, ddof=1))
        if n_pairs >= 2
        else np.nan
    )
    modeled_sd = (
        float(np.std(modeled, ddof=1))
        if n_pairs >= 2
        else np.nan
    )

    observed_constant = (
        n_pairs < 2
        or np.isclose(
            observed_sd,
            0.0,
            atol=ZERO_TOL,
            rtol=0.0,
        )
    )
    modeled_constant = (
        n_pairs < 2
        or np.isclose(
            modeled_sd,
            0.0,
            atol=ZERO_TOL,
            rtol=0.0,
        )
    )

    if observed_constant or modeled_constant:
        correlation = np.nan
    else:
        correlation = float(
            pearsonr(observed, modeled)[0]
        )

    error = modeled - observed
    bias = float(np.mean(error))
    ubrmse = float(
        np.sqrt(
            np.mean((error - bias) ** 2)
        )
    )

    return {
        "pearson_r": correlation,
        "ubrmse": ubrmse,
        "bias_signed": bias,
        "n": n_pairs,
    }


def pair_daily_series(
    observed: pd.DataFrame,
    modeled: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
) -> pd.DataFrame:
    """Pair daily observed and modeled values on exact dates."""
    paired = observed[
        ["date", observed_column]
    ].merge(
        modeled[["date", modeled_column]],
        on="date",
        how="inner",
        validate="one_to_one",
    )

    return paired.dropna(
        subset=[observed_column, modeled_column]
    )


def select_exact_month_end_pairs(
    daily_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Retain common valid pairs observed on exact calendar month ends."""
    if daily_pairs.empty:
        return daily_pairs.copy()

    month_end_pairs = (
        daily_pairs.loc[
            daily_pairs["date"].dt.is_month_end
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    return month_end_pairs


def load_station_info() -> pd.DataFrame:
    if not STATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing final-QC station file: {STATION_CSV}"
        )

    stations = pd.read_csv(
        STATION_CSV,
        dtype={"station_id": "string"},
        usecols=["station_id", "year"],
    )

    stations["station_id"] = stations[
        "station_id"
    ].map(normalize_station_id)

    stations["year"] = pd.to_numeric(
        stations["year"],
        errors="raise",
    ).astype(int)

    return stations.drop_duplicates(
        ["station_id", "year"]
    )


def load_theta() -> pd.DataFrame:
    if not THETA_PATH.exists():
        raise FileNotFoundError(
            f"Missing theta file: {THETA_PATH}"
        )

    theta = pd.read_csv(
        THETA_PATH,
        dtype={"station_id": "string"},
    )
    theta["station_id"] = theta[
        "station_id"
    ].map(normalize_station_id)

    required = {
        "station_id",
        "surface_theta_wp",
        "surface_theta_fc",
        "rootzone_theta_wp",
        "rootzone_theta_fc",
    }
    missing = sorted(
        required.difference(theta.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing theta columns: {missing}"
        )

    return theta


def observation_path(
    year: int,
    station_id: str,
) -> Path:
    return (
        OBS_ROOT
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


def monthly_model_path(station_id: str) -> Path:
    station_folder = station_id.lstrip("0") or "0"

    return (
        MONTHLY_DIR
        / station_folder
        / f"monthly_results_{station_folder}.csv"
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

    observed = pd.read_csv(
        path,
        parse_dates=["date"],
    )
    observed = add_bulk_columns(
        observed,
        prefer_5cm=PREFER_5CM_SURFACE,
    )
    observed = observed[
        observed["date"].dt.year == year
    ].copy()

    if observed.empty:
        raise RuntimeError(
            f"No observations for {station_id} in {year}"
        )

    return observed


def load_daily_model(
    year: int,
    station_id: str,
) -> pd.DataFrame:
    path = model_path(station_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing final D-NSM file: {path}"
        )

    modeled = pd.read_csv(
        path,
        parse_dates=["date"],
    )
    modeled = modeled[
        modeled["date"].dt.year == year
    ].copy()

    if modeled.empty:
        raise RuntimeError(
            f"No D-NSM values for {station_id} in {year}"
        )

    return modeled


def load_monthly_model(
    year: int,
    station_id: str,
) -> pd.DataFrame:
    path = monthly_model_path(station_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing monthly NSM file: {path}"
        )

    monthly = pd.read_csv(path)

    required = {
        "month",
        "FAW_0_12_5cm",
        "FAW_0_100cm",
        "spinup_converged",
    }
    missing = sorted(
        required.difference(monthly.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing monthly NSM columns for "
            f"{station_id}: {missing}"
        )

    monthly["month"] = pd.to_numeric(
        monthly["month"],
        errors="raise",
    ).astype(int)

    monthly = (
        monthly
        .sort_values("month")
        .reset_index(drop=True)
    )

    actual_months = monthly["month"].tolist()
    expected_months = list(range(1, 13))

    if actual_months != expected_months:
        raise RuntimeError(
            f"Incomplete monthly NSM results for "
            f"{station_id}: months={actual_months}"
        )

    converged = (
        monthly["spinup_converged"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    )

    if not converged.all():
        raise RuntimeError(
            f"Monthly NSM spinup did not converge "
            f"for station {station_id}"
        )

    monthly_dates = pd.to_datetime(
        {
            "year": [year] * len(monthly),
            "month": monthly["month"],
            "day": [1] * len(monthly),
        }
    )
    monthly["date"] = (
        monthly_dates + pd.offsets.MonthEnd(0)
    )

    return monthly


def add_faw_columns(
    observed: pd.DataFrame,
    modeled: pd.DataFrame,
    theta_row: pd.Series,
) -> None:
    """Add clipped observed and D-NSM FAW columns."""
    surface_wp = float(
        theta_row["surface_theta_wp"]
    )
    surface_fc = float(
        theta_row["surface_theta_fc"]
    )
    rootzone_wp = float(
        theta_row["rootzone_theta_wp"]
    )
    rootzone_fc = float(
        theta_row["rootzone_theta_fc"]
    )

    if surface_fc <= surface_wp:
        raise RuntimeError(
            "Surface theta_fc must exceed theta_wp"
        )
    if rootzone_fc <= rootzone_wp:
        raise RuntimeError(
            "Rootzone theta_fc must exceed theta_wp"
        )

    observed["FAW_surf_obs"] = (
        observed["sm_surf_bulk"] - surface_wp
    ) / (surface_fc - surface_wp)

    observed["FAW_rz_obs"] = (
        observed["sm_rz_bulk"] - rootzone_wp
    ) / (rootzone_fc - rootzone_wp)

    modeled["FAW_surf_mod"] = (
        modeled["VWC_0_12_5cm"] - surface_wp
    ) / (surface_fc - surface_wp)

    modeled["FAW_rz_mod"] = (
        modeled["VWC_0_100cm"] - rootzone_wp
    ) / (rootzone_fc - rootzone_wp)

    for column in [
        "FAW_surf_obs",
        "FAW_rz_obs",
    ]:
        observed[column] = observed[column].clip(
            0.0,
            1.0,
        )

    for column in [
        "FAW_surf_mod",
        "FAW_rz_mod",
    ]:
        modeled[column] = modeled[column].clip(
            0.0,
            1.0,
        )


def append_metrics(
    records: list[dict],
    station_id: str,
    year: int,
    version: str,
    frequency: str,
    depth: str,
    metrics: dict[str, float | int],
) -> None:
    records.append(
        {
            "station_id": station_id,
            "year": year,
            "run_id": MODEL_VERSION,
            "soil_tag": SOIL_TAG,
            "version": version,
            "freq": frequency,
            "depth": depth,
            **metrics,
        }
    )


def summarize(
    station_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []

    for (
        version,
        frequency,
        depth,
    ), group in station_metrics.groupby(
        ["version", "freq", "depth"],
        sort=False,
    ):
        record = {
            "run_id": MODEL_VERSION,
            "soil_tag": SOIL_TAG,
            "version": version,
            "freq": frequency,
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
            "paired_n_min": int(
                group["n"].min()
            ),
            "paired_n_median": float(
                group["n"].median()
            ),
            "paired_n_max": int(
                group["n"].max()
            ),
        }

        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            record[f"{metric}_median"] = (
                values.median()
            )
            record[f"{metric}_p25"] = (
                values.quantile(0.25)
            )
            record[f"{metric}_p75"] = (
                values.quantile(0.75)
            )
            record[f"{metric}_mean"] = (
                values.mean()
            )
            record[f"{metric}_sd"] = (
                values.std()
            )

        records.append(record)

    result = pd.DataFrame(records)

    result["version"] = pd.Categorical(
        result["version"],
        categories=VERSION_ORDER,
        ordered=True,
    )
    result["depth"] = pd.Categorical(
        result["depth"],
        categories=DEPTH_ORDER,
        ordered=True,
    )

    return (
        result
        .sort_values(["version", "depth"])
        .reset_index(drop=True)
    )


def make_monthly_comparison(
    station_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare NSM and D-NSM using metric-common stations."""
    monthly = station_metrics[
        (station_metrics["freq"] == "monthly")
        & station_metrics["version"].isin(
            [
                VERSION_DNSM_MONTHLY,
                VERSION_NSM_MONTHLY,
            ]
        )
    ].copy()

    metric_specs = [
        ("pearson_r", "r", "higher"),
        ("ubrmse", "ubRMSE", "lower"),
        ("bias_signed", "bias", "absolute_lower"),
    ]

    records: list[dict] = []

    for depth in DEPTH_ORDER:
        depth_metrics = monthly[
            monthly["depth"] == depth
        ]

        for metric, label, direction in metric_specs:
            pivot = depth_metrics.pivot_table(
                index="station_id",
                columns="version",
                values=metric,
                aggfunc="first",
            )

            required_versions = [
                VERSION_DNSM_MONTHLY,
                VERSION_NSM_MONTHLY,
            ]

            if all(
                version in pivot.columns
                for version in required_versions
            ):
                common = pivot.dropna(
                    subset=required_versions
                )
            else:
                common = pd.DataFrame(
                    columns=required_versions
                )

            if common.empty:
                dnsm_better = pd.Series(dtype=bool)
            elif direction == "higher":
                dnsm_better = (
                    common[VERSION_DNSM_MONTHLY]
                    > common[VERSION_NSM_MONTHLY]
                )
            elif direction == "lower":
                dnsm_better = (
                    common[VERSION_DNSM_MONTHLY]
                    < common[VERSION_NSM_MONTHLY]
                )
            else:
                dnsm_better = (
                    common[VERSION_DNSM_MONTHLY].abs()
                    < common[VERSION_NSM_MONTHLY].abs()
                )

            records.append(
                {
                    "soil_tag": SOIL_TAG,
                    "depth": depth,
                    "metric": label,
                    "n_common": len(common),
                    "NSM": (
                        common[
                            VERSION_NSM_MONTHLY
                        ].median()
                        if len(common)
                        else np.nan
                    ),
                    "D-NSM": (
                        common[
                            VERSION_DNSM_MONTHLY
                        ].median()
                        if len(common)
                        else np.nan
                    ),
                    "n_dnsm_better": int(
                        dnsm_better.sum()
                    ),
                    "fraction_dnsm_better": (
                        float(dnsm_better.mean())
                        if len(common)
                        else np.nan
                    ),
                }
            )

    comparison_long = pd.DataFrame(records)

    comparison_wide_records: list[dict] = []

    for depth in DEPTH_ORDER:
        record = {
            "soil_tag": SOIL_TAG,
            "depth": depth,
        }

        depth_rows = comparison_long[
            comparison_long["depth"] == depth
        ]

        for _, row in depth_rows.iterrows():
            metric = row["metric"]

            record[f"{metric}_n_common"] = int(
                row["n_common"]
            )
            record[f"{metric}_NSM"] = row["NSM"]
            record[f"{metric}_D-NSM"] = row["D-NSM"]
            record[
                f"{metric}_n_DNSM_better"
            ] = int(row["n_dnsm_better"])
            record[
                f"{metric}_fraction_DNSM_better"
            ] = row["fraction_dnsm_better"]

        comparison_wide_records.append(record)

    comparison_wide = pd.DataFrame(
        comparison_wide_records
    )

    return comparison_long, comparison_wide


def main() -> None:
    stations = load_station_info()
    theta = load_theta()

    records: list[dict] = []
    failures: list[tuple[str, int, str]] = []

    for station in stations.itertuples(index=False):
        station_id = normalize_station_id(
            station.station_id
        )
        year = int(station.year)

        try:
            theta_match = theta[
                theta["station_id"] == station_id
            ]

            if "year" in theta.columns:
                theta_match = theta_match[
                    pd.to_numeric(
                        theta_match["year"],
                        errors="coerce",
                    ) == year
                ]

            if len(theta_match) != 1:
                raise RuntimeError(
                    f"Expected one theta row for "
                    f"{station_id}-{year}, found "
                    f"{len(theta_match)}"
                )

            theta_row = theta_match.iloc[0]

            observed = load_observations(
                year,
                station_id,
            )
            modeled = load_daily_model(
                year,
                station_id,
            )
            monthly = load_monthly_model(
                year,
                station_id,
            )

            add_faw_columns(
                observed,
                modeled,
                theta_row,
            )

            for specification in DEPTH_SPECS:
                depth = specification["depth"]
                observed_faw = specification[
                    "observed_faw"
                ]
                modeled_faw = specification[
                    "modeled_faw"
                ]
                monthly_faw = specification[
                    "monthly_faw"
                ]

                daily_pairs = pair_daily_series(
                    observed,
                    modeled,
                    observed_faw,
                    modeled_faw,
                )

                append_metrics(
                    records,
                    station_id,
                    year,
                    VERSION_DAILY,
                    "daily",
                    depth,
                    calculate_paired_metrics(
                        daily_pairs,
                        observed_faw,
                        modeled_faw,
                    ),
                )

                eom_pairs = select_exact_month_end_pairs(
                    daily_pairs
                )

                # Use one common exact-month-end sample for both models.
                # Months without a valid observation, D-NSM value, or NSM
                # value on the calendar month end are excluded from both
                # comparisons.
                common_eom_pairs = eom_pairs.merge(
                    monthly[
                        ["date", monthly_faw]
                    ],
                    on="date",
                    how="inner",
                    validate="one_to_one",
                ).dropna(
                    subset=[
                        observed_faw,
                        modeled_faw,
                        monthly_faw,
                    ]
                )

                append_metrics(
                    records,
                    station_id,
                    year,
                    VERSION_DNSM_MONTHLY,
                    "monthly",
                    depth,
                    calculate_paired_metrics(
                        common_eom_pairs,
                        observed_faw,
                        modeled_faw,
                    ),
                )

                append_metrics(
                    records,
                    station_id,
                    year,
                    VERSION_NSM_MONTHLY,
                    "monthly",
                    depth,
                    calculate_paired_metrics(
                        common_eom_pairs,
                        observed_faw,
                        monthly_faw,
                    ),
                )

        except Exception as error:
            failures.append(
                (station_id, year, str(error))
            )
            print(
                f"{station_id}-{year}: {error}"
            )

    if failures:
        detail = "; ".join(
            f"{station_id}-{year}: {message}"
            for station_id, year, message
            in failures
        )

        raise RuntimeError(
            f"FAW evaluation failed for "
            f"{len(failures)} station-years: "
            f"{detail}"
        )

    if not records:
        raise RuntimeError(
            "No FAW evaluation records were produced."
        )

    station_metrics = pd.DataFrame(records)

    station_metrics = station_metrics[
        [
            "station_id",
            "year",
            "run_id",
            "soil_tag",
            "version",
            "freq",
            "depth",
            *METRIC_COLUMNS,
            "n",
        ]
    ]

    summary = summarize(station_metrics)
    comparison_long, comparison_wide = (
        make_monthly_comparison(station_metrics)
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = f"{YEAR}_{MODEL_VERSION}"

    station_path = (
        OUT_DIR
        / f"faw_station_metrics_{base_name}.csv"
    )
    summary_path = (
        OUT_DIR
        / f"faw_summary_metrics_{base_name}.csv"
    )
    comparison_long_path = (
        OUT_DIR
        / f"faw_nsm_dnsm_comparison_long_{base_name}.csv"
    )
    comparison_wide_path = (
        OUT_DIR
        / f"faw_nsm_dnsm_comparison_wide_{base_name}.csv"
    )
    workbook_path = (
        OUT_DIR
        / f"faw_metrics_{base_name}.xlsx"
    )

    station_metrics.to_csv(
        station_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    comparison_long.to_csv(
        comparison_long_path,
        index=False,
        encoding="utf-8-sig",
    )
    comparison_wide.to_csv(
        comparison_wide_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(workbook_path) as writer:
        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        comparison_long.to_excel(
            writer,
            sheet_name="comparison_long",
            index=False,
        )
        comparison_wide.to_excel(
            writer,
            sheet_name="comparison_wide",
            index=False,
        )
        station_metrics.to_excel(
            writer,
            sheet_name="station_metrics",
            index=False,
        )

    print(
        f"Evaluated final station-years: "
        f"{len(stations)}"
    )
    print(f"Saved: {station_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {comparison_long_path}")
    print(f"Saved: {comparison_wide_path}")
    print(f"Saved: {workbook_path}")


if __name__ == "__main__":
    main()