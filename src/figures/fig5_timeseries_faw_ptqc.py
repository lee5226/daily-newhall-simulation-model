"""Create Figure 5 FAW time series for three representative stations.

Observed FAW, daily D-NSM FAW, and monthly NSM FAW are shown for the
surface and rootzone layers. All station and hydraulic-property inputs use
the final_qc_w33_dbthirdbar soil configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.utils.utils_obs_bulk import add_bulk_columns


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"
RUN_LABEL = "PP_warmup_ptqc"
THETA_TAG = f"layers_ptf_df_{SOIL_TAG}"
RUN_ID = f"{RUN_LABEL}_{THETA_TAG}_dual"

PREFER_5CM_SURFACE = False
ZERO_TOL = 1.0e-12

# TARGET_STATIONS_FOR_FIG = ["12987", "53878", "03739"]

# STATION_PANEL_TEXT = {
#     "12987": "(a) Edinburg, TX",
#     "53878": "(b) Asheville, NC",
#     "03739": "(c) Cape Charles, VA",
# }

TARGET_STATIONS_FOR_FIG = ["12987", "53927", "23904"]

STATION_PANEL_TEXT = {
    "12987": "(a) Edinburg, TX",
    "53927": "(b) Stillwater, OK",
    "23904": "(c) Batesville, AR",
}

COMBINED_FIG_BASENAME = "Figure_5_faw_timeseries_3stations"

PREVIEW_ONLY_ONE = False
SAVE_FIG = True
TARGET_WBAN = "12987"


# ---------------------------------------------------------------------
# Input and output paths
# ---------------------------------------------------------------------

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG

STATION_CSV = (
    SOIL_DIR
    / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
)

THETA_PATH = (
    SOIL_DIR
    / f"theta_by_station_final_{YEAR}_{SOIL_TAG}.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "daily_newhall"
    / SOIL_TAG
)

MONTHLY_DIR = BASE_DIR / "monthly_newhall" / SOIL_TAG
OBS_ROOT = OUTPUTS_DIR

OUT_DIR = (
    BASE_DIR
    / "figures"
    / "faw"
    / SOIL_TAG
    / RUN_LABEL
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
    }
)

COLOR_OBS = "#000000"
COLOR_SURF = "#3E7F6E"
COLOR_RZ = "#845B53"

FS_LABEL = 20
FS_TICK = 16
FS_DATE = 16
FS_INPANEL = 17

LW_OBS = 2.0
LW_DAILY = 2.0
LW_MONTHLY = 2.2
MS_MONTHLY = 6

STATION_LABEL_CFG = {
    "03739": {
        "x": 0.01,
        "y": 0.20,
        "ha": "left",
        "va": "top",
    },
    "53878": {
        "x": 0.01,
        "y": 0.20,
        "ha": "left",
        "va": "top",
    },
    "12987": {
        "x": 0.01,
        "y": 0.90,
        "ha": "left",
        "va": "top",
    },
}

DEFAULT_LABEL_CFG = {
    "x": 0.03,
    "y": 0.88,
    "ha": "left",
    "va": "top",
}


def normalize_station_id(value: object) -> str:
    """Return a five-character station identifier."""

    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def station_folder(station_id: str) -> str:
    """Return the unpadded station folder used by model outputs."""

    return normalize_station_id(station_id).lstrip("0") or "0"


def observation_path(year: int, station_id: str) -> Path:
    """Return the observation file for one station-year."""

    station_id = normalize_station_id(station_id)

    return (
        OBS_ROOT
        / str(year)
        / "per_station_sm"
        / f"obs_sm_{year}_{station_id}.csv"
    )


def model_path(station_id: str) -> Path:
    """Return the final-QC daily D-NSM result file."""

    folder = station_folder(station_id)

    return (
        MODEL_DIR
        / folder
        / f"daily_results_{folder}_{RUN_ID}.csv"
    )


def monthly_model_path(station_id: str) -> Path:
    """Return the monthly NSM result file."""

    folder = station_folder(station_id)

    return (
        MONTHLY_DIR
        / folder
        / f"monthly_results_{folder}.csv"
    )


def load_station_info() -> pd.DataFrame:
    """Load the final set of selected station-years."""

    if not STATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing final-QC station file: {STATION_CSV}"
        )

    station_info = pd.read_csv(
        STATION_CSV,
        dtype={"station_id": "string"},
        usecols=["station_id", "year"],
    )

    station_info["station_id"] = station_info[
        "station_id"
    ].map(normalize_station_id)

    station_info["year"] = pd.to_numeric(
        station_info["year"],
        errors="raise",
    ).astype(int)

    station_info = station_info.drop_duplicates(
        ["station_id", "year"]
    )

    duplicate_stations = station_info.duplicated(
        "station_id",
        keep=False,
    )

    if duplicate_stations.any():
        duplicates = (
            station_info.loc[
                duplicate_stations,
                ["station_id", "year"],
            ]
            .sort_values(["station_id", "year"])
            .to_dict("records")
        )

        raise RuntimeError(
            "Expected one selected year per station; found "
            f"duplicates: {duplicates}"
        )

    return station_info


def load_theta() -> pd.DataFrame:
    """Load final-QC surface and rootzone hydraulic thresholds."""

    if not THETA_PATH.exists():
        raise FileNotFoundError(
            f"Missing final-QC theta file: {THETA_PATH}"
        )

    theta = pd.read_csv(
        THETA_PATH,
        dtype={"station_id": "string"},
    )

    required_columns = {
        "station_id",
        "surface_theta_wp",
        "surface_theta_fc",
        "rootzone_theta_wp",
        "rootzone_theta_fc",
    }

    missing_columns = sorted(
        required_columns.difference(theta.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing theta columns: {missing_columns}"
        )

    theta["station_id"] = theta[
        "station_id"
    ].map(normalize_station_id)

    if "year" in theta.columns:
        theta["year"] = pd.to_numeric(
            theta["year"],
            errors="raise",
        ).astype(int)

    return theta


def select_theta_row(
    theta: pd.DataFrame,
    station_id: str,
    year: int,
) -> pd.Series:
    """Select exactly one theta row for a station-year."""

    station_id = normalize_station_id(station_id)
    match = theta[theta["station_id"] == station_id]

    if "year" in theta.columns:
        match = match[match["year"] == year]

    if len(match) != 1:
        raise RuntimeError(
            "Expected one theta row for "
            f"{station_id}-{year}, found {len(match)}"
        )

    return match.iloc[0]


def load_observations(
    year: int,
    station_id: str,
) -> pd.DataFrame:
    """Load observed surface and rootzone VWC for one station-year."""

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
    """Load final-QC daily D-NSM results for one station-year."""

    path = model_path(station_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing final-QC D-NSM file: {path}"
        )

    modeled = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "VWC_0_12_5cm",
        "VWC_0_100cm",
    }

    missing_columns = sorted(
        required_columns.difference(modeled.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing D-NSM columns for "
            f"{station_id}: {missing_columns}"
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
    """Load and validate the 12 monthly NSM results."""

    path = monthly_model_path(station_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing monthly NSM file: {path}"
        )

    monthly = pd.read_csv(path)

    required_columns = {
        "month",
        "FAW_0_12_5cm",
        "FAW_0_100cm",
        "spinup_converged",
    }

    missing_columns = sorted(
        required_columns.difference(monthly.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing monthly NSM columns for "
            f"{station_id}: {missing_columns}"
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

    expected_months = list(range(1, 13))
    actual_months = monthly["month"].tolist()

    if actual_months != expected_months:
        raise RuntimeError(
            "Incomplete monthly NSM results for "
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
            "Monthly NSM spinup did not converge for "
            f"station {station_id}"
        )

    monthly["date"] = (
        pd.to_datetime(
            {
                "year": [year] * len(monthly),
                "month": monthly["month"],
                "day": [1] * len(monthly),
            }
        )
        + pd.offsets.MonthEnd(0)
    )

    return monthly


def add_faw_columns(
    observed: pd.DataFrame,
    modeled: pd.DataFrame,
    theta_row: pd.Series,
) -> None:
    """Add clipped observed and D-NSM FAW columns."""

    surface_wp = float(theta_row["surface_theta_wp"])
    surface_fc = float(theta_row["surface_theta_fc"])
    rootzone_wp = float(theta_row["rootzone_theta_wp"])
    rootzone_fc = float(theta_row["rootzone_theta_fc"])

    thresholds = np.array(
        [
            surface_wp,
            surface_fc,
            rootzone_wp,
            rootzone_fc,
        ],
        dtype=float,
    )

    if not np.isfinite(thresholds).all():
        raise RuntimeError(
            "Non-finite FAW hydraulic thresholds"
        )

    if surface_fc <= surface_wp + ZERO_TOL:
        raise RuntimeError(
            "Surface theta_fc must exceed theta_wp"
        )

    if rootzone_fc <= rootzone_wp + ZERO_TOL:
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


def load_station_series(
    station_info: pd.DataFrame,
    theta: pd.DataFrame,
    target_station: str,
) -> tuple[str, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all Figure 5 series for one target station."""

    station_id = normalize_station_id(target_station)
    station_match = station_info[
        station_info["station_id"] == station_id
    ]

    if len(station_match) != 1:
        raise RuntimeError(
            "Expected one selected station-year for "
            f"{station_id}, found {len(station_match)}"
        )

    year = int(station_match.iloc[0]["year"])
    theta_row = select_theta_row(
        theta,
        station_id,
        year,
    )

    observed = load_observations(year, station_id)
    modeled = load_daily_model(year, station_id)
    monthly = load_monthly_model(year, station_id)

    add_faw_columns(observed, modeled, theta_row)

    observed = observed.sort_values("date")
    modeled = modeled.sort_values("date")
    monthly = monthly.sort_values("date")

    return station_id, year, observed, modeled, monthly


def legend_handles() -> list[Line2D]:
    """Return the common Figure 5 legend handles."""

    return [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=LW_OBS,
            linestyle="-",
            label="Observation",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=LW_DAILY,
            linestyle="--",
            label="D-NSM",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=LW_MONTHLY,
            linestyle="-",
            marker="s",
            markersize=MS_MONTHLY,
            label="NSM",
        ),
    ]


def draw_station_pair(
    ax_surface: plt.Axes,
    ax_rootzone: plt.Axes,
    observed: pd.DataFrame,
    modeled: pd.DataFrame,
    monthly: pd.DataFrame,
    label_config: dict[str, float | str],
) -> None:
    """Draw surface and rootzone FAW series on a pair of axes."""

    ax_surface.plot(
        observed["date"],
        observed["FAW_surf_obs"],
        color=COLOR_OBS,
        linestyle="-",
        linewidth=LW_OBS,
        zorder=3,
    )

    ax_surface.plot(
        modeled["date"],
        modeled["FAW_surf_mod"],
        color=COLOR_SURF,
        linestyle="--",
        linewidth=LW_DAILY,
        zorder=3,
    )

    ax_surface.step(
        monthly["date"],
        monthly["FAW_0_12_5cm"],
        where="post",
        color=COLOR_SURF,
        linewidth=LW_MONTHLY,
        zorder=3,
    )

    ax_surface.plot(
        monthly["date"],
        monthly["FAW_0_12_5cm"],
        linestyle="None",
        marker="s",
        markersize=MS_MONTHLY,
        color=COLOR_SURF,
        zorder=3,
    )

    ax_rootzone.plot(
        observed["date"],
        observed["FAW_rz_obs"],
        color=COLOR_OBS,
        linestyle="-",
        linewidth=LW_OBS,
        zorder=3,
    )

    ax_rootzone.plot(
        modeled["date"],
        modeled["FAW_rz_mod"],
        color=COLOR_RZ,
        linestyle="--",
        linewidth=LW_DAILY,
        zorder=3,
    )

    ax_rootzone.step(
        monthly["date"],
        monthly["FAW_0_100cm"],
        where="post",
        color=COLOR_RZ,
        linewidth=LW_MONTHLY,
        zorder=3,
    )

    ax_rootzone.plot(
        monthly["date"],
        monthly["FAW_0_100cm"],
        linestyle="None",
        marker="s",
        markersize=MS_MONTHLY,
        color=COLOR_RZ,
        zorder=3,
    )

    for axis in [ax_surface, ax_rootzone]:
        axis.set_ylim(-0.05, 1.05)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.grid(False)
        axis.tick_params(
            axis="y",
            labelsize=FS_TICK,
        )

    ax_surface.tick_params(
        axis="x",
        labelbottom=False,
    )

    ax_rootzone.tick_params(
        axis="x",
        labelsize=FS_DATE,
    )

    ax_surface.text(
        label_config["x"],
        label_config["y"],
        "Surface",
        transform=ax_surface.transAxes,
        fontsize=FS_INPANEL,
        ha=label_config["ha"],
        va=label_config["va"],
    )

    ax_rootzone.text(
        label_config["x"],
        label_config["y"],
        "Rootzone",
        transform=ax_rootzone.transAxes,
        fontsize=FS_INPANEL,
        ha=label_config["ha"],
        va=label_config["va"],
    )


def plot_faw_timeseries_by_station() -> None:
    """Create a separate two-panel FAW plot for each selected station."""

    station_info = load_station_info()
    theta = load_theta()

    if TARGET_WBAN is None:
        target_stations = station_info["station_id"].tolist()
    else:
        target_stations = [normalize_station_id(TARGET_WBAN)]

    preview_done = False

    for target_station in target_stations:
        (
            station_id,
            _,
            observed,
            modeled,
            monthly,
        ) = load_station_series(
            station_info,
            theta,
            target_station,
        )

        label_config = STATION_LABEL_CFG.get(
            station_id,
            DEFAULT_LABEL_CFG,
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13.0, 3.8),
            sharex=True,
        )

        ax_surface, ax_rootzone = axes

        draw_station_pair(
            ax_surface,
            ax_rootzone,
            observed,
            modeled,
            monthly,
            label_config,
        )

        ax_rootzone.xaxis.set_major_locator(
            mdates.AutoDateLocator(
                minticks=4,
                maxticks=6,
            )
        )

        ax_rootzone.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        fig.supylabel(
            "FAW",
            fontsize=FS_LABEL,
            x=0.03,
        )

        fig.legend(
            handles=legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
            fontsize=FS_INPANEL,
            handlelength=2.6,
            columnspacing=1.6,
        )

        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.80,
            bottom=0.18,
            hspace=0.08,
        )

        save_path = (
            OUT_DIR
            / f"FAW_timeseries_station_{station_folder(station_id)}.png"
        )

        if PREVIEW_ONLY_ONE:
            plt.show()
            plt.close(fig)
            preview_done = True
            break

        if SAVE_FIG:
            plt.savefig(
                save_path,
                dpi=350,
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"Saved: {save_path}")

        plt.close(fig)

    if PREVIEW_ONLY_ONE and not preview_done:
        raise RuntimeError(
            "No station was previewed. Check TARGET_WBAN."
        )


def plot_faw_timeseries_three_stations_vertical() -> None:
    """Create the final six-panel Figure 5."""

    station_info = load_station_info()
    theta = load_theta()

    fig = plt.figure(figsize=(13.0, 12.2))

    outer = gridspec.GridSpec(
        8,
        1,
        figure=fig,
        height_ratios=[
            1.0,
            1.0,
            0.50,
            1.0,
            1.0,
            0.50,
            1.0,
            1.0,
        ],
        hspace=0.10,
    )

    station_label_y = {
        "12987": -0.28,
        "53878": -0.28,
        "03739": -0.28,
    }

    panel_text_y = {
        "12987": DEFAULT_LABEL_CFG["y"],
        "53878": DEFAULT_LABEL_CFG["y"] - 0.65,
        "03739": DEFAULT_LABEL_CFG["y"] - 0.65,
    }

    surface_rows = [0, 3, 6]
    rootzone_rows = [1, 4, 7]
    axis_pairs = []

    for station_index, target_station in enumerate(
        TARGET_STATIONS_FOR_FIG
    ):
        (
            station_id,
            year,
            observed,
            modeled,
            monthly,
        ) = load_station_series(
            station_info,
            theta,
            target_station,
        )

        ax_surface = fig.add_subplot(
            outer[surface_rows[station_index]]
        )

        ax_rootzone = fig.add_subplot(
            outer[rootzone_rows[station_index]],
            sharex=ax_surface,
        )

        axis_pairs.append(
            (ax_surface, ax_rootzone)
        )

        base_config = STATION_LABEL_CFG.get(
            station_id,
            DEFAULT_LABEL_CFG,
        )

        label_config = dict(base_config)
        label_config["y"] = panel_text_y.get(
            station_id,
            base_config["y"],
        )

        draw_station_pair(
            ax_surface,
            ax_rootzone,
            observed,
            modeled,
            monthly,
            label_config,
        )

        ax_rootzone.text(
            0.5,
            station_label_y.get(station_id, -0.24),
            STATION_PANEL_TEXT.get(
                station_id,
                f"{station_id} ({year})",
            ),
            transform=ax_rootzone.transAxes,
            fontsize=FS_LABEL,
            ha="center",
            va="top",
        )

        ax_rootzone.xaxis.set_major_locator(
            mdates.MonthLocator(interval=2)
        )

        ax_rootzone.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=3,
        frameon=False,
        fontsize=FS_INPANEL,
        handlelength=2.6,
        columnspacing=1.6,
    )

    plt.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.93,
        bottom=0.05,
    )

    for ax_surface, ax_rootzone in axis_pairs:
        y_middle = (
            ax_surface.get_position().y0
            + ax_rootzone.get_position().y1
        ) / 2.0

        fig.text(
            0.04,
            y_middle,
            "FAW",
            rotation=90,
            va="center",
            ha="center",
            fontsize=FS_LABEL,
        )

    output_pdf = (
        OUT_DIR
        / f"{COMBINED_FIG_BASENAME}.pdf"
    )

    output_png = (
        OUT_DIR
        / f"{COMBINED_FIG_BASENAME}.png"
    )

    if PREVIEW_ONLY_ONE:
        plt.show()
        plt.close(fig)
        return

    if SAVE_FIG:
        plt.savefig(
            output_pdf,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.savefig(
            output_png,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {output_pdf}")
        print(f"Saved: {output_png}")

    plt.close(fig)


def main() -> None:
    plot_faw_timeseries_three_stations_vertical()


if __name__ == "__main__":
    main()
