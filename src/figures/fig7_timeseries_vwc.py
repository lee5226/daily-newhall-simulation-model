import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.utils.utils_obs_bulk import add_bulk_columns


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)

OBS_ROOT = OUTPUTS_DIR
PET_TAG = "pet_PP"
RUN_ID = "PP_spinup_layers_ptf_df_dual"

MODEL_DIR = BASE / "daily_newhall" / PET_TAG
FIG_DIR = BASE / "figures" / "vwc"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STATION_CSV = BASE / f"aws_by_station_final_{YEAR}.csv"
CLIMATE_DIR = BASE / "daily_climate"

PREFER_5CM_SURF = False

PREVIEW_ONLY_ONE = False
SAVE_FIG = True

TARGET_STATIONS_FOR_FIG = ["12987", "53878", "03739"]

STATION_PANEL_TEXT = {
    "12987": "(a) Edinburg, TX",
    "53878": "(b) Asheville, NC",
    "03739": "(c) Cape Charles, VA",
}

COMBINED_FIG_BASENAME = "Figure_7_vwc_precip_3stations"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)

COLOR_SURF = "#3E7F6E"
COLOR_RZ = "#845B53"
COLOR_PRECIP = "#A6A6A6"

FS_LABEL = 18
FS_LABEL_RIGHT = 16
FS_TICK = 17
FS_TICK_RIGHT = 16
FS_DATE = 16

LW_OBS = 2.5
LW_MODEL = 2.0

PRECIP_YMAX = 200


def load_station_info() -> pd.DataFrame:
    station_df = pd.read_csv(STATION_CSV)
    station_df["station_id"] = station_df["station_id"].astype(str).str.zfill(5)
    return station_df


def main() -> None:
    target_df = load_station_info()

    fig = plt.figure(figsize=(13, 12.2))
    outer = plt.GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1, 0.24, 1, 0.24, 1],
        hspace=0.05,
    )

    for s_idx, wban in enumerate(TARGET_STATIONS_FOR_FIG):
        row_match = target_df.loc[target_df["station_id"] == wban]
        if row_match.empty:
            print(f"Station not found in station file: {wban}")
            continue

        yr = int(row_match.iloc[0]["year"])

        obs_path = OBS_ROOT / str(yr) / "per_station_sm" / f"obs_sm_{yr}_{wban}.csv"
        model_path = MODEL_DIR / wban.lstrip("0") / f"daily_results_{wban.lstrip('0')}_{RUN_ID}.csv"
        clim_path = CLIMATE_DIR / f"daily_climate_{YEAR}_{wban}.csv"

        if not obs_path.exists() or not model_path.exists():
            print(f"Missing observation or model file: {wban}")
            continue

        if not clim_path.exists():
            print(f"Missing climate file: {clim_path.name}")
            continue

        obs_raw = pd.read_csv(obs_path, parse_dates=["date"])
        obs = add_bulk_columns(obs_raw, prefer_5cm=PREFER_5CM_SURF)

        mod = pd.read_csv(model_path, parse_dates=["date"])
        clim = pd.read_csv(clim_path, parse_dates=["date"])[["date", "precipitation"]]

        df = obs[["date", "sm_surf_bulk", "sm_rz_bulk"]].merge(
            mod[["date", "VWC_0_12_5cm", "VWC_0_100cm"]],
            on="date",
            how="inner",
        )
        if df.empty:
            print(f"No overlapping dates: {wban}")
            continue

        df = df.merge(clim, on="date", how="left")
        df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
        df.loc[df["precipitation"] < 0, "precipitation"] = np.nan
        df["precipitation"] = df["precipitation"].fillna(0.0)

        if s_idx == 0:
            ax = fig.add_subplot(outer[0])
        elif s_idx == 1:
            ax = fig.add_subplot(outer[2])
        else:
            ax = fig.add_subplot(outer[4])

        axp = ax.twinx()

        dates = pd.to_datetime(df["date"])
        vals = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0)

        if len(dates) > 1:
            d = np.diff(dates.values).astype("timedelta64[D]").astype(float)
            step = np.nanmedian(d) if d.size else 1.0
            bar_w = max(step * 1.15, 0.9)
        else:
            bar_w = 0.8

        axp.bar(
            dates,
            vals,
            width=bar_w,
            align="center",
            color=COLOR_PRECIP,
            alpha=0.60,
            edgecolor="none",
            zorder=0,
        )
        axp.set_ylim(0, PRECIP_YMAX)
        axp.set_yticks([0, 25, 50])
        axp.set_ylabel("Precipitation (mm)", fontsize=FS_LABEL_RIGHT)
        axp.tick_params(axis="y", labelsize=FS_TICK_RIGHT)
        axp.spines["right"].set_color("black")
        axp.grid(False)
        axp.yaxis.set_label_coords(1.04, 0.30)

        ax.set_zorder(2)
        ax.patch.set_alpha(0)

        ax.plot(
            df["date"],
            df["sm_surf_bulk"],
            color=COLOR_SURF,
            linewidth=LW_OBS,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            df["date"],
            df["sm_rz_bulk"],
            color=COLOR_RZ,
            linewidth=LW_OBS,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            df["date"],
            df["VWC_0_12_5cm"],
            color=COLOR_SURF,
            linewidth=LW_MODEL,
            linestyle="--",
            zorder=3,
        )
        ax.plot(
            df["date"],
            df["VWC_0_100cm"],
            color=COLOR_RZ,
            linewidth=LW_MODEL,
            linestyle="--",
            zorder=3,
        )

        ax.set_ylabel("VWC (m³/m³)", fontsize=FS_LABEL)
        ax.set_xlabel("")
        ax.set_ylim(0, 0.5)
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=FS_DATE)
        ax.tick_params(axis="y", labelsize=FS_TICK)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        ax.text(
            0.5,
            -0.14,
            STATION_PANEL_TEXT[wban],
            transform=ax.transAxes,
            fontsize=FS_LABEL,
            ha="center",
            va="top",
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_SURF, lw=LW_OBS, linestyle="-", label="Surface Observation"),
        Line2D([0], [0], color=COLOR_SURF, lw=LW_MODEL, linestyle="--", label="Surface D-NSM"),
        Line2D([0], [0], color=COLOR_RZ, lw=LW_OBS, linestyle="-", label="Rootzone Observation"),
        Line2D([0], [0], color=COLOR_RZ, lw=LW_MODEL, linestyle="--", label="Rootzone D-NSM"),
        Patch(facecolor=COLOR_PRECIP, edgecolor="none", alpha=0.60, label="Precipitation"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=16,
        handlelength=2.5,
        columnspacing=1.4,
    )

    plt.subplots_adjust(
        left=0.10,
        right=0.90,
        top=0.93,
        bottom=0.05,
    )

    out_pdf = FIG_DIR / f"{COMBINED_FIG_BASENAME}.pdf"
    out_png = FIG_DIR / f"{COMBINED_FIG_BASENAME}.png"

    if PREVIEW_ONLY_ONE:
        plt.show()
        plt.close()
        return

    if SAVE_FIG:
        plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")

    plt.close()


if __name__ == "__main__":
    main()