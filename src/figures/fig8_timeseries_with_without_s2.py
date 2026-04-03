import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.utils.utils_obs_bulk import add_bulk_columns


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)

OBS_ROOT = OUTPUTS_DIR
PET_TAG = "pet_PP"

RUN_ID_DUAL = "PP_spinup_layers_ptf_df_dual"
RUN_ID_SINGLE = "PP_spinup_layers_ptf_df_single"

MODEL_DIR = BASE / "daily_newhall" / PET_TAG
CLIMATE_DIR = BASE / "daily_climate"

FIG_DIR = BASE / "figures" / "comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STATION_CSV = BASE / f"aws_by_station_final_{YEAR}.csv"

PREFER_5CM_SURF = False

SURFACE_WBAN = "53150"
ROOTZONE_WBAN = "63829"

PANEL_LABEL_SURF = "(a) Newton, GA"
PANEL_LABEL_RZ = "(b) Yosemite Village, CA"

PREVIEW_ONLY_ONE = True
SAVE_FIG = False

SHOW_OBS = True
SHOW_PRECIP = True
SHOW_OBS_INSET = False
SHOW_INSET_SURF = True
SHOW_INSET_RZ = True

INSET_SURF_X1 = "2014-01-25"
INSET_SURF_X2 = "2014-04-20"
INSET_SURF_Y1 = 0.12
INSET_SURF_Y2 = 0.34
INSET_SURF_LOC = "upper left"
INSET_SURF_WIDTH = "30%"
INSET_SURF_HEIGHT = "50%"
INSET_SURF_BORDERPAD = 1.0
INSET_SURF_LOC1 = 3
INSET_SURF_LOC2 = 4

INSET_RZ_X1 = "2013-02-03"
INSET_RZ_X2 = "2013-03-08"
INSET_RZ_Y1 = 0.05
INSET_RZ_Y2 = 0.18
INSET_RZ_LOC = "upper left"
INSET_RZ_WIDTH = "26%"
INSET_RZ_HEIGHT = "50%"
INSET_RZ_BORDERPAD = 1.0
INSET_RZ_LOC1 = 3
INSET_RZ_LOC2 = 4

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)

COLOR_OBS = "#000000"
COLOR_SURF_DUAL = "#3E7F6E"
COLOR_RZ_DUAL = "#845B53"
COLOR_SURF_SINGLE = "#C44E52"
COLOR_RZ_SINGLE = "#8C6BB1"
COLOR_PRECIP = "#A6A6A6"

FS_LABEL = 18
FS_LABEL_RIGHT = 14
FS_TICK = 18
FS_TICK_RIGHT = 14
FS_DATE = 16
FS_LEGEND = 16

LW_OBS = 2.0
LW_SINGLE = 2.0
LW_DUAL = 2.0

LW_OBS_INSET = 2.6
LW_SINGLE_INSET = 2.4
LW_DUAL_INSET = 2.4

FIG_W = 15.0
FIG_H = 3.5

YMAX_SURF = 1.00
YMAX_RZ = 1.00
PRECIP_YMAX = 200


def add_inset(
    ax,
    df,
    obs_col,
    single_col,
    dual_col,
    dual_color,
    single_color,
    x1,
    x2,
    y1,
    y2,
    loc,
    width,
    height,
    borderpad,
    loc1,
    loc2,
):
    axins = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        borderpad=borderpad,
    )

    if SHOW_OBS and SHOW_OBS_INSET:
        axins.plot(
            df["date"],
            df[obs_col],
            color=COLOR_OBS,
            linewidth=LW_OBS_INSET,
            linestyle="-",
            zorder=3,
        )

    axins.plot(
        df["date"],
        df[single_col],
        color=single_color,
        linewidth=LW_SINGLE_INSET,
        linestyle="-",
        zorder=3,
    )
    axins.plot(
        df["date"],
        df[dual_col],
        color=dual_color,
        linewidth=LW_DUAL_INSET,
        linestyle="--",
        zorder=3,
    )

    axins.set_xlim(pd.to_datetime(x1), pd.to_datetime(x2))
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(False)
    axins.set_facecolor("white")

    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)

    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="black", lw=1.0)


def load_station_info() -> pd.DataFrame:
    station_df = pd.read_csv(STATION_CSV)
    station_df["station_id"] = station_df["station_id"].astype(str).str.zfill(5)
    return station_df


def build_panel_dataframe(target_df: pd.DataFrame, wban: str) -> pd.DataFrame | None:
    row_match = target_df.loc[target_df["station_id"] == wban]
    if row_match.empty:
        print(f"Station not found: {wban}")
        return None

    yr = int(row_match.iloc[0]["year"])

    obs_path = OBS_ROOT / str(yr) / "per_station_sm" / f"obs_sm_{yr}_{wban}.csv"
    model_path_dual = MODEL_DIR / wban.lstrip("0") / f"daily_results_{wban.lstrip('0')}_{RUN_ID_DUAL}.csv"
    model_path_single = MODEL_DIR / wban.lstrip("0") / f"daily_results_{wban.lstrip('0')}_{RUN_ID_SINGLE}.csv"
    clim_path = CLIMATE_DIR / f"daily_climate_{YEAR}_{wban}.csv"

    if not obs_path.exists():
        print(f"Missing observation file: {wban}")
        return None
    if not model_path_dual.exists():
        print(f"Missing dual model file: {wban}")
        return None
    if not model_path_single.exists():
        print(f"Missing single model file: {wban}")
        return None
    if not clim_path.exists():
        print(f"Missing climate file: {clim_path.name}")
        return None

    obs_raw = pd.read_csv(obs_path, parse_dates=["date"])
    obs = add_bulk_columns(obs_raw, prefer_5cm=PREFER_5CM_SURF)

    mod_dual = pd.read_csv(model_path_dual, parse_dates=["date"])
    mod_single = pd.read_csv(model_path_single, parse_dates=["date"])
    clim = pd.read_csv(clim_path, parse_dates=["date"])[["date", "precipitation"]]

    mod_dual = mod_dual[["date", "VWC_0_12_5cm", "VWC_0_100cm"]].rename(
        columns={
            "VWC_0_12_5cm": "VWC_0_12_5cm_dual",
            "VWC_0_100cm": "VWC_0_100cm_dual",
        }
    )
    mod_single = mod_single[["date", "VWC_0_12_5cm", "VWC_0_100cm"]].rename(
        columns={
            "VWC_0_12_5cm": "VWC_0_12_5cm_single",
            "VWC_0_100cm": "VWC_0_100cm_single",
        }
    )

    df = (
        obs[["date", "sm_surf_bulk", "sm_rz_bulk"]]
        .merge(mod_single, on="date", how="inner")
        .merge(mod_dual, on="date", how="inner")
        .merge(clim, on="date", how="left")
    )

    if df.empty:
        print(f"No overlapping dates among observation, single, and dual runs: {wban}")
        return None

    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df.loc[df["precipitation"] < 0, "precipitation"] = np.nan
    df["precipitation"] = df["precipitation"].fillna(0.0)

    return df


def draw_panel(ax, df: pd.DataFrame, info: dict) -> None:
    if SHOW_PRECIP:
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
        axp.set_yticks([0, 30, 60])
        axp.set_ylabel("Precipitation (mm)", fontsize=FS_LABEL_RIGHT)
        axp.tick_params(axis="y", labelsize=FS_TICK_RIGHT, colors="black")
        axp.spines["right"].set_color("black")
        axp.yaxis.set_label_coords(1.03, 0.30)
        axp.grid(False)

        ax.set_zorder(2)
        ax.patch.set_alpha(0)

    if SHOW_OBS:
        ax.plot(
            df["date"],
            df[info["obs_col"]],
            color=COLOR_OBS,
            linewidth=LW_OBS,
            linestyle="-",
            zorder=3,
        )

    ax.plot(
        df["date"],
        df[info["single_col"]],
        color=info["single_color"],
        linewidth=LW_SINGLE,
        linestyle="-",
        zorder=3,
    )
    ax.plot(
        df["date"],
        df[info["dual_col"]],
        color=info["dual_color"],
        linewidth=LW_DUAL,
        linestyle="--",
        zorder=3,
    )

    ax.set_ylabel("VWC (m³/m³)", fontsize=FS_LABEL)
    ax.set_xlabel("")
    ax.set_ylim(0, info["y_max"])
    ax.set_yticks([0, 0.2, 0.4])
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=FS_DATE)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    if info["depth_name"] == "surface" and SHOW_INSET_SURF:
        add_inset(
            ax=ax,
            df=df,
            obs_col=info["obs_col"],
            single_col=info["single_col"],
            dual_col=info["dual_col"],
            dual_color=info["dual_color"],
            single_color=info["single_color"],
            x1=INSET_SURF_X1,
            x2=INSET_SURF_X2,
            y1=INSET_SURF_Y1,
            y2=INSET_SURF_Y2,
            loc=INSET_SURF_LOC,
            width=INSET_SURF_WIDTH,
            height=INSET_SURF_HEIGHT,
            borderpad=INSET_SURF_BORDERPAD,
            loc1=INSET_SURF_LOC1,
            loc2=INSET_SURF_LOC2,
        )

    if info["depth_name"] == "rootzone" and SHOW_INSET_RZ:
        add_inset(
            ax=ax,
            df=df,
            obs_col=info["obs_col"],
            single_col=info["single_col"],
            dual_col=info["dual_col"],
            dual_color=info["dual_color"],
            single_color=info["single_color"],
            x1=INSET_RZ_X1,
            x2=INSET_RZ_X2,
            y1=INSET_RZ_Y1,
            y2=INSET_RZ_Y2,
            loc=INSET_RZ_LOC,
            width=INSET_RZ_WIDTH,
            height=INSET_RZ_HEIGHT,
            borderpad=INSET_RZ_BORDERPAD,
            loc1=INSET_RZ_LOC1,
            loc2=INSET_RZ_LOC2,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_OBS, linewidth=LW_OBS, linestyle="-", label="Observation"),
        Line2D([0], [0], color=info["dual_color"], linewidth=LW_DUAL, linestyle="--", label=r"D-NSM with $S_2$"),
        Line2D([0], [0], color=info["single_color"], linewidth=LW_SINGLE, linestyle="-", label=r"D-NSM without $S_2$"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=False,
        facecolor="white",
        edgecolor="0.6",
        framealpha=0.95,
        fontsize=FS_LEGEND,
        handlelength=2.6,
        borderpad=0.4,
        labelspacing=0.35,
    )

    ax.text(
        0.5,
        -0.14,
        info["panel_label"],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FS_LABEL,
    )


def make_two_panel_surface_rootzone() -> None:
    target_df = load_station_info()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIG_W, FIG_H * 2.15),
        constrained_layout=False,
    )

    panel_info = [
        {
            "wban": SURFACE_WBAN,
            "depth_name": "surface",
            "obs_col": "sm_surf_bulk",
            "single_col": "VWC_0_12_5cm_single",
            "dual_col": "VWC_0_12_5cm_dual",
            "dual_color": COLOR_SURF_DUAL,
            "single_color": COLOR_SURF_SINGLE,
            "y_max": YMAX_SURF,
            "panel_label": PANEL_LABEL_SURF,
        },
        {
            "wban": ROOTZONE_WBAN,
            "depth_name": "rootzone",
            "obs_col": "sm_rz_bulk",
            "single_col": "VWC_0_100cm_single",
            "dual_col": "VWC_0_100cm_dual",
            "dual_color": COLOR_RZ_DUAL,
            "single_color": COLOR_RZ_SINGLE,
            "y_max": YMAX_RZ,
            "panel_label": PANEL_LABEL_RZ,
        },
    ]

    for ax, info in zip(axes, panel_info):
        df = build_panel_dataframe(target_df, info["wban"])
        if df is None:
            continue
        draw_panel(ax, df, info)

    plt.subplots_adjust(
        left=0.10,
        right=0.90,
        top=0.95,
        bottom=0.08,
        hspace=0.35,
    )

    out_pdf = FIG_DIR / "Figure_8.pdf"
    out_png = FIG_DIR / "Figure_8.png"

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


def main() -> None:
    make_two_panel_surface_rootzone()


if __name__ == "__main__":
    main()