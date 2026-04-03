import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.utils.utils_obs_bulk import add_bulk_columns


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)

PET_TAGS = ["pet_PP"]
RUN_IDS = ["PP_spinup_layers_ptf_df_dual"]

PREFER_5CM_SURF = False

OBS_ROOT = OUTPUTS_DIR
STATION_CSV = BASE / f"aws_by_station_final_{YEAR}.csv"
THETA_PATH = BASE / f"theta_by_station_final_{YEAR}.csv"
MONTHLY_DIR = BASE / "monthly_newhall"

OUT_DIR = BASE / "figures" / "faw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STATIONS_FOR_FIG = ["12987", "53878", "03739"]

STATION_PANEL_TEXT = {
    "12987": "(a) Edinburg, TX",
    "53878": "(b) Asheville, NC",
    "03739": "(c) Cape Charles, VA",
}

COMBINED_FIG_BASENAME = "Figure_5_faw_timeseries_3stations"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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

PREVIEW_ONLY_ONE = False
SAVE_FIG = True
TARGET_WBAN = "12987"

STATION_LABEL_CFG = {
    "03739": {"x": 0.01, "y": 0.20, "ha": "left", "va": "top"},
    "53878": {"x": 0.01, "y": 0.20, "ha": "left", "va": "top"},
    "12987": {"x": 0.01, "y": 0.90, "ha": "left", "va": "top"},
}

DEFAULT_LABEL_CFG = {"x": 0.03, "y": 0.88, "ha": "left", "va": "top"}


def get_model_path(model_dir: Path, station: str, run_id: str) -> Path:
    fname = f"daily_results_{station}_{run_id}.csv"
    return model_dir / station / fname


def get_obs_path(year: int, station_obs: str) -> Path:
    return OBS_ROOT / str(year) / "per_station_sm" / f"obs_sm_{year}_{station_obs}.csv"


def load_station_info() -> pd.DataFrame:
    if not STATION_CSV.exists():
        raise FileNotFoundError(f"Missing station file: {STATION_CSV}")

    station_info = pd.read_csv(STATION_CSV, usecols=["station_id", "year"]).drop_duplicates()
    station_info["station_id"] = station_info["station_id"].astype(str).str.zfill(5)
    station_info["year"] = station_info["year"].astype(int)
    return station_info


def add_faw_columns(obs_df: pd.DataFrame, mod_df: pd.DataFrame, theta_row: pd.Series) -> None:
    th_wp_sf = theta_row["surface_theta_wp"]
    th_fc_sf = theta_row["surface_theta_fc"]
    th_wp_rz = theta_row["rootzone_theta_wp"]
    th_fc_rz = theta_row["rootzone_theta_fc"]

    obs_df["FAW_surf_obs"] = (obs_df["sm_surf_bulk"] - th_wp_sf) / (th_fc_sf - th_wp_sf)
    obs_df["FAW_rz_obs"] = (obs_df["sm_rz_bulk"] - th_wp_rz) / (th_fc_rz - th_wp_rz)

    mod_df["FAW_surf_mod"] = (mod_df["VWC_0_12_5cm"] - th_wp_sf) / (th_fc_sf - th_wp_sf)
    mod_df["FAW_rz_mod"] = (mod_df["VWC_0_100cm"] - th_wp_rz) / (th_fc_rz - th_wp_rz)

    for col in ["FAW_surf_obs", "FAW_rz_obs"]:
        if col in obs_df.columns:
            obs_df[col] = obs_df[col].clip(0, 1)

    for col in ["FAW_surf_mod", "FAW_rz_mod"]:
        if col in mod_df.columns:
            mod_df[col] = mod_df[col].clip(0, 1)


def plot_faw_timeseries_by_station() -> None:
    theta_df = pd.read_csv(THETA_PATH)

    pet_tag = PET_TAGS[0]
    model_dir = BASE / "daily_newhall" / pet_tag
    if not model_dir.exists():
        model_dir = BASE / "daily_newhall"

    station_info = load_station_info()
    preview_done = False

    for _, row in station_info.iterrows():
        station_id_num = int(row["station_id"])
        station_obs = row["station_id"]
        station_model = row["station_id"].lstrip("0")
        year = int(row["year"])

        if TARGET_WBAN is not None and station_obs != TARGET_WBAN:
            continue

        theta_row = theta_df.loc[theta_df["station_id"] == station_id_num]
        if theta_row.empty:
            continue
        theta_row = theta_row.iloc[0]

        obs_path = get_obs_path(year, station_obs)
        if not obs_path.exists():
            continue

        obs = pd.read_csv(obs_path, parse_dates=["date"])
        obs = add_bulk_columns(obs, prefer_5cm=PREFER_5CM_SURF)
        obs = obs[obs["date"].dt.year == year].copy()
        if obs.empty:
            continue

        model_path = get_model_path(model_dir, station_model, RUN_IDS[0])
        if not model_path.exists():
            continue

        mod = pd.read_csv(model_path, parse_dates=["date"])
        mod = mod[mod["date"].dt.year == year].copy()
        if mod.empty:
            continue

        monthly_path = MONTHLY_DIR / station_model / f"monthly_results_{station_model}.csv"
        monthly_df = None
        if monthly_path.exists():
            monthly_df = pd.read_csv(monthly_path)
            monthly_df["date"] = pd.to_datetime(
                {
                    "year": [year] * len(monthly_df),
                    "month": monthly_df["month"].values,
                    "day": [1] * len(monthly_df),
                }
            ) + pd.offsets.MonthEnd(0)

        add_faw_columns(obs, mod, theta_row)

        obs = obs.sort_values("date")
        mod = mod.sort_values("date")
        if monthly_df is not None:
            monthly_df = monthly_df.sort_values("date")

        label_cfg = STATION_LABEL_CFG.get(station_obs, DEFAULT_LABEL_CFG)

        fig, axes = plt.subplots(2, 1, figsize=(13, 3.8), sharex=True, constrained_layout=True)
        ax_sf, ax_rz = axes

        ax_sf.plot(
            obs["date"],
            obs["FAW_surf_obs"],
            color=COLOR_OBS,
            linestyle="-",
            linewidth=LW_OBS,
            zorder=3,
        )
        ax_sf.plot(
            mod["date"],
            mod["FAW_surf_mod"],
            color=COLOR_SURF,
            linestyle="--",
            linewidth=LW_DAILY,
            zorder=3,
        )

        if monthly_df is not None and "FAW_0_12_5cm" in monthly_df.columns:
            ax_sf.step(
                monthly_df["date"],
                monthly_df["FAW_0_12_5cm"],
                where="post",
                color=COLOR_SURF,
                linewidth=LW_MONTHLY,
                zorder=3,
            )
            ax_sf.plot(
                monthly_df["date"],
                monthly_df["FAW_0_12_5cm"],
                linestyle="None",
                marker="s",
                markersize=MS_MONTHLY,
                color=COLOR_SURF,
                zorder=3,
            )

        ax_sf.set_ylim(-0.05, 1.05)
        ax_sf.set_yticks([0, 0.5, 1.0])
        ax_sf.grid(False)
        ax_sf.tick_params(axis="y", labelsize=FS_TICK)
        ax_sf.text(
            label_cfg["x"],
            label_cfg["y"],
            "Surface",
            transform=ax_sf.transAxes,
            fontsize=FS_INPANEL,
            ha=label_cfg["ha"],
            va=label_cfg["va"],
        )

        ax_rz.plot(
            obs["date"],
            obs["FAW_rz_obs"],
            color=COLOR_OBS,
            linestyle="-",
            linewidth=LW_OBS,
            zorder=3,
        )
        ax_rz.plot(
            mod["date"],
            mod["FAW_rz_mod"],
            color=COLOR_RZ,
            linestyle="--",
            linewidth=LW_DAILY,
            zorder=3,
        )

        if monthly_df is not None and "FAW_0_100cm" in monthly_df.columns:
            ax_rz.step(
                monthly_df["date"],
                monthly_df["FAW_0_100cm"],
                where="post",
                color=COLOR_RZ,
                linewidth=LW_MONTHLY,
                zorder=3,
            )
            ax_rz.plot(
                monthly_df["date"],
                monthly_df["FAW_0_100cm"],
                linestyle="None",
                marker="s",
                markersize=MS_MONTHLY,
                color=COLOR_RZ,
                zorder=3,
            )

        ax_rz.set_ylim(-0.05, 1.05)
        ax_rz.set_yticks([0, 0.5, 1.0])
        ax_rz.grid(False)
        ax_rz.tick_params(axis="x", labelsize=FS_DATE)
        ax_rz.tick_params(axis="y", labelsize=FS_TICK)
        ax_rz.text(
            label_cfg["x"],
            label_cfg["y"],
            "Rootzone",
            transform=ax_rz.transAxes,
            fontsize=FS_INPANEL,
            ha=label_cfg["ha"],
            va=label_cfg["va"],
        )

        fig.supylabel("FAW", fontsize=FS_LABEL, x=-0.02)
        ax_rz.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax_rz.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        legend_handles = [
            Line2D([0], [0], color="black", lw=LW_OBS, linestyle="-", label="Observation"),
            Line2D([0], [0], color="black", lw=LW_DAILY, linestyle="--", label="D-NSM"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=LW_MONTHLY,
                linestyle="-",
                marker="s",
                markersize=MS_MONTHLY,
                label="NSM",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
            frameon=False,
            fontsize=FS_INPANEL,
            handlelength=2.6,
            columnspacing=1.6,
        )
        fig.subplots_adjust(top=0.82)

        save_path = OUT_DIR / f"FAW_timeseries_station_{station_id_num}.png"

        if PREVIEW_ONLY_ONE:
            plt.show()
            plt.close()
            preview_done = True
            break

        if SAVE_FIG:
            plt.savefig(save_path, dpi=350, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        plt.close()

    if PREVIEW_ONLY_ONE and not preview_done:
        print("No station was previewed. Check TARGET_WBAN or input files.")


def plot_faw_timeseries_three_stations_vertical() -> None:
    import matplotlib.gridspec as gridspec

    theta_df = pd.read_csv(THETA_PATH)

    pet_tag = PET_TAGS[0]
    model_dir = BASE / "daily_newhall" / pet_tag
    if not model_dir.exists():
        model_dir = BASE / "daily_newhall"

    station_info = load_station_info()
    station_order = TARGET_STATIONS_FOR_FIG

    fig = plt.figure(figsize=(13, 12.2))
    outer = gridspec.GridSpec(
        8,
        1,
        height_ratios=[1, 1, 0.50, 1, 1, 0.50, 1, 1],
        hspace=0.10,
    )

    legend_handles = [
        Line2D([0], [0], color="black", lw=LW_OBS, linestyle="-", label="Observation"),
        Line2D([0], [0], color="black", lw=LW_DAILY, linestyle="--", label="D-NSM"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=LW_MONTHLY,
            linestyle="-",
            marker="s",
            markersize=MS_MONTHLY,
            label="NSM",
        ),
    ]

    found_count = 0
    axis_pairs = []

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

    for s_idx, target_station in enumerate(station_order):
        row_match = station_info.loc[station_info["station_id"] == str(target_station).zfill(5)]
        if row_match.empty:
            print(f"Missing station row: {target_station}")
            continue

        row = row_match.iloc[0]
        station_id_num = int(row["station_id"])
        station_obs = row["station_id"]
        station_model = row["station_id"].lstrip("0")
        year = int(row["year"])

        theta_row = theta_df.loc[theta_df["station_id"] == station_id_num]
        if theta_row.empty:
            print(f"Missing theta row: {target_station}")
            continue
        theta_row = theta_row.iloc[0]

        obs_path = get_obs_path(year, station_obs)
        if not obs_path.exists():
            print(f"Missing observation file: {target_station}")
            continue

        obs = pd.read_csv(obs_path, parse_dates=["date"])
        obs = add_bulk_columns(obs, prefer_5cm=PREFER_5CM_SURF)
        obs = obs[obs["date"].dt.year == year].copy()
        if obs.empty:
            print(f"Empty observation data: {target_station}")
            continue

        model_path = get_model_path(model_dir, station_model, RUN_IDS[0])
        if not model_path.exists():
            print(f"Missing model file: {target_station}")
            continue

        mod = pd.read_csv(model_path, parse_dates=["date"])
        mod = mod[mod["date"].dt.year == year].copy()
        if mod.empty:
            print(f"Empty model data: {target_station}")
            continue

        monthly_path = MONTHLY_DIR / station_model / f"monthly_results_{station_model}.csv"
        monthly_df = None
        if monthly_path.exists():
            monthly_df = pd.read_csv(monthly_path)
            monthly_df["date"] = pd.to_datetime(
                {
                    "year": [year] * len(monthly_df),
                    "month": monthly_df["month"].values,
                    "day": [1] * len(monthly_df),
                }
            ) + pd.offsets.MonthEnd(0)

        add_faw_columns(obs, mod, theta_row)

        obs = obs.sort_values("date")
        mod = mod.sort_values("date")
        if monthly_df is not None:
            monthly_df = monthly_df.sort_values("date")

        base_label_cfg = STATION_LABEL_CFG.get(station_obs, DEFAULT_LABEL_CFG)
        label_y = panel_text_y.get(station_obs, base_label_cfg["y"])

        if s_idx == 0:
            ax_sf = fig.add_subplot(outer[0])
            ax_rz = fig.add_subplot(outer[1], sharex=ax_sf)
        elif s_idx == 1:
            ax_sf = fig.add_subplot(outer[3])
            ax_rz = fig.add_subplot(outer[4], sharex=ax_sf)
        else:
            ax_sf = fig.add_subplot(outer[6])
            ax_rz = fig.add_subplot(outer[7], sharex=ax_sf)

        axis_pairs.append((ax_sf, ax_rz))

        ax_sf.plot(
            obs["date"],
            obs["FAW_surf_obs"],
            color=COLOR_OBS,
            linestyle="-",
            linewidth=LW_OBS,
            zorder=3,
        )
        ax_sf.plot(
            mod["date"],
            mod["FAW_surf_mod"],
            color=COLOR_SURF,
            linestyle="--",
            linewidth=LW_DAILY,
            zorder=3,
        )

        if monthly_df is not None and "FAW_0_12_5cm" in monthly_df.columns:
            ax_sf.step(
                monthly_df["date"],
                monthly_df["FAW_0_12_5cm"],
                where="post",
                color=COLOR_SURF,
                linewidth=LW_MONTHLY,
                zorder=3,
            )
            ax_sf.plot(
                monthly_df["date"],
                monthly_df["FAW_0_12_5cm"],
                linestyle="None",
                marker="s",
                markersize=MS_MONTHLY,
                color=COLOR_SURF,
                zorder=3,
            )

        ax_sf.set_ylim(-0.05, 1.05)
        ax_sf.set_yticks([0, 0.5, 1.0])
        ax_sf.grid(False)
        ax_sf.tick_params(axis="y", labelsize=FS_TICK)
        ax_sf.tick_params(axis="x", labelbottom=False)
        ax_sf.text(
            base_label_cfg["x"],
            label_y,
            "Surface",
            transform=ax_sf.transAxes,
            fontsize=FS_INPANEL,
            ha=base_label_cfg["ha"],
            va=base_label_cfg["va"],
        )

        ax_rz.plot(
            obs["date"],
            obs["FAW_rz_obs"],
            color=COLOR_OBS,
            linestyle="-",
            linewidth=LW_OBS,
            zorder=3,
        )
        ax_rz.plot(
            mod["date"],
            mod["FAW_rz_mod"],
            color=COLOR_RZ,
            linestyle="--",
            linewidth=LW_DAILY,
            zorder=3,
        )

        if monthly_df is not None and "FAW_0_100cm" in monthly_df.columns:
            ax_rz.step(
                monthly_df["date"],
                monthly_df["FAW_0_100cm"],
                where="post",
                color=COLOR_RZ,
                linewidth=LW_MONTHLY,
                zorder=3,
            )
            ax_rz.plot(
                monthly_df["date"],
                monthly_df["FAW_0_100cm"],
                linestyle="None",
                marker="s",
                markersize=MS_MONTHLY,
                color=COLOR_RZ,
                zorder=3,
            )

        ax_rz.set_ylim(-0.05, 1.05)
        ax_rz.set_yticks([0, 0.5, 1.0])
        ax_rz.grid(False)
        ax_rz.tick_params(axis="y", labelsize=FS_TICK)
        ax_rz.tick_params(axis="x", labelsize=FS_DATE)
        ax_rz.text(
            base_label_cfg["x"],
            label_y,
            "Rootzone",
            transform=ax_rz.transAxes,
            fontsize=FS_INPANEL,
            ha=base_label_cfg["ha"],
            va=base_label_cfg["va"],
        )

        ax_rz.text(
            0.5,
            station_label_y.get(station_obs, -0.24),
            STATION_PANEL_TEXT.get(station_obs, station_obs),
            transform=ax_rz.transAxes,
            fontsize=FS_LABEL,
            ha="center",
            va="top",
        )

        ax_rz.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_rz.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        found_count += 1

    if found_count == 0:
        print("No stations were available for plotting.")
        plt.close(fig)
        return

    fig.legend(
        handles=legend_handles,
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

    for ax_sf, ax_rz in axis_pairs:
        y_mid = (ax_sf.get_position().y0 + ax_rz.get_position().y1) / 2
        fig.text(
            0.04,
            y_mid,
            "FAW",
            rotation=90,
            va="center",
            ha="center",
            fontsize=FS_LABEL,
        )

    out_pdf = OUT_DIR / f"{COMBINED_FIG_BASENAME}.pdf"
    out_png = OUT_DIR / f"{COMBINED_FIG_BASENAME}.png"

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
    plot_faw_timeseries_three_stations_vertical()


if __name__ == "__main__":
    main()