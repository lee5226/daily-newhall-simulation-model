import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016

FONT_FAMILY = "Arial"
TITLE_FONTSIZE = 24
XTICK_FONTSIZE = 20
YTICK_FONTSIZE = 20
LEGEND_FONTSIZE = 24

COLOR_SURF = "#3E7F6E"
COLOR_RZ = "#845B53"

BOX_WIDTH = 0.14
WHISKER_LINEWIDTH = 2.0
MEDIAN_LINEWIDTH = 2.5
SPINE_LINEWIDTH = 1.3

SHOW_ONLY = False
SAVE_BASENAME = "Figure_6_boxplots"

PEARSON_YLIM = (0.5, 1.0)
PEARSON_TICKS = np.arange(0.5, 1.01, 0.1)

UBRMSE_YLIM = (0.0, 0.10)
UBRMSE_TICKS = np.arange(0.0, 0.101, 0.02)

BIAS_YLIM = (-0.4, 0.2)
BIAS_TICKS = np.arange(-0.4, 0.201, 0.2)

MAE_YLIM = (0.0, 0.20)
MAE_TICKS = np.arange(0.0, 0.201, 0.05)

BASE = OUTPUTS_DIR / str(YEAR)
PET_TAG = "pet_PP"
RUN_IDS = ["PP_spinup_layers_ptf_df_dual"]

FIG_DIR = BASE / "figures" / "vwc"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def format_ticks(vals):
    labels = []
    for v in vals:
        if np.isclose(v, round(v)):
            labels.append(f"{int(round(v))}")
        elif np.isclose(v * 10, round(v * 10)):
            labels.append(f"{v:.1f}")
        else:
            labels.append(f"{v:.2f}")
    return labels


def resolve_input_csv() -> Path:
    run_slug = "__".join(slugify(r) for r in RUN_IDS)
    fname = f"station_metrics_long_{YEAR}_{run_slug}.csv"

    preferred = BASE / "metrics" / "vwc" / fname
    legacy = BASE / "metrics" / PET_TAG / fname

    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "VWC metrics file not found.\n"
        f"Checked:\n"
        f"  - {preferred}\n"
        f"  - {legacy}"
    )


def main() -> None:
    input_csv = resolve_input_csv()
    df = pd.read_csv(input_csv)

    df_plot = df[df["season"] == "ALL"].copy()
    if df_plot.empty:
        raise ValueError(f"No rows found with season == 'ALL' in {input_csv.name}")

    depth_map = {
        "surface": "Surface",
        "rootzone": "Rootzone",
    }
    df_plot["depth_label"] = df_plot["depth"].map(depth_map)

    metrics = [
        ("pearson_r", "Pearson r"),
        ("ubrmse", "ubRMSE"),
        ("bias_signed", "Bias"),
        ("mae", "MAE"),
    ]

    positions = {
        "Surface": 0.86,
        "Rootzone": 1.14,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9))
    axes = axes.flatten()

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        surf_vals = (
            df_plot.loc[df_plot["depth_label"] == "Surface", metric_col]
            .dropna()
            .astype(float)
            .values
        )
        rz_vals = (
            df_plot.loc[df_plot["depth_label"] == "Rootzone", metric_col]
            .dropna()
            .astype(float)
            .values
        )

        data_arrays = [surf_vals, rz_vals]
        pos_arrays = [positions["Surface"], positions["Rootzone"]]
        color_arrays = [COLOR_SURF, COLOR_RZ]

        bp = ax.boxplot(
            data_arrays,
            positions=pos_arrays,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            boxprops=dict(linewidth=0, edgecolor="none"),
            whiskerprops=dict(linewidth=WHISKER_LINEWIDTH),
            capprops=dict(linewidth=WHISKER_LINEWIDTH),
            medianprops=dict(color="white", linewidth=MEDIAN_LINEWIDTH),
        )

        for i, patch in enumerate(bp["boxes"]):
            fc = color_arrays[i]
            patch.set_facecolor(fc)
            patch.set_edgecolor("none")
            patch.set_linewidth(0)

            bp["whiskers"][2 * i].set_color(fc)
            bp["whiskers"][2 * i + 1].set_color(fc)
            bp["whiskers"][2 * i].set_linewidth(WHISKER_LINEWIDTH)
            bp["whiskers"][2 * i + 1].set_linewidth(WHISKER_LINEWIDTH)

            bp["caps"][2 * i].set_color(fc)
            bp["caps"][2 * i + 1].set_color(fc)
            bp["caps"][2 * i].set_linewidth(WHISKER_LINEWIDTH)
            bp["caps"][2 * i + 1].set_linewidth(WHISKER_LINEWIDTH)

            bp["medians"][i].set_color("white")
            bp["medians"][i].set_linewidth(MEDIAN_LINEWIDTH)

        if metric_col == "pearson_r":
            y_min, y_max = PEARSON_YLIM
            ticks = PEARSON_TICKS
        elif metric_col == "ubrmse":
            y_min, y_max = UBRMSE_YLIM
            ticks = UBRMSE_TICKS
        elif metric_col == "bias_signed":
            y_min, y_max = BIAS_YLIM
            ticks = BIAS_TICKS
        else:
            y_min, y_max = MAE_YLIM
            ticks = MAE_TICKS

        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ticks)
        ax.set_yticklabels(format_ticks(ticks), fontsize=YTICK_FONTSIZE)

        ax.set_xlim(0.7, 1.3)
        ax.set_xticks([])
        ax.set_title(metric_label, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

        ax.tick_params(axis="x", length=0, labelsize=XTICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(SPINE_LINEWIDTH)

    legend_handles = [
        Patch(facecolor=COLOR_SURF, edgecolor="none", label="Surface"),
        Patch(facecolor=COLOR_RZ, edgecolor="none", label="Rootzone"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if SHOW_ONLY:
        plt.show()
        return

    out_pdf = FIG_DIR / f"{SAVE_BASENAME}.pdf"
    out_png = FIG_DIR / f"{SAVE_BASENAME}.png"

    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()