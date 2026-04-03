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

BOX_WIDTH = 0.24
WHISKER_LINEWIDTH = 1.8
MEDIAN_LINEWIDTH = 2.2
SPINE_LINEWIDTH = 1.1

SHOW_ONLY = False

PEARSON_YLIM = (0.0, 1.0)
UBRMSE_YLIM = (0.0, 0.5)
BIAS_YLIM = (-1.0, 1.0)
MAE_YLIM = (0.0, 1.0)

PEARSON_TICKS = np.linspace(0, 1, 5)
UBRMSE_TICKS = np.linspace(0, 0.5, 6)
BIAS_TICKS = np.linspace(-1, 1, 5)
MAE_TICKS = np.linspace(0, 1, 5)

POSITIONS = {
    ("Monthly", "surface"): 0.86,
    ("Monthly", "rootzone"): 1.14,
    ("Daily", "surface"): 1.86,
    ("Daily", "rootzone"): 2.14,
}

BASE = OUTPUTS_DIR / str(YEAR)
METRICS_DIR = BASE / "metrics" / "faw"
FIG_DIR = BASE / "figures" / "faw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VERSION_MAP = {
    "Daily_FAWeom_pet_PP": "Daily",
    "Monthly_orig_FAW_pet_PP": "Monthly",
}
VERSION_ORDER = ["Monthly", "Daily"]
VERSION_DISPLAY = {
    "Monthly": "NSM",
    "Daily": "D-NSM",
}
DEPTH_ORDER = ["surface", "rootzone"]
METRICS = [
    ("pearson_r", "Pearson r"),
    ("ubrmse", "ubRMSE"),
    ("bias_signed", "Bias"),
    ("mae", "MAE"),
]

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


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


def main() -> None:
    df = pd.read_csv(METRICS_DIR / "faw_metrics_station_long.csv")

    df_eom = df[
        df["version"].isin(
            [
                "Daily_FAWeom_pet_PP",
                "Monthly_orig_FAW_pet_PP",
            ]
        )
    ].copy()
    df_eom["version_label"] = df_eom["version"].map(VERSION_MAP)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9))
    axes = axes.flatten()

    for ax, (metric_col, metric_label) in zip(axes, METRICS):
        data_arrays = []
        pos_arrays = []
        color_arrays = []

        for ver in VERSION_ORDER:
            for dep in DEPTH_ORDER:
                vals = df_eom.loc[
                    (df_eom["version_label"] == ver) & (df_eom["depth"] == dep),
                    metric_col,
                ].dropna().astype(float).values

                data_arrays.append(vals)
                pos_arrays.append(POSITIONS[(ver, dep)])
                color_arrays.append(COLOR_SURF if dep == "surface" else COLOR_RZ)

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
            ticks = PEARSON_TICKS
            y_min, y_max = PEARSON_YLIM
        elif metric_col == "ubrmse":
            ticks = UBRMSE_TICKS
            y_min, y_max = UBRMSE_YLIM
        elif metric_col == "bias_signed":
            ticks = BIAS_TICKS
            y_min, y_max = BIAS_YLIM
        else:
            ticks = MAE_TICKS
            y_min, y_max = MAE_YLIM

        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ticks)
        ax.set_yticklabels(format_ticks(ticks), fontsize=YTICK_FONTSIZE)

        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(
            [VERSION_DISPLAY[v] for v in VERSION_ORDER],
            fontsize=XTICK_FONTSIZE,
        )
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

    out_pdf = FIG_DIR / "Figure_4_boxplots.pdf"
    out_png = FIG_DIR / "Figure_4_boxplots.png"

    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()