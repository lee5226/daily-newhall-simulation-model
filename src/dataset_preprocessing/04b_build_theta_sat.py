from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
BASE = OUTPUTS_DIR / str(YEAR)
INP = BASE / f"layers_by_station_{YEAR}.csv"
OUT = BASE / f"layers_by_station_{YEAR}_satPTF_only.csv"

CLIP_US = (0.20, 0.65)
CLIP_DF = (0.90, 1.30)
EPS_SATFC = 0.02


def theta_sat_saxton_rawls(sand_pct, clay_pct, om_pct, bulk_density=None):
    """
    Estimate theta_sat using the Saxton and Rawls (2006) pedotransfer function.
    Inputs are percentages, and bulk density is optional.
    """
    sand = np.asarray(sand_pct, dtype=float) / 100.0
    clay = np.asarray(clay_pct, dtype=float) / 100.0
    om = np.asarray(om_pct, dtype=float) / 100.0

    u33t = (
        -0.251 * sand
        + 0.195 * clay
        + 0.011 * om
        + 0.006 * (sand * om)
        - 0.027 * (clay * om)
        + 0.452 * (sand * clay)
        + 0.299
    )
    u33 = u33t + (1.283 * (u33t**2) - 0.374 * u33t - 0.015)

    uS33t = (
        0.278 * sand
        + 0.034 * clay
        + 0.022 * om
        - 0.018 * (sand * om)
        - 0.027 * (clay * om)
        - 0.584 * (sand * clay)
        + 0.078
    )
    uS33 = uS33t + (0.636 * uS33t - 0.107)

    uS = u33 + uS33 - 0.097 * sand + 0.043
    uS = np.clip(uS, *CLIP_US)

    out = {"uS": uS, "DF": None, "uS_DF": None, "rN": (1.0 - uS) * 2.65}

    if bulk_density is not None:
        rb = np.asarray(bulk_density, dtype=float)
        rN = out["rN"]
        DF = np.divide(rb, rN, out=np.full_like(rN, np.nan), where=(rN > 0))
        DF = np.clip(DF, *CLIP_DF)

        uS_DF = 1.0 - (rN * DF) / 2.65
        uS_DF = np.clip(uS_DF, *CLIP_US)

        out.update({"DF": DF, "uS_DF": uS_DF})

    return out


def main() -> None:
    df = pd.read_csv(INP)
    res = df[["station", "station_id", "year"]].copy()

    for k in range(1, 9):
        sand = df.get(f"sand_k{k}")
        clay = df.get(f"clay_k{k}")
        om = df.get(f"om_k{k}")
        rhob = df.get(f"rhob_k{k}")
        fc = df.get(f"fc_k{k}")

        ptf = theta_sat_saxton_rawls(sand, clay, om, bulk_density=rhob)
        sat_ptf = pd.Series(ptf["uS"])
        sat_ptf_df = pd.Series(ptf["uS_DF"])
        DF_series = pd.Series(ptf["DF"])

        rhop_series = (
            df[f"rhop_k{k}"] if f"rhop_k{k}" in df.columns else pd.Series(2.65, index=df.index)
        ).fillna(2.65)

        sat_phys = 0.90 * (1.0 - (df[f"rhob_k{k}"] / rhop_series))
        sat_phys = np.clip(sat_phys, *CLIP_US)

        if fc is not None:
            th_fc = df[f"fc_k{k}"]
            sat_ptf = np.where(sat_ptf < th_fc + EPS_SATFC, th_fc + EPS_SATFC, sat_ptf)
            sat_ptf_df = np.where(sat_ptf_df < th_fc + EPS_SATFC, th_fc + EPS_SATFC, sat_ptf_df)
            sat_phys = np.where(sat_phys < th_fc + EPS_SATFC, th_fc + EPS_SATFC, sat_phys)

        res[f"sat_ptf_k{k}"] = sat_ptf
        res[f"sat_ptf_df_k{k}"] = sat_ptf_df
        res[f"sat_phys_k{k}"] = sat_phys
        res[f"DF_k{k}"] = DF_series

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"Saved: {OUT} (rows={len(res)})")


if __name__ == "__main__":
    main()