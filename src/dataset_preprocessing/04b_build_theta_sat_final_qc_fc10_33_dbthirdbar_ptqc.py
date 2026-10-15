"""Calculate layer saturation for final_qc_fc10_33_dbthirdbar_ptqc soil inputs.

The density-adjusted Saxton and Rawls (2006) estimate is used directly.
No theta_sat = theta_fc + 0.02 floor is applied. Instead, the script requires
the raw density-adjusted theta_sat to be strictly greater than theta_fc in
every station-layer and saves the resulting theta_sat - theta_fc gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_fc10_33_dbthirdbar_ptqc"

SOIL_DIR = OUTPUTS_DIR / str(YEAR) / SOIL_TAG
INPUT_CSV = SOIL_DIR / f"layers_by_station_{YEAR}_{SOIL_TAG}.csv"
OUTPUT_CSV = SOIL_DIR / f"layers_by_station_{YEAR}_{SOIL_TAG}_satPTF_only.csv"

CLIP_US = (0.20, 0.65)
CLIP_DF = (0.90, 1.30)
TOL = 1.0e-12


def theta_sat_saxton_rawls(
    sand_pct: pd.Series,
    clay_pct: pd.Series,
    om_pct: pd.Series,
    bulk_density: pd.Series,
) -> dict[str, np.ndarray]:
    """Saxton and Rawls (2006) saturation estimate with density adjustment."""
    sand = np.asarray(sand_pct, dtype=float) / 100.0
    clay = np.asarray(clay_pct, dtype=float) / 100.0
    om = np.asarray(om_pct, dtype=float)
    rho_b = np.asarray(bulk_density, dtype=float)

    u33t = (
        -0.251 * sand
        + 0.195 * clay
        + 0.011 * om
        + 0.006 * sand * om
        - 0.027 * clay * om
        + 0.452 * sand * clay
        + 0.299
    )
    u33 = u33t + 1.283 * u33t**2 - 0.374 * u33t - 0.015

    u_s33t = (
        0.278 * sand
        + 0.034 * clay
        + 0.022 * om
        - 0.018 * sand * om
        - 0.027 * clay * om
        - 0.584 * sand * clay
        + 0.078
    )
    u_s33 = u_s33t + 0.636 * u_s33t - 0.107

    u_s = np.clip(u33 + u_s33 - 0.097 * sand + 0.043, *CLIP_US)
    normal_bulk_density = (1.0 - u_s) * 2.65
    density_factor = np.divide(
        rho_b,
        normal_bulk_density,
        out=np.full_like(normal_bulk_density, np.nan),
        where=normal_bulk_density > 0,
    )
    density_factor = np.clip(density_factor, *CLIP_DF)
    u_s_density_adjusted = np.clip(
        1.0 - normal_bulk_density * density_factor / 2.65,
        *CLIP_US,
    )
    return {
        "uS": u_s,
        "DF": density_factor,
        "uS_DF": u_s_density_adjusted,
    }


def main() -> None:
    layers = pd.read_csv(INPUT_CSV, dtype={"station_id": "string"})
    layers["station_id"] = layers["station_id"].str.zfill(5)

    metadata_columns = ["station", "station_id", "year", "mukey", "soil_tag"]
    missing_metadata = [column for column in metadata_columns if column not in layers]
    if missing_metadata:
        raise RuntimeError(f"Missing metadata columns: {missing_metadata}")
    if not layers["soil_tag"].eq(SOIL_TAG).all():
        observed_tags = sorted(layers["soil_tag"].dropna().astype(str).unique())
        raise RuntimeError(
            f"Input soil_tag does not match {SOIL_TAG!r}: {observed_tags}"
        )

    result = layers[metadata_columns].copy()

    required_columns: list[str] = []
    for k in range(1, 9):
        required_columns.extend(
            [f"sand_k{k}", f"clay_k{k}", f"om_k{k}", f"rhob_k{k}", f"fc_k{k}"]
        )
    missing_columns = [column for column in required_columns if column not in layers]
    if missing_columns:
        raise RuntimeError(f"Missing final soil-layer columns: {missing_columns}")
    if layers[required_columns].isna().any().any():
        bad = layers.loc[
            layers[required_columns].isna().any(axis=1), "station_id"
        ].tolist()
        raise RuntimeError(f"Missing inputs remain after final soil QC: {bad}")

    for k in range(1, 9):
        ptf = theta_sat_saxton_rawls(
            layers[f"sand_k{k}"],
            layers[f"clay_k{k}"],
            layers[f"om_k{k}"],
            layers[f"rhob_k{k}"],
        )
        fc = layers[f"fc_k{k}"].to_numpy(float)
        sat_ptf = np.asarray(ptf["uS"], dtype=float)
        sat_ptf_df = np.asarray(ptf["uS_DF"], dtype=float)

        # Raw estimates are final estimates in this version; no FC-based floor.
        result[f"fc_k{k}"] = fc
        result[f"sat_ptf_raw_k{k}"] = sat_ptf
        result[f"sat_ptf_k{k}"] = sat_ptf
        result[f"sat_ptf_fc_gap_k{k}"] = sat_ptf - fc
        result[f"sat_ptf_df_raw_k{k}"] = sat_ptf_df
        result[f"sat_ptf_df_k{k}"] = sat_ptf_df
        result[f"sat_ptf_df_fc_gap_k{k}"] = sat_ptf_df - fc
        result[f"DF_k{k}"] = ptf["DF"]

    sat_columns = [f"sat_ptf_df_k{k}" for k in range(1, 9)]
    fc_columns = [f"fc_k{k}" for k in range(1, 9)]
    sat_matrix = result[sat_columns].to_numpy(float)
    fc_matrix = result[fc_columns].to_numpy(float)
    gap_matrix = sat_matrix - fc_matrix

    if not np.isfinite(sat_matrix).all():
        raise RuntimeError("Non-finite saturation estimates were produced.")

    violation_rows, violation_layers = np.where(gap_matrix <= TOL)
    if violation_rows.size:
        violations = [
            {
                "station_id": str(result.iloc[row]["station_id"]),
                "layer": int(layer + 1),
                "theta_sat": float(sat_matrix[row, layer]),
                "theta_fc": float(fc_matrix[row, layer]),
                "gap": float(gap_matrix[row, layer]),
            }
            for row, layer in zip(violation_rows, violation_layers)
        ]
        raise RuntimeError(
            "Raw density-adjusted theta_sat must exceed theta_fc in every "
            f"station-layer. Violations: {violations}"
        )

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} (rows={len(result)})")
    print(
        "Density-adjusted theta_sat - theta_fc gap: "
        f"min={float(gap_matrix.min()):.6f}, "
        f"median={float(np.median(gap_matrix)):.6f}, "
        f"max={float(gap_matrix.max()):.6f}"
    )

    df_columns = [f"DF_k{k}" for k in range(1, 9)]
    df_matrix = result[df_columns].to_numpy(float)
    print(
        "Density-factor clipping: "
        f"lower={int(np.isclose(df_matrix, CLIP_DF[0], atol=TOL).sum())}, "
        f"upper={int(np.isclose(df_matrix, CLIP_DF[1], atol=TOL).sum())}"
    )


if __name__ == "__main__":
    main()
