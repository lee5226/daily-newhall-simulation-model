# utils_obs_bulk.py

import numpy as np
import pandas as pd


# USCRN soil moisture sensor depths, cm
DEFAULT_DEPTHS = [5.0, 10.0, 20.0, 50.0, 100.0]
DEFAULT_VALCOLS = ["sm5", "sm10", "sm20", "sm50", "sm100"]

# Surface layer requirement:
# 0-12.5 cm bulk observation is calculated when either sm5 or sm10 is valid.
SURFACE_CANDIDATES = ["sm5", "sm10"]

# Rootzone layer requirement:
# 0-100 cm bulk observation requires:
#   - sm5 as the upper-profile anchor
#   - sm100 as the lower-profile anchor
#   - at least one additional subsurface measurement at sm20 or sm50
#
# sm10 is not required as the additional subsurface constraint because it is
# close to the upper-profile anchor, but it is still included in the
# midpoint-based weighting when available.
ROOTZONE_REQUIRED = ["sm5", "sm100"]
ROOTZONE_INTERIOR = ["sm20", "sm50"]
MIN_ROOTZONE_INTERIOR = 1


def bulk_from_points(row, depths, valcols, z1, z2):
    """
    Compute thickness-weighted bulk VWC over [z1, z2] from available
    point-depth observations.

    Weights are determined from midpoints between available sensor depths
    and clipped to the target layer. Missing sensor values are ignored after
    the layer-specific minimum-depth requirements are checked.
    """
    pts = []

    for d, c in zip(depths, valcols):
        v = row.get(c, np.nan)
        if pd.notna(v):
            pts.append((float(d), float(v)))

    if not pts:
        return np.nan

    pts.sort(key=lambda x: x[0])

    weights = []

    for i, (z, val) in enumerate(pts):
        z_prev = pts[i - 1][0] if i > 0 else None
        z_next = pts[i + 1][0] if i < len(pts) - 1 else None

        mid_minus = 0.5 * (z_prev + z) if z_prev is not None else z1
        mid_plus = 0.5 * (z + z_next) if z_next is not None else z2

        a = max(mid_minus, z1)
        b = min(mid_plus, z2)
        w = max(0.0, b - a)

        if w > 0.0:
            weights.append((w, val))

    if not weights:
        return np.nan

    num = sum(w * v for w, v in weights)
    den = sum(w for w, _ in weights)

    return num / den if den > 0 else np.nan


def bulk_surface_from_obs(
    row,
    prefer_5cm=False,
    depths=DEFAULT_DEPTHS,
    valcols=DEFAULT_VALCOLS,
):
    """
    Compute surface soil moisture for 0-12.5 cm.

    Rules:
      - If prefer_5cm=True, use sm5 directly.
      - If prefer_5cm=False, require at least one valid measurement
        at sm5 or sm10.
      - Compute midpoint-based 0-12.5 cm bulk using all available
        valid depths.
    """
    if prefer_5cm:
        return row.get("sm5", np.nan)

    has_surface_depth = any(
        pd.notna(row.get(col, np.nan)) for col in SURFACE_CANDIDATES
    )

    if not has_surface_depth:
        return np.nan

    return bulk_from_points(row, depths, valcols, 0.0, 12.5)


def bulk_rootzone_from_obs(
    row,
    depths=DEFAULT_DEPTHS,
    valcols=DEFAULT_VALCOLS,
    min_interior=MIN_ROOTZONE_INTERIOR,
):
    """
    Compute rootzone soil moisture for 0-100 cm.

    Rules:
      - Require sm5 as the upper-profile anchor.
      - Require sm100 as the lower-profile anchor.
      - Require at least one additional subsurface measurement
        among sm20 and sm50.
      - Compute midpoint-based 0-100 cm bulk using all available
        valid depths, including sm10 when available.
    """
    for col in ROOTZONE_REQUIRED:
        if pd.isna(row.get(col, np.nan)):
            return np.nan

    n_interior = sum(
        pd.notna(row.get(col, np.nan)) for col in ROOTZONE_INTERIOR
    )

    if n_interior < min_interior:
        return np.nan

    return bulk_from_points(row, depths, valcols, 0.0, 100.0)


def add_bulk_columns(
    obs_df,
    prefer_5cm=False,
    depths=DEFAULT_DEPTHS,
    valcols=DEFAULT_VALCOLS,
    min_rootzone_interior=MIN_ROOTZONE_INTERIOR,
):
    """
    Add bulk soil moisture columns to a USCRN observation dataframe.

    Output columns:
      - sm_surf_bulk: surface soil moisture over 0-12.5 cm
      - sm_rz_bulk: rootzone soil moisture over 0-100 cm
    """
    df = obs_df.copy()

    # Ensure expected columns exist. Missing columns are created as NaN so that
    # requirement checks fail safely instead of raising unexpected errors.
    for col in valcols:
        if col not in df.columns:
            df[col] = np.nan

    for col in valcols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sm_surf_bulk"] = df.apply(
        lambda r: bulk_surface_from_obs(
            r,
            prefer_5cm=prefer_5cm,
            depths=depths,
            valcols=valcols,
        ),
        axis=1,
    )

    df["sm_rz_bulk"] = df.apply(
        lambda r: bulk_rootzone_from_obs(
            r,
            depths=depths,
            valcols=valcols,
            min_interior=min_rootzone_interior,
        ),
        axis=1,
    )

    return df