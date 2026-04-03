import numpy as np
import pandas as pd


def load_climate_data(filepath, p_gap_limit=1, t_gap_limit=3):
    """
    Load daily climate data and fill short gaps in precipitation and temperature.
    """
    df = pd.read_csv(filepath, parse_dates=["date"]).sort_values("date")

    for col in ["precipitation", "tmin", "tmax", "solar"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "precipitation" in df.columns:
        df.loc[df["precipitation"] < 0, "precipitation"] = np.nan

    for col in ["tmin", "tmax"]:
        if col in df.columns:
            df.loc[(df[col] < -80) | (df[col] > 60), col] = np.nan

    df["precip_missing_flag"] = df["precipitation"].isna().astype(int)
    df["tmin_missing_flag"] = df["tmin"].isna().astype(int)
    df["tmax_missing_flag"] = df["tmax"].isna().astype(int)

    if "precipitation" in df.columns:
        p0 = df["precipitation"].copy()
        p_interp = p0.interpolate(method="linear", limit=p_gap_limit, limit_direction="both")

        df["precip_interpolated_flag"] = (p0.isna() & p_interp.notna()).astype(int)

        long_gap = p_interp.isna()
        df["precip_longgap_flag"] = long_gap.astype(int)
        df["precipitation"] = p_interp.fillna(0.0)

        df.loc[df["precipitation"] > 300.0, "precipitation"] = 300.0

    for col in ["tmin", "tmax"]:
        if col in df.columns:
            s0 = df[col].copy()
            s1 = s0.interpolate(method="linear", limit=t_gap_limit, limit_direction="both")
            s2 = s1.fillna(s1.rolling(window=7, min_periods=1, center=True).mean())
            s3 = s2.ffill().bfill()

            df[col] = s3
            df[f"{col}_interpolated_flag"] = s0.isna().astype(int)

    if {"tmin", "tmax"}.issubset(df.columns):
        df["tavg"] = (df["tmin"] + df["tmax"]) / 2.0

    return df