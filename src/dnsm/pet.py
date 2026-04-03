import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils.paths import DATA_REFERENCE_DIR
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.paths import DATA_REFERENCE_DIR


CONSTANTS_CSV = DATA_REFERENCE_DIR / "constants_classic.csv"
constants_df = pd.read_csv(CONSTANTS_CSV)


def get_pe_value(tavg):
    """
    Return lookup-table PE values for high temperatures.
    """
    return np.interp(
        tavg,
        constants_df["temp_bins"].dropna(),
        constants_df["pe_bins"].dropna(),
    )


def calculate_annual_heat_index(monthly_tavg):
    """
    Calculate the annual Thornthwaite heat index from monthly mean temperature.
    """
    mt = np.asarray(monthly_tavg, dtype=float)
    mt = np.clip(mt, 0, None)
    I_m = (mt / 5.0) ** 1.514
    I_m[~np.isfinite(I_m)] = 0.0
    return float(I_m.sum())


def daylength_hours(latitude, date):
    """
    Return daylength in hours for a single date.
    """
    phi = np.radians(latitude)
    doy = date.timetuple().tm_yday
    delta = np.radians(23.44) * np.sin(np.radians((360.0 / 365.0) * (doy - 81)))
    x = -np.tan(phi) * np.tan(delta)
    x = np.clip(x, -1.0, 1.0)
    ws = np.arccos(x)
    return (24.0 / np.pi) * ws


def thornthwaite_raw_monthly(Tm, I):
    """
    Return uncorrected monthly Thornthwaite PET.
    """
    if I <= 0 or Tm <= 0:
        return 0.0
    if Tm >= 38.0:
        return 185.0
    if Tm > 26.5:
        return float(get_pe_value(Tm))

    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
    return 16.0 * ((10.0 * Tm / I) ** a)


def thornthwaite_weight_daily(Td, I):
    """
    Return daily Thornthwaite weighting values.
    """
    if I <= 0 or Td <= 0:
        return 0.0
    if Td >= 38.0:
        return 185.0
    if Td > 26.5:
        return float(get_pe_value(Td))

    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
    return 16.0 * ((10.0 * Td / I) ** a)


def prepare_E0_raw_by_month(monthly_tavg, I):
    """
    Prepare uncorrected monthly Thornthwaite PET anchors.
    """
    mt = np.asarray(monthly_tavg, dtype=float)
    mt = np.clip(mt, 0, None)
    return np.array([thornthwaite_raw_monthly(Tm, I) for Tm in mt])


def _petA_core(E0m, Ld):
    return E0m * (Ld / 12.0) * (1.0 / 30.0)


def calculate_daily_pet_A(date, latitude, E0_raw_by_month):
    """
    Return daily PET using method A.
    """
    m = date.month - 1
    Ld = daylength_hours(latitude, date)
    return _petA_core(E0_raw_by_month[m], Ld)


def daylength_hours_vec(latitude: float, dates: pd.Series) -> np.ndarray:
    """
    Return vectorized daylength in hours.
    """
    phi = np.radians(latitude)
    doy = dates.dt.dayofyear.to_numpy()
    delta = np.radians(23.44) * np.sin(np.radians((360.0 / 365.0) * (doy - 81)))
    x = -np.tan(phi) * np.tan(delta)
    x = np.clip(x, -1.0, 1.0)
    ws = np.arccos(x)
    return (24.0 / np.pi) * ws


def pet_series_A(daily_df: pd.DataFrame, latitude: float, E0_raw_by_month: np.ndarray) -> pd.Series:
    """
    Return daily PET series using method A.
    """
    m = daily_df["date"].dt.month.values - 1
    Ld = daylength_hours_vec(latitude, daily_df["date"])
    pet = E0_raw_by_month[m] * (Ld / 12.0) * (1.0 / 30.0)
    return pd.Series(pet, index=daily_df.index, name="PET_A")


def pet_series_B(
    daily_df: pd.DataFrame,
    latitude: float,
    I: float,
    E0_raw_by_month: np.ndarray,
) -> pd.Series:
    """
    Return daily PET series using method B.
    """
    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    df["Ld"] = daylength_hours_vec(latitude, df["date"])
    df["w"] = df["tavg"].apply(lambda T: thornthwaite_weight_daily(T, I)) * df["Ld"]

    Ld_sum_m = df.groupby("month")["Ld"].transform("sum")
    W_sum_m = df.groupby("month")["w"].transform("sum")
    e0_map = {m + 1: E0_raw_by_month[m] for m in range(12)}

    df["E0m"] = df["month"].map(e0_map)
    df["PET_target_m"] = df["E0m"] * (Ld_sum_m / (12.0 * 30.0))

    petB = np.where(
        (W_sum_m > 0) & (Ld_sum_m > 0),
        df["PET_target_m"] * (df["w"] / W_sum_m),
        df["E0m"] * (df["Ld"] / 12.0) * (1.0 / 30.0),
    )
    return pd.Series(petB, index=df.index, name="PET_B")


def _willmott_poly(T):
    return -415.85 + 32.24 * T - 0.43 * (T**2)


def _ETM_from_T_PP(T, I, a):
    """
    Return monthly-equivalent PET weights for PP-based methods.
    """
    T = np.asarray(T, float)
    ETM = np.zeros_like(T)
    cool = (T > 0) & (T <= 26.5)
    hot = T > 26.5

    ETM[cool] = 16.0 * np.power((10.0 * T[cool] / I), a)
    ETM[hot] = _willmott_poly(T[hot])
    return ETM


def pet_series_TEF(daily_df: pd.DataFrame, latitude: float, I: float, k: float = 0.69) -> pd.Series:
    """
    Return PP-based PET using effective temperature.
    """
    tavg = daily_df["tavg"].to_numpy()
    tmin = daily_df["tmin"].to_numpy()
    tmax = daily_df["tmax"].to_numpy()
    N = daylength_hours_vec(latitude, daily_df["date"])
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239

    Tef = k * (tavg + (tmax - tmin))
    ETM = _ETM_from_T_PP(Tef, I, a)
    PET = ETM * (N / 360.0)
    return pd.Series(PET, index=daily_df.index, name="PET_PP_TEF")


def pet_series_TEFSTAR(
    daily_df: pd.DataFrame,
    latitude: float,
    I: float,
    k: float = 0.69,
    clip_star: bool = True,
) -> pd.Series:
    """
    Return PP-based PET using effective temperature with day/night correction.
    """
    tavg = daily_df["tavg"].to_numpy()
    tmin = daily_df["tmin"].to_numpy()
    tmax = daily_df["tmax"].to_numpy()
    N = daylength_hours_vec(latitude, daily_df["date"])
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239

    Tef = k * (tavg + (tmax - tmin))
    Tstar = Tef * (N / np.maximum(24.0 - N, 1e-6))
    if clip_star:
        Tstar = np.clip(Tstar, tavg, tmax)

    ETM = _ETM_from_T_PP(Tstar, I, a)
    PET = ETM * (N / 360.0)
    return pd.Series(PET, index=daily_df.index, name="PET_PP_TEFSTAR")


def ra_extraterrestrial_vec(latitude: float, dates: pd.Series) -> np.ndarray:
    """
    Return extraterrestrial radiation for each date.
    """
    phi = np.radians(latitude)
    doy = dates.dt.dayofyear.to_numpy()
    delta = np.radians(23.44) * np.sin(np.radians((360.0 / 365.0) * (doy - 81)))
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    Gsc = 0.0820
    RA = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    return RA


def pet_series_HS(daily_df: pd.DataFrame, latitude: float) -> pd.Series:
    """
    Return daily PET using Hargreaves-Samani.
    """
    tmax = daily_df["tmax"].to_numpy()
    tmin = daily_df["tmin"].to_numpy()
    tavg = daily_df["tavg"].to_numpy()
    RA = ra_extraterrestrial_vec(latitude, daily_df["date"])
    TD = np.maximum(tmax - tmin, 0.0)

    PET = 0.0023 * RA * np.sqrt(TD) * (tavg + 17.8)
    PET[np.isnan(PET)] = 0.0
    return pd.Series(PET, index=daily_df.index, name="PET_HS")


def pet_series_PT(
    daily_df: pd.DataFrame,
    latitude: float,
    elevation_m: float = 0.0,
    albedo: float = 0.23,
    alpha_PT: float = 1.26,
) -> pd.Series:
    """
    Return daily PET using Priestley-Taylor.
    """
    tmax = daily_df["tmax"].to_numpy(dtype=float)
    tmin = daily_df["tmin"].to_numpy(dtype=float)
    tavg = daily_df["tavg"].to_numpy(dtype=float)
    rs = daily_df["solar"].to_numpy(dtype=float)

    if np.nanmedian(rs) > 70:
        rs = rs * 0.0864

    RA = ra_extraterrestrial_vec(latitude, daily_df["date"])
    Rso = (0.75 + 2e-5 * elevation_m) * RA
    Rns = (1.0 - albedo) * np.maximum(rs, 0.0)

    sigma = 4.903e-9
    tmaxK = tmax + 273.16
    tminK = tmin + 273.16

    def es(T):
        return 0.6108 * np.exp(17.27 * T / (T + 237.3))

    ea = es(tmin)
    Rs_Rso = np.clip(rs / np.maximum(Rso, 1e-6), 0.3, 1.0)
    Rnl = sigma * ((tmaxK**4 + tminK**4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (
        1.35 * Rs_Rso - 0.35
    )

    Rn = np.clip(Rns - np.maximum(Rnl, 0.0), 0.0, None)

    Delta = 4098 * es(tavg) / (tavg + 237.3) ** 2
    P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * P

    lambda_MJ = 2.45
    PET = alpha_PT * (Delta / (Delta + gamma)) * (Rn / lambda_MJ)
    PET[(tavg <= 0) | ~np.isfinite(PET)] = 0.0

    return pd.Series(PET, index=daily_df.index, name="PET_PT")


def prepare_E0_camargo_anchor(
    monthly_tavg: np.ndarray,
    monthly_tmax: np.ndarray,
    monthly_tmin: np.ndarray,
    I: float,
    k_mon: float = 0.72,
    use_willmott: bool = True,
) -> np.ndarray:
    """
    Return monthly Camargo PET anchors.
    """
    Tef_m = k_mon * (
        np.asarray(monthly_tavg, float) + (np.asarray(monthly_tmax, float) - np.asarray(monthly_tmin, float))
    )
    Tef_m = np.clip(Tef_m, 0.0, None)

    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239

    if use_willmott:
        E0 = _ETM_from_T_PP(Tef_m, I, a)
    else:
        cool = (Tef_m > 0) & (Tef_m <= 26.5)
        hot1 = (Tef_m > 26.5) & (Tef_m < 38.0)
        hot2 = Tef_m >= 38.0

        E0 = np.zeros_like(Tef_m, float)
        E0[cool] = 16.0 * np.power((10.0 * Tef_m[cool] / I), a)
        E0[hot1] = get_pe_value(Tef_m[hot1])
        E0[hot2] = 185.0

    return E0


def prepare_E0_camargo_anchor_from_daily(
    daily_df: pd.DataFrame,
    I: float,
    k_mon: float = 0.72,
    use_willmott: bool = True,
) -> np.ndarray:
    """
    Aggregate daily temperature to monthly values and return Camargo PET anchors.
    """
    grp = daily_df.groupby(daily_df["date"].dt.month)
    tavg_m = grp["tavg"].mean().reindex(range(1, 13), fill_value=0).to_numpy()
    tmax_m = grp["tmax"].mean().reindex(range(1, 13), fill_value=0).to_numpy()
    tmin_m = grp["tmin"].mean().reindex(range(1, 13), fill_value=0).to_numpy()

    return prepare_E0_camargo_anchor(
        tavg_m,
        tmax_m,
        tmin_m,
        I,
        k_mon=k_mon,
        use_willmott=use_willmott,
    )


def pet_series_B_camargo(
    daily_df: pd.DataFrame,
    latitude: float,
    I: float,
    E0_raw_by_month: np.ndarray,
    k_tef: float = 0.69,
) -> pd.Series:
    """
    Return Camargo method B PET using T_eff daily weighting.
    """
    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    df["Ld"] = daylength_hours_vec(latitude, df["date"])

    tavg = df["tavg"].to_numpy(float)
    tmax = df["tmax"].to_numpy(float)
    tmin = df["tmin"].to_numpy(float)

    Tef = k_tef * (tavg + (tmax - tmin))
    Tef = np.clip(Tef, 0.0, None)

    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
    ETM = _ETM_from_T_PP(Tef, I, a)
    df["w"] = ETM * df["Ld"]

    Ld_sum_m = df.groupby("month")["Ld"].transform("sum")
    W_sum_m = df.groupby("month")["w"].transform("sum")
    e0_map = {m + 1: E0_raw_by_month[m] for m in range(12)}

    df["E0m"] = df["month"].map(e0_map)
    df["PET_target_m"] = df["E0m"] * (Ld_sum_m / (12.0 * 30.0))

    petB = np.where(
        (W_sum_m > 0) & (Ld_sum_m > 0),
        df["PET_target_m"] * (df["w"] / W_sum_m),
        df["E0m"] * (df["Ld"] / 12.0) * (1.0 / 30.0),
    )
    return pd.Series(petB, index=df.index, name="PET_B_CAM")


def hs_monthsum_from_daily(daily_df: pd.DataFrame, latitude: float) -> pd.Series:
    """
    Return monthly Hargreaves-Samani PET totals from daily data.
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["Ra"] = ra_extraterrestrial_vec(latitude, df["date"])

    g = df.groupby("month", sort=True)
    tavg_m = g["tavg"].mean()
    tmax_m = g["tmax"].mean()
    tmin_m = g["tmin"].mean()
    Ra_sum = g["Ra"].sum()
    dtr_m = (tmax_m - tmin_m).clip(lower=0.0)

    E0_m = 0.0023 * (tavg_m.clip(lower=0.0) + 17.8) * np.sqrt(dtr_m) * Ra_sum
    return E0_m.reindex(range(1, 13)).fillna(0.0)


def pet_series_HS_M_A(daily_df: pd.DataFrame, latitude: float) -> pd.Series:
    """
    Return daily PET using monthly HS totals distributed by daylength.
    """
    E0m = hs_monthsum_from_daily(daily_df, latitude)
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["Ld"] = daylength_hours_vec(latitude, df["date"])

    Lsum = df.groupby("month")["Ld"].transform("sum")
    pet = np.where(Lsum > 0, E0m.loc[df["month"]].to_numpy() * (df["Ld"] / Lsum), 0.0)
    return pd.Series(pet, index=df.index, name="PET_HS_M_A")


def pet_series_HS_M_B(daily_df: pd.DataFrame, latitude: float) -> pd.Series:
    """
    Return daily PET using monthly HS totals distributed by daylength and temperature weight.
    """
    E0m = hs_monthsum_from_daily(daily_df, latitude)
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["Ld"] = daylength_hours_vec(latitude, df["date"])
    df["Tw"] = np.clip(df["tavg"] + 17.8, 0.0, None)
    df["w"] = df["Ld"] * df["Tw"]

    Wsum = df.groupby("month")["w"].transform("sum")
    pet = np.where(Wsum > 0, E0m.loc[df["month"]].to_numpy() * (df["w"] / Wsum), 0.0)
    return pd.Series(pet, index=df.index, name="PET_HS_M_B")


def calculate_daily_pet_legacy(tavg, latitude, date, I):
    """
    Return daily Thornthwaite PET using the legacy daylength-normalized method.
    """
    a = (6.75e-7 * (I**3)) - (7.71e-5 * (I**2)) + (1.792e-2 * I) + 0.49239

    if 0 < tavg < 26.5:
        raw_pet = 16 * np.power(10 * tavg / I, a)
    elif tavg >= 38:
        raw_pet = 185.0
    elif tavg <= 0:
        raw_pet = 0
    else:
        raw_pet = get_pe_value(tavg)

    month = date.month
    day_of_year = date.timetuple().tm_yday

    phi = np.radians(latitude)
    delta = np.radians(23.45) * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    daily_daylength = (24 / np.pi) * ws

    days_in_month = pd.Period(date.strftime("%Y-%m"), freq="M").days_in_month
    monthly_daylength = 0.0

    for day in range(1, days_in_month + 1):
        temp_date = pd.Timestamp(date.year, month, day)
        temp_day_of_year = temp_date.timetuple().tm_yday
        delta = np.radians(23.45) * np.sin(np.radians((360 / 365) * (temp_day_of_year - 81)))
        ws = np.arccos(-np.tan(phi) * np.tan(delta))
        monthly_daylength += (24 / np.pi) * ws

    pet = raw_pet * (daily_daylength / monthly_daylength)
    return pet