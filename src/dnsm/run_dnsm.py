import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR
from src.dnsm.climate import load_climate_data
from src.dnsm.moisture_condition import moisture_condition
from src.dnsm.pet import (
    calculate_annual_heat_index,
    pet_series_TEFSTAR,
)
from src.dnsm.soil_profile import (
    apply_water_change_dual_slots,
    capacities_from_layers,
    theta_by_layer_from_slots,
)


YEAR = 1016
PET_METHOD = "PP_TEFSTAR"
RUN_ID = "PP_spinup"
THETA_TAG = "layers_ptf_df"
K_PARAM = 0.69

USE_SPINUP = True
ALPHA_FAST_ROOT = 0.4

BASE_ROOT = OUTPUTS_DIR
BASE_IN = BASE_ROOT / str(YEAR)
BASE_OUT = BASE_ROOT / str(YEAR) / "daily_newhall" / "pet_PP"
BASE_OUT.mkdir(parents=True, exist_ok=True)

LAYERS_BASE_CSV = BASE_IN / f"layers_by_station_{YEAR}.csv"
layers_base_df = pd.read_csv(LAYERS_BASE_CSV)
layers_base_df["station_id"] = layers_base_df["station_id"].astype(str).str.lstrip("0")

try:
    LAYERS_SAT_CSV = BASE_IN / f"layers_by_station_{YEAR}_satPTF_only.csv"
    layers_sat_df = pd.read_csv(LAYERS_SAT_CSV)
    layers_sat_df["station_id"] = layers_sat_df["station_id"].astype(str).str.lstrip("0")
    layers_df = layers_base_df.merge(
        layers_sat_df,
        on=["station", "station_id", "year"],
        how="left",
    )
except Exception:
    layers_df = layers_base_df.copy()

need_base = [*(f"wp_k{i}" for i in range(1, 9)), *(f"fc_k{i}" for i in range(1, 9))]
missing = [c for c in need_base if c not in layers_df.columns]
if missing:
    raise RuntimeError(f"Missing layer base columns: {missing}")

SAT_COLS = [f"sat_ptf_df_k{i}" for i in range(1, 9)]


def assign_pet(daily_df, latitude, heat_index):
    return pet_series_TEFSTAR(
        daily_df,
        latitude,
        heat_index,
        k=K_PARAM,
        clip_star=True,
    ).values


def run_newhall_daily(station_id, coord_df):
    try:
        sid5 = str(station_id).zfill(5)
        sid = str(station_id).lstrip("0")

        daily_data_path = BASE_IN / "daily_climate" / f"daily_climate_{YEAR}_{sid5}.csv"
        output_dir = BASE_OUT / sid
        output_dir.mkdir(parents=True, exist_ok=True)

        coord_row = coord_df[coord_df["station_id"].astype(str).str.lstrip("0") == sid]
        if coord_row.empty:
            raise ValueError(f"Coordinates not found for station_id {sid}")

        latitude = float(coord_row["latitude"].values[0])
        if "elev_m" in coord_row.columns and pd.notna(coord_row["elev_m"].values[0]):
            elevation_m = float(coord_row["elev_m"].values[0])
        else:
            elevation_m = 0.0

        daily = load_climate_data(daily_data_path)
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily[daily["date"].notna()]

        monthly_tavg = daily.groupby(daily["date"].dt.month)["tavg"].mean().values
        monthly_tavg = np.clip(monthly_tavg, 0, None)
        heat_index = calculate_annual_heat_index(monthly_tavg)
        daily["PET_pre"] = assign_pet(daily, latitude, heat_index)
        daily["k_param"] = K_PARAM

        lay_row = layers_df[layers_df["station_id"] == sid]
        if lay_row.empty:
            raise ValueError(f"Layers not found for station_id {sid}")
        lay_row = lay_row.iloc[0]

        wp_k = np.array([lay_row[f"wp_k{i}"] for i in range(1, 9)], float)
        fc_k = np.array([lay_row[f"fc_k{i}"] for i in range(1, 9)], float)
        missing_sat = [c for c in SAT_COLS if c not in lay_row.index]
        if missing_sat:
            raise RuntimeError(f"Missing layer sat columns for {sid}: {missing_sat}")
        sat_k = np.array([lay_row[c] for c in SAT_COLS], float)

        fc_k = np.maximum(fc_k, wp_k + 1e-4)
        sat_k = np.maximum(sat_k, fc_k + 2e-2)

        C1_k, C2_k, w1_slot_arr, w2_slot_arr = capacities_from_layers(wp_k, fc_k, sat_k)
        C1_total = float(C1_k.sum())
        C2_total = float(C2_k.sum())
        soil_s1 = np.zeros(64)
        soil_s2 = np.zeros(64)

        if USE_SPINUP:
            spinup_path = BASE_IN / "daily_climate_spinup" / f"daily_climate_{YEAR}_{sid5}_spinup_y1.csv"
            spin = load_climate_data(spinup_path)
            spin["date"] = pd.to_datetime(spin["date"], errors="coerce")
            spin = spin[spin["date"].notna()]

            m_spin = spin.groupby(spin["date"].dt.month)["tavg"].mean().values
            m_spin = np.clip(m_spin, 0, None)
            i_spin = calculate_annual_heat_index(m_spin)
            spin["PET_pre"] = assign_pet(spin, latitude, i_spin)

            for _, row in spin.iterrows():
                wb_spin = float(row["precipitation"] - row["PET_pre"])
                apply_water_change_dual_slots(
                    amount_mm=wb_spin,
                    s1_slots=soil_s1,
                    s2_slots=soil_s2,
                    w1_slot_arr=w1_slot_arr,
                    w2_slot_arr=w2_slot_arr,
                    C1_total=C1_total,
                    C2_total=C2_total,
                    alpha_fast=ALPHA_FAST_ROOT,
                )

        for idx, row in daily.iterrows():
            pet = float(row["PET_pre"])
            water_balance = float(row["precipitation"] - pet)

            apply_water_change_dual_slots(
                amount_mm=water_balance,
                s1_slots=soil_s1,
                s2_slots=soil_s2,
                w1_slot_arr=w1_slot_arr,
                w2_slot_arr=w2_slot_arr,
                C1_total=C1_total,
                C2_total=C2_total,
                alpha_fast=ALPHA_FAST_ROOT,
            )

            _, vwc_sf, vwc_rz = theta_by_layer_from_slots(
                soil_s1,
                soil_s2,
                C1_k,
                C2_k,
                wp_k,
                fc_k,
                sat_k,
            )

            soil_profile_for_condition = soil_s1 + soil_s2
            total_soil_moisture = float(soil_profile_for_condition.sum())

            daily.at[idx, "PET"] = pet
            daily.at[idx, "WaterBalance"] = water_balance
            daily.at[idx, "MoistureCondition"] = moisture_condition(soil_profile_for_condition)
            daily.at[idx, "SoilMoistureTotal"] = total_soil_moisture
            daily.at[idx, "VWC_0_12_5cm"] = vwc_sf
            daily.at[idx, "VWC_0_100cm"] = vwc_rz
            daily.at[idx, "PET_method"] = PET_METHOD

        daily["run_id"] = RUN_ID
        daily["theta_tag"] = THETA_TAG

        out_path = output_dir / f"daily_results_{sid}_{RUN_ID}_{THETA_TAG}_dual.csv"
        daily.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"{station_id}: {e}")


def main() -> None:
    coord_df = pd.read_csv(BASE_IN / f"coordinates_final_{YEAR}.csv")
    station_ids = coord_df["station_id"].astype(str).str.lstrip("0").unique().tolist()
    for sid in station_ids:
        run_newhall_daily(sid, coord_df)


if __name__ == "__main__":
    main()
