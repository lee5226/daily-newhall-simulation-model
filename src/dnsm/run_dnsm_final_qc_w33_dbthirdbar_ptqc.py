"""Run the single-year D-NSM with strict P/T-QC climate handling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnsm.climate import load_climate_pair
from src.dnsm.moisture_condition import moisture_condition
from src.dnsm.pet import calculate_annual_heat_index, pet_series_TEFSTAR
from src.dnsm.soil_profile import (
    apply_water_change_dual_slots,
    capacities_from_layers,
    theta_by_layer_from_slots,
)
from src.utils.paths import OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"
PET_METHOD = "PP_TEFSTAR"
RUN_ID = "PP_warmup_ptqc"
THETA_TAG = f"layers_ptf_df_{SOIL_TAG}"
K_PARAM = 0.69
USE_SPINUP = True
ALPHA_FAST_ROOT = 0.35

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
OUTPUT_DIR = BASE_DIR / "daily_newhall" / SOIL_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_BASE_CSV = SOIL_DIR / f"layers_by_station_{YEAR}_{SOIL_TAG}.csv"
LAYERS_SAT_CSV = SOIL_DIR / f"layers_by_station_{YEAR}_{SOIL_TAG}_satPTF_only.csv"
COORDINATE_CSV = SOIL_DIR / f"coordinates_final_{YEAR}_{SOIL_TAG}.csv"

def normalize_station_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)


def load_layers() -> pd.DataFrame:
    base = pd.read_csv(LAYERS_BASE_CSV, dtype={"station_id": "string"})
    sat = pd.read_csv(LAYERS_SAT_CSV, dtype={"station_id": "string"})
    base["station_id"] = normalize_station_id(base["station_id"])
    sat["station_id"] = normalize_station_id(sat["station_id"])

    merge_keys = ["station", "station_id", "year", "mukey", "soil_tag"]
    sat_columns = [f"sat_ptf_df_k{k}" for k in range(1, 9)]
    missing_sat = [
        column for column in merge_keys + sat_columns if column not in sat
    ]
    if missing_sat:
        raise RuntimeError(
            f"Missing theta_sat columns in saturation file: {missing_sat}"
        )

    # FC and WP come from the base layer file. Import only theta_sat from the
    # saturation file to avoid duplicate fc_k* columns after the merge.
    layers = base.merge(
        sat[merge_keys + sat_columns],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(layers) != len(base) or len(layers) != len(sat):
        raise RuntimeError(
            "Base and saturation layer files do not contain identical station keys: "
            f"base={len(base)}, saturation={len(sat)}, merged={len(layers)}"
        )

    required = [
        *(f"wp_k{k}" for k in range(1, 9)),
        *(f"fc_k{k}" for k in range(1, 9)),
        *(f"sat_ptf_df_k{k}" for k in range(1, 9)),
    ]
    missing = [column for column in required if column not in layers]
    if missing:
        raise RuntimeError(f"Missing final layer columns: {missing}")
    if layers[required].isna().any().any():
        bad = layers.loc[
            layers[required].isna().any(axis=1), "station_id"
        ].tolist()
        raise RuntimeError(f"Missing final model inputs for stations: {bad}")
    return layers


LAYERS = load_layers()


def assign_pet(daily: pd.DataFrame, latitude: float, heat_index: float) -> np.ndarray:
    return pet_series_TEFSTAR(
        daily,
        latitude,
        heat_index,
        k=K_PARAM,
        clip_star=True,
    ).values


def assign_pet_for_year(climate: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """Calculate PET after strict climate gap treatment for one calendar year."""
    result = climate.copy()
    monthly_tavg = (
        result.groupby(result["date"].dt.month)["tavg"].mean().clip(lower=0.0)
    )
    if len(monthly_tavg) != 12:
        raise RuntimeError(
            f"Expected 12 monthly temperatures for PET heat index, got {len(monthly_tavg)}"
        )
    heat_index = calculate_annual_heat_index(monthly_tavg.to_numpy())
    result["PET_pre"] = assign_pet(result, latitude, heat_index)
    result["k_param"] = K_PARAM
    return result


def prepare_climate_pair_for_station(
    warmup_path: Path,
    evaluation_path: Path,
    warmup_year: int,
    evaluation_year: int,
    latitude: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jointly process P/T across the year boundary, then calculate annual PET."""
    warmup, evaluation = load_climate_pair(
        warmup_filepath=warmup_path,
        evaluation_filepath=evaluation_path,
        warmup_year=warmup_year,
        evaluation_year=evaluation_year,
    )
    warmup = assign_pet_for_year(warmup, latitude)
    evaluation = assign_pet_for_year(evaluation, latitude)
    return warmup, evaluation

def run_station(station_id: str, coordinates: pd.DataFrame) -> None:
    sid5 = str(station_id).zfill(5)
    sid_output = sid5.lstrip("0") or "0"

    layer_match = LAYERS[LAYERS["station_id"] == sid5]
    if layer_match.empty:
        raise RuntimeError(f"Final layers not found for station {sid5}")
    layer_row = layer_match.iloc[0]

    coordinate_match = coordinates[coordinates["station_id"] == sid5]
    if coordinate_match.empty:
        raise RuntimeError(f"Coordinates not found for station {sid5}")
    latitude = float(coordinate_match.iloc[0]["latitude"])

    evaluation_year = int(layer_row["year"])
    warmup_year = evaluation_year - 1
    daily_path = (
        BASE_DIR
        / "daily_climate_ptqc"
        / f"daily_climate_{YEAR}_{sid5}.csv"
    )
    warmup_path = (
        BASE_DIR
        / "daily_climate_warmup_ptqc"
        / f"daily_climate_{YEAR}_{sid5}_warmup_y1.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(f"Evaluation climate file not found: {daily_path}")
    if USE_SPINUP and not warmup_path.exists():
        raise FileNotFoundError(f"Warm-up climate file not found: {warmup_path}")

    if USE_SPINUP:
        spinup, daily = prepare_climate_pair_for_station(
            warmup_path=warmup_path,
            evaluation_path=daily_path,
            warmup_year=warmup_year,
            evaluation_year=evaluation_year,
            latitude=latitude,
        )
    else:
        raise RuntimeError(
            "This strict single-year run is designed to use the preceding-year warm-up. "
            "Set USE_SPINUP=True."
        )

    wp = np.array([layer_row[f"wp_k{k}"] for k in range(1, 9)], dtype=float)
    fc = np.array([layer_row[f"fc_k{k}"] for k in range(1, 9)], dtype=float)
    sat = np.array(
        [layer_row[f"sat_ptf_df_k{k}"] for k in range(1, 9)], dtype=float
    )
    if not np.isfinite(np.concatenate([wp, fc, sat])).all():
        raise RuntimeError(f"Non-finite hydraulic inputs for station {sid5}")

    if np.any(fc <= wp):
        raise RuntimeError(
            f"theta_fc must exceed theta_wp for station {sid5}"
        )
    if np.any(sat <= fc):
        raise RuntimeError(
            f"theta_sat must exceed theta_fc for station {sid5}"
        )
    c1, c2, w1_slots, w2_slots = capacities_from_layers(wp, fc, sat)
    c1_total = float(c1.sum())
    c2_total = float(c2.sum())
    soil_s1 = np.zeros(64, dtype=float)
    soil_s2 = np.zeros(64, dtype=float)

    if USE_SPINUP:
        for row in spinup.itertuples(index=False):
            apply_water_change_dual_slots(
                amount_mm=float(row.precipitation - row.PET_pre),
                s1_slots=soil_s1,
                s2_slots=soil_s2,
                w1_slot_arr=w1_slots,
                w2_slot_arr=w2_slots,
                C1_total=c1_total,
                C2_total=c2_total,
                alpha_fast=ALPHA_FAST_ROOT,
            )

    for index, row in daily.iterrows():
        pet = float(row["PET_pre"])
        water_balance = float(row["precipitation"] - pet)
        apply_water_change_dual_slots(
            amount_mm=water_balance,
            s1_slots=soil_s1,
            s2_slots=soil_s2,
            w1_slot_arr=w1_slots,
            w2_slot_arr=w2_slots,
            C1_total=c1_total,
            C2_total=c2_total,
            alpha_fast=ALPHA_FAST_ROOT,
        )

        _, vwc_surface, vwc_rootzone = theta_by_layer_from_slots(
            soil_s1, soil_s2, c1, c2, wp, fc, sat
        )
        combined_slots = soil_s1 + soil_s2
        daily.at[index, "PET"] = pet
        daily.at[index, "WaterBalance"] = water_balance
        daily.at[index, "MoistureCondition"] = moisture_condition(combined_slots)
        daily.at[index, "SoilMoistureTotal"] = float(combined_slots.sum())
        daily.at[index, "VWC_0_12_5cm"] = vwc_surface
        daily.at[index, "VWC_0_100cm"] = vwc_rootzone
        daily.at[index, "PET_method"] = PET_METHOD

    print(
        f"{sid5}: eval_year={evaluation_year}, warmup_year={warmup_year}, "
        f"warmup_P_zero_filled={int(spinup.get('precip_zero_filled_flag', pd.Series(dtype=int)).sum())}, "
        f"Tmin_interp={int(spinup.get('tmin_interpolated_flag', pd.Series(dtype=int)).sum() + daily.get('tmin_interpolated_flag', pd.Series(dtype=int)).sum())}, "
        f"Tmax_interp={int(spinup.get('tmax_interpolated_flag', pd.Series(dtype=int)).sum() + daily.get('tmax_interpolated_flag', pd.Series(dtype=int)).sum())}"
    )

    daily["run_id"] = RUN_ID
    daily["theta_tag"] = THETA_TAG
    daily["soil_tag"] = SOIL_TAG

    station_output_dir = OUTPUT_DIR / sid_output
    station_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = station_output_dir / (
        f"daily_results_{sid_output}_{RUN_ID}_{THETA_TAG}_dual.csv"
    )
    daily.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main() -> None:
    coordinates = pd.read_csv(COORDINATE_CSV, dtype={"station_id": "string"})
    coordinates["station_id"] = normalize_station_id(coordinates["station_id"])

    failures: list[tuple[str, str]] = []
    for station_id in LAYERS["station_id"].tolist():
        try:
            run_station(station_id, coordinates)
        except Exception as error:
            failures.append((station_id, str(error)))
            print(f"{station_id}: {error}")

    if failures:
        detail = "; ".join(f"{sid}: {message}" for sid, message in failures)
        raise RuntimeError(f"D-NSM failed for {len(failures)} stations: {detail}")
    print(f"Completed final D-NSM runs: {len(LAYERS)} stations")


if __name__ == "__main__":
    main()
