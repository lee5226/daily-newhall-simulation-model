"""Run final D-NSM parameterization across alpha_fast sensitivity values.

This runner is based on ``run_dnsm_final_qc_w33_dbthirdbar_ptqc.py``. The
soil, climate, PET, warm-up, and VWC calculations match the PT-QC primary run;
only alpha_fast is varied. Outputs are isolated from the primary D-NSM results
under ``outputs/<year>/alpha_fast_sensitivity/<soil_tag>/daily_newhall``.
"""

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
THETA_TAG = f"layers_ptf_df_{SOIL_TAG}"
K_PARAM = 0.69
USE_SPINUP = True

ALPHA_FAST_VALUES = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
]
FINAL_ALPHA_FAST = 0.35
OVERWRITE_EXISTING = True

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG
SENSITIVITY_DIR = (
    BASE_DIR / "alpha_fast_sensitivity" / SOIL_TAG
)
OUTPUT_ROOT = SENSITIVITY_DIR / "daily_newhall"

LAYERS_BASE_CSV = (
    SOIL_DIR
    / f"layers_by_station_{YEAR}_{SOIL_TAG}.csv"
)
LAYERS_SAT_CSV = (
    SOIL_DIR
    / f"layers_by_station_{YEAR}_{SOIL_TAG}_satPTF_only.csv"
)
COORDINATE_CSV = (
    SOIL_DIR
    / f"coordinates_final_{YEAR}_{SOIL_TAG}.csv"
)


def normalize_station_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def alpha_tag(alpha_fast: float) -> str:
    return f"alpha_{alpha_fast:.2f}".replace(".", "p")


def run_id_for_alpha(alpha_fast: float) -> str:
    return f"PP_warmup_ptqc_alpha_fast_{alpha_fast:.2f}".replace(".", "p")


def output_path_for_alpha(
    station_id: str,
    alpha_fast: float,
) -> Path:
    sid5 = str(station_id).zfill(5)
    sid_output = sid5.lstrip("0") or "0"
    return (
        OUTPUT_ROOT
        / alpha_tag(alpha_fast)
        / sid_output
        / (
            f"daily_results_{sid_output}_{alpha_tag(alpha_fast)}.csv"
        )
    )


def validate_alpha_values() -> None:
    if not ALPHA_FAST_VALUES:
        raise RuntimeError("ALPHA_FAST_VALUES is empty")
    if len(set(ALPHA_FAST_VALUES)) != len(ALPHA_FAST_VALUES):
        raise RuntimeError("ALPHA_FAST_VALUES contains duplicates")
    invalid = [
        value
        for value in ALPHA_FAST_VALUES
        if not np.isfinite(value) or value < 0.0 or value > 1.0
    ]
    if invalid:
        raise RuntimeError(
            f"alpha_fast values must be within [0, 1]: {invalid}"
        )
    if FINAL_ALPHA_FAST not in ALPHA_FAST_VALUES:
        raise RuntimeError(
            f"Final alpha {FINAL_ALPHA_FAST} is not in ALPHA_FAST_VALUES"
        )


def load_layers() -> pd.DataFrame:
    base = pd.read_csv(
        LAYERS_BASE_CSV,
        dtype={"station_id": "string"},
    )
    sat = pd.read_csv(
        LAYERS_SAT_CSV,
        dtype={"station_id": "string"},
    )
    base["station_id"] = normalize_station_id(base["station_id"])
    sat["station_id"] = normalize_station_id(sat["station_id"])

    merge_keys = [
        "station",
        "station_id",
        "year",
        "mukey",
        "soil_tag",
    ]
    sat_columns = [
        f"sat_ptf_df_k{k}" for k in range(1, 9)
    ]
    missing_sat = [
        column
        for column in merge_keys + sat_columns
        if column not in sat
    ]
    if missing_sat:
        raise RuntimeError(
            f"Missing theta_sat columns in saturation file: {missing_sat}"
        )

    layers = base.merge(
        sat[merge_keys + sat_columns],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(layers) != len(base) or len(layers) != len(sat):
        raise RuntimeError(
            "Base and saturation layer files do not contain identical "
            f"station keys: base={len(base)}, saturation={len(sat)}, "
            f"merged={len(layers)}"
        )

    required = [
        *(f"wp_k{k}" for k in range(1, 9)),
        *(f"fc_k{k}" for k in range(1, 9)),
        *(f"sat_ptf_df_k{k}" for k in range(1, 9)),
    ]
    missing = [
        column for column in required if column not in layers
    ]
    if missing:
        raise RuntimeError(
            f"Missing final layer columns: {missing}"
        )
    if layers[required].isna().any().any():
        bad = layers.loc[
            layers[required].isna().any(axis=1),
            "station_id",
        ].tolist()
        raise RuntimeError(
            f"Missing final model inputs for stations: {bad}"
        )

    return layers


def assign_pet(
    daily: pd.DataFrame,
    latitude: float,
    heat_index: float,
) -> np.ndarray:
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

def prepare_station_inputs(
    station_id: str,
    layers: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> dict[str, object]:
    sid5 = str(station_id).zfill(5)

    layer_match = layers[layers["station_id"] == sid5]
    if len(layer_match) != 1:
        raise RuntimeError(
            f"Expected one final layer row for {sid5}, "
            f"found {len(layer_match)}"
        )
    layer_row = layer_match.iloc[0]

    coordinate_match = coordinates[
        coordinates["station_id"] == sid5
    ]
    if len(coordinate_match) != 1:
        raise RuntimeError(
            f"Expected one coordinate row for {sid5}, "
            f"found {len(coordinate_match)}"
        )
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
            "This PT-QC alpha sensitivity run requires the preceding-year warm-up. "
            "Set USE_SPINUP=True."
        )

    wp = np.array(
        [layer_row[f"wp_k{k}"] for k in range(1, 9)],
        dtype=float,
    )
    fc = np.array(
        [layer_row[f"fc_k{k}"] for k in range(1, 9)],
        dtype=float,
    )
    sat = np.array(
        [
            layer_row[f"sat_ptf_df_k{k}"]
            for k in range(1, 9)
        ],
        dtype=float,
    )
    if not np.isfinite(
        np.concatenate([wp, fc, sat])
    ).all():
        raise RuntimeError(
            f"Non-finite hydraulic inputs for station {sid5}"
        )
    if np.any(fc <= wp):
        raise RuntimeError(
            f"theta_fc must exceed theta_wp for station {sid5}"
        )
    if np.any(sat <= fc):
        raise RuntimeError(
            f"theta_sat must exceed theta_fc for station {sid5}"
        )

    c1, c2, w1_slots, w2_slots = capacities_from_layers(
        wp, fc, sat
    )
    return {
        "sid5": sid5,
        "daily": daily,
        "spinup": spinup,
        "wp": wp,
        "fc": fc,
        "sat": sat,
        "c1": c1,
        "c2": c2,
        "w1_slots": w1_slots,
        "w2_slots": w2_slots,
        "c1_total": float(c1.sum()),
        "c2_total": float(c2.sum()),
    }


def run_station_alpha(
    station_inputs: dict[str, object],
    alpha_fast: float,
) -> Path:
    sid5 = str(station_inputs["sid5"])
    output_path = output_path_for_alpha(sid5, alpha_fast)
    if output_path.exists() and not OVERWRITE_EXISTING:
        print(
            f"Skipped existing: {sid5}, "
            f"alpha_fast={alpha_fast:.2f}"
        )
        return output_path

    daily = station_inputs["daily"].copy()
    spinup = station_inputs["spinup"]
    wp = station_inputs["wp"]
    fc = station_inputs["fc"]
    sat = station_inputs["sat"]
    c1 = station_inputs["c1"]
    c2 = station_inputs["c2"]
    w1_slots = station_inputs["w1_slots"]
    w2_slots = station_inputs["w2_slots"]
    c1_total = float(station_inputs["c1_total"])
    c2_total = float(station_inputs["c2_total"])

    soil_s1 = np.zeros(64, dtype=float)
    soil_s2 = np.zeros(64, dtype=float)

    if USE_SPINUP:
        if spinup is None:
            raise RuntimeError(
                f"Missing prepared spinup data for {sid5}"
            )
        for row in spinup.itertuples(index=False):
            apply_water_change_dual_slots(
                amount_mm=float(
                    row.precipitation - row.PET_pre
                ),
                s1_slots=soil_s1,
                s2_slots=soil_s2,
                w1_slot_arr=w1_slots,
                w2_slot_arr=w2_slots,
                C1_total=c1_total,
                C2_total=c2_total,
                alpha_fast=alpha_fast,
            )

    for index, row in daily.iterrows():
        pet = float(row["PET_pre"])
        water_balance = float(
            row["precipitation"] - pet
        )
        apply_water_change_dual_slots(
            amount_mm=water_balance,
            s1_slots=soil_s1,
            s2_slots=soil_s2,
            w1_slot_arr=w1_slots,
            w2_slot_arr=w2_slots,
            C1_total=c1_total,
            C2_total=c2_total,
            alpha_fast=alpha_fast,
        )

        _, vwc_surface, vwc_rootzone = (
            theta_by_layer_from_slots(
                soil_s1,
                soil_s2,
                c1,
                c2,
                wp,
                fc,
                sat,
            )
        )
        combined_slots = soil_s1 + soil_s2
        daily.at[index, "PET"] = pet
        daily.at[index, "WaterBalance"] = water_balance
        daily.at[index, "MoistureCondition"] = (
            moisture_condition(combined_slots)
        )
        daily.at[index, "SoilMoistureTotal"] = float(
            combined_slots.sum()
        )
        daily.at[index, "VWC_0_12_5cm"] = vwc_surface
        daily.at[index, "VWC_0_100cm"] = vwc_rootzone
        daily.at[index, "PET_method"] = PET_METHOD

    daily["run_id"] = run_id_for_alpha(alpha_fast)
    daily["theta_tag"] = THETA_TAG
    daily["soil_tag"] = SOIL_TAG
    daily["alpha_fast_root"] = alpha_fast

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    print(
        f"Saved: {sid5}, alpha_fast={alpha_fast:.2f} "
        f"-> {output_path}"
    )
    return output_path


def main() -> None:
    validate_alpha_values()
    layers = load_layers()
    coordinates = pd.read_csv(
        COORDINATE_CSV,
        dtype={"station_id": "string"},
    )
    coordinates["station_id"] = normalize_station_id(
        coordinates["station_id"]
    )

    failures: list[tuple[str, float | None, str]] = []
    completed = 0

    for station_id in layers["station_id"].tolist():
        try:
            station_inputs = prepare_station_inputs(
                station_id,
                layers,
                coordinates,
            )
        except Exception as error:
            failures.append((station_id, None, str(error)))
            print(f"{station_id}: {error}")
            continue

        for alpha_fast in ALPHA_FAST_VALUES:
            try:
                run_station_alpha(
                    station_inputs,
                    alpha_fast,
                )
                completed += 1
            except Exception as error:
                failures.append(
                    (station_id, alpha_fast, str(error))
                )
                print(
                    f"{station_id}, alpha_fast={alpha_fast:.2f}: "
                    f"{error}"
                )

    if failures:
        detail = "; ".join(
            (
                f"{station_id}, alpha=NA: {message}"
                if alpha_fast is None
                else (
                    f"{station_id}, alpha={alpha_fast:.2f}: "
                    f"{message}"
                )
            )
            for station_id, alpha_fast, message in failures
        )
        raise RuntimeError(
            f"Alpha-fast sensitivity runs failed for "
            f"{len(failures)} cases: {detail}"
        )

    expected = len(layers) * len(ALPHA_FAST_VALUES)
    if completed != expected:
        raise RuntimeError(
            f"Unexpected run count: completed={completed}, "
            f"expected={expected}"
        )

    print(
        f"Completed alpha-fast sensitivity runs: "
        f"{len(layers)} stations x "
        f"{len(ALPHA_FAST_VALUES)} alpha values = {completed}"
    )


if __name__ == "__main__":
    main()
