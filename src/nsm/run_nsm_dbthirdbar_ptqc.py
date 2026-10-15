"""Run the classical monthly NSM for the final P/T-QC station set."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DATA_REFERENCE_DIR, OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"

BASE_DIR = OUTPUTS_DIR / str(YEAR)
SOIL_DIR = BASE_DIR / SOIL_TAG

STATION_CSV = (
    SOIL_DIR
    / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv"
)
COORDINATE_CSV = (
    SOIL_DIR
    / f"coordinates_final_{YEAR}_{SOIL_TAG}.csv"
)

MONTHLY_INPUT_DIR = BASE_DIR / "monthly_climate_ptqc"
OUTPUT_DIR = BASE_DIR / "monthly_newhall" / SOIL_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPINUP_TOL = 0.01
SPINUP_MAX_ITER = 100
SAVE_SPINUP_INFO = True


def normalize_station_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def run_newhall_monthly(
    station_id: str,
    coordinates: pd.DataFrame,
    monthly_input_dir: Path,
    output_dir: Path,
    reference_dir: Path,
) -> None:
    sid5 = normalize_station_id(station_id)
    sid_output = sid5.lstrip("0") or "0"

    # The classical NSM uses a fixed 200-mm available water capacity.
    awc = 200.0
    water_per_slot = awc / 64.0

    coordinate_match = coordinates[
        coordinates["station_id"] == sid5
    ]
    if coordinate_match.empty:
        raise RuntimeError(
            f"Coordinates not found for station {sid5}"
        )

    coordinate_row = coordinate_match.iloc[0]
    latitude = float(coordinate_row["latitude"])
    ns_hemisphere = str(coordinate_row["nsHemisphere"]).upper()

    if ns_hemisphere not in {"N", "S"}:
        raise RuntimeError(
            f"Invalid hemisphere for station {sid5}: "
            f"{ns_hemisphere}"
        )

    monthly_path = (
        monthly_input_dir
        / f"monthly_climate_{YEAR}_{sid5}.csv"
    )
    if not monthly_path.exists():
        raise FileNotFoundError(
            f"Missing monthly climate file: {monthly_path}"
        )

    monthly = pd.read_csv(monthly_path)

    required_monthly = {"month", "PPT", "TAVG"}
    missing_monthly = sorted(
        required_monthly.difference(monthly.columns)
    )
    if missing_monthly:
        raise RuntimeError(
            f"Missing monthly climate columns for {sid5}: "
            f"{missing_monthly}"
        )

    monthly = monthly.sort_values("month").reset_index(drop=True)

    expected_months = list(range(1, 13))
    actual_months = monthly["month"].astype(int).tolist()
    if actual_months != expected_months:
        raise RuntimeError(
            f"Incomplete monthly climate for {sid5}: "
            f"months={actual_months}"
        )

    if monthly[["PPT", "TAVG"]].isna().any().any():
        raise RuntimeError(
            f"Missing monthly PPT or TAVG values for {sid5}"
        )

    ppt = monthly["PPT"].astype(float).to_numpy()
    tavg = monthly["TAVG"].astype(float).to_numpy()

    constants_path = reference_dir / "constants_classic.csv"
    knorth_path = reference_dir / "knorth_classic.csv"
    ksouth_path = reference_dir / "ksouth_classic.csv"

    for path in (constants_path, knorth_path, ksouth_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing classical NSM reference file: {path}"
            )

    constants = pd.read_csv(constants_path)
    depletion_order = (
        constants["depletion_order"].to_numpy(dtype=int) - 1
    )
    depletion_req = constants["depletion_req"].to_numpy(
        dtype=float
    )

    knorth = pd.read_csv(knorth_path).set_index("latitude")
    ksouth = pd.read_csv(ksouth_path).set_index("latitude")

    def get_pe_value(temp_avg: float) -> float:
        return float(
            np.interp(
                temp_avg,
                constants["temp_bins"].dropna(),
                constants["pe_bins"].dropna(),
            )
        )

    def get_k_value(
        latitude_value: float,
        hemisphere: str,
        month: int,
    ) -> float:
        if hemisphere == "N":
            return float(
                np.interp(
                    latitude_value,
                    knorth.index.astype(float),
                    knorth[f"knorth_{month}"],
                )
            )

        return float(
            np.interp(
                latitude_value,
                ksouth.index.astype(float),
                ksouth[f"ksouth_{month}"],
            )
        )

    def apply_water_change(
        amount: float,
        soil_profile: np.ndarray,
    ) -> None:
        if amount > 0:
            current_nma = amount

            for position in range(64):
                amount_to_fill = max(
                    0.0,
                    water_per_slot - soil_profile[position],
                )

                if current_nma > amount_to_fill:
                    soil_profile[position] += amount_to_fill
                    current_nma -= amount_to_fill
                else:
                    soil_profile[position] += current_nma
                    break

        elif amount < 0:
            current_nma = abs(amount)

            for index, position in enumerate(depletion_order):
                water_in_slot = soil_profile[position]

                if water_in_slot == 0:
                    continue

                energy_to_drain_slot = (
                    water_in_slot * depletion_req[index]
                )

                if current_nma > energy_to_drain_slot:
                    soil_profile[position] = 0.0
                    current_nma -= energy_to_drain_slot
                else:
                    soil_profile[position] *= (
                        1.0
                        - current_nma / energy_to_drain_slot
                    )
                    break

    def run_one_year(
        initial_soil_profile: np.ndarray,
    ) -> tuple[np.ndarray, list[dict], list[float]]:
        soil_profile = initial_soil_profile.copy()
        raw_results: list[dict] = []
        monthly_pe_list: list[float] = []

        positive_tavg = np.clip(tavg, a_min=0.0, a_max=None)
        monthly_heat_indices = np.power(
            positive_tavg / 5.0,
            1.514,
        )
        heat_index = float(np.sum(monthly_heat_indices))

        a = (
            6.75e-7 * heat_index**3
            - 7.71e-5 * heat_index**2
            + 1.792e-2 * heat_index
            + 0.49239
        )

        for month_index in range(12):
            monthly_precipitation = float(ppt[month_index])
            low_intensity_precipitation = (
                monthly_precipitation / 2.0
            )
            high_intensity_precipitation = (
                monthly_precipitation / 2.0
            )

            monthly_temperature = float(tavg[month_index])

            if 0 < monthly_temperature < 26.5:
                if heat_index > 0:
                    raw_pe = 16.0 * np.power(
                        10.0
                        * monthly_temperature
                        / heat_index,
                        a,
                    )
                else:
                    raw_pe = 0.0
            elif monthly_temperature >= 38.0:
                raw_pe = 185.0
            elif monthly_temperature < 0.0:
                raw_pe = 0.0
            else:
                raw_pe = get_pe_value(monthly_temperature)

            k_value = get_k_value(
                latitude,
                ns_hemisphere,
                month_index + 1,
            )
            monthly_pe = float(raw_pe * k_value)
            monthly_pe_list.append(monthly_pe)

            net_moisture_activity = (
                low_intensity_precipitation - monthly_pe
            )

            apply_water_change(
                net_moisture_activity / 2.0,
                soil_profile,
            )
            apply_water_change(
                high_intensity_precipitation,
                soil_profile,
            )
            apply_water_change(
                net_moisture_activity / 2.0,
                soil_profile,
            )

            raw_results.append(
                {
                    "month": month_index + 1,
                    "soil_profile": soil_profile.copy(),
                }
            )

        return soil_profile, raw_results, monthly_pe_list

    soil_profile = np.zeros(64, dtype=float)
    converged = False
    spinup_records: list[dict] = []

    for iteration in range(1, SPINUP_MAX_ITER + 1):
        previous_profile = soil_profile.copy()
        soil_profile, _, _ = run_one_year(previous_profile)

        maximum_difference = float(
            np.max(np.abs(soil_profile - previous_profile))
        )

        spinup_records.append(
            {
                "iteration": iteration,
                "max_abs_diff": maximum_difference,
                "end_total_moisture": float(
                    np.sum(soil_profile)
                ),
            }
        )

        if maximum_difference <= SPINUP_TOL:
            converged = True
            break

    final_initial_profile = soil_profile.copy()

    _, raw_results, monthly_pe_list = run_one_year(
        final_initial_profile
    )

    summary_results: list[dict] = []

    for month_index, result in enumerate(raw_results):
        soil_profile = result["soil_profile"]
        monthly_pe = monthly_pe_list[month_index]

        total_moisture = float(np.sum(soil_profile))
        surface_moisture = float(np.sum(soil_profile[:8]))

        rootzone_faw = total_moisture / awc
        surface_faw = surface_moisture / (
            water_per_slot * 8.0
        )

        rootzone_faw = float(
            np.clip(rootzone_faw, 0.0, 1.0)
        )
        surface_faw = float(
            np.clip(surface_faw, 0.0, 1.0)
        )

        summary_results.append(
            {
                "station_id": sid_output,
                "month": month_index + 1,
                "Monthly_PET": monthly_pe,
                "SoilMoistureTotal": round(
                    total_moisture,
                    3,
                ),
                "FAW_0_100cm": round(
                    rootzone_faw,
                    3,
                ),
                "FAW_0_12_5cm": round(
                    surface_faw,
                    3,
                ),
                "spinup_converged": converged,
                "spinup_iterations": len(
                    spinup_records
                ),
            }
        )

    station_output_dir = output_dir / sid_output
    station_output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_results)
    summary_path = (
        station_output_dir
        / f"monthly_results_{sid_output}.csv"
    )
    summary.to_csv(summary_path, index=False)

    if SAVE_SPINUP_INFO:
        spinup = pd.DataFrame(spinup_records)
        spinup_path = (
            station_output_dir
            / f"spinup_info_{sid_output}.csv"
        )
        spinup.to_csv(spinup_path, index=False)

    print(
        f"{sid5}: monthly simulation complete; "
        f"spinup_converged={converged}, "
        f"iterations={len(spinup_records)}"
    )


def main() -> None:
    if not STATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing final P/T-QC station file: {STATION_CSV}"
        )
    if not COORDINATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing coordinate file: {COORDINATE_CSV}"
        )

    stations = pd.read_csv(
        STATION_CSV,
        dtype={"station_id": "string"},
    )
    coordinates = pd.read_csv(
        COORDINATE_CSV,
        dtype={"station_id": "string"},
    )

    required_station = {"station_id"}
    missing_station = sorted(
        required_station.difference(stations.columns)
    )
    if missing_station:
        raise RuntimeError(
            f"Missing station columns: {missing_station}"
        )

    required_coordinate = {
        "station_id",
        "latitude",
        "nsHemisphere",
    }
    missing_coordinate = sorted(
        required_coordinate.difference(coordinates.columns)
    )
    if missing_coordinate:
        raise RuntimeError(
            f"Missing coordinate columns: {missing_coordinate}"
        )

    stations["station_id"] = stations["station_id"].map(
        normalize_station_id
    )
    coordinates["station_id"] = coordinates["station_id"].map(
        normalize_station_id
    )

    station_ids = (
        stations["station_id"]
        .drop_duplicates()
        .tolist()
    )

    failures: list[tuple[str, str]] = []

    for station_id in station_ids:
        try:
            run_newhall_monthly(
                station_id=station_id,
                coordinates=coordinates,
                monthly_input_dir=MONTHLY_INPUT_DIR,
                output_dir=OUTPUT_DIR,
                reference_dir=DATA_REFERENCE_DIR,
            )
        except Exception as error:
            failures.append((station_id, str(error)))
            print(f"{station_id}: {error}")

    if failures:
        detail = "; ".join(
            f"{station_id}: {message}"
            for station_id, message in failures
        )
        raise RuntimeError(
            f"Monthly NSM failed for {len(failures)} stations: "
            f"{detail}"
        )

    print(
        f"Completed monthly PT-QC NSM runs: "
        f"{len(station_ids)} stations"
    )


if __name__ == "__main__":
    main()