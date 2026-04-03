import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import DATA_REFERENCE_DIR, OUTPUTS_DIR


YEAR = 1016
BASE_IN = OUTPUTS_DIR / str(YEAR)

MONTHLY_INPUT_DIR = BASE_IN / "monthly_climate"
BASE_OUT = BASE_IN / "monthly_newhall"
BASE_OUT.mkdir(parents=True, exist_ok=True)

SPINUP_TOL = 0.01
SPINUP_MAX_ITER = 100
SAVE_SPINUP_INFO = True


def run_newhall_monthly(
    station_id,
    coord_df: pd.DataFrame,
    monthly_input_dir: Path,
    output_dir: Path,
    reference_dir: Path,
) -> None:
    try:
        awc = 200.0
        water_per_slot = awc / 64.0

        coord_row = coord_df[coord_df["station_id"].astype(str).str.lstrip("0") == str(station_id)]
        if coord_row.empty:
            raise ValueError(f"Coordinates not found for station {station_id}")

        latitude = coord_row["latitude"].values[0]
        ns_hemisphere = coord_row["nsHemisphere"].values[0]

        monthly_path = monthly_input_dir / f"monthly_climate_{YEAR}_{str(station_id).zfill(5)}.csv"
        monthly_df = pd.read_csv(monthly_path)
        ppt = monthly_df["PPT"].tolist()
        tavg = monthly_df["TAVG"].tolist()

        constants_df = pd.read_csv(reference_dir / "constants_classic.csv")
        depletion_order = constants_df["depletion_order"].to_numpy(copy=True) - 1
        depletion_req = constants_df["depletion_req"].to_numpy()

        knorth_df = pd.read_csv(reference_dir / "knorth_classic.csv").set_index("latitude")
        ksouth_df = pd.read_csv(reference_dir / "ksouth_classic.csv").set_index("latitude")

        def get_pe_value(temp_avg):
            return np.interp(
                temp_avg,
                constants_df["temp_bins"].dropna(),
                constants_df["pe_bins"].dropna(),
            )

        def get_k_value(lat, hemisphere, month):
            if hemisphere == "N":
                return np.interp(lat, knorth_df.index.astype(float), knorth_df[f"knorth_{month}"])
            return np.interp(lat, ksouth_df.index.astype(float), ksouth_df[f"ksouth_{month}"])

        def apply_water_change(amount, soil_profile):
            if amount > 0:
                current_nma = amount
                for pos in range(64):
                    to_fill = max(0.0, water_per_slot - soil_profile[pos])
                    if current_nma > to_fill:
                        soil_profile[pos] += to_fill
                        current_nma -= to_fill
                    else:
                        soil_profile[pos] += current_nma
                        break
            elif amount < 0:
                current_nma = abs(amount)
                for i, pos in enumerate(depletion_order):
                    water_in_slot = soil_profile[pos]
                    if water_in_slot == 0:
                        continue
                    energy_to_drain_slot = water_in_slot * depletion_req[i]
                    if current_nma > energy_to_drain_slot:
                        soil_profile[pos] = 0.0
                        current_nma -= energy_to_drain_slot
                    else:
                        soil_profile[pos] *= 1 - current_nma / energy_to_drain_slot
                        break

        def run_one_year(initial_soil_profile):
            soil_profile = initial_soil_profile.copy()
            raw_results = []
            monthly_pe_list = []

            i_m = np.power(np.array(tavg) / 5.0, 1.514)
            i_m[np.isnan(i_m)] = 0.0
            heat_index = np.sum(i_m)
            a = (
                (6.75e-7 * (heat_index**3))
                - (7.71e-5 * (heat_index**2))
                + (1.792e-2 * heat_index)
                + 0.49239
            )

            for m in range(12):
                mp = ppt[m]
                lp = mp / 2.0
                hp = mp / 2.0

                if 0 < tavg[m] < 26.5:
                    raw_pe = 16 * np.power(10 * tavg[m] / heat_index, a) if heat_index > 0 else 0.0
                elif tavg[m] >= 38:
                    raw_pe = 185.0
                elif tavg[m] < 0:
                    raw_pe = 0.0
                else:
                    raw_pe = get_pe_value(tavg[m])

                k = get_k_value(latitude, ns_hemisphere, m + 1)
                pe = raw_pe * k
                monthly_pe_list.append(pe)

                nma = lp - pe

                apply_water_change(nma / 2.0, soil_profile)
                apply_water_change(hp, soil_profile)
                apply_water_change(nma / 2.0, soil_profile)

                raw_results.append(
                    {
                        "month": m + 1,
                        "soil_profile": soil_profile.copy(),
                    }
                )

            return soil_profile, raw_results, monthly_pe_list

        soil_profile = np.zeros(64)
        converged = False
        spinup_records = []

        for iteration in range(1, SPINUP_MAX_ITER + 1):
            prev_profile = soil_profile.copy()
            soil_profile, _, _ = run_one_year(prev_profile)

            max_diff = float(np.max(np.abs(soil_profile - prev_profile)))
            spinup_records.append(
                {
                    "iteration": iteration,
                    "max_abs_diff": max_diff,
                    "end_total_moisture": float(np.sum(soil_profile)),
                }
            )

            if max_diff <= SPINUP_TOL:
                converged = True
                break

        final_initial_profile = soil_profile.copy()
        _, raw_results, monthly_pe_list = run_one_year(final_initial_profile)

        summary_results = []
        for m, res in enumerate(raw_results):
            profile = res["soil_profile"]
            monthly_pe = monthly_pe_list[m]

            total_moisture = float(np.sum(profile))
            surface_moisture = float(np.sum(profile[:8]))

            faw_rz = total_moisture / awc
            faw_sf = surface_moisture / (water_per_slot * 8.0)

            faw_rz = max(0.0, min(1.0, faw_rz))
            faw_sf = max(0.0, min(1.0, faw_sf))

            summary_results.append(
                {
                    "station_id": station_id,
                    "month": m + 1,
                    "Monthly_PET": monthly_pe,
                    "SoilMoistureTotal": round(total_moisture, 3),
                    "FAW_0_100cm": round(faw_rz, 3),
                    "FAW_0_12_5cm": round(faw_sf, 3),
                    "spinup_converged": converged,
                    "spinup_iterations": len(spinup_records),
                }
            )

        station_dir = output_dir / str(station_id)
        station_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(station_dir / f"monthly_results_{station_id}.csv", index=False)

        if SAVE_SPINUP_INFO:
            spinup_df = pd.DataFrame(spinup_records)
            spinup_df.to_csv(station_dir / f"spinup_info_{station_id}.csv", index=False)

        print(
            f"{station_id}: monthly simulation complete. "
            f"spinup_converged={converged}, iterations={len(spinup_records)}"
        )

    except Exception as e:
        print(f"{station_id}: {e}")


def main() -> None:
    coord_df = pd.read_csv(BASE_IN / f"coordinates_final_{YEAR}.csv")
    station_ids = coord_df["station_id"].astype(str).str.lstrip("0").unique().tolist()

    for sid in station_ids:
        run_newhall_monthly(
            station_id=sid,
            coord_df=coord_df,
            monthly_input_dir=MONTHLY_INPUT_DIR,
            output_dir=BASE_OUT,
            reference_dir=DATA_REFERENCE_DIR,
        )


if __name__ == "__main__":
    main()