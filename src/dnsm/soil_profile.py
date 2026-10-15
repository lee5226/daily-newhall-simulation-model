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


DK_CM = 12.5

CONSTANTS_CSV = DATA_REFERENCE_DIR / "constants_classic.csv"
constants_df = pd.read_csv(CONSTANTS_CSV)

accretion_order = np.arange(64)
depletion_order = constants_df["depletion_order"].to_numpy(copy=True) - 1
depletion_req = constants_df["depletion_req"].to_numpy()


def capacities_from_layers(wp_k, fc_k, sat_k, dk_cm=None):
    """
    Build layer capacities and slot capacities from layer-scale theta values.

    Returns:
        C1_k: storage from wp to fc for each of 8 layers (mm)
        C2_k: storage from fc to sat for each of 8 layers (mm)
        w1_slot_arr: 64-slot capacities for C1 (mm)
        w2_slot_arr: 64-slot capacities for C2 (mm)
    """
    wp_k = np.asarray(wp_k, float)
    fc_k = np.asarray(fc_k, float)
    sat_k = np.asarray(sat_k, float)

    if dk_cm is None:
        dk = np.full(8, DK_CM, float)
    else:
        dk = np.asarray(dk_cm, float)
        dk[:] = DK_CM

    C1_k = np.maximum(fc_k - wp_k, 0.0) * dk * 10.0
    C2_k = np.maximum(sat_k - fc_k, 0.0) * dk * 10.0

    w1_slot_arr = np.repeat(C1_k / 8.0, 8)
    w2_slot_arr = np.repeat(C2_k / 8.0, 8)

    return C1_k, C2_k, w1_slot_arr, w2_slot_arr


def apply_water_change(amount, soil_profile, water_per_slot, total_capacity):
    """
    Apply a daily water balance to a single 64-slot soil profile.
    """
    if amount > 0:
        current_nma = amount
        for pos in accretion_order:
            to_fill = max(0.0, water_per_slot - soil_profile[pos])
            if current_nma > to_fill:
                soil_profile[pos] += to_fill
                current_nma -= to_fill
            else:
                soil_profile[pos] += current_nma
                current_nma = 0.0

            if np.sum(soil_profile) >= total_capacity:
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
                remaining_energy_proportion = current_nma / energy_to_drain_slot
                soil_profile[pos] = remaining_energy_proportion * water_in_slot
                current_nma = 0.0

            if current_nma == 0 or np.sum(soil_profile) == 0:
                break


def apply_water_change_dual(
    amount_mm: float,
    soil_s1: np.ndarray,
    soil_s2: np.ndarray,
    w1_per_slot: float,
    w2_per_slot: float,
    C1: float,
    C2: float,
    alpha_fast: float = 0.4,
    acc_order=None,
    dep_order=None,
    dep_req=None,
):
    """
    Apply daily water balance to dual storage profiles.

    s1 stores water between wp and fc.
    s2 stores water between fc and sat.
    """
    if acc_order is None:
        acc_order = accretion_order
    if dep_order is None:
        dep_order = depletion_order
    if dep_req is None:
        dep_req = depletion_req

    C1 = max(C1, 0.0)
    C2 = max(C2, 0.0)
    w1 = max(w1_per_slot, 0.0)
    w2 = max(w2_per_slot, 0.0)

    if amount_mm > 0.0:
        inflow = amount_mm

        if C1 > 0.0:
            for pos in acc_order:
                if inflow <= 0.0:
                    break
                cap = w1 - soil_s1[pos]
                if cap <= 1e-12:
                    continue
                d = cap if inflow >= cap else inflow
                soil_s1[pos] += d
                inflow -= d

        if inflow > 0.0 and C2 > 0.0:
            for pos in acc_order:
                if inflow <= 0.0:
                    break
                cap = w2 - soil_s2[pos]
                if cap <= 1e-12:
                    continue
                d = cap if inflow >= cap else inflow
                soil_s2[pos] += d
                inflow -= d

    if C2 > 0.0:
        s2_total = float(soil_s2.sum())
        if s2_total > 1e-12 and alpha_fast > 0.0:
            drain = min(alpha_fast * s2_total, s2_total)
            frac = drain / s2_total
            soil_s2 *= max(1.0 - frac, 0.0)

    if amount_mm < 0.0 and C1 > 0.0:
        demand = -amount_mm
        cur = demand
        for i, pos in enumerate(dep_order):
            water_in = soil_s1[pos]
            if water_in <= 1e-12:
                continue
            energy = water_in * dep_req[i]
            if cur > energy:
                soil_s1[pos] = 0.0
                cur -= energy
            else:
                ratio_left = max(1.0 - cur / max(energy, 1e-12), 0.0)
                soil_s1[pos] = water_in * ratio_left
                cur = 0.0
            if cur <= 0.0:
                break

    if C1 > 0.0:
        np.clip(soil_s1, 0.0, w1, out=soil_s1)
    else:
        soil_s1[:] = 0.0

    if C2 > 0.0:
        np.clip(soil_s2, 0.0, w2, out=soil_s2)
    else:
        soil_s2[:] = 0.0


def theta_by_layer_from_slots(s1_slots, s2_slots, C1_k, C2_k, wp_k, fc_k, sat_k, dk_cm=None):
    """
    Convert slot-level storage to layer theta values and aggregated VWC.
    """
    s1_layers = np.asarray(s1_slots, float).reshape(8, 8).sum(axis=1)
    s2_layers = np.asarray(s2_slots, float).reshape(8, 8).sum(axis=1)

    C1 = np.asarray(C1_k, float)
    C2 = np.asarray(C2_k, float)
    wp = np.asarray(wp_k, float)
    fc = np.asarray(fc_k, float)
    sat = np.asarray(sat_k, float)

    f1 = np.divide(s1_layers, C1, out=np.zeros_like(C1), where=C1 > 0)
    f2 = np.divide(s2_layers, C2, out=np.zeros_like(C2), where=C2 > 0)

    theta_k = wp + f1 * (fc - wp) + f2 * (sat - fc)
    theta_surface = float(theta_k[0])

    if dk_cm is None:
        dk = np.full(8, DK_CM, float)
    else:
        dk = np.asarray(dk_cm, float)
        dk[:] = DK_CM

    theta_root = float((theta_k * dk).sum() / dk.sum()) if dk.sum() > 0 else float("nan")
    return theta_k, theta_surface, theta_root


def apply_water_change_dual_slots(
    amount_mm,
    s1_slots,
    s2_slots,
    w1_slot_arr,
    w2_slot_arr,
    C1_total,
    C2_total,
    acc_order=None,
    dep_order=None,
    dep_req=None,
    alpha_fast=0.4,
):
    """
    Apply daily water balance to dual-storage slot arrays.
    """
    inflow = float(amount_mm)

    if acc_order is None:
        acc_order = accretion_order
    if dep_order is None:
        dep_order = depletion_order
    if dep_req is None:
        dep_req = depletion_req

    if inflow > 0:
        for idx in acc_order:
            if inflow <= 0:
                break
            cap = w1_slot_arr[idx] - s1_slots[idx]
            if cap <= 1e-12:
                continue
            d = cap if inflow >= cap else inflow
            s1_slots[idx] += d
            inflow -= d

        if inflow > 0:
            for idx in acc_order:
                if inflow <= 0:
                    break
                cap = w2_slot_arr[idx] - s2_slots[idx]
                if cap <= 1e-12:
                    continue
                d = cap if inflow >= cap else inflow
                s2_slots[idx] += d
                inflow -= d

    s2_sum = float(s2_slots.sum())
    if s2_sum > 1e-12 and alpha_fast > 0.0:
        drain = min(alpha_fast * s2_sum, s2_sum)
        frac = drain / s2_sum
        s2_slots *= max(1.0 - frac, 0.0)

    if amount_mm < 0:
        demand = -amount_mm
        for i, idx in enumerate(dep_order):
            if demand <= 0:
                break
            water_in = s1_slots[idx]
            if water_in <= 1e-12:
                continue
            energy = water_in * dep_req[i]
            if demand >= energy:
                s1_slots[idx] = 0.0
                demand -= energy
            else:
                s1_slots[idx] = water_in * max(1.0 - demand / energy, 0.0)
                demand = 0.0
        inflow = -demand

    np.clip(s1_slots, 0.0, w1_slot_arr, out=s1_slots)
    np.clip(s2_slots, 0.0, w2_slot_arr, out=s2_slots)

    return s1_slots, s2_slots, inflow