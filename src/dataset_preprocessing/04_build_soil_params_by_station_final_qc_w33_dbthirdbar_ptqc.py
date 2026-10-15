"""Build final gSSURGO soil inputs for the single-year P/T-QC station set.

Final soil-data rules
---------------------
1. Components with continuous horizon coverage over 0--100 cm must account
   for at least 80 percentage points of the map-unit component composition.
2. In every 12.5-cm model layer, valid records for each required property
   must account for at least 50 percentage points of the map-unit
   component-by-depth support.
3. Layer means use only valid records and renormalize
   component-percentage x layer-overlap weights. No cross-layer, map-unit
   mean, nearest-horizon, or generic-default imputation is used.

Field capacity is wthirdbar_r (33 kPa) for all soils in this version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import GSSURGO_GDB, OUTPUTS_DIR


YEAR = 1016
SOIL_TAG = "final_qc_w33_dbthirdbar_ptqc"

BASE_DIR = OUTPUTS_DIR / str(YEAR)
OUT_DIR = BASE_DIR / SOIL_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_CSV = OUTPUTS_DIR / "selected_station_year_1016_pt_qc.csv"
MAP_PQT = OUTPUTS_DIR / "station_mukey_map.parquet"

LAYER_EDGES_CM = np.array(
    [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0],
    dtype=float,
)
MIN_FULL_DEPTH_COMPONENT_PCT = 80.0
MIN_LAYER_PROPERTY_SUPPORT_PCT = 50.0
TOL = 1.0e-8

REQUIRED_PROPERTIES = {
    "theta_fc": "wthirdbar_r",
    "theta_wp": "wfifteenbar_r",
    "rho_b": "rho_b",
    "sand": "sandtotal_r",
    "clay": "claytotal_r",
    "organic_matter": "om_r",
}


def _as_key(series: pd.Series) -> pd.Series:
    """Convert gSSURGO identifiers to stable strings."""
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _sql_in(values: list[str]) -> str:
    escaped = ["'" + str(value).replace("'", "''") + "'" for value in values]
    return "(" + ",".join(escaped) + ")"


def interval_overlap(top: float, bottom: float, a: float, b: float) -> float:
    """Length of the overlap between [top, bottom) and [a, b), in cm."""
    if pd.isna(top) or pd.isna(bottom):
        return 0.0
    return max(0.0, min(float(bottom), b) - max(float(top), a))


def merged_intervals(
    horizons: pd.DataFrame,
    lower: float = 0.0,
    upper: float = 100.0,
) -> list[tuple[float, float]]:
    """Return merged horizon intervals clipped to the requested depth."""
    intervals: list[tuple[float, float]] = []
    for row in horizons[["hzdept_r", "hzdepb_r"]].itertuples(index=False):
        if pd.isna(row.hzdept_r) or pd.isna(row.hzdepb_r):
            continue
        top = max(lower, float(row.hzdept_r))
        bottom = min(upper, float(row.hzdepb_r))
        if bottom > top:
            intervals.append((top, bottom))

    intervals.sort()
    merged: list[list[float]] = []
    for top, bottom in intervals:
        if not merged or top > merged[-1][1] + TOL:
            merged.append([top, bottom])
        else:
            merged[-1][1] = max(merged[-1][1], bottom)
    return [(top, bottom) for top, bottom in merged]


def continuous_0_100(horizons: pd.DataFrame) -> bool:
    """Whether a component has uninterrupted horizon coverage over 0--100 cm."""
    intervals = merged_intervals(horizons)
    return bool(
        len(intervals) == 1
        and intervals[0][0] <= 0.0 + TOL
        and intervals[0][1] >= 100.0 - TOL
    )


def load_station_map() -> pd.DataFrame:
    selected = pd.read_csv(STATION_CSV, dtype={"station_id": "string"})
    selected["station_id"] = selected["station_id"].str.zfill(5)
    if "station" not in selected.columns:
        selected["station"] = selected["station_id"]

    station_mukey = pd.read_parquet(MAP_PQT)
    station_mukey["station_id"] = _as_key(station_mukey["station_id"]).str.zfill(5)
    station_mukey["mukey"] = _as_key(station_mukey["mukey"])
    station_mukey = station_mukey[["station_id", "mukey"]].drop_duplicates(
        "station_id"
    )

    result = selected.merge(station_mukey, on="station_id", how="left")
    if result["mukey"].isna().any():
        missing = result.loc[result["mukey"].isna(), "station_id"].tolist()
        raise RuntimeError(f"Missing station-to-mukey matches: {missing}")
    return result


def load_gssurgo(station_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mukeys = station_map["mukey"].dropna().astype(str).unique().tolist()
    component = gpd.read_file(
        GSSURGO_GDB,
        layer="component",
        engine="pyogrio",
        columns=["mukey", "cokey", "comppct_r"],
        where=f"mukey IN {_sql_in(mukeys)}",
        use_arrow=True,
    ).drop(columns="geometry", errors="ignore")

    component["mukey"] = _as_key(component["mukey"])
    component["cokey"] = _as_key(component["cokey"])
    component["component_pct"] = pd.to_numeric(
        component["comppct_r"], errors="coerce"
    )
    component = component.dropna(subset=["mukey", "cokey", "component_pct"])

    cokeys = component["cokey"].astype(str).unique().tolist()
    horizon_columns = [
        "chkey",
        "cokey",
        "hzdept_r",
        "hzdepb_r",
        "dbovendry_r",
        "dbthirdbar_r",
        "wthirdbar_r",
        "wfifteenbar_r",
        "sandtotal_r",
        "claytotal_r",
        "om_r",
    ]
    horizon = gpd.read_file(
        GSSURGO_GDB,
        layer="chorizon",
        engine="pyogrio",
        columns=horizon_columns,
        where=f"cokey IN {_sql_in(cokeys)}",
        use_arrow=True,
    ).drop(columns="geometry", errors="ignore")

    horizon["cokey"] = _as_key(horizon["cokey"])
    for column in horizon_columns[2:]:
        horizon[column] = pd.to_numeric(horizon[column], errors="coerce")

    # Prefer one-third-bar bulk density because its reference volume is
    # measured at a field-capacity-like water state. Use oven-dry bulk
    # density only when dbthirdbar_r is unavailable.
    horizon["rho_b"] = horizon["dbthirdbar_r"].where(
        horizon["dbthirdbar_r"].notna(), horizon["dbovendry_r"]
    )
    horizon["theta_fc"] = horizon["wthirdbar_r"] / 100.0
    horizon["theta_wp"] = horizon["wfifteenbar_r"] / 100.0
    horizon = horizon.merge(
        component[["mukey", "cokey", "component_pct"]],
        on="cokey",
        how="left",
        validate="many_to_one",
    )
    return component, horizon


def identify_full_depth_components(
    component: pd.DataFrame,
    horizon: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    for comp in component.itertuples(index=False):
        comp_horizon = horizon[horizon["cokey"] == comp.cokey]
        records.append(
            {
                "mukey": comp.mukey,
                "cokey": comp.cokey,
                "component_pct": float(comp.component_pct),
                "continuous_0_100": continuous_0_100(comp_horizon),
            }
        )
    return pd.DataFrame(records)


def layer_property_support_pct(
    horizons: pd.DataFrame,
    source_column: str,
    layer_top: float,
    layer_bottom: float,
) -> float:
    """Absolute map-unit component-by-depth support in one model layer."""
    layer_thickness = layer_bottom - layer_top
    weighted_valid_cm = 0.0
    valid = horizons[horizons[source_column].notna()]
    for row in valid.itertuples(index=False):
        overlap = interval_overlap(
            row.hzdept_r, row.hzdepb_r, layer_top, layer_bottom
        )
        weighted_valid_cm += overlap * float(row.component_pct)
    return weighted_valid_cm / layer_thickness


def weighted_layer_mean(
    horizons: pd.DataFrame,
    source_column: str,
    layer_top: float,
    layer_bottom: float,
) -> float:
    """Component-percentage x overlap weighted mean using valid records only."""
    numerator = 0.0
    denominator = 0.0
    valid = horizons[horizons[source_column].notna()]
    for row in valid.itertuples(index=False):
        overlap = interval_overlap(
            row.hzdept_r, row.hzdepb_r, layer_top, layer_bottom
        )
        if overlap <= 0:
            continue
        weight = float(row.component_pct) * overlap
        numerator += float(getattr(row, source_column)) * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else np.nan


def build_mapunit_qc(
    component_status: pd.DataFrame,
    horizon: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_records: list[dict] = []
    support_records: list[dict] = []

    for mukey, comps in component_status.groupby("mukey"):
        total_component_pct = float(comps["component_pct"].sum())
        full = comps[comps["continuous_0_100"]]
        full_depth_pct = float(full["component_pct"].sum())
        full_cokeys = set(full["cokey"])
        eligible = horizon[
            (horizon["mukey"] == mukey)
            & (horizon["cokey"].isin(full_cokeys))
            & (horizon["hzdept_r"] < 100.0)
            & (horizon["hzdepb_r"] > 0.0)
        ].copy()

        minimum_support = np.inf
        limiting_layer = pd.NA
        limiting_property = pd.NA
        for layer_index in range(8):
            layer_top = float(LAYER_EDGES_CM[layer_index])
            layer_bottom = float(LAYER_EDGES_CM[layer_index + 1])
            for property_name, source_column in REQUIRED_PROPERTIES.items():
                support = layer_property_support_pct(
                    eligible, source_column, layer_top, layer_bottom
                )
                support_records.append(
                    {
                        "mukey": mukey,
                        "layer": layer_index + 1,
                        "layer_top_cm": layer_top,
                        "layer_bottom_cm": layer_bottom,
                        "property": property_name,
                        "mapunit_support_pct": support,
                    }
                )
                if support < minimum_support:
                    minimum_support = support
                    limiting_layer = layer_index + 1
                    limiting_property = property_name

        structural_pass = full_depth_pct >= MIN_FULL_DEPTH_COMPONENT_PCT - TOL
        property_pass = minimum_support >= MIN_LAYER_PROPERTY_SUPPORT_PCT - TOL
        qc_records.append(
            {
                "mukey": mukey,
                "total_component_pct": total_component_pct,
                "full_depth_component_pct": full_depth_pct,
                "minimum_layer_property_support_pct": minimum_support,
                "limiting_layer": limiting_layer,
                "limiting_property": limiting_property,
                "pass_full_depth_component_80": structural_pass,
                "pass_all_layer_properties_50": property_pass,
                "soil_qc_pass": structural_pass and property_pass,
            }
        )

    return pd.DataFrame(qc_records), pd.DataFrame(support_records)


def build_layer_table(
    station_pass: pd.DataFrame,
    component_status: pd.DataFrame,
    horizon: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    for station_row in station_pass.itertuples(index=False):
        full_cokeys = set(
            component_status.loc[
                (component_status["mukey"] == station_row.mukey)
                & component_status["continuous_0_100"],
                "cokey",
            ]
        )
        eligible = horizon[
            (horizon["mukey"] == station_row.mukey)
            & (horizon["cokey"].isin(full_cokeys))
            & (horizon["hzdept_r"] < 100.0)
            & (horizon["hzdepb_r"] > 0.0)
        ].copy()

        record = {
            "station": station_row.station,
            "station_id": station_row.station_id,
            "year": station_row.year,
            "mukey": station_row.mukey,
            "soil_tag": SOIL_TAG,
        }
        for layer_index in range(8):
            k = layer_index + 1
            layer_top = float(LAYER_EDGES_CM[layer_index])
            layer_bottom = float(LAYER_EDGES_CM[layer_index + 1])

            fc = weighted_layer_mean(eligible, "theta_fc", layer_top, layer_bottom)
            wp = weighted_layer_mean(eligible, "theta_wp", layer_top, layer_bottom)
            rho_b = weighted_layer_mean(eligible, "rho_b", layer_top, layer_bottom)
            sand = weighted_layer_mean(eligible, "sandtotal_r", layer_top, layer_bottom)
            clay = weighted_layer_mean(eligible, "claytotal_r", layer_top, layer_bottom)
            om = weighted_layer_mean(eligible, "om_r", layer_top, layer_bottom)

            values = [fc, wp, rho_b, sand, clay, om]
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Non-finite layer input after QC: station={station_row.station_id}, "
                    f"layer={k}, values={values}"
                )

            record[f"wp_k{k}"] = wp
            record[f"fc_k{k}"] = fc
            record[f"dk_k{k}"] = layer_bottom - layer_top
            record[f"rhob_k{k}"] = rho_b
            record[f"sand_k{k}"] = sand
            record[f"clay_k{k}"] = clay
            record[f"silt_k{k}"] = float(np.clip(100.0 - sand - clay, 0.0, 100.0))
            record[f"om_k{k}"] = om
            record[f"awcmm_k{k}"] = max(fc - wp, 0.0) * (layer_bottom - layer_top) * 10.0

        records.append(record)
    return pd.DataFrame(records)


def build_theta_table(layers: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in layers.itertuples(index=False):
        fc = np.array([getattr(row, f"fc_k{k}") for k in range(1, 9)], dtype=float)
        wp = np.array([getattr(row, f"wp_k{k}") for k in range(1, 9)], dtype=float)
        records.append(
            {
                "station": row.station,
                "station_id": row.station_id,
                "year": row.year,
                "mukey": row.mukey,
                "soil_tag": SOIL_TAG,
                "surface_theta_fc": fc[0],
                "surface_theta_wp": wp[0],
                "rootzone_theta_fc": float(fc.mean()),
                "rootzone_theta_wp": float(wp.mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    station_map = load_station_map()
    component, horizon = load_gssurgo(station_map)
    component_status = identify_full_depth_components(component, horizon)
    mapunit_qc, support = build_mapunit_qc(component_status, horizon)

    station_qc = station_map.merge(mapunit_qc, on="mukey", how="left")
    station_qc["soil_qc_pass"] = station_qc["soil_qc_pass"].fillna(False)
    station_pass = station_qc[station_qc["soil_qc_pass"]].copy()
    
    support = station_map[["station", "station_id", "year", "mukey"]].merge(
        support, on="mukey", how="left"
    )
    layers = build_layer_table(station_pass, component_status, horizon)
    theta = build_theta_table(layers)

    station_qc.to_csv(OUT_DIR / f"station_soil_qc_{YEAR}_{SOIL_TAG}.csv", index=False)
    support.to_csv(
        OUT_DIR / f"layer_property_support_{YEAR}_{SOIL_TAG}.csv", index=False
    )
    station_pass[["station", "station_id", "year", "mukey"]].to_csv(
        OUT_DIR / f"selected_station_year_{YEAR}_{SOIL_TAG}.csv", index=False
    )
    layers.to_csv(
        OUT_DIR / f"layers_by_station_{YEAR}_{SOIL_TAG}.csv", index=False
    )
    theta.to_csv(
        OUT_DIR / f"theta_by_station_final_{YEAR}_{SOIL_TAG}.csv", index=False
    )

    print(f"Candidate stations: {len(station_qc)}")
    print(f"Passed final soil QC: {len(station_pass)}")
    print(f"Saved final soil inputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()

