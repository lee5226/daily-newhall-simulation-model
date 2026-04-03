import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import GSSURGO_GDB, OUTPUTS_DIR


YEAR = 1016
GDB = GSSURGO_GDB
OUT_DIR = OUTPUTS_DIR / str(YEAR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PQT = OUTPUTS_DIR / "station_mukey_map.parquet"
STATION_CSV = OUTPUTS_DIR / "selected_station_year.csv"

LAYER_EDGES = np.array([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100.0], dtype=float)
FIXED_DK_CM = 12.5

EPS_SATFC = 0.02
EPS_FCWP = 0.01
TOL_AWC_ABS = 0.02
TOL_AWC_REL = 0.25

USDA_CLASSES = [
    "sandy clay loam",
    "silty clay loam",
    "clay loam",
    "sandy clay",
    "silty clay",
    "loamy sand",
    "sandy loam",
    "silt loam",
    "sand",
    "silt",
    "clay",
    "loam",
]
USDA_CLASSES.sort(key=len, reverse=True)


def bin_horizons_to_layers(hz_top_cm, hz_bot_cm, values, weights):
    """
    Project horizon values to the eight fixed 12.5 cm layers and return
    thickness-weighted layer means and overlapped thickness.
    """
    hz_top_cm = np.asarray(hz_top_cm, float)
    hz_bot_cm = np.asarray(hz_bot_cm, float)
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)

    layer_vals = np.zeros(8, float)
    layer_wsum = np.zeros(8, float)
    layer_thk = np.zeros(8, float)

    for i in range(8):
        a, b = LAYER_EDGES[i], LAYER_EDGES[i + 1]
        for ht, hb, v, w in zip(hz_top_cm, hz_bot_cm, values, weights):
            overlap = max(0.0, min(hb, b) - max(ht, a))
            if overlap <= 0:
                continue
            ww = w * overlap
            layer_vals[i] += v * ww
            layer_wsum[i] += ww
            layer_thk[i] += overlap

    with np.errstate(invalid="ignore", divide="ignore"):
        layer_vals = np.where(layer_wsum > 0, layer_vals / layer_wsum, np.nan)

    return layer_vals, layer_thk


def enforce_agg(prefix: str, depth_cm: float, rec: dict) -> tuple[dict, dict]:
    """
    Apply physical and consistency constraints to aggregated values and
    store before/after information in a flags dictionary.
    """
    import math

    def _isnan(x) -> bool:
        try:
            return math.isnan(x)
        except Exception:
            return False

    fc0 = rec.get(f"{prefix}_theta_fc")
    wp0 = rec.get(f"{prefix}_theta_wp")
    sat0 = rec.get(f"{prefix}_theta_sat")
    aws_mm = rec.get(f"{prefix}_aws")

    flags = {
        f"{prefix}_agg_action": "none",
        f"{prefix}_agg_satfix": False,
        f"{prefix}_fc_before": fc0,
        f"{prefix}_wp_before": wp0,
        f"{prefix}_sat_before": sat0,
        f"{prefix}_agg_diff_before": None,
        f"{prefix}_agg_desired": None,
        f"{prefix}_agg_abs_err": None,
        f"{prefix}_agg_rel_err": None,
        f"{prefix}_fc_after": None,
        f"{prefix}_wp_after": None,
        f"{prefix}_sat_after": None,
        f"{prefix}_agg_diff_after": None,
    }

    if (fc0 is None) or (wp0 is None) or _isnan(fc0) or _isnan(wp0):
        flags[f"{prefix}_fc_after"] = fc0
        flags[f"{prefix}_wp_after"] = wp0
        flags[f"{prefix}_sat_after"] = sat0
        return rec, flags

    diff = fc0 - wp0
    desired = EPS_FCWP
    if (aws_mm is not None) and (not _isnan(aws_mm)):
        desired = max(aws_mm / (depth_cm * 10.0), EPS_FCWP)

    abs_err = abs(diff - desired)
    rel_err = abs_err / max(abs(desired), 1e-6)

    flags[f"{prefix}_agg_diff_before"] = diff
    flags[f"{prefix}_agg_desired"] = desired
    flags[f"{prefix}_agg_abs_err"] = abs_err
    flags[f"{prefix}_agg_rel_err"] = rel_err

    fc, wp, sat = fc0, wp0, sat0

    if (diff + TOL_AWC_ABS < desired) or (rel_err > TOL_AWC_REL and diff < desired):
        wp = max(0.0, fc - desired)
        flags[f"{prefix}_agg_action"] = "wpdown"
    elif (diff - TOL_AWC_ABS > desired) or (rel_err > TOL_AWC_REL and diff > desired):
        fc = max(0.0, wp + desired)
        flags[f"{prefix}_agg_action"] = "fcdown"

    if (fc - wp) < EPS_FCWP:
        wp = max(0.0, fc - EPS_FCWP)
        if flags[f"{prefix}_agg_action"] == "none":
            flags[f"{prefix}_agg_action"] = "wpdown_min"

    if (sat is not None) and (not _isnan(sat)) and (sat < fc + EPS_SATFC):
        sat = fc + EPS_SATFC
        flags[f"{prefix}_agg_satfix"] = True

    rec[f"{prefix}_theta_fc"] = fc
    rec[f"{prefix}_theta_wp"] = wp
    rec[f"{prefix}_theta_sat"] = sat

    flags[f"{prefix}_fc_after"] = fc
    flags[f"{prefix}_wp_after"] = wp
    flags[f"{prefix}_sat_after"] = sat
    flags[f"{prefix}_agg_diff_after"] = fc - wp

    return rec, flags


def overlap(top: float, bot: float, depth_max: float) -> float:
    """
    Return overlap between horizon [top, bot) and [0, depth_max).
    """
    if pd.isna(top) or pd.isna(bot):
        return 0.0
    return max(0.0, min(bot, depth_max) - max(0.0, top))


def summarize(df: pd.DataFrame, depth_cm: float, prefix: str) -> dict:
    """
    Return thickness × component-weighted summaries up to a target depth.
    """
    empty_result = {
        f"{prefix}_aws": pd.NA,
        f"{prefix}_theta_sat": pd.NA,
        f"{prefix}_rho_b": pd.NA,
        f"{prefix}_theta_fc": pd.NA,
        f"{prefix}_theta_wp": pd.NA,
    }

    if df.empty:
        return empty_result

    df = df.copy()
    df["thick_used"] = df.apply(lambda r: overlap(r.hzdept_r, r.hzdepb_r, depth_cm), axis=1)
    df = df[df["thick_used"] > 0]
    if df.empty:
        return empty_result

    w = df["thick_used"] * df["comppct"]
    wsum = w.sum()
    if wsum <= 0:
        return empty_result

    aws_mm = (df["awc_r"] * df["thick_used"] * 10 * df["comppct"]).sum()
    theta_sat_w = (df["theta_sat"] * w).sum() / wsum
    rho_b_w = (df["rho_b"] * w).sum() / wsum
    theta_fc_w = (df["theta_fc"] * w).sum() / wsum
    theta_wp_w = (df["theta_wp"] * w).sum() / wsum

    return {
        f"{prefix}_aws": aws_mm,
        f"{prefix}_theta_sat": theta_sat_w,
        f"{prefix}_rho_b": rho_b_w,
        f"{prefix}_theta_fc": theta_fc_w,
        f"{prefix}_theta_wp": theta_wp_w,
    }


def choose_rho_p(row) -> float:
    """
    Select particle density using partdensity when available, otherwise
    use a simple fallback rule based on organic matter and bulk density.
    """
    rp = row.get("partdensity", np.nan)
    if pd.notna(rp) and 1.2 <= rp <= 2.75:
        return float(rp)

    om = row.get("om_r", np.nan)
    bd = row.get("rho_b", np.nan)

    if (pd.notna(om) and om >= 20) or (pd.notna(bd) and bd <= 1.0):
        return 1.4
    if pd.notna(om) and 8 <= om < 20:
        return 2.4
    return 2.65


def pick_usda_class(desc: str | None) -> str | None:
    """
    Map a texture description string to a USDA texture class using
    substring matching.
    """
    if not isinstance(desc, str):
        return None

    s = desc.lower()
    for cls in USDA_CLASSES:
        if cls in s:
            return cls.title()
    return None


def summarize_texture_om(df: pd.DataFrame, depth_cm: float) -> tuple:
    """
    Return weighted average sand, silt, clay, and OM percentages.
    """
    if df.empty:
        return (pd.NA,) * 4

    df = df.copy()
    df["thick_used"] = df.apply(lambda r: overlap(r.hzdept_r, r.hzdepb_r, depth_cm), axis=1)
    df = df[df["thick_used"] > 0]
    if df.empty:
        return (pd.NA,) * 4

    w = df["thick_used"] * df["comppct"]
    tw = w.sum()
    if tw <= 0:
        return (pd.NA,) * 4

    sand = (df["sandtotal_r"] * w).sum() / tw
    silt = (df["silttotal_r"] * w).sum() / tw
    clay = (df["claytotal_r"] * w).sum() / tw
    om = (df["om_r"] * w).sum() / tw

    return sand, silt, clay, om


def main() -> None:
    stations = pd.read_csv(STATION_CSV)["station_id"].astype(str).str.zfill(5).tolist()

    year_map = (
        pd.read_csv(STATION_CSV)[["station_id", "year"]]
        .assign(station_id=lambda d: d.station_id.astype(str).str.zfill(5))
    )

    sta_mk = pd.read_parquet(MAP_PQT)
    sta_mk["station_id"] = sta_mk["station_id"].astype(str).str.zfill(5)
    sta_mk = sta_mk[sta_mk["station_id"].isin(stations)]

    valid_mukeys = sta_mk["mukey"].dropna().astype(str).unique()
    valid_mukeys_str = [f"'{mk}'" for mk in valid_mukeys]
    where_clause_mukey = f"mukey IN ({','.join(valid_mukeys_str)})"

    comp_cols = ["mukey", "cokey", "comppct_r"]
    component = gpd.read_file(
        GDB,
        layer="component",
        engine="pyogrio",
        columns=comp_cols,
        where=where_clause_mukey,
        use_arrow=True,
    ).drop(columns="geometry", errors="ignore")
    component["comppct"] = component["comppct_r"] / 100.0

    valid_cokeys = component["cokey"].astype(str).unique()
    valid_cokeys_str = [f"'{ck}'" for ck in valid_cokeys]
    where_clause_cokey = f"cokey IN ({','.join(valid_cokeys_str)})"

    hz_cols_required = [
        "chkey",
        "cokey",
        "hzdept_r",
        "hzdepb_r",
        "awc_r",
        "dbovendry_r",
        "dbthirdbar_r",
        "wthirdbar_r",
        "wfifteenbar_r",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "om_r",
    ]
    hz_cols_optional = ["partdensity"]

    try:
        chorizon = gpd.read_file(
            GDB,
            layer="chorizon",
            engine="pyogrio",
            columns=hz_cols_required + hz_cols_optional,
            where=where_clause_cokey,
            use_arrow=True,
        )
    except Exception:
        chorizon = gpd.read_file(
            GDB,
            layer="chorizon",
            engine="pyogrio",
            columns=hz_cols_required,
            where=where_clause_cokey,
            use_arrow=True,
        )
        for col in hz_cols_optional:
            chorizon[col] = np.nan

    chorizon = chorizon.drop(columns="geometry", errors="ignore")
    chorizon = chorizon.merge(component[["cokey", "mukey", "comppct"]], on="cokey", how="left")

    rho_b = np.where(
        chorizon["dbovendry_r"].notna(),
        chorizon["dbovendry_r"],
        chorizon["dbthirdbar_r"],
    )
    chorizon["rho_b"] = rho_b
    chorizon["rho_p"] = chorizon.apply(choose_rho_p, axis=1)

    chorizon["theta_fc"] = chorizon["wthirdbar_r"] / 100.0
    chorizon["theta_wp"] = chorizon["wfifteenbar_r"] / 100.0

    theta_sat_raw = 1.0 - (chorizon["rho_b"] / chorizon["rho_p"])
    chorizon["theta_sat"] = np.clip(0.90 * theta_sat_raw, 0.0, 1.0)

    records: list[dict] = []
    records_before: list[dict] = []
    qc_flags: list[dict] = []

    null_cols = ["awc_r", "theta_fc", "theta_wp", "rho_b"]

    for mukey, grp in chorizon.groupby("mukey"):
        rec = {"mukey": mukey}

        pct_sum = component.loc[component["mukey"] == mukey, "comppct"].sum()
        pct_ok = pct_sum >= 0.80

        thick100 = grp.apply(lambda r: overlap(r.hzdept_r, r.hzdepb_r, 100), axis=1).sum()
        thick_ok = thick100 >= 95

        grp_100 = grp[(grp["hzdept_r"] < 100) & (grp["hzdepb_r"] > 0)].copy()

        grp_top = grp_100[(grp_100["hzdept_r"] < 50) & (grp_100["hzdepb_r"] > 0)]
        top_null_any = grp_top[null_cols].isna().any().any()
        top_ok = not top_null_any

        grp_bot = grp_100[(grp_100["hzdept_r"] < 100) & (grp_100["hzdepb_r"] > 50)]
        bot_null_frac = grp_bot[null_cols].isna().any(axis=1).mean() if len(grp_bot) > 0 else 0.0
        bottom_ok = bot_null_frac <= 0.20

        null_any = not (top_ok and bottom_ok)

        bottom_note_cols = ""
        fill_vals = {c: None for c in null_cols}
        grp_100_filled = None

        if not null_any:
            group_all = chorizon[chorizon["mukey"] == mukey]
            mean_vals = group_all[null_cols].mean(skipna=True).to_dict()

            grp_100_filled = grp_100.copy()
            for col in null_cols:
                mask = grp_100_filled[col].isna()
                if mask.any():
                    grp_100_filled.loc[mask, col] = mean_vals[col]
                    fill_vals[col] = mean_vals[col]

            if len(grp_bot) > 0:
                bottom_null_cols = grp_bot[null_cols].isna().any()
                bottom_note_cols = ",".join(bottom_null_cols.index[bottom_null_cols].tolist())

        src_df = grp_100_filled if grp_100_filled is not None else grp_100
        rec.update(summarize(src_df, 12.5, "surface"))
        rec.update(summarize(src_df, 100.0, "rootzone"))

        grp_layers = grp[(grp["hzdept_r"] < 100) & (grp["hzdepb_r"] > 0)].copy()
        if not grp_layers.empty:
            hz_top = grp_layers["hzdept_r"].to_numpy(float)
            hz_bot = grp_layers["hzdepb_r"].to_numpy(float)
            thk = hz_bot - hz_top
            w_comp = grp_layers["comppct"].to_numpy(float)
            w_base = thk * w_comp

            v_fc = grp_layers["theta_fc"].to_numpy(float)
            v_wp = grp_layers["theta_wp"].to_numpy(float)
            v_sat = grp_layers["theta_sat"].to_numpy(float)
            v_rhob = grp_layers["rho_b"].to_numpy(float)
            v_rhop = grp_layers["rho_p"].to_numpy(float)
            v_sand = grp_layers["sandtotal_r"].to_numpy(float)
            v_silt = grp_layers["silttotal_r"].to_numpy(float)
            v_clay = grp_layers["claytotal_r"].to_numpy(float)
            v_om = grp_layers["om_r"].to_numpy(float)
            v_awc = grp_layers["awc_r"].to_numpy(float)

            fc_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_fc, w_base)
            wp_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_wp, w_base)
            sat_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_sat, w_base)
            rhob_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_rhob, w_base)
            rhop_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_rhop, w_base)
            sand_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_sand, w_base)
            silt_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_silt, w_base)
            clay_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_clay, w_base)
            om_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_om, w_base)
            awcmm_layers, _ = bin_horizons_to_layers(hz_top, hz_bot, v_awc, w_comp * 10.0)

            def _fill_nan(a, fill=0.0):
                if np.all(np.isnan(a)):
                    return np.zeros_like(a) + fill
                m = np.nanmean(a) if not np.isnan(np.nanmean(a)) else fill
                return np.where(np.isnan(a), m, a)

            fc_layers = np.clip(_fill_nan(fc_layers, 0.25), 0.00, 0.60)
            wp_layers = np.clip(_fill_nan(wp_layers, 0.12), 0.00, 0.40)
            sat_layers = np.clip(_fill_nan(sat_layers, 0.45), 0.20, 0.85)
            rhob_layers = _fill_nan(rhob_layers, 1.4)
            rhop_layers = _fill_nan(rhop_layers, 2.65)
            sand_layers = _fill_nan(sand_layers, 45.0)
            silt_layers = _fill_nan(silt_layers, 35.0)
            clay_layers = _fill_nan(clay_layers, 20.0)
            om_layers = _fill_nan(om_layers, 2.0)
            awcmm_layers = _fill_nan(awcmm_layers, 0.0)

            for k in range(8):
                rec[f"wp_k{k+1}"] = float(wp_layers[k])
                rec[f"fc_k{k+1}"] = float(fc_layers[k])
                rec[f"sat_k{k+1}"] = float(sat_layers[k])
                rec[f"dk_k{k+1}"] = FIXED_DK_CM
                rec[f"rhob_k{k+1}"] = float(rhob_layers[k])
                rec[f"rhop_k{k+1}"] = float(rhop_layers[k])
                rec[f"sand_k{k+1}"] = float(sand_layers[k])
                rec[f"silt_k{k+1}"] = float(silt_layers[k])
                rec[f"clay_k{k+1}"] = float(clay_layers[k])
                rec[f"om_k{k+1}"] = float(om_layers[k])
                rec[f"awcmm_k{k+1}"] = float(awcmm_layers[k])
        else:
            sf_fc = rec.get("surface_theta_fc", 0.25)
            sf_wp = rec.get("surface_theta_wp", 0.12)
            sf_sat = rec.get("surface_theta_sat", 0.45)

            for k in range(8):
                rec[f"wp_k{k+1}"] = float(sf_wp)
                rec[f"fc_k{k+1}"] = float(sf_fc)
                rec[f"sat_k{k+1}"] = float(sf_sat)
                rec[f"dk_k{k+1}"] = FIXED_DK_CM
                rec[f"rhob_k{k+1}"] = 1.4
                rec[f"rhop_k{k+1}"] = 2.65
                rec[f"sand_k{k+1}"] = 45.0
                rec[f"silt_k{k+1}"] = 35.0
                rec[f"clay_k{k+1}"] = 20.0
                rec[f"om_k{k+1}"] = 2.0
                rec[f"awcmm_k{k+1}"] = 0.0

        rec_before = rec.copy()

        rec, surf_flags = enforce_agg("surface", 12.5, rec)
        rec, root_flags = enforce_agg("rootzone", 100.0, rec)

        records.append(rec)
        records_before.append(rec_before)

        qc_flags.append(
            {
                "mukey": mukey,
                "pct_sum": round(float(pct_sum), 3),
                "thick_used_cm": round(float(thick100), 1),
                "pct_ok": pct_ok,
                "thick_ok": thick_ok,
                "null_any": null_any,
                "awc_r_fill": fill_vals.get("awc_r"),
                "theta_fc_fill": fill_vals.get("theta_fc"),
                "theta_wp_fill": fill_vals.get("theta_wp"),
                "rho_b_fill": fill_vals.get("rho_b"),
                "bottom_note": bottom_note_cols,
                **surf_flags,
                **root_flags,
            }
        )

    params_df = pd.DataFrame(records)
    params_df_before = pd.DataFrame(records_before)

    qc_df = pd.DataFrame(qc_flags).merge(
        sta_mk[["mukey", "station", "station_id"]],
        on="mukey",
        how="left",
    )
    qc_df = qc_df[["station", "station_id"] + [c for c in qc_df.columns if c not in ["station", "station_id"]]]
    qc_df.to_csv(OUT_DIR / f"aws_theta_QC_flags_{YEAR}.csv", index=False)

    valid_mukeys_qc = qc_df.query("pct_sum >= 0.8 and thick_used_cm >= 95 and not null_any")["mukey"]
    params_df = params_df[params_df["mukey"].isin(valid_mukeys_qc)]
    params_df_before = params_df_before[params_df_before["mukey"].isin(valid_mukeys_qc)]

    sta_mk_valid = sta_mk[sta_mk["mukey"].isin(valid_mukeys_qc)][["station", "station_id", "mukey"]]

    aws_df = (
        sta_mk_valid.merge(params_df[["mukey", "rootzone_aws"]], on="mukey", how="left")
        .rename(columns={"rootzone_aws": "aws_0_100cm"})
    )

    val_cols = ["mukey", "aws0_5", "aws5_20", "aws20_50", "aws50_100"]
    valu1 = gpd.read_file(
        GDB,
        layer="Valu1",
        engine="pyogrio",
        columns=val_cols,
        where=where_clause_mukey,
        use_arrow=True,
    ).drop(columns="geometry", errors="ignore").rename(
        columns={
            "aws0_5": "aws_val_0_5",
            "aws5_20": "aws_val_5_20",
            "aws20_50": "aws_val_20_50",
            "aws50_100": "aws_val_50_100",
        }
    )

    aws_df = aws_df.merge(
        valu1[["mukey", "aws_val_0_5", "aws_val_5_20", "aws_val_20_50", "aws_val_50_100"]],
        on="mukey",
        how="left",
    )
    aws_df = aws_df.merge(year_map, on="station_id", how="left")
    aws_df = aws_df[["station", "station_id", "year"] + [c for c in aws_df.columns if c not in ["station", "station_id", "year"]]]
    aws_df.to_csv(OUT_DIR / f"aws_by_station_final_{YEAR}.csv", index=False)

    theta_df = sta_mk_valid.merge(
        params_df[
            [
                "mukey",
                "surface_theta_sat",
                "surface_theta_fc",
                "surface_theta_wp",
                "surface_rho_b",
                "rootzone_theta_sat",
                "rootzone_theta_fc",
                "rootzone_theta_wp",
                "rootzone_rho_b",
            ]
        ],
        on="mukey",
        how="left",
    )
    theta_df = theta_df.merge(year_map, on="station_id", how="left")
    theta_df = theta_df[["station", "station_id", "year"] + [c for c in theta_df.columns if c not in ["station", "station_id", "year"]]]
    theta_df.to_csv(OUT_DIR / f"theta_by_station_final_{YEAR}.csv", index=False)

    print(f"Saved: {OUT_DIR / f'aws_by_station_final_{YEAR}.csv'}")
    print(f"Saved: {OUT_DIR / f'theta_by_station_final_{YEAR}.csv'}")

    pre_cols_map = {
        "surface_fc_before": "surface_theta_fc",
        "surface_wp_before": "surface_theta_wp",
        "surface_sat_before": "surface_theta_sat",
        "rootzone_fc_before": "rootzone_theta_fc",
        "rootzone_wp_before": "rootzone_theta_wp",
        "rootzone_sat_before": "rootzone_theta_sat",
    }

    theta_df_before = (
        qc_df.loc[
            qc_df["mukey"].isin(valid_mukeys_qc),
            ["mukey", "station", "station_id"] + list(pre_cols_map.keys()),
        ]
        .rename(columns=pre_cols_map)
        .merge(year_map, on="station_id", how="left")
        .merge(params_df_before[["mukey", "surface_rho_b", "rootzone_rho_b"]], on="mukey", how="left")
        .drop(columns=["mukey"])
    )

    theta_df_before = theta_df_before[
        [
            "station",
            "station_id",
            "year",
            "surface_theta_sat",
            "surface_theta_fc",
            "surface_theta_wp",
            "surface_rho_b",
            "rootzone_theta_sat",
            "rootzone_theta_fc",
            "rootzone_theta_wp",
            "rootzone_rho_b",
        ]
    ]

    for c_fc, c_sat in [
        ("surface_theta_fc", "surface_theta_sat"),
        ("rootzone_theta_fc", "rootzone_theta_sat"),
    ]:
        mask = theta_df_before[c_sat].astype(float) < (theta_df_before[c_fc].astype(float) + EPS_SATFC)
        theta_df_before.loc[mask, c_sat] = theta_df_before.loc[mask, c_fc] + EPS_SATFC
        print(f"{c_sat} sat-fix count: {int(mask.sum())}")

    theta_df_before.to_csv(OUT_DIR / f"theta_by_station_final_{YEAR}_before_90.csv", index=False)
    print(f"Saved: {OUT_DIR / f'theta_by_station_final_{YEAR}_before_90.csv'}")

    layer_cols = []
    for i in range(1, 9):
        layer_cols += [
            f"wp_k{i}",
            f"fc_k{i}",
            f"sat_k{i}",
            f"dk_k{i}",
            f"rhob_k{i}",
            f"rhop_k{i}",
            f"sand_k{i}",
            f"silt_k{i}",
            f"clay_k{i}",
            f"om_k{i}",
            f"awcmm_k{i}",
        ]

    layers_df = (
        sta_mk_valid.merge(params_df[["mukey"] + layer_cols], on="mukey", how="left")
        .merge(year_map, on="station_id", how="left")
    )
    layers_df = layers_df[["station", "station_id", "year"] + layer_cols]
    layers_df.to_csv(OUT_DIR / f"layers_by_station_{YEAR}.csv", index=False)
    print(f"Saved: {OUT_DIR / f'layers_by_station_{YEAR}.csv'}")

    tex_cols = ["chkey", "rvindicator", "texdesc"]
    valid_chkeys = chorizon["chkey"].astype(str).unique()
    valid_chkeys_str = [f"'{ck}'" for ck in valid_chkeys]
    where_clause_chkey = f"chkey IN ({','.join(valid_chkeys_str)})"

    chtexgrp = gpd.read_file(
        GDB,
        layer="chtexturegrp",
        engine="pyogrio",
        columns=tex_cols,
        where=where_clause_chkey,
        use_arrow=True,
    ).drop(columns="geometry", errors="ignore")

    chtexgrp["is_rv"] = chtexgrp["rvindicator"].astype(str).str.upper().isin(["Y", "YES", "1"])
    rep_tex = (
        chtexgrp.sort_values("is_rv", ascending=False)
        .drop_duplicates("chkey")
        .set_index("chkey")[["texdesc"]]
    )

    texture_records: list[dict] = []
    for mukey, cgrp in component.groupby("mukey"):
        dom_idx = cgrp["comppct"].idxmax()
        if pd.isna(dom_idx):
            continue

        dom_cokey = cgrp.loc[dom_idx, "cokey"]
        top_hz = chorizon[chorizon["cokey"] == dom_cokey].sort_values("hzdept_r").head(1)
        if top_hz.empty:
            continue

        chkey = top_hz.iloc[0]["chkey"]
        if chkey not in rep_tex.index:
            continue

        texture_records.append(
            {
                "mukey": mukey,
                "texture_desc": rep_tex.loc[chkey, "texdesc"],
            }
        )

    for rec in texture_records:
        rec["usda_class"] = pick_usda_class(rec["texture_desc"])

    texture_df = pd.DataFrame(texture_records)

    texture_sta_df = (
        sta_mk_valid.merge(texture_df, on="mukey", how="left")
        .dropna(subset=["texture_desc"])
        .merge(year_map, on="station_id", how="left")
    )
    cols = ["station", "station_id", "year"] + [c for c in texture_sta_df.columns if c not in ["station", "station_id", "year"]]
    texture_sta_df = texture_sta_df[cols]
    texture_sta_df.to_csv(OUT_DIR / f"texture_by_station_final_{YEAR}.csv", index=False)
    print(f"Saved: {OUT_DIR / f'texture_by_station_final_{YEAR}.csv'}")

    dom_cokey_map = (
        component.sort_values("comppct", ascending=False)
        .drop_duplicates("mukey")[["mukey", "cokey"]]
        .set_index("mukey")["cokey"]
        .to_dict()
    )

    full_records: list[dict] = []
    for _, row in component.iterrows():
        mukey = row["mukey"]
        cokey = row["cokey"]
        comppct = row["comppct"]
        is_dom = cokey == dom_cokey_map.get(mukey)

        top_hz = chorizon[chorizon["cokey"] == cokey].sort_values("hzdept_r").head(1)
        if top_hz.empty:
            continue

        chkey = top_hz.iloc[0]["chkey"]
        if chkey not in rep_tex.index:
            continue

        texdesc = rep_tex.loc[chkey, "texdesc"]
        usda_class = pick_usda_class(texdesc)

        full_records.append(
            {
                "mukey": mukey,
                "cokey": cokey,
                "component_pct": round(comppct * 100, 1),
                "is_dominant": is_dom,
                "texture_desc": texdesc,
                "usda_class": usda_class,
            }
        )

    allcomp_df = pd.DataFrame(full_records)
    allcomp_sta_df = sta_mk_valid.merge(allcomp_df, on="mukey", how="inner")
    allcomp_sta_df.to_csv(OUT_DIR / f"all_components_with_texture_{YEAR}.csv", index=False)
    print(f"Saved: {OUT_DIR / f'all_components_with_texture_{YEAR}.csv'}")

    texture_list: list[dict] = []
    for _, row in sta_mk_valid.iterrows():
        mukey = row["mukey"]
        station = row["station"]
        sub = chorizon[chorizon["mukey"] == mukey]

        sand_sf, silt_sf, clay_sf, om_sf = summarize_texture_om(sub, 12.5)
        sum_pct_sf = (sand_sf + silt_sf + clay_sf) if pd.notna(sand_sf) else pd.NA

        sand_rz, silt_rz, clay_rz, om_rz = summarize_texture_om(sub, 100.0)
        sum_pct_rz = (sand_rz + silt_rz + clay_rz) if pd.notna(sand_rz) else pd.NA

        texture_list.append(
            {
                "station": station,
                "sand_pct_sf": sand_sf,
                "silt_pct_sf": silt_sf,
                "clay_pct_sf": clay_sf,
                "om_pct_sf": om_sf,
                "sum_pct_sf": sum_pct_sf,
                "sand_pct": sand_rz,
                "silt_pct": silt_rz,
                "clay_pct": clay_rz,
                "om_pct": om_rz,
                "sum_pct": sum_pct_rz,
            }
        )

    texture_pct_df = pd.DataFrame(texture_list)

    texture_sta_df = (
        sta_mk_valid.merge(texture_df, on="mukey", how="left")
        .merge(texture_pct_df, on="station", how="left")
        .dropna(subset=["texture_desc"])
        .merge(year_map, on="station_id", how="left")
    )

    cols = [
        "station",
        "station_id",
        "year",
        "sand_pct_sf",
        "silt_pct_sf",
        "clay_pct_sf",
        "om_pct_sf",
        "sum_pct_sf",
        "sand_pct",
        "silt_pct",
        "clay_pct",
        "om_pct",
        "sum_pct",
    ] + [
        c
        for c in texture_sta_df.columns
        if c
        not in [
            "station",
            "station_id",
            "year",
            "sand_pct_sf",
            "silt_pct_sf",
            "clay_pct_sf",
            "om_pct_sf",
            "sum_pct_sf",
            "sand_pct",
            "silt_pct",
            "clay_pct",
            "om_pct",
            "sum_pct",
        ]
    ]
    texture_sta_df = texture_sta_df[cols]

    texture_sta_df.to_csv(OUT_DIR / f"texture_by_station_with_pct_{YEAR}.csv", index=False)
    print(f"Saved: {OUT_DIR / f'texture_by_station_with_pct_{YEAR}.csv'}")


if __name__ == "__main__":
    main()