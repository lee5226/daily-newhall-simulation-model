"""Final non-destructive USCRN station-year QC audit for soil moisture and P/T.

Purpose
-------
This script only records QC/completeness statistics.  It does NOT decide
whether a station-year is eligible as an evaluation year or as a warm-up year.
Those decisions are made later by the single-year and multi-year selectors.

Core audit years are 2009-2016.  The adjacent years 2008 and 2017 are audited
when raw files are available because they can bound short Tmin/Tmax gaps that
cross the outer edges of a 2009-2016 simulation window.  Missing optional
buffer years do not stop the audit.

For each audited year, a new file is written as
``uscrn_<year>_qc_summary_pt_qc.csv``.  Existing QC summaries and cleaned VWC
files are never overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR, USCRN_RAW_DIR


# ------------------------------ Configuration ------------------------------
CORE_YEARS = list(range(2009, 2017))
OPTIONAL_T_BUFFER_YEARS = [2008, 2017]

SOIL_MAX_MISS_RATE = 0.15
SOIL_MAX_CONSEC_GAP = 20
VWC_RANGE = (0.0, 0.6)

SM_COLS = ["SM_5", "SM_10", "SM_20", "SM_50", "SM_100"]
CLIMATE_COLS = ["P", "TMIN", "TMAX"]

# Zero-based columns in the USCRN Daily01 product.
DAILY01_COLUMN_MAP = {
    0: "WBAN",
    1: "DATE",
    3: "LON",
    4: "LAT",
    5: "TMAX",
    6: "TMIN",
    9: "P",
    18: "SM_5",
    19: "SM_10",
    20: "SM_20",
    21: "SM_50",
    22: "SM_100",
}
# ---------------------------------------------------------------------------


def as_bool(value: object) -> bool:
    """Return a plain Python bool for stable CSV output."""
    return bool(value)


def read_station_file(fp: Path) -> pd.DataFrame:
    """Read fields needed for soil-moisture and climate completeness QC."""
    df = pd.read_csv(
        fp,
        sep=r"\s+",
        header=None,
        usecols=list(DAILY01_COLUMN_MAP),
        engine="python",
    ).rename(columns=DAILY01_COLUMN_MAP)

    missing_cols = set(DAILY01_COLUMN_MAP.values()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{fp.name} is missing expected columns: {missing_cols}")

    date_text = df["DATE"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["DATE"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")

    numeric_cols = ["WBAN", "LON", "LAT", *CLIMATE_COLS, *SM_COLS]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # USCRN missing-value sentinels.  Additional physical checks are applied
    # below before missingness statistics are computed.
    df.replace([-9999.0, -99.0], np.nan, inplace=True)
    return df


def apply_value_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply soil range QC plus basic physical checks for P/T."""
    out = df.copy()

    for col in SM_COLS:
        out.loc[~out[col].between(*VWC_RANGE), col] = np.nan

    p_invalid = out["P"].notna() & (out["P"] < 0.0)
    tmin_invalid = out["TMIN"].notna() & ~out["TMIN"].between(-80.0, 60.0)
    tmax_invalid = out["TMAX"].notna() & ~out["TMAX"].between(-80.0, 60.0)

    out.loc[p_invalid, "P"] = np.nan
    out.loc[tmin_invalid, "TMIN"] = np.nan
    out.loc[tmax_invalid, "TMAX"] = np.nan

    order_invalid = (
        out["TMIN"].notna()
        & out["TMAX"].notna()
        & (out["TMIN"] > out["TMAX"])
    )
    out.loc[order_invalid, ["TMIN", "TMAX"]] = np.nan

    return out, {
        "p_invalid_n": int(p_invalid.sum()),
        "tmin_invalid_n": int(tmin_invalid.sum()),
        "tmax_invalid_n": int(tmax_invalid.sum()),
        "temp_order_invalid_n": int(order_invalid.sum()),
    }


def normalize_calendar(
    df: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Reindex to every calendar day so absent rows count as missing values."""
    expected_index = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    invalid_date_n = int(df["DATE"].isna().sum())
    dated = df.loc[df["DATE"].notna()].copy()
    out_of_year_n = int((dated["DATE"].dt.year != year).sum())
    dated = dated.loc[dated["DATE"].dt.year == year].copy()

    rows_in_year = len(dated)
    duplicate_date_n = int(dated["DATE"].duplicated(keep="first").sum())
    dated = dated.drop_duplicates(subset="DATE", keep="first")

    calendar = dated.set_index("DATE").reindex(expected_index)
    calendar.index.name = "DATE"
    missing_date_n = int(len(expected_index.difference(dated["DATE"])))

    # This flag is informational only.  In particular, a missing date row is
    # reindexed as NaN and counted by the normal missingness/gap statistics;
    # it does not automatically fail soil QC.
    date_structure_clean = (
        invalid_date_n == 0
        and out_of_year_n == 0
        and duplicate_date_n == 0
        and missing_date_n == 0
    )

    stats: dict[str, int | bool] = {
        "expected_days": len(expected_index),
        "rows_in_year": rows_in_year,
        "invalid_date_n": invalid_date_n,
        "out_of_year_n": out_of_year_n,
        "duplicate_date_n": duplicate_date_n,
        "missing_date_n": missing_date_n,
        "date_structure_clean_flag": as_bool(date_structure_clean),
    }
    return calendar, stats


def missing_stats(series: pd.Series) -> dict[str, int | float]:
    """Return missing count/rate and consecutive-gap statistics.

    ``leading_gap`` and ``trailing_gap`` are retained so selectors can combine
    gaps across Dec-31/Jan-01 without imposing a false requirement that each
    calendar-year boundary itself be complete.
    """
    missing = series.isna().to_numpy(dtype=bool)
    n_total = len(missing)
    missing_n = int(missing.sum())
    missing_rate = float(missing_n / n_total) if n_total else np.nan
    valid_n = int(n_total - missing_n)

    if missing_n == 0:
        max_gap = 0
        leading_gap = 0
        trailing_gap = 0
    else:
        # Maximum run of True values.
        padded = np.concatenate(([False], missing, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        run_lengths = changes[1::2] - changes[::2]
        max_gap = int(run_lengths.max()) if len(run_lengths) else 0

        leading_gap = 0
        for is_missing in missing:
            if not is_missing:
                break
            leading_gap += 1

        trailing_gap = 0
        for is_missing in missing[::-1]:
            if not is_missing:
                break
            trailing_gap += 1

    return {
        "missing_n": missing_n,
        "missing_rate": missing_rate,
        "valid_n": valid_n,
        "max_gap": max_gap,
        "leading_gap": int(leading_gap),
        "trailing_gap": int(trailing_gap),
    }


def station_year_record(fp: Path, year: int) -> dict[str, object]:
    """Build one station-year audit record."""
    station = fp.stem.split("-")[-1]
    raw = read_station_file(fp)

    wban_values = raw["WBAN"].dropna()
    if wban_values.empty:
        raise ValueError(f"WBAN ID not found in {fp}")
    station_id = str(int(wban_values.iloc[0])).zfill(5)

    cleaned, invalid_counts = apply_value_qc(raw)
    calendar, calendar_stats = normalize_calendar(cleaned, year)

    record: dict[str, object] = {
        "station": station,
        "station_id": station_id,
        "year": year,
        **calendar_stats,
        **invalid_counts,
    }

    soil_exact: dict[str, tuple[float, int]] = {}
    for col in SM_COLS:
        depth = col.replace("SM_", "")
        stats = missing_stats(calendar[col])
        miss_rate = float(stats["missing_rate"])
        max_gap = int(stats["max_gap"])
        soil_exact[depth] = (miss_rate, max_gap)
        record[f"miss_rate_{depth}cm"] = round(miss_rate, 6)
        record[f"max_gap_{depth}cm"] = max_gap

    # Final soil-observation QC requested for station-year eligibility.
    # Date-structure diagnostics are retained above but are not an additional
    # automatic failure criterion.
    soil_pass = (
        soil_exact["5"][0] <= SOIL_MAX_MISS_RATE
        and soil_exact["5"][1] <= SOIL_MAX_CONSEC_GAP
        and soil_exact["100"][0] <= SOIL_MAX_MISS_RATE
        and soil_exact["100"][1] <= SOIL_MAX_CONSEC_GAP
    )

    for col, prefix in [("P", "p"), ("TMIN", "tmin"), ("TMAX", "tmax")]:
        stats = missing_stats(calendar[col])
        record[f"{prefix}_missing_n"] = int(stats["missing_n"])
        record[f"{prefix}_missing_rate"] = round(float(stats["missing_rate"]), 6)
        record[f"{prefix}_valid_n"] = int(stats["valid_n"])
        record[f"{prefix}_max_gap"] = int(stats["max_gap"])

        # Leading/trailing gaps are especially useful for Tmin/Tmax, but are
        # harmless and informative for P as well.
        record[f"{prefix}_leading_gap"] = int(stats["leading_gap"])
        record[f"{prefix}_trailing_gap"] = int(stats["trailing_gap"])

    record["soil_pass_flag"] = as_bool(soil_pass)
    return record


def audit_year(year: int, *, required: bool) -> Path | None:
    """Audit one year and write a separate final P/T-aware summary."""
    data_dir = USCRN_RAW_DIR / str(year)
    out_dir = OUTPUTS_DIR / str(year)
    files = sorted(data_dir.glob(f"CRND0103-{year}-*.txt"))

    if not files:
        message = f"No USCRN Daily01 files found in {data_dir}"
        if required:
            raise FileNotFoundError(message)
        print(f"Optional buffer year skipped: {message}")
        return None

    records = [station_year_record(fp, year) for fp in files]
    summary = pd.DataFrame(records).sort_values(["station_id", "station"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"uscrn_{year}_qc_summary_pt_qc.csv"
    summary.to_csv(out_path, index=False)

    print(
        f"Saved: {out_path} | records={len(summary)} | "
        f"soil_pass={int(summary['soil_pass_flag'].sum())}"
    )
    return out_path


def main() -> None:
    outputs: list[Path] = []

    for year in CORE_YEARS:
        out = audit_year(year, required=True)
        if out is not None:
            outputs.append(out)

    for year in OPTIONAL_T_BUFFER_YEARS:
        out = audit_year(year, required=False)
        if out is not None:
            outputs.append(out)

    print(f"Finished final P/T QC audit for {len(outputs)} available years.")
    print("No evaluation/warm-up climate eligibility was decided here.")
    print("Existing QC summaries and cleaned VWC files were not modified.")


if __name__ == "__main__":
    main()
