"""Select one 2010-2016 evaluation year per USCRN station using final P/T QC.

Eligibility for candidate evaluation year Y
-------------------------------------------
1. Soil-observation QC passes in Y.
2. Evaluation-year precipitation is complete: P missing_n == 0 in Y.
3. Warm-up precipitation in Y-1 has at most one missing day and max gap <= 1.
4. Tmin and Tmax are assessed over the continuous Y-1 through Y window:
   every consecutive missing gap must be <= 3 days and must be bounded by
   valid observations.  Gaps that cross Dec-31/Jan-01 are combined correctly.
   The first day of Y-1 and the last day of Y must therefore both be observed;
   no temperature values outside the two-year simulation window are used.

Among eligible candidate years for each station, selection follows the prior
rule: miss_sum -> gap_sum -> earliest year.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import OUTPUTS_DIR


EVAL_YEARS = list(range(2010, 2017))
AUDIT_YEARS = list(range(EVAL_YEARS[0] - 1, EVAL_YEARS[-1] + 1))
AUDIT_TEMPLATE = "uscrn_{year}_qc_summary_pt_qc.csv"

T_MAX_GAP = 3
WARMUP_P_MAX_MISSING = 1
WARMUP_P_MAX_GAP = 1

OUT_PATH = OUTPUTS_DIR / "selected_station_year_1016_pt_qc.csv"
AUDIT_OUT_PATH = OUTPUTS_DIR / "selected_station_year_1016_pt_qc_candidates.csv"


REQUIRED_COLUMNS = {
    "station",
    "station_id",
    "year",
    "miss_rate_5cm",
    "max_gap_5cm",
    "miss_rate_100cm",
    "max_gap_100cm",
    "soil_pass_flag",
    "p_missing_n",
    "p_max_gap",
    "tmin_max_gap",
    "tmin_leading_gap",
    "tmin_trailing_gap",
    "tmax_max_gap",
    "tmax_leading_gap",
    "tmax_trailing_gap",
}


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse CSV booleans safely; do not use astype(bool) on strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none"}

    def parse_one(value: object) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        raise ValueError(f"Unrecognized boolean value: {value!r}")

    return series.map(parse_one)


def load_audits() -> dict[int, pd.DataFrame]:
    """Load the required 2009-2016 final audit summaries."""
    audits: dict[int, pd.DataFrame] = {}

    for year in AUDIT_YEARS:
        path = OUTPUTS_DIR / str(year) / AUDIT_TEMPLATE.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Required audit summary missing: {path}")

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

        df = df.copy()
        df["station_id"] = df["station_id"].astype(str).str.zfill(5)
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
        df["soil_pass_flag"] = parse_bool_series(df["soil_pass_flag"])

        if df["station_id"].duplicated().any():
            dup = df.loc[df["station_id"].duplicated(False), "station_id"].tolist()
            raise ValueError(f"Duplicate station_id in {path.name}: {dup[:10]}")

        audits[year] = df.set_index("station_id", drop=False)

    return audits


def get_row(
    audits: dict[int, pd.DataFrame],
    year: int,
    station_id: str,
) -> pd.Series | None:
    df = audits.get(year)
    if df is None or station_id not in df.index:
        return None
    row = df.loc[station_id]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Duplicate station_id {station_id} in audit year {year}")
    return row


def continuous_t_gap_check(
    audits: dict[int, pd.DataFrame],
    station_id: str,
    start_year: int,
    end_year: int,
    prefix: str,
) -> tuple[bool, int | None, str]:
    """Check all bounded T gaps across a continuous multi-year window.

    Annual leading/trailing missing runs are combined at year boundaries.
    The first and last days of the requested window must be observed, so no
    outside-year temperature buffer is used.
    """
    rows: dict[int, pd.Series] = {}
    for year in range(start_year, end_year + 1):
        row = get_row(audits, year, station_id)
        if row is None:
            return False, None, f"missing_{year}_audit"
        rows[year] = row

    observed_gaps: list[int] = []

    # Within-year gaps, including annual leading/trailing partial runs.
    for year, row in rows.items():
        annual_gap = int(row[f"{prefix}_max_gap"])
        observed_gaps.append(annual_gap)
        if annual_gap > T_MAX_GAP:
            return False, max(observed_gaps), f"{prefix}_{year}_gap_gt_{T_MAX_GAP}"

    # Gaps crossing internal Dec-31/Jan-01 boundaries.
    for year in range(start_year, end_year):
        left = rows[year]
        right = rows[year + 1]
        cross_gap = int(left[f"{prefix}_trailing_gap"]) + int(
            right[f"{prefix}_leading_gap"]
        )
        observed_gaps.append(cross_gap)
        if cross_gap > T_MAX_GAP:
            return (
                False,
                max(observed_gaps),
                f"{prefix}_{year}_{year + 1}_cross_gap_gt_{T_MAX_GAP}",
            )

    # The first day of the warm-up year must be observed.
    first = rows[start_year]
    first_leading = int(first[f"{prefix}_leading_gap"])
    if first_leading > 0:
        return (
            False,
            max(observed_gaps, default=first_leading),
            f"{prefix}_{start_year}_unbounded_leading_gap",
        )

    # The last day of the evaluation year must be observed.
    last = rows[end_year]
    last_trailing = int(last[f"{prefix}_trailing_gap"])
    if last_trailing > 0:
        return (
            False,
            max(observed_gaps, default=last_trailing),
            f"{prefix}_{end_year}_unbounded_trailing_gap",
        )

    return True, max(observed_gaps, default=0), "pass"


def evaluate_candidate(
    audits: dict[int, pd.DataFrame],
    station_id: str,
    eval_year: int,
) -> dict[str, object] | None:
    eval_row = get_row(audits, eval_year, station_id)
    warm_row = get_row(audits, eval_year - 1, station_id)
    if eval_row is None or warm_row is None:
        return None

    soil_pass = bool(eval_row["soil_pass_flag"])
    eval_p_pass = int(eval_row["p_missing_n"]) == 0
    warmup_p_pass = (
        int(warm_row["p_missing_n"]) <= WARMUP_P_MAX_MISSING
        and int(warm_row["p_max_gap"]) <= WARMUP_P_MAX_GAP
    )

    tmin_pass, tmin_window_max_gap, tmin_reason = continuous_t_gap_check(
        audits, station_id, eval_year - 1, eval_year, "tmin"
    )
    tmax_pass, tmax_window_max_gap, tmax_reason = continuous_t_gap_check(
        audits, station_id, eval_year - 1, eval_year, "tmax"
    )

    climate_pass = eval_p_pass and warmup_p_pass and tmin_pass and tmax_pass
    eligible = soil_pass and climate_pass

    miss_sum = float(eval_row["miss_rate_5cm"]) + float(
        eval_row["miss_rate_100cm"]
    )
    gap_sum = int(eval_row["max_gap_5cm"]) + int(eval_row["max_gap_100cm"])

    return {
        "station": eval_row["station"],
        "station_id": station_id,
        "year": eval_year,
        "warmup_year": eval_year - 1,
        "miss_sum": miss_sum,
        "gap_sum": gap_sum,
        "soil_pass_flag": soil_pass,
        "eval_p_missing_n": int(eval_row["p_missing_n"]),
        "eval_p_pass_flag": eval_p_pass,
        "warmup_p_missing_n": int(warm_row["p_missing_n"]),
        "warmup_p_max_gap": int(warm_row["p_max_gap"]),
        "warmup_p_pass_flag": warmup_p_pass,
        "tmin_window_max_gap": tmin_window_max_gap,
        "tmin_window_pass_flag": tmin_pass,
        "tmin_reason": tmin_reason,
        "tmax_window_max_gap": tmax_window_max_gap,
        "tmax_window_pass_flag": tmax_pass,
        "tmax_reason": tmax_reason,
        "climate_pass_flag": climate_pass,
        "eligible_flag": eligible,
    }


def main() -> None:
    audits = load_audits()

    # Candidate station IDs are taken from evaluation years; warm-up presence is
    # checked explicitly per candidate.
    station_ids = sorted(
        set().union(
            *[
                set(audits[year].index)
                for year in EVAL_YEARS
                if year in audits
            ]
        )
    )

    records: list[dict[str, object]] = []
    for station_id in station_ids:
        for eval_year in EVAL_YEARS:
            record = evaluate_candidate(audits, station_id, eval_year)
            if record is not None:
                records.append(record)

    if not records:
        raise RuntimeError("No candidate station-year records could be evaluated.")

    candidates = pd.DataFrame(records).sort_values(["station_id", "year"])
    AUDIT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(AUDIT_OUT_PATH, index=False)

    eligible = candidates.loc[candidates["eligible_flag"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No station-year records satisfy the final soil/P/T eligibility rules."
        )

    eligible = eligible.sort_values(
        by=["station_id", "miss_sum", "gap_sum", "year"],
        ascending=[True, True, True, True],
    )
    selected = eligible.drop_duplicates(subset=["station_id"], keep="first").copy()

    selected_cols = [
        "station",
        "station_id",
        "year",
        "warmup_year",
        "miss_sum",
        "gap_sum",
        "eval_p_missing_n",
        "warmup_p_missing_n",
        "warmup_p_max_gap",
        "tmin_window_max_gap",
        "tmax_window_max_gap",
    ]
    selected = selected[selected_cols].sort_values(["station_id", "year"])
    selected.to_csv(OUT_PATH, index=False)

    print(f"Saved candidate audit: {AUDIT_OUT_PATH}")
    print(f"Candidate station-years checked: {len(candidates)}")
    print(f"Eligible candidate station-years: {len(eligible)}")
    print(f"Saved selected station-years: {OUT_PATH}")
    print(f"Selected stations: {len(selected)}")


if __name__ == "__main__":
    main()
