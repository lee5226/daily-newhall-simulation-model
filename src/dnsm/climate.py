"""Strict climate preparation for the single-year D-NSM analysis.

Final rules implemented here
----------------------------
* Evaluation-year precipitation must be complete.
* Warm-up precipitation may have at most 3 missing days in total, and every
  missing day must be isolated (maximum consecutive gap = 1 day). Those
  accepted warm-up P gaps are set to 0 mm.
* Tmin and Tmax are processed on the concatenated warm-up + evaluation window.
  Only bounded missing gaps of at most 3 consecutive days are linearly
  interpolated. No rolling-mean, forward-fill, or backward-fill is used.
* Tavg is recalculated from the final Tmin and Tmax.

The extractor is expected to retain the full calendar and preserve missing
values. This module is the single source of truth for climate gap treatment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WARMUP_P_MAX_MISSING = 1
WARMUP_P_MAX_GAP = 1
T_MAX_GAP = 3


def _max_consecutive_missing(series: pd.Series) -> int:
    missing = series.isna()
    if not missing.any():
        return 0
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    return int(missing.groupby(groups).sum().max())


def _missing_runs(series: pd.Series) -> list[tuple[int, int, int]]:
    """Return inclusive positional start/end and length for missing runs."""
    missing = series.isna().to_numpy()
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(missing) - 1, len(missing) - start))
    return runs


def _normalize_one_file(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required = {"date", "precipitation", "tmin", "tmax"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise RuntimeError(f"{path.name} missing required columns: {sorted(missing_columns)}")

    if df["date"].isna().any():
        raise RuntimeError(f"Invalid dates remain in {path.name}")
    if df["date"].duplicated().any():
        duplicates = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Duplicate dates in {path.name}: {duplicates}")

    for col in ["precipitation", "tmin", "tmax", "solar", "tavg_raw", "tavg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Match the physical-value QC used by the P/T audit.
    df.loc[df["precipitation"].notna() & (df["precipitation"] < 0.0), "precipitation"] = np.nan
    for col in ["tmin", "tmax"]:
        invalid = df[col].notna() & ~df[col].between(-80.0, 60.0)
        df.loc[invalid, col] = np.nan

    order_invalid = (
        df["tmin"].notna()
        & df["tmax"].notna()
        & (df["tmin"] > df["tmax"])
    )
    df.loc[order_invalid, ["tmin", "tmax"]] = np.nan
    return df


def _require_complete_calendar(df: pd.DataFrame, year: int, label: str) -> pd.DataFrame:
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    observed = pd.DatetimeIndex(df["date"])
    if len(df) != len(expected) or not observed.equals(expected):
        missing_dates = expected.difference(observed)
        extra_dates = observed.difference(expected)
        raise RuntimeError(
            f"{label} climate calendar is not exactly {year}: "
            f"rows={len(df)}, expected={len(expected)}, "
            f"missing_dates={missing_dates.strftime('%Y-%m-%d').tolist()[:10]}, "
            f"extra_dates={extra_dates.strftime('%Y-%m-%d').tolist()[:10]}"
        )
    return df


def _validate_and_fill_precipitation(
    warmup: pd.DataFrame,
    evaluation: pd.DataFrame,
    warmup_p_max_missing: int,
    warmup_p_max_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warmup = warmup.copy()
    evaluation = evaluation.copy()

    eval_missing = int(evaluation["precipitation"].isna().sum())
    if eval_missing != 0:
        dates = evaluation.loc[evaluation["precipitation"].isna(), "date"]
        raise RuntimeError(
            "Evaluation precipitation must be complete; "
            f"missing={eval_missing}, dates={dates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    warmup_missing = int(warmup["precipitation"].isna().sum())
    warmup_gap = _max_consecutive_missing(warmup["precipitation"])
    if warmup_missing > warmup_p_max_missing or warmup_gap > warmup_p_max_gap:
        dates = warmup.loc[warmup["precipitation"].isna(), "date"]
        raise RuntimeError(
            "Warm-up precipitation exceeds allowed missingness: "
            f"missing={warmup_missing} (limit={warmup_p_max_missing}), "
            f"max_gap={warmup_gap} (limit={warmup_p_max_gap}), "
            f"dates={dates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    warmup["precip_missing_flag"] = warmup["precipitation"].isna().astype(int)
    warmup["precip_zero_filled_flag"] = warmup["precip_missing_flag"]
    warmup["precipitation"] = warmup["precipitation"].fillna(0.0)

    evaluation["precip_missing_flag"] = 0
    evaluation["precip_zero_filled_flag"] = 0
    return warmup, evaluation


def _validate_and_interpolate_temperature(combined: pd.DataFrame, t_gap_limit: int) -> pd.DataFrame:
    result = combined.copy()

    for col in ["tmin", "tmax"]:
        original_missing = result[col].isna()
        runs = _missing_runs(result[col])
        bad_runs: list[dict[str, object]] = []

        for start, end, length in runs:
            bounded = start > 0 and end < len(result) - 1
            if not bounded or length > t_gap_limit:
                bad_runs.append(
                    {
                        "start": result.iloc[start]["date"].strftime("%Y-%m-%d"),
                        "end": result.iloc[end]["date"].strftime("%Y-%m-%d"),
                        "length": length,
                        "bounded": bounded,
                    }
                )

        if bad_runs:
            raise RuntimeError(
                f"{col} contains unfillable gaps; only bounded gaps <= {t_gap_limit} days "
                f"are allowed. Gaps: {bad_runs}"
            )

        if runs:
            result[col] = result[col].interpolate(method="linear", limit_area="inside")

        if result[col].isna().any():
            remaining = result.loc[result[col].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
            raise RuntimeError(f"{col} still contains NaN after interpolation: {remaining}")

        result[f"{col}_missing_flag"] = original_missing.astype(int)
        result[f"{col}_interpolated_flag"] = original_missing.astype(int)

    if (result["tmin"] > result["tmax"]).any():
        bad = result.loc[result["tmin"] > result["tmax"], ["date", "tmin", "tmax"]]
        raise RuntimeError(
            "Tmin exceeds Tmax after interpolation: "
            + bad.to_dict(orient="records").__repr__()
        )

    result["tavg"] = (result["tmin"] + result["tmax"]) / 2.0
    return result


def load_climate_pair(
    warmup_filepath: str | Path,
    evaluation_filepath: str | Path,
    warmup_year: int,
    evaluation_year: int,
    warmup_p_max_missing: int = WARMUP_P_MAX_MISSING,
    warmup_p_max_gap: int = WARMUP_P_MAX_GAP,
    t_gap_limit: int = T_MAX_GAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and jointly prepare one warm-up year and one evaluation year."""
    if evaluation_year != warmup_year + 1:
        raise ValueError(
            f"Expected consecutive warm-up/evaluation years, got {warmup_year} and {evaluation_year}"
        )

    warmup = _require_complete_calendar(
        _normalize_one_file(warmup_filepath), warmup_year, "Warm-up"
    )
    evaluation = _require_complete_calendar(
        _normalize_one_file(evaluation_filepath), evaluation_year, "Evaluation"
    )

    warmup, evaluation = _validate_and_fill_precipitation(
        warmup,
        evaluation,
        warmup_p_max_missing=warmup_p_max_missing,
        warmup_p_max_gap=warmup_p_max_gap,
    )

    warmup = warmup.copy()
    evaluation = evaluation.copy()
    warmup["_period"] = "warmup"
    evaluation["_period"] = "evaluation"
    combined = pd.concat([warmup, evaluation], ignore_index=True).sort_values("date").reset_index(drop=True)

    expected_all = pd.date_range(
        f"{warmup_year}-01-01", f"{evaluation_year}-12-31", freq="D"
    )
    if not pd.DatetimeIndex(combined["date"]).equals(expected_all):
        raise RuntimeError("Combined warm-up + evaluation calendar is not continuous.")

    combined = _validate_and_interpolate_temperature(combined, t_gap_limit=t_gap_limit)

    warmup_final = combined.loc[combined["_period"] == "warmup"].drop(columns="_period").reset_index(drop=True)
    evaluation_final = combined.loc[combined["_period"] == "evaluation"].drop(columns="_period").reset_index(drop=True)

    required_final = ["precipitation", "tmin", "tmax", "tavg"]
    if warmup_final[required_final].isna().any().any() or evaluation_final[required_final].isna().any().any():
        raise RuntimeError("NaN remains in required climate forcing after strict processing.")

    return warmup_final, evaluation_final


def load_climate_data(filepath: str | Path) -> pd.DataFrame:
    """Read/physical-QC one extracted climate file without filling gaps.

    This compatibility helper deliberately performs no imputation. Single-year
    D-NSM runs should use ``load_climate_pair`` so temperature gaps spanning the
    warm-up/evaluation boundary are handled correctly.
    """
    return _normalize_one_file(filepath)
