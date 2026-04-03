"""
Filename: extract_daily_climate_inputs.py
Author: Moonyoung Lee
Date: 2025-06-19
Edited: 2025-07-06 YEAR 변수를 도입
Edited: 2025-07-22 selected_station_year 기반으로 수정


Description:
    This script extracts daily climate variables for QC-passed USCRN stations.
    It formats the data for use as input to the Daily Newhall Model by reading 
    raw daily01 files and exporting station-level tidy CSVs.

    Output files follow this column format:
        station, station_id, date, precipitation, tmin, tmax, tavg, solar

Inputs:
    - uscrn_daily01_2024/ : Folder containing raw NOAA daily text files (CRND0103-2024-*.txt)
    - outputs/aws_by_station_final.csv : List of stations that passed QC (contains 'station' field)

Outputs:
    - outputs/daily_climate/daily_climate_2024_<station_id>.csv : One file per station

Dependencies:
    - pandas
    - Python 3.9+
    - Assumes fixed header structure in NOAA files (29-column layout)

Notes:
    - Date parsing and numeric conversions are performed to ensure clean output
    - Station identifiers (WBANNO) are zero-padded to 5 digits
"""
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. 기본 설정
YEAR        = 1016  # selected station-year 세트 식별자
BASE        = Path(f"outputs/{YEAR}")
INPUT_ROOT  = Path("uscrn_daily01")
QC_CSV      = BASE / f"aws_by_station_final_{YEAR}.csv"
OUTPUT_DIR  = BASE / "daily_climate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 2. NOAA fixed 헤더 정의
HEADERS = [
    "WBANNO", "LST_DATE", "CRX_VN", "LONGITUDE", "LATITUDE",
    "T_DAILY_MAX", "T_DAILY_MIN", "T_DAILY_MEAN", "T_DAILY_AVG",
    "P_DAILY_CALC", "SOLARAD_DAILY", "SUR_TEMP_DAILY_TYPE",
    "SUR_TEMP_DAILY_MAX", "SUR_TEMP_DAILY_MIN", "SUR_TEMP_DAILY_AVG",
    "RH_DAILY_MAX", "RH_DAILY_MIN", "RH_DAILY_AVG",
    "SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY", "SOIL_MOISTURE_20_DAILY",
    "SOIL_MOISTURE_50_DAILY", "SOIL_MOISTURE_100_DAILY",
    "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY", "SOIL_TEMP_20_DAILY",
    "SOIL_TEMP_50_DAILY", "SOIL_TEMP_100_DAILY"
]

# ─────────────────────────────────────────────────────────────
# 3. QC 통과 station 목록 로드
qc_df = pd.read_csv(QC_CSV)
qc_df["station_id"] = qc_df["station_id"].astype(str).str.zfill(5)

# ─────────────────────────────────────────────────────────────
# 4. station–year 단위로 기후 추출
for _, row in qc_df.iterrows():
    station_name = row["station"]
    station_id   = row["station_id"]
    real_year    = row["year"]

    input_dir = INPUT_ROOT / str(real_year)
    pattern = f"CRND0103-{real_year}-*{station_name}.txt"
    matches = list(input_dir.glob(pattern))

    if not matches:
        print(f"X {station_name} ({real_year}): raw file not found")
        continue

    raw_path = matches[0]

    try:
        df_raw = pd.read_csv(
            raw_path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=HEADERS,
            dtype=str,
            engine="python"
        )

        df = df_raw[[
            "WBANNO", "LST_DATE", "T_DAILY_MIN", "T_DAILY_MAX",
            "T_DAILY_MEAN", "P_DAILY_CALC", "SOLARAD_DAILY"
        ]].copy()

        df["station"] = station_name
        df["station_id"] = df["WBANNO"].str.zfill(5)
        df["date"] = pd.to_datetime(df["LST_DATE"], format="%Y%m%d", errors="coerce")

        df = df.rename(columns={
            "T_DAILY_MIN": "tmin",
            "T_DAILY_MAX": "tmax",
            "T_DAILY_MEAN": "tavg",
            "P_DAILY_CALC": "precipitation",
            "SOLARAD_DAILY": "solar"
        })

        df = df[[
            "station", "station_id", "date", "precipitation",
            "tmin", "tmax", "tavg", "solar"
        ]]
        
        df = df.dropna(subset=["date", "tavg", "precipitation"])

        num_cols = ["precipitation", "tmin", "tmax", "tavg", "solar"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # 저장
        out_path = OUTPUT_DIR / f"daily_climate_{YEAR}_{station_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"✔ {station_name} ({real_year}) → {out_path.name}")

    except Exception as e:
        print(f"X {station_name} ({real_year}): {e}")