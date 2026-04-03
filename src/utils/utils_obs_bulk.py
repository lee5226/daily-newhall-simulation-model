# utils_obs_bulk.py
import pandas as pd
import numpy as np

# 기본 USCRN 센서 깊이(단위: cm). 필요시 외부에서 주입 가능하게 두되, 기본값은 고정.
DEFAULT_DEPTHS = [5.0, 10.0, 20.0, 50.0, 100.0]
DEFAULT_VALCOLS = ["sm5", "sm10", "sm20", "sm50", "sm100"]

def bulk_from_points(row, depths, valcols, z1, z2):
    """
    [z1, z2] 구간의 두께가중 평균(bulk VWC)을 센서 포인트로부터 계산.
    - 센서 i의 가중치 w_i = clamp(mid_{i+} , z2) - clamp(mid_{i-} , z1),
      여기서 mid_{i-}, mid_{i+}는 인접 센서들과의 '중간 깊이'.
    - 경계 센서는 z1 또는 z2를 경계로 사용(반가중).
    - 결측 센서는 자동으로 제외되며, 남아있는 센서들로 중간깊이를 재계산(트라페zoidal).
    """
    # 유효 센서만 추출
    pts = []
    for d, c in zip(depths, valcols):
        v = row.get(c, np.nan)
        if pd.notna(v):
            pts.append((float(d), float(v)))

    if not pts:
        return np.nan

    # 센서 깊이 정렬
    pts.sort(key=lambda x: x[0])

    # 대상 구간과 겹치는 센서만 고려(센서가 구간 밖이라도 중간 경계로 영향이 있을 수 있어 남겨둠)
    # 다만 가중치 단계에서 clamp로 처리되므로 여기서는 전체 pts 사용

    # 각 센서에 대한 중간 경계(mids) 계산
    weights = []
    for i, (z, val) in enumerate(pts):
        z_prev = pts[i-1][0] if i > 0 else None
        z_next = pts[i+1][0] if i < len(pts)-1 else None
        mid_minus = 0.5 * (z_prev + z) if z_prev is not None else z1
        mid_plus  = 0.5 * (z + z_next) if z_next is not None else z2

        # 타깃 구간으로 클램프
        a = max(mid_minus, z1)
        b = min(mid_plus,  z2)
        w = max(0.0, b - a)
        if w > 0.0:
            weights.append((w, val))

    if not weights:
        return np.nan

    num = sum(w * v for (w, v) in weights)
    den = sum(w for (w, _) in weights)
    return num / den if den > 0 else np.nan

def bulk_surface_from_obs(row, prefer_5cm=False,
                          depths=DEFAULT_DEPTHS, valcols=DEFAULT_VALCOLS):
    """
    표층 비교용.
    - 기본(prefer_5cm=True): 관행적으로 5 cm 단일값 사용(민감도/재현 목적).
    - prefer_5cm=False: 0–12.5 cm 두께가중(사다리꼴)로 표층 bulk 생성(권장).
    """
    if prefer_5cm:
        return row.get("sm5", np.nan)
    return bulk_from_points(row, depths, valcols, 0.0, 12.5)

def bulk_rootzone_from_obs(row,
                           depths=DEFAULT_DEPTHS, valcols=DEFAULT_VALCOLS):
    """0–100 cm 두께가중 bulk(트라페zoidal, 결측 적응)."""
    return bulk_from_points(row, depths, valcols, 0.0, 100.0)

def add_bulk_columns(obs_df, prefer_5cm=True,
                     depths=DEFAULT_DEPTHS, valcols=DEFAULT_VALCOLS):
    """
    관측 DF에 다음 컬럼 추가:
      - sm_surf_bulk : 표층(기본은 sm5, 옵션으로 0–12.5 cm bulk)
      - sm_rz_bulk   : 0–100 cm bulk
    """
    df = obs_df.copy()
    df["sm_surf_bulk"] = df.apply(
        lambda r: bulk_surface_from_obs(r, prefer_5cm, depths, valcols), axis=1
    )
    df["sm_rz_bulk"] = df.apply(
        lambda r: bulk_rootzone_from_obs(r, depths, valcols), axis=1
    )
    return df
