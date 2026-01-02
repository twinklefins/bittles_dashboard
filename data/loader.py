# data/loader.py
from pathlib import Path
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "time" not in df.columns:
        st.error("CSV에 'time' 컬럼이 없습니다.")
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # ✅ 숫자 컬럼만 변환 (문자 컬럼 보호)
    numeric_cols = [
        "price_close",
        "risk_total_0_100",
        "risk_level_score",
        "ml_10d_prob",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ✅ 레벨/메시지 컬럼은 문자열 유지 (선택이지만 추천)
    text_cols = [
        "final_risk_level",
        "final_driver",
        "ml_10d_level",
        "msg_headline",
        "msg_summary",
        "msg_action",
        "msg_details",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df
