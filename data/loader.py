# data/loader.py
from pathlib import Path
import numpy as np
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

    # object → numeric 시도
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
