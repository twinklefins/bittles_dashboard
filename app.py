import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt


# ======================
# Paths
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "df_draft_1209_w.sti.csv"


# ======================
# CII helpers
# ======================
def z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    mu = x.mean()
    sd = x.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def build_cii(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    need = [
        "gt_bitcoin", "gt_btc_z14",
        "rd_avg_sent", "rd_pos_ratio", "rd_neg_ratio",
        "rd_rolling_mean_neg", "rd_rolling_std_neg", "rd_count"
    ]
    if any(c not in out.columns for c in need):
        out["CII"] = np.nan
        return out

    out["rd_pos_minus_neg"] = out["rd_pos_ratio"] - out["rd_neg_ratio"]

    out["cii_attention"] = (z(out["gt_bitcoin"]) + z(out["gt_btc_z14"])) / 2
    out["cii_sentiment"] = (
        z(out["rd_avg_sent"]) +
        z(out["rd_pos_minus_neg"]) -
        z(out["rd_rolling_mean_neg"])
    ) / 3
    out["cii_volatility"] = (
        z(out["rd_rolling_std_neg"]) +
        z(out["rd_count"])
    ) / 2

    out["CII"] = (
        0.4 * out["cii_attention"] +
        0.4 * out["cii_sentiment"] +
        0.2 * out["cii_volatility"]
    )

    return out


# ======================
# Data loader
# ======================
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error("데이터 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    df = build_cii(df)
    return df


# ======================
# Risk helpers
# ======================
def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True):
    if series.empty or pd.isna(value):
        return "⚪️", 1

    lower, upper = series.quantile([0.33, 0.66])

    if higher_is_risky:
        if value >= upper:
            return "🔴", 2
        if value <= lower:
            return "🟢", 0
    else:
        if value <= lower:
            return "🔴", 2
        if value >= upper:
            return "🟢", 0

    return "🟡", 1


def overall_risk_text(total, cnt):
    avg = total / max(cnt, 1)
    if avg < 0.5:
        return "🟢 과열 신호는 약합니다."
    if avg < 1.2:
        return "🟡 변동성 확대 가능 구간입니다."
    if avg < 1.8:
        return "🟠 레버리지·쏠림 주의 구간입니다."
    return "🔴 단기 충격 가능성이 큽니다."


# ======================
# VAR helpers
# ======================
def zscore_df(df):
    return (df - df.mean()) / df.std().replace(0, np.nan)


def run_var_bundle(df, selected_cols, target, lag, horizon, standardize):
    data = df[selected_cols].dropna()
    if standardize:
        data = zscore_df(data).dropna()

    model = VAR(data)
    res = model.fit(lag)

    rows = []
    for x in selected_cols:
        if x == target:
            continue
        test = res.test_causality(caused=target, causing=[x], kind="f")
        rows.append({
            "causing(x)": x,
            "caused(target)": target,
            "stat": test.test_statistic,
            "pvalue": test.pvalue
        })

    granger = pd.DataFrame(rows).sort_values("pvalue")

    irf = res.irf(horizon)
    fevd = res.fevd(horizon)
    names = res.names
    t_idx = names.index(target)

    fevd_tbl = pd.DataFrame(
        fevd.decomp[1:, t_idx, :] * 100,
        columns=names,
        index=range(1, horizon + 1)
    ).round(2)

    return granger, irf, fevd_tbl


# ======================
# Main
# ======================
def main():
    st.set_page_config("시장 위험도 대시보드", "📊", layout="wide")
    st.title("📊 시장 위험도 대시보드")

    df = load_data(DATA_PATH)
    if df.empty:
        return

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 Pipeline"])

    # ------------------
    # TAB 1
    # ------------------
    with tab1:
        st.sidebar.header("설정")

        dates = sorted(pd.unique(df.index.date), reverse=True)
        sel_date = st.sidebar.selectbox(
            "기준 날짜",
            dates,
            format_func=lambda d: d.strftime("%Y-%m-%d")
        )

        today = df[df.index.date == sel_date].iloc[-1]
        idx = dates.index(sel_date)
        prev = df[df.index.date == dates[idx + 1]].iloc[-1] if idx < len(dates) - 1 else None

        cols = st.columns(5)
        indicators = {
            "oi_close_diff": ("OI 변화", True),
            "funding_close": ("펀딩비", True),
            "liq_total_usd_diff": ("청산", True),
            "taker_buy_ratio": ("테이커 비중", True),
            "global_m2_yoy_diff": ("M2", False),
        }

        total, used = 0, 0
        for ui, (c, meta) in zip(cols, indicators.items()):
            if c not in df.columns:
                ui.metric(meta[0], "N/A")
                continue

            v = today[c]
            sig, sc = percentile_signal(df[c], v, meta[1])
            delta = v - prev[c] if prev is not None else None

            ui.metric(meta[0], f"{v:.4g}", delta=f"{delta:+.4g}" if delta is not None else None)
            ui.caption(sig)

            total += sc
            used += 1

        st.success(overall_risk_text(total, used))

        st.subheader("📈 소비자 투자 인덱스 (CII)")
        if df["CII"].dropna().empty:
            st.warning("CII 계산 불가 (컬럼 부족)")
        else:
            st.metric("CII (latest)", f"{df['CII'].iloc[-1]:.2f}")
            st.line_chart(df["CII"].tail(200))

    # ------------------
    # TAB 2
    # ------------------
    with tab2:
        st.sidebar.header("VAR 설정")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        sel = st.sidebar.multiselect("VAR 변수", num_cols, default=num_cols[:4])
        target = st.sidebar.selectbox("Target", sel) if sel else None
        lag = st.sidebar.slider("lag", 1, 5, 1)
        horizon = st.sidebar.slider("horizon", 5, 20, 10)
        standardize = st.sidebar.checkbox("z-score", True)

        if st.button("VAR 실행"):
            g, irf, fevd = run_var_bundle(df, sel, target, lag, horizon, standardize)

            st.subheader("Granger")
            st.dataframe(g)

            st.subheader("IRF")
            fig = irf.plot()
            st.pyplot(fig)

            st.subheader("FEVD")
            st.dataframe(fevd)

    # ------------------
    # TAB 3
    # ------------------
    with tab3:
        st.markdown("""
### 분석 파이프라인
1. df_draft_1209_w.sti.csv 로드
2. CII(소비자 투자 인덱스) 계산
3. Risk Signal 요약
4. VAR → Granger / IRF / FEVD
""")


if __name__ == "__main__":
    main()
