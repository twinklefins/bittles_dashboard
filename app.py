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
        st.error(f"데이터 파일을 찾을 수 없습니다: {path}")
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
def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    if series.empty or pd.isna(value):
        return "⚪️", 1

    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "⚪️", 1

    lower, upper = s.quantile([0.33, 0.66])

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


def overall_risk_text(total: int, cnt: int) -> str:
    avg = total / max(cnt, 1)
    if avg < 0.5:
        return "🟢 과열 신호는 약합니다."
    if avg < 1.2:
        return "🟡 변동성 확대 가능 구간입니다."
    if avg < 1.8:
        return "🟠 레버리지·쏠림 주의 구간입니다."
    return "🔴 단기 충격 가능성이 큽니다."


def fmt_delta(col: str, v: float, prev_v: Optional[float]) -> Optional[str]:
    if prev_v is None or pd.isna(prev_v) or pd.isna(v):
        return None
    d = v - prev_v
    if col in ["funding_close", "taker_buy_ratio", "global_m2_yoy_diff", "CII"]:
        return f"{d:+.4f}"
    return f"{d:+,.0f}"


# ======================
# VAR helpers
# ======================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std().replace(0, np.nan)


def fevd_to_table(fevd, target: str, names: List[str], horizon: int) -> pd.DataFrame:
    """
    statsmodels 버전에 따라 fevd.decomp shape이 다를 수 있어 방어적으로 처리.
    목표: index=1..horizon, columns=names, 값=target의 분산 기여도(%)
    """
    decomp = np.array(fevd.decomp)

    if decomp.ndim != 3:
        raise ValueError(f"FEVD decomp shape이 예상과 다릅니다: {decomp.shape}")

    if target not in names:
        raise ValueError("FEVD에서 target 변수를 찾지 못했습니다.")

    t_idx = names.index(target)

    # decomp[t, response, impulse] 가 일반적
    # t가 horizon+1(0 포함)인 경우가 많음 → 0 step 제거
    t_len = decomp.shape[0]
    if t_len == horizon + 1:
        use = decomp[1:horizon + 1, t_idx, :]
        idx = range(1, horizon + 1)
    else:
        # 혹시 이미 1..horizon로 나온 경우
        use = decomp[:horizon, t_idx, :]
        idx = range(1, min(horizon, use.shape[0]) + 1)

    tbl = pd.DataFrame(use * 100.0, columns=names, index=idx).round(2)
    tbl.index.name = "horizon(step)"
    return tbl


def run_var_bundle(df: pd.DataFrame, selected_cols: List[str], target: str, impulse: str, lag: int, horizon: int, standardize: bool):
    if len(selected_cols) < 2:
        raise ValueError("VAR 변수는 2개 이상 선택해야 합니다.")
    if target not in selected_cols:
        raise ValueError("Target은 선택된 VAR 변수 안에 있어야 합니다.")
    if impulse not in selected_cols:
        raise ValueError("Impulse는 선택된 VAR 변수 안에 있어야 합니다.")
    if impulse == target:
        raise ValueError("Impulse와 Target은 달라야 합니다.")

    data = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < max(80, lag * 10):
        raise ValueError(f"데이터가 부족합니다. 현재 {data.shape[0]} rows")

    if standardize:
        data = zscore_df(data).dropna()

    model = VAR(data)
    res = model.fit(lag)

    # Granger: x -> target
    rows = []
    for x in selected_cols:
        if x == target:
            continue
        try:
            test = res.test_causality(caused=target, causing=[x], kind="f")
            rows.append({
                "causing(x)": x,
                "caused(target)": target,
                "stat": float(test.test_statistic),
                "pvalue": float(test.pvalue),
            })
        except Exception as e:
            rows.append({
                "causing(x)": x,
                "caused(target)": target,
                "stat": np.nan,
                "pvalue": np.nan,
                "error": str(e),
            })

    granger = pd.DataFrame(rows).sort_values("pvalue", na_position="last").reset_index(drop=True)

    irf = res.irf(horizon)
    fevd = res.fevd(horizon)
    names = list(res.names)
    fevd_tbl = fevd_to_table(fevd, target=target, names=names, horizon=horizon)

    return granger, irf, fevd_tbl, res, data.shape[0]


# ======================
# Main
# ======================
def main():
    st.set_page_config("시장 위험도 대시보드", "📊", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.9rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 시장 위험도 대시보드")
    st.caption("Risk Signal(🟢🟡🔴) + VAR Insight(Granger/IRF/FEVD) + Pipeline")

    df = load_data(DATA_PATH)
    if df.empty:
        return

    # 탭
    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 Pipeline"])

    # 공통 sidebar
    st.sidebar.header("설정")

    # ✅ 최근 날짜가 위로: reverse=True
    dates = sorted(pd.unique(df.index.date), reverse=True)
    sel_date = st.sidebar.selectbox(
        "기준 날짜",
        dates,
        index=0,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        key="risk_date",
    )

    # ------------------
    # TAB 1: Risk
    # ------------------
    with tab1:
        st.subheader(f"기준 데이터 날짜: {pd.Timestamp(sel_date):%Y-%m-%d}")

        day_df = df[df.index.date == sel_date]
        if day_df.empty:
            st.warning("해당 날짜 데이터가 없습니다.")
            return
        today = day_df.iloc[-1]

        # 전일(=dates에서 다음 index)
        idx = dates.index(sel_date)
        prev_date = dates[idx + 1] if idx < len(dates) - 1 else None
        prev_row = None
        if prev_date is not None:
            prev_df = df[df.index.date == prev_date]
            if not prev_df.empty:
                prev_row = prev_df.iloc[-1]

        # 지표 정의
        indicators = [
            ("oi_close_diff", "OI 변화", True),
            ("funding_close", "펀딩비", True),
            ("liq_total_usd_diff", "청산", True),
            ("taker_buy_ratio", "테이커 비중(쏠림)", True),   # <- 쏠림은 0.5 중심
            ("global_m2_yoy_diff", "M2", False),
        ]

        cols = st.columns(len(indicators), gap="large")
        total, used = 0, 0

        for ui, (col, label, higher_risky) in zip(cols, indicators):
            if col not in df.columns:
                ui.metric(label, "N/A")
                ui.caption("⚪️")
                continue

            v = today[col]
            prev_v = prev_row[col] if (prev_row is not None and col in prev_row.index) else None

            # ✅ 테이커: abs(x-0.5)로 쏠림을 위험도로
            if col == "taker_buy_ratio":
                series = (df[col] - 0.5).abs()
                vv = abs(float(v) - 0.5) if pd.notna(v) else np.nan
                sig, sc = percentile_signal(series, vv, higher_is_risky=True)
                value_txt = f"{float(v):.3f}" if pd.notna(v) else "N/A"
                extra = f"쏠림 |{vv:.3f}| (0.5에서 멀수록 쏠림)" if pd.notna(v) else ""
                delta_txt = fmt_delta(col, float(v), float(prev_v)) if (prev_v is not None and pd.notna(v) and pd.notna(prev_v)) else None
                ui.metric(label, value_txt, delta=delta_txt)
                if extra:
                    ui.caption(extra)
                ui.caption(sig)
            else:
                sig, sc = percentile_signal(df[col], float(v) if pd.notna(v) else np.nan, higher_is_risky=higher_risky)
                value_txt = f"{float(v):.4g}" if pd.notna(v) else "N/A"
                delta_txt = fmt_delta(col, float(v), float(prev_v)) if (prev_v is not None and pd.notna(v) and pd.notna(prev_v)) else None
                ui.metric(label, value_txt, delta=delta_txt)
                ui.caption(sig)

            total += sc
            used += 1

        st.success(overall_risk_text(total, used))

        st.subheader("📈 소비자 투자 인덱스 (CII)")
        if "CII" not in df.columns or df["CII"].dropna().empty:
            st.warning("CII 계산 불가 (컬럼 부족)")
        else:
            st.metric("CII (latest)", f"{df['CII'].iloc[-1]:.2f}")
            st.line_chart(df["CII"].tail(200))

        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df.tail(50), use_container_width=True)

    # ------------------
    # TAB 2: VAR
    # ------------------
    with tab2:
        st.sidebar.header("VAR 설정")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        default_vars = [c for c in ["ret_log_1d", "oi_close_diff", "funding_close", "liq_total_usd_diff", "taker_buy_ratio"] if c in num_cols]

        sel = st.sidebar.multiselect("VAR 변수(2개 이상)", num_cols, default=default_vars if len(default_vars) >= 2 else num_cols[:3], key="var_vars")
        if sel:
            target = st.sidebar.selectbox("Target(반응)", sel, index=0, key="var_target")
            impulses = [c for c in sel if c != target]
            impulse = st.sidebar.selectbox("Impulse(충격)", impulses, index=0 if impulses else 0, key="var_impulse")
        else:
            target, impulse = None, None

        lag = st.sidebar.slider("lag", 1, 10, 1, key="var_lag")
        horizon = st.sidebar.slider("horizon", 5, 30, 10, key="var_h")
        standardize = st.sidebar.checkbox("z-score", True, key="var_z")
        show_grid = st.sidebar.checkbox("IRF 전체 그리드도 보기(옵션)", False, key="var_grid")

        if "var_out" not in st.session_state:
            st.session_state["var_out"] = None

        run_clicked = st.button("VAR 실행", type="primary")

        if run_clicked:
            try:
                with st.spinner("VAR 실행 중... (조금 걸릴 수 있어요)"):
                    g, irf, fevd_tbl, res, nrows = run_var_bundle(df, sel, target, impulse, lag, horizon, standardize)
                st.session_state["var_out"] = {
                    "g": g, "irf": irf, "fevd": fevd_tbl,
                    "target": target, "impulse": impulse,
                    "horizon": horizon, "nrows": nrows
                }
                st.success(f"완료! (학습 데이터 rows: {nrows})")
            except Exception as e:
                st.session_state["var_out"] = None
                st.error(f"VAR 실행 실패: {e}")

        out = st.session_state.get("var_out")

        if out is None:
            st.info("왼쪽에서 VAR 변수를 선택하고 **VAR 실행**을 눌러주세요.")
        else:
            st.subheader("1) Granger 인과테스트 (x → target)")
            st.caption("p-value가 작을수록 ‘x가 target을 그랜저 인과한다’ 근거가 강합니다(통상 0.05 기준).")
            st.dataframe(out["g"], use_container_width=True)

            st.divider()

            st.subheader("2) IRF (Impulse Response Functions)")
            st.caption("기본은 ‘impulse 1개 → target 1개’만 크게 보여줍니다(발표용으로 가장 깔끔).")

            # ✅ 1개 impulse -> 1개 target
            fig = out["irf"].plot(impulse=out["impulse"], response=out["target"])
            fig.set_size_inches(10, 4)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            if show_grid:
                st.caption("전체 그리드는 변수가 많으면 글자가 겹쳐 보일 수 있어요.")
                fig2 = out["irf"].plot()
                fig2.set_size_inches(12, 10)
                fig2.tight_layout()
                st.pyplot(fig2, clear_figure=True)

            st.divider()

            st.subheader("3) FEVD 분산분해 (target 기준)")
            st.caption("각 horizon에서 target 변동을 ‘어떤 shock(변수)이 얼마나 설명하는지(%)’를 보여줍니다.")
            st.dataframe(out["fevd"], use_container_width=True)

    # ------------------
    # TAB 3: Pipeline
    # ------------------
    with tab3:
        st.markdown(
            """
### 분석 파이프라인
1) `df_draft_1209_w.sti.csv` 로드 → `time` 기준 정렬  
2) CII(소비자 투자 인덱스) 계산  
3) Risk Signal(🟢🟡🔴) 요약 + 테이커 쏠림(0.5 기준) 반영  
4) VAR → Granger(표) / IRF(impulse→target 그래프) / FEVD(표)  
"""
        )


if __name__ == "__main__":
    main()
