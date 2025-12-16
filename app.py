# app.py
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
DATA_DIR = BASE_DIR / "data"

# ✅ 우선순위: draft(최신) -> var(구버전) 순으로 자동 선택
CANDIDATE_FILES = [
    "df_draft_1209_w.sti.csv",
    "df_var_1209.csv",
]


def resolve_data_path() -> Path:
    for fname in CANDIDATE_FILES:
        p = DATA_DIR / fname
        if p.exists():
            return p
    return DATA_DIR / CANDIDATE_FILES[0]


# ======================
# Utils
# ======================
def z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    mu = x.mean()
    sd = x.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# ======================
# CII (Consumer Investment Index)
# ======================
def build_cii(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터에 있는 컬럼명을 자동 매핑해서 CII를 계산합니다.
    - GT(관심): bitcoin / gtrend_btc_z14 등
    - Sentiment: avg_sent / pos_ratio / neg_ratio / rolling_mean_neg / rolling_std_neg / count 등
    """
    out = df.copy()

    alias = {
        "gt_bitcoin": ["gt_bitcoin", "bitcoin", "gtrend_bitcoin", "gt_bitcoin_raw"],
        "gt_btc_z14": ["gt_btc_z14", "gtrend_btc_z14", "gt_btc_z14"],
        "rd_avg_sent": ["rd_avg_sent", "avg_sent", "reddit_avg_sent", "sent_avg"],
        "rd_pos_ratio": ["rd_pos_ratio", "pos_ratio", "sent_pos_ratio"],
        "rd_neg_ratio": ["rd_neg_ratio", "neg_ratio", "sent_neg_ratio"],
        "rd_rolling_mean_neg": ["rd_rolling_mean_neg", "rolling_mean_neg", "neg_rolling_mean"],
        "rd_rolling_std_neg": ["rd_rolling_std_neg", "rolling_std_neg", "neg_rolling_std"],
        "rd_count": ["rd_count", "count", "rd_cnt", "doc_count"],
    }

    def pick(key: str) -> Optional[str]:
        for c in alias[key]:
            if c in out.columns:
                return c
        return None

    col = {k: pick(k) for k in alias.keys()}
    missing = [k for k, v in col.items() if v is None]

    out.attrs["cii_colmap"] = col
    out.attrs["cii_missing"] = missing

    if missing:
        out["CII"] = np.nan
        return out

    out["rd_pos_minus_neg"] = out[col["rd_pos_ratio"]] - out[col["rd_neg_ratio"]]

    out["cii_attention"] = (z(out[col["gt_bitcoin"]]) + z(out[col["gt_btc_z14"]])) / 2
    out["cii_sentiment"] = (
        z(out[col["rd_avg_sent"]]) +
        z(out["rd_pos_minus_neg"]) -
        z(out[col["rd_rolling_mean_neg"]])
    ) / 3
    out["cii_volatility"] = (
        z(out[col["rd_rolling_std_neg"]]) +
        z(out[col["rd_count"]])
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
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # 숫자형 캐스팅(문자열 숫자 대비)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df = build_cii(df)
    return df


# ======================
# Risk helpers
# ======================
def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    if series.empty or pd.isna(value):
        return "⚪️", 1

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
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


def overall_risk_text(total_score: int, count: int) -> str:
    avg = total_score / max(count, 1)
    if avg < 0.5:
        return "🟢 과열 신호는 약합니다. 급변 구간에서도 감정적 매매보다 구조(청산/펀딩/레버리지)를 먼저 확인하세요."
    if avg < 1.2:
        return "🟡 단기 변동성이 커질 수 있는 구간입니다. ‘청산/펀딩/쏠림’ 요인을 먼저 점검하세요."
    if avg < 1.8:
        return "🟠 레버리지·쏠림 주의 구간입니다. 포지션 크기/리스크 관리가 필요합니다."
    return "🔴 단기 충격 가능성이 큽니다. 무리한 레버리지는 피하고 변동성 확대를 전제로 대응하세요."


# ======================
# VAR helpers
# ======================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std().replace(0, np.nan)
    return (df - df.mean()) / std


def run_var_bundle(
    df: pd.DataFrame,
    selected_cols: List[str],
    target: str,
    lag: int,
    horizon: int,
    standardize: bool,
) -> Dict[str, object]:
    if len(selected_cols) < 2:
        raise ValueError("VAR 변수는 2개 이상 선택해야 합니다.")
    if target not in selected_cols:
        raise ValueError("타겟(반응) 변수는 선택된 VAR 변수 안에 있어야 합니다.")

    data = df[selected_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if standardize:
        data = zscore_df(data).dropna()

    if data.shape[0] < max(60, lag * 15):
        raise ValueError(f"데이터가 너무 적습니다. (현재 {data.shape[0]} rows) lag={lag} 기준 최소 60~권장")

    res = VAR(data).fit(lag)

    # --- Granger table
    rows = []
    for x in selected_cols:
        if x == target:
            continue
        try:
            test = res.test_causality(caused=target, causing=[x], kind="f")
            rows.append(
                {
                    "causing(x)": x,
                    "caused(target)": target,
                    "stat(F)": float(test.test_statistic),
                    "pvalue": float(test.pvalue),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "causing(x)": x,
                    "caused(target)": target,
                    "stat(F)": np.nan,
                    "pvalue": np.nan,
                    "error": str(e),
                }
            )
    granger = pd.DataFrame(rows).sort_values("pvalue", na_position="last").reset_index(drop=True)

    # --- IRF
    irf = res.irf(horizon)

    # --- FEVD (robust shape)
    fevd = res.fevd(horizon)
    decomp = np.array(fevd.decomp)  # (steps, response, impulse)

    names = list(res.names)
    t_idx = names.index(target)

    # step0 포함 여부 처리
    # decomp[t, response, impulse]
    if decomp.shape[0] == horizon + 1:
        decomp_use = decomp[1:, t_idx, :]  # (horizon, impulses)
        idx = list(range(1, horizon + 1))
    elif decomp.shape[0] == horizon:
        decomp_use = decomp[:, t_idx, :]
        idx = list(range(1, horizon + 1))
    else:
        # 예상치 못한 경우라도 최대한 안전하게
        decomp_use = decomp[min(1, decomp.shape[0] - 1):, t_idx, :]
        idx = list(range(1, decomp_use.shape[0] + 1))

    fevd_tbl = pd.DataFrame(decomp_use * 100.0, columns=names, index=idx).round(2)
    fevd_tbl.index.name = "horizon(step)"

    return {
        "granger": granger,
        "irf": irf,
        "fevd_tbl": fevd_tbl,
        "rows_used": int(data.shape[0]),
        "names": names,
    }


# ======================
# UI
# ======================
def main():
    st.set_page_config(page_title="시장 위험도 대시보드", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
        h1 { margin-bottom: 0.15rem; }
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.92rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 시장 위험도 대시보드")
    st.caption("Risk Signal(신호등) + VAR Insight(Granger/IRF/FEVD) + Pipeline")

    data_path = resolve_data_path()
    df = load_data(data_path)

    if df.empty:
        st.error(f"데이터 파일을 찾을 수 없습니다: `{data_path}`\n\n`data/` 폴더에 CSV가 있는지 확인해 주세요.")
        return

    # ---- Tabs
    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 Pipeline"])

    # ---- Sidebar common controls
    st.sidebar.header("설정")

    # ✅ 날짜: 최신이 위로(내림차순)
    unique_dates = sorted(pd.unique(df.index.date), reverse=True)
    sel_date = st.sidebar.selectbox(
        "기준 날짜",
        options=unique_dates,
        index=0,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        key="base_date",
    )

    # 해당 날짜 마지막 row
    day_df = df[df.index.date == sel_date]
    if day_df.empty:
        day_df = df.iloc[[-1]]
    latest_row = day_df.iloc[-1]

    # 전일 row (내림차순 리스트 기준: idx+1이 전일)
    i = unique_dates.index(sel_date)
    prev_row = None
    if i + 1 < len(unique_dates):
        prev_date = unique_dates[i + 1]
        prev_day_df = df[df.index.date == prev_date]
        if not prev_day_df.empty:
            prev_row = prev_day_df.iloc[-1]

    # ======================
    # TAB 1: Risk Signal
    # ======================
    with tab1:
        st.subheader(f"기준 데이터 날짜: {sel_date:%Y-%m-%d}")

        with st.expander("✅ 데이터 로드 정보(디버깅)"):
            st.write("사용 파일:", str(data_path))
            st.write("행/열:", df.shape)
            st.write("컬럼 수:", len(df.columns))

        # 컬럼 자동 매핑
        colmap = {
            "oi": find_column(df, ["oi_close", "oi_close_diff", "open_interest", "oi"]),
            "funding": find_column(df, ["funding_close", "funding", "funding_rate"]),
            "liq": find_column(df, ["liq_total_usd", "liq_total_usd_diff", "liquidation_usd", "liq_usd"]),
            "taker": find_column(df, ["taker_buy_ratio", "taker_ratio"]),
            "m2": find_column(df, ["global_m2_yoy_diff", "m2_yoy_diff", "global_m2_yoy"]),
        }

        # 표시용 지표 정의
        indicators = [
            ("oi", "OI", True),
            ("funding", "펀딩비", True),
            ("liq", "청산(USD)", True),
            ("taker", "테이커 비중(쏠림)", True),
            ("m2", "M2", False),
        ]

        cols = st.columns(len(indicators), gap="large")
        total_score, used = 0, 0

        for ui_col, (k, label, higher_is_risky) in zip(cols, indicators):
            real_col = colmap.get(k)
            if not real_col or real_col not in df.columns:
                ui_col.metric(label, "N/A")
                ui_col.caption("⚪️ 컬럼 없음")
                continue

            raw_val = latest_row.get(real_col, np.nan)

            # 값 표시
            display_value = "N/A"
            extra_line = ""

            # m2: 0이면 결측 취급
            if k == "m2":
                if pd.isna(raw_val) or float(raw_val) == 0.0:
                    sig, sc = "⚪️", 1
                    display_value = "0"
                    extra_line = "0값 → 결측 가능"
                else:
                    sig, sc = percentile_signal(df[real_col], float(raw_val), higher_is_risky=higher_is_risky)
                    display_value = f"{float(raw_val):,.4g}"
            elif k == "taker":
                # 쏠림: |x-0.5|
                if pd.isna(raw_val):
                    sig, sc = "⚪️", 1
                    display_value = "N/A"
                else:
                    v = abs(float(raw_val) - 0.5)
                    series = (df[real_col] - 0.5).abs()
                    sig, sc = percentile_signal(series, v, higher_is_risky=True)
                    display_value = f"{float(raw_val):.3f}"
                    extra_line = f"쏠림 |x-0.5| = {v:.3f}"
            else:
                if pd.isna(raw_val):
                    sig, sc = "⚪️", 1
                    display_value = "N/A"
                else:
                    sig, sc = percentile_signal(df[real_col], float(raw_val), higher_is_risky=higher_is_risky)
                    display_value = f"{float(raw_val):,.4g}"

            # 전일 대비 delta
            delta_txt = None
            if prev_row is not None and real_col in prev_row.index:
                try:
                    pv = prev_row.get(real_col, np.nan)
                    if pd.isna(pv) or pd.isna(raw_val):
                        delta_txt = None
                    else:
                        dv = float(raw_val) - float(pv)
                        # 소수형은 4자리, 큰 값은 천단위 구분
                        if k in ["funding", "taker", "m2"]:
                            delta_txt = f"{dv:+.4f}"
                        else:
                            delta_txt = f"{dv:+,.0f}"
                except Exception:
                    delta_txt = None

            ui_col.metric(label, display_value, delta=delta_txt)
            ui_col.caption(f"신호: {sig} · 컬럼: `{real_col}`")
            if extra_line:
                ui_col.caption(extra_line)

            total_score += sc
            used += 1

        st.divider()
        st.subheader("신호등 요약")
        st.write("🟢 낮음 | 🟡 중간 | 🔴 높음 | ⚪️ 데이터 부족/결측/컬럼 없음")
        st.success(overall_risk_text(total_score, used))

        # ---- CII
        st.subheader("📈 소비자 투자 인덱스 (CII)")
        if "CII" not in df.columns or df["CII"].dropna().empty:
            st.warning("CII 계산 불가 (필요 컬럼 부족)")
            with st.expander("CII 디버깅 정보"):
                st.write("missing:", df.attrs.get("cii_missing"))
                st.write("colmap:", df.attrs.get("cii_colmap"))
        else:
            # 기준 날짜의 CII 마지막 값(가능하면)
            cii_val = day_df["CII"].dropna().iloc[-1] if ("CII" in day_df.columns and not day_df["CII"].dropna().empty) else df["CII"].dropna().iloc[-1]
            st.metric("CII (latest)", f"{float(cii_val):.2f}")
            st.line_chart(df["CII"].tail(200))

        # ---- Raw preview (가로 스크롤 가능)
        with st.expander("원본 데이터 미리보기 (가로 스크롤 가능)"):
            st.dataframe(df.tail(50), use_container_width=True, height=520)

    # ======================
    # TAB 2: VAR Insight
    # ======================
    with tab2:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger(표) / IRF(그래프) / FEVD(표)")

        st.sidebar.header("VAR 설정")

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # 디폴트 후보
        default_candidates = [c for c in [
            "ret_log_1d",
            "funding_close",
            "taker_buy_ratio",
            "oi_close",
            "oi_close_diff",
            "liq_total_usd",
            "liq_total_usd_diff",
            "global_m2_yoy_diff",
        ] if c in numeric_cols]

        sel_cols = st.sidebar.multiselect(
            "VAR 변수(2개 이상)",
            options=numeric_cols,
            default=(default_candidates[:4] if len(default_candidates) >= 2 else numeric_cols[:3]),
        )

        target = st.sidebar.selectbox("Target(반응)", options=sel_cols, index=0) if sel_cols else None

        impulse_options = [c for c in sel_cols if c != target] if sel_cols else []
        impulse = st.sidebar.selectbox("Impulse(충격)", options=impulse_options, index=0) if impulse_options else None

        lag = st.sidebar.slider("lag", 1, 10, 1)
        horizon = st.sidebar.slider("horizon", 5, 30, 10)
        standardize = st.sidebar.checkbox("z-score 표준화", True)
        show_full_grid = st.sidebar.checkbox("IRF 전체 그리드(옵션)", False)

        run_btn = st.button("VAR 실행", type="primary")

        if "var_out" not in st.session_state:
            st.session_state["var_out"] = None
            st.session_state["var_params"] = None

        if run_btn:
            try:
                with st.spinner("VAR 적합 중…"):
                    out = run_var_bundle(
                        df=df,
                        selected_cols=sel_cols,
                        target=target,
                        lag=lag,
                        horizon=horizon,
                        standardize=standardize,
                    )
                st.session_state["var_out"] = out
                st.session_state["var_params"] = {
                    "sel_cols": sel_cols,
                    "target": target,
                    "impulse": impulse,
                    "lag": lag,
                    "horizon": horizon,
                    "standardize": standardize,
                    "show_full_grid": show_full_grid,
                }
                st.success(f"완료! (rows used: {out['rows_used']})")
            except Exception as e:
                st.session_state["var_out"] = None
                st.session_state["var_params"] = None
                st.error(f"VAR 실행 실패: {e}")

        out = st.session_state.get("var_out")
        params = st.session_state.get("var_params")

        if out is None:
            st.info("왼쪽에서 변수 선택 후 **VAR 실행** 버튼을 눌러주세요.")
        else:
            # Granger
            st.subheader("1) Granger 인과테스트 (x → target)")
            st.caption("p-value가 작을수록 x가 target을 그랜저 인과한다는 근거가 강합니다(보통 0.05 기준).")
            st.dataframe(out["granger"], use_container_width=True, height=320)

            st.divider()

            # IRF (1개 impulse -> 1개 target)
            st.subheader("2) IRF (Impulse Response Functions)")
            st.caption("발표/데모에서는 ‘impulse 1개 → target 1개’ 그래프가 가장 읽기 좋습니다.")

            irf = out["irf"]
            imp = params.get("impulse")
            tgt = params.get("target")

            if imp is None or tgt is None:
                st.warning("Impulse/Target 설정이 필요합니다.")
            else:
                fig = irf.plot(impulse=imp, response=tgt)
                fig.set_size_inches(10, 4.2)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                if params.get("show_full_grid", False):
                    st.caption("전체 그리드는 변수가 많으면 겹쳐 보여요(옵션).")
                    fig2 = irf.plot()
                    fig2.set_size_inches(12, 9)
                    fig2.tight_layout()
                    st.pyplot(fig2, clear_figure=True)

            st.divider()

            # FEVD
            st.subheader("3) FEVD 분산분해 (target 기준)")
            st.caption("각 horizon에서 target 변동을 ‘어떤 shock(변수)이 얼마나 설명하는지(%)’를 보여줍니다.")
            st.dataframe(out["fevd_tbl"], use_container_width=True, height=420)

    # ======================
    # TAB 3: Pipeline
    # ======================
    with tab3:
        st.subheader("🧭 분석 파이프라인")
        st.markdown(
            f"""
1. **데이터 로드**
   - `data/{data_path.name}` 로드 → `time` 기준 정렬

2. **CII(소비자 투자 인덱스) 계산**
   - Google Trends(관심) + Sentiment(정서/부정 변동/볼륨)
   - 컬럼명은 alias로 자동 매핑 (없으면 CII는 NaN)

3. **Risk Signal**
   - OI / Funding / Liquidation / Taker / M2
   - 분위수(33/66%) 기반 🟢🟡🔴 신호

4. **VAR Insight**
   - 변수 선택 → (옵션) z-score → VAR 적합
   - **Granger** 표 / **IRF** 그래프 / **FEVD** 표
"""
        )


if __name__ == "__main__":
    main()
