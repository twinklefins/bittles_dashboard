import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR


# ======================
# Paths
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DRAFT_PATH = DATA_DIR / "df_draft_1209_w.sti.csv"
VAR_PATH = DATA_DIR / "df_var_1209.csv"


# ======================
# Utils
# ======================
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def zscore(series: pd.Series) -> pd.Series:
    x = series.replace([np.inf, -np.inf], np.nan)
    mu = x.mean()
    sd = x.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    """
    Return (emoji, score)
    score: 🟢0, 🟡1, 🔴2, ⚪️1
    """
    if series is None or series.empty or pd.isna(value):
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


def overall_risk_text(total_score: int, count: int) -> str:
    avg = total_score / max(count, 1)
    if avg < 0.5:
        return "🟢 과열/쏠림 신호가 약합니다. 단기 노이즈 가능성이 큽니다."
    if avg < 1.2:
        return "🟡 단기 변동성이 커질 수 있는 구간입니다. 패닉셀보다는 원인(레버리지/쏠림/유동성)을 먼저 점검하세요."
    if avg < 1.8:
        return "🟠 레버리지·쏠림 신호가 관측됩니다. 포지션 크기/리스크 관리가 필요합니다."
    return "🔴 단기 충격(강제청산/쏠림) 가능성이 높습니다. 무리한 레버리지는 피하세요."


# ======================
# CII (optional)
# ======================
def build_cii(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ✅ 실제 데이터 컬럼명 ↔ 코드에서 쓰고 싶은 논리명 매핑
    alias = {
        "gt_bitcoin": ["gt_bitcoin", "bitcoin", "gtrend_bitcoin"],
        "gt_btc_z14": ["gt_btc_z14", "gtrend_btc_z14", "gt_btc_z14"],
        "rd_avg_sent": ["rd_avg_sent", "avg_sent", "reddit_avg_sent"],
        "rd_pos_ratio": ["rd_pos_ratio", "pos_ratio"],
        "rd_neg_ratio": ["rd_neg_ratio", "neg_ratio"],
        "rd_rolling_mean_neg": ["rd_rolling_mean_neg", "rolling_mean_neg"],
        "rd_rolling_std_neg": ["rd_rolling_std_neg", "rolling_std_neg"],
        "rd_count": ["rd_count", "count", "rd_cnt"],
    }

    def pick(name: str) -> Optional[str]:
        for c in alias[name]:
            if c in out.columns:
                return c
        return None

    # 실제로 존재하는 컬럼명으로 resolve
    col = {k: pick(k) for k in alias.keys()}

    missing = [k for k, v in col.items() if v is None]
    if missing:
        out["CII"] = np.nan
        # 디버깅용: 어떤 컬럼이 없어서 실패했는지 확인
        out.attrs["cii_missing"] = missing
        out.attrs["cii_colmap"] = col
        return out

    # ---- 계산 ----
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

    out.attrs["cii_missing"] = []
    out.attrs["cii_colmap"] = col
    return out


# ======================
# Data loader
# ======================
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, str]:
    """
    draft 우선 로드, 없으면 var 로드.
    return (df, source_name)
    """
    if DRAFT_PATH.exists():
        path = DRAFT_PATH
        source = "df_draft_1209_w.sti.csv"
    elif VAR_PATH.exists():
        path = VAR_PATH
        source = "df_var_1209.csv"
    else:
        return pd.DataFrame(), "NO_FILE"

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = safe_to_datetime(df["time"])
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # CII 시도(없으면 자동 NaN)
    df = build_cii(df)

    return df, source


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

    data = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < max(60, lag * 12):
        raise ValueError(f"데이터가 너무 적습니다. (rows={data.shape[0]}). lag={lag}면 최소 60행 이상 권장")

    if standardize:
        data = zscore_df(data).dropna()

    res = VAR(data).fit(lag)

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

    # IRF
    irf = res.irf(horizon)

    # FEVD (shape 방어)
    fevd = res.fevd(horizon)
    decomp = np.array(fevd.decomp)  # (steps, response, impulse)
    names = list(res.names)
    t_idx = names.index(target)

    # step0 포함 여부 처리
    if decomp.shape[0] == horizon + 1:
        use = decomp[1:, t_idx, :]   # (horizon, impulse)
        idx = list(range(1, horizon + 1))
    else:
        use = decomp[:, t_idx, :]
        idx = list(range(1, decomp.shape[0] + 1))

    fevd_tbl = pd.DataFrame((use * 100.0), columns=names, index=idx).round(2)
    fevd_tbl.index.name = "horizon(step)"

    return {
        "granger": granger,
        "irf": irf,
        "fevd": fevd_tbl,
        "rows": int(data.shape[0]),
        "names": names,
    }


# ======================
# App
# ======================
def main():
    st.set_page_config(page_title="시장 위험도 대시보드", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.9rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 시장 위험도 대시보드")
    st.caption("Risk Signal(신호등) + VAR Insight(Granger/IRF/FEVD) + Pipeline")

    df, source = load_data()
    if df.empty:
        st.error("데이터 파일을 찾을 수 없습니다. repo의 data/ 폴더에 CSV가 있는지 확인하세요.")
        st.info("필요 파일: data/df_draft_1209_w.sti.csv 또는 data/df_var_1209.csv")
        return

    with st.expander("✅ 데이터 로드 정보(디버깅)"):
        st.write(f"source: **{source}**")
        st.write(f"rows: **{len(df):,}**, cols: **{len(df.columns):,}**")
        st.write("columns sample:", list(df.columns)[:30])

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 Pipeline"])

    # ======================
    # Sidebar: 공용 설정 (날짜)
    # ======================
    st.sidebar.header("설정")
    unique_dates = sorted(pd.unique(df.index.date), reverse=True)  # ✅ 내림차순
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
        day_df = df
    latest_row = day_df.iloc[-1]

    # 전일(바로 다음 날짜가 전일) row
    idx = unique_dates.index(sel_date)
    prev_row = None
    if idx + 1 < len(unique_dates):
        prev_date = unique_dates[idx + 1]
        prev_df = df[df.index.date == prev_date]
        if not prev_df.empty:
            prev_row = prev_df.iloc[-1]

    # ======================
    # TAB 1: Risk Signal
    # ======================
    with tab1:
        st.subheader(f"기준 데이터 날짜: {latest_row.name:%Y-%m-%d}")

        # ✅ 데이터 스키마가 달라도 자동으로 컬럼 찾기
        col_oi = find_column(df, ["oi_close_diff", "oi_close", "open_interest", "oi"])
        col_funding = find_column(df, ["funding_close", "funding_rate", "funding"])
        col_liq = find_column(df, ["liq_total_usd_diff", "liq_total_usd", "liquidation_usd", "liq_usd"])
        col_taker = find_column(df, ["taker_buy_ratio", "taker_ratio"])
        col_m2 = find_column(df, ["global_m2_yoy_diff", "global_m2_yoy", "m2_yoy_diff", "m2_yoy"])

        indicators = [
            ("OI", col_oi, True, "level_or_diff"),
            ("펀딩비", col_funding, True, "level"),
            ("청산(USD)", col_liq, True, "level_or_diff"),
            ("테이커 비중(쏠림)", col_taker, True, "taker"),
            ("M2", col_m2, False, "level_or_diff"),
        ]

        cols = st.columns(len(indicators), gap="large")
        total_score, used = 0, 0

        for ui_col, (label, colname, higher_is_risky, mode) in zip(cols, indicators):
            if colname is None or colname not in df.columns:
                ui_col.metric(label, "N/A")
                ui_col.caption("⚪️ 컬럼 없음")
                continue

            v = latest_row[colname]
            if pd.isna(v):
                ui_col.metric(label, "N/A")
                ui_col.caption("⚪️ 결측")
                continue

            # signal 계산
            extra = ""
            if mode == "taker":
                # 0.5에서 멀수록 쏠림
                series = (df[colname] - 0.5).abs()
                val = float(abs(v - 0.5))
                signal, score = percentile_signal(series, val, higher_is_risky=True)
                display_value = f"{float(v):.3f}"
                extra = f"쏠림 |x-0.5| = {val:.3f}"
            else:
                series = df[colname]
                signal, score = percentile_signal(series, float(v), higher_is_risky=higher_is_risky)
                display_value = f"{float(v):,.4g}"

            # delta (전일 대비)
            delta_txt = None
            if prev_row is not None and colname in prev_row.index and not pd.isna(prev_row[colname]):
                dv = float(v) - float(prev_row[colname])
                # funding/taker/m2는 소수
                if colname in [col_funding, col_taker, col_m2]:
                    delta_txt = f"{dv:+.4f}"
                else:
                    delta_txt = f"{dv:+,.0f}"

            ui_col.metric(label, display_value, delta=delta_txt)
            ui_col.caption(f"신호: {signal}  ·  컬럼: `{colname}`")
            if extra:
                ui_col.caption(extra)

            total_score += score
            used += 1

        st.divider()
        st.subheader("신호등 요약")
        st.write("🟢 낮음 | 🟡 중간 | 🔴 높음 | ⚪️ 데이터 부족/결측/컬럼 없음")
        st.success(overall_risk_text(total_score, used))

        st.subheader("📈 소비자 투자 인덱스 (CII)")
        if "CII" not in df.columns or df["CII"].dropna().empty:
            st.warning("CII 계산 불가 (필요 컬럼 부족)")
        else:
            st.metric("CII (latest)", f"{df['CII'].iloc[-1]:.2f}")
            st.line_chart(df["CII"].tail(200))

        with st.expander("원본 데이터 미리보기 (가로 스크롤 가능)"):
            st.caption("표 내부에서 가로 스크롤하면 모든 컬럼을 확인할 수 있습니다.")
            st.dataframe(df.tail(50), use_container_width=True)

    # ======================
    # TAB 2: VAR Insight
    # ======================
    with tab2:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger(표) / IRF(그래프) / FEVD(표)")

        st.sidebar.header("VAR 설정")

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # 기본 추천: ret_log_1d, funding_close, taker_buy_ratio 등 (있으면)
        default_candidates = [c for c in ["ret_log_1d", "funding_close", "taker_buy_ratio", "oi_close", "spot_volume_usd", "price_close"] if c in numeric_cols]
        default_sel = default_candidates[:4] if len(default_candidates) >= 2 else numeric_cols[:4]

        selected_cols = st.sidebar.multiselect(
            "VAR 변수(2개 이상)",
            options=numeric_cols,
            default=default_sel,
            key="var_cols",
        )

        if selected_cols:
            target = st.sidebar.selectbox("Target(반응)", options=selected_cols, index=0, key="var_target")
            impulse_opts = [c for c in selected_cols if c != target]
            impulse = st.sidebar.selectbox("Impulse(충격)", options=impulse_opts, index=0 if impulse_opts else 0, key="var_impulse")
        else:
            target, impulse = None, None

        lag = st.sidebar.slider("lag", 1, 10, 1, key="var_lag")
        horizon = st.sidebar.slider("horizon", 5, 30, 10, key="var_h")
        standardize = st.sidebar.checkbox("z-score 표준화", True, key="var_z")
        show_grid = st.sidebar.checkbox("IRF 전체 그리드(옵션)", False, key="var_grid")

        run_btn = st.button("VAR 실행", type="primary")

        if "var_out" not in st.session_state:
            st.session_state["var_out"] = None
            st.session_state["var_params"] = None

        if run_btn:
            try:
                with st.spinner("VAR 적합 중…"):
                    out = run_var_bundle(df, selected_cols, target, lag, horizon, standardize)
                st.session_state["var_out"] = out
                st.session_state["var_params"] = {
                    "target": target,
                    "impulse": impulse,
                    "horizon": horizon,
                    "show_grid": show_grid,
                }
                st.success(f"완료! (학습 rows: {out['rows']})")
            except Exception as e:
                st.error(f"VAR 실패: {e}")
                st.session_state["var_out"] = None
                st.session_state["var_params"] = None

        out = st.session_state.get("var_out")
        params = st.session_state.get("var_params")

        if out is None:
            st.info("왼쪽에서 변수 선택 후 **VAR 실행**을 눌러주세요.")
        else:
            st.subheader("1) Granger 인과테스트 (x → target)")
            st.caption("p-value가 작을수록 ‘x가 target을 그랜저 인과한다’는 근거가 강합니다(보통 0.05 기준).")
            st.dataframe(out["granger"], use_container_width=True)

            st.divider()

            st.subheader("2) IRF (Impulse Response)")
            st.caption("발표용으로는 impulse 1개 → target 1개만 크게 보여주는 게 가장 읽기 좋아요.")
            irf = out["irf"]
            tgt = params["target"]
            imp = params["impulse"]

            if imp is None or tgt is None:
                st.warning("Impulse/Target 설정이 필요합니다.")
            else:
                fig = irf.plot(impulse=imp, response=tgt)
                fig.set_size_inches(10, 4)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                if params.get("show_grid"):
                    st.caption("전체 그리드는 변수 많으면 겹쳐 보일 수 있어요.")
                    fig2 = irf.plot()
                    fig2.set_size_inches(12, 10)
                    fig2.tight_layout()
                    st.pyplot(fig2, clear_figure=True)

            st.divider()

            st.subheader("3) FEVD 분산분해 (target 기준)")
            st.caption("각 horizon에서 target 변동을 ‘어떤 shock(변수)이 얼마나 설명하는지(%)’를 보여줍니다.")
            st.dataframe(out["fevd"], use_container_width=True)

    # ======================
    # TAB 3: Pipeline
    # ======================
    with tab3:
        st.subheader("🧭 분석 파이프라인")
        st.markdown(
            f"""
1. **데이터 로드**
   - 우선: `data/df_draft_1209_w.sti.csv`
   - 없으면: `data/df_var_1209.csv`
   - 현재 로드 소스: **{source}**

2. **Risk Signal**
   - 데이터 스키마가 달라도 자동으로 컬럼 후보를 탐색해 신호등 생성
   - 테이커 비중은 `|taker_buy_ratio - 0.5|`로 쏠림 계산

3. **(옵션) CII**
   - GT/Reddit 관련 컬럼이 존재할 때만 CII 계산 및 시각화

4. **VAR Insight**
   - VAR 적합 → **Granger(표) / IRF(그래프) / FEVD(표)**
"""
        )


if __name__ == "__main__":
    main()
