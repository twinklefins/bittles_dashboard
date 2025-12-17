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
DATA_PATH = BASE_DIR / "data" / "df_draft_1209_w.sti.csv"


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
        st.error("CSV에 'time' 컬럼이 없습니다.")
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # 숫자형 컬럼 내 inf 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ======================
# Utils
# ======================
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    """Return emoji + score (green=0, yellow=1, red=2, neutral=1)."""
    if series.empty or pd.isna(value):
        return "⚪️", 1

    s = series.dropna()
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


def overall_risk_text(avg_score: float) -> str:
    if avg_score < 0.5:
        return "🟢 현재는 구조적 과열 신호가 약합니다. 급락 시에도 패닉셀보다 ‘원인(청산/펀딩/유동성)’ 확인이 우선입니다."
    if avg_score < 1.2:
        return "🟡 단기 변동성이 커질 수 있는 구간입니다. 패닉셀보다는 ‘청산/펀딩/쏠림’ 요인이 있는지 먼저 점검하세요."
    if avg_score < 1.8:
        return "🟠 레버리지·쏠림 신호가 관측됩니다. 포지션 크기/리스크 관리를 권장합니다."
    return "🔴 단기 충격(청산/레버리지) 가능성이 높습니다. 무리한 레버리지는 피하고 변동성 확대를 전제로 대응하세요."


def market_mood_label(avg_score: float) -> Tuple[str, str]:
    """Market Mood (쉬운 말 + 재밌는 톤)"""
    if avg_score < 0.5:
        return "🔵 Calm", "시장 과열/쏠림 신호가 낮은 편입니다. 단기 변동은 ‘구조’보다 ‘이벤트’ 요인일 가능성이 큽니다."
    if avg_score < 1.0:
        return "🟢 Stable", "심리와 레버리지 지표가 비교적 균형입니다. 과열 신호는 제한적입니다."
    if avg_score < 1.4:
        return "🟡 Warm", "시장 관심이 올라오는 구간입니다. 변동성이 커질 수 있어요(쏠림/청산 체크 권장)."
    if avg_score < 1.8:
        return "🟠 Hot", "레버리지·쏠림 신호가 관측됩니다. 단기 조정/변동성 확대 가능성에 유의하세요."
    return "🔴 Too Hot", "과열 경고 구간입니다. 청산/급변동 리스크가 커질 수 있어 ‘감정 매매’는 특히 위험합니다."


def build_cause_summary(signals: Dict[str, str]) -> str:
    """Short, human-readable summary."""
    red = [k for k, s in signals.items() if s == "🔴"]
    yellow = [k for k, s in signals.items() if s == "🟡"]
    neutral = [k for k, s in signals.items() if s == "⚪️"]

    desc = {
        "oi": "레버리지(OI)",
        "funding": "펀딩(쏠림/과열)",
        "liq": "청산(강제 체결)",
        "taker": "테이커 쏠림",
        "m2": "유동성(M2)",
    }

    lines = []
    if red:
        lines.append("🔴 **강한 단기 리스크 신호**: " + ", ".join(desc[x] for x in red))
        lines.append("→ 급변 구간에서는 ‘원인 확인(청산/펀딩/레버리지)’이 먼저이고, 즉흥 매매는 손실 확률을 키웁니다.")
    elif yellow:
        lines.append("🟡 **변동성 확대 신호**: " + ", ".join(desc[x] for x in yellow))
        lines.append("→ 당일 급락/급등이 ‘쏠림’으로 과장될 수 있어 추세인지 구조인지 확인하세요.")
    else:
        lines.append("🟢 **과열 신호가 뚜렷하지 않습니다.** 단기 노이즈 가능성이 큽니다.")

    if neutral:
        lines.append("⚪️ 데이터 부족/결측 가능: " + ", ".join(desc[x] for x in neutral))

    return "\n\n".join(lines)


# ======================
# VAR helpers
# ======================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std().replace(0, np.nan)
    return (df - mu) / sd


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
        raise ValueError("타겟 변수는 선택된 VAR 변수 안에 있어야 합니다.")

    data = df[selected_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.shape[0] < max(80, lag * 12):
        raise ValueError(f"데이터가 부족합니다. (현재 {data.shape[0]} rows) lag={lag}면 80행 이상 권장")

    if standardize:
        data = zscore_df(data).dropna()

    model = VAR(data)
    results = model.fit(lag)

    # Granger: x -> target
    rows = []
    for x in selected_cols:
        if x == target:
            continue
        try:
            test = results.test_causality(caused=target, causing=[x], kind="f")
            rows.append(
                {
                    "causing(x)": x,
                    "caused(target)": target,
                    "stat(F)": float(test.test_statistic),
                    "pvalue": float(test.pvalue),
                }
            )
        except Exception as e:
            rows.append({"causing(x)": x, "caused(target)": target, "stat(F)": np.nan, "pvalue": np.nan, "error": str(e)})

    granger_df = pd.DataFrame(rows).sort_values("pvalue", na_position="last").reset_index(drop=True)

    # IRF / FEVD
    irf = results.irf(horizon)
    fevd = results.fevd(horizon)

    varnames = list(results.names)
    t_idx = varnames.index(target)

    decomp = np.array(fevd.decomp)  # (steps, response, impulse)
    # step 0 제거 (표 보기 좋게)
    if decomp.shape[0] >= 2:
        decomp = decomp[1:, :, :]
        step_index = list(range(1, decomp.shape[0] + 1))
    else:
        step_index = list(range(decomp.shape[0]))

    fevd_target = decomp[:, t_idx, :]  # (steps, impulse)
    fevd_table = pd.DataFrame(fevd_target * 100.0, columns=varnames, index=step_index).round(2)
    fevd_table.index.name = "horizon(step)"

    return {
        "granger_table": granger_df,
        "irf": irf,
        "fevd_table_target": fevd_table,
        "var_results": results,
        "var_rows": data.shape[0],
        "var_names": varnames,
    }


# ======================
# App
# ======================
def main() -> None:
    st.set_page_config(page_title="Bittles Dashboard", page_icon="📊", layout="wide")

    # UI polish: metric 라벨 줄바꿈/잘림 방지
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.9rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Bittles Dashboard")
    st.caption("Risk Signal → Market Mood → VAR(Granger/IRF/FEVD)로 ‘시장 상태’를 해석하는 대시보드")

    df = load_data(DATA_PATH)
    if df.empty:
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧠 Market Mood", "🧩 VAR Insight"])

    # ======================
    # Tab 1: Risk Signal
    # ======================
    with tab1:
        st.sidebar.header("설정")

        # ✅ 날짜: 최근 날짜가 맨 위 (내림차순)
        unique_dates = sorted(pd.unique(df.index.date), reverse=True)

        selected_date = st.sidebar.selectbox(
            "기준 날짜(최근이 위)",
            options=unique_dates,
            index=0,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="risk_date",
        )

        selected_df = df[df.index.date == selected_date]
        if selected_df.empty:
            st.error("선택한 날짜 데이터가 없습니다.")
            st.stop()

        today = selected_df.iloc[-1]

        # 전일 (unique_dates는 내림차순이므로 +1이 전일)
        idx = unique_dates.index(selected_date)
        prev_date = unique_dates[idx + 1] if idx < len(unique_dates) - 1 else None
        prev_row = df[df.index.date == prev_date].iloc[-1] if prev_date else None

        st.subheader(f"기준 시점: {today.name:%Y-%m-%d %H:%M:%S}")

        # 컬럼 매핑(자동)
        colmap: Dict[str, Optional[str]] = {
            "oi": find_column(df, ["oi_close_diff", "oi_diff", "open_interest_diff", "open_interest", "oi"]),
            "funding": find_column(df, ["funding_close", "funding", "funding_rate"]),
            "liq": find_column(df, ["liq_total_usd_diff", "liquidation_usd", "liq_usd", "liq_total_usd"]),
            "taker": find_column(df, ["taker_buy_ratio", "taker_ratio"]),
            "m2": find_column(df, ["global_m2_yoy_diff", "m2_yoy_diff", "global_m2_yoy", "m2_yoy"]),
        }

        indicators = {
            "oi": {"title": "OI 변화", "higher_is_risky": True},
            "funding": {"title": "펀딩비", "higher_is_risky": True},
            "liq": {"title": "청산(USD)", "higher_is_risky": True},
            "taker": {"title": "테이커 비중", "higher_is_risky": True},
            "m2": {"title": "M2(유동성)", "higher_is_risky": False},
        }

        cols = st.columns(len(indicators), gap="large")

        total_score, used = 0, 0
        signal_map: Dict[str, str] = {}

        for ui_col, (k, meta) in zip(cols, indicators.items()):
            real_col = colmap.get(k)
            if not real_col:
                ui_col.metric(meta["title"], "N/A")
                ui_col.caption("⚪️ (컬럼 없음)")
                signal_map[k] = "⚪️"
                continue

            value = today.get(real_col, np.nan)
            if pd.isna(value):
                ui_col.metric(meta["title"], "N/A")
                ui_col.caption(f"⚪️ `{real_col}` 결측")
                signal_map[k] = "⚪️"
                continue

            value = float(value)

            # taker는 0.5에서 멀수록 쏠림(리스크)로 보기 좋음
            extra_line = ""
            if k == "taker":
                series = (df[real_col] - 0.5).abs()
                v = abs(value - 0.5)
                signal, score = percentile_signal(series, v, higher_is_risky=True)
                display_value = f"{value:.3f}"
                extra_line = f"쏠림 |{v:.3f}| (0.5에서 멀수록 쏠림)"
            else:
                series = df[real_col]
                signal, score = percentile_signal(series, value, higher_is_risky=meta["higher_is_risky"])
                display_value = f"{value:,.4g}"

            # 전일 대비
            delta_txt = None
            if prev_row is not None and real_col in prev_row.index:
                pv = prev_row.get(real_col, np.nan)
                if not pd.isna(pv):
                    pv = float(pv)
                    delta = value - pv
                    if k in ["funding", "taker", "m2"]:
                        delta_txt = f"{delta:+.4f}"
                    else:
                        delta_txt = f"{delta:+,.0f}"

            ui_col.metric(meta["title"], display_value, delta=delta_txt)
            ui_col.caption(f"{signal}  ·  `{real_col}`")
            if extra_line:
                ui_col.caption(extra_line)

            signal_map[k] = signal
            total_score += score
            used += 1

        st.divider()

        avg_score = total_score / max(used, 1)
        st.subheader("요약")
        st.success(overall_risk_text(avg_score))
        st.info(build_cause_summary(signal_map))

        with st.expander("데이터 미리보기(컬럼 선택)"):
            all_cols = list(df.columns)
            default_cols = [c for c in ["ret_log_1d", colmap.get("oi"), colmap.get("funding"), colmap.get("liq"), colmap.get("taker"), colmap.get("m2")] if c]
            picked = st.multiselect("표시할 컬럼", options=all_cols, default=default_cols)
            if picked:
                st.dataframe(df[picked].tail(200), use_container_width=True)
            else:
                st.dataframe(df.tail(50), use_container_width=True)

    # ======================
    # Tab 2: Market Mood
    # ======================
    with tab2:
        st.subheader("🧠 Market Mood")
        st.caption("Risk Signal(지표 신호등) 점수를 ‘사람이 이해하기 쉬운 언어’로 번역한 상태 지표입니다.")

        # Risk 탭에서 계산한 avg_score가 없을 수 있으니, 여기서도 안전 계산
        # (같은 selected_date를 써서 일관되게)
        unique_dates = sorted(pd.unique(df.index.date), reverse=True)
        selected_date = st.sidebar.selectbox(
            "Market Mood 기준 날짜(최근이 위)",
            options=unique_dates,
            index=0,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="mood_date",
        )
        selected_df = df[df.index.date == selected_date]
        today = selected_df.iloc[-1] if not selected_df.empty else df.iloc[-1]

        # Risk score 재계산(간단히 동일 로직)
        colmap2 = {
            "oi": find_column(df, ["oi_close_diff", "oi_diff", "open_interest_diff", "open_interest", "oi"]),
            "funding": find_column(df, ["funding_close", "funding", "funding_rate"]),
            "liq": find_column(df, ["liq_total_usd_diff", "liquidation_usd", "liq_usd", "liq_total_usd"]),
            "taker": find_column(df, ["taker_buy_ratio", "taker_ratio"]),
            "m2": find_column(df, ["global_m2_yoy_diff", "m2_yoy_diff", "global_m2_yoy", "m2_yoy"]),
        }
        indicators2 = {
            "oi": {"higher_is_risky": True},
            "funding": {"higher_is_risky": True},
            "liq": {"higher_is_risky": True},
            "taker": {"higher_is_risky": True},
            "m2": {"higher_is_risky": False},
        }

        total, cnt = 0, 0
        for k, meta in indicators2.items():
            c = colmap2.get(k)
            if not c or c not in df.columns:
                continue
            v = today.get(c, np.nan)
            if pd.isna(v):
                continue
            v = float(v)
            if k == "taker":
                s = (df[c] - 0.5).abs()
                vv = abs(v - 0.5)
                _, sc = percentile_signal(s, vv, higher_is_risky=True)
            else:
                _, sc = percentile_signal(df[c], v, higher_is_risky=meta["higher_is_risky"])
            total += sc
            cnt += 1

        avg = total / max(cnt, 1)
        mood, mood_desc = market_mood_label(avg)

        st.markdown(
            f"""
            ### {mood}

            {mood_desc}

            **(참고)** Market Mood는 ‘가격 예측’이 아니라 **현재 시장의 구조/심리 상태를 요약**합니다.
            """
        )

        st.divider()
        st.write("**구간 안내**")
        st.write("- 🔵 Calm  → 🟢 Stable → 🟡 Warm → 🟠 Hot → 🔴 Too Hot")

    # ======================
    # Tab 3: VAR Insight
    # ======================
    with tab3:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger(표) / IRF(그래프) / FEVD(표)")

        st.sidebar.header("VAR 설정")

        all_numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        default_candidates = [c for c in ["ret_log_1d", "oi_close_diff", "funding_close", "liq_total_usd_diff", "taker_buy_ratio", "global_m2_yoy_diff"] if c in df.columns]
        default_sel = default_candidates[:4] if len(default_candidates) >= 2 else all_numeric_cols[:3]

        selected_cols = st.sidebar.multiselect(
            "VAR 변수 선택(2개 이상)",
            options=all_numeric_cols,
            default=default_sel,
            key="var_cols",
        )

        if selected_cols:
            target = st.sidebar.selectbox("타겟(반응) 변수", options=selected_cols, index=0, key="var_target")
            impulse_options = [c for c in selected_cols if c != target]
            impulse_var = st.sidebar.selectbox(
                "IRF Impulse(충격) 변수",
                options=impulse_options if impulse_options else selected_cols,
                index=0,
                key="var_impulse",
            )
        else:
            target, impulse_var = None, None

        lag = st.sidebar.slider("VAR lag", min_value=1, max_value=10, value=1, key="var_lag")
        horizon = st.sidebar.slider("IRF/FEVD horizon", min_value=5, max_value=30, value=10, key="var_h")
        standardize = st.sidebar.checkbox("표준화(z-score) 후 적합", value=True, key="var_z")
        show_full_grid = st.sidebar.checkbox("IRF 전체 그리드도 보기", value=False, key="var_grid")

        run_btn = st.button("VAR 실행", type="primary")

        if run_btn:
            try:
                with st.spinner("VAR 적합 중…"):
                    out = run_var_bundle(
                        df=df,
                        selected_cols=selected_cols,
                        target=target,
                        lag=lag,
                        horizon=horizon,
                        standardize=standardize,
                    )
                st.session_state["var_out"] = out
                st.session_state["var_params"] = {
                    "target": target,
                    "impulse": impulse_var,
                    "horizon": horizon,
                    "show_full_grid": show_full_grid,
                }
                st.success(f"완료! (학습 데이터 rows: {out['var_rows']})")
            except Exception as e:
                st.error(f"VAR 실행 실패: {e}")
                st.session_state["var_out"] = None
                st.session_state["var_params"] = None

        out = st.session_state.get("var_out")
        params = st.session_state.get("var_params")

        if not out:
            st.info("왼쪽에서 변수 선택 후 **VAR 실행**을 눌러주세요.")
            st.stop()

        # 1) Granger
        st.subheader("1) Granger 인과테스트 (x → target)")
        st.caption("p-value가 작을수록 ‘x가 target을 그랜저 인과한다’는 근거가 강합니다(관행적으로 0.05 기준).")
        st.dataframe(out["granger_table"], use_container_width=True)

        st.divider()

        # 2) IRF
        st.subheader("2) IRF (Impulse → Target)")
        irf = out["irf"]
        imp = params.get("impulse")
        tgt = params.get("target")

        if imp and tgt:
            fig = irf.plot(impulse=imp, response=tgt)
            fig.set_size_inches(9, 4)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("IRF를 위해 impulse/target 설정이 필요합니다.")

        if params.get("show_full_grid", False):
            st.caption("전체 그리드는 변수가 많으면 겹쳐 보일 수 있어요.")
            fig2 = irf.plot()
            fig2.set_size_inches(12, 10)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True)

        st.divider()

        # 3) FEVD
        st.subheader("3) FEVD 분산분해 (target 기준)")
        st.caption("각 horizon에서 target 변동을 ‘어떤 shock(변수)이 얼마나 설명하는지(%)’")
        st.dataframe(out["fevd_table_target"], use_container_width=True)


if __name__ == "__main__":
    main()
