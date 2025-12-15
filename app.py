import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "df_var_1209.csv"


# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV into a DataFrame with a datetime index."""
    if not path.exists():
        st.warning(
            "데이터 파일을 찾을 수 없습니다. 'data/df_var_1209.csv' 경로를 확인해주세요. "
            "샘플 데이터로 화면을 구성합니다."
        )
        sample_index = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=180, freq="D")
        return pd.DataFrame(
            {
                "ret_log_1d": [0.0002 * ((i % 11) - 5) for i in range(180)],
                "oi_close_diff": pd.Series(range(180)).mul(5e7).tolist(),
                "funding_close": [0.00003 + (i % 10) * 0.00002 for i in range(180)],
                "liq_total_usd_diff": [2e7 + (i % 12) * 1e7 for i in range(180)],
                "taker_buy_ratio": [0.5 + ((i % 14) - 7) * 0.01 for i in range(180)],
                "global_m2_yoy_diff": [0.02 if i % 7 else 0 for i in range(180)],
            },
            index=sample_index,
        )

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df


# -----------------------------
# Risk Signal utilities
# -----------------------------
def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    """Return an emoji signal and numeric score based on percentile thresholds.
    Scores: green=0, yellow=1, red=2. Neutral/unknown=⚪️(score 1)
    """
    if series.empty or (isinstance(value, float) and math.isnan(value)):
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
    """Panic-sell prevention tone."""
    average_score = total_score / max(count, 1)
    if average_score < 0.5:
        return "🟢 현재는 구조적 과열 신호가 약합니다. 급락 시에도 패닉셀보다 ‘원인(청산/펀딩/유동성)’ 확인이 우선입니다."
    if average_score < 1.2:
        return "🟡 단기 변동성이 커질 수 있는 구간입니다. 패닉셀보다는 ‘청산/펀딩/쏠림’ 요인이 있는지 먼저 점검하세요."
    if average_score < 1.8:
        return "🟠 레버리지·쏠림 신호가 관측됩니다. 과거에는 변동성 확대가 잦았던 구간이니 포지션 크기/리스크 관리를 권장합니다."
    return "🔴 단기 충격(청산/레버리지) 가능성이 높습니다. 무리한 레버리지는 피하고, 변동성 확대를 전제로 대응하세요."


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def build_cause_summary(signals: Dict[str, str]) -> str:
    """Build short explanation based on signal emojis."""
    red = [k for k, s in signals.items() if s == "🔴"]
    yellow = [k for k, s in signals.items() if s == "🟡"]
    neutral = [k for k, s in signals.items() if s == "⚪️"]

    desc = {
        "oi": "레버리지(미결제약정)",
        "funding": "펀딩(포지션 쏠림/과열)",
        "liq": "청산(강제 매도/매수 압력)",
        "taker": "테이커 쏠림(공포/탐욕)",
        "m2": "유동성(M2)",
    }

    lines = []
    if red:
        lines.append("🔴 **강한 단기 충격 신호**가 있습니다: " + ", ".join(desc[x] for x in red))
        lines.append("→ 급변 구간에서는 ‘원인 확인(청산/펀딩/레버리지)’이 우선이고, 감정적 매매는 손실 확률을 키웁니다.")
    elif yellow:
        lines.append("🟡 **변동성 확대 신호**가 있습니다: " + ", ".join(desc[x] for x in yellow))
        lines.append("→ 당일 급락/급등은 ‘쏠림’ 때문에 과장될 수 있어, 추세가 아니라 구조인지 확인하세요.")
    else:
        lines.append("🟢 **과열 신호가 뚜렷하지 않습니다.** 단기 노이즈 가능성이 큽니다.")

    if neutral:
        lines.append("⚪️ 데이터가 부족/결측 가능: " + ", ".join(desc[x] for x in neutral))

    return "\n\n".join(lines)


# -----------------------------
# Pipeline tab utilities
# -----------------------------
def get_analysis_pipeline() -> Dict[str, List[str]]:
    """1210_VAR_시범.py의 흐름을 '카테고리'로 묶어 지도 형태로 보여주기."""
    return {
        "데이터 준비": [
            "CSV 로드 (time → datetime index)",
            "결측/이상치 처리, 정렬",
            "분석 대상 변수 선택/매핑 (Risk / VAR 공통)",
        ],
        "전처리 & 정상성": [
            "수익률/차분 등 변환(필요 시)",
            "정상성 확인(ADF Test) 및 변환 결정",
        ],
        "VAR 모델링": [
            "VAR 입력 데이터 구성(선택 변수 집합)",
            "Lag 선택(AIC/BIC 등) 및 VAR 적합",
            "Granger 인과성 테스트",
        ],
        "IRF / FEVD": [
            "IRF(충격 반응) 시각화",
            "FEVD(분산분해) 표로 기여도 출력",
            "요약 인사이트 생성(어떤 요인이 컸는지)",
        ],
        "대시보드 출력": [
            "Risk Signal(🟢🟡🔴) + 패닉셀 방지 메시지",
            "VAR Insight(Granger/IRF/FEVD) 결과 제공",
            "공유(Cloud 링크/README/데이터 공유 정책)",
        ],
    }


def render_pipeline_visual(pipeline: Dict[str, List[str]]) -> None:
    """Graphviz 있으면 흐름도, 없으면 텍스트로 표시."""
    st.subheader("🧭 분석 파이프라인 전체 보기")
    st.caption("Risk Signal / VAR Insight가 어떤 분석 단계를 거쳐 만들어지는지 설명하는 지도입니다.")

    try:
        import graphviz  # type: ignore

        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")

        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightgrey")
        categories = list(pipeline.keys())
        for c in categories:
            dot.node(c, c)

        dot.attr("node", shape="box", style="rounded,filled", fillcolor="white")
        for c, steps in pipeline.items():
            prev = c
            for i, s in enumerate(steps, start=1):
                sid = f"{c}_{i}"
                dot.node(sid, f"{i}. {s}")
                dot.edge(prev, sid)
                prev = sid

        st.graphviz_chart(dot, use_container_width=True)

    except Exception:
        st.info("ℹ️ Graphviz가 없어 텍스트로 표시합니다. (원하면 requirements.txt에 graphviz 추가)")
        for c, steps in pipeline.items():
            st.markdown(f"### {c}")
            for i, s in enumerate(steps, start=1):
                st.markdown(f"- {i}. {s}")

    st.divider()
    st.subheader("카테고리별 단계(체크리스트)")
    st.caption("팀 내부에서 ‘어디까지 구현/검증됐는지’ 표시용으로 사용할 수 있습니다.")
    for c, steps in pipeline.items():
        with st.expander(f"📌 {c}", expanded=(c == "IRF / FEVD")):
            for s in steps:
                st.checkbox(s, value=False, key=f"pipeline_{c}_{s}")


# -----------------------------
# VAR / Granger / IRF / FEVD
# -----------------------------
@dataclass
class VarOutputs:
    granger_matrix: pd.DataFrame
    granger_to_target: pd.DataFrame
    fevd_table: pd.DataFrame
    irf_fig: Optional["object"]  # matplotlib Figure (typing 회피)


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0).replace(0, pd.NA))


@st.cache_data(show_spinner=False)
def run_var_bundle(
    df: pd.DataFrame,
    cols: List[str],
    target: str,
    maxlags: int,
    horizon: int,
    standardize: bool,
) -> VarOutputs:
    """Run VAR and return Granger matrix/table, IRF figure, FEVD table."""
    # lazy import (Cloud에서 requirements 없을 때 에러 메시지 깔끔하게)
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests

    import matplotlib.pyplot as plt  # noqa: F401

    x = df[cols].copy().dropna()
    if len(x) < (maxlags + 25):
        raise ValueError(f"VAR 실행을 위한 데이터 길이가 부족합니다. dropna 후 {len(x)}행 (lag={maxlags})")

    if standardize:
        x = _zscore(x).dropna()

    model = VAR(x)
    res = model.fit(maxlags=maxlags)

    # ---- Granger Matrix (pairwise p-values) ----
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for caused in cols:
        for causing in cols:
            if caused == causing:
                pvals.loc[caused, causing] = float("nan")
                continue
            try:
                test = res.test_causality(caused=caused, causing=[causing], kind="f")
                pvals.loc[caused, causing] = float(test.pvalue)
            except Exception:
                pvals.loc[caused, causing] = float("nan")
    granger_matrix = pvals

    # ---- Granger to target (stable, based on grangercausalitytests) ----
    rows = []
    causes = [c for c in cols if c != target]
    for c in causes:
        try:
            g = grangercausalitytests(x[[target, c]], maxlag=maxlags, verbose=False)
            # 선택 lag의 p-value
            p = float(g[maxlags][0]["ssr_ftest"][1])
        except Exception:
            p = float("nan")
        rows.append({"cause": c, "target": target, "lag": maxlags, "p_value": p})
    granger_to_target = pd.DataFrame(rows).sort_values("p_value")

    # ---- IRF ----
    irf_fig = None
    try:
        irf = res.irf(horizon)
        fig = irf.plot(orth=False)
        if fig is not None:
            fig.suptitle("IRF (Impulse Response Functions)", fontsize=12)
            irf_fig = fig
    except Exception:
        irf_fig = None

    # ---- FEVD (last horizon summary) ----
    fevd = res.fevd(horizon)
    # horizon 마지막 시점의 분산분해 (k x k)
    decomp = fevd.decomp[-1]
    fevd_table = pd.DataFrame(decomp, index=cols, columns=cols)
    fevd_table.index.name = "Explained (target)"
    fevd_table.columns.name = "Explainer (shock)"

    return VarOutputs(
        granger_matrix=granger_matrix,
        granger_to_target=granger_to_target,
        fevd_table=fevd_table,
        irf_fig=irf_fig,
    )


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="시장 위험도 대시보드", page_icon="📊", layout="wide")
    st.title("📊 시장 위험도 대시보드")
    st.caption("선택한 날짜 기준으로 주요 지표를 신호등(🟢🟡🔴)으로 확인하고, VAR 기반(Granger/IRF/FEVD) 인사이트를 제공합니다.")

    df = load_data(DATA_PATH)
    if df.empty:
        st.error("표시할 데이터가 없습니다.")
        return

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 분석 파이프라인"])

    # -------------------------
    # Tab1: Risk Signal
    # -------------------------
    with tab1:
        st.sidebar.header("설정")
        unique_dates = sorted(pd.unique(df.index.date))

        selected_date = st.sidebar.selectbox(
            "기준 날짜 선택",
            options=unique_dates,
            index=len(unique_dates) - 1,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )

        selected_df = df[df.index.date == selected_date]
        latest_row = selected_df.iloc[-1] if not selected_df.empty else df.iloc[-1]

        date_idx = unique_dates.index(selected_date)
        prev_date = unique_dates[date_idx - 1] if date_idx > 0 else None
        prev_row = None
        if prev_date is not None:
            prev_df = df[df.index.date == prev_date]
            prev_row = prev_df.iloc[-1] if not prev_df.empty else None

        st.subheader(f"기준 데이터 날짜: {latest_row.name:%Y-%m-%d %H:%M:%S}")

        with st.expander("컬럼 목록 보기(문제 해결용)"):
            st.write(list(df.columns))

        colmap: Dict[str, Optional[str]] = {
            "oi": find_column(df, ["oi_close_diff", "oi_diff", "open_interest_diff", "oi", "OI"]),
            "funding": find_column(df, ["funding_close", "funding", "funding_rate", "Funding"]),
            "liq": find_column(df, ["liq_total_usd_diff", "liquidation_usd", "liq_usd", "Liquidation"]),
            "taker": find_column(df, ["taker_buy_ratio", "taker_ratio", "Taker Buy Ratio"]),
            "m2": find_column(df, ["global_m2_yoy_diff", "m2_yoy_diff", "global_m2_yoy", "M2"]),
        }

        indicators = {
            "oi": {"description": "OI 변화량", "higher_is_risky": True},
            "funding": {"description": "펀딩비", "higher_is_risky": True},
            "liq": {"description": "청산(USD)", "higher_is_risky": True},
            "taker": {"description": "테이커 매수비중(쏠림)", "higher_is_risky": True},
            "m2": {"description": "글로벌 M2(YoY diff)", "higher_is_risky": False},
        }

        cols_ui = st.columns(len(indicators))
        total_score = 0
        used = 0
        signal_map: Dict[str, str] = {}

        for ui_col, (k, meta) in zip(cols_ui, indicators.items()):
            real_col = colmap.get(k)
            if not real_col:
                ui_col.warning(f"{meta['description']} 컬럼을 찾지 못했습니다.")
                signal_map[k] = "⚪️"
                continue

            value = float(latest_row[real_col])

            if k == "taker":
                series = (df[real_col] - 0.5).abs()
                v = abs(value - 0.5)
                signal, score = percentile_signal(series, v, higher_is_risky=True)
                display_value = f"{value:.3f} (쏠림:{v:.3f})"

            elif k == "m2":
                series = df[real_col].replace(0, pd.NA).dropna()
                if value == 0:
                    signal, score = "⚪️", 1
                    display_value = f"{value:,.4g} (결측가능)"
                else:
                    signal, score = percentile_signal(series, value, higher_is_risky=meta["higher_is_risky"])
                    display_value = f"{value:,.4g}"

            else:
                series = df[real_col]
                signal, score = percentile_signal(series, value, higher_is_risky=meta["higher_is_risky"])
                display_value = f"{value:,.4g}"

            delta_txt = "전일: N/A"
            if prev_row is not None and real_col in prev_row.index:
                try:
                    prev_val = float(prev_row[real_col])
                    if k == "m2" and (prev_val == 0 or value == 0):
                        delta_txt = "전일: N/A"
                    else:
                        delta_val = value - prev_val
                        if k in ["funding", "taker", "m2"]:
                            delta_txt = f"전일 대비 {delta_val:+.4f}"
                        else:
                            delta_txt = f"전일 대비 {delta_val:+,.0f}"
                except Exception:
                    delta_txt = "전일: N/A"

            signal_map[k] = signal
            total_score += score
            used += 1

            ui_col.metric(
                label=f"{meta['description']} ({real_col})",
                value=display_value,
                delta=delta_txt,
            )
            ui_col.caption(f"신호: {signal}")

        st.divider()
        st.subheader("신호등 요약")
        st.write("🟢 낮음 | 🟡 중간 | 🔴 높음 | ⚪️ 데이터 부족/결측 가능")

        if used == 0:
            st.error("핵심 지표 컬럼을 하나도 찾지 못했습니다. 컬럼명을 확인해 매핑 후보를 추가해주세요.")
            return

        st.success(overall_risk_text(total_score, used))

        st.subheader("오늘의 원인 요약(자동)")
        st.info(build_cause_summary(signal_map))

        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df.tail(50))

    # -------------------------
    # Tab2: VAR Insight (Granger / IRF / FEVD)
    # -------------------------
    with tab2:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger 인과 테스트(표) / IRF(그래프) / FEVD 분산분해(표)")

        # 추천 변수셋(있으면 자동 포함)
        recommended = [
            "ret_log_1d",
            "oi_close_diff",
            "funding_close",
            "liq_total_usd_diff",
            "taker_buy_ratio",
            "sth_sopr",
            "lth_sopr",
            "sth_realized_price_usd_diff",
            "lth_realized_price_usd_diff",
            "rhodl_ratio",
            "global_m2_yoy_diff",
            "sp500_ret",
            "nasdaq_ret",
            "etf_aum_diff",
            "etf_flow_shock_pos",
            "etf_flow_shock_neg",
        ]
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        available = [c for c in recommended if c in numeric_cols]
        if "ret_log_1d" not in numeric_cols:
            st.error("CSV에 ret_log_1d 컬럼이 없습니다. VAR/IRF/FEVD 타겟을 바꾸거나 컬럼명을 확인해주세요.")
            st.stop()

        # 사이드바 설정
        st.sidebar.header("VAR 설정")
        default_cols = available if len(available) >= 5 else (["ret_log_1d"] + numeric_cols[:6])
        selected_cols = st.sidebar.multiselect(
            "VAR 변수 선택(2개 이상)",
            options=numeric_cols,
            default=list(dict.fromkeys([c for c in default_cols if c in numeric_cols]))[:10],
        )
        target = st.sidebar.selectbox("타겟(반응) 변수", options=selected_cols if selected_cols else ["ret_log_1d"], index=0)
        maxlags = st.sidebar.slider("VAR lag", min_value=1, max_value=14, value=1)
        horizon = st.sidebar.slider("IRF/FEVD horizon", min_value=5, max_value=30, value=10)
        standardize = st.sidebar.checkbox("표준화(z-score) 후 VAR 적합", value=True)

        if len(selected_cols) < 2:
            st.warning("VAR 실행을 위해 변수 2개 이상을 선택해주세요.")
            st.stop()
        if target not in selected_cols:
            st.warning("타겟 변수는 선택된 VAR 변수 목록 안에 있어야 합니다.")
            st.stop()

        run_btn = st.button("VAR 실행(Granger / IRF / FEVD)", type="primary")

        if run_btn:
            try:
                out = run_var_bundle(
                    df=df,
                    cols=selected_cols,
                    target=target,
                    maxlags=maxlags,
                    horizon=horizon,
                    standardize=standardize,
                )

                # --- Granger (target 중심) ---
                st.subheader("1) Granger 인과 테스트 (타겟 기준 p-value)")
                st.caption("p-value가 낮을수록 ‘cause → target’ 인과성 신호로 해석합니다. (보수적으로 해석 권장)")
                st.dataframe(out.granger_to_target.style.format({"p_value": "{:.4f}"}), use_container_width=True)

                with st.expander("Granger p-value 전체 매트릭스 보기(advanced)"):
                    st.dataframe(out.granger_matrix.style.format("{:.4f}"), use_container_width=True)

                st.divider()

                # --- IRF ---
                st.subheader("2) IRF (Impulse Response Functions)")
                st.caption("기본은 전체 IRF. 발표용으로는 ‘impulse 선택 → target 반응’ 1개만 보여줘도 좋아요.")
                if out.irf_fig is None:
                    st.warning("IRF 그래프 생성에 실패했습니다. (결측/lag/horizon/변수 수를 조정해보세요.)")
                else:
                    st.pyplot(out.irf_fig, clear_figure=True)

                st.divider()

                # --- FEVD ---
                st.subheader("3) FEVD 분산분해 (기여도)")
                st.caption("선택한 horizon의 ‘마지막 스텝’ 기준 기여도입니다. (행=설명되는 변수, 열=충격 제공 변수)")
                st.dataframe(out.fevd_table.style.format("{:.3f}"), use_container_width=True)

                st.info(
                    "해석 팁: target 행에서 값이 큰 열(변수)이 ‘target 변동을 많이 설명하는 shock’로 해석될 수 있습니다."
                )

            except ModuleNotFoundError as e:
                st.error(f"필요 라이브러리가 없습니다: {e}")
                st.info("requirements.txt에 statsmodels, matplotlib를 추가하고 재배포하세요.")
            except Exception as e:
                st.error(f"VAR/Granger/IRF/FEVD 실행 중 오류: {e}")
                st.info("해결 팁: (1) 변수 수를 3~6개로 줄이기 (2) lag=1부터 시작 (3) horizon 10~15 (4) 표준화 on/off 변경")

    # -------------------------
    # Tab3: Pipeline
    # -------------------------
    with tab3:
        pipeline = get_analysis_pipeline()
        render_pipeline_visual(pipeline)
        st.divider()
        st.markdown(
            """
**이 탭의 목적**  
- 팀원/멘토가 “Risk Signal / VAR Insight가 어떤 분석 과정을 통해 나오는지”를 즉시 이해하도록 돕습니다.  
- 1210_VAR_시범.py의 연구/실험 코드는 유지하되, 대시보드에서는 “과정 지도 + 결과 출력” 형태로 설명합니다.
"""
        )


if __name__ == "__main__":
    main()

