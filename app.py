import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# VAR / IRF / FEVD
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt


# ======================
# Paths
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "df_var_1209.csv"


# ======================
# Data loader
# ======================
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV into a DataFrame with a datetime index."""
    if not path.exists():
        st.warning(
            "데이터 파일을 찾을 수 없습니다. 'data/df_var_1209.csv' 경로를 확인해주세요. "
            "샘플 데이터로 화면을 구성합니다."
        )
        sample_index = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=200, freq="D")
        return pd.DataFrame(
            {
                "ret_log_1d": np.random.normal(0, 0.02, size=len(sample_index)),
                "oi_close_diff": pd.Series(range(len(sample_index))).mul(1e8).tolist(),
                "funding_close": [0.00005 + (i % 10) * 0.00002 for i in range(len(sample_index))],
                "liq_total_usd_diff": [2e7 + (i % 12) * 1e7 for i in range(len(sample_index))],
                "taker_buy_ratio": [0.5 + ((i % 14) - 7) * 0.01 for i in range(len(sample_index))],
                "global_m2_yoy_diff": [0.03 if i % 7 else 0 for i in range(len(sample_index))],
            },
            index=sample_index,
        )

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    return df


# ======================
# Risk signal helpers
# ======================
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


# ======================
# VAR helpers
# ======================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    std = df.std().replace(0, np.nan)
    return (df - mean) / std


def run_var_bundle(
    df: pd.DataFrame,
    selected_cols: List[str],
    target: str,
    lag: int,
    horizon: int,
    standardize: bool,
) -> Dict[str, object]:
    """
    Fit VAR and produce:
    - granger_table: DataFrame (x -> target)
    - irf_result: IRF object
    - fevd_table_target: DataFrame (steps x impulses) for target only
    - var_results: VARResults
    """
    if len(selected_cols) < 2:
        raise ValueError("VAR 변수는 2개 이상 선택해야 합니다.")

    if target not in selected_cols:
        raise ValueError("타겟(반응) 변수는 선택된 VAR 변수 안에 있어야 합니다.")

    data = df[selected_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.shape[0] < max(50, lag * 10):
        raise ValueError(f"데이터가 너무 적습니다. (현재 {data.shape[0]} rows). lag={lag}면 최소 50~100행 권장")

    if standardize:
        data = zscore_df(data).dropna()

    model = VAR(data)
    results = model.fit(lag)

    # ---- Granger: x -> target (p-value)
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
                    "test": "F",
                    "stat": float(test.test_statistic),
                    "pvalue": float(test.pvalue),
                    "df_denom": getattr(test, "df_denom", None),
                    "df_num": getattr(test, "df_num", None),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "causing(x)": x,
                    "caused(target)": target,
                    "test": "F",
                    "stat": np.nan,
                    "pvalue": np.nan,
                    "df_denom": None,
                    "df_num": None,
                    "error": str(e),
                }
            )
    granger_df = pd.DataFrame(rows).sort_values("pvalue", na_position="last").reset_index(drop=True)

    # ---- IRF
    irf = results.irf(horizon)

    # ---- FEVD to target (steps x impulses)
    fevd = results.fevd(horizon)
    # fevd.decomp shape: (horizon+1, neq, neq) or (horizon, neq, neq) depending
    decomp = np.array(fevd.decomp)

    varnames = list(results.names)
    if target not in varnames:
        raise ValueError("VAR 결과에서 target 변수를 찾지 못했습니다.")

    target_idx = varnames.index(target)

    # steps 축 처리(0 포함할 수 있어 0 제외하고 1..horizon로 표기)
    # decomp[t, response, impulse]
    # t=0이 포함되면 제거
    steps = list(range(decomp.shape[0]))
    if 0 in steps:
        # drop step 0 for nicer display
        decomp_use = decomp[1:, :, :]
        step_labels = list(range(1, decomp.shape[0]))
    else:
        decomp_use = decomp
        step_labels = list(range(1, decomp.shape[0] + 1))

    fevd_target = decomp_use[:, target_idx, :]  # (steps, impulse_vars)
    fevd_table = pd.DataFrame(fevd_target, columns=varnames, index=step_labels)
    fevd_table = (fevd_table * 100.0).round(2)
    fevd_table.index.name = "horizon(step)"

    return {
        "granger_table": granger_df,
        "irf": irf,
        "fevd_table_target": fevd_table,
        "var_results": results,
        "var_data_rows": data.shape[0],
    }


def main() -> None:
    st.set_page_config(page_title="시장 위험도 대시보드", page_icon="📊", layout="wide")
    # --- (추가) UI 다듬기용 CSS ---
    st.markdown(
        """
        <style>
        /* 전체 여백/타이포 정리 */
        .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
        h1 { margin-bottom: 0.25rem; }
        /* metric label 줄바꿈 허용 + 폰트 살짝 줄이기 */
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.9rem; }
        /* metric value 너무 커서 답답한 느낌 완화 */
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        /* sidebar 간격 */
        section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📊 시장 위험도 대시보드")
    st.caption("선택한 날짜 기준으로 주요 지표를 신호등(🟢🟡🔴) 형태로 확인하고, VAR 기반(Granger/IRF/FEVD) 인사이트를 제공합니다.")

    df = load_data(DATA_PATH)
    if df.empty:
        st.error("표시할 데이터가 없습니다.")
        return

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight", "🧭 분석 파이프라인"])

    # ======================
    # Tab 1: Risk Signal
    # ======================
    with tab1:
        # ---- Sidebar: 날짜 선택(YYYY-MM-DD로 깔끔하게) ----
        st.sidebar.header("설정")
        unique_dates = sorted(pd.unique(df.index.date))

        selected_date = st.sidebar.selectbox(
            "기준 날짜 선택",
            options=unique_dates,
            index=len(unique_dates) - 1,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="risk_date",
        )

        selected_mask = (df.index.date == selected_date)
        selected_df = df[selected_mask] if selected_mask.any() else df
        latest_row = selected_df.iloc[-1]

        # ---- 전일 row 구하기 ----
        date_idx = unique_dates.index(selected_date)
        prev_date = unique_dates[date_idx - 1] if date_idx > 0 else None

        if prev_date is not None:
            prev_mask = (df.index.date == prev_date)
            prev_df = df[prev_mask] if prev_mask.any() else df
            prev_row = prev_df.iloc[-1]
        else:
            prev_row = None

        st.subheader(f"기준 데이터 날짜: {latest_row.name:%Y-%m-%d %H:%M:%S}")

        with st.expander("컬럼 목록 보기(문제 해결용)"):
            st.write(list(df.columns))

        # ---- 컬럼 매핑(자동 탐지) ----
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

        cols = st.columns(len(indicators), gap="large")
        total_score = 0
        used = 0
        signal_map: Dict[str, str] = {}

        for ui_col, (k, meta) in zip(cols, indicators.items()):
            real_col = colmap.get(k)
            if not real_col:
                ui_col.warning(f"{meta['description']} 컬럼을 찾지 못했습니다.")
                signal_map[k] = "⚪️"
                continue

            value = float(latest_row[real_col])

            # ---- 지표별 신호 계산 ----
            extra_line = ""  # metric 아래에 붙일 보조 정보(쏠림 등)
            if k == "taker":
                series = (df[real_col] - 0.5).abs()
                v = abs(value - 0.5)
                signal, score = percentile_signal(series, v, higher_is_risky=True)

                # ✅ (핵심) metric 값은 짧게: 0.509 이런 식으로만
                display_value = f"{value:.3f}"
                extra_line = f"쏠림 |{v:.3f}| (0.5에서 멀수록 쏠림)"
            elif k == "m2":
                series = df[real_col].replace(0, pd.NA).dropna()
                if value == 0:
                    signal, score = "⚪️", 1
                    display_value = "N/A"
                    extra_line = "결측 가능(0값 처리)"
                else:
                    signal, score = percentile_signal(series, value, higher_is_risky=meta["higher_is_risky"])
                    display_value = f"{value:,.4g}"
            else:
                series = df[real_col]
                signal, score = percentile_signal(series, value, higher_is_risky=meta["higher_is_risky"])
                display_value = f"{value:,.4g}"

            # ---- 전일 대비(DoD) 계산 ----
            delta_txt = None
            if prev_row is not None and real_col in prev_row.index:
                try:
                    prev_val = float(prev_row[real_col])
                    if k == "m2" and (prev_val == 0 or value == 0):
                        delta_txt = None
                    else:
                        delta_val = value - prev_val
                        if k in ["funding", "taker", "m2"]:
                            delta_txt = f"{delta_val:+.4f}"
                        else:
                            delta_txt = f"{delta_val:+,.0f}"
                except Exception:
                    delta_txt = None

            signal_map[k] = signal
            total_score += score
            used += 1

            # ✅ 라벨은 짧게(잘림 방지) + 실제 컬럼명은 caption으로
            ui_col.metric(
                label=f"{meta['description']}",
                value=display_value,
                delta=delta_txt,
            )
            ui_col.caption(f"컬럼: `{real_col}`  ·  신호: {signal}")
            if extra_line:
                ui_col.caption(extra_line)

        st.divider()

        st.subheader("신호등 요약")
        st.write("🟢 낮음 | 🟡 중간 | 🔴 높음 | ⚪️ 데이터 부족/결측 가능")

        if used == 0:
            st.error("핵심 지표 컬럼을 하나도 찾지 못했습니다. '컬럼 목록 보기'에서 실제 컬럼명을 확인해 매핑 후보를 추가해주세요.")
            return

        overall_text = overall_risk_text(total_score, used)
        st.success(overall_text)

        st.subheader("오늘의 원인 요약(자동)")
        st.info(build_cause_summary(signal_map))

        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df.tail(50))

    # ======================
    # Tab 2: VAR Insight
    # ======================
    with tab2:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger 인과테스트(표) / IRF(그래프) / FEVD 분산분해(표)를 한 번에 확인합니다.")

        # ---- Sidebar: VAR controls ----
        st.sidebar.header("VAR 설정")

        # 추천 후보 컬럼들(있는 것만 자동 포함)
        default_candidates = [c for c in ["ret_log_1d", "oi_close_diff", "funding_close", "liq_total_usd_diff", "taker_buy_ratio", "global_m2_yoy_diff"] if c in df.columns]
        all_numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        selected_cols = st.sidebar.multiselect(
            "VAR 변수 선택(2개 이상)",
            options=all_numeric_cols,
            default=default_candidates[:4] if len(default_candidates) >= 2 else all_numeric_cols[:3],
        )

        if selected_cols:
            target = st.sidebar.selectbox("타겟(반응) 변수", options=selected_cols, index=0)
            impulse_options = [c for c in selected_cols if c != target]
            impulse_var = st.sidebar.selectbox("IRF Impulse(충격) 변수", options=impulse_options, index=0 if impulse_options else 0)
        else:
            target = None
            impulse_var = None

        lag = st.sidebar.slider("VAR lag", min_value=1, max_value=10, value=1)
        horizon = st.sidebar.slider("IRF/FEVD horizon", min_value=5, max_value=30, value=10)
        standardize = st.sidebar.checkbox("표준화(z-score) 후 VAR 적합", value=True)
        show_full_grid = st.sidebar.checkbox("IRF 전체 그리드(변수×변수)도 보기", value=False)

        run_btn = st.button("VAR 실행(Granger / IRF / FEVD)", type="primary")

        if "var_out" not in st.session_state:
            st.session_state["var_out"] = None
        if "var_params" not in st.session_state:
            st.session_state["var_params"] = None

        if run_btn:
            try:
                with st.spinner("VAR 적합 중… (조금 걸릴 수 있어요)"):
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
                    "selected_cols": selected_cols,
                    "target": target,
                    "impulse_var": impulse_var,
                    "lag": lag,
                    "horizon": horizon,
                    "standardize": standardize,
                    "show_full_grid": show_full_grid,
                }
                st.success(f"완료! (학습 데이터 rows: {out['var_data_rows']})")
            except Exception as e:
                st.error(f"VAR 실행 실패: {e}")
                st.session_state["var_out"] = None
                st.session_state["var_params"] = None

        # ---- Render results if exists
        out = st.session_state.get("var_out")
        params = st.session_state.get("var_params")

        if out is None:
            st.info("왼쪽에서 변수 선택 후 **VAR 실행** 버튼을 눌러주세요.")
        else:
            # 1) Granger table
            st.subheader("1) Granger 인과테스트 (x → target)")
            st.caption("p-value가 작을수록 ‘x가 target을 그랜저 인과한다’는 근거가 강합니다(통상 0.05 기준).")
            st.dataframe(out["granger_table"], use_container_width=True)

            st.divider()

            # 2) IRF (nice: impulse 1 -> target 1)
            st.subheader("2) IRF (Impulse Response Functions)")
            st.caption("데모용으로는 ‘impulse 1개 → target 1개’만 크게 보여주는 게 가장 읽기 좋아요.")

            irf = out["irf"]
            imp = params.get("impulse_var")
            tgt = params.get("target")
            h = params.get("horizon", 10)

            if imp is None or tgt is None:
                st.warning("IRF를 위해 impulse/target 설정이 필요합니다.")
            else:
                # ✅ 1개 impulse -> 1개 target
                fig = irf.plot(impulse=imp, response=tgt)
                fig.set_size_inches(9, 4)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                # (옵션) full grid
                if params.get("show_full_grid", False):
                    with st.expander("전체 IRF 그리드 보기(변수×변수)"):
                        fig2 = irf.plot()
                        fig2.set_size_inches(12, 10)
                        fig2.tight_layout()
                        st.pyplot(fig2, clear_figure=True)

            st.divider()

            # 3) FEVD table
            st.subheader("3) FEVD 분산분해 (target 기준)")
            st.caption("각 horizon에서 target 변동을 ‘어떤 shock(변수)이 얼마나 설명하는지(%)’를 보여줍니다.")
            st.dataframe(out["fevd_table_target"], use_container_width=True)

    # ======================
    # Tab 3: Pipeline visualization
    # ======================
    with tab3:
        st.subheader("🧭 분석 파이프라인 (전체 흐름 시각화)")
        st.caption("팀 요청대로, 대시보드가 ‘어떤 순서로 분석을 수행하는지’를 한 눈에 보여주는 뷰입니다.")

        st.markdown(
            """
### ✅ 전체 분석 단계

1. **데이터 로드**
   - `data/df_var_1209.csv` 로드 → `time` 기준 정렬

2. **Risk Signal (신호등)**
   - OI / Funding / Liquidation / Taker / M2 지표
   - 분위수(33%/66%) 기반 🟢🟡🔴 신호 생성
   - 전일 대비 변화 + 원인 요약(자동 메시지)

3. **VAR Insight (인사이트)**
   - (사용자 선택) 변수 2개 이상 선택
   - (선택) z-score 표준화
   - VAR(lag) 적합
   - **Granger**: `x → target` 인과테스트 결과 표
   - **IRF**: (impulse 1개 → target 1개) 반응 그래프
   - **FEVD**: target 분산 분해(%) 표

---

### 🧩 데모용 추천 사용법 (멘토/팀원 발표 기준)

- Target(반응): `ret_log_1d`
- Impulse(충격): `liq_total_usd_diff` 또는 `funding_close` 또는 `oi_close_diff`
- Lag: 1~2
- Horizon: 10

👉 이렇게 설정하면 ‘청산/펀딩/레버리지 충격이 수익률에 미치는 동학’을 **깔끔하게 1장 그래프로** 보여줄 수 있습니다.
"""
        )


if __name__ == "__main__":
    main()
