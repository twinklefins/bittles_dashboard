import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

DATA_PATH = Path("data/df_var_1209.csv")


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV into a DataFrame with a datetime index."""
    if not path.exists():
        st.warning(
            "데이터 파일을 찾을 수 없습니다. 'data/df_var_1209.csv' 경로를 확인해주세요. "
            "샘플 데이터로 화면을 구성합니다."
        )
        sample_index = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=60, freq="D")
        return pd.DataFrame(
            {
                "oi_close_diff": pd.Series(range(60)).mul(1e8).tolist(),
                "funding_close": [0.00005 + (i % 10) * 0.00002 for i in range(60)],
                "liq_total_usd_diff": [2e7 + (i % 12) * 1e7 for i in range(60)],
                "taker_buy_ratio": [0.5 + ((i % 14) - 7) * 0.01 for i in range(60)],
                "global_m2_yoy_diff": [0.03 if i % 7 else 0 for i in range(60)],
            },
            index=sample_index,
        )

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV에 'time' 컬럼이 없습니다.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df


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


def main() -> None:
    st.set_page_config(page_title="시장 위험도 대시보드", page_icon="📊", layout="wide")
    st.title("📊 시장 위험도 대시보드")
    st.caption("선택한 날짜 기준으로 주요 지표를 신호등(🟢🟡🔴) 형태로 확인하고, ‘패닉셀 방지’ 메시지를 제공합니다.")

    df = load_data(DATA_PATH)
    if df.empty:
        st.error("표시할 데이터가 없습니다.")
        return

    tab1, tab2 = st.tabs(["🚦 Risk Signal", "🧩 VAR Insight(준비중)"])

    with tab1:
        # ---- Sidebar: 날짜 선택(YYYY-MM-DD로 깔끔하게) ----
        st.sidebar.header("설정")
        unique_dates = sorted(pd.unique(df.index.date))

        selected_date = st.sidebar.selectbox(
            "기준 날짜 선택",
            options=unique_dates,
            index=len(unique_dates) - 1,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
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

        cols = st.columns(len(indicators))
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

            # ---- 전일 대비(DoD) 계산 ----
            delta_txt = "전일: N/A"
            if prev_row is not None and real_col in prev_row.index:
                try:
                    prev_val = float(prev_row[real_col])

                    if k == "m2" and (prev_val == 0 or value == 0):
                        delta_txt = "전일: N/A"
                    else:
                        if k in ["funding", "taker", "m2"]:
                            delta_val = value - prev_val
                            delta_txt = f"전일 대비 {delta_val:+.4f}"
                        else:
                            delta_val = value - prev_val
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
            st.error("핵심 지표 컬럼을 하나도 찾지 못했습니다. '컬럼 목록 보기'에서 실제 컬럼명을 확인해 매핑 후보를 추가해주세요.")
            return

        overall_text = overall_risk_text(total_score, used)
        st.success(overall_text)

        st.subheader("오늘의 원인 요약(자동)")
        st.info(build_cause_summary(signal_map))

        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df.tail(50))

    with tab2:
        st.subheader("🧩 VAR Insight (준비중)")
        st.write(
            "여기는 다음 단계에서 VAR 결과(IRF/FEVD) 시각화를 붙일 자리입니다.\n\n"
            "- IRF: 특정 shock(예: 청산/펀딩/OI)이 수익률(ret_log_1d)에 미치는 동학\n"
            "- FEVD: 변동성 분해로 ‘무슨 요인이 설명력이 큰지’\n\n"
            "원하면, 지금 df_var_1209 기준으로 IRF 1개 그래프부터 바로 붙여줄게요."
        )


if __name__ == "__main__":
    main()
