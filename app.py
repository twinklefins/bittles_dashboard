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

    # 숫자형으로 강제 변환(문자 섞여있을 때 대비)
    for c in df.columns:
        if c == "time":
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ======================
# Utility
# ======================
def zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    mu, sd = s.mean(), s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def percentile_signal(series: pd.Series, value: float, higher_is_risky: bool = True) -> Tuple[str, int]:
    """
    return (signal_emoji, score)
    score: 🟢0, 🟡1, 🔴2, ⚪️(missing)=1
    """
    if series is None or series.dropna().empty or pd.isna(value):
        return "⚪️", 1

    q1, q2 = series.quantile([0.33, 0.66])

    if higher_is_risky:
        if value >= q2:
            return "🔴", 2
        if value <= q1:
            return "🟢", 0
    else:
        if value <= q1:
            return "🔴", 2
        if value >= q2:
            return "🟢", 0

    return "🟡", 1


# ======================
# Risk → Market Mood Index (0~100)
# ======================
def compute_risk_signals(df: pd.DataFrame, row: pd.Series) -> Dict[str, Dict]:
    """
    returns dict with per-indicator: value, signal, score, colname, note
    """
    # 네 파일에 실제로 있는 컬럼 기준으로 “우선순위 후보”를 둠
    col_oi = find_col(df, ["oi_close", "oi_close_diff", "open_interest", "oi"])
    col_funding = find_col(df, ["funding_close", "funding_rate", "funding"])
    col_liq = find_col(df, ["liq_total_usd", "liq_total_usd_diff", "liquidation_usd", "liq_usd"])
    col_taker = find_col(df, ["taker_buy_ratio", "taker_ratio"])
    col_m2 = find_col(df, ["global_m2_yoy_diff", "global_m2_yoy", "m2_yoy_diff"])

    indicators = [
        ("oi", "OI", col_oi, True),
        ("funding", "Funding", col_funding, True),
        ("liq", "Liquidation(USD)", col_liq, True),
        ("taker", "Taker Bias", col_taker, True),        # 0.5에서 멀수록 쏠림(위험)
        ("m2", "Global M2", col_m2, False),              # 유동성은 낮을수록 위험(방어적으로)
    ]

    out = {}
    for key, label, col, higher_is_risky in indicators:
        if col is None or col not in df.columns:
            out[key] = {
                "label": label, "col": None, "value": np.nan,
                "signal": "⚪️", "score": 1, "note": "컬럼 없음"
            }
            continue

        v = row.get(col, np.nan)

        # taker는 0.5 기준 거리로 판단
        if key == "taker" and not pd.isna(v):
            dist = abs(float(v) - 0.5)
            series = (df[col] - 0.5).abs()
            sig, sc = percentile_signal(series, dist, higher_is_risky=True)
            out[key] = {
                "label": label, "col": col, "value": float(v),
                "signal": sig, "score": sc,
                "note": f"|x-0.5|={dist:.3f}"
            }
            continue

        # m2는 0이 결측표시일 수도 있어 방어 처리
        if key == "m2" and (pd.isna(v) or float(v) == 0.0):
            out[key] = {
                "label": label, "col": col, "value": float(v) if not pd.isna(v) else np.nan,
                "signal": "⚪️", "score": 1, "note": "0/NaN (결측 가능)"
            }
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        sig, sc = percentile_signal(series, float(v) if not pd.isna(v) else np.nan, higher_is_risky=higher_is_risky)

        out[key] = {
            "label": label, "col": col, "value": float(v) if not pd.isna(v) else np.nan,
            "signal": sig, "score": sc, "note": ""
        }

    return out


def compute_market_mood_index(df: pd.DataFrame, row: pd.Series, signals: Dict[str, Dict]) -> Tuple[float, str, str]:
    """
    MMI: 0~100 (낮을수록 Calm, 높을수록 Too Hot)
    - 기본은 Risk Signal 점수 평균(0~2)을 0~100으로 스케일
    - 데이터에 avg_sent / gtrend_btc_z14가 있으면 약간 가중치로 보정
    """
    # 1) risk score 기반
    scores = [v["score"] for v in signals.values() if v["signal"] != "⚪️"]
    base = float(np.mean(scores)) if scores else 1.0  # 0~2
    mmi = (base / 2.0) * 100.0

    # 2) sentiment / attention 있으면 보정(있을 때만)
    col_sent = find_col(df, ["avg_sent", "sentiment", "rd_avg_sent"])
    col_gt = find_col(df, ["gtrend_btc_z14", "gt_btc_z14", "gt_bitcoin", "gtrend_btc"])

    bonus = 0.0
    if col_sent and col_sent in df.columns:
        zsent = zscore(pd.to_numeric(df[col_sent], errors="coerce"))
        if row.name in zsent.index and not pd.isna(zsent.loc[row.name]):
            # 긍정이면 과열(탐욕) 방향, 부정이면 공포 방향으로 살짝 이동
            bonus += float(zsent.loc[row.name]) * 6.0

    if col_gt and col_gt in df.columns:
        zgt = zscore(pd.to_numeric(df[col_gt], errors="coerce"))
        if row.name in zgt.index and not pd.isna(zgt.loc[row.name]):
            # 관심 급증은 과열/변동성 확대 방향으로 살짝
            bonus += float(zgt.loc[row.name]) * 4.0

    mmi = float(np.clip(mmi + bonus, 0, 100))

    # 3) 레벨/문구
    if mmi < 20:
        level = "Calm"
        desc = "조용한 바다. 과열 신호가 거의 없고, 노이즈 장세일 확률이 큽니다."
    elif mmi < 40:
        level = "Stable"
        desc = "안정 구간. 레버리지/쏠림이 크지 않아 급격한 흔들림 가능성이 낮습니다."
    elif mmi < 60:
        level = "Warm"
        desc = "미지근한 긴장감. 단기 변동성 확대 신호가 섞여 있어 원인(펀딩/청산/쏠림) 점검이 좋아요."
    elif mmi < 80:
        level = "Hot"
        desc = "뜨거운 구간. 레버리지/쏠림 신호가 늘어 변동성 확대가 잦을 수 있습니다."
    else:
        level = "Too Hot"
        desc = "과열 경보. 급변(청산/쏠림) 가능성이 높아 레버리지/포지션 관리를 권장합니다."

    return mmi, level, desc


def draw_gauge(score: float, level: str):
    """
    반원 게이지(0~100) - matplotlib
    """
    bands = [
        (0, 20, "#2E86FF"),   # Calm
        (20, 40, "#2ECC71"),  # Stable
        (40, 60, "#F1C40F"),  # Warm
        (60, 80, "#E67E22"),  # Hot
        (80, 100, "#E74C3C"), # Too Hot
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # 반원 밴드
    for a, b, color in bands:
        theta1 = 180 * (1 - a / 100)
        theta2 = 180 * (1 - b / 100)
        wedge = plt.matplotlib.patches.Wedge(
            (0, 0), 1.0, theta2, theta1,
            width=0.18, color=color, alpha=0.95
        )
        ax.add_patch(wedge)

    # 눈금
    for t in range(0, 101, 10):
        ang = math.radians(180 * (1 - t / 100))
        x1, y1 = 0.82 * math.cos(ang), 0.82 * math.sin(ang)
        x2, y2 = 0.90 * math.cos(ang), 0.90 * math.sin(ang)
        ax.plot([x1, x2], [y1, y2], linewidth=1, color="#D0D0D0")
        if t in [0, 50, 100]:
            xt, yt = 0.68 * math.cos(ang), 0.68 * math.sin(ang)
            ax.text(xt, yt, str(t), ha="center", va="center", fontsize=11, color="#777777")

    # ✅ 바늘 각도 & 좌표(이게 누락돼서 꼬였던 부분)
    ang = math.radians(180 * (1 - score / 100))
    nx, ny = 0.74 * math.cos(ang), 0.74 * math.sin(ang)  # 바늘 길이(겹침 방지)

    # 바늘
    ax.plot([0, nx], [0, ny], linewidth=4, color="#222222", zorder=2)
    ax.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.04, color="#222222", zorder=3))

    # 중앙 텍스트(바늘보다 위 + 흰 배경)
    ax.text(
        0, 0.20, f"{score:.0f}",
        ha="center", va="center",
        fontsize=36, fontweight="bold", color="#111111",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    )
    ax.text(
        0, 0.06, level,
        ha="center", va="center",
        fontsize=14, color="#333333",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.9)
    )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.10, 1.05)
    return fig


# ======================
# VAR helpers (FEVD shape 안정화)
# ======================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    std = df.std().replace(0, np.nan)
    return (df - mean) / std


def run_var_bundle(df: pd.DataFrame, selected_cols: List[str], target: str, lag: int, horizon: int, standardize: bool):
    if len(selected_cols) < 2:
        raise ValueError("VAR 변수는 2개 이상 선택해야 합니다.")
    if target not in selected_cols:
        raise ValueError("Target은 선택된 VAR 변수 안에 있어야 합니다.")

    data = df[selected_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if standardize:
        data = zscore_df(data).dropna()

    if data.shape[0] < max(60, lag * 10):
        raise ValueError(f"데이터가 너무 적습니다: {data.shape[0]} rows (lag={lag})")

    model = VAR(data)
    res = model.fit(lag)

    # Granger
    rows = []
    for x in selected_cols:
        if x == target:
            continue
        try:
            test = res.test_causality(caused=target, causing=[x], kind="f")
            rows.append({"causing(x)": x, "pvalue": float(test.pvalue), "stat": float(test.test_statistic)})
        except Exception as e:
            rows.append({"causing(x)": x, "pvalue": np.nan, "stat": np.nan, "error": str(e)})
    granger = pd.DataFrame(rows).sort_values("pvalue", na_position="last").reset_index(drop=True)

    # IRF
    irf = res.irf(horizon)

    # FEVD (shape 대응)
    fevd = res.fevd(horizon)
    decomp = np.array(fevd.decomp)  # (steps, response, impulse)
    names = list(res.names)
    t_idx = names.index(target)

    # step 0이 포함되는 경우 제거
    if decomp.shape[0] == horizon + 1:
        decomp_use = decomp[1:, t_idx, :]
        idx = list(range(1, horizon + 1))
    else:
        decomp_use = decomp[:, t_idx, :]
        idx = list(range(1, decomp.shape[0] + 1))

    fevd_tbl = pd.DataFrame(decomp_use * 100.0, columns=names, index=idx).round(2)
    fevd_tbl.index.name = "horizon(step)"

    return granger, irf, fevd_tbl


# ======================
# UI
# ======================
def main():
    st.set_page_config(page_title="Bittles Dashboard", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
        h1 { margin-bottom: 0.1rem; }
        [data-testid="stMetricLabel"] { white-space: normal; font-size: 0.9rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Bittles Dashboard")
    st.caption("Risk Signal → Market Mood → VAR(Granger/IRF/FEVD)로 시장 상태를 해석하는 대시보드")

    df = load_data(DATA_PATH)
    if df.empty:
        return

    tab1, tab2, tab3 = st.tabs(["🚦 Risk Signal", "🧠 Market Mood", "🧩 VAR Insight"])

    # -----------------------
    # Sidebar - 공통 날짜(내림차순)
    # -----------------------
    st.sidebar.header("설정")
    dates = sorted(pd.unique(df.index.date), reverse=True)  # ✅ 최근 날짜가 위로
    sel_date = st.sidebar.selectbox("기준 날짜(최근이 위)", dates, format_func=lambda d: d.strftime("%Y-%m-%d"))

    day_df = df[df.index.date == sel_date]
    if day_df.empty:
        st.warning("선택 날짜에 데이터가 없어서 가장 마지막 row로 표시합니다.")
        row = df.iloc[-1]
    else:
        row = day_df.iloc[-1]

    # 이전 날짜 row
    try:
        i = dates.index(sel_date)
        prev_row = None
        if i + 1 < len(dates):
            prev_day_df = df[df.index.date == dates[i + 1]]
            prev_row = prev_day_df.iloc[-1] if not prev_day_df.empty else None
    except Exception:
        prev_row = None

    # 리스크 시그널 계산
    signals = compute_risk_signals(df, row)

    # -----------------------
    # TAB 1: Risk Signal
    # -----------------------
    with tab1:
        st.subheader(f"기준 데이터 날짜: {row.name:%Y-%m-%d}")

        cols = st.columns(5, gap="large")
        order = ["oi", "funding", "liq", "taker", "m2"]

        for ui, key in zip(cols, order):
            item = signals[key]
            label = item["label"]
            colname = item["col"]
            v = item["value"]
            sig = item["signal"]

            if colname is None or pd.isna(v):
                ui.metric(label, "N/A")
                ui.caption(f"{sig} · {item['note']}")
                continue

            # delta
            delta_txt = None
            if prev_row is not None and colname in prev_row.index:
                try:
                    pv = float(prev_row[colname])
                    if key == "m2" and (pv == 0 or float(v) == 0):
                        delta_txt = None
                    else:
                        dv = float(v) - pv
                        if abs(dv) < 1:
                            delta_txt = f"{dv:+.4f}"
                        else:
                            delta_txt = f"{dv:+,.0f}" if abs(dv) > 1000 else f"{dv:+.4g}"
                except Exception:
                    delta_txt = None

            # format value
            if key in ["funding", "taker", "m2"]:
                val_txt = f"{float(v):.4g}"
            else:
                val_txt = f"{float(v):,.4g}"

            ui.metric(label, val_txt, delta=delta_txt)
            cap = f"{sig} · 컬럼: `{colname}`"
            if item["note"]:
                cap += f" · {item['note']}"
            ui.caption(cap)

        st.divider()

        score_list = [signals[k]["score"] for k in order if signals[k]["signal"] != "⚪️"]
        avg_score = float(np.mean(score_list)) if score_list else 1.0

        if avg_score < 0.5:
            st.success("🟢 과열 신호는 약합니다. 급변 시에도 ‘원인(청산/펀딩/쏠림)’부터 확인!")
        elif avg_score < 1.2:
            st.info("🟡 변동성 확대 가능 구간입니다. 패닉셀보다는 구조(청산/펀딩/쏠림)를 점검하세요.")
        else:
            st.warning("🟠~🔴 과열/쏠림 신호가 늘었습니다. 레버리지/포지션 크기 관리가 중요합니다.")

        with st.expander("원본 데이터 미리보기 (가로 스크롤 가능)"):
            st.dataframe(df.tail(50), use_container_width=True, height=420)

    # -----------------------
    # TAB 2: Market Mood (Upbit 느낌 게이지 + 기간별 카드)
    # -----------------------
    with tab2:
        st.subheader("🧠 Market Mood")
        st.caption("Risk Signal(레버리지/쏠림/유동성) + (가능하면) 관심/감성 정보를 합쳐 0~100 지수로 요약합니다.")

        mmi, level, desc = compute_market_mood_index(df, row, signals)

        # 게이지 + 설명 카드 2열
        left, right = st.columns([1.25, 1], gap="large")

        with left:
            fig = draw_gauge(mmi, level)
            st.pyplot(fig, clear_figure=True)

        with right:
            # 업비트 “현재지수” 느낌
            st.markdown(
                f"""
                <div style="border:1px solid #E8E8E8; border-radius:14px; padding:16px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:18px; font-weight:700;">현재 지수</div>
                    <div style="background:#F4F6FA; border-radius:999px; padding:6px 12px; font-weight:700;">
                      {level} · {mmi:.0f}
                    </div>
                  </div>
                  <div style="margin-top:10px; color:#333; line-height:1.55;">
                    {desc}
                  </div>
                  <div style="margin-top:10px; color:#777; font-size:13px;">
                    (참고) Market Mood는 가격 예측이 아니라 “현재 시장의 구조/심리 상태” 요약입니다.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 기간별(1d/7d/30d/90d) 카드
            def get_past_value(days: int) -> Optional[float]:
                ts = row.name - pd.Timedelta(days=days)
                # 가장 가까운 과거 시점
                past = df.loc[:ts]
                if past.empty:
                    return None
                past_row = past.iloc[-1]
                past_signals = compute_risk_signals(df, past_row)
                val, _, _ = compute_market_mood_index(df, past_row, past_signals)
                return val

            p1 = get_past_value(1)
            p7 = get_past_value(7)
            p30 = get_past_value(30)
            p90 = get_past_value(90)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='border:1px solid #E8E8E8; border-radius:14px; padding:16px;'>"
                "<div style='font-size:16px; font-weight:700; margin-bottom:10px;'>기간별 지수</div>",
                unsafe_allow_html=True
            )

            r1, r2 = st.columns(2)
            with r1:
                st.metric("1일 전", "N/A" if p1 is None else f"{p1:.0f}")
                st.metric("1주 전", "N/A" if p7 is None else f"{p7:.0f}")
            with r2:
                st.metric("1개월 전", "N/A" if p30 is None else f"{p30:.0f}")
                st.metric("3개월 전", "N/A" if p90 is None else f"{p90:.0f}")

            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 구간 안내")
        st.write("🔵 Calm → 🟢 Stable → 🟡 Warm → 🟠 Hot → 🔴 Too Hot")

    # -----------------------
    # TAB 3: VAR Insight
    # -----------------------
    with tab3:
        st.subheader("🧩 VAR Insight")
        st.caption("Granger(표) / IRF(그래프) / FEVD(표)")

        st.sidebar.header("VAR 설정")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        default_sel = [c for c in ["ret_log_1d", "funding_close", "taker_buy_ratio", "oi_close", "liq_total_usd"] if c in numeric_cols]
        if len(default_sel) < 2:
            default_sel = numeric_cols[:3]

        sel = st.sidebar.multiselect("VAR 변수 선택(2개 이상)", numeric_cols, default=default_sel)
        target = st.sidebar.selectbox("Target(반응)", sel, index=0) if sel else None
        impulse_candidates = [c for c in sel if c != target] if sel and target else []
        impulse = st.sidebar.selectbox("Impulse(충격)", impulse_candidates, index=0) if impulse_candidates else None
        lag = st.sidebar.slider("VAR lag", 1, 10, 1)
        horizon = st.sidebar.slider("IRF/FEVD horizon", 5, 30, 10)
        standardize = st.sidebar.checkbox("z-score 표준화", True)

        if st.button("VAR 실행", type="primary"):
            try:
                with st.spinner("VAR 적합 중…"):
                    g, irf, fevd = run_var_bundle(df, sel, target, lag, horizon, standardize)

                st.success("완료!")

                st.markdown("### 1) Granger (x → target)")
                st.dataframe(g, use_container_width=True)

                st.markdown("### 2) IRF")
                if impulse and target:
                    fig = irf.plot(impulse=impulse, response=target)
                    fig.set_size_inches(9, 4)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                else:
                    fig = irf.plot()
                    st.pyplot(fig, clear_figure=True)

                st.markdown("### 3) FEVD (target 기준, %)")
                st.dataframe(fevd, use_container_width=True)

            except Exception as e:
                st.error(f"VAR 실행 실패: {e}")


if __name__ == "__main__":
    main()
