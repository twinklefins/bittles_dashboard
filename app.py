import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


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

    # object → numeric 시도 (문자 섞임 대비)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ======================
# Utility
# ======================
def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(s: pd.Series) -> pd.Series:
    s = safe_to_numeric(s)
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

    series = safe_to_numeric(series).dropna()
    if series.empty:
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


def last_valid_z_at_or_before(df: pd.DataFrame, col: str, ts: pd.Timestamp) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    """
    ts 시점 '이전/당일'에서 col의 마지막 유효값을 찾아 z-score 값을 반환
    returns: (z_value, used_timestamp)
    """
    if col is None or col not in df.columns:
        return None, None

    s = safe_to_numeric(df[col])
    s_upto = s.loc[:ts].dropna()
    if s_upto.empty:
        return None, None

    used_ts = s_upto.index[-1]
    z = zscore(s)  # 전체 기간 기준 zscore
    zv = z.loc[used_ts] if used_ts in z.index else None
    if zv is None or pd.isna(zv):
        return None, None
    return float(zv), used_ts

def build_mmi_series(df: pd.DataFrame, lookback_days: int = 60) -> pd.Series:
    """df 전체에 대해 MMI 시계열을 계산해서 Series로 반환"""
    mmi_vals = []
    for ts, row in df.iterrows():
        sig = compute_risk_signals(df, row)
        mmi, _, _, _ = compute_market_mood_index(df, row, sig, lookback_days=lookback_days)
        mmi_vals.append(mmi)
    return pd.Series(mmi_vals, index=df.index, name="MMI")


def run_significance_bundle(
    df: pd.DataFrame,
    mmi_col: str = "MMI",
    ret_col: str = "ret_log_1d",
    forward_days: int = 1,
    hac_lags: int = 5,
    granger_maxlag: int = 5,
):
    """
    1) 상관(Pearson)
    2) 회귀(OLS + HAC robust)
    3) Granger (MMI -> returns)
    """
    out = {}

    if mmi_col not in df.columns:
        raise ValueError(f"'{mmi_col}' 컬럼이 없습니다. 먼저 MMI 시계열을 생성하세요.")
    if ret_col not in df.columns:
        raise ValueError(f"'{ret_col}' 컬럼이 없습니다. (예: ret_log_1d)")

    tmp = df[[mmi_col, ret_col]].copy()
    tmp[mmi_col] = pd.to_numeric(tmp[mmi_col], errors="coerce")
    tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")
    tmp = tmp.dropna()

    # 미래 수익률(선행 검정)
    tmp["ret_fwd"] = tmp[ret_col].shift(-forward_days)
    tmp["absret_fwd"] = tmp["ret_fwd"].abs()
    tmp = tmp.dropna()

    # z-score
    tmp["MMI_z"] = (tmp[mmi_col] - tmp[mmi_col].mean()) / tmp[mmi_col].std()

    # 1) correlation
    out["corr_ret"] = float(tmp[mmi_col].corr(tmp["ret_fwd"]))
    out["corr_absret"] = float(tmp[mmi_col].corr(tmp["absret_fwd"]))

    # 2) regression HAC
    X = sm.add_constant(tmp["MMI_z"])
    y1 = tmp["ret_fwd"]
    y2 = tmp["absret_fwd"]

    res1 = sm.OLS(y1, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    res2 = sm.OLS(y2, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # 표로 쓰기 쉬운 요약
    def coef_table(res):
        coef = res.params.get("MMI_z", np.nan)
        pval = res.pvalues.get("MMI_z", np.nan)
        tval = res.tvalues.get("MMI_z", np.nan)
        return {"coef(MMI_z)": float(coef), "t": float(tval), "pvalue": float(pval)}

    out["reg_ret"] = coef_table(res1)
    out["reg_absret"] = coef_table(res2)

    # 3) Granger (MMI -> ret)
    # grangercausalitytests는 [y, x] 순서
    gdf = tmp[["ret_fwd", mmi_col]].dropna().rename(columns={mmi_col: "MMI"})
    g = grangercausalitytests(gdf[["ret_fwd", "MMI"]], maxlag=granger_maxlag, verbose=False)

    pvals = []
    for lag in range(1, granger_maxlag + 1):
        p = g[lag][0]["ssr_ftest"][1]
        pvals.append({"lag": lag, "pvalue": float(p)})

    out["granger_pvals"] = pd.DataFrame(pvals)

    return out


# ======================
# Risk → Market Mood Index (0~100)
# ======================
def compute_risk_signals(df: pd.DataFrame, row: pd.Series) -> Dict[str, Dict]:
    """
    returns dict with per-indicator: value, signal, score, colname, note
    """
    col_oi = find_col(df, ["oi_close", "oi_close_diff", "open_interest", "oi"])
    col_funding = find_col(df, ["funding_close", "funding_rate", "funding"])
    col_liq = find_col(df, ["liq_total_usd", "liq_total_usd_diff", "liquidation_usd", "liq_usd"])
    col_taker = find_col(df, ["taker_buy_ratio", "taker_ratio"])
    col_m2 = find_col(df, ["global_m2_yoy_diff", "global_m2_yoy", "m2_yoy_diff"])

    indicators = [
        ("oi", "OI", col_oi, True),
        ("funding", "Funding", col_funding, True),
        ("liq", "Liquidation(USD)", col_liq, True),
        ("taker", "Taker Bias", col_taker, True),   # 0.5에서 멀수록 위험
        ("m2", "Global M2", col_m2, False),         # 낮을수록 위험(방어적으로)
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
            series = (safe_to_numeric(df[col]) - 0.5).abs()
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

        series = safe_to_numeric(df[col])
        sig, sc = percentile_signal(series, float(v) if not pd.isna(v) else np.nan, higher_is_risky=higher_is_risky)

        out[key] = {
            "label": label, "col": col, "value": float(v) if not pd.isna(v) else np.nan,
            "signal": sig, "score": sc, "note": ""
        }

    return out


def compute_market_mood_index(
    df: pd.DataFrame,
    row: pd.Series,
    signals: Dict[str, Dict],
    lookback_days: int = 60,
) -> Tuple[float, str, str, Dict[str, object]]:
    """
    MMI: 0~100 (낮을수록 Calm, 높을수록 Too Hot)
    - base: Risk Signal 점수 평균(0~2) → 0~100
    - bonus: (가능하면) sentiment / google trends를 lookback으로 보정
    - explain: 어떤 컬럼이 실제로 반영되었는지 리포트(dict) 반환
    """
    ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)

    # 1) Base = Risk score 평균
    used_signals = {k: v for k, v in signals.items() if v.get("signal") != "⚪️"}
    scores = [v["score"] for v in used_signals.values()]  # 0~2
    base = float(np.mean(scores)) if scores else 1.0
    mmi_base = (base / 2.0) * 100.0

    # 2) Optional bonus (lookback)
    # 네가 어제 정리해둔 컬럼명 우선 반영
    col_sent = find_col(df, ["rd_avg_sent", "avg_sent", "sentiment"])
    col_gt = find_col(df, ["gt_btc_z14", "gtrend_btc_z14", "gt_bitcoin", "gtrend_btc"])

    bonus = 0.0
    used_inputs = []
    min_ts = ts - pd.Timedelta(days=lookback_days)

    # sentiment bonus
    if col_sent and col_sent in df.columns:
        zv, used_ts = last_valid_z_at_or_before(df, col_sent, ts)
        if used_ts is not None and used_ts >= min_ts:
            w = 6.0
            contrib = float(zv) * w
            bonus += contrib
            used_inputs.append({
                "type": "sentiment",
                "col": col_sent,
                "z": float(zv),
                "weight": w,
                "contrib": contrib,
                "used_ts": used_ts,
            })

    # attention bonus
    if col_gt and col_gt in df.columns:
        zv, used_ts = last_valid_z_at_or_before(df, col_gt, ts)
        if used_ts is not None and used_ts >= min_ts:
            w = 4.0
            contrib = float(zv) * w
            bonus += contrib
            used_inputs.append({
                "type": "attention",
                "col": col_gt,
                "z": float(zv),
                "weight": w,
                "contrib": contrib,
                "used_ts": used_ts,
            })

    mmi = float(np.clip(mmi_base + bonus, 0, 100))

    # 3) Level & description
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

    explain = {
        "ts": ts,
        "lookback_days": lookback_days,
        "base_avg_score(0~2)": base,
        "mmi_base(0~100)": mmi_base,
        "bonus": float(bonus),
        "final_mmi": float(mmi),
        "risk_inputs_used": [
            {
                "key": k,
                "label": v.get("label"),
                "col": v.get("col"),
                "score": v.get("score"),
                "signal": v.get("signal"),
                "note": v.get("note", ""),
            }
            for k, v in signals.items()
        ],
        "optional_inputs_used": used_inputs,
        "optional_candidates": {"sentiment_col": col_sent, "attention_col": col_gt},
    }

    return mmi, level, desc, explain


def draw_gauge(score: float, level: str):
    """
    반원 게이지(0~100) - matplotlib
    """
    bands = [
        (0, 20, "#2E86FF"),    # Calm
        (20, 40, "#2ECC71"),   # Stable
        (40, 60, "#F1C40F"),   # Warm
        (60, 80, "#E67E22"),   # Hot
        (80, 100, "#E74C3C"),  # Too Hot
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # 밴드
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
        ang_t = math.radians(180 * (1 - t / 100))
        x1, y1 = 0.82 * math.cos(ang_t), 0.82 * math.sin(ang_t)
        x2, y2 = 0.90 * math.cos(ang_t), 0.90 * math.sin(ang_t)
        ax.plot([x1, x2], [y1, y2], linewidth=1, color="#D0D0D0")
        if t in [0, 50, 100]:
            xt, yt = 0.68 * math.cos(ang_t), 0.68 * math.sin(ang_t)
            ax.text(xt, yt, str(t), ha="center", va="center", fontsize=11, color="#777777")

    # 바늘(각도/좌표 반드시 정의)
    ang = math.radians(180 * (1 - score / 100))
    nx, ny = 0.74 * math.cos(ang), 0.74 * math.sin(ang)

    ax.plot([0, nx], [0, ny], linewidth=4, color="#222222", zorder=2)
    ax.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.04, color="#222222", zorder=3))

    # 중앙 텍스트 (바늘 위로 + 배경)
    ax.text(
        0, 0.20, f"{score:.0f}",
        ha="center", va="center",
        fontsize=36, fontweight="bold", color="#111111",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92)
    )
    ax.text(
        0, 0.06, level,
        ha="center", va="center",
        fontsize=14, color="#333333",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.92)
    )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.10, 1.05)
    return fig


# ======================
# VAR helpers
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

    irf = res.irf(horizon)

    fevd = res.fevd(horizon)
    decomp = np.array(fevd.decomp)  # (steps, response, impulse)
    names = list(res.names)
    t_idx = names.index(target)

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

    tab1, tab2, tab3, tab4 = st.tabs(["🚦 Risk Signal", "🧠 Market Mood", "🧩 VAR Insight", "🧪 Significance Test"])

    # Sidebar 날짜 (최근이 위)
    st.sidebar.header("설정")
    dates = sorted(pd.unique(df.index.date), reverse=True)
    sel_date = st.sidebar.selectbox("기준 날짜(최근이 위)", dates, format_func=lambda d: d.strftime("%Y-%m-%d"))

    day_df = df[df.index.date == sel_date]
    row = day_df.iloc[-1] if not day_df.empty else df.iloc[-1]

    # 이전 날짜 row
    prev_row = None
    try:
        i = dates.index(sel_date)
        if i + 1 < len(dates):
            prev_day_df = df[df.index.date == dates[i + 1]]
            prev_row = prev_day_df.iloc[-1] if not prev_day_df.empty else None
    except Exception:
        prev_row = None

    signals = compute_risk_signals(df, row)

    # -----------------------
    # TAB 1
    # -----------------------
    with tab1:
        st.subheader(f"기준 데이터 날짜: {row.name:%Y-%m-%d}")

        cols = st.columns(5, gap="large")
        order = ["oi", "funding", "liq", "taker", "m2"]

        for ui, key in zip(cols, order):
            item = signals[key]
            label, colname = item["label"], item["col"]
            v, sig = item["value"], item["signal"]

            if colname is None or pd.isna(v):
                ui.metric(label, "N/A")
                ui.caption(f"{sig} · {item['note']}")
                continue

            delta_txt = None
            if prev_row is not None and colname in prev_row.index:
                try:
                    pv = float(prev_row[colname])
                    if key == "m2" and (pv == 0 or float(v) == 0):
                        delta_txt = None
                    else:
                        dv = float(v) - pv
                        delta_txt = f"{dv:+.4g}"
                except Exception:
                    delta_txt = None

            val_txt = f"{float(v):.4g}" if key in ["funding", "taker", "m2"] else f"{float(v):,.4g}"
            ui.metric(label, val_txt, delta=delta_txt)

            cap = f"{sig} · 컬럼: `{colname}`"
            if item["note"]:
                cap += f" · {item['note']}"
            ui.caption(cap)

        st.divider()
        score_list = [signals[k]["score"] for k in order if signals[k]["signal"] != "⚪️"]
        avg_score = float(np.mean(score_list)) if score_list else 1.0

        if avg_score < 0.5:
            st.success("🟢 과열 신호는 약합니다.")
        elif avg_score < 1.2:
            st.info("🟡 변동성 확대 가능 구간입니다.")
        else:
            st.warning("🟠~🔴 과열/쏠림 신호가 늘었습니다. 레버리지/포지션 크기 관리가 중요합니다.")

        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df.tail(50), use_container_width=True, height=420)

    # -----------------------
    # TAB 2
    # -----------------------
    with tab2:
        st.subheader("🧠 Market Mood")
        st.caption("Risk Signal + (가능하면) Google Trends / Reddit Sentiment 보정으로 0~100 지수로 요약합니다.")

        mmi, level, desc, explain = compute_market_mood_index(df, row, signals, lookback_days=60)

        left, right = st.columns([1.25, 1], gap="large")

        with left:
            fig = draw_gauge(mmi, level)
            st.pyplot(fig, clear_figure=True)

        with right:
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

            # ✅ 여기! 과거값 계산에서 “리턴값 4개” 안전하게 처리
            def get_past_value(days: int) -> Optional[float]:
                ts = row.name - pd.Timedelta(days=days)
                past = df.loc[:ts]
                if past.empty:
                    return None
                past_row = past.iloc[-1]
                past_signals = compute_risk_signals(df, past_row)
                val, _, _, _ = compute_market_mood_index(df, past_row, past_signals, lookback_days=60)
                return float(val)

            p1, p7, p30, p90 = get_past_value(1), get_past_value(7), get_past_value(30), get_past_value(90)

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

            with st.expander("🔍 Market Mood 계산 상세 (어떤 컬럼이 반영됐는지)"):
                st.markdown("### 1) Base: Risk Signal 평균 → 0~100")
                st.write(f"- 평균 점수(0~2): **{explain['base_avg_score(0~2)']:.2f}**")
                st.write(f"- Base MMI(0~100): **{explain['mmi_base(0~100)']:.1f}**")

                st.markdown("### 2) Risk Signal에 실제로 사용된 컬럼")
                st.dataframe(pd.DataFrame(explain["risk_inputs_used"]), use_container_width=True)

                st.markdown("### 3) Bonus: Sentiment / Google Trends (있을 때만)")
                st.markdown(
                    """
                **왜 Bonus를 쓰나?**  
                행동 데이터(OI/청산/펀딩)가 시장의 ‘구조’를 보여준다면,  
                Sentiment와 관심도는 **그 구조에 사람들이 얼마나 반응하고 있는지**를 보여줍니다.

                **가중치 설계**
                - Sentiment × 6  
                → 공포·탐욕은 단기 변동성에 직접적인 영향을 주기 때문
                - Google Trends × 4  
                → 관심 급증은 과열의 보조 신호 (후행 가능성 고려)

                ⚠️ Bonus는 Base를 뒤집지 않고, **설명력만 보강**합니다.
                """
                )
                if len(explain["optional_inputs_used"]) == 0:
                    st.info(
                        "이번 날짜에는 **심리/관심 보정이 적용되지 않았습니다.**\n\n"
                        "- 해당 컬럼이 데이터에 없거나\n"
                        "- 최근 60일 이내 유효한 값이 없기 때문입니다.\n\n"
                        "→ 이 경우 Market Mood는 **행동 데이터만으로 계산**됩니다."
                    )
                    st.write("🔎 탐지된 후보 컬럼:", explain["optional_candidates"])
                else:
                    bonus_df = pd.DataFrame(explain["optional_inputs_used"])
                    bonus_df["used_ts"] = bonus_df["used_ts"].astype(str)
                    st.dataframe(bonus_df, use_container_width=True)

                st.markdown("### 4) 최종")
                st.write(f"- Bonus 합계: **{explain['bonus']:+.2f}**")
                st.write(f"- 최종 MMI: **{explain['final_mmi']:.1f} ({level})**")

        st.divider()
        st.write("구간 안내: 🔵 Calm → 🟢 Stable → 🟡 Warm → 🟠 Hot → 🔴 Too Hot")

    # -----------------------
    # TAB 3
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

    # -----------------------
    # TAB 4
    # -----------------------
    with tab4:
        st.subheader("🧪 유의성 검정: Market Mood → Price")
        st.caption("MMI가 미래 수익률/변동성을 선행 설명하는지 (상관/회귀/HAC/Granger)로 확인합니다.")

        # 1) MMI 시계열 생성(캐시 추천)
        @st.cache_data(show_spinner=False)
        def get_mmi_df(_df: pd.DataFrame) -> pd.DataFrame:
            dfx = _df.copy()
            dfx["MMI"] = build_mmi_series(dfx, lookback_days=60)
            return dfx

        df2 = get_mmi_df(df)

        c1, c2, c3 = st.columns(3)
        with c1:
            ret_col = st.selectbox("수익률 컬럼", [c for c in df2.columns if "ret" in c.lower()], index=0)
        with c2:
            fwd = st.selectbox("미래 시차 (days)", [1, 3, 7], index=0)
        with c3:
            maxlag = st.selectbox("Granger maxlag", [3, 5, 7, 10], index=1)

        if st.button("검정 실행", type="primary"):
            try:
                out = run_significance_bundle(
                    df=df2,
                    mmi_col="MMI",
                    ret_col=ret_col,
                    forward_days=int(fwd),
                    hac_lags=5,
                    granger_maxlag=int(maxlag),
                )

                st.markdown("### 1) 상관관계")
                st.write(f"- Corr(MMI, future return): **{out['corr_ret']:.4f}**")
                st.write(f"- Corr(MMI, future |return|): **{out['corr_absret']:.4f}**")

                st.markdown("### 2) 회귀 (OLS + HAC robust)")
                reg_tbl = pd.DataFrame([
                    {"model": f"future return (t+{fwd})", **out["reg_ret"]},
                    {"model": f"future |return| (t+{fwd})", **out["reg_absret"]},
                ])
                st.dataframe(reg_tbl, use_container_width=True)

                st.markdown("### 3) Granger (MMI → future return)")
                st.dataframe(out["granger_pvals"], use_container_width=True)

                best = out["granger_pvals"].sort_values("pvalue").iloc[0]
                if best["pvalue"] < 0.05:
                    st.success(f"✅ Granger 유의: lag={int(best['lag'])}, p={best['pvalue']:.4f} → MMI가 수익률을 선행 설명할 가능성이 있습니다.")
                else:
                    st.info(f"ℹ️ Granger 유의 증거 약함: 최저 p={best['pvalue']:.4f} (maxlag={maxlag})")

            except Exception as e:
                st.error(f"검정 실패: {e}")

        with st.expander("해석 가이드(팀 공유용)"):
            st.markdown(
                """
    - **상관**: 같이 움직이는 경향(인과 아님)
    - **회귀 p-value < 0.05**: MMI가 미래 수익률/변동성을 설명하는 통계적 근거
    - **Granger p-value < 0.05**: MMI가 수익률을 '선행'하는 패턴이 있다는 근거
    - 일반적으로 **방향(수익률)** 보다 **위험(|수익률|/변동성)** 쪽이 더 잘 나오는 경우가 많습니다.
    """
            )

if __name__ == "__main__":
    main()
    