# pages/01_Market_Mood.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data.loader import load_df
from components.header import render_header
from components.styles import inject_styles

# ======================
# Config
# ======================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "df_draft_1209_w.sti.csv"
KST = "Asia/Seoul"


# ======================
# Helpers
# ======================
def md_html(s: str) -> None:
    """
    Streamlit markdown은 '4칸 들여쓰기'를 코드블록으로 인식해서
    <div>가 그대로 글자로 찍히는 이슈가 자주 생김.
    → HTML 라인들은 왼쪽 들여쓰기를 제거해서 안전하게 렌더링.
    """
    raw = dedent(s).strip("\n")
    lines = raw.splitlines()
    fixed = []
    for line in lines:
        if line.lstrip().startswith("<"):
            fixed.append(line.lstrip())
        else:
            fixed.append(line)
    st.markdown("\n".join(fixed), unsafe_allow_html=True)


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
    """return (signal_emoji, score) score: 🟢0, 🟡1, 🔴2, ⚪️(missing)=1"""
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
    if col is None or col not in df.columns:
        return None, None

    s = safe_to_numeric(df[col])
    s_upto = s.loc[:ts].dropna()
    if s_upto.empty:
        return None, None

    used_ts = s_upto.index[-1]
    zv = zscore(s).loc[used_ts]
    if pd.isna(zv):
        return None, None
    return float(zv), used_ts


def fmt(v: Optional[float]) -> str:
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


def delta_str(now: float, past: Optional[float]) -> str:
    if past is None or (isinstance(past, float) and np.isnan(past)):
        return ""
    d = now - float(past)
    if d > 0:
        return f"▲ {abs(d):.0f}"
    if d < 0:
        return f"▼ {abs(d):.0f}"
    return ""

def trend_class(now: float, past: Optional[float]) -> str:
    if past is None or (isinstance(past, float) and np.isnan(past)):
        return "flat"
    d = now - float(past)
    if d > 0:
        return "up"
    if d < 0:
        return "down"
    return "flat"

def trend_delta_class(now: float, past: Optional[float]) -> str:
    return trend_class(now, past)

def level_pill_dot(level: str) -> str:
    # mood key color (dot / glow)
    m = {
        "Calm":    "#2D6BFF",
        "Stable":  "#19D3FF",
        "Warm":    "#FFB020",
        "Hot":     "#FF6B00",
        "Too Hot": "#FF2D55",
    }
    return m.get(level, "#FF6B00")

def inject_mm_css() -> None:
    st.markdown(
        """
<style>
/* =========================
   컨테이너 위치 조절 (안정판)
========================= */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-mm="score"]){
  margin-top: -12px !important;   /* 🔼 Market Mood Score 위로 */
  margin-bottom: 12px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-mm="btc"]){
  margin-top: 0px !important;
  margin-bottom: 8px !important;
}

/* =========================
   HERO
========================= */
.mm-hero{
  margin-top: 10px;
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 16px 44px rgba(0,0,0,0.26);
  backdrop-filter: blur(14px);
  position: relative;
  overflow: hidden;
}
.mm-hero-top{ display:flex; align-items:flex-start; justify-content:space-between; gap: 12px; }
.mm-hero-title{ font-weight: 900; letter-spacing:-0.01em; color: rgba(255,255,255,.94); }
.mm-hero-desc{ margin-top:8px; color: rgba(255,255,255,.74); line-height:1.45; }

.mm-title-accent{
  display:inline-block; width:6px; height:14px; border-radius:4px; margin-right:8px;
  background: var(--mood-color); box-shadow: 0 0 12px var(--mood-color);
  vertical-align: middle;
}

.mm-badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px; border-radius:999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.20);
  color: rgba(255,255,255,.90);
  font-weight: 900; font-size: 13px; white-space:nowrap;
}
.mm-dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }

.mm-hero.mm-today{
  border-color: rgba(255,176,32,0.45);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset,
              0 16px 44px rgba(0,0,0,0.26),
              0 0 18px -12px var(--mood-color);
}

/* =========================
   KPI GRID
========================= */
.mm-kpi-grid{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
.mm-kpi-item{
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 10px 28px rgba(0,0,0,0.18);
  backdrop-filter: blur(12px);
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
.mm-kpi-item:hover{ transform: translateY(-2px); border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.055); }
.mm-kpi-label{ font-size:12px; text-transform:uppercase; opacity:.72; font-weight:800; color: rgba(255,255,255,.76); margin-bottom:4px; }
.mm-kpi-val{ font-size:19px; font-weight:900; color: rgba(255,255,255,.92); }
.mm-kpi-delta{ margin-left:6px; font-size:12px; font-weight:900; opacity:.90; }

.mm-kpi-item.up{ border-color: rgba(255,176,32,0.38) !important; }
.mm-kpi-item.down{ border-color: rgba(45,107,255,0.38) !important; }
.mm-kpi-delta.up{ color: rgba(255,176,32,0.98) !important; }
.mm-kpi-delta.down{ color: rgba(45,107,255,0.98) !important; }
.mm-kpi-delta.flat{ color: rgba(255,255,255,0.70) !important; }

/* =========================
   SCALE BAR
========================= */
.mm-scale{ margin-top:10px; }
.mm-scale-bar{
  height:6px; border-radius:999px;
  background: linear-gradient(90deg,#2D6BFF,#19D3FF,#FFB020,#FF6B00,#FF2D55);
  opacity:.90;
  box-shadow: 0 0 0 1px rgba(255,255,255,.08) inset;
}
.mm-scale-labels{
  margin-top:6px;
  display:flex; justify-content:space-between;
  font-size:11px; color: rgba(255,255,255,.55);
  letter-spacing:.01em;
}

/* =========================
   BTC MINI CARDS
========================= */
.mm-btc-row{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 14px;
}

.mm-mini-card{
  position: relative;
  border-radius: 16px;
  padding: 12px 12px 14px 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 10px 28px rgba(0,0,0,0.18);
  backdrop-filter: blur(12px);
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
.mm-mini-card:hover{
  transform: translateY(-2px);
  border-color: rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.055);
}

.mm-mini-label{
  font-size: 11px;
  letter-spacing: .06em;
  font-weight: 900;
  color: rgba(255,255,255,0.62);
  text-transform: uppercase;
}

.mm-mini-sep{
  height: 0.25px;
  border-radius: 999px;
  margin: 8px 0 10px;
  background: linear-gradient(90deg,#2D6BFF,#19D3FF,#FFB020,#FF6B00,#FF2D55);
  opacity: .92;
  box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset;
}

.mm-mini-value{
  font-size: 20px;
  font-weight: 900;
  color: rgba(255,255,255,0.92);
}
.mm-mini-sub{
  margin-top: 6px;
  font-size: 11px;
  color: rgba(255,255,255,0.52);
}


</style>
        """,
        unsafe_allow_html=True,
    )

# ======================
# Risk → Market Mood Index
# ======================
def compute_risk_signals(df: pd.DataFrame, row: pd.Series) -> Dict[str, Dict]:
    col_oi = find_col(df, ["oi_close", "oi_close_diff", "open_interest", "oi"])
    col_funding = find_col(df, ["funding_close", "funding_rate", "funding"])
    col_liq = find_col(df, ["liq_total_usd", "liq_total_usd_diff", "liquidation_usd", "liq_usd"])
    col_taker = find_col(df, ["taker_buy_ratio", "taker_ratio"])
    col_m2 = find_col(df, ["global_m2_yoy_diff", "global_m2_yoy", "m2_yoy_diff"])

    indicators = [
        ("oi", "OI", col_oi, True),
        ("funding", "Funding", col_funding, True),
        ("liq", "Liquidation(USD)", col_liq, True),
        ("taker", "Taker Bias", col_taker, True),
        ("m2", "Global M2", col_m2, False),
    ]

    out: Dict[str, Dict] = {}
    for key, label, col, higher_is_risky in indicators:
        if col is None or col not in df.columns:
            out[key] = {"label": label, "col": None, "value": np.nan, "signal": "⚪️", "score": 1}
            continue

        v = row.get(col, np.nan)

        if key == "taker" and not pd.isna(v):
            dist = abs(float(v) - 0.5)
            series = (safe_to_numeric(df[col]) - 0.5).abs()
            sig, sc = percentile_signal(series, dist, higher_is_risky=True)
            out[key] = {"label": label, "col": col, "value": float(v), "signal": sig, "score": sc}
            continue

        if key == "m2" and (pd.isna(v) or float(v) == 0.0):
            out[key] = {"label": label, "col": col, "value": float(v) if not pd.isna(v) else np.nan, "signal": "⚪️", "score": 1}
            continue

        series = safe_to_numeric(df[col])
        sig, sc = percentile_signal(series, float(v) if not pd.isna(v) else np.nan, higher_is_risky=higher_is_risky)
        out[key] = {"label": label, "col": col, "value": float(v) if not pd.isna(v) else np.nan, "signal": sig, "score": sc}

    return out


def compute_market_mood_index(df: pd.DataFrame, row: pd.Series, signals: Dict[str, Dict], lookback_days: int = 60):
    ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)

    used_signals = {k: v for k, v in signals.items() if v.get("signal") != "⚪️"}
    scores = [v["score"] for v in used_signals.values()]
    base = float(np.mean(scores)) if scores else 1.0
    mmi_base = (base / 2.0) * 100.0

    col_sent = find_col(df, ["rd_avg_sent", "avg_sent", "sentiment"])
    col_gt = find_col(df, ["gt_btc_z14", "gtrend_btc_z14", "gt_bitcoin", "gtrend_btc"])

    bonus = 0.0
    min_ts = ts - pd.Timedelta(days=lookback_days)

    if col_sent and col_sent in df.columns:
        zv, used_ts = last_valid_z_at_or_before(df, col_sent, ts)
        if used_ts is not None and used_ts >= min_ts:
            bonus += float(zv) * 6.0

    if col_gt and col_gt in df.columns:
        zv, used_ts = last_valid_z_at_or_before(df, col_gt, ts)
        if used_ts is not None and used_ts >= min_ts:
            bonus += float(zv) * 4.0

    mmi = float(np.clip(mmi_base + bonus, 0, 100))

    if mmi < 20:
        level, desc = "Calm", "과열 신호가 거의 없고, 노이즈 장세일 가능성이 큽니다."
    elif mmi < 40:
        level, desc = "Stable", "레버리지/쏠림이 크지 않아 급격한 변동 가능성이 낮습니다."
    elif mmi < 60:
        level, desc = "Warm", "변동성 신호가 일부 섞여 있어 원인(펀딩/청산/쏠림) 점검이 필요합니다."
    elif mmi < 80:
        level, desc = "Hot", "레버리지/쏠림 신호가 늘어 변동성 확대가 잦을 수 있습니다."
    else:
        level, desc = "Too Hot", "급변(청산/쏠림) 가능성이 높아 포지션/레버리지 관리가 필요합니다."

    return mmi, level, desc


# ======================
# Gauge
# ======================
def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c1, c2, t):
    return (_lerp(c1[0], c2[0], t), _lerp(c1[1], c2[1], t), _lerp(c1[2], c2[2], t))


def _gradient_color(t: float) -> Tuple[float, float, float]:
    stops = [
        (0.00, _hex_to_rgb("#2D6BFF")),
        (0.28, _hex_to_rgb("#19D3FF")),
        (0.55, _hex_to_rgb("#FFB020")),
        (0.78, _hex_to_rgb("#FF6B00")),
        (1.00, _hex_to_rgb("#FF2D55")),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0 + 1e-9)
            return _lerp_rgb(c0, c1, u)
    return stops[-1][1]


def draw_gauge(score: float, level: str):
    # Apple Watch-ish thin arc + low glow
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    fig.patch.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")

    R, W = 1.0, 0.10
    N = 320

    # base arc
    for i in range(N):
        t0, t1 = i / N, (i + 1) / N
        a0, a1 = 180 * (1 - t0), 180 * (1 - t1)
        ax.add_patch(
            plt.matplotlib.patches.Wedge(
                (0, 0), R, a1, a0, width=W, color=(1, 1, 1, 0.045)
            )
        )

    # progress arc
    p = max(0.0, min(1.0, score / 100.0))
    progN = max(1, int(N * p))
    for i in range(progN):
        t0, t1 = i / N, (i + 1) / N
        col = _gradient_color((t0 + t1) / 2)
        a0, a1 = 180 * (1 - t0), 180 * (1 - t1)
        ax.add_patch(
            plt.matplotlib.patches.Wedge(
                (0, 0), R, a1, a0, width=W, color=col, alpha=0.92
            )
        )

    # end glow (reduced)
    end_ang = math.radians(180 * (1 - p))
    ex, ey = R * math.cos(end_ang), R * math.sin(end_ang)

    for k in range(6):
        ax.add_patch(
            plt.matplotlib.patches.Circle(
                (ex, ey),
                0.018 + k * 0.010,
                color=(*_gradient_color(p),),
                alpha=0.055 / (k + 1),
                zorder=8,
            )
        )
    ax.add_patch(
        plt.matplotlib.patches.Circle(
            (ex, ey), 0.026,
            color=(*_gradient_color(p),),
            alpha=0.92,
            zorder=9,
        )
    )

    # center text
    ax.text(
        0, 0.18, f"{score:.0f}",
        ha="center", va="center",
        fontsize=52, fontweight="heavy",
        color=(234/255, 240/255, 255/255, 0.96),
    )
    ax.text(
        0, 0.01, level,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color=(234/255, 240/255, 255/255, 0.70),
    )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.02, 1.03)
    return fig


def get_past_value(df: pd.DataFrame, anchor_ts: pd.Timestamp, days: int) -> Optional[float]:
    anchor_ts = pd.Timestamp(anchor_ts)
    ts = anchor_ts - pd.Timedelta(days=days)
    past = df.loc[:ts]
    if past.empty:
        return None
    past_row = past.iloc[-1]
    past_signals = compute_risk_signals(df, past_row)
    val, _, _ = compute_market_mood_index(df, past_row, past_signals, lookback_days=60)
    return float(val)


# ======================
# BTC mini cards
# ======================
def compute_btc_general_info(df: pd.DataFrame, anchor_ts: pd.Timestamp) -> dict:
    col_close = find_col(df, ["close", "btc_close", "price_close", "spot_close", "px_close"])
    if not col_close:
        return {"close": None, "chg": None, "chg_pct": None, "vol7d": None}

    s = safe_to_numeric(df[col_close]).dropna()
    s_upto = s.loc[:anchor_ts].dropna()
    if s_upto.empty:
        return {"close": None, "chg": None, "chg_pct": None, "vol7d": None}

    now = float(s_upto.iloc[-1])
    prev = float(s_upto.iloc[-2]) if len(s_upto) >= 2 else np.nan

    chg = None if np.isnan(prev) else now - prev
    chg_pct = None if np.isnan(prev) or prev == 0 else (now / prev - 1) * 100

    ret = s_upto.pct_change().dropna()
    vol7d = float(ret.tail(7).std() * 100) if len(ret) >= 7 else None

    return {"close": now, "chg": chg, "chg_pct": chg_pct, "vol7d": vol7d}


def fmt_money(v: Optional[float]) -> str:
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}"


def fmt_signed(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.0f}"


def fmt_signed_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


# ======================
# Run
# ======================
inject_styles()  # ✅ 여기로 올려서 페이지 시작하자마자 1번만

df_utc = load_df(DATA_PATH)
if df_utc.empty:
    st.stop()

df = df_utc.copy()
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert(KST)

anchor_ts, row = render_header(
    df,
    title="Market Mood",
    subtitle="시장 상태 요약과 BTC 주요 지표를 함께 보여줍니다.",
    date_key="mm_date",
)

signals = compute_risk_signals(df, row)
mmi, level, desc = compute_market_mood_index(df, row, signals, lookback_days=60)
MOOD_COLOR = level_pill_dot(level)

p1 = get_past_value(df, row.name, 1)
p7 = get_past_value(df, row.name, 7)
p30 = get_past_value(df, row.name, 30)
p90 = get_past_value(df, row.name, 90)

d1 = delta_str(mmi, p1)
d7 = delta_str(mmi, p7)
d30 = delta_str(mmi, p30)
d90 = delta_str(mmi, p90)

# ======================
# ✅ Market Mood Score
# ======================
inject_mm_css()

with st.container(border=True):
    md_html('<div data-mm="score"></div>')

    md_html("""
      <div class="mm-block-head">
        <div class="mm-block-title">Market Mood Score</div>
        <div class="mm-block-sub">
          현재 시장의 리스크와 심리 상태를 종합해 0–100 점수로 나타냅니다.
        </div>
      </div>
    """)

    left, right = st.columns([1.22, 1.0], gap="large")

    with left:
        fig = draw_gauge(mmi, level)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        md_html("""
        <div class="mm-scale">
          <div class="mm-scale-bar"></div>
          <div class="mm-scale-labels">
            <span>Calm</span><span>Stable</span><span>Warm</span><span>Hot</span><span>Too Hot</span>
          </div>
        </div>
        """)

    with right:
        md_html(f"""
        <div class="mm-hero mm-today" style="--mood-color:{MOOD_COLOR};">
          <div class="mm-hero-top">
            <div class="mm-hero-title">
              <span class="mm-title-accent"></span>
              오늘의 시장 온도
            </div>
            <span class="mm-badge">
              <span class="mm-dot" style="background:{MOOD_COLOR};"></span>
              {level} · {mmi:.0f}
            </span>
          </div>
          <div class="mm-hero-desc">{desc}</div>
          <div style="margin-top:10px; font-size:12px; color:rgba(255,255,255,.55);">
            Market Mood는 가격 예측이 아니라 현재 시장 상태를 읽기 위한 일간 지표입니다.
          </div>
        </div>
        <div style="height:14px"></div>
        """)

        t1  = trend_class(mmi, p1)
        t7  = trend_class(mmi, p7)
        t30 = trend_class(mmi, p30)
        t90 = trend_class(mmi, p90)

        md_html(f"""
        <div style="font-weight:900; letter-spacing:-.01em; margin-bottom:8px;">시장 온도 추이</div>
        <div class="mm-kpi-grid">
          <div class="mm-kpi-item {t1}">
            <div class="mm-kpi-label">1D ago</div>
            <div><span class="mm-kpi-val">{fmt(p1)}</span> <span class="mm-kpi-delta {t1}">{d1}</span></div>
          </div>
          <div class="mm-kpi-item {t7}">
            <div class="mm-kpi-label">1W ago</div>
            <div><span class="mm-kpi-val">{fmt(p7)}</span> <span class="mm-kpi-delta {t7}">{d7}</span></div>
          </div>
          <div class="mm-kpi-item {t30}">
            <div class="mm-kpi-label">1M ago</div>
            <div><span class="mm-kpi-val">{fmt(p30)}</span> <span class="mm-kpi-delta {t30}">{d30}</span></div>
          </div>
          <div class="mm-kpi-item {t90}">
            <div class="mm-kpi-label">3M ago</div>
            <div><span class="mm-kpi-val">{fmt(p90)}</span> <span class="mm-kpi-delta {t90}">{d90}</span></div>
          </div>
        </div>
        """)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# ======================
# ✅ BTC Daily Snapshot
# ======================
btc = compute_btc_general_info(df, row.name)
v1 = fmt_money(btc["close"])
v2 = fmt_signed(btc["chg"])
v3 = fmt_signed_pct(btc["chg_pct"])
v4 = "N/A" if btc["vol7d"] is None else f"{btc['vol7d']:.2f}%"

st.markdown('<div class="btc-snapshot">', unsafe_allow_html=True)

with st.container(border=True):
    md_html('<div data-mm="btc"></div>')

    md_html("""
      <div class="mm-block-head">
        <div class="mm-block-title">BTC Daily Snapshot</div>
        <div class="mm-block-sub">
          전일 종가 기준으로 본 비트코인 주요 지표를 정리합니다.
        </div>
      </div>
    """)

    md_html(f"""
    <div class="mm-btc-row">
      <div class="mm-mini-card" data-mood style="--mood-color:{MOOD_COLOR};">
        <div class="mm-mini-label">Previous Close</div>
        <div class="mm-mini-sep"></div>
        <div class="mm-mini-value">{v1}</div>
        <div class="mm-mini-sub">USD</div>
      </div>

      <div class="mm-mini-card" data-mood style="--mood-color:{MOOD_COLOR};">
        <div class="mm-mini-label">Daily Change</div>
        <div class="mm-mini-sep"></div>
        <div class="mm-mini-value">{v2}</div>
      </div>

      <div class="mm-mini-card" data-mood style="--mood-color:{MOOD_COLOR};">
        <div class="mm-mini-label">Daily Change (%)</div>
        <div class="mm-mini-sep"></div>
        <div class="mm-mini-value">{v3}</div>
      </div>

      <div class="mm-mini-card" data-mood style="--mood-color:{MOOD_COLOR};">
        <div class="mm-mini-label">7D Volatility</div>
        <div class="mm-mini-sep"></div>
        <div class="mm-mini-value">{v4}</div>
        <div class="mm-mini-sub">Std. dev</div>
      </div>
    </div>
    """)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)