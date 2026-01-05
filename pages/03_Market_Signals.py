# pages/03_Market_Signals.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components

from components.styles import inject_styles
from components.header import render_header
from data.loader import load_df


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "df_draft_1209_w.sti.csv"


# ----------------------
# UI helpers
# ----------------------
def spacer(h: int = 32) -> None:
    # negative spacer는 Streamlit 레이아웃상 기대처럼 안 먹는 경우가 많아서 0 이상만 허용
    st.markdown(f"<div style='height:{max(0, int(h))}px'></div>", unsafe_allow_html=True)


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='mm-block-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='mm-block-sub'>{subtitle}</div>", unsafe_allow_html=True)


def _window_df(df: pd.DataFrame, end_ts: pd.Timestamp, days: int = 120) -> pd.DataFrame:
    if end_ts is None or df.empty:
        return df
    start = end_ts - pd.Timedelta(days=days)
    return df.loc[(df.index >= start) & (df.index <= end_ts)]

def safe_to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None

def fmt_compact(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v/1e12:.{decimals}f}T"
    if abs_v >= 1e9:
        return f"{v/1e9:.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{v/1e6:.{decimals}f}M"
    if abs_v >= 1e3:
        return f"{v/1e3:.{decimals}f}K"
    return f"{v:,.0f}"

def fmt_signed_compact(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return ""
    sign = "+" if v > 0 else ""
    # 작은 값은 compact가 이상할 수 있어서 예외
    if abs(v) < 1e3:
        return f"{sign}{v:.{decimals}f}"
    return f"{sign}{fmt_compact(v, decimals=decimals)}"

def get_prev_row(df: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.Series | None:
    upto = df.loc[:anchor_ts]
    if len(upto) < 2:
        return None
    return upto.iloc[-2]

def calc_delta(now: float | None, prev: float | None) -> tuple[float | None, float | None]:
    if now is None or prev is None:
        return None, None
    d = now - prev
    pct = None if prev == 0 else (d / prev) * 100
    return d, pct

def delta_class(d: float | None) -> str:
    if d is None:
        return "flat"
    if d > 0:
        return "up"
    if d < 0:
        return "down"
    return "flat"

def fmt_delta_line(d: float | None, pct: float | None, decimals: int = 2) -> str:
    if d is None or pct is None:
        return ""
    sign = "+" if d > 0 else ""
    # 예: +123.45M (+1.23%)
    return f"{fmt_signed_compact(d, decimals=2)} ({sign}{pct:.{decimals}f}%)"

def fmt_delta_line_small(d: float | None, pct: float | None, decimals: int = 2) -> str:
    # taker ratio / premium처럼 작은 값용
    if d is None or pct is None:
        return ""
    sign = "+" if d > 0 else ""
    return f"{d:+.{decimals}f} ({sign}{pct:.{decimals}f}%)"

def fmt_pct_only(pct: float | None, decimals: int = 2) -> str:
    if pct is None:
        return ""
    return f"{abs(pct):.{decimals}f}%"


# ----------------------
# Metrics: mini chart blocks (Plotly in components.html)
# ----------------------
def mini_chart_block(
    key: str,
    title: str,
    series: pd.Series | None,
    value_fmt: str = "{:,.2f}",
    value_text: str | None = None,      # ✅ 값 텍스트 강제(예: 149.96B)
    delta_text: str = "",               # ✅ 전일대비 텍스트
    delta_cls: str = "flat",            # ✅ up/down/flat
) -> None:
    last_txt = "—"

    if series is None or series.dropna().shape[0] < 2:
        html = f"""
        <div class="ms-card">
          <div class="ms-title">{title}</div>
          <div class="ms-value">{last_txt}</div>
          <div class="ms-chart-placeholder"></div>
        </div>
        """
        components.html(_wrap_ms_card_html(html), height=280, scrolling=False)
        return

    s = series.dropna().sort_index()
    last = s.iloc[-1]

    # ✅ value_text가 들어오면 그걸 우선 사용
    if value_text is not None:
        last_txt = value_text
    else:
        try:
            last_txt = value_fmt.format(last)
        except Exception:
            last_txt = str(last)

    baseline = 0.0 if key == "etf_flow" else float(s.mean())
    is_up = bool(s.iloc[-1] >= s.iloc[0])
    line_color = "#2d6bff" if is_up else "#ffb020"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            line=dict(color=line_color, width=1.2, shape="linear"),
            hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:,.2f}</b><extra></extra>",
        )
    )
    fig.add_hline(
        y=baseline,
        line_width=1,
        line_dash="solid",
        line_color="rgba(255,255,255,0.28)",
    )

    fig.update_xaxes(visible=False, rangeslider_visible=False, rangeselector=dict(visible=False))
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=150,
        margin=dict(l=4, r=4, t=2, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,12,18,0.18)",
        hovermode="x",
        showlegend=False,
    )

    plot_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False},
    )

    delta_html = ""
    if delta_text:
        if delta_cls == "up":
            delta_html = f"<span class='ms-delta up'>▲ {delta_text}</span>"
        elif delta_cls == "down":
            delta_html = f"<span class='ms-delta down'>▼ {delta_text}</span>"
        else:  # flat
            delta_html = f"<span class='ms-delta flat'>{delta_text}</span>"

    html = f"""
    <div class="ms-card {'up' if is_up else 'down'}">
      <div class="ms-title">{title}</div>

      <div class="ms-vrow">
        <div class="ms-value">{last_txt}</div>
        {delta_html}
      </div>

      <div class="ms-chart">{plot_html}</div>
    </div>
    """
    components.html(_wrap_ms_card_html(html), height=280, scrolling=False)

def _wrap_ms_card_html(inner: str) -> str:
    # f-string으로 안전하게 (format/중괄호 충돌 방지)
    return f"""
    <div class="ms-root">
      <style>
        .ms-root {{
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Apple SD Gothic Neo", "Noto Sans KR", Arial;
        }}

        .ms-card {{
          border-radius: 18px;
          padding: 18px 18px 16px 18px;
          background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
          border: 1px solid rgba(255,255,255,0.18);
          box-shadow:
            0 0 0 1px rgba(255,255,255,0.06) inset,
            0 22px 60px rgba(0,0,0,0.45);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
          overflow: hidden;
        }}
        .ms-card:hover {{
          transform: translateY(-2px);
          border-color: rgba(255,255,255,0.30);
          box-shadow:
            0 0 0 1px rgba(255,255,255,0.10) inset,
            0 28px 78px rgba(0,0,0,0.52);
        }}

        .ms-title {{
          font-size: 12px;
          font-weight: 900;
          line-height: 1.1;
          margin: 0 0 6px 0;
          color: rgba(255,255,255,0.92);
        }}

        .ms-vrow{{
          display:flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 10px;
        }}

        .ms-delta{{
          font-size: 12px;
          font-weight: 900;
          line-height: 1;
          white-space: nowrap;
          opacity: .92;
        }}

        .ms-delta.up{{ color: rgba(255,176,32,0.95); }}
        .ms-delta.down{{ color: rgba(45,107,255,0.95); }}
        .ms-delta.flat{{ color: rgba(255,255,255,0.60); }}

        .ms-value {{
          font-size: 20px;
          font-weight: 900;
          line-height: 1.05;
          margin: 0 0 14px 0;
          color: rgba(255,255,255,0.92);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }}

        .ms-chart {{
          margin-top: 4px;
          border-radius: 14px;
          overflow: hidden;
          background: rgba(10,12,18,0.18);
          min-height: 160px;
        }}

        .ms-chart .plot-container,
        .ms-chart .js-plotly-plot {{
          width: 100% !important;
        }}

        .ms-chart .js-plotly-plot,
        .ms-chart .plot-container,
        .ms-chart .svg-container {{
          border-radius: 14px !important;
          overflow: hidden !important;
          background: transparent !important;
        }}
        .ms-chart svg {{
          background: transparent !important;
        }}

        .ms-card.up {{
          border-color: rgba(45,107,255,0.34);
          box-shadow:
            0 0 0 1px rgba(45,107,255,0.10) inset,
            0 22px 60px rgba(0,0,0,0.45);
        }}
        .ms-card.down {{
          border-color: rgba(255,176,32,0.34);
          box-shadow:
            0 0 0 1px rgba(255,176,32,0.10) inset,
            0 22px 60px rgba(0,0,0,0.45);
        }}

        .ms-chart-placeholder {{
          height: 150px;
          background: rgba(10,12,18,0.18);
          border-radius: 12px;
          margin-top: 2px;
        }}

        .ms-delta{{
          margin-top: 6px;
          font-size: 12px;
          font-weight: 800;
          opacity: .9;
        }}
        .ms-delta.up{{ color: rgba(255,176,32,0.95); }}
        .ms-delta.down{{ color: rgba(45,107,255,0.95); }}
        .ms-delta.flat{{ color: rgba(255,255,255,0.60); }}
      </style>

      {inner}
    </div>
    """

# ----------------------
# Futures Market Snapshot helpers
# ----------------------
def _fmt_ratio(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.2f}"


def _fmt_pct(x) -> str:
    """x is already percent units (e.g. 0.38 means 0.38%)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_usd_full(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${float(x):,.0f}"


def _inject_fs_styles_once() -> None:
    st.markdown(
        """
<style>
:root{
  --mood-cool:    rgba(80, 200, 180, 0.95);
  --mood-neutral: rgba(255, 255, 255, 0.60);
  --mood-warm:    rgba(255, 170, 80, 0.95);

  --tag-bg-cool:    rgba(80, 200, 180, 0.14);
  --tag-bg-neutral: rgba(255, 255, 255, 0.10);
  --tag-bg-warm:    rgba(255, 170, 80, 0.14);

  --tag-bd-cool:    rgba(80, 200, 180, 0.35);
  --tag-bd-neutral: rgba(255, 255, 255, 0.25);
  --tag-bd-warm:    rgba(255, 170, 80, 0.35);
}

/* Grid */
.fs-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 12px;
}

/* Boxes */
.fs-box{
  padding: 14px 14px;
  border-radius: 18px;
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px);
  min-height: 96px;
  transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
  transform: translateZ(0);
}
.fs-box:hover{
  background: rgba(255,255,255,0.065);
  border-color: rgba(255,255,255,0.13);
  transform: translateY(-2px);
}

.fs-k{
  font-size:12px;
  font-weight:750;
  letter-spacing:-0.01em;
  color: rgba(255,255,255,0.72);
  line-height:1.2;
}

.fs-vrow{
  margin-top: 8px;
  display:flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.fs-v{
  font-size: 28px;
  font-weight: 850;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,0.92);
  line-height: 1.0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Tags */
.tag{
  display:inline-flex;
  align-items:center;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.015em;
  line-height: 1;
  backdrop-filter: blur(6px);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset;
  white-space: nowrap;
  flex: 0 0 auto;
  opacity: 0.95;   /* 기본이 0.85~0.9라면 살짝 또렷 */
}
.tag.cool{ color: var(--mood-cool); background: var(--tag-bg-cool); border:1px solid var(--tag-bd-cool); }
.tag.neutral{ color: var(--mood-neutral); background: var(--tag-bg-neutral); border:1px solid var(--tag-bd-neutral); }
.tag.warm{ color: var(--mood-warm); background: var(--tag-bg-warm); border:1px solid var(--tag-bd-warm); }

/* Futures 아래 바닥 확보 */
.fs-bottom-space{ height: 30px; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ---- tag rules ----
def _tag_oi(oi_chg_pct_24h: float | None) -> tuple[str, str]:
    if oi_chg_pct_24h is None or (isinstance(oi_chg_pct_24h, float) and np.isnan(oi_chg_pct_24h)):
        return ("OI Flat", "neutral")
    if oi_chg_pct_24h > 1.0:
        return ("OI Expanding", "cool")
    if oi_chg_pct_24h < -1.0:
        return ("OI Deleveraging", "warm")
    return ("OI Flat", "neutral")


def _tag_taker(tr: float | None) -> tuple[str, str]:
    if tr is None or (isinstance(tr, float) and np.isnan(tr)):
        return ("Flow Balanced", "neutral")
    if tr >= 1.05:
        return ("Buy-side Aggressive", "cool")
    if tr <= 0.95:
        return ("Sell-side Aggressive", "warm")
    return ("Flow Balanced", "neutral")


def _tag_liq(total_usd: float | None, long_usd: float | None, short_usd: float | None) -> tuple[str, str]:
    if total_usd is None or (isinstance(total_usd, float) and np.isnan(total_usd)):
        return ("Liquidation Light", "neutral")

    total = float(total_usd)
    if total < 50e6:
        return ("Liquidation Light", "neutral")

    lv = 0.0 if long_usd is None or (isinstance(long_usd, float) and np.isnan(long_usd)) else float(long_usd)
    sv = 0.0 if short_usd is None or (isinstance(short_usd, float) and np.isnan(short_usd)) else float(short_usd)

    if lv > sv * 1.3:
        return ("Longs Flushed", "warm")
    if sv > lv * 1.3:
        return ("Shorts Squeezed", "cool")
    return ("Liquidation Light", "neutral")


def _tag_premium(pct: float | None) -> tuple[str, str]:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return ("Premium Neutral", "neutral")
    v = float(pct)
    if v >= 0.10:
        return ("US Spot Bid", "cool")
    if v <= -0.10:
        return ("US Demand Fading", "warm")
    return ("Premium Neutral", "neutral")


def render_futures_market_snapshot(df: pd.DataFrame, row: pd.Series) -> None:
    _inject_fs_styles_once()

    COL_OI    = "oi_close"
    COL_TAKER = "taker_buy_ratio"
    COL_LIQ_T = "liq_total_usd"
    COL_LIQ_L = "liq_long_usd"
    COL_LIQ_S = "liq_short_usd"
    COL_PREM  = "coinbase_premium_rate"

    oi    = row.get(COL_OI)
    taker = row.get(COL_TAKER)
    liq_t = row.get(COL_LIQ_T)
    liq_l = row.get(COL_LIQ_L)
    liq_s = row.get(COL_LIQ_S)
    prem  = row.get(COL_PREM)

    # OI 전일 대비(%)
    oi_chg_pct_24h = None
    try:
        if row.name in df.index:
            pos = df.index.get_loc(row.name)
            if isinstance(pos, int) and pos > 0:
                prev = df.iloc[pos - 1].get(COL_OI)
                if prev is not None and float(prev) != 0:
                    oi_chg_pct_24h = (float(oi) / float(prev) - 1) * 100
    except Exception:
        pass

    oi_tag, oi_tone = _tag_oi(oi_chg_pct_24h)
    tr_tag, tr_tone = _tag_taker(taker)
    lq_tag, lq_tone = _tag_liq(liq_t, liq_l, liq_s)
    cp_tag, cp_tone = _tag_premium(prem)

    # ---- 전일 row ----
    pos = None
    try:
        pos = df.index.get_loc(row.name)
    except Exception:
        pos = None

    prev = None
    if isinstance(pos, int) and pos > 0:
        prev = df.iloc[pos - 1]

    oi_prev = safe_to_float(prev.get(COL_OI)) if prev is not None else None
    taker_prev = safe_to_float(prev.get(COL_TAKER)) if prev is not None else None
    liq_prev = safe_to_float(prev.get(COL_LIQ_T)) if prev is not None else None
    prem_prev = safe_to_float(prev.get(COL_PREM)) if prev is not None else None

    oi_now = safe_to_float(oi)
    taker_now = safe_to_float(taker)
    liq_now = safe_to_float(liq_t)
    prem_now = safe_to_float(prem)

    oi_d, oi_pct = calc_delta(oi_now, oi_prev)
    taker_d, taker_pct = calc_delta(taker_now, taker_prev)
    liq_d, liq_pct = calc_delta(liq_now, liq_prev)
    prem_d, prem_pct = calc_delta(prem_now, prem_prev)

    oi_delta = fmt_delta_line(oi_d, oi_pct)
    liq_delta = fmt_delta_line(liq_d, liq_pct)

    # taker/premium은 값이 작아서 small 추천
    taker_delta = fmt_delta_line_small(taker_d, taker_pct, decimals=2)
    prem_delta = fmt_delta_line_small(prem_d, prem_pct, decimals=2)

    oi_cls = delta_class(oi_d)
    taker_cls = delta_class(taker_d)
    liq_cls = delta_class(liq_d)
    prem_cls = delta_class(prem_d)


    with st.container(border=True):
        # ✅ 마커 통일 (컨테이너 wrapper 잡는 용도)
        st.markdown("<div data-ms='futures'></div>", unsafe_allow_html=True)

        section_title(
            "Futures Market Snapshot",
            "파생·현물 수급 지표를 통해 현재 시장의 포지션 구조와 레버리지 상태를 요약합니다.",
        )
        spacer(36)

        st.markdown(
            f"""
<div class="fs-grid">
  <div class="fs-box">
    <div class="fs-k">OPEN INTEREST</div>
    <div class="fs-vrow">
      <div class="fs-v">${fmt_compact(safe_to_float(oi), decimals=2)}</div>
      <span class="tag {oi_tone}">{oi_tag}</span>
    </div>
    <div class="fs-delta {oi_cls}">{oi_delta}</div>
  </div>

  <div class="fs-box">
    <div class="fs-k">TAKER RATIO</div>
    <div class="fs-vrow">
      <div class="fs-v">{_fmt_ratio(taker)}</div>
      <span class="tag {tr_tone}">{tr_tag}</span>
    </div>
    <div class="fs-delta {taker_cls}">{taker_delta}</div>
  </div>

  <div class="fs-box">
    <div class="fs-k">LIQUIDATION (24H)</div>
    <div class="fs-vrow">
      <div class="fs-v">${fmt_compact(safe_to_float(liq_t), decimals=2)}</div>
      <span class="tag {lq_tone}">{lq_tag}</span>
    </div>
    <div class="fs-delta {liq_cls}">{liq_delta}</div>
  </div>

  <div class="fs-box">
    <div class="fs-k">COINBASE PREMIUM</div>
    <div class="fs-vrow">
      <div class="fs-v">{_fmt_pct(prem)}</div>
      <span class="tag {cp_tone}">{cp_tag}</span>
    </div>
    <div class="fs-delta {prem_cls}">{prem_delta}</div>
  </div>
</div>
<div class="fs-bottom-space"></div>
            """,
            unsafe_allow_html=True,
        )


# ======================
# Page
# ======================
inject_styles()

# ✅ 페이지 스코프 마커
st.markdown("<span class='ms-page'></span>", unsafe_allow_html=True)

# ✅ Market Mood 방식: wrapper margin/padding으로 간격 제어 (확실히 먹음)
st.markdown(
    """
<style>
/* 1) 섹션(컨테이너) 프레임 간 간격: Metrics 아래를 줄이고, Futures를 위로 당김 */
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="metrics"]){
  margin-bottom: 8px !important;      /* <- 필요하면 0~12 사이로 */
}
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="futures"]){
  margin-top: -10px !important;       /* <- -6 ~ -18 사이로 취향 */
}

/* 2) Futures 섹션의 '바닥'은 wrapper 내부 padding으로 확보 (간격 줄여도 바닥 유지) */
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="futures"]) > div{
  padding-bottom: 2px !important;    /* <- 16~28 */
}

/* 3) 혹시 전체 vertical gap을 건드리고 싶으면 "살짝만" (과하면 바닥 사라짐)
:has(.ms-page) [data-testid="stVerticalBlock"]{ gap: 4px !important; }
*/

/* Delta line (Futures) */
.fs-delta{ margin-top: 6px; font-size: 12px; font-weight: 800; opacity: .9; }
.fs-delta.up{ color: rgba(255,176,32,0.95); }
.fs-delta.down{ color: rgba(45,107,255,0.95); }
.fs-delta.flat{ color: rgba(255,255,255,0.60); }

</style>
""",
    unsafe_allow_html=True,
)

df = load_df(DATA_PATH)

anchor_ts, row = render_header(
    df=df,
    title="Market Signals",
    subtitle="주요 시장 지표를 구조적으로 확인합니다.",
    date_key="ms_date",
)

prev_row = get_prev_row(df, row.name)  # df는 KST index인 df

# (헤더 아래 여백)
spacer(16)

COL_ETF_FLOW = "etf_flow_usd"
COL_ETF_AUM  = "etf_aum_usd"
COL_FFR      = "ffr"
COL_SP500    = "sp500"


# ----------------------
# Metrics section
# ----------------------
with st.container(border=True):
    # ✅ 마커 통일
    st.markdown("<div data-ms='metrics'></div>", unsafe_allow_html=True)

    section_title("Metrics", "자금 흐름과 매크로 환경의 변화를 미니 차트로 보여줍니다.")
    spacer(2)

    st.markdown('<div class="ms-cards-row">', unsafe_allow_html=True)

    w = _window_df(df, anchor_ts, days=120)

    # ---- 전일대비(delta) 계산용 ----
    aum_now = safe_to_float(row.get(COL_ETF_AUM))
    aum_prev = safe_to_float(prev_row.get(COL_ETF_AUM)) if prev_row is not None else None
    aum_d, aum_pct = calc_delta(aum_now, aum_prev)

    sp_now = safe_to_float(row.get(COL_SP500))
    sp_prev = safe_to_float(prev_row.get(COL_SP500)) if prev_row is not None else None
    sp_d, sp_pct = calc_delta(sp_now, sp_prev)

    aum_delta_text = fmt_pct_only(aum_pct)
    sp_delta_text  = fmt_pct_only(sp_pct)

    aum_delta_cls = delta_class(aum_d)
    sp_delta_cls = delta_class(sp_d)


    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        mini_chart_block("etf_flow", "ETF Flow", w.get(COL_ETF_FLOW), "{:,.0f}")
    with c2:
        mini_chart_block(
            "etf_aum",
            "ETF AUM 변화량",
            w.get(COL_ETF_AUM),
            "{:,.0f}",
            value_text=fmt_compact(aum_now, decimals=2),   # ✅ B 표기
            delta_text=aum_delta_text,                    # ✅ 전일대비
            delta_cls=aum_delta_cls,
        )
    with c3:
        mini_chart_block("ffr", "FFR 기준금리", w.get(COL_FFR), "{:.2f}%")
    with c4:
        mini_chart_block(
            "sp500",
            "S&P 500 (Tech)",
            w.get(COL_SP500),
            "{:,.2f}",
            value_text=(f"{sp_now:,.2f}" if sp_now is not None else "—"),
            delta_text=sp_delta_text,
            delta_cls=sp_delta_cls,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------
# Futures section
# ----------------------
render_futures_market_snapshot(df, row)
