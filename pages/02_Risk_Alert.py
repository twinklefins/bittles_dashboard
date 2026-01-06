# pages/02_Risk_Alert.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import html as _html

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from components.styles import inject_styles
from data.loader import load_df
from components.header import render_header


# ======================
# Config
# ======================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "df_final_dashboard.csv"
KST = "Asia/Seoul"
DATA_PATH_7D = ROOT_DIR / "data" / "band_updown_dash_f.csv"

# ======================
# Palette
# ======================
PALETTE = {
    "BLUE":   "#2D6BFF",
    "CYAN":   "#19D3FF",
    "YELLOW": "#FFB020",
    "ORANGE": "#FF6B00",
    "RED":    "#FF2D55",
}

LEVEL_COLOR = {
    "GREEN":  PALETTE["CYAN"],
    "YELLOW": PALETTE["YELLOW"],
    "RED":    PALETTE["RED"],
}


# ======================
# Helpers
# ======================
def _split_bullets(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    if isinstance(s, float) and pd.isna(s):
        return []
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    out: List[str] = []
    for ln in lines:
        if ln.startswith("-"):
            ln = ln.lstrip("-").strip()
        out.append(ln)
    return out


def _safe_str(x, fallback: str = "—") -> str:
    if x is None:
        return fallback
    if isinstance(x, float) and pd.isna(x):
        return fallback
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return fallback
    return s


def _format_mlprob(v) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        fv = float(v)
        if pd.isna(fv) or fv == 0:
            return "—"
        return f"{fv:.3f}"
    except Exception:
        return "—"


def _to_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def level_theme(level: str) -> dict:
    lvl = _safe_str(level, "N/A").upper()
    c = LEVEL_COLOR.get(lvl, PALETTE["CYAN"])
    return {
        "lvl": lvl,
        "color": c,
        "border": f"1px solid {c}33",
        "glow":   f"0 0 0 1px {c}22 inset, 0 0 28px {c}18",
    }


def _st_html(s: str) -> None:
    cleaned = "\n".join(line.lstrip() for line in s.splitlines()).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def _format_prob_pct(v) -> str:
    """0.363 -> 36.3%"""
    try:
        fv = float(v)
        if pd.isna(fv) or fv == 0:
            return "N/A"
        return f"{fv*100:.1f}%"
    except Exception:
        return "N/A"


def _delta_pp(curr: Optional[float], prev: Optional[float]) -> str:
    """확률 변화량을 %p로: (0.363-0.379)=-0.016 -> -1.6%p"""
    if curr is None or prev is None:
        return "<span class='ra-snap-delta ra-snap-na'>Δ N/A</span>"

    d = curr - prev
    if abs(d) < 1e-12:
        return "<span class='ra-snap-delta ra-snap-flat'>• 0.0%p</span>"

    up = d > 0
    cls = "ra-snap-up" if up else "ra-snap-down"
    arrow = "▲" if up else "▼"   # ✅ 속이 찬 화살표
    return f"<span class='ra-snap-delta {cls}'>{arrow} {abs(d)*100:.1f}%p</span>"


def _bullets_block(title: str, lines: List[str], max_n: int = 3) -> str:
    if not lines:
        body = "<div class='ra-empty'>—</div>"
    else:
        lis = "".join([f"<li>{_html.escape(str(ln))}</li>" for ln in lines[:max_n]])
        body = f"<ul class='ra-ul'>{lis}</ul>"

    return f"""
<div class="ra-body-card">
  <div class="ra-card-h">{_html.escape(title)}</div>
  {body}
</div>
"""


def _pick_prev_row(df: pd.DataFrame, sel_date: pd.Timestamp) -> Optional[pd.Series]:
    dates = sorted(pd.Series(df.index.date).unique())
    try:
        idx = dates.index(sel_date.date())
    except ValueError:
        return None
    if idx <= 0:
        return None
    prev_date = dates[idx - 1]
    prev_df = df[df.index.date == prev_date]
    if prev_df.empty:
        return None
    return prev_df.iloc[-1]


@st.cache_data(show_spinner=False)
def load_df_7d(path: Path) -> pd.DataFrame:
    df7 = pd.read_csv(path)
    df7["date"] = pd.to_datetime(df7["date"])
    df7 = df7.sort_values("date")
    return df7


def _fmt_pct(v, digits: int = 1) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.{digits}f}%"
    except Exception:
        return "—"


def _flag_badge(flag: int, kind: str) -> str:
    try:
        f = int(flag)
    except Exception:
        f = 0

    if kind == "up":
        labels = {0: "Neutral", 1: "Opportunity", 2: "Strong Up"}
        cls_map = {0: "ra-flag-neutral", 1: "ra-flag-up1", 2: "ra-flag-up2"}
    else:
        labels = {0: "Neutral", 1: "Defense On", 2: "High Risk"}
        cls_map = {0: "ra-flag-neutral", 1: "ra-flag-down1", 2: "ra-flag-down2"}

    cls = cls_map.get(f, "ra-flag-neutral")
    return f"<span class='ra-flag {cls}'>{labels.get(f,'Neutral')}</span>"


def _inject_ra_css() -> None:
    _st_html("""
<style>
.ra-page{ display:none; }

/* =========================
   HERO
========================= */
.ra-hero{
  padding:18px 18px 14px 18px;
  border-radius:16px;
  background: rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.10);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset;
}
.ra-hero-top{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
}
.ra-hero-title{
  font-size:15px;
  font-weight:900;
  letter-spacing:-0.01em;
  color: rgba(255,255,255,.92);
}
.ra-hero-sub{
  margin-top:4px;
  font-size:12px;
  opacity:.70;
  line-height:1.45;
}
.ra-headline{
  margin-top:10px;
  margin-bottom:12px;
  font-size:30px;
  font-weight:950;
  letter-spacing:-0.02em;
  line-height:1.15;
}

/* =========================
   PILLS
========================= */
.ra-pill-row{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  justify-content:flex-end;
  margin-top: 6px;
}
.ra-pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:6px 12px;
  border-radius:999px;
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  white-space:nowrap;
  font-weight:900;
}
.ra-dot{ width:10px; height:10px; border-radius:50%; }
.ra-pill-strong{ font-weight:950; }
.ra-pill-sep{ opacity:.45; font-weight:800; padding:0 2px; }
.ra-pill-muted{ opacity:.78; font-weight:850; }
.ra-pill-final{ padding: 7px 14px; }
.ra-pill-final .ra-pill-muted{ opacity:.72; }
.ra-pill-final .ra-pill-kpi{ font-weight:950; }

/* =========================
   SNAPSHOT
========================= */
.ra-snap{
  margin-top:8px;
  font-size:12px;
  opacity:.70;
  display:flex;
  align-items:center;
  gap:10px;
  flex-wrap:wrap;
}
.ra-snap-kpi{ font-weight:900; opacity:.92; }
.ra-snap-delta{
  display:inline-flex;
  align-items:center;
  padding:2px 8px;
  border-radius:999px;
  font-size:12px;
  font-weight:900;
  letter-spacing: -0.01em;       /* 살짝 조이기 */
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  border:1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.55);
  opacity:.92;
}
.ra-snap-up{ color: rgba(255,176,32,.95); }
.ra-snap-down{ color: rgba(45,107,255,.95); }
.ra-snap-flat{ color: rgba(255,255,255,.65); }
.ra-snap-na{ color: rgba(255,255,255,.55); }

/* =========================
   BULLET CARDS
========================= */
.ra-card-h{
  font-size: 14px;
  font-weight: 900;
  letter-spacing: -0.01em;
  margin-bottom: 10px;
  color: rgba(255,255,255,.92);
}
.ra-ul{
  margin: 0;
  padding-left: 18px;
  color: rgba(255,255,255,.78);
  line-height: 1.55;
}
.ra-ul li{ margin: 10px 0; }
.ra-empty{ opacity: .55; }
.ra-body-card{ padding-bottom: 8px; }

/* =========================
   EXPANDER
========================= */
div[data-testid="stExpander"]{
  margin-top: 6px !important;
  margin-bottom: 12px !important;
}

/* =========================
   BADGES / MINI
========================= */
.ra-flag{
  display:inline-flex;
  align-items:center;
  padding:3px 10px;
  border-radius:999px;
  font-size:11px;
  font-weight:900;
  border:1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.05);
  opacity:.92;
  white-space: nowrap !important;
  line-height: 1 !important;
  height: 22px;
  flex: 0 0 auto;             /* ✅ 배지가 밀려 아래로 내려가는 것 방지 */
}
.ra-flag-neutral{ color: rgba(255,255,255,.70); }
.ra-flag-up1{ color: rgba(25,211,255,.95); }
.ra-flag-up2{ color: rgba(25,211,255,1.0); box-shadow: 0 0 0 1px rgba(25,211,255,.25) inset; }
.ra-flag-down1{ color: rgba(255,176,32,.95); }
.ra-flag-down2{ color: rgba(255,45,85,.95); box-shadow: 0 0 0 1px rgba(255,45,85,.22) inset; }

.ra-mini{
  margin-top: 8px;
  font-size: 12px;
  opacity: .75;
  line-height: 1.5;
}

/* =========================
   ✅ Direction Wrapper (통 glow는 여기만!)
========================= */
.ra-7d-wrap{
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 24px 18px 44px 18px;
  margin-top: 10px;
}
.ra-7d-title{
  font-size:15px;
  font-weight:900;
  letter-spacing:-0.01em;
  color: rgba(255,255,255,.92);
  margin: 0;
}
.ra-7d-sub{
  margin-top:4px;
  margin-bottom: 14px;
  font-size:12px;
  line-height:1.45;
  color: rgba(255,255,255,.68);
}

/* ✅ 3카드 높이 유동 유지 */
.ra-7d-grid{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  align-items: start;     /* ✅ stretch 금지 */
  margin-top: 12px;
}

/* 카드(개별 글로우 X) */
.ra-7d-card{
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: none !important;
}
.ra-7d-card-inner{
  padding: 14px;
}

/* =========================
   KPI tiles
========================= */
.ra-kpi-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
  margin-top:10px;
  margin-bottom:10px;
}
.ra-kpi-tile{
  padding:12px 12px;
  border-radius:14px;
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
}
.ra-kpi-tile.big{
  grid-column: 1 / span 2;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:14px 14px;
}
.ra-kpi-label{
  font-size:12px;
  font-weight:850;
  color: rgba(255,255,255,.70);
}
.ra-kpi-val{
  margin-top:4px;
  font-size:22px;
  font-weight:950;
  color: rgba(255,255,255,.92);
}
.ra-kpi-tile.big .ra-kpi-val{ font-size:26px; }

.ra-kpi-full{
  padding:14px 14px;
  border-radius:14px;
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  width:100%;
  margin-top:10px;
  margin-bottom:10px;
}
.ra-kpi-full .ra-kpi-val{ font-size:26px; }

/* =========================
   ✅ 핵심: "KPI 박스들만" 하단 기준선 통일
   - 전체 카드/영역 고정 높이 X
   - 타일에 min-height만 부여해서
     Up/Down 박스가 Band80/q50 타일과 '비슷한 높이'로 정렬됨
========================= */
:root{
  --ra-kpi-tile-h: 80px;  /* ✅ 여기만 조절하면 됨 (120~145 추천) */
}
.ra-kpi-tile.big{ min-height: var(--ra-kpi-tile-h); }
.ra-kpi-full{ min-height: var(--ra-kpi-tile-h); }

/* =========================
   Card2 split (Up/Down) — SAFE RESET
========================= */
.ra-split2{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
  margin-top:10px;
  margin-bottom:10px;
  align-items: stretch;
}

/* box는 무조건 flex column */
.ra-split-box{
  padding:12px;
  border-radius:14px;
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);

  /* ✅ 타일 높이 컨벤션에 맞춤 */
  min-height: var(--ra-kpi-tile-h);
  display:flex;
  flex-direction:column;
}

/* 헤더 한 줄 고정 */
.ra-split-top{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  flex-wrap: nowrap;
  min-height: 26px;
}

.ra-split-kicker{
  font-size:12px;
  font-weight:850;
  opacity:.72;
  white-space: nowrap;
}

/* ✅ 미니차트(= bar)는 아래쪽으로 */
.ra-split-bar{
  margin-top: auto;      /* 남는 공간을 위로 밀어서 아래로 */
  margin-bottom: 2px;    /* 바닥에 너무 붙지 않게 */
  min-height: 32px;

  font-size:18px;        /* 18이 커 보이면 16이 안정적 */
  font-weight:950;
  line-height:1.05;
  opacity:.92;
}

/* 설명 */
.ra-split-desc{
  margin-top: 6px;
  font-size:12px;
  line-height:1.55;
  opacity:.80;
}

.ra-split-up.active{
  border-color: rgba(25,211,255,.30);
  box-shadow: 0 0 0 1px rgba(25,211,255,.12) inset;
}
.ra-split-down.active{
  border-color: rgba(255,45,85,.30);
  box-shadow: 0 0 0 1px rgba(255,45,85,.10) inset;
}
             
.ra-split-bar{ padding-top: 14px; }

</style>
""")


# ======================
# UI
# ======================
def hero_banner(
    level: str,
    driver: str,
    headline: str,
    as_of: str,
    ml_prob,
    prev_ml_prob=None,
) -> None:
    t = level_theme(level)
    drv = _safe_str(driver, "N/A")
    head = _safe_str(headline, "해당 날짜의 Risk Alert 메시지가 아직 생성되지 않았습니다.")

    curr_prob_f = _to_float_or_none(ml_prob)
    prev_prob_f = _to_float_or_none(prev_ml_prob)

    prob_pct = _format_prob_pct(ml_prob)
    prob_delta_pp = _delta_pp(curr_prob_f, prev_prob_f)

    html = f"""
<div class="ra-hero"
     style="border:{t['border']}; box-shadow:{t['glow']};">
  <div class="ra-hero-top">
    <div>
      <div class="ra-hero-title">Risk Alert (Crash_3)</div>
      <div class="ra-hero-sub">향후 10일 내 급락(Downside) 리스크 확률과 경고 메시지를 요약합니다.</div>
    </div>

    <div class="ra-pill-row">
      <div class="ra-pill ra-pill-final">
        <span class="ra-dot" style="background:{t["color"]};"></span>
        <span class="ra-pill-strong">Final: {t["lvl"]}</span>
        <span class="ra-pill-sep">|</span>
        <span class="ra-pill-muted">driver: {drv}</span>
      </div>
    </div>
  </div>

  <div class="ra-headline">{_html.escape(head)}</div>

  <div class="ra-snap">
    <span>ML Prob(10D) <span class="ra-snap-kpi">{_html.escape(prob_pct)}</span></span>
    {prob_delta_pp}
    <span style="opacity:.65;">vs prev day</span>
  </div>
</div>
"""
    _st_html(html)


def render_risk_alert_main(
    *,
    final_level: str,
    driver: str,
    headline: str,
    summary_lines: List[str],
    action_lines: List[str],
    detail_lines: List[str],
    sel_date_str: str,
    ml_prob,
    prev_ml_prob=None,
) -> None:
    hero_banner(
        final_level,
        driver,
        headline,
        as_of=sel_date_str,
        ml_prob=ml_prob,
        prev_ml_prob=prev_ml_prob,
    )

    st.markdown("")

    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            _st_html(_bullets_block("근거 요약", summary_lines, max_n=3))
    with b2:
        with st.container(border=True):
            _st_html(_bullets_block("운영 액션", action_lines, max_n=3))

    with st.expander("상세 근거 보기", expanded=False):
        if detail_lines:
            for ln in detail_lines:
                st.markdown(f"- {ln}")
        else:
            st.markdown("- (상세 근거 문구가 아직 없습니다)")


# ======================
# Run
# ======================
inject_styles()
_st_html("<span class='ra-page'></span>")
_inject_ra_css()

df_utc = load_df(DATA_PATH)
if df_utc.empty:
    st.stop()

df = df_utc.copy()
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert(KST)

anchor_ts, row = render_header(
    df,
    title="Risk Alert",
    subtitle="시장 리스크 신호를 종합해 날짜별로 핵심만 요약합니다.",
    date_key="ra_date",
)

sel_date_str = pd.Timestamp(anchor_ts).strftime("%Y-%m-%d")
prev_row = _pick_prev_row(df, pd.Timestamp(anchor_ts))

final_level = row.get("final_risk_level", None)
driver = row.get("final_driver", None)
headline = row.get("msg_headline", None)

summary_lines = _split_bullets(row.get("msg_summary", None))
action_lines = _split_bullets(row.get("msg_action", None))
detail_lines = _split_bullets(row.get("msg_details", None))

ml_prob = row.get("ml_10d_prob", None)
prev_ml_prob = prev_row.get("ml_10d_prob", None) if prev_row is not None else None

render_risk_alert_main(
    final_level=_safe_str(final_level, "N/A"),
    driver=_safe_str(driver, "N/A"),
    headline=_safe_str(headline, "해당 날짜의 Risk Alert 메시지가 아직 생성되지 않았습니다."),
    summary_lines=summary_lines,
    action_lines=action_lines,
    detail_lines=detail_lines,
    sel_date_str=sel_date_str,
    ml_prob=ml_prob,
    prev_ml_prob=prev_ml_prob,
)

# ======================
# 7D Direction
# ======================
df7 = load_df_7d(DATA_PATH_7D)
sel_dt = pd.to_datetime(sel_date_str)
row7 = df7.loc[df7["date"] == sel_dt]

if row7.empty:
    st.caption("선택 날짜에 대한 7D 데이터가 없습니다.")
else:
    r = row7.iloc[0]
    t_final = level_theme(_safe_str(final_level, "N/A"))

    # 카드 1
    card1 = f"""
    <div class="ra-7d-card">
      <div class="ra-7d-card-inner">
        <div class="ra-card-h">7일 뒤 수익전망</div>

        <div class="ra-kpi-grid">
          <div class="ra-kpi-tile big">
            <div>
              <div class="ra-kpi-label">중앙값(q50)</div>
              <div class="ra-kpi-val">{_fmt_pct(r.get("num_1_card1_q50_center_pct"), 1)}</div>
            </div>
          </div>

          <div class="ra-kpi-tile">
            <div class="ra-kpi-label">상방(q90)</div>
            <div class="ra-kpi-val">{_fmt_pct(r.get("num_1_card1_q90_upper_pct"), 1)}</div>
          </div>

          <div class="ra-kpi-tile">
            <div class="ra-kpi-label">하방(q10)</div>
            <div class="ra-kpi-val">{_fmt_pct(r.get("num_1_card1_q10_lower_pct"), 1)}</div>
          </div>
        </div>

        <div class="ra-mini">{_html.escape(_safe_str(r.get("txt_1_card1_main"), "—"))}</div>
      </div>
    </div>
    """

    # 카드 2
    up_flag = r.get("num_2_card2_flag_up", 0)
    dn_flag = r.get("num_2_card2_flag_down", 0)
    up_bar = _safe_str(r.get("txt_2_card2_bar_up"), "—")
    dn_bar = _safe_str(r.get("txt_2_card2_bar_down"), "—")
    up_cls = "active" if int(up_flag) == 1 else ""
    dn_cls = "active" if int(dn_flag) == 1 else ""

    card2 = f"""
    <div class="ra-7d-card">
      <div class="ra-7d-card-inner">
        <div class="ra-card-h">{_html.escape(_safe_str(r.get("txt_2_card2_title"), "상·하위 20% 수익 이벤트 지표"))}</div>

        <div class="ra-split2">
          <div class="ra-split-box ra-split-up {up_cls}">
            <div class="ra-split-top">
              <div class="ra-split-kicker">Up20</div>
              {_flag_badge(up_flag, "up")}
            </div>
            <div class="ra-split-spacer"></div>
            <div class="ra-split-bar">{_html.escape(up_bar)}</div>
            <div class="ra-split-desc">{_html.escape(_safe_str(r.get("txt_2_card2_up_desc"), ""))}</div>
          </div>

          <div class="ra-split-box ra-split-down {dn_cls}">
            <div class="ra-split-top">
              <div class="ra-split-kicker">Down20</div>
              {_flag_badge(dn_flag, "down")}
            </div>
            <div class="ra-split-spacer"></div>
            <div class="ra-split-bar">{_html.escape(dn_bar)}</div>
            <div class="ra-split-desc">{_html.escape(_safe_str(r.get("txt_2_card2_down_desc"), ""))}</div>
          </div>
        </div>

        <div class="ra-mini">{_html.escape(_safe_str(r.get("txt_2_card2_summary"), "—"))}</div>
      </div>
    </div>
    """

    # 카드 3
    card3 = f"""
    <div class="ra-7d-card">
      <div class="ra-7d-card-inner">
        <div class="ra-card-h">시장 예측 변동성</div>

        <div class="ra-kpi-full">
          <div class="ra-kpi-label">Band80</div>
          <div class="ra-kpi-val">{_fmt_pct(r.get("num_3_card3_band80_val_pct"), 1)}</div>
        </div>

        <div class="ra-mini">{_html.escape(_safe_str(r.get("txt_3_card3_band80_rank"), "—"))}</div>
        <div class="ra-mini" style="margin-top:10px;">{_html.escape(_safe_str(r.get("txt_3_card3_main"), "—"))}</div>
        <div class="ra-mini" style="opacity:.70;">{_html.escape(_safe_str(r.get("txt_3_card3_extra"), ""))}</div>
      </div>
    </div>
    """

    _st_html(f"""
    <div class="ra-7d-wrap" style="border:{t_final['border']}; box-shadow:{t_final['glow']};">
      <div class="ra-7d-title">7D Direction</div>
      <div class="ra-7d-sub">향후 7일 수익률 ‘가능 범위(q10/q50/q90)’와 비대칭 리스크 신호(Down20)를 보여줍니다. </div>

      <div class="ra-7d-grid">
        {card1}
        {card2}
        {card3}
      </div>
    </div>
    """)

    # --- Chart section (FULL WIDTH) ---
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    lookback = st.slider(
        "차트 기간(일)",
        min_value=30,
        max_value=365,
        value=90,
        step=10,
        key="ra_7d_lb"
    )

    end_t = sel_dt
    start_t = end_t - pd.Timedelta(days=lookback)
    hist = df7[(df7["date"] >= start_t) & (df7["date"] <= end_t)].copy()

    if hist.empty:
        st.caption("선택 구간 차트 데이터가 없습니다.")
    else:
        down_cut = float(df7["num_0_q10_7d_pct"].quantile(0.20))
        up_cut = float(df7["num_0_q90_7d_pct"].quantile(0.80))

        fig = go.Figure()

        # q10~q90 band (fill 영역)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["num_0_q90_7d_pct"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["num_0_q10_7d_pct"],
            mode="lines",
            fill="tonexty",
            name="q10~q90 band",
            line=dict(width=0),
            fillcolor="rgba(45,107,255,0.18)",  # ✅ BLUE 살짝 투명
        ))

        # q50 (pred)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["num_0_q50_7d_pct"],
            mode="lines",
            name="q50 (pred)",
            line=dict(width=2, color=PALETTE["CYAN"]),  # ✅ CYAN
        ))

        # actual
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["num_0_y_reg_7d_actual_pct"],
            mode="lines",
            name="7D actual",
            line=dict(width=2, color=PALETTE["RED"]),   # ✅ RED (또는 ORANGE)
        ))

        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y", y0=down_cut, y1=down_cut,
            line=dict(dash="dash", width=2),
        )
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y", y0=up_cut, y1=up_cut,
            line=dict(dash="dash", width=2),
        )

        fig.add_trace(go.Scatter(
            x=[sel_dt], y=[float(r["num_0_q50_7d_pct"])],
            mode="markers", name="selected q50",
            marker=dict(size=10),
            hovertemplate="selected<br>%{x|%Y-%m-%d}<br>q50=%{y:.1f}%<extra></extra>"
        ))

        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=44, b=10),  # ⬅️ 위 여백 조금 늘려서 제목 공간 확보
            title=dict(
                text=f"7D Target Prediction vs Actual (lookback {lookback} days)",
                x=0.01, xanchor="left",
                y=0.98, yanchor="top",
                font=dict(size=14)  # 색은 지정 안 해도 테마 따라가고, 필요하면 나중에 조정
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True}
        )
