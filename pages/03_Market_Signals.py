# pages/03_Market_Signals.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from components.styles import inject_styles
from components.header import render_header
from data.loader import load_df


# ======================
# Paths
# ======================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "df_draft_1209_w.sti.csv"


# ======================
# UI helpers
# ======================
def spacer(h: int = 32) -> None:
    st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='mm-block-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='mm-block-sub'>{subtitle}</div>", unsafe_allow_html=True)


def render_signal_tile(name: str, status: str, one_liner: str, meta: str = "") -> None:
    st.markdown(
        f"""
        <div class="mm-mini-card">
          <div style="display:flex; justify-content:space-between; gap:10px; align-items:center;">
            <div style="font-weight:800;">{name}</div>
            <div style="font-weight:900;">{status}</div>
          </div>
          <div style="margin-top:8px; opacity:.70; line-height:1.4;">{one_liner}</div>
          {"<div style='margin-top:8px; opacity:.45; font-size:12px;'>" + meta + "</div>" if meta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_placeholder(title: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="mm-mini-card" style="padding:16px;">
          <div style="font-weight:900; margin-bottom:10px;">{title}</div>
          <div style="opacity:.55; line-height:1.4;">(차트 자리) {note}</div>
          <div style="margin-top:14px; height:180px; border-radius:14px; border:1px dashed rgba(255,255,255,.18);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================
# Page
# ======================
st.set_page_config(page_title="Market Signals", layout="wide")
inject_styles()

df = load_df(DATA_PATH)  # ✅ DatetimeIndex(UTC)

# ✅ Market Mood와 동일 헤더 (title + 날짜 선택)
anchor_ts, row = render_header(
    df=df,
    title="Market Signals",
    subtitle="시장 분위기를 구성하는 개별 지표를 상세하게 확인합니다.",
    date_key="ms_date",
)
spacer(16)

row_dict = row.to_dict() if hasattr(row, "to_dict") else {}


# ======================
# Section 1: Signals Overview
# ======================
with st.container(border=True):
    st.markdown("<div data-ms='overview'></div>", unsafe_allow_html=True)
    section_title("Signals Overview", "핵심 지표를 상태(🟢🟡🔴)로 빠르게 스캔합니다.")

    tiles: List[Dict[str, str]] = [
        {"name": "Open Interest", "status": "🔴", "one": "단기 급증 구간", "meta": "Percentile • 60d"},
        {"name": "Funding Rate", "status": "🔴", "one": "상단 구간(과열)", "meta": "Percentile • 60d"},
        {"name": "Liquidations", "status": "🟡", "one": "증가 추세(주의)", "meta": "Rolling sum"},
        {"name": "Taker Flow", "status": "🟡", "one": "공격적 매수·매도 쏠림 관측", "meta": "Imbalance"},
        {"name": "M2", "status": "🟢", "one": "완만한 유동성 흐름", "meta": "Macro"},
    ]

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    for col, t in zip([c1, c2, c3, c4, c5], tiles):
        with col:
            render_signal_tile(t["name"], t["status"], t["one"], t.get("meta", ""))

spacer(32)


# ======================
# Section 2: Derivatives Signals
# ======================
with st.container(border=True):
    st.markdown("<div data-ms='derivatives'></div>", unsafe_allow_html=True)
    section_title("Derivatives Signals", "파생 지표를 차트로 확인합니다.")

    left, right = st.columns([2, 1], gap="large")

    with left:
        chart_placeholder("Funding Rate (60d)", "percentile band / 최근 급등 구간 강조")
        spacer(12)
        chart_placeholder("Open Interest (60d)", "급증/감소 구간 강조")
        spacer(12)
        chart_placeholder("Liquidations (30d)", "청산 스파이크 표시")

    with right:
        section_title("Quick Notes", "차트 옆 한 줄 해석(자동 생성 영역)")
        render_signal_tile("Funding", "🔴", "펀딩 과열 구간 — 레버리지 쏠림 점검", "Rule-based note")
        spacer(10)
        render_signal_tile("OI", "🔴", "포지션 과밀 — 변동성 확대 가능", "Rule-based note")
        spacer(10)
        render_signal_tile("Liq", "🟡", "청산 증가 — 급변 구간 발생 가능", "Rule-based note")

spacer(32)


# ======================
# Section 3: Liquidity / Macro
# ======================
with st.container(border=True):
    st.markdown("<div data-ms='macro'></div>", unsafe_allow_html=True)
    section_title("Liquidity / Macro", "구조적 환경(유동성)을 확인합니다.")

    l, r = st.columns([2, 1], gap="large")
    with l:
        chart_placeholder("M2 (macro)", "느린 지표 — 장기 흐름 위주")
    with r:
        render_signal_tile("M2", "🟢", "완만한 흐름 — 급격한 악화 신호는 제한적", "Macro note")

spacer(32)


# ======================
# Section 4: Sentiment & Attention
# ======================
with st.container(border=True):
    st.markdown("<div data-ms='sentiment'></div>", unsafe_allow_html=True)
    section_title("Sentiment & Attention", "시장 심리와 관심도(보너스 요인)를 확인합니다.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_placeholder("Market Sentiment (z-score)", "60d z-score 기반")
    with c2:
        chart_placeholder("Google Trends (z-score)", "BTC 관심도")

spacer(32)
