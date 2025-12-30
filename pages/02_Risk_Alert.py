# pages/02_Risk_Alert.py
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


def pill(sev: str) -> str:
    if sev == "HIGH":
        return "🔴 HIGH"
    if sev == "MED":
        return "🟡 MED"
    return "🟢 LOW"


def render_alert_card(title: str, sev: str, desc: str, meta: str = "") -> None:
    st.markdown(
        f"""
        <div class="mm-mini-card">
          <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
            <div style="font-weight:800; letter-spacing:-0.01em;">{title}</div>
            <div style="opacity:.9; font-weight:800;">{pill(sev)}</div>
          </div>
          <div style="margin-top:8px; opacity:.70; line-height:1.4;">{desc}</div>
          {"<div style='margin-top:8px; opacity:.45; font-size:12px;'>" + meta + "</div>" if meta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_driver_tile(name: str, status: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="mm-mini-card">
          <div style="display:flex; justify-content:space-between; gap:10px; align-items:center;">
            <div style="font-weight:800;">{name}</div>
            <div style="font-weight:900;">{status}</div>
          </div>
          <div style="margin-top:8px; opacity:.70; line-height:1.4;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================
# Page
# ======================
st.set_page_config(page_title="Risk Alert", layout="wide")
inject_styles()

df = load_df(DATA_PATH)  # ✅ 이미 DatetimeIndex(UTC)로 세팅됨

# ✅ Market Mood와 동일 헤더 (title + 날짜 선택)
anchor_ts, row = render_header(
    df=df,
    title="Risk Alert",
    subtitle="오늘 시장에서 특히 주의해야 할 신호를 우선순위로 정리합니다.",
    date_key="ra_date",
)
spacer(16)

# row: 선택 날짜의 최신 row (Series)
row_dict = row.to_dict() if hasattr(row, "to_dict") else {}


# ======================
# Section 1: Today's Risk Alerts
# ======================
with st.container(border=True):
    st.markdown("<div data-ra='top'></div>", unsafe_allow_html=True)
    section_title("Today’s Risk Alerts", "가장 중요한 경고 신호만 먼저 보여줍니다.")

    # TODO: 실제 alert 생성 로직으로 교체
    alerts: List[Dict[str, str]] = [
        {
            "title": "Funding Overheat",
            "sev": "HIGH",
            "desc": "펀딩비가 최근 분포 상단에 위치해 레버리지 쏠림이 커질 수 있습니다.",
            "meta": "Source: Funding percentile • Lookback: 60d",
        },
        {
            "title": "Open Interest Spike",
            "sev": "MED",
            "desc": "미결제약정이 단기간 급증해 포지션 과밀 위험을 점검할 필요가 있습니다.",
            "meta": "Source: OI percentile • Lookback: 60d",
        },
    ]

    for a in alerts:
        render_alert_card(a["title"], a["sev"], a["desc"], a.get("meta", ""))
        spacer(10)

spacer(32)


# ======================
# Section 2: Alert Breakdown (Drivers)
# ======================
with st.container(border=True):
    st.markdown("<div data-ra='drivers'></div>", unsafe_allow_html=True)
    section_title("Alert Breakdown", "어떤 지표가 경고를 만들었는지 빠르게 확인합니다.")

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")

    # TODO: row 기반으로 상태 계산 연결
    drivers = [
        ("Open Interest", "🔴", "최근 급증 구간"),
        ("Funding", "🔴", "상단 percentile"),
        ("Liquidations", "🟡", "평균 이상 증가"),
        ("Taker Flow", "🟡", "단기 쏠림 관측"),
        ("M2", "🟢", "완만한 흐름"),
    ]

    for col, (name, status, note) in zip([c1, c2, c3, c4, c5], drivers):
        with col:
            render_driver_tile(name, status, note)

spacer(32)


# ======================
# Section 3: Recent Alert History
# ======================
with st.container(border=True):
    st.markdown("<div data-ra='history'></div>", unsafe_allow_html=True)
    section_title("Recent Alert History", "오늘만의 이슈인지, 누적 흐름인지 맥락을 제공합니다.")

    # TODO: 실제 history 생성 로직으로 교체
    history = [
        "3일 전: Funding 경고(🟡) → 오늘 🔴로 상승",
        "1주 전: OI 급증 경고(🟡)",
        "2주 전: 청산 증가 신호(🟡) 관측",
    ]
    for h in history:
        st.markdown(f"- {h}")

spacer(32)
