from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st
from components.styles import inject_styles


# =========================================================
# App Config (✅ 딱 1번만)
# =========================================================
st.set_page_config(
    page_title="BITTLES Dashboard",
    page_icon=" ",  # ✅ 공백 1칸 OK (""는 에러)
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Paths (✅ 먼저 정의)
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
SIDEBAR_LOGO_PATH = ASSETS_DIR / "Bittles_box.png"


# =========================================================
# Read logo as base64 (✅ 먼저 만들기)
# =========================================================
logo_b64: str | None = None
if SIDEBAR_LOGO_PATH.exists():
    logo_b64 = base64.b64encode(SIDEBAR_LOGO_PATH.read_bytes()).decode("utf-8")


# =========================================================
# Sidebar CSS
# =========================================================
def apply_sidebar_nav_style(logo_b64: str | None) -> None:
    logo_css = ""

    if logo_b64:
        logo_css = f"""
        [data-testid="stSidebarContent"]::before {{
            content: "";
            display: block;

            /* ✅ 박스는 '정사각 카드' 느낌으로 */
            width: 180px;
            height: 180px;

            /* ✅ 위에 붙는 느낌 해결: 상단 여백 늘리기 */
            margin: 34px auto 0px auto;

            border-radius: 28px;
            background: rgba(255,255,255,0.06);
            background-image: url("data:image/png;base64,{logo_b64}");
            background-repeat: no-repeat;
            background-position: center;

            /* ✅ 로고는 카드 안에서 좀 작게 (여백 컨셉) */
            background-size: 78% auto;

            backdrop-filter: blur(10px);
            box-shadow: 0 14px 34px rgba(0,0,0,0.42);
        }}
        """

    st.markdown(
        f"""
        <style>
        :root{{
        --sb-bg: #0b1220;
        --sb-text: rgba(255,255,255,.90);

        --brand-orange: #F87B1B;            /* ✅ 로고 오렌지 */
        --sb-hover: rgba(248,123,27,0.12);  /* ✅ hover도 로고 오렌지 기반 */
        --sb-active-bg: rgba(248,123,27,0.16);
        --sb-active-bar: var(--brand-orange);

        --sb-divider: rgba(255,255,255,0.10);
        }}

        [data-testid="stSidebar"]{{
            background: var(--sb-bg) !important;
        }}

        /* ✅ Sidebar 내부 기본 padding 제거 */
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 0px !important;
        }}

        /* ✅ 접기 버튼(<<)이 로고 위에 떠보이면 숨기는 게 가장 깔끔 */
        [data-testid="stSidebarCollapseButton"]{{
            display: none !important;
        }}

        /* 기본 구분선 숨김 */
        [data-testid="stSidebar"] hr{{ display:none !important; }}
        [data-testid="stSidebarNavSeparator"]{{ display:none !important; }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr{{ display:none !important; }}

        {logo_css}

        /* ✅ nav는 더 이상 큰 margin-top으로 밀지 말기 */
        [data-testid="stSidebarNav"]{{
            margin-top: -36px;
            border-bottom: none;
            padding-bottom: 10px;
            margin-bottom: 6px;
        }}

        /* 라벨 */
        [data-testid="stSidebarNav"]::before{{
            content: "BITTLES INSIGHT";
            display:block;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 2.2px;
            color: rgba(248,123,27,.88);
            margin: 12px 0 10px 2.25rem;
            text-transform: uppercase;
        }}

        /* Links */

        /* nav 링크 공통 */
        [data-testid="stSidebarNav"] a{{
        display: flex;
        align-items: center;

        /* ✅ 폭 통일 핵심 */
        width: 10.5rem;          /* <- 여기 숫자만 조절 */
        max-width: calc(100% - 1.2rem);

        padding: .62rem 1.0rem .62rem 2.12rem;  /* 왼쪽 여백 유지 */
        margin: .22rem auto;      /* ✅ 가운데 정렬 */

        border-radius: .85rem;
        text-decoration: none !important;
        color: var(--sb-text) !important;
        }}

        [data-testid="stSidebarNav"] a *{{ color: inherit !important; }}

        [data-testid="stSidebarNav"] a:hover{{ background: var(--sb-hover); }}

        /* Active link */
        [data-testid="stSidebarNav"] a[aria-current="page"]{{
            background: var(--sb-active-bg);
            font-weight: 700;
            position: relative;
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"]::before{{
            content:"";
            position:absolute;
            left:0.98rem;
            top:.42rem;
            bottom:.42rem;
            width:.22rem;
            border-radius:.25rem;
            background: var(--sb-active-bar);
            box-shadow: 0 0 14px rgba(247,147,26,0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# ✅ 실행 순서 (중요)
# =========================================================
inject_styles()                 # 전체 테마
apply_sidebar_nav_style(logo_b64)  # 사이드바 CSS (로고 포함)


# =========================================================
# Pages (st.navigation)
# =========================================================
market_mood = st.Page("pages/01_Market_Mood.py", title="Market Mood", default=True)
risk_alert = st.Page("pages/02_Risk_Alert.py", title="Risk Alert")
market_signals = st.Page("pages/03_Market_Signals.py", title="Market Signals")

nav = st.navigation([market_mood, risk_alert, market_signals], position="sidebar")
nav.run()
