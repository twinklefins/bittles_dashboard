# components/styles.py
import streamlit as st

def inject_styles() -> None:
    bg0 = "#070B14"
    bg1 = "#0B1220"

    text = "rgba(255,255,255,.92)"
    muted = "rgba(255,255,255,.62)"

    st.markdown(
        f"""
<style>
/* ======================================================
   GLOBAL LAYOUT
====================================================== */
.block-container {{
  padding-top: 1.35rem !important;
  padding-bottom: 2.2rem !important;
  max-width: 1220px;
}}

/* ======================================================
   FORCE DARK BACKGROUND
====================================================== */
html, body,
[data-testid="stAppViewContainer"],
section[data-testid="stMain"],
header[data-testid="stHeader"] {{
  background:
    radial-gradient(1200px 700px at 18% 8%, rgba(45,107,255,.20), transparent 55%),
    radial-gradient(1200px 700px at 80% 20%, rgba(247,147,26,.16), transparent 58%),
    radial-gradient(900px 600px at 55% 70%, rgba(255,45,85,.10), transparent 60%),
    linear-gradient(180deg, {bg0}, {bg1}) !important;
  color: {text} !important;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
}}

/* 캡션 */
.stCaption, [data-testid="stCaptionContainer"] {{
  color: {muted} !important;
}}

/* ======================================================
   ✅ st.container(border=True) ONLY
   - 일반 컨테이너 전체를 건드리면 다른 페이지도 깨짐
====================================================== */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.18) !important;
  border-radius: 18px !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset,
              0 18px 44px rgba(0,0,0,.35) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}}

div[data-testid="stVerticalBlockBorderWrapper"] > div {{
  padding: 14px 16px 16px 16px !important; /* 위아래 여유 줄임 */
}}

/* divider(가로줄) 제거 */
div[data-testid="stDivider"]{{ display:none !important; }}

/* ======================================================
   SELECTBOX / BASEWEB (날짜 선택 UI)
====================================================== */
div[data-baseweb="select"] > div {{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {{
  color: rgba(255,255,255,0.92) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.92) !important;
  caret-color: rgba(255,255,255,0.92) !important;
  font-weight: 800 !important;
}}

div[data-baseweb="select"] svg {{
  fill: rgba(255,255,255,0.75) !important;
}}

div[data-baseweb="popover"] div[role="listbox"] {{
  background: rgba(14,16,22,0.98) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45) !important;
}}

div[role="option"] {{
  color: rgba(255,255,255,0.86) !important;
}}
div[role="option"][aria-selected="true"] {{
  background: rgba(255,255,255,0.10) !important;
}}
div[role="option"]:hover {{
  background: rgba(255,255,255,0.08) !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )
