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
   BLOCK HEAD (공통)
====================================================== */
.mm-block-title {{
  font-size: 16px;
  font-weight: 900;
  letter-spacing: -0.01em;
  color: rgba(255,255,255,.92);
}}
.mm-block-sub {{
  margin-top: 6px !important;
  margin-bottom: 12px !important;
  font-size: 12px;
  color: rgba(255,255,255,.55);
  line-height: 1.45;
}}

/* ======================================================
   st.container(border=True) 기본 카드 스타일
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
  padding: 18px 16px 18px 16px !important;
}}

div[data-testid="stDivider"] {{ display:none !important; }}

/* ======================================================
   SELECTBOX / BASEWEB (날짜 선택 UI) - spacing fix (strong)
====================================================== */

/* 1) st.selectbox 라벨 자체 / 라벨 안 p(텍스트) 모두 커버 */
div[data-testid="stSelectbox"] label {{
  margin-bottom: 10px !important;          /* 라벨-박스 간격 */
}}

div[data-testid="stSelectbox"] label p {{
  margin: 0 0 10px 0 !important;           /* Streamlit이 p에 margin 주는 케이스 */
  padding: 0 !important;
  line-height: 1.2 !important;
}}

/* 2) select 박스(베이스웹) 위쪽 여백도 추가 */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
  margin-top: 8px !important;              /* 필요시 0~6 */
}}

/* 3) 혹시 label이 stWidgetLabel로 따로 렌더되는 버전도 같이 커버 */
div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {{
  margin-bottom: 10px !important;
}}


/* ======================================================
   Market Signals (scoped)
   ✅ 안정판: data-ms 마커로만 컨트롤
   - 03_Market_Signals.py에서 아래 2개만 넣으면 됨
     Metrics 컨테이너 안:  st.markdown("<div data-ms='metrics'></div>", unsafe_allow_html=True)
     Futures 컨테이너 안:  st.markdown("<div data-ms='futures'></div>", unsafe_allow_html=True)
   - 그리고 페이지 상단에 ms-page 마커는 유지:
     st.markdown("<span class='ms-page'></span>", unsafe_allow_html=True)
====================================================== */

/* (1) Market Signals 페이지 전체: 섹션(컨테이너)들 사이 기본 간격 */
:has(.ms-page) [data-testid="stVerticalBlock"] {{
  gap: 12px !important;            /* ✅ 10~16 취향 */
}}

/* (2) Metrics ↔ Futures 간격만 더 타이트하게 */
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="metrics"]) {{
  margin-bottom: 6px !important;   /* ✅ 0~12 */
}}
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="futures"]) {{
  margin-top: -10px !important;    /* ✅ -6 ~ -18 */
}}

/* (3) Futures 아래 바닥(패딩) 확보 */
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="futures"]) > div {{
  padding-bottom: 24px !important; /* ✅ 18~32 */
}}

/* (4) Market Signals에서 제목/서브 텍스트 간격 조금 정리(선택) */
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="metrics"]) .mm-block-sub,
:has(.ms-page) div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-ms="futures"]) .mm-block-sub {{
  margin-top: 4px !important;
  margin-bottom: 8px !important;
}}

/* =========================
   Mobile responsive (global)
========================= */
@media (max-width: 820px){{

  /* Streamlit columns 겹침 방지 */
  div[data-testid="stHorizontalBlock"]{{
    flex-wrap: wrap !important;
  }}
  div[data-testid="column"]{{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}

  /* 페이지 좌우 여백 */
  section.main > div{{
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
  }}

  /* =========================
     ✅ 2열 유지하고 싶은 그리드들
     - 화면 충분하면 2열
     - 부족하면 자동 1열
  ========================= */
  .mm-grid-2x2,
  .ms-futures-grid{{
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
  }}

  /* 아주 작은 화면(아이폰 SE급)은 1열로 폴백 */
  @media (max-width: 520px){{
    .mm-grid-2x2,
    .ms-futures-grid{{
      grid-template-columns: 1fr !important;
    }}
  }}

  /* =========================
     Risk Alert: 7D Direction는 모바일 1열이 안정적
  ========================= */
  .ra-7d-grid{{
    grid-template-columns: 1fr !important;
    gap: 12px !important;
  }}

  /* Up20 / Down20도 모바일에선 세로 */
  .ra-split2{{
    grid-template-columns: 1fr !important;
  }}
}}

/* ✅ 모바일에서 큰 숫자 KPI가 잘리는 문제 해결 */
@media (max-width: 640px){{
  .fs-kpi, .fs-value, .mm-value, .kpi-value, .tile-value{{
    font-size: clamp(18px, 5.6vw, 28px) !important;
    letter-spacing: -0.02em !important;
    white-space: nowrap !important;
  }}

  /* 카드 안에서 넘칠 경우 대비 */
  .fs-card, .mm-card, .kpi-tile, .fs-tile{{
    overflow: hidden !important;
  }}

  /* $5... 처럼 생략 표시 자연스럽게 */
  .fs-kpi, .fs-value, .mm-value, .kpi-value, .tile-value{{
    text-overflow: ellipsis !important;
    overflow: hidden !important;
  }}
}}

/* ✅ Market Signals - Metrics: 모바일 1열 */
@media (max-width: 640px){{
  .fs-grid{{
    grid-template-columns: 1fr !important;
  }}
}}
/* =========================
   ✅ Risk Alert 7D - Up20/Down20 뱃지 정렬 (PC/모바일 유지 + 아이패드만 분리)
========================= */

/* grid/box 안에서 내용 때문에 옆 칸을 밀지 못하게 */
.ra-7d-wrap, .ra-7d-grid, .ra-7d-card, .ra-7d-card-inner,
.ra-split2, .ra-split-box{{
  min-width: 0 !important;
}}

/* 헤더 줄: 기본은 한 줄 */
.ra-split-top{{
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 8px !important;
}}

/* Up20 / Down20 라벨 */
.ra-split-kicker{{
  flex: 0 0 auto !important;
  white-space: nowrap !important;
}}

/* ✅ 배지 기본 위치: 같은 줄 오른쪽 */
.ra-split-top .ra-flag-badge,
.ra-split-top .ra-badge,
.ra-split-top .ra-pill,
.ra-split-top .ra-flag{{
  margin-left: auto !important;
  flex: 0 0 auto !important;
  white-space: nowrap !important;
}}

/* ✅ 아이패드/중간 폭에서만: 배지만 아래 줄로 */
@media (min-width: 768px) and (max-width: 1100px){{
  .ra-split-top{{
    align-items: flex-start !important;
  }}

  .ra-split-top .ra-flag-badge,
  .ra-split-top .ra-badge,
  .ra-split-top .ra-pill,
  .ra-split-top .ra-flag{{
    margin-left: 0 !important;
    width: 100% !important;
    margin-top: 6px !important;
  }}
}}

/* overflow 잘림 방지 */
.ra-7d-wrap .ra-split2 .ra-split-box{{
  overflow: visible !important;
}}

</style>
        """,
        unsafe_allow_html=True,
    )
