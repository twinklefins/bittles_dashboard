from __future__ import annotations

from typing import Tuple
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _inject_header_css() -> None:
    st.markdown(
        """
<style>
/* =========================
   HEADER (anchor-based)
   - mm-header-anchor 바로 "다음"에 생성되는 columns 블록을 타겟팅
========================= */

/* 헤더 타이틀/서브 */
.mm-hdr{ margin: 0; }
.mm-hdr-title{
  font-size: 32px;
  font-weight: 900;
  letter-spacing: -0.02em;
  color: rgba(255,255,255,.95);
  line-height: 1.12;
  margin: 0 0 10px 0;
}
.mm-hdr-subtitle{
  font-size: 14px;
  color: rgba(255,255,255,.52);
  line-height: 1.45;
  margin: 0;
}

/* 오른쪽 라벨 */
.mm-date-label{
  font-size: 12px;
  color: rgba(255,255,255,.65);
  margin: 0 0 0 0;
}

/* ✅ 핵심 1) 헤더 앵커 다음에 나오는 "columns 블록" 자체를 flex-start로 */
div#mm-header-anchor + div [data-testid="stHorizontalBlock"]{
  align-items: flex-start !important;
}

/* ✅ 핵심 2) 오른쪽 column 내부의 위쪽 padding/여백을 눌러서 윗단 맞추기
   - Streamlit 버전에 따라 column 구조가 달라서 넓게 잡음 */
div#mm-header-anchor + div [data-testid="column"]:last-child > div{
  padding-top: 0 !important;
  margin-top: 0 !important;
}

/* ✅ 핵심 3) selectbox 자체가 위로 튀는 걸 살짝 눌러주기 */
div#mm-header-anchor + div [data-testid="column"]:last-child [data-testid="stSelectbox"]{
  margin-top: -2px !important;
}

/* (선택) 라벨도 살짝 올리고 싶으면 여기만 조절: -2 ~ 4 */
div#mm-header-anchor + div [data-testid="column"]:last-child .mm-date-label{
  margin-top: 4px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    df: pd.DataFrame,
    title: str,
    subtitle: str = "",
    date_key: str = "global_date",
) -> Tuple[pd.Timestamp, pd.Series]:
    _inject_header_css()

    # ✅ 앵커: 이 바로 아래 생성되는 columns 블록을 CSS로 정확히 타겟팅할 수 있음
    st.markdown('<div id="mm-header-anchor"></div>', unsafe_allow_html=True)

    left, right = st.columns([7.5, 2.5], vertical_alignment="top")

    date_options = sorted(pd.Series(df.index.date).unique(), reverse=True)

    with left:
        st.markdown(
            f"""
            <div class="mm-hdr">
              <div class="mm-hdr-title">{title}</div>
              {"<div class='mm-hdr-subtitle'>" + subtitle + "</div>" if subtitle else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="mm-date-label">날짜 선택</div>', unsafe_allow_html=True)
        sel_date = st.selectbox(
            label="날짜 선택",
            options=date_options,
            index=0,
            key=date_key,
            label_visibility="collapsed",
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )

    day_df = df[df.index.date == sel_date]
    row = day_df.iloc[-1] if not day_df.empty else df.iloc[-1]
    anchor_ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)
    return anchor_ts, row
