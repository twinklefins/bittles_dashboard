# 📊 BITTLES 시장 위험도 대시보드 (MVP)

비트코인 파생상품/온체인/거시 지표를 기반으로  
**시장 위험도를 신호등(🟢🟡🔴) 형태로 직관적으로 확인**할 수 있는 Streamlit 대시보드입니다.

> 목표: 특정 사건(ETF Flow, 온체인 변화, 규제·거시 이벤트 등) 이후에도  
> 사용자가 **패닉셀을 하지 않도록 “근거 기반 메시지”**를 제공하는 MVP를 구축합니다.

---

## ✨ 주요 기능
- ✅ 날짜 선택(기준일) 기반 지표 조회
- ✅ OI / Funding / Liquidation / Taker Buy Ratio / M2 신호등 표시
- ✅ 퍼센타일 기반 위험도 판별(낮음/중간/높음)
- ✅ 자동 요약 메시지(오늘의 원인 요약)
- ✅ 결측 데이터 안전 처리(표시/설명)

---

## 📁 프로젝트 구조
```text
BITTLES_DASHBOARD/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ data/
   └─ df_var_1209.csv   # 로컬 실행용(깃허브에는 포함되지 않음)