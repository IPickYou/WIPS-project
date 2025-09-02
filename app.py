from components import sidebar
from dotenv import load_dotenv
from methods import finetuning, prompt_engineering

import streamlit as st

load_dotenv()

# 페이지 설정
st.set_page_config(page_title="특허 분류기", layout="wide")

# 메인 타이틀
st.title(" 특허 분류기")
st.markdown("---")

# 세션 상태 초기화
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'categories' not in st.session_state:
    st.session_state.categories = {
        "C01B": "비금속 원소, 비금속 화합물 (예: 수소, 질소, 산소 관련 화합물)",
        "C01C": "무기산, 무기산의 염 (예: 황산, 질산, 인산 등)",
        "C01D": "할로겐 화합물 (예: 염소, 브롬, 플루오르 화합물)",
        "C01F": "알칼리 금속, 알칼리 토금속, 희토류 금속 화합물",
        "C01G": "귀금속, 기타 금속 화합물"
    }

classification_method = sidebar.show() # 사이드 바 출력

# 메인 컨텐츠 영역
if classification_method == "프롬프트 엔지니어링":
    prompt_engineering.show()

elif classification_method == "학습 및 추론":
    finetuning.show()