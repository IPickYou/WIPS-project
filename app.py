from components import sidebar
from dotenv import load_dotenv
from methods import finetuning, prompt_engineering
import streamlit as st

load_dotenv()

st.set_page_config(page_title = "특허 문서 분류 자동화 플랫폼", layout = "wide")

st.title("특허 문서 분류 자동화 플랫폼")

st.markdown("---")

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:1234/v1/chat/completions"
if "api_model" not in st.session_state:
    st.session_state.api_model = "qwen/qwen3-14b"
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
        "CPC_C01B" : "Non-metal elements and their compounds (excluding CO2). Inorganic compounds without metals.",
        "CPC_C01C" : "Ammonia, cyanide, and their compounds.",
        "CPC_C01D" : "Alkali metal compounds such as lithium, sodium, potassium, rubidium, cesium, or francium.",
        "CPC_C01F" : "Compounds of metals like beryllium, magnesium, aluminum, calcium, strontium, barium, radium, thorium, or rare earth metals.",
        "CPC_C01G" : "Compounds containing metals not included in C01D or C01F."
    }
if 'step1_prompt' not in st.session_state:
    st.session_state.step1_prompt = """다음은 특허 정보입니다.

{text}

아래의 CPC 코드가 이 특허에 얼마나 적합한지 평가해 주세요.  

0.0은 전혀 관련 없음을 의미하고,  
5.0은 부분적으로 관련 있음을 의미하며,  
10.0은 완벽히 일치함을 의미합니다.  

CPC 코드 : {code}
설명 : {desc}

0.0에서 10.0 사이의 소수점 1자리 숫자만 출력해 주세요.  
숫자 외에는 어떠한 단어도 출력하지 마세요."""

if 'step2_prompt' not in st.session_state:
    st.session_state.step2_prompt = """다음은 특허 정보입니다.

{text}

아래의 CPC 후보 중에서 위의 특허에 가장 적합한 **단 1개의 CPC 코드**를 골라 주세요.

[CPC 후보 및 설명]
{candidate_text}

규칙 :
1. 반드시 CPC 후보에 있는 코드 중 하나만 출력하세요.
2. 후보에 없는 다른 코드는 절대 출력하지 마세요.
3. 부가 설명이나 다른 단어 없이 코드만 출력하세요.
4. 출력 전에 신중히 검토하세요.
5. 출력 예시 : CPC_C01B"""

classification_method = sidebar.show()

if classification_method == "PROMPT ENGINEERING":
    prompt_engineering.show()

elif classification_method == "FINE TUNING":
    finetuning.show()

st.markdown("---")

st.markdown(
    """
    <div style = 'text-align : center; color : gray;'>
        특허 문서 분류 자동화 플랫폼 © IPickYou
    </div>
    """, 
    unsafe_allow_html = True
)