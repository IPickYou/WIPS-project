from utils import lmstudio
import streamlit as st

def show(selected_columns, df, custom_separator):

    with st.expander("**PROMPT TEMPLATE**", expanded = False):
    
        if 'step1_prompt' not in st.session_state:
            st.session_state.step1_prompt = """다음은 특허 정보입니다.

{text}

아래의 CPC 코드가 이 특허에 얼마나 적합한 지 평가해 주세요.  

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

        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.step1_prompt = st.text_area(
                "[STEP 1] RELEVANCE SCORING",
                value = st.session_state.step1_prompt,
                height = 300,
                help = "각 카테고리마다 적합도 점수를 계산합니다."
            )
        
        with col2:
            st.session_state.step2_prompt = st.text_area(
                "[STEP 2] BEST MATCH SELECTION", 
                value = st.session_state.step2_prompt,
                height = 300,
                help = "동점인 경우, 최적의 코드를 재선택합니다."
            )
        
        if st.button("INITIALIZE PROMPT"):
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
            st.rerun()

    lmstudio.settings()

    if selected_columns and len(st.session_state.categories) > 0:
        lmstudio.inference(selected_columns, df, custom_separator, 
                          st.session_state.step1_prompt, st.session_state.step2_prompt, 
                          st.session_state.categories)
    else:
        st.warning("Please select at least one column and one category.")