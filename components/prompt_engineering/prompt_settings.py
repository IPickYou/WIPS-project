from utils import lmstudio

import streamlit as st

def show(selected_columns, df, custom_separator):
    st.write("**자동 생성된 분류 프롬프트 미리보기:**")
    
    # 프롬프트 자동 생성
    categories_text = "\n".join([f"- {code}: {desc}" for code, desc in st.session_state.categories.items()])
    
    auto_prompt = f"""다음 특허 텍스트를 아래 카테고리 중 하나로 분류해주세요:

텍스트: {{text}}

분류 카테고리:
{categories_text}

위 카테고리 중 가장 적절한 것을 선택하여 코드만 정확히 답변해주세요 (예: C01B)."""
    
    # 프롬프트 표시 (읽기 전용)
    st.text_area(
        "생성된 프롬프트",
        value=auto_prompt,
        height=300,
        disabled=True,
        help="카테고리를 수정하면 자동으로 프롬프트가 업데이트됩니다"
    )
    
    # LM Studio API 설정
    lmstudio.settings()
    
    # 분류 실행
    if selected_columns and len(st.session_state.categories) > 0:
        lmstudio.inference(selected_columns, df, custom_separator, auto_prompt)
    else:
        st.warning("칼럼을 선택하고 최소 하나의 카테고리가 있어야 합니다.")