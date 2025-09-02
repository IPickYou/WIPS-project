import streamlit as st

def show():
    st.write("**현재 카테고리 목록 (편집 가능):**")
    
    categories_to_delete = []
    updated_categories = {}
    
    for i, (code, description) in enumerate(st.session_state.categories.items()):
        col_code, col_desc, col_delete = st.columns([1, 4, 1])
        
        with col_code:
            new_code = st.text_input(f"코드", value=code, key=f"code_{i}", label_visibility="collapsed")
        with col_desc:
            new_desc = st.text_area(f"설명", value=description, key=f"desc_{i}", height=68, label_visibility="collapsed")
        with col_delete:
            st.write("")  # 공간 확보
            if st.button( key=f"delete_{i}", help="삭제"):
                categories_to_delete.append(code)
        
        updated_categories[new_code] = new_desc
    
    # 삭제된 카테고리 처리
    if categories_to_delete:
        for code in categories_to_delete:
            if code in st.session_state.categories:
                del st.session_state.categories[code]
        st.rerun()
    
    # 변경사항 적용
    st.session_state.categories = updated_categories
    
    # 새 카테고리 추가
    st.write("**새 카테고리 추가:**")
    col_new_code, col_new_desc, col_add = st.columns([1, 4, 1])
    
    with col_new_code:
        new_category_code = st.text_input("카테고리 코드", placeholder="C01H", key="new_category_code")
    with col_new_desc:
        new_category_desc = st.text_area("카테고리 설명", placeholder="화학반응 - 특정 화학반응 과정 및 방법에 관한 분야", key="new_category_desc", height=68)
    with col_add:
        st.write("")  # 공간 확보
        if st.button("추가", key="add_category_btn"):
            if new_category_code.strip() and new_category_desc.strip():
                st.session_state.categories[new_category_code.strip()] = new_category_desc.strip()
                st.rerun()
            else:
                st.error("코드와 설명을 모두 입력해주세요")
    
    # 초기화 버튼
    if st.button("기본값으로 초기화", key="reset_categories_btn"):
        st.session_state.categories = {
            "C01B": "비금속 원소, 비금속 화합물 (예: 수소, 질소, 산소 관련 화합물)",
            "C01C": "무기산, 무기산의 염 (예: 황산, 질산, 인산 등)",
            "C01D": "할로겐 화합물 (예: 염소, 브롬, 플루오르 화합물)",
            "C01F": "알칼리 금속, 알칼리 토금속, 희토류 금속 화합물",
            "C01G": "귀금속, 기타 금속 화합물"
        }
        st.rerun()