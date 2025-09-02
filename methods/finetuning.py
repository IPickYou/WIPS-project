from tabs import inference, train

import streamlit as st

def show():
    st.subheader("학습 및 추론")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["추론", "학습"])

    # --- 추론 탭 ---
    with tab1:
        inference.show()
    # --- 학습 탭 ---
    with tab2:
        train.show()