# methods/finetuning.py

from tabs import inference, train
import streamlit as st
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def show():

    if st.session_state.uploaded_df is not None:
        
        st.header("FINE TUNING")

        st.write(" ")

        with st.expander("**TRANSFORMERS SETTING**", expanded = True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.ft_model_name = st.text_input(
                    "MODEL NAME", 
                    value=st.session_state.get('ft_model_name', 'google/gemma-2-2b'), 
                    key="ft_model_name_input"
                )
            with col2:
                st.session_state.ft_hf_token = st.text_input(
                    "HUGGING FACE TOKEN", 
                    value=st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN'), 
                    type="password", 
                    key="ft_hf_token_input"
                )

        tab1, tab2 = st.tabs(["**INFERENCE**", "**TRAIN**"])

        with tab1:
            inference.show()

        with tab2:
            train.show()

    else:
        st.info("⬅️ 특허 문서를 업로드해 주세요.")