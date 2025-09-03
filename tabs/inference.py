from utils import trained_model

import os
import streamlit as st
import torch

def show():
    st.write("### 모델 추론")
    st.write("**모델 설정**")

    # 모델 설정
    col1, col2 = st.columns(2)
    
    with col1:
        model_id = st.text_input(
            "베이스 모델 ID",
            value="meta-llama/Llama-3.2-1B",
            key="model_id"
        )
        
        tokenizer_path = st.text_input(
            "토크나이저 경로",
            value="C:/aimer/wips/models/wips_t",
            key="tokenizer_path"
        )
    
    with col2:
        hf_token = st.text_input(
            "HuggingFace Token",
            value=os.getenv("HF_TOKEN"),
            type="password",
            key="hf_token"
        )
        
        num_labels = st.number_input(
            "라벨 수",
            min_value=2,
            max_value=100,
            value=5,
            key="num_labels"
        )
    if st.button("🔄 모델 로드"):
        trained_model.load(model_id, hf_token, num_labels, tokenizer_path)
                
    # 모델 상태 표시
    if st.session_state.get('model_loaded', False):
        st.success(" 모델이 로드되어 추론 준비가 완료되었습니다!")
        
        # 모델 정보 표시
        with st.expander("모델 정보"):
            model = st.session_state.model
            st.write(f"- 모델 타입: {model.config.model_type}")
            st.write(f"- 라벨 수: {model.config.num_labels}")
            if hasattr(model.config, 'id2label') and model.config.id2label:
                st.write("- 라벨 매핑:")
                for id_val, label in model.config.id2label.items():
                    st.write(f"  - {id_val}: {label}")
    else:
        st.warning(" 모델을 먼저 로드해주세요.")

    st.markdown("---")
    
    # 추론 데이터 준비
    st.write("**추론 데이터 설정**")
    
    # 추론 실행
    if st.session_state.get('uploaded_df') is not None and st.session_state.get('model_loaded', False):
        trained_model.inference()

    elif st.session_state.get('uploaded_df') is None:
        st.info("먼저 사이드바에서 엑셀 파일을 업로드해주세요.")
    elif not st.session_state.get('model_loaded', False):
        st.info("먼저 모델을 로드해주세요.")

    # 추론 결과 표시
    if st.session_state.get('inference_results'):
        st.write(" **추론 결과**")
        results = st.session_state.inference_results
        
        # 결과를 DataFrame으로 변환
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        # 결과 요약
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 샘플 수", len(results_df))
        with col2:
            if 'predicted_class' in results_df.columns:
                unique_classes = results_df['predicted_class'].nunique()
                st.metric("예측된 클래스 수", unique_classes)
        
        # 클래스별 분포 표시
        if 'predicted_class' in results_df.columns:
            st.write("**클래스별 분포:**")
            class_counts = results_df['predicted_class'].value_counts()
            st.bar_chart(class_counts)
        
        # 결과 데이터프레임 표시
        st.write("**상세 결과:**")
        st.dataframe(results_df)
        
        # CSV 다운로드 버튼
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=" 추론 결과 CSV 다운로드",
            data=csv,
            file_name="inference_results.csv",
            mime="text/csv"
        )
        
        # 원본 데이터와 결합된 결과 다운로드
        if st.session_state.get('uploaded_df') is not None:
            original_df = st.session_state.uploaded_df.copy()
            original_df['predicted_class'] = [r['predicted_class'] for r in results]
            
            combined_csv = original_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=" 원본+예측결과 통합 CSV 다운로드",
                data=combined_csv,
                file_name="combined_results.csv",
                mime="text/csv"
            )