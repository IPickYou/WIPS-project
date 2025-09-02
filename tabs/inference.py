import os
import streamlit as st

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
        with st.spinner("모델을 로드하는 중..."):
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                # 모델 로드 (예시 코드 방식)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    device_map="cuda",
                    token=hf_token,
                    num_labels=num_labels,
                    offload_folder="C:/aimer/wips/temp_offload"
                )
                
                # 모델 evaluation 모드로 설정
                model.eval()
                
                # 토크나이저 로드
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
                
                st.success("모델 로드가 성공적으로 완료되었습니다!")

            except Exception as e:
                import traceback
                st.error(f"모델 로드 실패: {str(e)}")
                st.code(traceback.format_exc())
                
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
        df = st.session_state.uploaded_df
        
        st.write("사이드바에서 업로드된 데이터 미리보기:")
        st.dataframe(df.head())
        st.info(f"총 {len(df)}개의 행이 있습니다.")

        # 칼럼 선택
        text_column = st.selectbox(
            "추론에 사용할 텍스트 컬럼",
            df.columns.tolist(),
            key="inference_text_column"
        )

        if text_column:
            # 데이터 준비
            data_to_infer = df[text_column].dropna().astype(str).tolist()
            st.info(f" {len(data_to_infer)}개의 샘플이 추론 준비되었습니다.")

            # 추론 설정
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input(
                    "배치 크기",
                    min_value=1,
                    max_value=32,
                    value=8,
                    step=1,
                    key="inference_batch_size"
                )
            with col2:
                max_length = st.number_input(
                    "최대 토큰 길이",
                    min_value=128,
                    max_value=1024,
                    value=256,
                    step=32,
                    key="max_length"
                )

            # 추론 실행 버튼
            if st.button("추론 실행", type="primary"):
                with st.spinner("추론 실행 중..."):
                    progress_bar = st.progress(0)
                    
                    model = st.session_state.model
                    tokenizer = st.session_state.tokenizer
                    device = next(model.parameters()).device
                    
                    # 예시 코드와 동일한 방식으로 처리
                    preds = []
                    
                    for i in range(0, len(data_to_infer), batch_size):
                        batch_texts = data_to_infer[i : i + batch_size]
                        
                        try:
                            encodings = tokenizer(
                                batch_texts,
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                return_tensors="pt",
                            )
                            encodings = {k: v.to(device) for k, v in encodings.items()}

                            with torch.no_grad():
                                outputs = model(**encodings)
                                logits = outputs.logits
                                batch_preds = torch.argmax(logits, dim=-1).tolist()
                                preds.extend(batch_preds)
                                
                        except Exception as e:
                            # 배치 처리 실패 시
                            for _ in batch_texts:
                                preds.append(f"오류: {str(e)}")

                        progress_bar.progress((i + len(batch_texts)) / len(data_to_infer))

                    # 결과 정리
                    results = []
                    for j, text in enumerate(data_to_infer):
                        results.append({
                            'original_text': text,
                            'predicted_class': preds[j] if j < len(preds) else "오류",
                        })

                    st.session_state.inference_results = results
                    st.success(" 추론이 완료되었습니다!")
                    st.write(f"예측 결과: {preds}")

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