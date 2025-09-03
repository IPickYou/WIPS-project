import streamlit as st
import torch

def load(model_id, hf_token, num_labels, tokenizer_path):
    with st.spinner("모델을 로드하는 중..."):
        try:
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
            
def inference():
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