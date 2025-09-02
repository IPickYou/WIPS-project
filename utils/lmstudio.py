import requests
import streamlit as st
import time

def settings():
    with st.expander("LM Studio API 설정"):
        col_api1, col_api2 = st.columns(2)
        with col_api1:
            st.session_state.api_url = st.text_input(
                "API URL", 
                value=st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions")
            )
        with col_api2:
            st.session_state.api_model = st.text_input(
                "모델 이름", 
                value=st.session_state.get("api_model", "llama-3.2-1b")
            )
        
        # API 테스트
        if st.button("API 연결 테스트"):
            try:
                test_response = requests.post(
                    st.session_state.api_url,
                    json={
                        "model": st.session_state.api_model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10
                    },
                    timeout=10
                )
                if test_response.status_code == 200:
                    st.success("API 연결 성공!")
                else:
                    st.error(f"API 연결 실패: {test_response.status_code}")
            except Exception as e:
                st.error(f"API 연결 실패: {str(e)}")
                
def inference(selected_columns, df, custom_separator, auto_prompt):
    if st.button("분류 실행", type="primary"):
        with st.spinner("분류를 실행하는 중..."):
            # 데이터 준비
            if len(selected_columns) == 1:
                data_to_classify = df[selected_columns[0]].dropna().astype(str).tolist()
            else:
                clean_df = df[selected_columns].dropna()
                data_to_classify = clean_df.apply(
                    lambda row: custom_separator.join([str(row[col]) for col in selected_columns]),
                    axis=1
                ).tolist()
            
            # 빈 문자열 제거
            data_to_classify = [text for text in data_to_classify if text.strip()]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(data_to_classify):
                try:
                    # LM Studio API 호출
                    response = requests.post(
                        st.session_state.api_url,
                        json={
                            "model": st.session_state.api_model,
                            "messages": [
                                {"role": "user", "content": auto_prompt.format(text=text)}
                            ],
                            "max_tokens": 100,
                            "temperature": 0.1
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        classification = result['choices'][0]['message']['content'].strip()
                        results.append({
                            'index': i,
                            'text': text,
                            'classification': classification,
                            'text_preview': text[:100] + "..." if len(text) > 100 else text
                        })
                    else:
                        results.append({
                            'index': i,
                            'text': text,
                            'classification': "오류",
                            'text_preview': text[:100] + "..." if len(text) > 100 else text
                        })
                        
                except Exception as e:
                    results.append({
                        'index': i,
                        'text': text,
                        'classification': f"오류: {str(e)}",
                        'text_preview': text[:100] + "..." if len(text) > 100 else text
                    })
                
                progress_bar.progress((i + 1) / len(data_to_classify))
                time.sleep(0.1)  # API 부하 방지
            
            st.session_state.classification_results = results
            st.success("분류가 완료되었습니다!")