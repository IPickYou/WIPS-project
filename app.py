import streamlit as st
import pandas as pd
import requests
import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from io import StringIO, BytesIO
import time
import openpyxl
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
from dorenv import load_dotenv

load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="특허 분류기",
    layout="wide"
)

# 메인 타이틀
st.title(" 특허 분류기")
st.markdown("---")

# 세션 상태 초기화
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
        "C01B": "비금속 원소, 비금속 화합물 (예: 수소, 질소, 산소 관련 화합물)",
        "C01C": "무기산, 무기산의 염 (예: 황산, 질산, 인산 등)",
        "C01D": "할로겐 화합물 (예: 염소, 브롬, 플루오르 화합물)",
        "C01F": "알칼리 금속, 알칼리 토금속, 희토류 금속 화합물",
        "C01G": "귀금속, 기타 금속 화합물"
    }

# 사이드바 설정
st.sidebar.title("설정")

# 1) 작업 방법 선택
classification_method = st.sidebar.selectbox(
    "작업 선택",
    ["프롬프트 엔지니어링", "학습 및 추론"]
)

st.sidebar.markdown("---")

# 2) 데이터 업로드
st.sidebar.subheader("데이터 업로드")
uploaded_file = st.sidebar.file_uploader(
    "CSV 또는 Excel 파일을 업로드하세요", 
    type=['csv', 'xlsx', 'xls'],
    key="data_upload"
)

# 파일 처리
if uploaded_file is not None:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.sidebar.selectbox("시트 선택", sheet_names)
            else:
                selected_sheet = sheet_names[0]
            
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        
        st.session_state.uploaded_df = df
        st.sidebar.success(f"파일 업로드 완료 ({len(df)} 행)")
        
    except Exception as e:
        st.sidebar.error(f"파일 처리 오류: {str(e)}")

st.sidebar.markdown("---")

# 3) 사용법
st.sidebar.subheader("사용법")
if classification_method == "프롬프트 엔지니어링":
    st.sidebar.markdown("""
    **프롬프트 엔지니어링**
    1. 데이터 파일 업로드
    2. 분류 카테고리 관리
    3. 칼럼 선택
    4. 프롬프트 작성 및 실행
    5. 카테고리별 결과 확인
    """)
else:
    st.sidebar.markdown("""
    **학습 및 추론**
    1. 데이터 파일 업로드
    2. 학습 탭: 모델 학습 실행
    3. 추론 탭: 학습된 모델로 추론
    4. 결과 다운로드
    """)

# 메인 컨텐츠 영역
if classification_method == "프롬프트 엔지니어링":
    st.subheader("프롬프트 엔지니어링")
    
    # 업로드된 데이터 확인
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**업로드된 데이터 미리보기**")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"총 {len(df)} 행, {len(df.columns)} 열")
            
        with col2:
            # 칼럼 선택
            st.write("**칼럼 선택**")
            selected_columns = st.multiselect(
                "분석할 칼럼들을 선택하세요",
                df.columns.tolist(),
                help="선택된 칼럼들의 내용이 결합되어 프롬프트에 전달됩니다"
            )
            
            # 컬럼 결합 방식 선택
            if len(selected_columns) > 1:
                combine_method = st.selectbox(
                    "컬럼 결합 방식",
                    ["공백으로 연결", "줄바꿈으로 연결", "커스텀 구분자"]
                )
                
                if combine_method == "커스텀 구분자":
                    custom_separator = st.text_input("구분자 입력", value=" | ")
                else:
                    custom_separator = " " if combine_method == "공백으로 연결" else "\n"
            else:
                custom_separator = ""
        
        st.markdown("---")
        
        # 통합된 카테고리 및 프롬프트 관리
        st.write("**카테고리 및 프롬프트 관리**")
        
        # 현재 카테고리들 표시 및 편집
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
        
        st.markdown("---")
        
        # 자동 생성된 프롬프트 미리보기
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
        with st.expander("LM Studio API 설정"):
            col_api1, col_api2 = st.columns(2)
            with col_api1:
                api_url = st.text_input(
                    "API URL", 
                    value="http://localhost:1234/v1/chat/completions"
                )
            with col_api2:
                api_model = st.text_input(
                    "모델 이름", 
                    value="llama-3.2-1b"
                )
            
            # API 테스트
            if st.button("API 연결 테스트"):
                try:
                    test_response = requests.post(
                        api_url,
                        json={
                            "model": api_model,
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
        
        # 분류 실행
        if selected_columns and len(st.session_state.categories) > 0:
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
                                api_url,
                                json={
                                    "model": api_model,
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
cd 'C:/aimer/wips/WIPS-project'
                                'text': text,
                                'classification': f"오류: {str(e)}",
                                'text_preview': text[:100] + "..." if len(text) > 100 else text
                            })
                        
                        progress_bar.progress((i + 1) / len(data_to_classify))
                        time.sleep(0.1)  # API 부하 방지
                    
                    st.session_state.classification_results = results
                    st.success("분류가 완료되었습니다!")
        else:
            st.warning("칼럼을 선택하고 최소 하나의 카테고리가 있어야 합니다.")
    
    else:
        st.info("먼저 사이드바에서 데이터 파일을 업로드해주세요.")
    
    # 결과 표시 (카테고리별)
    if st.session_state.classification_results:
        st.markdown("---")
        st.subheader("분류 결과")
        
        results = st.session_state.classification_results
        results_df = pd.DataFrame(results)
        
        # 전체 결과 개요
        col_overview1, col_overview2, col_overview3 = st.columns(3)
        with col_overview1:
            st.metric("총 분류 건수", len(results))
        with col_overview2:
            unique_categories = results_df['classification'].nunique()
            st.metric("분류된 카테고리 수", unique_categories)
        with col_overview3:
            error_count = len([r for r in results if "오류" in r['classification']])
            st.metric("오류 건수", error_count)
        
        # 카테고리별 결과 표시
        st.write("**카테고리별 분류 결과**")
        
        # 분류 결과 그룹화
        classification_groups = results_df.groupby('classification')
        
        for category, group in classification_groups:
            with st.expander(f" {category} ({len(group)}건)", expanded=True):
                # 해당 카테고리의 결과만 표시
                display_df = group[['text_preview', 'classification']].copy()
                display_df.columns = ['텍스트 미리보기', '분류 결과']
                st.dataframe(display_df, use_container_width=True)
        
        # 전체 결과 다운로드
        st.write("**결과 다운로드**")

        download_df = results_df[['text', 'classification']].copy()
        download_df.columns = ['원본 텍스트', '분류 결과']

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            download_df.to_excel(writer, sheet_name='전체결과', index=False)
            
            # 카테고리별 시트 추가
            for category, group in classification_groups:
                safe_name = category.replace('/', '_').replace(':', '_')[:31]  # 시트명 길이 제한
                category_df = group[['text', 'classification']].copy()
                category_df.columns = ['원본 텍스트', '분류 결과']
                category_df.to_excel(writer, sheet_name=safe_name, index=False)
            
            # 통계 시트 추가
            stats_df = results_df['classification'].value_counts().reset_index()
            stats_df.columns = ['분류', '개수']
            stats_df.to_excel(writer, sheet_name='통계', index=False)

        excel_buffer.seek(0)
        st.download_button(
            label="Excel 다운로드",
            data=excel_buffer.getvalue(),
            file_name=f"patent_classification_prompt_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 분류 통계 차트
        if len(results) > 1:
            st.write("**분류 통계**")
            classification_counts = results_df['classification'].value_counts()
            st.bar_chart(classification_counts)

elif classification_method == "학습 및 추론":
    st.subheader("학습 및 추론")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["추론", "학습"])

    # --- 추론 탭 ---
    with tab1:
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
                value=os.getenv(HF_TOKEN)",
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
    # --- 학습 탭 ---
    with tab2:
        st.write("### 모델 학습")

        if st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df

            # 데이터 미리보기
            st.write("**학습 데이터 미리보기**")
            st.dataframe(df.head())
            st.caption(f"총 {len(df)} 행, {len(df.columns)} 열")

            # 텍스트 & 라벨 컬럼 선택
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox("텍스트 컬럼", df.columns.tolist())
            with col2:
                label_column = st.selectbox("라벨 컬럼", df.columns.tolist())

            if text_column and label_column:
                clean_df = df[[text_column, label_column]].dropna()
                clean_df[text_column] = clean_df[text_column].astype(str)

                # 라벨 처리
                st.write("**라벨 분포**")
                label_series = clean_df[label_column]
                if isinstance(label_series, pd.DataFrame):
                    label_series = label_series.iloc[:, 0]

                label_counts = label_series.value_counts()
                st.bar_chart(label_counts)

                # 샘플 1개인 클래스 제거
                rare_labels = label_counts[label_counts < 2].index.tolist()
                if rare_labels:
                    st.warning(f"샘플 1개인 라벨 제거: {rare_labels}")
                    clean_df = clean_df[~clean_df[label_column].isin(rare_labels)]
                    label_series = clean_df[label_column]

                if isinstance(label_series, pd.DataFrame):
                    label_series = label_series.iloc[:, 0]  # Series로 변환

                unique_labels = sorted(label_series.unique())
                label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                id_to_label = {idx: label for label, idx in label_to_id.items()}

# 이제 Series이므로 map이 정상 작동
                clean_df['label_id'] = label_series.map(label_to_id)

                st.write("**라벨 매핑**")
                for label, idx in label_to_id.items():
                    st.write(f"  - {label} → {idx}")

                # 하이퍼파라미터
                st.write("**하이퍼파라미터 설정**")
                col_param1, col_param2, col_param3 = st.columns(3)

                with col_param1:
                    base_model_id = st.selectbox(
                        "베이스 모델 선택",
                        ["meta-llama/Llama-3.2-1B", "bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
                        index=0,
                        key="training_base_model"
                    )
                    custom_model_id = st.text_input("직접 모델 입력 (HuggingFace Hub)", value="")
                    if custom_model_id.strip():
                        base_model_id = custom_model_id.strip()

                    learning_rate = st.number_input("학습률", value=2e-5, format="%.0e", key="training_lr")
                    num_epochs = st.number_input("에포크 수", value=3, min_value=1, max_value=20, key="training_epochs")

                with col_param2:
                    batch_size = st.number_input("배치 크기", value=2, min_value=1, max_value=16, key="training_batch_size")
                    max_length = st.number_input("최대 토큰 길이", value=512, min_value=128, max_value=2048, key="training_max_length")
                    train_ratio = st.slider("학습 데이터 비율", 0.6, 0.9, 0.8, 0.05, key="training_ratio")

                with col_param3:
                    lora_r = st.number_input("LoRA r", value=16, min_value=8, max_value=128, key="training_lora_r")
                    lora_alpha = st.number_input("LoRA alpha", value=32, min_value=16, max_value=256, key="training_lora_alpha")
                    lora_dropout = st.number_input("LoRA dropout", value=0.1, min_value=0.0, max_value=0.5, key="training_lora_dropout")

                output_dir = st.text_input("출력 디렉토리", value="./output", key="training_output_dir")
                model_name = st.text_input("저장할 모델 이름", value="my_model", key="training_model_name")

                # 데이터 분할
                clean_df['label_id'] = label_series.map(label_to_id)
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    clean_df[text_column].values.tolist(),
                    clean_df['label_id'].values.tolist(),
                    train_size=train_ratio,
                    random_state=42
                )

                st.write(f"**데이터 분할**: 학습 {len(train_texts)}개, 테스트 {len(test_texts)}개")

                # 학습 시작
                if st.button("학습 시작", type="primary"):
                    with st.spinner("모델 학습을 진행합니다..."):
                        try:
                            # Tokenizer
                            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=False)
                            if tokenizer.pad_token is None:
                                if tokenizer.eos_token:
                                    tokenizer.pad_token = tokenizer.eos_token
                                else:
                                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                            tokenizer.padding_side = 'right'

                            # Dataset
                            train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
                            test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

                            def preprocess_function(examples):
                                tokenized = tokenizer(
                                    examples['text'],
                                    truncation=True,
                                    max_length=max_length,
                                    padding=True
                                )
                                tokenized['labels'] = examples['label']
                                return tokenized

                            tokenized_train = train_dataset.map(preprocess_function, batched=True)
                            tokenized_test = test_dataset.map(preprocess_function, batched=True)
                            tokenized_train = tokenized_train.remove_columns(['text', 'label'])
                            tokenized_test = tokenized_test.remove_columns(['text', 'label'])

                            # Model
                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='float16',
                                bnb_4bit_use_double_quant=True
                            )
                            model = AutoModelForSequenceClassification.from_pretrained(
                                base_model_id,
                                num_labels=len(unique_labels),
                                device_map='auto',
                                quantization_config=bnb_config
                            )
                            model.config.use_cache = False
                            model.config.pad_token_id = tokenizer.pad_token_id

                            # PEFT
                            peft_config = LoraConfig(
                                lora_alpha=int(lora_alpha),
                                lora_dropout=lora_dropout,
                                r=int(lora_r),
                                bias='none',
                                task_type='SEQ_CLS',
                                target_modules=['k_proj','gate_proj','v_proj','up_proj','q_proj','o_proj','down_proj']
                            )
                            model = prepare_model_for_kbit_training(model)
                            model = get_peft_model(model, peft_config)

                            # SFTConfig 생성 (Streamlit 하이퍼파라미터 반영)
                            sft_args = SFTConfig(
                                output_dir=os.path.join(output_dir, model_name),
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                gradient_accumulation_steps=2,
                                optim='paged_adamw_32bit',
                                lr_scheduler_type='cosine',
                                num_train_epochs=num_epochs,
                                warmup_steps=50,
                                logging_steps=10,
                                fp16=True,
                                gradient_checkpointing=True,
                                dataset_text_field='text',
                                max_length=max_length,
                                label_names=['labels']
                            )

                            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                            # Trainer
                            trainer = SFTTrainer(
                                model=model,
                                train_dataset=tokenized_train,
                                eval_dataset=tokenized_test,
                                processing_class=tokenizer,
                                args=sft_args,
                                data_collator=data_collator,
                                peft_config=peft_config
                            )

                            trainer.train()
                            trainer.save_model(os.path.join(output_dir, model_name))
                            tokenizer.save_pretrained(os.path.join(output_dir, model_name))

                            st.success(f"모델 학습 완료! 저장 경로: {os.path.join(output_dir, model_name)}")

                        except Exception as e:
                            st.error(f"학습 중 오류 발생: {str(e)}")
                            st.exception(e)
        else:
            st.info("먼저 사이드바에서 학습 데이터를 업로드해주세요.")


