from io import BytesIO

import pandas as pd
import requests
import streamlit as st
import time

def show():
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