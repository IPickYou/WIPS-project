from components.prompt_engineering import category_settings, prompt_settings
from utils import excel_download

import pandas as pd
import streamlit as st

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
        category_settings.show()
        
        st.markdown("---")
        
        # 자동 생성된 프롬프트 미리보기
        prompt_settings.show(selected_columns, df, custom_separator)
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
        excel_download.show(results_df, classification_groups)
        
        # 분류 통계 차트
        if len(results) > 1:
            st.write("**분류 통계**")
            classification_counts = results_df['classification'].value_counts()
            st.bar_chart(classification_counts)