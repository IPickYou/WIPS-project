from io import BytesIO

import pandas as pd
import streamlit as st
import time

def show(results_df, classification_groups):
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