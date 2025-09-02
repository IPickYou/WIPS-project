import pandas as pd
import streamlit as st

def show():
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
        
    return classification_method