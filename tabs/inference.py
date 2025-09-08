# inference.py

from utils.finetuning_classifier import FineTuningClassifier
from utils import excel_download
import os
import glob
import streamlit as st
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def show():
    with st.expander("**COLUMNS TO USE FOR INFERENCE**", expanded=True):

        df = st.session_state.uploaded_df

        selected_cols = st.multiselect(
            "SELECTED COLUMNS",
            options=df.columns.tolist(),
            default=[col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns.tolist()],
            key="inference_cols"
        )

        if not selected_cols:
            st.warning("Please select at least one column.")

    with st.expander("**MODEL TO USE FOR INFERENCE**", expanded=False):

        model_selection_method = st.radio(
            "MODEL SELECTION METHOD",
            ["AUTOMATIC SEARCH", "MANUAL PATH ENTRY"],
            key="model_selection_method"
        )

        if model_selection_method == "MANUAL PATH ENTRY":
            model_path = st.text_input(
                "병합된 모델의 경로를 입력하세요.",
                value=r"C:\company\wips\excel_gemma_2_2b\merged_model",
                help="학습 완료 후 생성된 merged_model 폴더의 전체 경로를 입력하세요."
            )
        else:
            base_dir = r"C:\company\wips"

            if os.path.exists(base_dir):
                try:
                    # merged_model 폴더 검색
                    merged_model_paths = glob.glob(os.path.join(base_dir, "*", "merged_model"))

                    valid_models = []
                    for merged_path in merged_model_paths:
                        # config.json 또는 pytorch_model.bin 파일 존재 확인
                        if (os.path.exists(os.path.join(merged_path, 'config.json')) or
                                os.path.exists(os.path.join(merged_path, 'pytorch_model.bin')) or
                                os.path.exists(os.path.join(merged_path, 'model.safetensors'))):
                            # 상위 폴더 이름으로 표시
                            parent_dir = os.path.basename(os.path.dirname(merged_path))
                            valid_models.append((parent_dir, merged_path))

                    if valid_models:
                        # 사용자에게 보여줄 이름과 실제 경로 분리
                        model_names = [name for name, path in valid_models]
                        selected_model_name = st.selectbox(
                            "검색된 병합 모델 중 하나를 선택하세요.",
                            options=model_names
                        )

                        # 선택된 모델의 실제 경로 찾기
                        model_path = next(path for name, path in valid_models if name == selected_model_name)

                        st.info(f"선택된 모델 경로: {model_path}")
                    else:
                        st.warning("No merged model could be found using automatic search.")
                        model_path = st.text_input(
                            "병합된 모델의 경로를 직접 입력하세요.",
                            value=r"C:\company\wips\excel_gemma_2_2b\merged_model"
                        )
                except Exception as e:
                    st.error(f"모델 검색 중 오류 발생: {e}")
                    model_path = st.text_input(
                        "병합된 모델의 경로를 직접 입력하세요.",
                        value=r"C:\company\wips\excel_gemma_2_2b\merged_model"
                    )
            else:
                st.error(f"The default directory does not exist. : {base_dir}")
                model_path = st.text_input(
                    "병합된 모델의 경로를 직접 입력하세요.",
                    value=r"C:\company\wips\excel_gemma_2_2b\merged_model"
                )

        model_exists = False

        if model_path and os.path.exists(model_path):
            # 병합 모델 파일들 존재 확인
            config_exists = os.path.exists(os.path.join(model_path, 'config.json'))
            model_file_exists = (os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or
                                 os.path.exists(os.path.join(model_path, 'model.safetensors')))

            if config_exists and model_file_exists:
                model_exists = True
                st.success("병합된 모델을 사용할 수 있습니다.")

                # 라벨 정보 표시 (상위 폴더에서 확인)
                parent_dir = os.path.dirname(model_path)
                label_file_path = os.path.join(parent_dir, 'label_mappings.pkl')

                if os.path.exists(label_file_path):
                    try:
                        import pickle
                        with open(label_file_path, 'rb') as f:
                            mappings = pickle.load(f)
                            model_labels = mappings['labels_list']

                            with st.expander("**LABELS FOR THE TRAINED MODEL**", expanded=False):
                                st.write(sorted(model_labels))

                    except Exception as e:
                        st.warning(f"라벨 정보를 읽을 수 없습니다: {e}")
                else:
                    st.warning("라벨 정보 파일을 찾을 수 없습니다.")
            else:
                st.warning("병합된 모델 파일이 존재하지 않습니다.")
        else:
            st.warning("지정된 경로에 모델이 존재하지 않습니다.")

    if model_exists:
        with st.expander("**HYPERPARAMETER**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                chunk_max_length = st.number_input(
                    "MAX LENGTH", min_value=128, max_value=1024, value=512, key="chunk_max_length"
                )
            with col2:
                chunk_stride = st.number_input(
                    "STRIDE", min_value=10, max_value=100, value=50, key="chunk_stride"
                )

    if st.button("**I N F E R E N C E**", type="primary", use_container_width=True, disabled=not model_exists):
        try:
            model_name = st.session_state.get('ft_model_name', 'google/gemma-2-2b')
            hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')

            classifier = FineTuningClassifier(model_name, hf_token)

            with st.spinner("LOADING MERGED MODEL ..."):
                classifier.load_model(model_path, is_merged_model=True)

            with st.spinner("RUNNING INFERENCE ..."):
                results_df = classifier.predict_patents(
                    df, model_path,
                    selected_cols=selected_cols,
                    max_length=chunk_max_length,
                    stride=chunk_stride
                )

            st.toast("INFERENCE IS COMPLETE")

            st.subheader("INFERENCE RESULT")
            st.dataframe(results_df, use_container_width=True)

            st.subheader("PREDICTION DISTRIBUTION")
            pred_counts = results_df['예측_라벨'].value_counts()
            st.bar_chart(pred_counts)

            st.session_state.inference_results = results_df
            excel_download.show_finetuning(results_df)

        except Exception as e:
            st.error(f"추론 중 오류 발생: {e}")
            st.code(str(e))