# inference.py

from utils.transformers_settings import FineTuningClassifier
from utils import excel_download
import os
import streamlit as st
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def show():

    with st.expander("**COLUMNS TO USE FOR INFERENCE**", expanded = True):

        df = st.session_state.uploaded_df
        
        selected_cols = st.multiselect(
            "SELECTED COLUMNS",
            options = df.columns.tolist(),
            default = [col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns.tolist()],
            key = "inference_cols"
        )
        
        if not selected_cols:
            st.warning("Please select at least one column.")

    with st.expander("**MODEL TO USE FOR INFERENCE**", expanded = False):

        model_selection_method = st.radio(
            "MODEL SELECTION METHOD",
            ["AUTOMATIC SEARCH", "MANUAL PATH ENTRY"],
            key = "model_selection_method"
        )
        
        if model_selection_method == "MANUAL PATH ENTRY":
            model_path = st.text_input(
                "원하는 모델의 경로를 입력하세요.",
                value = r"C:\company\wips\excel_gemma_2_2b",
                help = "학습시킨 모델이 저장되어 있는 전체 경로를 입력하세요."
            )
        else:
            base_dir = r"C:\company\wips"
            
            if os.path.exists(base_dir):
                try:
                    all_items = [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))]
                    
                    valid_models = []
                    for item in all_items:
                        item_path = os.path.join(base_dir, item)
                        if os.path.exists(os.path.join(item_path, 'label_mappings.pkl')):
                            valid_models.append(item)
                    
                    if valid_models:
                        selected_model = st.selectbox(
                            "검색된 모델 중 하나를 선택하세요.",
                            options = valid_models
                        )
                        model_path = os.path.join(base_dir, selected_model)
                    else:
                        st.warning("No model could be found using automatic search.")
                        model_path = st.text_input(
                            "모델의 경로를 직접 입력하세요.",
                            value = r"C:\company\wips\excel_gemma_2_2b"
                        )
                except Exception as e:
                    st.error(e)
                    model_path = st.text_input(
                        "모델의 경로를 직접 입력하세요.",
                        value = r"C:\company\wips\excel_gemma_2_2b"
                    )
            else:
                st.error(f"The default directory does not exist. : {base_dir}")
                model_path = st.text_input(
                    "모델의 경로를 직접 입력하세요.",
                    value = r"C:\company\wips\excel_gemma_2_2b"
                )
        
        model_exists = False
        
        if model_path and os.path.exists(model_path):
            label_file_path = os.path.join(model_path, 'label_mappings.pkl')
            
            if os.path.exists(label_file_path):
                model_exists = True
                st.success("A model is available.")
                
                try:
                    import pickle
                    with open(label_file_path, 'rb') as f:
                        mappings = pickle.load(f)
                        model_labels = mappings['labels_list']
                        
                        with st.expander("**LABELS FOR THE TRAINED MODEL**", expanded = False):
                            st.write(sorted(model_labels))
                                
                except Exception as e:
                    st.error(e)
            else:
                st.warning("No model is available.")
        else:
            st.warning("No model is available.")
        
    if model_exists:
        with st.expander("**HYPERPARAMETER**", expanded = False):
            col1, col2 = st.columns(2)
            with col1:
                chunk_max_length = st.number_input("MAX LENGTH", min_value = 128, max_value = 1024, value = 512, key = "chunk_max_length")
            with col2:
                chunk_stride = st.number_input("STRIDE", min_value = 10, max_value = 100, value = 50, key = "chunk_stride")

    if st.button("**I N F E R E N C E**", type = "primary", use_container_width = True, disabled = not model_exists):
        try:
            model_name = st.session_state.get('ft_model_name', 'google/gemma-2-2b')
            hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')
            
            classifier = FineTuningClassifier(model_name, hf_token)
            
            with st.spinner("LOADING MODEL ..."):
                classifier.load_model(model_path)
            
            with st.spinner("RUNNING INFERENCE ..."):
                results_df = classifier.predict_patents(
                    df, model_path, 
                    selected_cols = selected_cols,
                    max_length = chunk_max_length,
                    stride = chunk_stride
                )
            
            st.toast("INFERENCE IS COMPLETE")
            
            st.subheader("INFERENCE RESULT")
            st.dataframe(results_df, use_container_width = True)
            
            st.subheader("PREDICTION DISTRIBUTION")
            pred_counts = results_df['예측_라벨'].value_counts()
            st.bar_chart(pred_counts)
            
            st.session_state.inference_results = results_df
            
            excel_download.show_finetuning(results_df)
            
        except Exception as e:
            st.error(e)
            st.code(str(e))