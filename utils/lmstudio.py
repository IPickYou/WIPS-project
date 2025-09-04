import requests
import streamlit as st
import time

def settings():

    with st.expander("**LM STUDIO SETTING**"):

        col_api1, col_api2 = st.columns(2)

        with col_api1:
            api_url_input = st.text_input(
                "BASE URL", 
                value = st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions")
            )
            st.session_state.api_url = api_url_input
            
        with col_api2:
            api_model_input = st.text_input(
                "MODEL", 
                value = st.session_state.get("api_model", "qwen/qwen3-14b")
            )
            st.session_state.api_model = api_model_input

        if st.button("API CONNECTION"):

            try:

                test_response = requests.post(
                    st.session_state.api_url,
                    json = {
                        "model": st.session_state.api_model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10
                    },
                    timeout = 30
                )

                if test_response.status_code != 200:
                    st.error(f"API CONNECTION FAILED ({test_response.status_code})")
                    st.session_state.api_connection_success = False
                    return
                
                response_data = test_response.json()
                actual_model = response_data.get("model", "")

                if actual_model != st.session_state.api_model:
                    st.error(
                        f"""API CONNECTION FAILED  
                        - REQUEST MODEL : {st.session_state.api_model}  
                        - LOAD MODEL : {actual_model}"""
                    )
                    st.session_state.api_connection_success = False
                    return

                st.success("API CONNECTION SUCCESSFUL")
                st.session_state.api_connection_success = True

            except Exception as e:
                st.error("API CONNECTION FAILED")
                st.error(e)
                st.session_state.api_connection_success = False

def get_score_for_candidate(text, code, desc, step1_prompt, api_url, api_model):

    prompt = step1_prompt.format(text = text, code = code, desc = desc)
    
    try:
        response = requests.post(
            api_url,
            json = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "당신은 특허를 CPC 체계로 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 8
            },
            timeout = 30
        )
        
        if response.status_code == 200:
            result = response.json()
            score_text = result['choices'][0]['message']['content'].strip()
            try:
                score = float(score_text)
                if score < 0.0 or score > 10.0:
                    score = 0.0
            except ValueError:
                score = 0.0

        else:
            score = 0.0
            
    except Exception as e:
        st.error(e)
        score = 0.0
    
    return score

def reselect_best_code(text, candidate_codes, cpc_candidates, step2_prompt, api_url, api_model):

    candidate_text = "\n".join([f"{code} : {cpc_candidates[code]}" for code in candidate_codes])
    prompt = step2_prompt.format(text = text, candidate_text = candidate_text)
    
    try:
        response = requests.post(
            api_url,
            json = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "당신은 특허를 CPC 체계로 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 15
            },
            timeout = 30
        )
        
        if response.status_code == 200:
            result = response.json()
            final_choice = result['choices'][0]['message']['content'].strip()
            for code in candidate_codes:
                if code in final_choice:
                    return code
            return "ERROR"
        
        else:
            return "ERROR"
            
    except Exception as e:
        st.error(e)
        return "ERROR"

def inference(selected_columns, df, custom_separator, step1_prompt, step2_prompt, cpc_candidates):

    if not st.session_state.get('api_connection_success', False):
        st.warning("Please complete API CONNECTION first.")
        return

    if st.button("**C L A S S I F Y**", type = "primary", use_container_width = True):

        if not hasattr(st.session_state, 'api_url') or not hasattr(st.session_state, 'api_model'):
            st.error("Please set up the LM STUDIO API first.")
            return
        
        with st.spinner("CLASSIFICATION IN PROGRESS ..."):

            if len(selected_columns) == 1:
                data_to_classify = df[selected_columns[0]].dropna().astype(str).tolist()
                
            else:
                clean_df = df[selected_columns].dropna()
                data_to_classify = clean_df.apply(
                    lambda row: custom_separator.join([str(row[col]) for col in selected_columns]),
                    axis = 1
                ).tolist()

            data_to_classify = [text for text in data_to_classify if text.strip()]
            
            if not data_to_classify:
                st.error("No data available for classification.")
                return
            
            results = []

            progress_bar = st.progress(0)
            
            for i, text in enumerate(data_to_classify):

                try:

                    scores = {}

                    for code, desc in cpc_candidates.items():
                        score = get_score_for_candidate(text, code, desc, step1_prompt, st.session_state.api_url, st.session_state.api_model)
                        scores[code] = score
                        time.sleep(0.5)

                    if scores:
                        max_score = max(scores.values())
                        candidates_with_max_score = [code for code, s in scores.items() if s == max_score]

                        if len(candidates_with_max_score) == 1:
                            classification = candidates_with_max_score[0]

                        else:
                            classification = reselect_best_code(text, candidates_with_max_score, cpc_candidates, step2_prompt, st.session_state.api_url, st.session_state.api_model)
                            time.sleep(0.5)

                    else:
                        classification = "ERROR"

                    results.append({
                        'index': i,
                        'text': text,
                        'classification': classification,
                        'text_preview': text[:100] + "..." if len(text) > 100 else text,
                        'scores': scores
                    })
                        
                except Exception as e:
                    results.append({
                        'index': i,
                        'text': text,
                        'classification': "ERROR",
                        'text_preview': text[:100] + "..." if len(text) > 100 else text,
                        'scores': {}
                    })
                
                progress_bar.progress((i + 1) / len(data_to_classify))

                time.sleep(1.0)
            
            st.session_state.classification_results = results
            st.toast("CLASSIFICATION IS COMPLETE")