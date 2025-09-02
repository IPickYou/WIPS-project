from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from trl import SFTTrainer, SFTConfig

import os
import pandas as pd
import streamlit as st

def show():
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