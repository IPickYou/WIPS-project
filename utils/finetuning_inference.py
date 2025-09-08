# utils/finetuning_inference.py

import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
import pickle
from safetensors.torch import load_file
from dotenv import load_dotenv
from utils.data_proceesor import DataProcessor

# .env 파일 로드
load_dotenv()

class FineTuningInference:
    def __init__(self, model_name="google/gemma-2-2b", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
        
    def load_model(self, model_path, manual_labels=None, is_merged_model=False):
        """병합된 모델 로드 옵션 추가 및 패딩 토큰 보정"""

        if not model_path or not os.path.exists(model_path):
            raise ValueError(model_path)

        # 병합된 모델 경로 확인
        merged_model_path = os.path.join(model_path, "merged_model")
        if not is_merged_model and os.path.exists(merged_model_path):
            model_path = merged_model_path
            is_merged_model = True

        label_file = os.path.join(model_path, 'label_mappings.pkl')

        if os.path.exists(label_file):
            try:
                with open(label_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.labels_list = mappings['labels_list']
                    self.label2id = mappings['label2id']
                    self.id2label = mappings['id2label']
            except Exception as e:
                raise ValueError(e)
        elif manual_labels:
            self.labels_list = sorted(manual_labels)
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        else:
            self.labels_list = ['CPC_C01B', 'CPC_C01C', 'CPC_C01D', 'CPC_C01F', 'CPC_C01G']
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}

        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=self.hf_token,
                trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        try:
            if is_merged_model:
                # 병합된 모델 직접 로드
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    token=self.hf_token,
                    num_labels=len(self.labels_list),
                    device_map='auto',
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                print("병합된 모델을 로드했습니다.")
            else:
                # 기존 방식 (베이스 모델 + 어댑터)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='float16',
                    bnb_4bit_use_double_quant=True
                )

                # 베이스 모델을 SEQ_CLS로 로드
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    num_labels=len(self.labels_list),
                    device_map='auto',
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )

                # 어댑터 로드 및 병합
                self.model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    device_map='auto'
                )

                # 병합
                self.model = self.model.merge_and_unload()
                print("어댑터를 병합하여 로드했습니다.")

            self.model.eval()

        except Exception as e:
            raise ValueError(e)

    def predict_patents(self, df, model_path=None, selected_cols=None, max_length=512, stride=50):
        try:
            # -------------------------------
            # 모델 로드
            # -------------------------------
            if model_path and not self.model:
                self.load_model(model_path)

            if not self.model or not self.tokenizer:
                raise ValueError("모델이 로드되지 않았습니다.")

            if selected_cols is None:
                selected_cols = ["발명의 명칭", "요약", "전체청구항"]

            processed_df = DataProcessor.prepare_data(self, df, selected_cols)
            
            chunker = SlidingWindowChunker(self.tokenizer)
            
            df_chunked = chunker.create_chunked_dataset(processed_df, max_length, stride)

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise

        try:
            test_data = Dataset.from_pandas(df_chunked)

            # -------------------------------
            # tokenizer pad_token 안전 처리
            # -------------------------------
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

            # -------------------------------
            # 모델 config pad_token_id 설정
            # -------------------------------
            if getattr(self.model.config, 'pad_token_id', None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            # -------------------------------
            # tokenization
            # -------------------------------
            def preprocess_function(examples):
                tokenized = self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding=True
                )
                return tokenized

            tokenized_test = test_data.map(preprocess_function, batched=True)

            remove_cols = ['text', 'patent_id']
            if 'label' in df_chunked.columns:
                remove_cols.append('label')
            tokenized_test = tokenized_test.remove_columns(remove_cols)

            # -------------------------------
            # DataLoader
            # -------------------------------
            from torch.utils.data import DataLoader
            from transformers import DataCollatorWithPadding

            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            dataloader = DataLoader(tokenized_test, batch_size=2, collate_fn=data_collator)

            # -------------------------------
            # 추론
            # -------------------------------
            self.model.eval()
            all_predictions = []

            with torch.no_grad():
                for batch in dataloader:
                    if next(self.model.parameters()).device.type == 'cuda':
                        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    outputs = self.model(**batch)
                    logits = outputs.logits
                    predictions = torch.softmax(logits, dim=-1)
                    all_predictions.append(predictions.cpu())

            probs = torch.cat(all_predictions, dim=0).numpy()

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise

        try:
            # -------------------------------
            # 결과 처리
            # -------------------------------
            df_chunked = df_chunked.reset_index(drop=True)
            df_chunked['chunk_index'] = range(len(df_chunked))

            patent_results = []

            for patent_id, group in df_chunked.groupby('patent_id'):
                indices = group['chunk_index'].tolist()
                group = group.copy()
                group['chunk_len'] = group['text'].apply(lambda x: len(x.split()))
                weights = group['chunk_len'].values

                if len(weights) > 0 and len(indices) > 0:
                    weighted_probs = probs[indices] * weights[:, None]
                    mean_prob = weighted_probs.sum(axis=0) / weights.sum()

                    pred_idx = mean_prob.argmax()
                    pred_label = self.id2label[pred_idx]

                    patent_results.append({
                        "출원번호": patent_id,
                        "예측_라벨": pred_label,
                        "신뢰도": round(mean_prob[pred_idx], 4)
                    })

            return pd.DataFrame(patent_results)

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise