# utils/transformers_settings.py

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

# .env 파일 로드
load_dotenv()


class FineTuningClassifier:

    def __init__(self, model_name="google/gemma-2-2b", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

    def prepare_data(self, df, selected_cols=None):

        if selected_cols is None:
            selected_cols = ["발명의 명칭", "요약", "전체청구항"]

        def combine_text(row):
            text_parts = []
            for col in selected_cols:
                if col in row.index and pd.notna(row.get(col, "")):
                    text_parts.append(f"{col} : {row[col]}")
            return " ".join(text_parts)

        df_copy = df.copy()
        df_copy["combined_text"] = df_copy.apply(combine_text, axis=1)

        if "사용자태그" in df_copy.columns:

            self.labels_list = sorted(df_copy["사용자태그"].unique())
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}

            processed_df = pd.DataFrame({
                "text": df_copy["combined_text"],
                "labels": df_copy["사용자태그"],
                "patent_id": df_copy["출원번호"]
            })
            processed_df["label"] = processed_df["labels"].map(self.label2id)

        else:
            processed_df = pd.DataFrame({
                "text": df_copy["combined_text"],
                "patent_id": df_copy["출원번호"]
            })

        return processed_df

    def chunk_text(self, text, max_length=512, stride=50):

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunks.append(self.tokenizer.decode(tokens[start:end], skip_special_tokens=True))

            if end == len(tokens):
                break

            start += max_length - stride

        return chunks

    def create_chunked_dataset(self, df, max_length=512, stride=50):

        chunked_rows = []

        for _, row in df.iterrows():
            chunks = self.chunk_text(row["text"], max_length, stride)

            for chunk in chunks:
                chunk_row = {
                    "text": chunk,
                    "patent_id": row["patent_id"]
                }
                if "label" in row:
                    chunk_row["label"] = row["label"]
                chunked_rows.append(chunk_row)

        return pd.DataFrame(chunked_rows)

    def create_balanced_datasetdict(self, df_chunked, test_size=0.2, random_state=25):
        """라벨별 동일한 개수로 train/test 분할하여 DatasetDict 생성"""

        # 각 라벨별 최소 개수 찾기
        label_counts = df_chunked['label'].value_counts()
        min_count = label_counts.min()

        train_samples_per_label = int(min_count * (1 - test_size))
        test_samples_per_label = min_count - train_samples_per_label

        print(f"각 라벨별 train: {train_samples_per_label}개, test: {test_samples_per_label}개")

        train_dfs = []
        test_dfs = []

        # 각 라벨별로 동일한 개수만큼 샘플링
        for label in sorted(df_chunked['label'].unique()):
            label_data = df_chunked[df_chunked['label'] == label].sample(
                n=min_count,
                random_state=random_state
            ).reset_index(drop=True)

            train_data = label_data.iloc[:train_samples_per_label]
            test_data = label_data.iloc[train_samples_per_label:]

            train_dfs.append(train_data)
            test_dfs.append(test_data)

        # 합치기
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_state)

        # DatasetDict 생성
        train_dataset_dict = {
            'text': train_df['text'].tolist(),
            'label': train_df['label'].tolist()
        }

        test_dataset_dict = {
            'text': test_df['text'].tolist(),
            'label': test_df['label'].tolist()
        }

        train_data = Dataset.from_dict(train_dataset_dict)
        test_data = Dataset.from_dict(test_dataset_dict)

        dataset = DatasetDict({
            'train': train_data,
            'test': test_data
        })

        # 토큰화 적용
        def preprocess_function(examples):
            tokenized = self.tokenizer(examples['text'], truncation=True, max_length=512)
            tokenized['labels'] = [int(l) for l in examples['label']]
            return tokenized

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])

        return tokenized_dataset, test_df

    def prepare_datasets(self, df_chunked, test_size=0.2, random_state=25):
        """기존 방식 (호환성 유지)"""

        train_df, test_df = train_test_split(
            df_chunked,
            test_size=test_size,
            stratify=df_chunked['label'],
            random_state=random_state
        )

        train_data = Dataset.from_pandas(train_df)
        test_data = Dataset.from_pandas(test_df)

        def preprocess_function(examples):
            tokenized = self.tokenizer(examples['text'], truncation=True, max_length=512)
            tokenized['labels'] = [int(l) for l in examples['label']]
            return tokenized

        tokenized_train = train_data.map(preprocess_function, batched=True)
        tokenized_test = test_data.map(preprocess_function, batched=True)

        tokenized_train = tokenized_train.remove_columns(['text', 'label', 'patent_id'])
        tokenized_test = tokenized_test.remove_columns(['text', 'label', 'patent_id'])

        return tokenized_train, tokenized_test, test_df

    def setup_model(self, bnb_config_params=None, lora_config_params=None):

        if bnb_config_params is None:
            bnb_config_params = {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_use_double_quant': True
            }

        bnb_config = BitsAndBytesConfig(**bnb_config_params)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            token=self.hf_token,
            num_labels=len(self.labels_list),
            device_map='auto',
            quantization_config=bnb_config
        )

        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if lora_config_params is None:
            lora_config_params = {
                'lora_alpha': 128,
                'lora_dropout': 0.1,
                'r': 64,
                'bias': 'none',
                'task_type': 'SEQ_CLS',
                'target_modules': ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            }

        peft_config = LoraConfig(**lora_config_params)

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        return peft_config

    def compute_metrics(self, pred):

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)

        logits_tensor = torch.tensor(pred.predictions)
        labels_tensor = torch.tensor(pred.label_ids)
        loss = F.cross_entropy(logits_tensor, labels_tensor).item()

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'eval_loss': loss
        }

    def train_model(self, tokenized_dataset, output_dir, bnb_config_params=None, lora_config_params=None,
                    training_config_params=None, use_balanced_split=True):
        """DatasetDict을 받도록 수정"""

        # DatasetDict인지 확인
        if isinstance(tokenized_dataset, DatasetDict):
            tokenized_train = tokenized_dataset['train']
            tokenized_test = tokenized_dataset['test']
        else:
            # 기존 방식 호환성
            tokenized_train, tokenized_test = tokenized_dataset

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        default_training_args = {
            'output_dir': output_dir,
            'learning_rate': 2e-5,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 2,
            'optim': 'paged_adamw_32bit',
            'lr_scheduler_type': 'cosine',
            'num_train_epochs': 5,
            'warmup_steps': 50,
            'logging_steps': 10,
            'fp16': True,
            'gradient_checkpointing': True,
            'dataset_text_field': 'text',
            'max_length': 512,
            'label_names': ['labels']
        }

        if training_config_params:
            default_training_args.update(training_config_params)

        training_arguments = SFTConfig(**default_training_args)

        if lora_config_params is None:
            lora_config_params = {
                'lora_alpha': 128,
                'lora_dropout': 0.1,
                'r': 64,
                'bias': 'none',
                'task_type': 'SEQ_CLS',
                'target_modules': ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            }

        peft_config = LoraConfig(**lora_config_params)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=self.tokenizer,
            args=training_arguments,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            peft_config=peft_config
        )

        self.trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'label_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'labels_list': self.labels_list,
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f)

        return self.trainer.evaluate()

    def save_model(self, output_dir, merge_adapter=True):
        """어댑터 병합 옵션 추가"""

        if self.trainer:
            if merge_adapter:
                # 어댑터 병합
                merged_model = self.trainer.model.merge_and_unload()

                # 병합된 모델 저장
                merged_output_dir = os.path.join(output_dir, "merged_model")
                os.makedirs(merged_output_dir, exist_ok=True)
                merged_model.save_pretrained(merged_output_dir)
                self.tokenizer.save_pretrained(merged_output_dir)

                print(f"병합된 모델이 {merged_output_dir}에 저장되었습니다.")
            else:
                # 기존 방식 (어댑터만 저장)
                self.trainer.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

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

        # << 여기가 핵심 >>
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

            processed_df = self.prepare_data(df, selected_cols)
            df_chunked = self.create_chunked_dataset(processed_df, max_length, stride)

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


class DataProcessor:

    @staticmethod
    def validate_dataframe(df, required_cols=None):
        if df is None or len(df) == 0:
            raise ValueError("데이터가 비어 있습니다.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(missing_cols)

        return True

    @staticmethod
    def get_available_columns(df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = ["사용자태그", "WINTELIPS KEY"]

        return [col for col in df.columns if col not in exclude_cols]

    @staticmethod
    def create_results_summary(results_df, label_col='예측_라벨'):

        if results_df is None or len(results_df) == 0:
            return None

        summary = {
            'total_count': len(results_df),
            'label_distribution': results_df[label_col].value_counts().to_dict(),
            'unique_labels': results_df[label_col].nunique()
        }

        if '신뢰도' in results_df.columns:
            summary['confidence_stats'] = {
                'mean': results_df['신뢰도'].mean(),
                'min': results_df['신뢰도'].min(),
                'max': results_df['신뢰도'].max(),
                'std': results_df['신뢰도'].std()
            }

        return summary