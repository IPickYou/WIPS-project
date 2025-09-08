# utils/transformers_settings.py

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
import pickle
from safetensors.torch import load_file
from dotenv import load_dotenv

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

    def prepare_datasets(self, df_chunked, test_size=0.2, random_state=25):
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

    def train_model(self, tokenized_train, tokenized_test, output_dir, bnb_config_params=None, lora_config_params=None,
                    training_config_params=None):
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

    def save_model(self, output_dir, save_merged=False, merged_output_dir=None):
        """
        모델 저장 (어댑터만 또는 머지된 모델)

        Args:
            output_dir: 어댑터 저장 경로
            save_merged: 머지된 모델 저장 여부
            merged_output_dir: 머지된 모델 저장 경로 (None이면 output_dir + "_merged")
        """
        if self.trainer:
            # 1. 기본 어댑터 저장
            self.trainer.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # 2. 머지된 모델 저장 (선택사항)
            if save_merged:
                if merged_output_dir is None:
                    merged_output_dir = output_dir + "_merged"

                print(f"머지된 모델을 {merged_output_dir}에 저장 중...")

                # 베이스 모델 로드 (양자화 없이)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    num_labels=len(self.labels_list),
                    torch_dtype=torch.float16
                )

                # 어댑터 로드 및 머지
                model_with_adapter = PeftModel.from_pretrained(base_model, output_dir)
                merged_model = model_with_adapter.merge_and_unload()

                # 머지된 모델 저장
                merged_model.save_pretrained(merged_output_dir)
                self.tokenizer.save_pretrained(merged_output_dir)

                print(f"머지 완료: {merged_output_dir}")

    def load_model(self, model_path, use_merged=False, manual_labels=None):
        """
        모델 로드 (어댑터 방식 또는 머지된 모델)

        Args:
            model_path: 모델 경로
            use_merged: True면 머지된 모델 직접 로드, False면 베이스+어댑터 로드
            manual_labels: 수동 라벨 지정
        """
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"모델 경로가 존재하지 않습니다: {model_path}")

        # 라벨 매핑 로드
        label_file = os.path.join(model_path, 'label_mappings.pkl')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as f:
                mappings = pickle.load(f)
                self.labels_list = mappings['labels_list']
                self.label2id = mappings['label2id']
                self.id2label = mappings['id2label']
        elif manual_labels:
            self.labels_list = sorted(manual_labels)
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        else:
            raise ValueError("라벨 매핑 정보가 없습니다. manual_labels를 지정해주세요.")

        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=self.hf_token)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        if use_merged:
            # 머지된 모델 직접 로드
            print("머지된 모델을 로드 중...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(self.labels_list),
                torch_dtype=torch.float16,
                device_map='auto'
            )
        else:
            # 베이스 모델 + 어댑터 로드 (기존 방식)
            print("베이스 모델 + 어댑터를 로드 중...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='float16',
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token=self.hf_token,
                num_labels=len(self.labels_list),
                device_map='auto',
                quantization_config=bnb_config
            )

            peft_config = LoraConfig(
                lora_alpha=128, lora_dropout=0.1, r=64, bias='none',
                task_type='SEQ_CLS',
                target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            )

            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)

            # 어댑터 가중치 로드
            adapter_path = os.path.join(model_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                checkpoint = load_file(adapter_path)
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"어댑터 파일을 찾을 수 없습니다: {adapter_path}")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def predict_patents(self, df, model_path=None, use_merged=False, selected_cols=None, max_length=512, stride=50,
                        batch_size=2):
        """
        특허 예측 (청킹 + 가중평균 방식 유지)

        Args:
            use_merged: 머지된 모델 사용 여부
            batch_size: 배치 크기
        """
        if model_path and not self.model:
            self.load_model(model_path, use_merged=use_merged)

        if not self.model or not self.tokenizer:
            raise ValueError("모델이 로드되지 않았습니다.")

        if selected_cols is None:
            selected_cols = ["발명의 명칭", "요약", "전체청구항"]

        # 데이터 전처리 및 청킹 (기존 방식 유지)
        processed_df = self.prepare_data(df, selected_cols)
        df_chunked = self.create_chunked_dataset(processed_df, max_length, stride)

        # 토크나이징
        test_data = Dataset.from_pandas(df_chunked)

        def preprocess_function(examples):
            tokenized = self.tokenizer(examples['text'], truncation=True, max_length=512, padding=True)
            return tokenized

        tokenized_test = test_data.map(preprocess_function, batched=True)
        remove_cols = ['text', 'patent_id']
        if 'label' in df_chunked.columns:
            remove_cols.append('label')
        tokenized_test = tokenized_test.remove_columns(remove_cols)

        # 추론 수행
        from torch.utils.data import DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(tokenized_test, batch_size=batch_size, collate_fn=data_collator)

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

        # 청크별 예측을 특허별로 통합 (가중평균 방식 유지)
        df_chunked = df_chunked.reset_index(drop=True)
        df_chunked['chunk_index'] = range(len(df_chunked))

        patent_results = []

        for patent_id, group in df_chunked.groupby('patent_id'):
            indices = group['chunk_index'].tolist()
            weights = group['text'].apply(lambda x: len(x.split())).values

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


class DataProcessor:

    @staticmethod
    def load_and_combine_excel_files(train_path, test_path=None, sheet_name=0):
        """
        학습/테스트 엑셀 파일들을 로드하고 결합

        Args:
            train_path: 학습용 엑셀 파일 경로
            test_path: 테스트용 엑셀 파일 경로 (선택사항)
            sheet_name: 시트 이름 또는 인덱스

        Returns:
            dict: {'train': DataFrame, 'test': DataFrame} 또는 {'combined': DataFrame}
        """
        try:
            train_df = pd.read_excel(train_path, sheet_name=sheet_name)
            print(f"학습 데이터 로드: {len(train_df)}행")

            result = {'train': train_df}

            if test_path:
                test_df = pd.read_excel(test_path, sheet_name=sheet_name)
                print(f"테스트 데이터 로드: {len(test_df)}행")
                result['test'] = test_df
            else:
                result['combined'] = train_df

            return result

        except Exception as e:
            raise ValueError(f"엑셀 파일 로드 실패: {str(e)}")

    @staticmethod
    def validate_dataframe(df, required_cols=None):
        if df is None or len(df) == 0:
            raise ValueError("데이터가 비어 있습니다.")

        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

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