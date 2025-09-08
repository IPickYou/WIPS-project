# utils/finetuning_trainer.py

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
from utils.text_chunker import SlidingWindowChunker

# .env 파일 로드
load_dotenv()

class FineTuningTrainer:
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