from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from trl import SFTTrainer, SFTConfig

import os

class LLMModel():
    def __init__(self, base_model_id, unique_labels, max_length):
        self.max_length = max_length
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = 'right'
        
        # Model
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=len(unique_labels),
            device_map='auto',
            quantization_config=self.bnb_config
        )
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def preprocess_function(self, examples):
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        tokenized['labels'] = examples['label']
        return tokenized
        
    def lora_config(self, lora_alpha, lora_dropout, lora_r):
        # PEFT
        self.peft_config = LoraConfig(
            lora_alpha=int(lora_alpha),
            lora_dropout=lora_dropout,
            r=int(lora_r),
            bias='none',
            task_type='SEQ_CLS',
            target_modules=['k_proj','gate_proj','v_proj','up_proj','q_proj','o_proj','down_proj']
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        
    def sft_config(self, output_dir, model_name, learning_rate, batch_size, num_epochs, max_length):
        # SFTConfig 생성 (Streamlit 하이퍼파라미터 반영)
        self.sft_args = SFTConfig(
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

    def train(self, output_dir, model_name, tokenized_train, tokenized_test):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=self.tokenizer,
            args=self.sft_args,
            data_collator=data_collator,
            peft_config=self.peft_config
        )
        
        trainer.train()
        trainer.save_model(os.path.join(output_dir, model_name))
        self.tokenizer.save_pretrained(os.path.join(output_dir, model_name))