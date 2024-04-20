'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2024-04-17 15:26:22
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-04-20 18:41:31
FilePath: /2024_president/ml/sft_train.py
Description:
'''
from yaml import safe_load
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import PartialState

train_setting = safe_load(open("train_setting.yaml"))

device_string = PartialState().process_index
print(f"Device: {device_string}")

dataset = load_dataset(train_setting['dataset']['name'], split=train_setting['dataset']['split'])

model = AutoModelForCausalLM.from_pretrained(train_setting['model']['name'],
                                             device_map={'':device_string})
tokenizer = AutoTokenizer.from_pretrained(train_setting['model']['name'])

peft_config = LoraConfig(
    r=int(train_setting['peft']['r']),
    lora_alpha=int(train_setting['peft']['lora_alpha']),
    lora_dropout=float(train_setting['peft']['lora_dropout']),
    bias=str(train_setting['peft']['bias']),
    task_type=str(train_setting['peft']['task_type']),
)
model = get_peft_model(model, peft_config) if train_setting['fine_tune']['method'] == 'peft' else model

print("LoRA method can't be used in to mergekit. Please use the full-finetuning method.") if train_setting['fine_tune']['method'] == 'peft' else None

traine_args = TrainingArguments(
    output_dir=train_setting['training_args']['output_dir'],
    per_device_train_batch_size=int(train_setting['training_args']['per_device_train_batch_size']),
    gradient_accumulation_steps=int(train_setting['training_args']['gradient_accumulation_steps']),
    learning_rate=float(train_setting['training_args']['learning_rate']),
    logging_steps = int(train_setting['training_args']['logging_steps']),
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field=train_setting['dataset']['text_field'],
    max_seq_length=int(train_setting['trainer']['max_seq_length']),
    tokenizer=tokenizer,
    args=traine_args
)

print("Start training")
trainer.train()

print("Save model")
trainer.save_model()

print("Finish training")