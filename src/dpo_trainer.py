'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-01 09:35:08
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-31 10:54:45
FilePath: \TRL_trainer_wapper\src\dpo_trainer.py
Description: 
'''
from yaml import safe_load
from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from accelerate import PartialState
from os import environ
from warnings import filterwarnings

filterwarnings("ignore")
# Load train settings from YAML file
train_setting = safe_load(open("train_setting.yaml"))


# Set env
environ['HF_TOKEN'] = train_setting['api_tokens']['huggingface']

# Get device string
device_string = PartialState().process_index
print(f"Device: {device_string}")

# Load dataset
dataset = load_dataset(train_setting['dataset']['name'], split=train_setting['dataset']['split'],token=train_setting['api_tokens']['huggingface'])

# change data type to string
dataset = dataset.map(lambda x: {train_setting['trainer']['dataset_text_field']: str(x[train_setting['trainer']['dataset_text_field']])})

# Load model
model = AutoModelForCausalLM.from_pretrained(train_setting['model']['name'],
                                             device_map='auto',
                                             token=train_setting['api_tokens']['huggingface'])

# Configure LoRA if fine-tuning method is 'lora'
if 'fine_tuning' in train_setting and 'method' in train_setting['fine_tuning'] and train_setting['fine_tuning']['method'] == 'lora':
    if 'lora_config' in train_setting:
        peft_config = LoraConfig(
            r=int(train_setting['lora_config'].get('r', 0)),
            lora_alpha=int(train_setting['lora_config'].get('lora_alpha', 0)),
            lora_dropout=float(train_setting['lora_config'].get('lora_dropout', 0.0)),
            bias=str(train_setting['lora_config'].get('bias', '')),
            task_type=str(train_setting['lora_config'].get('task_type', '')),
        )
        model = get_peft_model(model, peft_config)
        print("LoRA method can't be used in to mergekit. Please use the full-finetuning method.")
        print(model)
else:
    print(model)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=train_setting['training_args']['output_dir'],
    per_device_train_batch_size=int(train_setting['training_args']['per_device_train_batch_size']),
    gradient_accumulation_steps=int(train_setting['training_args']['gradient_accumulation_steps']),
    learning_rate=float(train_setting['training_args']['learning_rate']),
    logging_steps=int(train_setting['training_args']['logging_steps']),
    save_total_limit=1,
    push_to_hub=train_setting['training_args']['push_to_hub'],
    push_to_hub_organization=train_setting['training_args']['push_to_organization'] if 'push_to_organization' != '' else None,
)

# Create DPOTrainer instance

trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    max_seq_length=int(train_setting['trainer']['max_seq_length']),
    args=training_args
)
# We don't pass the tokenizer to the trainer, because it will automatically use the original tokenizer of the model

# Start training
print("Start training")
trainer.train()

# Save model
print("Save model")
trainer.save_model()

# Upload model to Hugging Face Model Hub
trainer.push_to_hub()

# Finish training
print("Finish training")
