'''
Author: hibana2077 hibana2077@gmaill.com
Date: 2024-04-17 15:26:22
LastEditors: hibana2077 hibana2077@gmaill.com
LastEditTime: 2024-04-17 15:55:21
FilePath: /2024_president/ml/sft_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from yaml import safe_load
from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import PartialState

train_setting = safe_load(open("train_setting.yaml"))

device_string = PartialState().process_index
print(f"Device: {device_string}")

dataset = load_dataset("NTTUNLPTEAM/class-textbook", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

traine_args = TrainingArguments(
    output_dir="./opt-350m",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps = 100)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=traine_args
)

trainer.train()