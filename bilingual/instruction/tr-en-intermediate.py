# -*- coding: utf-8 -*-
"""Instruction tuning script — SSH/GPU ready, safe local save + push"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# -----------------------------
# ENVIRONMENT (HF token + cache)
# -----------------------------
os.environ["HF_HOME"] = "/local/scratch/lgb35/hf_cache"
os.environ["HUGGINGFACE_HUB_TOKEN"] = "TUQhUgrNyKvaGtoMHbzKIGjvwApbaLQfJc"
HF_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "RA-ALTA/tr-en-intermediate"
MAX_LENGTH = 512
IGNORE_INDEX = -100
HF_REPO = "RA-ALTA/tr-en-intermediate-alpaca-english"
LOCAL_SAVE_DIR = "./dummy_output"  # local save before pushing

# -----------------------------
# TOKENIZER + MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_auth_token=HF_TOKEN
)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)

# Resize embeddings if tokenizer changed
model.resize_token_embeddings(len(tokenizer))

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Loaded model on:", device)

# -----------------------------
# DATASET
# -----------------------------
alpaca_english = load_dataset("tatsu-lab/alpaca", split="train")

# Keep only empty input and non-empty output
alpaca_english = alpaca_english.filter(lambda e: e['input'] == "")
alpaca_english = alpaca_english.filter(lambda e: e.get("output") and e["output"].strip() != "")

# Rename columns
alpaca_english = alpaca_english.rename_columns({
    "instruction": "instruction",
    "output": "response"
})

dataset = alpaca_english.shuffle(seed=42)

# -----------------------------
# FORMAT FUNCTION
# -----------------------------
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    answer = example["response"]

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens["input_ids"])

    max_response_len = MAX_LENGTH - prompt_len - 1
    if max_response_len <= 0:
        raise ValueError("Prompt too long for MAX_LENGTH")

    answer_tokens = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_response_len)

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len  # mask prompt

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# -----------------------------
# SAFETY CHECK
# -----------------------------
vocab_size = model.get_input_embeddings().weight.shape[0]

def validate(example):
    for t in example["input_ids"]:
        assert 0 <= t < vocab_size, f"Token id {t} outside vocab"
    return example

dataset = dataset.map(validate)

# -----------------------------
# DATA COLLATOR
# -----------------------------
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

# -----------------------------
# TRAINING ARGS
# -----------------------------
training_args = TrainingArguments(
    output_dir=LOCAL_SAVE_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="epoch",  # save locally at each epoch
    save_total_limit=1,
    dataloader_num_workers=8,
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# SAVE LOCALLY BEFORE PUSH
# -----------------------------
print(f"Saving model and tokenizer locally to {LOCAL_SAVE_DIR}...")
model.save_pretrained(LOCAL_SAVE_DIR)
tokenizer.save_pretrained(LOCAL_SAVE_DIR)

# -----------------------------
# PUSH TO HUGGING FACE HUB
# -----------------------------
print(f"Pushing model and tokenizer to Hugging Face Hub ({HF_REPO})...")
model.push_to_hub(HF_REPO, use_auth_token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, use_auth_token=HF_TOKEN)
print("Training and push complete!")
