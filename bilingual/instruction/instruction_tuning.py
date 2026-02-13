# ===== TRUE INSTRUCTION TUNING (MASK PROMPT LOSS) – END-TO-END =====
# Works for causal LM models (GPT-style, OPT, LLaMA, etc.)
# Masks loss so model ONLY learns from assistant / response tokens

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
# CONFIG
# -----------------------------
MODEL_NAME = "gpt2"          # change to your model
MAX_LENGTH = 128
OUTPUT_DIR = "./inst_tuned_model"
IGNORE_INDEX = -100

# -----------------------------
# LOAD TOKENIZER + MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT2 has no pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Resize embeddings if tokenizer changed
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loaded model on:", device)

# -----------------------------
# LOAD DATASET
# Expected fields:
#   instruction
#   response
# -----------------------------
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

# -----------------------------
# FORMAT FUNCTION
# -----------------------------
def format_example(example):

    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    answer = example["output"]

    full_text = prompt + answer + tokenizer.eos_token

    # Tokenize full sequence
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Tokenize prompt alone (to know where to mask)
    tokenized_prompt = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(tokenized_prompt["input_ids"])

    # Mask prompt tokens
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# -----------------------------
# SAFETY CHECK (prevents CUDA assert)
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
collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=IGNORE_INDEX
)

# -----------------------------
# TRAINING ARGS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    bf16=torch.cuda.is_available(),
    report_to="none",
)

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# SAVE MODEL
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")
