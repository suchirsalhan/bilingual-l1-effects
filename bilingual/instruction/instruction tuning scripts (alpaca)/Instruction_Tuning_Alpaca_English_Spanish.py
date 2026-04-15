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
HF_TOKEN = os.environ.get("HF_TOKEN")  # make sure you exported this in SSH
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not set! Run: export HF_TOKEN='your_token_here'")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "RA-ALTA/es-en-advanced"
MAX_LENGTH = 512
IGNORE_INDEX = -100
HF_REPO = "RA-ALTA/es-en-advanced-alpaca-english-L1"
LOCAL_DIR = "./dummy_output"  # local save directory

# -----------------------------
# TOKENIZER + MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Loaded model on:", device)


# -----------------------------
# DATASET
# -----------------------------

from datasets import load_dataset, concatenate_datasets

alpaca_english = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_spanish = load_dataset("bertin-project/alpaca-spanish", split="train")

print(alpaca_english.column_names)
print(alpaca_spanish.column_names)

print(len(alpaca_english))
print(len(alpaca_spanish))

# remove the rows with input:
# Keep only rows where input is empty
alpaca_english = alpaca_english.filter(lambda example: example['input'] == "")
# Check
print(len(alpaca_english))
print(alpaca_english.column_names)

# remove rows with empty output
alpaca_english = alpaca_english.filter(
    lambda example: (
        example.get("output") is not None
        and isinstance(example["output"], str)
        and example["output"].strip() != ""
    )
)
# Check
print(len(alpaca_english))
print(alpaca_english.column_names)

# normalise alpaca polish:
import unicodedata

def normalize_example(example):
    for col in ["instruction", "input", "output"]:
        if col in example and example[col] is not None:
            example[col] = unicodedata.normalize("NFKC", example[col])
    return example

alpaca_spanish = alpaca_spanish.map(normalize_example)

# remove the rows with input:
# Keep only rows where input is empty
alpaca_spanish = alpaca_spanish.filter(
    lambda example: example.get("input") is None
                    or example["input"].strip() == ""
                    or example["input"].strip() == "nan"
)
# Check
print(len(alpaca_spanish))
print(alpaca_spanish.column_names)

# many rows have empty output:
alpaca_spanish = alpaca_spanish.filter(
    lambda example: (
        example.get("output") is not None
        and isinstance(example["output"], str)
        and example["output"].strip() != ""
    )
)
# Check
print(len(alpaca_spanish))
print(alpaca_spanish.column_names)

# rename dataset columns:
alpaca_english = alpaca_english.rename_columns({
    "instruction": "instruction",
    "output": "response",
})

# rename dataset columns:
alpaca_spanish = alpaca_spanish.rename_columns({
    "instruction": "instruction",
    "output": "response",
})

# shuffle each
alpaca_english = alpaca_english.shuffle(seed=42)
alpaca_spanish = alpaca_spanish.shuffle(seed=42)

min_size = min(len(alpaca_english), len(alpaca_spanish))
print(min_size)

target_per_dataset = 31322 // 2
print(target_per_dataset)

target_size = target_per_dataset

alpaca_english_small = alpaca_english.select(range(target_size))
alpaca_spanish_small = alpaca_spanish.select(range(target_size))

print(len(alpaca_english_small))
print(len(alpaca_spanish_small))

# randomly halve each
#half_en = len(alpaca_english) // 2
#half_pl = len(alpaca_polish) // 2

#alpaca_english = alpaca_english.select(range(half_en))
#alpaca_polish = alpaca_polish.select(range(half_pl))

dataset = concatenate_datasets([alpaca_english_small, alpaca_spanish_small])
dataset = dataset.shuffle(seed=42)

print(len(dataset))

dataset_raw = dataset

# RUN THIS ONE!
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    answer = example["response"]

    # Tokenize prompt (never truncate)
    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False
    )
    prompt_len = len(prompt_tokens["input_ids"])

    # Compute available space for answer
    max_response_len = MAX_LENGTH - prompt_len - 1  # EOS

    if max_response_len <= 0:
        raise ValueError("Prompt too long for MAX_LENGTH")

    # Tokenize answer (truncate ONLY here)
    answer_tokens = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_response_len
    )

    # Concatenate
    input_ids = (
        prompt_tokens["input_ids"]
        + answer_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    attention_mask = [1] * len(input_ids)

    # Labels: mask prompt tokens
    labels = input_ids.copy()
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names
)

print(dataset.column_names)

# CHECK!
ex = dataset[0]
print(tokenizer.decode(ex["input_ids"], skip_special_tokens=False))

print(repr(dataset_raw[0]["response"]))

sum(1 for x in dataset_raw if x["response"].strip() == "")

print(dataset_raw.column_names)
print(dataset_raw[0])

for x in dataset_raw:
    if x["text"] is not None:
        print(x["text"])
        break


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
    tokenizer=tokenizer,
    padding=True,
)

# -----------------------------
# TRAINING ARGS
# -----------------------------

training_args = TrainingArguments(
    output_dir=LOCAL_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="epoch",      # save at end of epoch
    dataloader_num_workers=8,
    report_to="none",
    disable_tqdm=False
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

print("Saving model and tokenizer locally...")
model.save_pretrained(LOCAL_DIR)
tokenizer.save_pretrained(LOCAL_DIR)
print(f"Model saved to {LOCAL_DIR}")

# -----------------------------
# PUSH TO HUGGING FACE HUB (safe)
# -----------------------------
print("Pushing model and tokenizer to Hugging Face Hub...")
# Do NOT pass use_auth_token inside tokenizer files; HF_TOKEN is read from environment
os.environ["HF_TOKEN"] = HF_TOKEN  # ensure env variable is set
model.push_to_hub(HF_REPO)         # reads HF_TOKEN automatically
tokenizer.push_to_hub(HF_REPO)
print("Training, local save, and push complete!")
