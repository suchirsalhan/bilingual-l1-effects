

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)


MODEL_NAME = "RA-ALTA/es-en-advanced" 
MAX_LENGTH = 512
OUTPUT_DIR = "./inst_tuned_model"
IGNORE_INDEX = -100


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "right"


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loaded model on:", device)


from datasets import load_dataset, concatenate_datasets

learner_english = load_dataset("", split="train")

print(learner_english.column_names)

print(len(learner_english))


learner_english = learner_english.filter(
    lambda example: (
        example.get("Output") is not None
        and isinstance(example["Output"], str)
        and example["Output"].strip() != ""
    )
)

print(len(learner_english))
print(learner_english.column_names)

learner_english = learner_english.rename_columns({
    "Instruction": "instruction",
    "Output": "response",
})

learner_english = learner_english.shuffle(seed=42)

print(learner_english[0])

dataset = learner_english

from datasets import Dataset

def clean_example(example):
    example['instruction'] = example['instruction'].replace('\xa0', ' ').strip()
    example['response'] = example['response'].replace('\xa0', ' ').strip()
    return example

dataset = dataset.map(clean_example)

print(dataset[0])

print(len(dataset))

dataset_raw = dataset


def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    answer = example["response"]


    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False
    )
    prompt_len = len(prompt_tokens["input_ids"])


    max_response_len = MAX_LENGTH - prompt_len - 1  # EOS

    if max_response_len <= 0:
        raise ValueError("Prompt too long for MAX_LENGTH")


    answer_tokens = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_response_len
    )


    input_ids = (
        prompt_tokens["input_ids"]
        + answer_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )

    attention_mask = [1] * len(input_ids)


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


ex = dataset[0]
print(tokenizer.decode(ex["input_ids"], skip_special_tokens=False))

print(repr(dataset_raw[0]["response"]))

sum(1 for x in dataset_raw if x["response"].strip() == "")

print(dataset_raw.column_names)
print(dataset_raw[0])


vocab_size = model.get_input_embeddings().weight.shape[0]

def validate(example):
    for t in example["input_ids"]:
        assert 0 <= t < vocab_size, f"Token id {t} outside vocab"
    return example

dataset = dataset.map(validate)


collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,  # log more frequently
    save_steps=500,
    bf16=torch.cuda.is_available(),
    report_to="none",
    disable_tqdm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)


trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")

model.push_to_hub()
tokenizer.push_to_hub()
