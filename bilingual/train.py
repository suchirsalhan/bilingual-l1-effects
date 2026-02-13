# TO DO – 1024 OR 512 =  sequence length + context window, max_len = prompt_len + 220 tokens (or 500 tokens). 
# generate a config.json before running. 
import random
import json
from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ------------------------------
# CONFIG
# ------------------------------
l1_languages = ["es", "fr", "de", "pl", "tr", "ar", "zh"]
l1_dataset_name = "uonlp/CulturaX"
l2_datasets = {
    "narrow": "HuggingFaceFW/fineweb-edu",
    "broad": {
        "dataset": "togethercomputer/RedPajama-Data-V2",
        "snapshot": ["2023-06"],
        "language": "en",
        "partition": "head_middle"
    }
}

total_tokens = 20_000_000_000
l1_tokens_total = int(total_tokens * 2 / 3)  # 13.3B
l2_tokens_total = int(total_tokens * 1 / 3)  # 6.7B

phase1_fraction = 0.5
phase2_fraction = 0.5
l2_checkpoints = [0.25, 0.5, 0.75, 1.0]
batch_size = 4
save_dir_root = "./bilingual_models"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def gopher_rules_pass(sample):
    try:
        signals = json.loads(sample["quality_signals"])
        wc = signals["rps_doc_word_count"][0][2]
        if wc < 50 or wc > 100_000: return False
        mean_len = signals["rps_doc_mean_word_length"][0][2]
        if mean_len < 3 or mean_len > 10: return False
        sym_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
        if sym_ratio > 0.1: return False
        n_lines = signals["ccnet_nlines"][0][2]
        bullets = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
        if bullets / n_lines > 0.9: return False
        top2 = signals["rps_doc_frac_chars_top_2gram"][0][2]
        if top2 > 0.2: return False
        return True
    except Exception:
        return False

def load_and_tokenize_l1(l1_lang, tokenizer, max_tokens=None):
    ds = load_dataset(f"{l1_dataset_name}","{l1_lang}", split="train")
    def tok(batch):
        enc = tokenizer(batch["text"], truncation=False)
        batch["input_ids"] = enc["input_ids"]
        batch["n_tokens"] = [len(ids) for ids in enc["input_ids"]]
        return batch
    ds = ds.map(tok, batched=True)
    if max_tokens:
        current = 0
        indices = []
        for i, n in enumerate(ds["n_tokens"]):
            if current + n > max_tokens: break
            indices.append(i)
            current += n
        ds = ds.select(indices)
    return ds

def stream_redpajama(tokenizer, max_tokens):
    cfg = l2_datasets["broad"]
    ds_iter = load_dataset(
        cfg["dataset"],
        snapshots=cfg["snapshot"],
        languages=[cfg["language"]],
        partition=cfg["partition"],
        streaming=True
    )["train"]

    total_tokens = 0
    tokenized_samples = []
    for sample in ds_iter:
        if not gopher_rules_pass(sample):
            continue
        ids = tokenizer(sample["text"], truncation=False)["input_ids"]
        n_ids = len(ids)
        if total_tokens + n_ids > max_tokens:
            break
        tokenized_samples.append({"input_ids": ids})
        total_tokens += n_ids
    return tokenized_samples

def interleave_l1_l2(l1_ds, l2_ds):
    l1_iter = iter(l1_ds)
    l2_iter = iter(l2_ds)
    merged = []
    while True:
        try:
            for _ in range(2):
                merged.append(next(l1_iter))
            merged.append(next(l2_iter))
        except StopIteration:
            break
    merged.extend(list(l1_iter))
    merged.extend(list(l2_iter))
    random.shuffle(merged)
    return merged

# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train_bilingual(l1_lang, l2_type="narrow"):
    print(f"\n=== Training {l1_lang.upper()} + L2 ({l2_type}) ===")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Phase 1: L1 only
    phase1_tokens = int(total_tokens * phase1_fraction)
    l1_phase1_ds = load_and_tokenize_l1(l1_lang, tokenizer, max_tokens=phase1_tokens)

    # Phase 2: remaining L1 + L2
    remaining_l1_tokens = l1_tokens_total - sum(l1_phase1_ds["n_tokens"])
    l1_phase2_ds = load_and_tokenize_l1(l1_lang, tokenizer, max_tokens=remaining_l1_tokens)

    if l2_type == "narrow":
        l2_phase2_ds = load_and_tokenize_l1(l2_datasets["narrow"], tokenizer, max_tokens=l2_tokens_total)
    else:
        l2_phase2_ds = stream_redpajama(tokenizer, max_tokens=l2_tokens_total)

    phase2_ds = interleave_l1_l2(l1_phase2_ds, l2_phase2_ds)

    # Combine datasets
    input_ids = sum([x["input_ids"] for x in l1_phase1_ds] + [x["input_ids"] for x in phase2_ds], [])
    train_ds = Dataset.from_dict({"input_ids": input_ids})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=f"{save_dir_root}/{l1_lang}_{l2_type}",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=500,
        fp16=True,
        push_to_hub=False
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=data_collator)

    # Checkpointing by L2 tokens
    total_l2_tokens = sum([len(x["input_ids"]) for x in l2_phase2_ds])
    checkpoint_tokens = [int(frac * total_l2_tokens) for frac in l2_checkpoints]
    current_l2_tokens = 0
    checkpoint_idx = 0

    for batch in trainer.get_train_dataloader():
        trainer.training_step(batch)
        batch_l2_tokens = sum(len(ids) for ids in batch["input_ids"] if ids in l2_phase2_ds)
        current_l2_tokens += batch_l2_tokens
        while checkpoint_idx < len(checkpoint_tokens) and current_l2_tokens >= checkpoint_tokens[checkpoint_idx]:
            cp_percent = int(l2_checkpoints[checkpoint_idx]*100)
            trainer.save_model(f"{save_dir_root}/{l1_lang}_{l2_type}_checkpoint_{cp_percent}")
            checkpoint_idx += 1

# ------------------------------
# RUN ALL EXPERIMENTS
# ------------------------------
for l1 in l1_languages:
    train_bilingual(l1, l2_type="narrow")
    train_bilingual(l1, l2_type="broad")
