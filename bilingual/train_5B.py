#!/usr/bin/env python3
"""
End-to-End 7-Language Trainer & Orchestrator

- Streams Hugging Face datasets
- Applies L1:L2 curriculum (5B total tokens)
- Runs 2–3 languages at a time safely
- Logs shard usage live
- Checkpoints frequently and pushes to HF
"""

import os
import subprocess
import time
from pathlib import Path
from datasets import load_dataset

# -----------------------------
# CONFIGURATION
# -----------------------------
LANGUAGES = [
    "zh-5B",
    "en-5B",
    "es-5B",
    "de-5B",
    "pl-5B",
    "fr-5B",
    "ar-5B",
]

HF_USER = "RA-ALTA"
MODEL_NAME = "gpt2"  # base model for finetuning
TOTAL_TOKENS = 5_000_000_000
L1_L2_RATIO = 2 / 3  # Example: 2/3 L1, 1/3 L2
BATCH_SIZE_GPUS = 2  # languages at a time
CHECKPOINT_INTERVAL = 500_000  # tokens
PYTHON_EXEC = "python3"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_shards(dataset_name: str, total_tokens: int, token_ratio: float) -> int:
    """Compute number of shards to use for this dataset based on L1:L2 split"""
    # Load dataset metadata without downloading full data
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    # Estimate shards using first shard length
    first_shard = next(iter(dataset))
    avg_tokens_per_item = first_shard.get("tokens", 128)  # fallback
    total_items = int(total_tokens * token_ratio / avg_tokens_per_item)
    shards = max(1, total_items // 1_000_000)  # crude approx: 1M items per shard
    return shards

def run_language_train(lang: str, gpu: int, shards: int):
    """Launch torchrun for a single language"""
    dataset_name = f"{HF_USER}/{lang}"
    log_file = LOG_DIR / f"{lang}.log"
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--standalone",
        __file__,  # run this script in "train mode"
        "--train_lang", lang,
        "--dataset_name", dataset_name,
        "--model_name", MODEL_NAME,
        "--gpu", str(gpu),
        "--shards", str(shards),
        "--checkpoint_interval", str(CHECKPOINT_INTERVAL),
    ]
    print(f"[INFO] Launching {lang} on GPU {gpu} using {shards} shards")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc, log_file

# -----------------------------
# TRAINING LOGIC
# -----------------------------
def train_language(args):
    """Train one language — streaming dataset, logging shards, checkpointing"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

    lang = args.train_lang
    dataset_name = args.dataset_name
    model_name = args.model_name
    gpu = int(args.gpu)
    shards = int(args.shards)
    checkpoint_interval = int(args.checkpoint_interval)

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[{lang}] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Streaming dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.take(shards * 1_000_000)  # approximate shard usage
    print(f"[{lang}] Streaming {shards} shards (~{shards*1_000_000} items)")

    # Dummy trainer for demonstration (replace with your actual tokenization + batching)
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{lang}",
        per_device_train_batch_size=1,
        logging_steps=10,
        save_steps=checkpoint_interval,
        save_total_limit=3,
        push_to_hub=True,
        report_to="none"
    )

    def batch_generator(ds_stream):
        count = 0
        for item in ds_stream:
            # Assume 'text' field exists
            tokens = tokenizer(item["text"], truncation=True, return_tensors="pt").to(device)
            yield {"input_ids": tokens["input_ids"].squeeze(0)}
            count += 1
            if count % 1000 == 0:
                print(f"[{lang}] Processed {count} items")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=list(batch_generator(ds))
    )

    trainer.train()
    trainer.push_to_hub()
    print(f"[{lang}] Training complete and pushed to HF")

# -----------------------------
# MAIN ORCHESTRATOR
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_lang", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=500_000)
    args = parser.parse_args()

    if args.train_lang:
        # If run in train mode (torchrun subprocess)
        train_language(args)
        return

    # Orchestrator mode
    lang_queue = LANGUAGES.copy()
    gpu_count = 8  # adjust based on your node
    while lang_queue:
        batch = lang_queue[:BATCH_SIZE_GPUS]
        lang_queue = lang_queue[BATCH_SIZE_GPUS:]
        processes = []
        log_files = []

        for i, lang in enumerate(batch):
            gpu_id = i % gpu_count
            shards = compute_shards(f"{HF_USER}/{lang}", TOTAL_TOKENS, L1_L2_RATIO)
            proc, log_file = run_language_train(lang, gpu_id, shards)
            processes.append(proc)
            log_files.append(log_file)

        # Monitor batch
        while any(p.poll() is None for p in processes):
            for lang, log_file, proc in zip(batch, log_files, processes):
                if proc.poll() is None and Path(log_file).exists():
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            print(f"[{lang}] {lines[-1].strip()}")
            time.sleep(10)

        # Check exit codes
        for lang, proc in zip(batch, processes):
            if proc.returncode != 0:
                print(f"[ERROR] {lang} failed (exit {proc.returncode})")
            else:
                print(f"[INFO] {lang} completed successfully")

        print("[INFO] Batch finished\n")

    print("[INFO] All languages finished")

if __name__ == "__main__":
    main()
