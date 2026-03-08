#!/usr/bin/env python3
# run torchrun --nproc_per_node=8  ./tokenizers/pretokenize.py --lang_l1 fr or similar 
import os
import time
import csv
import queue
import threading
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

# ----------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, default="es")
parser.add_argument("--seq_len", type=int, default=512)
args = parser.parse_args()

L1 = args.lang_l1
SEQ_LEN = args.seq_len

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")

TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"

TARGETS = {
    L1: 3_500_000_000,
    "en": 2_000_000_000
}

REPO_NAMING = {
    L1: f"{L1}-en-5B",
    "en": f"en-{L1}-5B"
}

# ----------------------------------------------------
# PERFORMANCE SETTINGS
# ----------------------------------------------------

TOKENIZER_BATCH = 30000
SHARD_SEQS = 100000
TOKENS_PER_SHARD = SHARD_SEQS * SEQ_LEN

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUT_DIR / f"stats_{L1}.csv"

# ----------------------------------------------------
# HF API
# ----------------------------------------------------

api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------

if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp","lang","shard","tokens","sec","tok_per_sec"
        ])

def log_stats(lang, shard, tokens, sec):

    rate = tokens/sec if sec > 0 else 0

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            lang,
            shard,
            tokens,
            f"{sec:.2f}",
            int(rate)
        ])

# ----------------------------------------------------
# ASYNC UPLOAD
# ----------------------------------------------------

upload_queue = queue.Queue()

def uploader():

    while True:

        item = upload_queue.get()

        if item is None:
            break

        fname, repo_id = item

        try:

            api.upload_file(
                path_or_fileobj=str(fname),
                path_in_repo=fname.name,
                repo_id=repo_id,
                repo_type="dataset"
            )

            os.remove(fname)

        except Exception as e:

            print(f"Upload failed: {e}")

        upload_queue.task_done()

# ----------------------------------------------------
# DATA STREAM
# ----------------------------------------------------

def get_stream(lang):

    if lang == L1:

        return load_dataset(
            "uonlp/CulturaX",
            L1,
            split="train",
            streaming=True
        )

    return load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True
    )

# ----------------------------------------------------
# SHARD WRITER
# ----------------------------------------------------

def write_shard(token_array, shard_id, lang, repo_id):

    table = pa.Table.from_arrays(
        [token_array],
        names=["input_ids"]
    )

    fname = OUT_DIR / f"{lang}_train_{shard_id:05d}.parquet"

    pq.write_table(
        table,
        fname,
        compression="zstd",
        row_group_size=4096
    )

    upload_queue.put((fname, repo_id))

# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------

def run_pipeline(lang, target_tokens, tokenizer):

    repo_name = REPO_NAMING[lang]
    repo_id = f"{HF_USER}/{repo_name}"

    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=HF_TOKEN
    )

    print(f"\n🚀 Processing {lang} → {repo_id}")

    stream = iter(get_stream(lang))

    shard = 0
    tokens_seen = 0

    shard_start = time.perf_counter()

    flat_tokens = []

    pbar = tqdm(
        total=target_tokens,
        unit="tok",
        desc=lang,
        unit_scale=True
    )

    while tokens_seen < target_tokens:

        texts = []

        try:
            for _ in range(TOKENIZER_BATCH):
                texts.append(next(stream)["text"])
        except StopIteration:
            if not texts:
                break

        enc = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False
        )

        for ids in enc["input_ids"]:
            flat_tokens.extend(ids)

        while len(flat_tokens) >= TOKENS_PER_SHARD:

            shard_tokens = flat_tokens[:TOKENS_PER_SHARD]
            flat_tokens = flat_tokens[TOKENS_PER_SHARD:]

            arr = pa.FixedSizeListArray.from_arrays(
                pa.array(shard_tokens, type=pa.int32()),
                SEQ_LEN
            )

            write_shard(arr, shard, lang, repo_id)

            duration = time.perf_counter() - shard_start

            log_stats(lang, shard, TOKENS_PER_SHARD, duration)

            tokens_seen += TOKENS_PER_SHARD
            shard += 1

            pbar.update(TOKENS_PER_SHARD)

            shard_start = time.perf_counter()

    pbar.close()

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        token=HF_TOKEN,
        use_fast=True
    )

    uploader_thread = threading.Thread(target=uploader, daemon=True)
    uploader_thread.start()

    for lang, target in TARGETS.items():

        try:
            run_pipeline(lang, target, tokenizer)
        except Exception as e:
            print(f"Pipeline error: {e}")

    upload_queue.join()

    print("\nDataset generation complete.")

if __name__ == "__main__":
    main()
