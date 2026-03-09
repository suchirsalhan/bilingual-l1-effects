#!/usr/bin/env python3
"""
Stable multi-GPU pretokenizer for large datasets.
Safer shard sizes, full debugging output, rank-safe repo creation.
Run with:
torchrun --nproc_per_node=8 ./tokenizers/pretokenize_safe.py --lang_l1 fr --seq_len 512
"""

import os
import sys
import time
import csv
import queue
import threading
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

# ----------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, default="es")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--shard_seqs", type=int, default=50_000, help="Safer, smaller shard size")
parser.add_argument("--tokenizer_batch", type=int, default=20_000)
args = parser.parse_args()

L1 = args.lang_l1
SEQ_LEN = args.seq_len
SHARD_SEQS = args.shard_seqs
TOKENIZER_BATCH = args.tokenizer_batch
TOKENS_PER_SHARD = SHARD_SEQS * SEQ_LEN

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable not set!")

TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"

TARGETS = {
    L1: 3_500_000_000,
    "en": 2_500_000_000
}

REPO_NAMING = {
    L1: f"{L1}-en-5B",
    "en": f"en-{L1}-5B"
}

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# DISTRIBUTED INFO
# ----------------------------------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

LOG_FILE = OUT_DIR / f"stats_{L1}_rank{LOCAL_RANK}.csv"
if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "lang", "shard", "tokens", "sec", "tok_per_sec"])

# HF API
api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------
# DEBUGGING HELPER
# ----------------------------------------------------
def debug_print(msg):
    print(f"[Rank {LOCAL_RANK}] {msg}", flush=True)

# ----------------------------------------------------
# ASYNC UPLOAD THREAD
# ----------------------------------------------------
upload_queue = queue.Queue()

def uploader(max_retries=5, base_delay=2):
    """
    Async uploader with automatic retries and exponential backoff.
    """
    while True:
        item = upload_queue.get()
        if item is None:
            break
        fname, repo_id = item
        retries = 0
        while retries <= max_retries:
            try:
                api.upload_file(
                    path_or_fileobj=str(fname),
                    path_in_repo=fname.name,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                os.remove(fname)
                debug_print(f"Uploaded {fname.name} successfully.")
                time.sleep(1)  # small delay between uploads to avoid HF rate limits
                break  # success, exit retry loop
            except Exception as e:
                retries += 1
                wait_time = base_delay * (2 ** (retries - 1))  # exponential backoff
                debug_print(f"Upload failed ({retries}/{max_retries}) for {fname.name}: {e}")
                if retries > max_retries:
                    debug_print(f"Max retries exceeded for {fname.name}, skipping file.")
                    break
                debug_print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        upload_queue.task_done()
# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
def log_stats(lang, shard, tokens, sec):
    rate = tokens / sec if sec > 0 else 0
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
# DATA STREAM
# ----------------------------------------------------
def get_stream(lang):
    try:
        if lang == L1:
            return load_dataset("uonlp/CulturaX", L1, split="train", streaming=True)
        else:
            return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    except Exception as e:
        debug_print(f"Dataset load failed: {e}")
        sys.exit(1)

# ----------------------------------------------------
# SHARD WRITER
# ----------------------------------------------------
def write_shard(token_array, shard_id, lang, repo_id):
    try:
        table = pa.Table.from_arrays([token_array], names=["input_ids"])
        fname = OUT_DIR / f"{lang}_train_{shard_id:05d}_rank{LOCAL_RANK}.parquet"
        pq.write_table(table, fname, compression="zstd", row_group_size=4096)
        upload_queue.put((fname, repo_id))
    except Exception as e:
        debug_print(f"Shard write failed: {e}")
        sys.exit(1)

# ----------------------------------------------------
# MAIN PIPELINE PER RANK
# ----------------------------------------------------
def run_pipeline(lang, target_tokens, tokenizer):
    repo_name = REPO_NAMING[lang]
    repo_id = f"{HF_USER}/{repo_name}"

    # Rank 0 creates repo
    if LOCAL_RANK == 0:
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        except Exception as e:
            debug_print(f"Repo creation failed: {e}")
            sys.exit(1)

    debug_print(f"Starting processing {lang} → {repo_id}")
    stream = iter(get_stream(lang))
    shard = LOCAL_RANK  # Start shard offset by rank
    tokens_seen = 0
    flat_tokens = []
    shard_start = time.perf_counter()

    pbar = tqdm(total=target_tokens, unit="tok",
                desc=f"{lang}-R{LOCAL_RANK}", unit_scale=True, position=LOCAL_RANK)

    while tokens_seen < target_tokens:
        texts = []
        try:
            for _ in range(TOKENIZER_BATCH):
                texts.append(next(stream)["text"])
        except StopIteration:
            if not texts:
                break
        except Exception as e:
            debug_print(f"Error fetching texts: {e}")
            sys.exit(1)

        try:
            enc = tokenizer(texts, add_special_tokens=False, truncation=False)
        except Exception as e:
            debug_print(f"Tokenizer failed: {e}")
            sys.exit(1)

        for ids in enc["input_ids"]:
            flat_tokens.extend(ids)

        while len(flat_tokens) >= TOKENS_PER_SHARD:
            shard_tokens = flat_tokens[:TOKENS_PER_SHARD]
            flat_tokens = flat_tokens[TOKENS_PER_SHARD:]
            try:
                arr = pa.FixedSizeListArray.from_arrays(pa.array(shard_tokens, type=pa.int32()), SEQ_LEN)
            except Exception as e:
                debug_print(f"FixedSizeListArray creation failed: {e}")
                sys.exit(1)
            write_shard(arr, shard, lang, repo_id)
            duration = time.perf_counter() - shard_start
            log_stats(lang, shard, TOKENS_PER_SHARD, duration)
            tokens_seen += TOKENS_PER_SHARD
            shard += WORLD_SIZE  # Next shard for this rank
            pbar.update(TOKENS_PER_SHARD)
            shard_start = time.perf_counter()

    pbar.close()
    debug_print(f"Finished {lang}")

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    debug_print(f"Loading tokenizer {TOKENIZER_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, token=HF_TOKEN)
    except Exception as e:
        debug_print(f"Tokenizer load failed: {e}")
        sys.exit(1)

    uploader_thread = threading.Thread(target=uploader)
    uploader_thread.start()

    for lang, target in TARGETS.items():
        try:
            run_pipeline(lang, target, tokenizer)
        except Exception as e:
            debug_print(f"Pipeline error: {e}")
            sys.exit(1)

    # Stop uploader
    upload_queue.put(None)
    uploader_thread.join()

    debug_print("Dataset generation complete.")

if __name__ == "__main__":
    main()
