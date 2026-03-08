#!/usr/bin/env python3
import os
import time
import queue
import threading
import csv
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# =====================================================
# CONFIG & ARGPARSE
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, default="es", help="L1 language code (tr, zh, ar, pl, de, fr, es)")
args = parser.parse_args()

L1 = args.lang_l1
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Dynamically set tokenizer based on L1
TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"

SEQ_LEN = 512
SHARD_SIZE = 100_000 
BATCH_SIZE = 5000  
NUM_PROC = cpu_count()

# Replicating your REPO_NAMING and TARGETS structure for the chosen L1
REPO_NAMING = {
    L1: f"{L1}-en-5B",
    "en": f"en-{L1}-5B"
}

TARGETS = {
    L1: 3_500_000_000,
    "en": 2_000_000_000
}

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUT_DIR / f"processing_stats_{L1}.csv"

api = HfApi(token=HF_TOKEN)

if not LOG_FILE.exists():
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lang", "shard", "tokens", "duration_sec", "tokens_per_sec"])

# =====================================================
# 1. LOGGING & HELPERS
# =====================================================
def log_stats(lang, shard, tokens, duration):
    rate = tokens / duration if duration > 0 else 0
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lang, shard, tokens, f"{duration:.2f}", f"{rate:.0f}"])

# =====================================================
# 2. PARALLEL TOKENIZER WORKER
# =====================================================
_tokenizer = None

def init_worker():
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN, use_fast=True)

def tokenize_batch(texts):
    return _tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]

# =====================================================
# 3. ASYNC UPLOADER
# =====================================================
upload_queue = queue.Queue()

def uploader_thread():
    while True:
        item = upload_queue.get()
        if item is None: break
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
            print(f"\n❌ Upload failed for {fname}: {e}")
        upload_queue.task_done()

# =====================================================
# 4. MAIN PIPELINE
# =====================================================
def get_stream(lang):
    # If lang matches our L1 argument, use CulturaX, otherwise use FineWeb-Edu for 'en'
    if lang == L1:
        return load_dataset("uonlp/CulturaX", L1, split="train", streaming=True)
    return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

def run_pipeline(lang, target_tokens):
    repo_name = REPO_NAMING.get(lang, f"{lang}-5B-dataset")
    repo_id = f"{HF_USER}/{repo_name}"
    
    print(f"\n🚀 Initializing {lang.upper()} pipeline using {TOKENIZER_ID}...")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    stream = iter(get_stream(lang))
    buffer = []
    tokens_seen = 0
    shard = 0
    
    pbar = tqdm(total=target_tokens, unit="tok", desc=f"🔤 {lang.upper()}", dynamic_ncols=True, unit_scale=True)
    shard_start_time = time.perf_counter()

    with Pool(processes=NUM_PROC, initializer=init_worker) as pool:
        while tokens_seen < target_tokens:
            batch_texts = []
            try:
                num_to_pull = BATCH_SIZE * NUM_PROC
                for _ in range(num_to_pull):
                    batch_texts.append(next(stream)["text"])
            except StopIteration:
                if not batch_texts: break

            sub_batches = [batch_texts[i:i + BATCH_SIZE] for i in range(0, len(batch_texts), BATCH_SIZE)]
            results = pool.map(tokenize_batch, sub_batches, chunksize=1)
            
            for sub_result in results:
                for ids in sub_result:
                    buffer.extend(ids)

            while len(buffer) >= SEQ_LEN * SHARD_SIZE:
                valid_len = SHARD_SIZE * SEQ_LEN
                to_save = buffer[:valid_len]
                buffer = buffer[valid_len:]

                blocks = [np.array(to_save[i : i + SEQ_LEN], dtype=np.int32) for i in range(0, valid_len, SEQ_LEN)]
                table = pa.Table.from_arrays([pa.array(blocks)], names=["input_ids"])
                fname = OUT_DIR / f"{lang}_train_{shard:04d}.parquet"
                pq.write_table(table, fname, compression='snappy')
                
                upload_queue.put((fname, repo_id))
                
                shard_duration = time.perf_counter() - shard_start_time
                log_stats(lang, shard, valid_len, shard_duration)
                
                tokens_seen += valid_len
                shard += 1
                pbar.update(valid_len)
                shard_start_time = time.perf_counter()

    pbar.close()
    print(f"✅ Finished {lang} tokenization.")

if __name__ == "__main__":
    up_t = threading.Thread(target=uploader_thread, daemon=True)
    up_t.start()

    # Iterates through L1 and then EN
    for lang, target in TARGETS.items():
        try:
            run_pipeline(lang, target)
        except Exception as e:
            print(f"Critical error in {lang} pipeline: {e}")

    print("\n⏳ Finishing final uploads...")
    upload_queue.join()
    print(f"💎 Done. Stats saved to {LOG_FILE}")
