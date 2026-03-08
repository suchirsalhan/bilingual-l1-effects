#!/usr/bin/env python3
import os
import time
import queue
import threading
import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, HfApi
from multiprocessing import Pool, cpu_count

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
TOKENIZER_ID = f"{HF_USER}/tokenizer-es-en"

SEQ_LEN = 512
SHARD_SIZE = 100_000 
BATCH_SIZE = 1000    
NUM_PROC = cpu_count()

# Specific naming requested by user
REPO_NAMING = {
    "es": "es-en-5B",
    "en": "en-es-5B"
}

TARGETS = {
    "es": 3_500_000_000,
    "en": 2_000_000_000
}

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUT_DIR / "processing_stats.csv"

api = HfApi(token=HF_TOKEN)

# Initialize CSV log with headers if it doesn't exist
if not LOG_FILE.exists():
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lang", "shard", "tokens", "duration_sec", "tokens_per_sec"])

# =====================================================
# 1. LOGGING & HELPERS
# =====================================================
def log_stats(lang, shard, tokens, duration):
    """Writes a row to the CSV log file."""
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
            print(f"❌ Upload failed for {fname}: {e}")
        upload_queue.task_done()

# =====================================================
# 4. MAIN PIPELINE
# =====================================================
def get_stream(lang):
    if lang == "es":
        return load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

def run_pipeline(lang, target_tokens):
    repo_name = REPO_NAMING.get(lang, f"{lang}-5B-dataset")
    repo_id = f"{HF_USER}/{repo_name}"
    
    print(f"🚀 Starting {lang} -> {repo_id} ({NUM_PROC} cores)")
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    stream = iter(get_stream(lang))
    buffer = []
    tokens_seen = 0
    shard = 0
    
    # Track time for the entire language run
    pipeline_start = time.time()
    # Track time for individual shard
    shard_start_time = time.perf_counter()

    with Pool(processes=NUM_PROC, initializer=init_worker) as pool:
        while tokens_seen < target_tokens:
            batch_texts = []
            try:
                for _ in range(BATCH_SIZE * NUM_PROC // 2):
                    batch_texts.append(next(stream)["text"].replace("\n", " "))
            except StopIteration:
                if not batch_texts: break

            sub_batches = [batch_texts[i:i + BATCH_SIZE] for i in range(0, len(batch_texts), BATCH_SIZE)]
            results = pool.map(tokenize_batch, sub_batches)
            
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
                
                # Calculate timing for this specific shard
                shard_duration = time.perf_counter() - shard_start_time
                log_stats(lang, shard, valid_len, shard_duration)
                
                tokens_seen += valid_len
                shard += 1
                
                # Reset shard timer
                shard_start_time = time.perf_counter()
                
                total_elapsed = time.time() - pipeline_start
                print(f"[{lang.upper()}] Shard {shard-1} | {tokens_seen/1e9:.2f}B Total | Rate: {tokens_seen/total_elapsed:,.0f} tok/s")

    print(f"✅ Finished {lang} tokenization.")

if __name__ == "__main__":
    up_t = threading.Thread(target=uploader_thread, daemon=True)
    up_t.start()

    for lang, target in TARGETS.items():
        run_pipeline(lang, target)

    print("⏳ Finishing final uploads...")
    upload_queue.join()
    print(f"\n💎 Done. Stats saved to {LOG_FILE}")
