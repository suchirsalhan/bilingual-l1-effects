#!/usr/bin/env python3
import os
import time
import queue
import threading
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
SHARD_SIZE = 100_000  # Increased to ~50M tokens per file for better efficiency
BATCH_SIZE = 1000     # Number of docs to send to CPUs at once
NUM_PROC = cpu_count() # Detects all 128 cores

TARGETS = {
    "es": 3_500_000_000,
    "en": 2_000_000_000
}

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)
api = HfApi(token=HF_TOKEN)

# =====================================================
# 1. PARALLEL TOKENIZER WORKER
# =====================================================
# Global tokenizer instance per process to avoid re-loading
_tokenizer = None

def init_worker():
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN, use_fast=True)

def tokenize_batch(texts):
    # This runs in parallel on 128 cores
    return _tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]

# =====================================================
# 2. ASYNC UPLOADER
# =====================================================
upload_queue = queue.Queue()

def uploader_thread():
    """Background worker to upload files to HF Hub"""
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
# 3. MAIN PIPELINE
# =====================================================
def get_stream(lang):
    if lang == "es":
        return load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

def run_pipeline(lang, target_tokens):
    print(f"🚀 Starting {lang} pipeline with {NUM_PROC} cores...")
    repo_id = f"{HF_USER}/{lang}-5B-parquet"
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    stream = iter(get_stream(lang))
    buffer = []
    tokens_seen = 0
    shard = 0
    start_time = time.time()

    # Use Pool to distribute tokenization
    with Pool(processes=NUM_PROC, initializer=init_worker) as pool:
        while tokens_seen < target_tokens:
            # 1. Collect a massive batch from the stream
            batch_texts = []
            try:
                for _ in range(BATCH_SIZE * NUM_PROC // 2): # Feed the hungry CPUs
                    batch_texts.append(next(stream)["text"].replace("\n", " "))
            except StopIteration:
                if not batch_texts: break

            # 2. Parallelize Tokenization
            # Split batch into sub-batches for the workers
            sub_batches = [batch_texts[i:i + BATCH_SIZE] for i in range(0, len(batch_texts), BATCH_SIZE)]
            results = pool.map(tokenize_batch, sub_batches)
            
            # Flatten results into buffer
            for sub_result in results:
                for ids in sub_result:
                    buffer.extend(ids)

            # 3. Check if we have enough for a shard
            while len(buffer) >= SEQ_LEN * SHARD_SIZE:
                valid_len = SHARD_SIZE * SEQ_LEN
                to_save = buffer[:valid_len]
                buffer = buffer[valid_len:]

                # Chunk into SEQ_LEN rows
                blocks = [np.array(to_save[i : i + SEQ_LEN], dtype=np.int32) for i in range(0, valid_len, SEQ_LEN)]
                
                table = pa.Table.from_arrays([pa.array(blocks)], names=["input_ids"])
                fname = OUT_DIR / f"{lang}_train_{shard:04d}.parquet"
                pq.write_table(table, fname, compression='snappy')
                
                # Hand off to uploader thread
                upload_queue.put((fname, repo_id))
                
                tokens_seen += valid_len
                shard += 1
                
                elapsed = time.time() - start_time
                print(f"[{lang.upper()}] {tokens_seen/1e9:.2f}B / {target_tokens/1e9}B | Rate: {tokens_seen/elapsed:,.0f} tok/s")

    print(f"✅ Finished {lang} tokenization.")

if __name__ == "__main__":
    # Start uploader thread
    up_t = threading.Thread(target=uploader_thread, daemon=True)
    up_t.start()

    # Run languages sequentially (the Pool handles 128 cores per lang)
    # This avoids memory thrashing by trying to stream two huge datasets at once
    for lang, target in TARGETS.items():
        run_pipeline(lang, target)

    # Wait for remaining uploads
    print("⏳ Finishing final uploads...")
    upload_queue.join()
    print("\n💎 Done. All datasets tokenized and uploaded.")
