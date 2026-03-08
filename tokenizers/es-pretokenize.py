#!/usr/bin/env python3
import os
import time
import multiprocessing as mp
from threading import Thread
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer, HfApi

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
TOKENIZER_ID = f"{HF_USER}/tokenizer-es-en"

SEQ_LEN = 512
SHARD_SIZE = 50_000  # ~25M tokens per file

# Language-specific targets
TARGETS = {
    "es": 3_500_000_000,
    "en": 2_000_000_000
}

OUT_DIR = Path("parquet_factory")
OUT_DIR.mkdir(exist_ok=True)
api = HfApi(token=HF_TOKEN)

# =====================================================
# 1. STREAM SELECTOR
# =====================================================
def get_stream(lang):
    if lang == "es":
        return load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    else:
        return load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# =====================================================
# 2. CORE PIPELINE (Per Language)
# =====================================================
def processing_pipeline(lang, target_tokens):
    print(f"🏁 Starting pipeline for {lang} (Target: {target_tokens/1e9}B)")
    
    repo_id = f"{HF_USER}/{lang}-5B-parquet"
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)
    stream = iter(get_stream(lang))
    
    buffer = []
    tokens_seen = 0
    shard = 0
    start_time = time.time()

    while tokens_seen < target_tokens:
        try:
            item = next(stream)
            text = item["text"].replace("\n", " ")
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            
            # When we have enough for a shard
            if len(buffer) >= SEQ_LEN * SHARD_SIZE:
                blocks = []
                # Chunk into SEQ_LEN blocks
                for i in range(0, SHARD_SIZE * SEQ_LEN, SEQ_LEN):
                    blocks.append(np.array(buffer[i : i + SEQ_LEN], dtype=np.int32))
                
                # Save Parquet
                table = pa.Table.from_arrays([pa.array(blocks)], names=["input_ids"])
                fname = OUT_DIR / f"{lang}_train_{shard:04d}.parquet"
                pq.write_table(table, fname, compression='snappy')
                
                # Upload and Clean
                api.upload_file(path_or_fileobj=str(fname), path_in_repo=fname.name, repo_id=repo_id, repo_type="dataset")
                os.remove(fname)
                
                tokens_seen += (SHARD_SIZE * SEQ_LEN)
                buffer = buffer[SHARD_SIZE * SEQ_LEN:]
                shard += 1
                
                elapsed = time.time() - start_time
                print(f"[{lang.upper()}] {tokens_seen/1e9:.2f}B / {target_tokens/1e9}B | {tokens_seen/elapsed:,.0f} tok/s")
                
        except StopIteration:
            break

    print(f"✅ Finished {lang} dataset.")

# =====================================================
# EXECUTION
# =====================================================
if __name__ == "__main__":
    # We use multiprocessing to run both languages simultaneously
    processes = []
    for lang, target in TARGETS.items():
        p = mp.Process(target=processing_pipeline, args=(lang, target))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n💎 Both datasets are fully tokenized and uploaded.")
