#!/usr/bin/env python3
"""
Memory-safe multi-GPU pretokenizer + async shard uploader.

Features:
- Async upload thread to Hugging Face Hub with retries
- Fully resumable shard IDs per rank
- Memory-limited token buffer
- /tmp free space safety check
- Optional NVMe temp directory
- Multi-GPU safe (WORLD_SIZE & LOCAL_RANK)
"""

import os
import time
import argparse
import shutil
import threading
from pathlib import Path
import queue

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import shutil

# ------------------------------
# ARGUMENTS
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, default="es")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--shard_seqs", type=int, default=50_000)
parser.add_argument("--tokenizer_batch", type=int, default=20_000)
parser.add_argument("--upload_batch", type=int, default=40)
parser.add_argument("--nvme_tmp", type=str, default=None, help="Optional NVMe mount for temp shards")
args = parser.parse_args()

L1 = args.lang_l1
SEQ_LEN = args.seq_len
SHARD_SEQS = args.shard_seqs
TOKENIZER_BATCH = args.tokenizer_batch
UPLOAD_BATCH = args.upload_batch
TOKENS_PER_SHARD = SEQ_LEN * SHARD_SEQS
MAX_BUFFER_SHARDS = 2  # Max shards held in memory at once

# ------------------------------
# HF SETTINGS
# ------------------------------
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not set!")

TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"
TARGETS = {L1: 4_000_000_000, "en": 2_500_000_000}
REPOS = {L1: f"{L1}-en-5B", "en": f"en-{L1}-5B"}

# ------------------------------
# DISTRIBUTED INFO
# ------------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def log(msg):
    print(f"[Rank {LOCAL_RANK}] {msg}", flush=True)

api = HfApi(token=HF_TOKEN)
upload_queue = queue.Queue(maxsize=UPLOAD_BATCH*WORLD_SIZE)  # limit memory

# ------------------------------
# ASYNC UPLOADER THREAD
# ------------------------------
def uploader(repo_id):
    while True:
        fname = upload_queue.get()
        if fname is None:
            break
        retries = 0
        while retries < 5:
            try:
                api.upload_file(
                    path_or_fileobj=str(fname),
                    path_in_repo=fname.name,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                fname.unlink()
                log(f"✅ Uploaded {fname.name}")
                break
            except Exception as e:
                retries += 1
                wait = min(60, 2 ** retries)
                log(f"Upload failed {fname.name}, retry {retries}, waiting {wait}s: {e}")
                time.sleep(wait)
        upload_queue.task_done()

# ------------------------------
# DATA STREAM
# ------------------------------
def get_stream(lang):
    if lang == L1:
        return load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# ------------------------------
# WRITE SHARD
# ------------------------------
def write_shard(tokens, shard_id, lang, tmp_dir):
    flat = pa.array(tokens, type=pa.int32())
    arr = pa.FixedSizeListArray.from_arrays(flat, SEQ_LEN)
    table = pa.Table.from_arrays([arr], names=["input_ids"])
    fname = tmp_dir / f"{lang}_train_{shard_id:06d}_rank{LOCAL_RANK}.parquet"
    pq.write_table(table, fname, compression="zstd", row_group_size=4096)
    upload_queue.put(fname)
    return fname

# ------------------------------
# PIPELINE
# ------------------------------
def run_pipeline(lang, target_tokens, tokenizer, repo_id):
    # Setup temp dir
    tmp_base = Path(args.nvme_tmp) if args.nvme_tmp else Path("/tmp")
    tmp_dir = tmp_base / f"hf_shards_{lang}_rank{LOCAL_RANK}"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # Pre-flight /tmp space check
    usage = shutil.disk_usage(str(tmp_dir))
    needed_bytes = TOKENS_PER_SHARD * 4 * MAX_BUFFER_SHARDS  # 4 bytes per token
    if usage.free < needed_bytes:
        raise RuntimeError(f"/tmp has {usage.free/1e9:.2f} GB free, need at least {needed_bytes/1e9:.2f} GB")
    log(f"/tmp free space OK: {usage.free/1e9:.2f} GB")

    # Flush leftover shards
    leftover_shards = list(tmp_dir.glob("*.parquet"))
    for f in leftover_shards:
        upload_queue.put(f)
    if leftover_shards:
        log(f"Flushed {len(leftover_shards)} leftover shards")

    # Determine starting shard ID
    existing_shards = list(tmp_dir.glob(f"{lang}_train_*_rank{LOCAL_RANK}.parquet"))
    if existing_shards:
        max_id = max(int(f.stem.split("_")[2]) for f in existing_shards)
        shard = max_id + WORLD_SIZE
    else:
        shard = LOCAL_RANK

    per_rank_target = target_tokens // WORLD_SIZE
    tokens_seen = 0
    buffer = np.empty(0, dtype=np.int32)

    stream = iter(get_stream(lang))
    pbar = tqdm(total=per_rank_target, unit="tok", unit_scale=True,
                desc=f"{lang}-R{LOCAL_RANK}", position=LOCAL_RANK)

    while tokens_seen < per_rank_target:
        texts = []
        for _ in range(TOKENIZER_BATCH):
            try:
                texts.append(next(stream)["text"])
            except StopIteration:
                break
        if not texts:
            break

        enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        tokens = np.concatenate([np.array(ids, dtype=np.int32) for ids in enc["input_ids"]])
        buffer = np.concatenate([buffer, tokens])

        # Write shards while limiting memory
        while buffer.size >= TOKENS_PER_SHARD and buffer.size >= TOKENS_PER_SHARD * MAX_BUFFER_SHARDS:
            shard_tokens = buffer[:TOKENS_PER_SHARD]
            buffer = buffer[TOKENS_PER_SHARD:]
            write_shard(shard_tokens, shard, lang, tmp_dir)
            shard += WORLD_SIZE
            tokens_seen += TOKENS_PER_SHARD
            pbar.update(TOKENS_PER_SHARD)

    # Flush remaining tokens
    while buffer.size >= TOKENS_PER_SHARD:
        shard_tokens = buffer[:TOKENS_PER_SHARD]
        buffer = buffer[TOKENS_PER_SHARD:]
        write_shard(shard_tokens, shard, lang, tmp_dir)
        shard += WORLD_SIZE
        tokens_seen += TOKENS_PER_SHARD
        pbar.update(TOKENS_PER_SHARD)

    pbar.close()
    log(f"Finished {lang} tokenization")

# ------------------------------
# MAIN
# ------------------------------
def main():
    log("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, token=HF_TOKEN)

    for lang, target in TARGETS.items():
        repo_id = f"{HF_USER}/{REPOS[lang]}"

        # Rank 0 creates repo
        if LOCAL_RANK == 0:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

        # Barrier: wait for repo creation
        while True:
            try:
                api.dataset_info(repo_id)
                break
            except Exception:
                log("Waiting for repo creation...")
                time.sleep(2)

        # Start async uploader
        t = threading.Thread(target=uploader, args=(repo_id,), daemon=True)
        t.start()

        run_pipeline(lang, target, tokenizer, repo_id)

        # Signal uploader to stop
        upload_queue.put(None)
        t.join()

        # Ensure all remaining files uploaded
        while not upload_queue.empty():
            time.sleep(1)

        # Cleanup tmp
        tmp_dir = (Path(args.nvme_tmp) if args.nvme_tmp else Path("/tmp")) / f"hf_shards_{lang}_rank{LOCAL_RANK}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    log("✅ Dataset creation complete")

if __name__ == "__main__":
    main()
