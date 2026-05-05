#!/usr/bin/env python3
"""
Stable multi-GPU tokenizer + shard uploader with dynamic /tmp safety

• Memory-safe token buffering
• Per-rank deterministic temp directories
• Async HF upload queue
• Distributed barriers for repo creation
• Resumable shard IDs
• Dynamic shard batching based on available disk space
• NCCL debug + timeout + graceful error handling (SIGABRT/SIGTERM fix)
"""

import os
import sys
import time
import shutil
import signal
import argparse
import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
import torch.distributed as dist
from tqdm import tqdm

# ----------------------------
# NCCL / CUDA env hardening
# Set BEFORE any torch/NCCL initialization so these take effect
# ----------------------------
os.environ.setdefault("NCCL_DEBUG", "INFO")           # verbose NCCL logs → tells you exactly which primitive aborted
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALL")     # capture all subsystems (INIT, COLL, P2P, SHM, etc.)
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # surface NCCL errors as Python exceptions instead of SIGABRT
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")      # block instead of aborting on timeout → cleaner Python traceback
os.environ.setdefault("NCCL_SHM_DISABLE", "0")        # keep SHM on; flip to "1" if you see SHM-related abort
os.environ.setdefault("NCCL_IB_DISABLE", "0")         # keep IB on; flip to "1" if no InfiniBand in your tenant net
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")  # kill a stuck rank after 2 min, not silently hang

# Route NCCL debug output to a per-rank file so logs from 8 ranks don't interleave in stdout
_rank_for_log = int(os.environ.get("LOCAL_RANK", 0))
_nccl_log_path = f"/tmp/nccl_debug_rank{_rank_for_log}.log"
os.environ.setdefault("NCCL_DEBUG_FILE", _nccl_log_path)

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, default="es")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--shard_seqs", type=int, default=50000)
parser.add_argument("--tokenizer_batch", type=int, default=10000)
parser.add_argument("--upload_batch", type=int, default=40)
parser.add_argument("--tmp_root", type=str, default="/tmp")
parser.add_argument("--nccl_timeout_min", type=int, default=10,
                    help="Distributed barrier/op timeout in minutes (default 10)")
args = parser.parse_args()

L1 = args.lang_l1
SEQ_LEN = args.seq_len
SHARD_SEQS = args.shard_seqs
TOKENIZER_BATCH = args.tokenizer_batch
UPLOAD_BATCH = args.upload_batch
TMP_ROOT = Path(args.tmp_root)
NCCL_TIMEOUT = datetime.timedelta(minutes=args.nccl_timeout_min)

TOKENS_PER_SHARD = SEQ_LEN * SHARD_SEQS

# ----------------------------
# HF / Distributed settings
# ----------------------------
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not set")

TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

TARGETS = {
    L1: 3_500_000_000,
    "en": 2_500_000_000
}

REPOS = {
    L1: f"{L1}-en-5B",
    "en": f"en-{L1}-5B"
}

api = HfApi(token=HF_TOKEN)


def log(msg):
    print(f"[Rank {LOCAL_RANK}] {msg}", flush=True)


# ----------------------------
# Graceful distributed teardown
# Called on SIGTERM so ranks don't leave orphan process groups
# ----------------------------
def _sigterm_handler(signum, frame):
    log("SIGTERM received — destroying process group and exiting cleanly")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)


# ----------------------------
# Safe distributed barrier
# Wraps dist.barrier() so a single rank failure raises a Python
# exception rather than hanging all other ranks until watchdog SIGTERMs them.
# ----------------------------
def safe_barrier(tag: str = ""):
    """
    barrier() with a human-readable tag for easier log correlation.
    Raises RuntimeError (instead of SIGABRT/hang) on NCCL failure.
    """
    if WORLD_SIZE <= 1:
        return
    log(f"barrier enter  [{tag}]")
    try:
        dist.barrier()
        log(f"barrier passed [{tag}]")
    except Exception as e:
        log(f"BARRIER FAILED [{tag}]: {e}")
        log(f"  → Check NCCL debug log: {_nccl_log_path}")
        # Tear down the group cleanly so sibling processes get an error
        # rather than hanging.
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise RuntimeError(f"dist.barrier() failed at [{tag}]: {e}") from e


# ----------------------------
# Async uploader
# ----------------------------
class UploadWorker(Thread):
    def __init__(self, tmp_dir, repo_id, upload_batch):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.repo_id = repo_id
        self.upload_batch = upload_batch
        self.queue = Queue()
        self.stop_flag = False
        self.daemon = True
        self.start()

    def run(self):
        while not self.stop_flag or not self.queue.empty():
            try:
                batch = self.queue.get(timeout=1)
            except Exception:
                continue

            retries = 0
            while True:
                try:
                    api.upload_folder(
                        folder_path=str(self.tmp_dir),
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        allow_patterns=[f.name for f in batch]
                    )
                    for f in batch:
                        f.unlink()
                    break
                except Exception as e:
                    retries += 1
                    wait = min(120, 2 ** retries)
                    log(f"Upload failed (attempt {retries}), retrying in {wait}s — {e}")
                    time.sleep(wait)

    def add_batch(self, batch_files):
        self.queue.put(batch_files)

    def stop(self):
        self.stop_flag = True
        self.join()


# ----------------------------
# Temp space safety
# ----------------------------
def get_max_shards(tmp_dir: Path) -> int:
    """Determine maximum shards that can fit safely in tmp_dir."""
    usage = shutil.disk_usage(tmp_dir)
    free_bytes = usage.free
    bytes_per_shard = TOKENS_PER_SHARD * 4  # int32
    max_shards = max(1, free_bytes // bytes_per_shard)
    log(f"Free space: {free_bytes / 1e9:.2f} GB  →  max buffered shards: {max_shards}")
    return int(max_shards)


# ----------------------------
# Shard helpers
# ----------------------------
def get_next_shard_id(repo_id: str, lang: str) -> int:
    """Fetch highest shard ID from HF, then return the next one for this rank."""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        shard_ids = [
            int(f.split("_train_")[1].split("_")[0])
            for f in files
            if f.startswith(lang) and "_train_" in f
        ]
        return max(shard_ids) + 1 if shard_ids else LOCAL_RANK
    except Exception as e:
        log(f"get_next_shard_id failed ({e}), defaulting to LOCAL_RANK={LOCAL_RANK}")
        return LOCAL_RANK


def write_shard(tokens, shard_id: int, lang: str, tmp_dir: Path) -> Path:
    arr = pa.array(tokens, type=pa.int32())
    arr = pa.FixedSizeListArray.from_arrays(arr, SEQ_LEN)
    table = pa.Table.from_arrays([arr], names=["input_ids"])
    fname = tmp_dir / f"{lang}_train_{shard_id:06d}_rank{LOCAL_RANK}.parquet"
    pq.write_table(table, fname, compression="zstd", row_group_size=4096)
    return fname


# ----------------------------
# Dataset streaming
# ----------------------------
def get_stream(lang: str):
    if lang == L1:
        return load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(lang: str, target_tokens: int, tokenizer, repo_id: str):
    per_rank_tokens = target_tokens // WORLD_SIZE
    tmp_dir = TMP_ROOT / f"hf_shards_{lang}_rank{LOCAL_RANK}"
    tmp_dir.mkdir(exist_ok=True)

    max_shards = get_max_shards(tmp_dir)
    upload_batch_dynamic = min(UPLOAD_BATCH, max_shards)

    # Flush any leftover shards from a previous interrupted run
    leftover_files = sorted(tmp_dir.glob("*.parquet"))
    if leftover_files:
        log(f"Flushing {len(leftover_files)} leftover shard(s) from previous run")
        flusher = UploadWorker(tmp_dir, repo_id, upload_batch_dynamic)
        flusher.add_batch(leftover_files)
        flusher.stop()

    uploader = UploadWorker(tmp_dir, repo_id, upload_batch_dynamic)
    shard_id = get_next_shard_id(repo_id, lang)

    buffer = []
    tokens_seen = 0
    pbar = tqdm(
        total=per_rank_tokens,
        unit="tok",
        unit_scale=True,
        desc=f"{lang}-R{LOCAL_RANK}",
    )
    stream = iter(get_stream(lang))

    while tokens_seen < per_rank_tokens:
        texts = []
        for _ in range(TOKENIZER_BATCH):
            try:
                texts.append(next(stream)["text"])
            except StopIteration:
                break
        if not texts:
            break

        enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        token_arrays = [np.array(ids, dtype=np.int32) for ids in enc["input_ids"]]

        for arr in token_arrays:
            buffer.extend(arr.tolist())

        while len(buffer) >= TOKENS_PER_SHARD:
            shard_tokens = buffer[:TOKENS_PER_SHARD]
            buffer = buffer[TOKENS_PER_SHARD:]
            write_shard(shard_tokens, shard_id, lang, tmp_dir)
            shard_id += WORLD_SIZE
            tokens_seen += TOKENS_PER_SHARD
            pbar.update(TOKENS_PER_SHARD)

            parquet_files = sorted(tmp_dir.glob("*.parquet"))
            if len(parquet_files) >= upload_batch_dynamic:
                uploader.add_batch(parquet_files[:upload_batch_dynamic])

    # Upload remaining shards
    final_files = sorted(tmp_dir.glob("*.parquet"))
    if final_files:
        uploader.add_batch(final_files)

    uploader.stop()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    pbar.close()


# ----------------------------
# Main
# ----------------------------
def main():
    log(f"NCCL debug log → {_nccl_log_path}")
    log("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, token=HF_TOKEN)

    if WORLD_SIZE > 1:
        log(f"Initializing process group  world_size={WORLD_SIZE}  timeout={NCCL_TIMEOUT}")
        dist.init_process_group(
            backend="nccl",
            timeout=NCCL_TIMEOUT,   # prevents indefinite hangs that lead to SIGABRT
        )

        # Only rank 0 creates repos; all ranks wait at a named barrier afterwards
        # so no rank starts writing before the repo exists.
        try:
            if LOCAL_RANK == 0:
                for lang in TARGETS:
                    repo_id = f"{HF_USER}/{REPOS[lang]}"
                    log(f"Creating/verifying repo: {repo_id}")
                    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
            safe_barrier("repo_creation")
        except RuntimeError:
            log("Aborting — barrier failed during repo creation. "
                f"Inspect {_nccl_log_path} for the NCCL error.")
            sys.exit(1)

    for lang, target in TARGETS.items():
        repo_id = f"{HF_USER}/{REPOS[lang]}"
        log(f"Starting pipeline: lang={lang}  target={target:,}  repo={repo_id}")
        run_pipeline(lang, target, tokenizer, repo_id)

    # Final sync before exit so no rank tears down while others are still uploading
    if WORLD_SIZE > 1:
        safe_barrier("final_sync")
        dist.destroy_process_group()

    log("Dataset creation complete")


if __name__ == "__main__":
    main()
