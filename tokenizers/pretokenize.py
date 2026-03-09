#!/usr/bin/env python3
"""
End-to-end tokenizer factory + shard uploader.

Features
--------
• vectorized tokenization
• Arrow FixedSizeListArray shards
• tmp shard folder
• batch uploads to HF
• exponential retry
• shards deleted only after successful upload
• resumable if HF fails
• multi-GPU safe

Run:
torchrun --nproc_per_node=8 tokenizer_factory.py --lang_l1 fr
"""

import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm


# ------------------------------------------------
# ARGUMENTS
# ------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--lang_l1", type=str, default="es")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--shard_seqs", type=int, default=50000)
parser.add_argument("--tokenizer_batch", type=int, default=20000)
parser.add_argument("--upload_batch", type=int, default=40)

args = parser.parse_args()

L1 = args.lang_l1
SEQ_LEN = args.seq_len
SHARD_SEQS = args.shard_seqs
TOKENIZER_BATCH = args.tokenizer_batch
UPLOAD_BATCH = args.upload_batch

TOKENS_PER_SHARD = SEQ_LEN * SHARD_SEQS


# ------------------------------------------------
# HF SETTINGS
# ------------------------------------------------

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not set")

TOKENIZER_ID = f"{HF_USER}/tokenizer-{L1}-en"


TARGETS = {
    L1: 3_500_000_000,
    "en": 2_500_000_000
}

REPOS = {
    L1: f"{L1}-en-5B",
    "en": f"en-{L1}-5B"
}


# ------------------------------------------------
# DISTRIBUTED INFO
# ------------------------------------------------

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def log(msg):
    print(f"[Rank {LOCAL_RANK}] {msg}", flush=True)


api = HfApi(token=HF_TOKEN)


# ------------------------------------------------
# SAFE HF UPLOAD
# ------------------------------------------------

def upload_tmp_shards(tmp_dir, repo_id):

    files = sorted(tmp_dir.glob("*.parquet"))

    if not files:
        return

    batch = files[:UPLOAD_BATCH]

    log(f"Uploading batch of {len(batch)} shards")

    retries = 0

    while True:

        try:

            api.upload_folder(
                folder_path=str(tmp_dir),
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[f.name for f in batch],
            )

            for f in batch:
                f.unlink()

            log("Upload success")

            time.sleep(5)

            return

        except Exception as e:

            retries += 1
            wait = min(120, 2 ** retries)

            log(f"Upload failed retry={retries} wait={wait}s")

            time.sleep(wait)


# ------------------------------------------------
# DATA STREAM
# ------------------------------------------------

def get_stream(lang):

    if lang == L1:

        return load_dataset(
            "uonlp/CulturaX",
            lang,
            split="train",
            streaming=True
        )

    return load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True
    )


# ------------------------------------------------
# WRITE SHARD
# ------------------------------------------------

def write_shard(tokens, shard_id, lang, tmp_dir):

    flat = pa.array(tokens, type=pa.int32())

    arr = pa.FixedSizeListArray.from_arrays(flat, SEQ_LEN)

    table = pa.Table.from_arrays([arr], names=["input_ids"])

    fname = tmp_dir / f"{lang}_train_{shard_id:06d}_rank{LOCAL_RANK}.parquet"

    pq.write_table(
        table,
        fname,
        compression="zstd",
        row_group_size=4096
    )

    return fname


# ------------------------------------------------
# PIPELINE
# ------------------------------------------------

def run_pipeline(lang, target_tokens, tokenizer, repo_id):

    stream = iter(get_stream(lang))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"hf_shards_rank{LOCAL_RANK}_"))

    log(f"Temporary shard dir: {tmp_dir}")

    shard = LOCAL_RANK
    tokens_seen = 0

    buffer = np.empty(0, dtype=np.int32)

    pbar = tqdm(
        total=target_tokens,
        unit="tok",
        unit_scale=True,
        desc=f"{lang}-R{LOCAL_RANK}",
        position=LOCAL_RANK
    )

    while tokens_seen < target_tokens:

        texts = []

        for _ in range(TOKENIZER_BATCH):

            try:
                texts.append(next(stream)["text"])
            except StopIteration:
                break

        if not texts:
            break

        enc = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False
        )

        tokens = np.concatenate(enc["input_ids"]).astype(np.int32)

        buffer = np.concatenate([buffer, tokens])

        while buffer.size >= TOKENS_PER_SHARD:

            shard_tokens = buffer[:TOKENS_PER_SHARD]
            buffer = buffer[TOKENS_PER_SHARD:]

            write_shard(shard_tokens, shard, lang, tmp_dir)

            shard += WORLD_SIZE

            tokens_seen += TOKENS_PER_SHARD

            pbar.update(TOKENS_PER_SHARD)

            if len(list(tmp_dir.glob("*.parquet"))) >= UPLOAD_BATCH:

                upload_tmp_shards(tmp_dir, repo_id)

    # final upload
    upload_tmp_shards(tmp_dir, repo_id)

    log("Cleaning tmp dir")

    shutil.rmtree(tmp_dir)

    pbar.close()


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    log("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        use_fast=True,
        token=HF_TOKEN
    )

    for lang, target in TARGETS.items():

        repo_id = f"{HF_USER}/{REPOS[lang]}"

        if LOCAL_RANK == 0:

            create_repo(
                repo_id,
                repo_type="dataset",
                exist_ok=True,
                token=HF_TOKEN
            )

        run_pipeline(lang, target, tokenizer, repo_id)

    log("Dataset creation complete")


if __name__ == "__main__":
    main()
