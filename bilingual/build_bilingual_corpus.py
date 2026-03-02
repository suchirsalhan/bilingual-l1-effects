#!/usr/bin/env python3
import os
import json
import random
import argparse
import multiprocessing as mp
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer
from huggingface_hub import HfApi, create_repo

# ======================================================
# GLOBAL CONFIG
# ======================================================

HF_USER = "RA-ALTA"

SEQ_LEN = 512
VOCAB_SIZE = 50_000
MAX_SENTENCES = 500_000

SHARD_SIZE = 200_000
NUM_WORKERS = max(8, os.cpu_count() // 2)

SEED = 1337

TOKENIZER_DIR = Path("tokenizers")
CACHE_DIR = Path("factory_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ======================================================
# STREAMS
# ======================================================

def culturax_stream(lang):
    ds = load_dataset(
        "uonlp/CulturaX",
        lang,
        split="train",
        streaming=True,
    )
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")

def fineweb_stream():
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")

# ======================================================
# LANGUAGE BALANCER
# ======================================================

def balanced_stream(lang):

    l1 = culturax_stream(lang)
    en = fineweb_stream()

    while True:
        yield next(l1)
        yield next(en)

# ======================================================
# TOKENIZER
# ======================================================

def train_tokenizer(lang):

    out_dir = TOKENIZER_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = str(out_dir / "spm")

    print(f"[Tokenizer] Training {lang}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=balanced_stream(lang),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        input_sentence_size=MAX_SENTENCES,
        shuffle_input_sentence=True,
        byte_fallback=True,
        num_threads=os.cpu_count(),
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
    )

    tokenizer = LlamaTokenizer(
        vocab_file=str(out_dir / "spm.model"),
        legacy=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)

    repo = f"{HF_USER}/tokenizer-{lang}"
    create_repo(repo, exist_ok=True)

    HfApi().upload_folder(
        folder_path=str(out_dir),
        repo_id=repo,
        repo_type="model",
    )

    return tokenizer

# ======================================================
# TOKEN PACKER
# ======================================================

def pack_blocks(stream, tokenizer):

    buf = []

    for text in stream:

        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)

        while len(buf) >= SEQ_LEN:
            yield buf[:SEQ_LEN]
            buf = buf[SEQ_LEN:]

# ======================================================
# RESUMABLE STATE
# ======================================================

def state_path(lang):
    return CACHE_DIR / f"{lang}_state.json"

def load_state(lang):
    p = state_path(lang)
    if p.exists():
        return json.load(open(p))
    return {"shard": 0}

def save_state(lang, state):
    json.dump(state, open(state_path(lang), "w"))

# ======================================================
# WORKER
# ======================================================

def worker(lang):

    random.seed(SEED)

    print(f"[Worker] Starting {lang}")

    tokenizer = LlamaTokenizer.from_pretrained(
        f"{HF_USER}/tokenizer-{lang}"
    )

    repo = f"{HF_USER}/bilingual-{lang}-512"
    create_repo(repo, repo_type="dataset", exist_ok=True)

    state = load_state(lang)

    stream = balanced_stream(lang)
    blocks = pack_blocks(stream, tokenizer)

    shard = []
    shard_id = state["shard"]

    for block in blocks:

        shard.append({"input_ids": block})

        if len(shard) >= SHARD_SIZE:

            print(f"{lang} → shard {shard_id}")

            ds = Dataset.from_list(shard)

            # =================================
            # DETERMINISTIC GLOBAL SHUFFLE
            # =================================
            ds = ds.shuffle(seed=SEED + shard_id)

            ds.push_to_hub(
                repo,
                split=f"train_{shard_id}",
            )

            shard.clear()

            shard_id += 1
            state["shard"] = shard_id
            save_state(lang, state)

# ======================================================
# PARALLEL BUILDER
# ======================================================

def build_languages(langs):

    print(f"\nLaunching {len(langs)} builders")

    with mp.Pool(len(langs)) as pool:
        pool.map(worker, langs)

# ======================================================
# MAIN
# ======================================================

def main(langs):

    random.seed(SEED)

    # ---------- Train tokenizers ----------
    for lang in langs:
        train_tokenizer(lang)

    # ---------- Parallel dataset builders ----------
    build_languages(langs)

# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs",
        nargs="+",
        required=True,
    )

    args = parser.parse_args()

    main(args.langs)
