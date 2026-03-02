#!/usr/bin/env python3
import os
import argparse
import random
import json
import gzip
import requests

import sentencepiece as spm
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer
from huggingface_hub import HfApi, create_repo

# ============================================================
# CONFIG
# ============================================================

CULTURAX_LANGS = ["es", "fr", "de", "pl", "tr", "ar", "zh"]

HF_USER = "RA-ALTA"

TOKENIZER_ROOT = "./tokenizers"
VOCAB_SIZE = 50000
MAX_SENTENCES = 500_000

SEQ_LEN = 512
SHARD_SIZE = 100_000
SEED = 42

# ============================================================
# DATA STREAMS
# ============================================================

def culturax_stream(lang):
    ds = load_dataset("uonlp/CulturaX", lang,
                      split="train",
                      streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")


def fineweb_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
                      split="train",
                      streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")


def redpajama_stream(lang):
    BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"

    listing = requests.get(
        f"{BASE}/listings/{lang}-2023-06-head_middle.txt"
    ).text.splitlines()

    session = requests.Session()

    for shard in listing:
        try:
            r = session.get(
                f"{BASE}/documents/{shard}.json.gz",
                stream=True,
                timeout=60,
            )

            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc = json.loads(line)
                    txt = doc.get("raw_content")
                    if txt:
                        yield txt.replace("\n", " ")
        except Exception:
            continue


# ============================================================
# BALANCED ITERATOR
# ============================================================

def bilingual_iterator(lang, source):

    l1 = culturax_stream(lang)
    en = fineweb_stream() if source == "fineweb" else redpajama_stream("en")

    while True:
        yield next(l1)
        yield next(en)


# ============================================================
# TOKENIZER TRAINING
# ============================================================

def train_tokenizer(lang, source):

    out_dir = f"{TOKENIZER_ROOT}/{source}_{lang}"
    os.makedirs(out_dir, exist_ok=True)

    model_prefix = f"{out_dir}/spm"

    print(f"\n=== Training tokenizer {source}_{lang} ===")

    spm.SentencePieceTrainer.train(
        sentence_iterator=bilingual_iterator(lang, source),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        input_sentence_size=MAX_SENTENCES,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
    )

    return out_dir


# ============================================================
# PUSH TOKENIZER TO HF
# ============================================================

def push_tokenizer(tokenizer_dir, lang, source):

    repo_name = f"tokenizer-{source}-{lang}"
    repo_id = f"{HF_USER}/{repo_name}"

    print(f"Pushing tokenizer → {repo_id}")

    create_repo(repo_id, exist_ok=True)

    tokenizer = LlamaTokenizer(
        vocab_file=os.path.join(tokenizer_dir, "spm.model"),
        legacy=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(tokenizer_dir)

    api = HfApi()
    api.upload_folder(
        folder_path=tokenizer_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    return tokenizer


# ============================================================
# PRETOKENIZATION
# ============================================================

def token_blocks(stream, tokenizer):

    buffer = []

    for text in stream:

        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)

        while len(buffer) >= SEQ_LEN:
            yield {"input_ids": buffer[:SEQ_LEN]}
            buffer = buffer[SEQ_LEN:]


# ============================================================
# DATASET BUILDER
# ============================================================

def build_dataset(lang, source, tokenizer):

    repo_id = f"{HF_USER}/bilingual-{lang}-512"

    print(f"\n=== Building dataset {repo_id} ===")

    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    stream = bilingual_iterator(lang, source)
    blocks = token_blocks(stream, tokenizer)

    shard = []
    shard_id = 0

    for example in blocks:

        shard.append(example)

        if len(shard) >= SHARD_SIZE:

            print(f"Uploading shard {shard_id}")

            ds = Dataset.from_list(shard)

            ds = ds.shuffle(seed=SEED)

            ds.push_to_hub(
                repo_id,
                split=f"train_{shard_id}",
            )

            shard = []
            shard_id += 1


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(lang, source):

    # 1️⃣ Train tokenizer
    tok_dir = train_tokenizer(lang, source)

    # 2️⃣ Push tokenizer
    tokenizer = push_tokenizer(tok_dir, lang, source)

    # 3️⃣ Pretokenize + shuffle + upload dataset
    build_dataset(lang, source, tokenizer)


# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True,
                        choices=CULTURAX_LANGS)
    parser.add_argument("--source", required=True,
                        choices=["fineweb", "redpajama"])

    args = parser.parse_args()

    main(args.lang, args.source)
