#!/usr/bin/env python3
import os
import argparse
import sentencepiece as spm
from datasets import load_dataset
import requests, gzip, json
import itertools

# ---------------- Config ----------------
LANGS = ["es","fr","de","pl","tr","ar","zh"]
TOKENIZER_ROOT = "./tokenizers"
VOCAB_SIZE = 50000
MAX_SENTENCES = 500_000  # ~30 min cap per tokenizer

# ---------- CulturaX ----------
def culturax_stream(lang):
    ds = load_dataset("uonlp/CulturaX", lang,
                      split="train",
                      streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

# ---------- FineWeb ----------
def fineweb_stream():
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True
    )
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

# ---------- RedPajama ----------
def redpajama_stream(lang):
    BASE="https://data.together.xyz/redpajama-data-v2/v1.0.0"
    listing = requests.get(
        f"{BASE}/listings/{lang}-2023-06-head_middle.txt"
    ).text.splitlines()
    session = requests.Session()
    for shard in listing:
        try:
            r=session.get(
                f"{BASE}/documents/{shard}.json.gz",
                stream=True,
                timeout=60
            )
            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc=json.loads(line)
                    text=doc.get("raw_content")
                    if text:
                        yield text
        except Exception:
            continue

# ---------- Combined iterator with sentence cap ----------
def bilingual_iterator(lang, source):
    """Yield sentences, stop after MAX_SENTENCES"""
    count = 0
    # CulturaX
    for s in culturax_stream(lang):
        yield s
        count += 1
        if count >= MAX_SENTENCES:
            return

    # FineWeb or RedPajama
    if source == "fineweb":
        stream = fineweb_stream()
    else:
        stream = redpajama_stream(lang)

    for s in stream:
        yield s
        count += 1
        if count >= MAX_SENTENCES:
            return

# ---------- Train tokenizer ----------
def build_tokenizer(lang, source):
    out_dir=f"{TOKENIZER_ROOT}/{source}_{lang}"
    os.makedirs(out_dir,exist_ok=True)

    print(f"Training tokenizer: {source}-{lang} (~{MAX_SENTENCES} sentences)")

    spm.SentencePieceTrainer.train(
        sentence_iterator=bilingual_iterator(lang,source),
        model_prefix=f"{out_dir}/spm",
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

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--source", choices=["fineweb","redpajama"], required=True)
    args=parser.parse_args()

    build_tokenizer(args.lang, args.source)
