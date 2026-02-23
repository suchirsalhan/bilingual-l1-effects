#!/usr/bin/env python3
import os
import argparse
import sentencepiece as spm
from datasets import load_dataset
import requests, gzip, json

# ---------------- Config ----------------
CULTURAX_LANGS = ["es", "fr", "de", "pl", "tr", "ar", "zh"]
TOKENIZER_ROOT = "./tokenizers"
VOCAB_SIZE = 50000
# Total sentences to sample (split between L1 and English)
MAX_SENTENCES = 500_000 

# ---------- CulturaX (L1 Stream) ----------
def culturax_stream(lang):
    """Streams the target L1 language from CulturaX."""
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")

# ---------- FineWeb-Edu (English Stream A) ----------
def fineweb_stream():
    """Streams high-quality English educational data."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n", " ")

# ---------- RedPajama (English/General Stream B) ----------
def redpajama_stream(lang):
    """Streams data from RedPajama V2 based on language listings."""
    BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
    try:
        listing = requests.get(f"{BASE}/listings/{lang}-2023-06-head_middle.txt").text.splitlines()
    except Exception as e:
        print(f"Error fetching RedPajama listing: {e}")
        return

    session = requests.Session()
    for shard in listing:
        try:
            r = session.get(f"{BASE}/documents/{shard}.json.gz", stream=True, timeout=60)
            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc.get("raw_content")
                    if text:
                        yield text.replace("\n", " ")
        except Exception:
            continue

# ---------- Balanced Bilingual Iterator ----------
def bilingual_iterator(lang, source, total_max):
    """
    Ensures a 50/50 split between the L1 language and the English source.
    Without this manual split, one stream might dominate the entire sample.
    """
    per_lang_cap = total_max // 2
    
    # 1. Pull L1 Data
    print(f"Sampling {per_lang_cap} sentences from CulturaX ({lang})...")
    l1_count = 0
    for text in culturax_stream(lang):
        yield text
        l1_count += 1
        if l1_count >= per_lang_cap:
            break
            
    # 2. Pull English Data
    print(f"Sampling {per_lang_cap} sentences from English source ({source})...")
    en_count = 0
    en_stream = fineweb_stream() if source == "fineweb" else redpajama_stream("en")
    for text in en_stream:
        yield text
        en_count += 1
        if en_count >= per_lang_cap:
            break
    
    print(f"Iterator finished. Total collected: {l1_count + en_count}")

# ---------- Train Tokenizer ----------
def build_tokenizer(lang, source):
    out_dir = os.path.join(TOKENIZER_ROOT, f"{source}_{lang}")
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "spm")

    print(f"--- Starting Training: {source}_{lang} ---")

    spm.SentencePieceTrainer.train(
        sentence_iterator=bilingual_iterator(lang, source, MAX_SENTENCES),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        # input_sentence_size tells SPM how many sentences to keep in RAM
        input_sentence_size=MAX_SENTENCES,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
        # Standard ID mapping
        bos_id=0, 
        eos_id=1, 
        pad_id=2, 
        unk_id=3,
    )
    print(f"Successfully saved tokenizer to {out_dir}")

# ---------- Main Execution ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a bilingual SentencePiece tokenizer.")
    parser.add_argument("--lang", required=True, choices=CULTURAX_LANGS, help="Target L1 language code.")
    parser.add_argument("--source", required=True, choices=["fineweb", "redpajama"], help="English data source.")
    args = parser.parse_args()

    build_tokenizer(args.lang, args.source)
