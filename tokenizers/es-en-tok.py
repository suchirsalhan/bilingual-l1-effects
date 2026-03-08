#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

# =====================================================
# CONFIG
# =====================================================
LANG = "es"
VOCAB_SIZE = 50_000
# 1M sentences (500k each) is the 'sweet spot' for bilingual parity
BOOTSTRAP_SENTENCES = 1_000_000 
OUT_DIR = Path(f"balanced-tokenizer-es-en")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 50:50 PARITY GENERATOR
# =====================================================
def get_training_corpus():
    # Stream Spanish (CulturaX) and English (FineWeb-Edu)
    es_ds = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    es_iter = iter(es_ds)
    en_iter = iter(en_ds)
    
    print(f"📊 Sampling {BOOTSTRAP_SENTENCES} interleaved sentences...")
    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            # Yield 1 Spanish, then 1 English
            yield next(es_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# TRAINING
# =====================================================
def train_balanced_tokenizer():
    # Initialize Byte-Level BPE (Gold standard for LLMs)
    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    
    # Pre-tokenizer handles whitespace and byte-level splitting
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2, # Filter out noise
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"🚀 Training Balanced BPE...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Save Native JSON
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    
    # Wrap for easy use with 'AutoTokenizer.from_pretrained()'
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(OUT_DIR / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )
    hf_tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Success! Balanced tokenizer saved to {OUT_DIR}")

if __name__ == "__main__":
    train_balanced_tokenizer()
