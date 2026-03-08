#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from transformers import PreTrainedTokenizerFast, AutoTokenizer, pipeline

# =====================================================
# CONFIG - Production Specs
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANG = "es"
VOCAB_SIZE = 50_000
BOOTSTRAP_SENTENCES = 2_000_000  # High enough for subword quality
REPO_ID = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. ROBUST DATA GENERATOR
# =====================================================
def get_training_corpus():
    es_ds = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    es_iter, en_iter = iter(es_ds), iter(en_ds)
    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(es_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration: break

# =====================================================
# 2. TRAINING LOGIC
# =====================================================
def train_and_push():
    print(f"🚀 Training BPE (50k Vocab)...")
    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    
    # Normalization & Pre-tokenization
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(" {2,}", " ")])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=10, 
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(OUT_DIR / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )
    hf_tokenizer.save_pretrained(OUT_DIR)
    
    if HF_TOKEN:
        hf_tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"✅ Tokenizer pushed to {REPO_ID}")

# =====================================================
# 3. LOADING & GENERATION TEST
# =====================================================
def run_benchmark():
    print(f"\n🧪 Starting Loading & Generation Benchmark...")
    
    # 1. Test Loading from Hub/Local
    try:
        my_tk = AutoTokenizer.from_pretrained(OUT_DIR)
        llama_tk = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Load failed: {e}"); return

    # 2. Test Sentences (Mixed Domain)
    test_cases = {
        "ES_Legal": "El contrato se rige por las leyes de la jurisdicción local.",
        "EN_Tech": "The neural network architecture utilizes a transformer backbone.",
        "Mixed": "Bilingual models are essential for traducción automática."
    }

    print(f"{'Test Case':<15} | {'Custom Tokens':<15} | {'Llama-3 Tokens':<15} | {'Ratio'}")
    print("-" * 65)

    for name, text in test_cases.items():
        my_len = len(my_tk.encode(text))
        llama_len = len(llama_tk.encode(text))
        ratio = my_len / llama_len
        print(f"{name:<15} | {my_len:<15} | {llama_len:<15} | {ratio:.2f}x")

    # 3. Mock Generation Test (Structural Verification)
    print("\n📝 Mock Generation Verification:")
    input_text = "La ciencia de datos es"
    encoded = my_tk(input_text, return_tensors="pt")
    
    print(f"Input: '{input_text}'")
    print(f"Token IDs: {encoded['input_ids'][0].tolist()}")
    print(f"Reconstructed: '{my_tk.decode(encoded['input_ids'][0])}'")
    
    # Check for empty tokens or decoding artifacts
    if my_tk.decode(encoded['input_ids'][0]).strip() == input_text.strip():
        print("💎 INTEGRITY CHECK PASSED: Round-trip decoding is 100% accurate.")

if __name__ == "__main__":
    train_and_push()
    run_benchmark()
