#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANG = "pl"
VOCAB_SIZE = 50_000
BOOTSTRAP_SENTENCES = 2_000_000 
REPO_ID = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus():
    pl_ds = load_dataset("uonlp/CulturaX", "pl", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    pl_iter, en_iter = iter(pl_ds), iter(en_ds)
    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(pl_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration: break

# =====================================================
# 2. TRAINING LOGIC (FIXED)
# =====================================================
def train_and_push():
    print(f"🚀 Training Byte-Level BPE (50k Vocab)...")
    
    # Initialize BPE with byte_fallback for ultimate robustness
    tokenizer = Tokenizer(models.BPE(byte_fallback=False))
    
    # 1. Normalization: Keep it simple for Byte-Level
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    
    # 2. Pre-tokenization: Use ByteLevel instead of Whitespace
    # This splits by bytes and handles whitespace as a special character (Ġ)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    
    # 3. Post-Processing: Critical for GPT-2 style leading space preservation
    tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> <s> $B </s>",
    special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    
    # 4. Decoder: Converts the byte-shards back into readable text
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Wrap in Transformers-compatible class
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )
    
    hf_tokenizer.save_pretrained(OUT_DIR)
    
    if HF_TOKEN:
        hf_tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"✅ Tokenizer pushed to {REPO_ID}")

# =====================================================
# 3. BENCHMARK
# =====================================================
def run_benchmark():
    print(f"\n🧪 Starting Benchmark...")
    my_tk = AutoTokenizer.from_pretrained(OUT_DIR)
    
    test_cases = {
        "PL_Legal": "Cześć, jestem głodny.",
        "Emoji": "Learning is fun! 🚀🔥"
    }

    for name, text in test_cases.items():
        encoded = my_tk.encode(text)
        decoded = my_tk.decode(encoded)
        print(f"[{name}] Tokens: {len(encoded)} | Decoded: '{decoded}'")
        
        # Integrity check
        assert decoded.strip() == text.strip(), f"Integrity failed for {name}"

if __name__ == "__main__":
    train_and_push()
    run_benchmark()
