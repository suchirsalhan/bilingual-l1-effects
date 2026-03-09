#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANG = "zh"
VOCAB_SIZE = 50_000
BOOTSTRAP_SENTENCES = 500_000  # you can reduce for testing
REPO_ID = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)
CORPUS_FILE = OUT_DIR / "cached_corpus.txt"

# =====================================================
# 1. CACHE DATASET TO FILE SAFELY
# =====================================================
def cache_training_corpus():
    if CORPUS_FILE.exists():
        print(f"📄 Cached corpus already exists at {CORPUS_FILE}")
        return

    print(f"💾 Caching {BOOTSTRAP_SENTENCES} sentences to {CORPUS_FILE}...")

    zh_ds = load_dataset("uonlp/CulturaX", "zh", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    zh_iter, en_iter = iter(zh_ds), iter(en_ds)

    written = 0
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for _ in range(BOOTSTRAP_SENTENCES // 2):
            try:
                zh_text = next(zh_iter).get("text", "")
                en_text = next(en_iter).get("text", "")
                if zh_text.strip():
                    f.write(zh_text.replace("\n", " ") + "\n")
                    written += 1
                if en_text.strip():
                    f.write(en_text.replace("\n", " ") + "\n")
                    written += 1
            except StopIteration:
                break

    print(f"✅ Corpus caching complete. {written} sentences written.")

# =====================================================
# 2. TRAIN TOKENIZER
# =====================================================
def train_and_push():
    from tokenizers import ByteLevelBPETokenizer
    from huggingface_hub import create_repo

    cache_training_corpus()

    print(f"🚀 Training Byte-Level BPE tokenizer (vocab size={VOCAB_SIZE})...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(CORPUS_FILE)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )

    hf_tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Tokenizer saved to {OUT_DIR}")

    # Ensure repo exists
    if HF_TOKEN:
        create_repo(REPO_ID, exist_ok=True, token=HF_TOKEN)
        hf_tokenizer.push_to_hub(REPO_ID, use_auth_token=HF_TOKEN)
        print(f"✅ Tokenizer pushed to HuggingFace Hub: {REPO_ID}")
# =====================================================
# 3. BENCHMARK
# =====================================================
def run_benchmark():
    from transformers import AutoTokenizer

    print(f"\n🧪 Starting Benchmark...")
    my_tk = AutoTokenizer.from_pretrained(OUT_DIR)

    test_cases = {
        "ZH_Legal": "你好，我饿了。",
        "Emoji": "Learning is fun! 🚀🔥"
    }

    for name, text in test_cases.items():
        encoded = my_tk.encode(text)
        decoded = my_tk.decode(encoded)
        print(f"[{name}] Tokens: {len(encoded)} | Decoded: '{decoded}'")
        # Integrity check
        assert decoded.strip() == text.strip(), f"Integrity failed for {name}"

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train_and_push()
    run_benchmark()
