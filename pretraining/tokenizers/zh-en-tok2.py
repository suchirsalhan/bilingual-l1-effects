#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import create_repo

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANG = "zh"
VOCAB_SIZE = 50_000
BOOTSTRAP_SENTENCES = 500_000  # adjust for testing or full corpus
REPO_ID = f"{HF_USER}/tokenizer-{LANG}-en"
OUT_DIR = Path("tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. DATA GENERATOR
# =====================================================
def get_training_corpus():
    zh_ds = load_dataset("uonlp/CulturaX", "zh", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    zh_iter, en_iter = iter(zh_ds), iter(en_ds)
    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(zh_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# 2. TRAIN TOKENIZER & PUSH TO HUB
# =====================================================
def train_and_push():
    print(f"🚀 Training Byte-Level BPE tokenizer (vocab size={VOCAB_SIZE})...")

    # 1. Initialize Byte-Level BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # 2. Normalizer: NFC (unicode normalization)
    tokenizer.normalizer = normalizers.NFC()
    
    # 3. Pre-tokenizer: ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 4. Post-processor: GPT-2 style
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # 5. Decoder: ByteLevel
    tokenizer.decoder = decoders.ByteLevel()

    # 6. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    # 7. Train
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 8. Wrap in Transformers PreTrainedTokenizerFast
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

    # 9. Ensure repo exists & push
    if HF_TOKEN:
        create_repo(REPO_ID, exist_ok=True, token=HF_TOKEN)
        hf_tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"✅ Tokenizer pushed to Hugging Face Hub: {REPO_ID}")

# =====================================================
# 3. BENCHMARK
# =====================================================
def run_benchmark():
    print("\n🧪 Starting Benchmark...")
    my_tk = AutoTokenizer.from_pretrained(OUT_DIR)

    test_cases = {
        "ZH_Legal": "你好，我饿了。",
        "Emoji": "Learning is fun! 🚀🔥",
        "Mixed": "Hello 你好!"
    }

    for name, text in test_cases.items():
        encoded = my_tk.encode(text)
        decoded = my_tk.decode(encoded)
        print(f"[{name}] Tokens: {len(encoded)} | Decoded: '{decoded}'")
        assert decoded.strip() == text.strip(), f"Integrity failed for {name}"

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train_and_push()
    run_benchmark()
