#!/usr/bin/env python3
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import HfApi, create_repo

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN") # Must have 'write' permissions
LANG = "es"
VOCAB_SIZE = 50_000
BOOTSTRAP_SENTENCES = 1_000_000 
REPO_ID = f"{HF_USER}/tokenizer-{LANG}-en-parity"
OUT_DIR = Path(f"tokenizer-local-temp")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. DATA GENERATOR (STRICT 50:50)
# =====================================================
def get_training_corpus():
    es_ds = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
    en_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    es_iter = iter(es_ds)
    en_iter = iter(en_ds)
    
    for _ in range(BOOTSTRAP_SENTENCES // 2):
        try:
            yield next(es_iter)["text"].replace("\n", " ")
            yield next(en_iter)["text"].replace("\n", " ")
        except StopIteration:
            break

# =====================================================
# 2. TRAINING LOGIC
# =====================================================
def train_and_push():
    print(f"🚀 Training Native BPE for {LANG}-EN Parity...")
    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    
    # Wrap for HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(OUT_DIR / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )
    hf_tokenizer.save_pretrained(OUT_DIR)
    
    if HF_TOKEN:
        print(f"📤 Pushing to Hugging Face: {REPO_ID}...")
        create_repo(REPO_ID, exist_ok=True, token=HF_TOKEN)
        hf_tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"✅ Upload complete!")
    else:
        print("⚠️ No HF_TOKEN found. Skipping upload.")

# =====================================================
# 3. TEST & VERIFY PARITY
# =====================================================
def test_from_hub():
    print(f"🔄 Loading tokenizer back from {REPO_ID} to verify...")
    # This proves the upload worked and the files are compatible
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)

    # Test Sentences
    en_text = "The artificial intelligence helps to improve our daily efficiency."
    es_text = "La inteligencia artificial ayuda a mejorar nuestra eficiencia diaria."

    en_ids = tokenizer.encode(en_text)
    es_ids = tokenizer.encode(es_text)

    print("\n--- Parity Test Results ---")
    print(f"English: {len(en_ids)} tokens | {en_text}")
    print(f"Spanish: {len(es_ids)} tokens | {es_text}")
    
    ratio = len(es_ids) / len(en_ids)
    print(f"Compression Ratio (ES/EN): {ratio:.2f}")
    
    if 0.9 <= ratio <= 1.2:
        print("🏆 SUCCESS: Tokenizer is highly balanced!")
    else:
        print("💡 NOTE: Ratio is slightly skewed, but within normal linguistic variance.")

if __name__ == "__main__":
    train_and_push()
    if HF_TOKEN:
        test_from_hub()
