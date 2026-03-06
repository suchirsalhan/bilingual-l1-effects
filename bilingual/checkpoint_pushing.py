#!/usr/bin/env python3
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ["HF_TOKEN"]

# =====================================================
# CONFIG
# =====================================================
CHECKPOINT_DIR = Path("checkpoints")  # your local checkpoints folder
LANGS = ["es", "fr", "de", "pl", "tr", "ar", "zh", "en"]  # L1 languages
MODEL_PREFIX = f"{HF_USER}/"  # push target model repos

VOCAB_SIZE = 50_000  
MODEL_CONFIG_DEFAULTS = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "vocab_size": VOCAB_SIZE,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "unk_token_id": 3
}

api = HfApi(token=HF_TOKEN)

# =====================================================
# CHECKPOINT PUSH
# =====================================================
def push_checkpoint(l1_lang: str, checkpoint_path: Path):
    """Push a single checkpoint to HF with proper model files and config.json"""
    # Compute HF repo name
    ckpt_name = checkpoint_path.name.replace("-", "_")
    repo_name = f"{MODEL_PREFIX}{l1_lang}-english-{ckpt_name.split('_')[-1]}"

    # Ensure repo exists
    create_repo(repo_name, repo_type="model", exist_ok=True, token=HF_TOKEN)

    # Ensure model file is named pytorch_model.bin
    model_file = checkpoint_path / "model.pt"
    if model_file.exists():
        target_file = checkpoint_path / "pytorch_model.bin"
        model_file.rename(target_file)
    elif not (checkpoint_path / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Neither model.pt nor pytorch_model.bin exists in {checkpoint_path}")

    # Write config.json if missing
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(MODEL_CONFIG_DEFAULTS, indent=2))

    # Push checkpoint folder
    upload_folder(folder_path=str(checkpoint_path), repo_id=repo_name, repo_type="model", token=HF_TOKEN)
    print(f"🎉 Checkpoint {checkpoint_path.name} pushed to {repo_name}")

# =====================================================
# MAIN
# =====================================================
def main():
    if not CHECKPOINT_DIR.exists():
        print(f"❌ Directory {CHECKPOINT_DIR} not found.")
        return

    for l1_lang in LANGS:
        # Push all checkpoints matching the language pattern
        found_ckpts = list(CHECKPOINT_DIR.glob(f"{l1_lang}-en-*"))
        
        if not found_ckpts:
            print(f"ℹ️ No checkpoints found for {l1_lang}")
            continue

        for ckpt in found_ckpts:
            if ckpt.is_dir():
                print(f"➡️ Pushing checkpoint {ckpt.name}")
                push_checkpoint(l1_lang, ckpt)

    print("✅ All checkpoints pushed successfully!")

if __name__ == "__main__":
    main()
