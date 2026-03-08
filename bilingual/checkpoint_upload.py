#!/usr/bin/env python3
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Update this to the absolute or relative path where your folders live
CHECKPOINT_DIR = Path("./checkpoints")  
LANGS = ["tr"] 

MODEL_CONFIG_DEFAULTS = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_embd": 1024,      
    "n_layer": 24,      
    "n_head": 16,       
    "n_positions": 512, 
    "vocab_size": 50000,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
}

# =====================================================
# CHECKPOINT PUSH
# =====================================================
def push_checkpoint(l1_lang: str, checkpoint_path: Path):
    ckpt_suffix = checkpoint_path.name.split("-")[-1]
    repo_name = f"{HF_USER}/{l1_lang}-en-{ckpt_suffix}"

    # Define the possible weight files
    safetensors_file = checkpoint_path / "model.safetensors"
    pytorch_file = checkpoint_path / "model.pt"
    bin_file = checkpoint_path / "pytorch_model.bin"

    # Check if ANY weight file exists
    if not (safetensors_file.exists() or pytorch_file.exists() or bin_file.exists()):
        print(f"⚠️ Skipping {checkpoint_path.name}: No weight files found.")
        return

    print(f"🚀 Preparing to push {checkpoint_path.name} to {repo_name}...")
    create_repo(repo_id=repo_name, repo_type="model", exist_ok=True, token=HF_TOKEN)

    # If it's model.pt, rename it to pytorch_model.bin for HF compatibility
    if pytorch_file.exists() and not bin_file.exists():
        print(f"📦 Renaming model.pt to pytorch_model.bin...")
        pytorch_file.replace(bin_file)

    # Update config.json
    config_path = checkpoint_path / "config.json"
    config_path.write_text(json.dumps(MODEL_CONFIG_DEFAULTS, indent=2))

    # Push the whole folder (will include model.safetensors if it's there)
    upload_folder(
        folder_path=str(checkpoint_path), 
        repo_id=repo_name, 
        repo_type="model", 
        token=HF_TOKEN,
        ignore_patterns=["optimizer.pt"] # Usually don't want to push heavy optimizer states
    )
    print(f"🎉 Success: {repo_name} is updated.")

def main():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable not set.")
        return

    if not CHECKPOINT_DIR.exists():
        print(f"❌ Directory {CHECKPOINT_DIR} not found.")
        return

    for l1_lang in LANGS:
        found_ckpts = list(CHECKPOINT_DIR.glob(f"{l1_lang}-english-*"))
        for ckpt in sorted(found_ckpts):
            if ckpt.is_dir():
                push_checkpoint(l1_lang, ckpt)

if __name__ == "__main__":
    main()
