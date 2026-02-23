#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, snapshot_download, upload_folder
from transformers import T5Tokenizer, AutoTokenizer
import os
import shutil
import tempfile

HF_USERNAME = "RA-ALTA"   # <-- your HF account
BASE_DIR = "."             # run inside tokenizers/
api = HfApi()
ORG = "RA-ALTA"

def valid_tokenizer_folder(path):
    files = os.listdir(path)
    return "spm.model" in files and "spm.vocab" in files


def main():
    subfolders = sorted(
        f for f in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, f))
    )

    print(f"Found {len(subfolders)} tokenizer folders\n")

    for folder in subfolders:
        folder_path = os.path.join(BASE_DIR, folder)

        if not valid_tokenizer_folder(folder_path):
            print(f"⚠️  Skipping {folder} (missing tokenizer files)")
            continue

        repo_id = f"{HF_USERNAME}/tokenizer-{folder}"

        print(f"\n🚀 Creating / pushing → {repo_id}")

        # Create repo if needed
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )

        # Upload tokenizer
        upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            repo_type="model",
            commit_message=f"Add tokenizer {folder}",
        )

        print(f"✅ Uploaded {folder}")

    print("\n🎉 All tokenizers processed.")



# --------------------------------------------------
# Find tokenizer repos
# --------------------------------------------------
models = api.list_models(author=ORG)

tokenizer_repos = [
    m.modelId for m in models
    if "tokenizer-" in m.modelId
]

print("Found tokenizers:")
for r in tokenizer_repos:
    print(" -", r)

# --------------------------------------------------
# Process each tokenizer
# --------------------------------------------------
for repo_id in tokenizer_repos:

    print(f"\n=== Processing {repo_id} ===")

    with tempfile.TemporaryDirectory() as tmp:

        repo_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=os.path.join(tmp, "repo"),
        )

        spm_path = os.path.join(repo_dir, "spm.model")

        if not os.path.exists(spm_path):
            print("No spm.model — skipping")
            continue

        # --------------------------------------------------
        # 1️⃣ Load as SentencePiece tokenizer (slow)
        # --------------------------------------------------
        slow_tok = T5Tokenizer(
            vocab_file=spm_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        slow_dir = os.path.join(tmp, "slow")
        slow_tok.save_pretrained(slow_dir)

        # --------------------------------------------------
        # 2️⃣ Convert automatically → FAST tokenizer
        # --------------------------------------------------
        fast_tok = AutoTokenizer.from_pretrained(
            slow_dir,
            use_fast=True,
        )

        out_dir = os.path.join(tmp, "hf_tokenizer")
        fast_tok.save_pretrained(out_dir)

        # keep original files
        shutil.copy(spm_path, out_dir)
        vocab = os.path.join(repo_dir, "spm.vocab")
        if os.path.exists(vocab):
            shutil.copy(vocab, out_dir)

        # --------------------------------------------------
        # Upload fixed tokenizer
        # --------------------------------------------------
        upload_folder(
            repo_id=repo_id,
            folder_path=out_dir,
            commit_message="Register tokenizer as HF SentencePiece tokenizer",
        )

        print("✅ Uploaded:", repo_id)

print("\n🎉 ALL TOKENIZERS REGISTERED")

if __name__ == "__main__":
    main()
