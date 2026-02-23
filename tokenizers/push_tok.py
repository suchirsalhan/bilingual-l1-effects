#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, upload_folder

HF_USERNAME = "RA-ALTA"   # <-- your HF account
BASE_DIR = "."             # run inside tokenizers/

api = HfApi()

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


if __name__ == "__main__":
    main()
