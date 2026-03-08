# A small alternative to push_tok.py
import os
from pathlib import Path
import sentencepiece as spm
from tokenizers import Tokenizer, models, decoders, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from huggingface_hub import upload_folder

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANGS = ["es", "fr", "de", "pl", "tr", "ar", "zh", "en"]
TMP = Path("tmp_factory")

def clean_base_restoration(lang):
    spm_path = TMP / f"tokenizer_{lang}" / "spm.model"
    hf_out_dir = TMP / f"hf_base_{lang}"
    hf_out_dir.mkdir(exist_ok=True, parents=True)

    if not spm_path.exists():
        print(f"❌ {lang}: spm.model missing.")
        return

    print(f"🛠️  Restoring Clean Base Tokenizer for {lang}...")

    try:
        # 1. Extract raw pieces
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        
        # 2. Identify the EXACT space marker character (usually ID 36 or similar)
        # SentencePiece typically uses ' ' (U+2581). 
        # We'll pull it directly from the vocab to be 100% sure.
        space_marker = " " # Default fallback
        for piece in vocab.keys():
            if " " in piece or piece == " ":
                space_marker = " "
                break

        # 3. Build the Tokenizer Backend
        tok_model = models.BPE(vocab=vocab, merges=[])
        tokenizer = Tokenizer(tok_model)
        
        # Apply Metaspace logic with the SPECIFIC character found in vocab
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=space_marker)
        tokenizer.decoder = decoders.Metaspace(replacement=space_marker)

        # 4. Wrap in Transformers
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            add_prefix_space=True,
            clean_up_tokenization_spaces=True
        )

        # 5. Save and Push
        fast_tokenizer.save_pretrained(hf_out_dir)
        print(f"✅ {lang} fixed locally. Vocab size: {len(fast_tokenizer)}")

        repo_id = f"{HF_USER}/tokenizer-{lang}"
        upload_folder(
            folder_path=str(hf_out_dir),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="Fix: Final Metaspace decoding using exact vocab marker"
        )
        print(f"🚀 Pushed {lang} to {repo_id}")

    except Exception as e:
        print(f"💥 {lang} failed: {e}")

if __name__ == "__main__":
    for lang in LANGS:
        clean_base_restoration(lang)
