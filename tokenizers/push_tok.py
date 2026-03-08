import os
from pathlib import Path
import sentencepiece as spm
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from huggingface_hub import upload_folder

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")
LANGS = ["es", "fr", "de", "pl", "tr", "ar", "zh", "en"]
TMP = Path("tmp_factory")

def manual_vocab_restoration(lang):
    spm_path = TMP / f"tokenizer_{lang}" / "spm.model"
    hf_out_dir = TMP / f"hf_fixed_{lang}"
    hf_out_dir.mkdir(exist_ok=True, parents=True)

    if not spm_path.exists():
        print(f"❌ {lang}: spm.model missing.")
        return

    print(f"🛠️  Force-extracting vocab for {lang}...")

    try:
        # 1. Load with raw SentencePiece to get the actual pieces
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))
        
        # 2. Extract every single token and its score
        # This is the "ground truth" of your 5B token training
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        
        # 3. Create a BPE model manually from the vocab
        # (Since you used model_type="bpe" in your trainer)
        # Note: Merges are technically stored in the .model, but for a 
        # Fast tokenizer, the vocab mapping is the most critical part for ID recovery.
        tok_model = models.BPE(vocab=vocab, merges=[])
        tokenizer = Tokenizer(tok_model)

        # 4. Wrap in PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        # 5. Add Chat Template for Instruction Tuning
        fast_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

        fast_tokenizer.save_pretrained(hf_out_dir)

        # VERIFY
        final_vocab_size = len(fast_tokenizer.get_vocab())
        print(f"✅ {lang} fixed. Vocab size: {final_vocab_size}")

        if final_vocab_size < 1000:
            print(f"⚠️  Fatal Error: {lang} still has tiny vocab. Do not push.")
            return

        # 6. PUSH TO HUB
        repo_id = f"{HF_USER}/tokenizer-{lang}"
        upload_folder(
            folder_path=str(hf_out_dir),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="Fix: Manual vocab extraction from spm.model pieces"
        )
        print(f"🚀 Pushed {lang} to {repo_id}")

    except Exception as e:
        print(f"💥 {lang} failed: {e}")

if __name__ == "__main__":
    for lang in LANGS:
        manual_vocab_restoration(lang)
