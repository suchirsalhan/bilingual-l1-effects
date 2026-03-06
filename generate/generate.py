import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel
from huggingface_hub import hf_hub_download

# ==========================================================
# 1. CUSTOM SENTENCEPIECE WRAPPER
# ==========================================================
class SPTokenizer:
    def __init__(self, model_file, unk="<unk>", bos="<s>", eos="</s>", pad="<pad>"):
        self.sp_model = spm.SentencePieceProcessor(model_file=model_file)
        # Map tokens to IDs based on your SentencePiece model
        self.unk_id = self.sp_model.PieceToId(unk)
        self.bos_id = self.sp_model.PieceToId(bos)
        self.eos_id = self.sp_model.PieceToId(eos)
        self.pad_id = self.sp_model.PieceToId(pad)

    def encode(self, text, add_special_tokens=True):
        ids = self.sp_model.EncodeAsIds(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return torch.tensor([ids])  # Returns shape [1, seq_len]

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        # Handle nested lists if it's a batch
        if isinstance(ids[0], list):
            return [self.sp_model.DecodeIds(i) for i in ids]
        return self.sp_model.DecodeIds(ids)

# ==========================================================
# 2. SETUP AND LOADING
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO = "RA-ALTA/de-english-32"
TOKENIZER_REPO = "RA-ALTA/tokenizer-de"

print(f"🔹 Downloading tokenizer from {TOKENIZER_REPO}...")
# This fetches the .model file from your HF tokenizer repo
spm_path = hf_hub_download(repo_id=TOKENIZER_REPO, filename="spm.model")
tokenizer = SPTokenizer(spm_path)

print(f"🔹 Loading model from {MODEL_REPO}...")
# Transformers automatically handles the config.json and pytorch_model.bin
model = GPT2LMHeadModel.from_pretrained(MODEL_REPO).to(device)
model.eval()

# ==========================================================
# 3. INFERENCE & GENERATION
# ==========================================================
text = "Hallo, wie geht es dir?"
input_ids = tokenizer.encode(text).to(device)

print(f"\n📝 Input: {text}")
print(f"🔢 IDs: {input_ids.tolist()}")

with torch.no_grad():
    # Basic forward pass to get loss/logits
    outputs = model(input_ids, labels=input_ids)
    print(f"📉 Loss: {outputs.loss.item():.4f}")

    # Simple text generation
    # Since it's a GPT2 model, we use the model's built-in generate method
    generated_ids = model.generate(
        input_ids, 
        max_length=50, 
        do_sample=True, 
        top_p=0.95, 
        temperature=0.8,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id
    )

print(f"🤖 Model Response: {tokenizer.decode(generated_ids[0])}")
