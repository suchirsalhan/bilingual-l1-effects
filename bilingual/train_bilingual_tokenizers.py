#!/usr/bin/env python3
import os, argparse, logging, json, gzip, requests
from datasets import load_dataset
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast, HfApi

# ---------------- CONFIG ----------------
VOCAB_SIZE = 50000
TOKENIZER_ROOT = "./tokenizers"
HF_USER = "RA-ALTA"

# ---------------- LANGS ----------------
CULTURAX_LANGS = ["es","fr","de","pl","tr","ar","zh"]

# ---------------- STREAMS ----------------
def culturax_stream(lang):
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

def fineweb_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

def redpajama_stream(lang):
    BASE="https://data.together.xyz/redpajama-data-v2/v1.0.0"
    listing = requests.get(f"{BASE}/listings/{lang}-2023-06-head_middle.txt").text.splitlines()
    session = requests.Session()
    for shard in listing:
        try:
            r = session.get(f"{BASE}/documents/{shard}.json.gz", stream=True, timeout=60)
            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc.get("raw_content")
                    if text:
                        yield text.replace("\n"," ")
        except Exception:
            continue

# ---------------- COMBINED ITERATOR ----------------
def bilingual_iterator(lang, source):
    yield from culturax_stream(lang)
    if source == "fineweb":
        yield from fineweb_stream()
    else:
        yield from redpajama_stream(lang)

# ---------------- TRAIN TOKENIZER ----------------
def build_tokenizer(lang, source):
    out_dir = os.path.join(TOKENIZER_ROOT, f"{source}_{lang}")
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "spm")
    logging.info(f"Training tokenizer: {source}-{lang} → {out_dir}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=bilingual_iterator(lang, source),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
        bos_id=0, eos_id=1, pad_id=2, unk_id=3,
    )

    # Load & save tokenizer for HF
    tokenizer = PreTrainedTokenizerFast(
        sp_model_file=model_prefix + ".model",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>"
    )
    tokenizer.save_pretrained(out_dir)
    logging.info(f"Tokenizer saved: {out_dir}")

    # Push to Hugging Face Hub
    repo_name = f"{HF_USER}/{source}_{lang}-tokenizer"
    try:
        api = HfApi()
        api.create_repo(repo_name, exist_ok=True, repo_type="model")
        tokenizer.push_to_hub(repo_name, use_auth_token=True)
        logging.info(f"Tokenizer pushed to HF Hub: {repo_name}")
    except Exception as e:
        logging.error(f"Failed to push to HF: {e}")

# ---------------- MAIN ----------------
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=CULTURAX_LANGS)
    parser.add_argument("--source", required=True, choices=["fineweb","redpajama"])
    args = parser.parse_args()

    build_tokenizer(args.lang, args.source)
