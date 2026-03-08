#!/usr/bin/env python3
import os
import time
import torch
import multiprocessing as mp
from threading import Thread
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ["HF_TOKEN"]

# Languages and existing HF tokenizer/dataset repo mapping
LANGS = ["en","zh","ar","tr","pl","de","fr","es"]

TOKENIZER_REPOS = {
    "en": f"{HF_USER}/tokenizer-en",
    "zh": f"{HF_USER}/tokenizer-zh",
    "ar": f"{HF_USER}/tokenizer-ar",
    "tr": f"{HF_USER}/tokenizer-tr",
    "pl": f"{HF_USER}/tokenizer-pl",
    "de": f"{HF_USER}/tokenizer-de",
    "fr": f"{HF_USER}/tokenizer-fr",
    "es": f"{HF_USER}/tokenizer-es",
}

DATASET_REPOS = {
    "en": f"{HF_USER}/en-5B",
    "zh": f"{HF_USER}/zh-5B",
    "ar": f"{HF_USER}/ar-5B",
    "tr": f"{HF_USER}/tr-5B",
    "pl": f"{HF_USER}/pl-5B",
    "de": f"{HF_USER}/de-5B",
    "fr": f"{HF_USER}/fr-5B",
    "es": f"{HF_USER}/es-5B",
}

VOCAB_SIZE = 50_000
SEQ_LEN = 512
BOOTSTRAP_SENTENCES = 500_000
TMP = Path("tmp_bpe")
TMP.mkdir(exist_ok=True)

TOTAL_TOKENS = 5_000_000_000  # Only 5B tokens
RATIO_L1 = 2/3
RATIO_L2 = 1/3
SHARD_BLOCKS = 10_000

api = HfApi(token=HF_TOKEN)

# =====================================================
# DATA STREAMS
# =====================================================
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

def bilingual_stream(lang):
    l1_gen = culturax_stream(lang)
    en_gen = fineweb_stream()
    while True:
        for t in l1_gen:
            yield t, "L1"
        for t in en_gen:
            yield t, "L2"

# =====================================================
# TRAIN HF BPE TOKENIZER
# =====================================================
def train_bilingual_bpe(lang):
    stream = bilingual_stream(lang)
    sample_texts = [next(stream)[0] for _ in range(BOOTSTRAP_SENTENCES)]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    print(f"🚀 Training bilingual BPE tokenizer for {lang}")
    tokenizer.train_from_iterator(sample_texts, trainer=trainer)

    out_dir = TMP / f"tokenizer_{lang}"
    out_dir.mkdir(exist_ok=True)
    tokenizer.model.save(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(out_dir / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>"
    )

    print(f"✅ Tokenizer ready for {lang} | Vocab size: {len(hf_tok)}")
    return hf_tok, out_dir

# =====================================================
# PUSH TOKENIZER TO HF
# =====================================================
def push_tokenizer(out_dir, lang):
    repo_id = TOKENIZER_REPOS[lang]
    create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model", token=HF_TOKEN)
    print(f"✅ Uploaded tokenizer for {lang} to {repo_id}")

# =====================================================
# TOKENIZE + ARROW WRITER
# =====================================================
def tokenizer_worker(text_q, block_q, tokenizer, max_tokens_L1, max_tokens_L2, counter):
    buffer = []
    count_L1 = 0
    count_L2 = 0
    start_time = time.time()
    while True:
        item = text_q.get()
        if item is None:
            break
        text, tag = item
        tokens = tokenizer.encode(text).ids

        if tag == "L1":
            if count_L1 >= max_tokens_L1: continue
            remaining = max_tokens_L1 - count_L1
            tokens = tokens[:remaining]
            count_L1 += len(tokens)
        else:
            if count_L2 >= max_tokens_L2: continue
            remaining = max_tokens_L2 - count_L2
            tokens = tokens[:remaining]
            count_L2 += len(tokens)

        buffer.extend(tokens)
        while len(buffer) >= SEQ_LEN:
            block_q.put(np.array(buffer[:SEQ_LEN], dtype=np.int32))
            buffer = buffer[SEQ_LEN:]
            with counter.get_lock():
                counter.value += SEQ_LEN
                elapsed = time.time() - start_time
                eta = ((max_tokens_L1 + max_tokens_L2) - counter.value) / (counter.value / elapsed + 1e-9)
                print(f"\rTokens processed: {counter.value/1e9:.2f}B | ETA: {eta/60:.1f} min", end="")

        if count_L1 >= max_tokens_L1 and count_L2 >= max_tokens_L2:
            break
    print()

def arrow_writer(block_q, upload_q, lang, counter):
    shard = 0
    arrays = []
    while True:
        block = block_q.get()
        if block is None: break
        arrays.append(pa.array([block]))
        if len(arrays) >= SHARD_BLOCKS:
            table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
            fname = TMP / f"{lang}_train_5B_{shard}.arrow"
            with ipc.new_file(fname, table.schema) as w: w.write(table)
            upload_q.put(fname)
            arrays = []
            shard += 1
            print(f"Shard {shard} written for {lang}-5B | Tokens so far: {counter.value/1e9:.2f}B")
    if arrays:
        table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
        fname = TMP / f"{lang}_train_5B_{shard}.arrow"
        with ipc.new_file(fname, table.schema) as w: w.write(table)
        upload_q.put(fname)
        print(f"Final shard {shard} written for {lang}-5B | Tokens so far: {counter.value/1e9:.2f}B")

# =====================================================
# UPLOADER
# =====================================================
def uploader(upload_q, lang):
    dataset_repo = DATASET_REPOS[lang]
    create_repo(dataset_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    while True:
        fname = upload_q.get()
        if fname is None: break
        api.upload_file(str(fname), path_in_repo=fname.name, repo_id=dataset_repo, repo_type="dataset")
        os.remove(fname)

# =====================================================
# LANGUAGE PIPELINE
# =====================================================
def process_language(lang, gpu_id):
    print(f"🔹 GPU {gpu_id}: Starting pipeline for {lang}")

    tokenizer, out_dir = train_bilingual_bpe(lang)
    push_tokenizer(out_dir, lang)

    max_tokens_L1 = int(TOTAL_TOKENS * RATIO_L1)
    max_tokens_L2 = int(TOTAL_TOKENS * RATIO_L2)

    counter = mp.Value("i", 0)
    text_q = mp.Queue(2000)
    block_q = mp.Queue(2000)
    upload_q = mp.Queue()

    tok_proc = mp.Process(target=tokenizer_worker,
                          args=(text_q, block_q, tokenizer, max_tokens_L1, max_tokens_L2, counter))
    writer_proc = mp.Process(target=arrow_writer, args=(block_q, upload_q, lang, counter))
    up_thread = Thread(target=uploader, args=(upload_q, lang))

    tok_proc.start(); writer_proc.start(); up_thread.start()

    stream = bilingual_stream(lang)
    for item in stream:
        text_q.put(item)
    text_q.put(None)
    tok_proc.join()

    block_q.put(None)
    writer_proc.join()

    upload_q.put(None)
    up_thread.join()
    print(f"🎉 GPU {gpu_id}: Finished 5B-token dataset for {lang}")

# =====================================================
# MAIN: DISTRIBUTED 8-GPU
# =====================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    gpus = min(8, len(LANGS))
    processes = []
    for i, lang in enumerate(LANGS):
        gpu_id = i % gpus
        p = mp.Process(target=process_language, args=(lang, gpu_id))
        processes.append(p)
        p.start()
    for p in processes: p.join()

    print("✅ All tokenizers and 5B-token datasets uploaded to existing HF repos")
