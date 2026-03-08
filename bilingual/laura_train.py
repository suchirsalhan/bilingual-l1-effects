#!/usr/bin/env python3
import os
import time
import math
import glob
import datetime
import contextlib
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from huggingface_hub import upload_folder

# ==========================================================
# ARGUMENTS
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, required=True,
                    help="Primary language (L1) dataset key, e.g., tr, zh, es")
parser.add_argument("--hf_user", type=str, default="RA-ALTA", help="HF username")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--total_tokens", type=int, default=5_000_000_000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr_max", type=float, default=2e-4)
parser.add_argument("--warmup_frac", type=float, default=0.01)
parser.add_argument("--vocab_size", type=int, default=50_000)
args = parser.parse_args()

HF_USER = args.hf_user
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = f"{HF_USER}/{args.lang_l1}-bilingual-5B"

SEQ_LEN = args.seq_len
TOTAL_TOKENS = args.total_tokens
BATCH_SIZE = args.batch_size
GRAD_ACCUM_STEPS = args.grad_accum
LR_MAX = args.lr_max
WARMUP_FRACTION = args.warmup_frac
VOCAB_SIZE = args.vocab_size

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "n_embd": 1024, "n_layer": 24, "n_head": 16,
    "n_positions": SEQ_LEN, "vocab_size": VOCAB_SIZE,
    "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
    "loss_type": "cross_entropy",
}

# Map language keys to HF dataset repos
DATASETS = {
    "tr": "RA-ALTA/tr-5B",
    "en": "RA-ALTA/en-5B",
    "zh": "RA-ALTA/zh-5B",
    "ar": "RA-ALTA/ar-5B",
    "pl": "RA-ALTA/pl-5B",
    "de": "RA-ALTA/de-5B",
    "fr": "RA-ALTA/fr-5B",
    "es": "RA-ALTA/es-5B"
}

# Phase 2 checkpointing
PHASE_2_START = TOTAL_TOKENS // 2
BASE_CHECKPOINTS = [0.25, 0.5, 0.75, 1.0]
LOG_CHECKPOINTS = [2 ** i / 100 for i in range(0, 7)]
CHECKPOINT_PERCENTAGES = sorted(set(BASE_CHECKPOINTS + LOG_CHECKPOINTS))
saved_checkpoints = set()

# ==========================================================
# DATASET & LOADER
# ==========================================================
class BilingualDataset(IterableDataset):
    def __init__(self, repo, skip_steps=0):
        self.repo = repo
        self.skip = skip_steps * BATCH_SIZE * GRAD_ACCUM_STEPS
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        num_shards = self.world_size * num_workers
        shard_idx = (self.rank * num_workers) + worker_id
        
        ds = load_dataset(self.repo, split="train", streaming=True)
        ds = ds.shard(num_shards=num_shards, index=shard_idx)
        
        if self.skip > 0:
            ds = ds.skip(self.skip // num_shards)
            
        for item in ds:
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            if ids.size(0) == SEQ_LEN:
                yield ids

def get_infinite_loader(repo, skip_steps):
    def data_generator():
        while True:
            dataset = BilingualDataset(repo, skip_steps)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
            for batch in loader:
                yield batch
    return iter(data_generator())

# ==========================================================
# HELPERS
# ==========================================================
def setup():
    dist.init_process_group(
        backend="nccl", 
        timeout=datetime.timedelta(seconds=7200)
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def log(msg):
    if dist.get_rank() == 0:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(LOG_DIR / f"{args.lang_l1}-en-training.log", "a") as f:
            f.write(line + "\n")

def get_latest_checkpoint(lang):
    ckpts = glob.glob(str(CHECKPOINT_DIR / f"{lang}-*"))
    if not ckpts:
        return None
    return Path(max(ckpts, key=lambda x: int(x.split("-")[-1])))

# ==========================================================
# TRAINING LOOP
# ==========================================================
def train():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # Pre-populate saved_checkpoints
    for cp in glob.glob(str(CHECKPOINT_DIR / f"{args.lang_l1}-*")):
        try:
            pct_val = int(cp.split("-")[-1]) / 100.0
            saved_checkpoints.add(pct_val)
        except ValueError:
            continue

    conf = GPT2Config(**MODEL_CONFIG)
    model = GPT2LMHeadModel(conf).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX)

    latest_cp = get_latest_checkpoint(args.lang_l1)
    start_step, tokens_seen = 0, 0
    if latest_cp:
        log(f"🔄 Resuming from {latest_cp}")
        model = GPT2LMHeadModel.from_pretrained(latest_cp).to(device)
        ckpt = torch.load(latest_cp / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        log(f"📈 Resuming at {tokens_seen/1e9:.2f}B tokens")

    model = DDP(model, device_ids=[local_rank])

    l1_loader = get_infinite_loader(DATASETS[args.lang_l1], start_step)
    en_loader = get_infinite_loader(DATASETS["en"], start_step)

    tokens_per_step = SEQ_LEN * BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
    warmup_tokens = int(TOTAL_TOKENS * WARMUP_FRACTION)
    start_time = time.time()
    step = start_step
    
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)
        loader = l1_loader if tokens_seen < PHASE_2_START or step % 3 == 0 else en_loader

        for micro_step in range(GRAD_ACCUM_STEPS):
            my_context = model.no_sync() if micro_step < GRAD_ACCUM_STEPS - 1 else contextlib.nullcontext()
            
            with my_context:
                batch = next(loader).to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch, labels=batch).loss / GRAD_ACCUM_STEPS
                loss.backward()

        optimizer.step()
        tokens_seen += tokens_per_step
        step += 1

        # LR Schedule
        if tokens_seen < warmup_tokens:
            lr = LR_MAX * (tokens_seen / warmup_tokens)
        else:
            lr = LR_MAX * 0.5 * (1 + math.cos(math.pi * (tokens_seen - warmup_tokens) / (TOTAL_TOKENS - warmup_tokens)))
        for g in optimizer.param_groups:
            g["lr"] = lr

        if rank == 0 and step % 20 == 0:
            elapsed = time.time() - start_time
            tok_sec = (tokens_seen - (start_step * tokens_per_step)) / max(elapsed, 1)
            log(f"Step {step} | {tokens_seen/1e9:.2f}B Tok | {tok_sec:,.0f} tok/s | LR {lr:.2e}")

        # SYNCHRONIZED CHECKPOINTING
        if tokens_seen >= PHASE_2_START:
            fraction = tokens_seen / TOTAL_TOKENS
            for cp_frac in CHECKPOINT_PERCENTAGES:
                if fraction >= cp_frac and cp_frac not in saved_checkpoints:
                    dist.barrier()
                    
                    if rank == 0:
                        saved_checkpoints.add(cp_frac)
                        pct = int(cp_frac * 100)
                        cp_path = CHECKPOINT_DIR / f"{args.lang_l1}-{pct}"
                        cp_path.mkdir(parents=True, exist_ok=True)

                        model.module.save_pretrained(cp_path, safe_serialization=False)
                        torch.save({
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "optimizer_state_dict": optimizer.state_dict()
                        }, cp_path / "optimizer.pt")

                        log(f"💾 Phase 2 checkpoint saved locally: {pct}%")
                        
                        if HF_TOKEN:
                            try:
                                upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=HF_TOKEN)
                                log(f"🚀 Uploaded to HF: {REPO_ID}")
                            except Exception as e:
                                log(f"⚠️ Upload failed: {e}")
                    
                    dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
