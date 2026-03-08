#!/usr/bin/env python3
import os
import time
import math
import glob
import datetime
import contextlib
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from huggingface_hub import upload_folder

# ==========================================================
# CONFIG
# ==========================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = f"{HF_USER}/tr-bilingual-5B"

SEQ_LEN = 512
TOTAL_TOKENS = 5_000_000_000
GRAD_ACCUM_STEPS = 8
BATCH_SIZE = 16  
LR_MAX = 2e-4
WARMUP_FRACTION = 0.01
VOCAB_SIZE = 50_000

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "n_embd": 896,
    "n_layer": 24,
    "n_head": 14,
    "n_positions": SEQ_LEN, 
    "vocab_size": VOCAB_SIZE,
    "bos_token_id": 0, 
    "eos_token_id": 1, 
    "pad_token_id": 2,
}

DATASETS = {"tr": "RA-ALTA/tr-5B", "en": "RA-ALTA/en-5B"}

# Checkpointing Milestones
PHASE_2_START = TOTAL_TOKENS // 2
TARGET_PERCENTAGES = [0.25, 0.50, 0.75, 1.0]
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
        with open(LOG_DIR / "tr-en-training.log", "a") as f:
            f.write(line + "\n")

def get_latest_checkpoint():
    """Finds the most recent checkpoint based on Phase and Percentage."""
    ckpts = glob.glob(str(CHECKPOINT_DIR / "l*_*"))
    if not ckpts:
        return None
    
    def sort_key(path_str):
        name = Path(path_str).name # e.g., l1_25
        phase = int(name.split('_')[0][1:]) # Extract 1 from l1
        pct = int(name.split('_')[1])       # Extract 25
        return (phase, pct)

    latest = max(ckpts, key=sort_key)
    return Path(latest)

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
    for cp in glob.glob(str(CHECKPOINT_DIR / "l*_*")):
        cp_path = Path(cp)
        # Store as "p1_0.25" etc to match loop logic
        phase_num = cp_path.name.split('_')[0][1:]
        pct_val = int(cp_path.name.split('_')[1]) / 100.0
        saved_checkpoints.add(f"p{phase_num}_{pct_val}")

    conf = GPT2Config(**MODEL_CONFIG)
    model = GPT2LMHeadModel(conf).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX)

    latest_cp = get_latest_checkpoint()
    start_step, tokens_seen = 0, 0
    if latest_cp:
        log(f"🔄 Resuming from {latest_cp}")
        model = GPT2LMHeadModel.from_pretrained(latest_cp).to(device)
        ckpt = torch.load(latest_cp / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        log(f"📈 Resuming at {tokens_seen/1e9:.2f}B tokens")

    model = DDP(model, device_ids=[local_rank])

    tr_loader = get_infinite_loader(DATASETS["tr"], start_step)
    en_loader = get_infinite_loader(DATASETS["en"], start_step)

    tokens_per_step = SEQ_LEN * BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
    warmup_tokens = int(TOTAL_TOKENS * WARMUP_FRACTION)
    start_time = time.time()
    step = start_step
    
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)
        # Phase 1: Mostly TR | Phase 2: Mixed
        loader = tr_loader if tokens_seen < PHASE_2_START or step % 3 == 0 else en_loader

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

        # ==========================================================
        # MULTI-PHASE CHECKPOINTING (L1 & L2)
        # ==========================================================
        current_phase = 1 if tokens_seen < PHASE_2_START else 2
        fraction = tokens_seen / TOTAL_TOKENS
        
        should_save = False
        save_name = ""

        for cp_frac in TARGET_PERCENTAGES:
            checkpoint_key = f"p{current_phase}_{cp_frac}"
            if fraction >= cp_frac and checkpoint_key not in saved_checkpoints:
                should_save = True
                pct = int(cp_frac * 100)
                save_name = f"l{current_phase}_{pct}"
                if rank == 0:
                    saved_checkpoints.add(checkpoint_key)
                break

        if should_save:
            dist.barrier()
            if rank == 0:
                cp_path = CHECKPOINT_DIR / save_name
                cp_path.mkdir(parents=True, exist_ok=True)

                model.module.save_pretrained(cp_path, safe_serialization=False)
                torch.save({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "optimizer_state_dict": optimizer.state_dict()
                }, cp_path / "optimizer.pt")

                log(f"💾 Checkpoint saved locally: {save_name}")
                
                if HF_TOKEN:
                    try:
                        upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=HF_TOKEN)
                        log(f"🚀 Uploaded to HF: {REPO_ID}/{save_name}")
                    except Exception as e:
                        log(f"⚠️ Upload failed: {e}")
            dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
