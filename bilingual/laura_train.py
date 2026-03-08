#!/usr/bin/env python3
import os
import time
import math
import glob
import datetime
import contextlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import upload_folder

# ==========================================================
# CONFIG
# ==========================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = f"{HF_USER}/es-en-bilingual-5B"
# Dedicated repository for inference-ready L2 English/L1 Spanish models
CURRICULUM_REPO_ID = f"{HF_USER}/es-en-l1-es-l2-en-curriculum"
TOKENIZER_ID = f"{HF_USER}/tokenizer-es-en"

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

DATASETS = {
    "es": f"{HF_USER}/es-en-5B", 
    "en": f"{HF_USER}/en-es-5B"
}

PHASE_2_START = TOTAL_TOKENS // 2 

# --- Milestone Definitions ---
# Phase 1 checkpoints (0, 25, 50, 100% of Phase 1)
P1_TARGETS = [0.0, 0.125, 0.25, 0.5] 

# Phase 2 Curriculum (L1 Spanish, L2 English)
# Maps total token fraction to descriptive names
P2_CURRICULUM = {
    0.625: "l1-es-l2-en-beginner",      # 25% thru P2
    0.75:  "l1-es-l2-en-intermediate",  # 50% thru P2
    0.875: "l1-es-l2-en-advanced",      # 75% thru P2
    1.0:   "l1-es-l2-en-fluent"         # 100% thru P2
}

ALL_CHECKPOINTS = sorted(set(P1_TARGETS + list(P2_CURRICULUM.keys())))
saved_checkpoints = set()

# ==========================================================
# DATASET & LOADER
# ==========================================================
class BilingualParquetDataset(IterableDataset):
    def __init__(self, repo, skip_steps=0):
        self.repo = repo
        self.skip = skip_steps * BATCH_SIZE * GRAD_ACCUM_STEPS
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        ds = load_dataset(self.repo, split="train", streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=10_000)
        
        num_shards = self.world_size * num_workers
        shard_idx = (self.rank * num_workers) + worker_id
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
            dataset = BilingualParquetDataset(repo, skip_steps)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
            for batch in loader:
                yield batch
    return iter(data_generator())

# ==========================================================
# HELPERS
# ==========================================================
def setup():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.FileStore("tmp_store", 1))

def log(msg):
    if dist.get_rank() == 0:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(LOG_DIR / "es-en-training.log", "a") as f:
            f.write(line + "\n")

def get_latest_checkpoint():
    ckpts = glob.glob(str(CHECKPOINT_DIR / "es-en-bilingual-*"))
    if not ckpts: return None
    return Path(max(ckpts, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0))

# ==========================================================
# TRAINING LOOP
# ==========================================================
def train():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size()

    # Load shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)

    conf = GPT2Config(**MODEL_CONFIG)
    model = GPT2LMHeadModel(conf).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)

    latest_cp = get_latest_checkpoint()
    start_step, tokens_seen = 0, 0
    if latest_cp:
        log(f"🔄 Resuming from {latest_cp}")
        model = GPT2LMHeadModel.from_pretrained(latest_cp).to(device)
        ckpt_data = torch.load(latest_cp / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        start_step, tokens_seen = ckpt_data["step"], ckpt_data["tokens_seen"]

    model = DDP(model, device_ids=[local_rank]) if torch.cuda.is_available() else model

    es_loader = get_infinite_loader(DATASETS["es"], start_step)
    en_loader = get_infinite_loader(DATASETS["en"], start_step)

    tokens_per_step = SEQ_LEN * BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
    warmup_tokens = int(TOTAL_TOKENS * WARMUP_FRACTION)
    start_time = time.time()
    step = start_step
    
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)
        
        # Sampling logic
        if tokens_seen < PHASE_2_START:
            loader = es_loader if step % 10 != 0 else en_loader
        else:
            loader = es_loader if step % 3 == 0 else en_loader

        for micro_step in range(GRAD_ACCUM_STEPS):
            my_context = model.no_sync() if (isinstance(model, DDP) and micro_step < GRAD_ACCUM_STEPS - 1) else contextlib.nullcontext()
            with my_context:
                batch = next(loader).to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(batch, labels=batch)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        tokens_seen += tokens_per_step
        step += 1

        # LR Schedule
        if tokens_seen < warmup_tokens:
            lr = LR_MAX * (tokens_seen / warmup_tokens)
        else:
            progress = (tokens_seen - warmup_tokens) / (TOTAL_TOKENS - warmup_tokens)
            lr = LR_MAX * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups: g["lr"] = lr

        # Periodic Logging
        if rank == 0 and step % 20 == 0:
            log(f"Step {step} | {tokens_seen/1e9:.3f}B Tok | LR {lr:.2e}")

        # --- CHECKPOINTING & CURRICULUM PUSH ---
        fraction = tokens_seen / TOTAL_TOKENS
        for cp_frac in ALL_CHECKPOINTS:
            if fraction >= cp_frac and cp_frac not in saved_checkpoints:
                dist.barrier()
                if rank == 0:
                    saved_checkpoints.add(cp_frac)
                    pct = int(cp_frac * 100)
                    
                    # Local save
                    folder_name = f"es-en-bilingual-{pct}"
                    cp_path = CHECKPOINT_DIR / folder_name
                    cp_path.mkdir(parents=True, exist_ok=True)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(cp_path, safe_serialization=False)
                    tokenizer.save_pretrained(cp_path)

                    torch.save({
                        "step": step, "tokens_seen": tokens_seen,
                        "optimizer_state_dict": optimizer.state_dict()
                    }, cp_path / "optimizer.pt")

                    log(f"💾 Checkpoint: {pct}%")
                    
                    if HF_TOKEN:
                        # 1. Push to main training repo
                        try:
                            upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=HF_TOKEN)
                            
                            # 2. Push to L1/L2 Curriculum repo if it's a P2 Milestone
                            if cp_frac in P2_CURRICULUM:
                                label = P2_CURRICULUM[cp_frac]
                                log(f"🚀 Pushing curriculum milestone: {label}")
                                upload_folder(
                                    folder_path=str(cp_path),
                                    repo_id=CURRICULUM_REPO_ID,
                                    path_in_repo=label, # Organized by label folder
                                    token=HF_TOKEN
                                )
                        except Exception as e:
                            log(f"⚠️ Upload failed: {e}")
                dist.barrier()

    if rank == 0: log("✅ Training Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
