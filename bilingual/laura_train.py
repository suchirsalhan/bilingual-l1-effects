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
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from huggingface_hub import upload_folder

# ==========================================================
# CONFIG
# ==========================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = f"{HF_USER}/es-en-bilingual-5B"

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

# Matching the repo names from your Parquet Factory script
DATASETS = {
    "es": f"{HF_USER}/es-en-5B", 
    "en": f"{HF_USER}/en-es-5B"
}

PHASE_2_START = TOTAL_TOKENS // 2 # 2.5B Tokens
BASE_CHECKPOINTS = [0.25, 0.5, 0.75, 1.0]
LOG_CHECKPOINTS = [2 ** i / 100 for i in range(0, 7)]
CHECKPOINT_PERCENTAGES = sorted(set(BASE_CHECKPOINTS + LOG_CHECKPOINTS))
saved_checkpoints = set()

# ==========================================================
# DATASET & LOADER (Optimized for Parquet)
# ==========================================================
class BilingualParquetDataset(IterableDataset):
    def __init__(self, repo, skip_steps=0):
        self.repo = repo
        # skip calculation: steps * batch * accumulation
        self.skip = skip_steps * BATCH_SIZE * GRAD_ACCUM_STEPS
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # Load the dataset in streaming mode (reads Parquet shards)
        ds = load_dataset(self.repo, split="train", streaming=True)
        
        # Shuffle shards to ensure different nodes see different data
        ds = ds.shuffle(seed=42, buffer_size=10_000)
        
        # Shard the dataset across GPUs and Workers
        num_shards = self.world_size * num_workers
        shard_idx = (self.rank * num_workers) + worker_id
        ds = ds.shard(num_shards=num_shards, index=shard_idx)
        
        # Skip logic for resuming
        if self.skip > 0:
            # We skip proportional to the amount of workers/shards
            ds = ds.skip(self.skip // num_shards)
            
        for item in ds:
            ids = torch.tensor(item["input_ids"], dtype=torch.long)
            # The factory script produces (SEQ_LEN,) blocks
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
        # Local non-distributed fallback
        print("⚠️ Not in distributed mode. Running locally.")
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
    if not ckpts:
        return None
    return Path(max(ckpts, key=lambda x: int(x.split("-")[-1])))

# ==========================================================
# TRAINING LOOP
# ==========================================================
def train():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size()

    # Resume Logic: Sync checkpoints set
    for cp in glob.glob(str(CHECKPOINT_DIR / "es-en-bilingual-*")):
        try:
            pct_val = int(cp.split("-")[-1]) / 100.0
            saved_checkpoints.add(pct_val)
        except ValueError: continue

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
        log(f"📈 Resuming at {tokens_seen/1e9:.2f}B tokens")

    model = DDP(model, device_ids=[local_rank]) if torch.cuda.is_available() else model

    # Data Loaders
    es_loader = get_infinite_loader(DATASETS["es"], start_step)
    en_loader = get_infinite_loader(DATASETS["en"], start_step)

    tokens_per_step = SEQ_LEN * BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
    warmup_tokens = int(TOTAL_TOKENS * WARMUP_FRACTION)
    start_time = time.time()
    step = start_step
    
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)
        
        # Phase 1: 100% Spanish (with rare English injection)
        # Phase 2: 33% Spanish / 66% English mix
        if tokens_seen < PHASE_2_START:
            loader = es_loader if step % 10 != 0 else en_loader
        else:
            loader = es_loader if step % 3 == 0 else en_loader

        for micro_step in range(GRAD_ACCUM_STEPS):
            # Optimizing syncs for grad accumulation
            my_context = model.no_sync() if (isinstance(model, DDP) and micro_step < GRAD_ACCUM_STEPS - 1) else contextlib.nullcontext()
            
            with my_context:
                batch = next(loader).to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(batch, labels=batch)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        tokens_seen += tokens_per_step
        step += 1

        # Cosine LR Schedule
        if tokens_seen < warmup_tokens:
            lr = LR_MAX * (tokens_seen / warmup_tokens)
        else:
            progress = (tokens_seen - warmup_tokens) / (TOTAL_TOKENS - warmup_tokens)
            lr = LR_MAX * 0.5 * (1 + math.cos(math.pi * progress))
        
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Reporting
        if rank == 0 and step % 20 == 0:
            elapsed = time.time() - start_time
            tok_sec = (tokens_seen - (start_step * tokens_per_step)) / max(elapsed, 1)
            log(f"Step {step} | {tokens_seen/1e9:.3f}B Tok | {tok_sec:,.0f} tok/s | LR {lr:.2e} | Phase {'2' if tokens_seen >= PHASE_2_START else '1'}")

        # Checkpointing
        fraction = tokens_seen / TOTAL_TOKENS
        for cp_frac in CHECKPOINT_PERCENTAGES:
            if fraction >= cp_frac and cp_frac not in saved_checkpoints:
                dist.barrier()
                if rank == 0:
                    saved_checkpoints.add(cp_frac)
                    pct = int(cp_frac * 100)
                    cp_path = CHECKPOINT_DIR / f"es-en-bilingual-{pct}"
                    cp_path.mkdir(parents=True, exist_ok=True)

                    # Save model and optimizer
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(cp_path, safe_serialization=False)
                    torch.save({
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "optimizer_state_dict": optimizer.state_dict()
                    }, cp_path / "optimizer.pt")

                    log(f"💾 Checkpoint saved: {pct}%")
                    
                    if HF_TOKEN:
                        try:
                            upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=HF_TOKEN)
                            log(f"🚀 Uploaded to HF: {REPO_ID}")
                        except Exception as e:
                            log(f"⚠️ Upload failed: {e}")
                dist.barrier()

    if rank == 0:
        log("✅ Training Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
