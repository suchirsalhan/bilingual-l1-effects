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
    "n_embd": 1024, "n_layer": 24, "n_head": 16,
    "n_positions": SEQ_LEN, "vocab_size": VOCAB_SIZE,
    "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
    "loss_type": "cross_entropy",  # Silences the loss_type warning
}

DATASETS = {"tr": "RA-ALTA/tr-5B", "en": "RA-ALTA/en-5B"}

# Phase 2 logic
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
        with open(LOG_DIR / "tr-en-training.log", "a") as f:
            f.write(line + "\n")

def get_latest_checkpoint():
    # Fixed to look for your tr-english-X folders
    ckpts = glob.glob(str(CHECKPOINT_DIR / "tr-english-*"))
    if not ckpts:
        return None
    # Sort numerically so that '64' > '8'
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

    # Pre-populate saved_checkpoints so we don't re-upload what we already have
    for cp in glob.glob(str(CHECKPOINT_DIR / "tr-english-*")):
        try:
            pct_val = int(cp.split("-")[-1]) / 100.0
            saved_checkpoints.add(pct_val)
        except ValueError:
            continue

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
        loader = tr_loader if tokens_seen < PHASE_2_START or step % 3 == 0 else en_loader

        for micro_step in range(GRAD_ACCUM_STEPS):
            my_context = model.no_sync() if micro_step < GRAD_ACCUM_STEPS - 1 else contextlib.nullcontext()
            
            with my_context:
                batch = next(loader).to(device, non_blocking=True)
                # Updated to new non-deprecated API
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
                        cp_path = CHECKPOINT_DIR / f"tr-english-{pct}"
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
