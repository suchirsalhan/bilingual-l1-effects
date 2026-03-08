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
# ARGUMENTS & CONFIG
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", type=str, required=True, help="Primary language (e.g., es, tr, zh)")
parser.add_argument("--hf_user", type=str, default="RA-ALTA")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr_max", type=float, default=2e-4)
parser.add_argument("--warmup_frac", type=float, default=0.01)
parser.add_argument("--vocab_size", type=int, default=50_000)
args = parser.parse_args()

# Curriculum Constants
PHASE_1_TOKENS = 3_500_000_000
PHASE_2_TOKENS = 2_000_000_000
TOTAL_TOKENS = PHASE_1_TOKENS + PHASE_2_TOKENS

# Dynamic Dataset Mapping
DATASETS = {
    "l1": f"{args.hf_user}/{args.lang_l1}-5B-parquet",
    "en": f"{args.hf_user}/en-5B-parquet"
}

REPO_ID = f"{args.hf_user}/{args.lang_l1}-en-bilingual-5B"
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
for d in [CHECKPOINT_DIR, LOG_DIR]: d.mkdir(exist_ok=True)

# Milestone Mapping (Restored Names)
P1_MILESTONES = {
    int(PHASE_1_TOKENS * 0.5): f"{args.lang_l1}-monolingual-50",
    int(PHASE_1_TOKENS * 0.75): f"{args.lang_l1}-monolingual-75",
    PHASE_1_TOKENS: f"{args.lang_l1}-monolingual-100"
}
P2_MILESTONES = {
    PHASE_1_TOKENS + int(PHASE_2_TOKENS * 0.25): "beginner",
    PHASE_1_TOKENS + int(PHASE_2_TOKENS * 0.50): "intermediate",
    PHASE_1_TOKENS + int(PHASE_2_TOKENS * 0.75): "advanced",
    TOTAL_TOKENS: "fluent"
}
ALL_MILESTONES = {**P1_MILESTONES, **P2_MILESTONES}

# ==========================================================
# DATASET & LOADER
# ==========================================================
class ParquetDataset(IterableDataset):
    def __init__(self, repo, skip_steps=0):
        self.repo = repo
        self.skip = skip_steps * args.batch_size * args.grad_accum
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        shard_idx = (self.rank * num_workers) + (worker_info.id if worker_info else 0)
        
        # Load the pre-tokenized parquet shards
        ds = load_dataset(self.repo, split="train", streaming=True)
        ds = ds.shard(num_shards=self.world_size * num_workers, index=shard_idx)
        
        if self.skip > 0:
            ds = ds.skip(self.skip // (self.world_size * num_workers))
            
        for item in ds:
            yield torch.tensor(item["input_ids"], dtype=torch.long)

def get_infinite_loader(repo, skip_steps):
    def data_generator():
        while True:
            dataset = ParquetDataset(repo, skip_steps)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
            for batch in loader: yield batch
    return iter(data_generator())

# ==========================================================
# TRAINING EXECUTION
# ==========================================================
def setup():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def log(msg):
    if dist.get_rank() == 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def train():
    setup()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    rank = dist.get_rank()

    # Model Init
    conf = GPT2Config(
        n_embd=1024, n_layer=24, n_head=16, 
        n_positions=args.seq_len, vocab_size=args.vocab_size,
        bos_token_id=0, eos_token_id=1, pad_token_id=2
    )
    model = GPT2LMHeadModel(conf).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max)
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Loaders: L1 and English
    l1_loader = get_infinite_loader(DATASETS["l1"], 0)
    en_loader = get_infinite_loader(DATASETS["en"], 0)

    tokens_per_step = args.seq_len * args.batch_size * dist.get_world_size() * args.grad_accum
    tokens_seen, step = 0, 0
    saved_milestones = set()

    model.train()
    log(f"🚀 Starting {args.lang_l1} Training Curriculum...")

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)
        
        # Phase 1: 100% L1 | Phase 2: 2:1 L1:EN ratio
        if tokens_seen < PHASE_1_TOKENS:
            loader = l1_loader
        else:
            loader = l1_loader if step % 3 != 0 else en_loader

        for micro_step in range(args.grad_accum):
            my_context = model.no_sync() if micro_step < args.grad_accum - 1 else contextlib.nullcontext()
            with my_context:
                batch = next(loader).to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch, labels=batch).loss / args.grad_accum
                loss.backward()

        optimizer.step()
        tokens_seen += tokens_per_step
        step += 1

        # Cosine LR Schedule
        warmup_tokens = TOTAL_TOKENS * args.warmup_frac
        if tokens_seen < warmup_tokens:
            lr = args.lr_max * (tokens_seen / warmup_tokens)
        else:
            lr = args.lr_max * 0.5 * (1 + math.cos(math.pi * (tokens_seen - warmup_tokens) / (TOTAL_TOKENS - warmup_tokens)))
        for g in optimizer.param_groups: g["lr"] = lr

        # Checkpointing with custom names
        for milestone, name in ALL_MILESTONES.items():
            if tokens_seen >= milestone and name not in saved_milestones:
                dist.barrier()
                if rank == 0:
                    cp_path = CHECKPOINT_DIR / name
                    model.module.save_pretrained(cp_path)
                    log(f"💾 Milestone Reached: {name} ({tokens_seen/1e9:.2f}B tokens)")
                    saved_milestones.add(name)
                    
                    if os.environ.get("HF_TOKEN"):
                        try:
                            upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=os.environ["HF_TOKEN"])
                        except Exception as e:
                            log(f"⚠️ Upload failed: {e}")
                dist.barrier()

        if rank == 0 and step % 50 == 0:
            log(f"Step {step} | {tokens_seen/1e9:.2f}B Tokens | LR {lr:.2e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
