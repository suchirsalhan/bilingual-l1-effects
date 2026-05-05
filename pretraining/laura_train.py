#!/usr/bin/env python3

import os
import math
import time
import glob
import argparse
import datetime
import contextlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from torch import optim

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import upload_folder, create_repo

# ------------------------------------------------
# ENVIRONMENT & ARGS
# ------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", required=True)
parser.add_argument("--lang_l2", default="en")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--total_tokens", type=int, default=5_000_000_000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

L1, L2 = args.lang_l1, args.lang_l2
PAIR = f"{L1}-{L2}"
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Constants
SEQ_LEN = args.seq_len
TOTAL_TOKENS = args.total_tokens
PHASE_2_START_TOKEN = TOTAL_TOKENS // 2  # 50% Mark
# Phase 2 has 50% of total tokens. At 1:2 ratio, L2 tokens = (2/3) * (TOTAL_TOKENS / 2)
TOTAL_L2_EXPECTED = (2/3) * (TOTAL_TOKENS - PHASE_2_START_TOKEN)

CHECKPOINT_DIR = Path(f"checkpoints_{PAIR}")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# DISTRIBUTED & LOGGING
# ------------------------------------------------
def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, local_rank, device

def log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ------------------------------------------------
# DATA LOADING
# ------------------------------------------------
class StreamingDataset(IterableDataset):
    def __init__(self, repo, rank, world):
        self.repo, self.rank, self.world = repo, rank, world
    def __iter__(self):
        ds = load_dataset(self.repo, split="train", streaming=True).shuffle(seed=42)
        if self.world > 1: ds = ds.shard(self.world, self.rank)
        for row in ds:
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            if ids.shape[0] == SEQ_LEN: yield ids

def infinite_loader(repo, rank, world):
    dataset = StreamingDataset(repo, rank, world)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    while True:
        for batch in loader: yield batch

# ------------------------------------------------
# CHECKPOINT & UPLOAD LOGIC
# ------------------------------------------------
def save_and_upload(rank, model, tokenizer, optimizer, step, tokens_seen, l2_tokens, label):
    dist.barrier()
    if rank != 0: 
        dist.barrier()
        return

    repo_id = f"{HF_USER}/{PAIR}-{label}"
    cp_path = CHECKPOINT_DIR / label
    cp_path.mkdir(exist_ok=True)

    # Save
    m = model.module if hasattr(model, "module") else model
    m.save_pretrained(cp_path, safe_serialization=False)
    tokenizer.save_pretrained(cp_path)
    torch.save({"step": step, "tokens": tokens_seen, "l2_tokens": l2_tokens, "optimizer": optimizer.state_dict()}, cp_path / "optimizer.pt")

    if HF_TOKEN:
        try:
            create_repo(repo_id, token=HF_TOKEN, exist_ok=True)
            upload_folder(folder_path=str(cp_path), repo_id=repo_id, token=HF_TOKEN)
            log(rank, f"🚀 Uploaded to {repo_id}")
        except Exception as e:
            log(rank, f"⚠️ Upload failed: {e}")
    dist.barrier()

# ------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------
def train():
    rank, world, local_rank, device = setup_distributed()
    tokenizer = AutoTokenizer.from_pretrained(f"{HF_USER}/tokenizer-{PAIR}")
    
    config = GPT2Config(vocab_size=50000, n_positions=SEQ_LEN, n_embd=896, n_layer=24, n_head=14)
    model = GPT2LMHeadModel(config).to(device)
    
    # State tracking
    step, tokens_seen, l2_tokens_seen = 0, 0, 0
    milestones_hit = set()

    # Resume Logic
    ckpts = glob.glob(str(CHECKPOINT_DIR / "*/optimizer.pt"))
    if ckpts:
        latest = max(ckpts, key=os.path.getmtime)
        parent = Path(latest).parent
        model = GPT2LMHeadModel.from_pretrained(parent).to(device)
        state = torch.load(latest, map_location=device)
        step, tokens_seen, l2_tokens_seen = state["step"], state["tokens"], state.get("l2_tokens", 0)
        log(rank, f"🔄 Resumed from {parent} at {tokens_seen/1e9:.2f}B tokens")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if world > 1: model = DDP(model, device_ids=[local_rank])

    l1_loader = infinite_loader(f"{HF_USER}/{L1}-{L2}-5B", rank, world)
    l2_loader = infinite_loader(f"{HF_USER}/{L2}-{L1}-5B", rank, world)

    tokens_per_step = SEQ_LEN * args.batch_size * world * args.grad_accum
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)

        # 1. PHASE LOGIC
        if tokens_seen < PHASE_2_START_TOKEN:
            # Phase 1: 100% L1
            loader = l1_loader
            is_l2_step = False
        else:
            # Phase 2: 1:2 Ratio (L1:L2)
            # Step 0=L1, Step 1=L2, Step 2=L2...
            if step % 3 == 0:
                loader = l1_loader
                is_l2_step = False
            else:
                loader = l2_loader
                is_l2_step = True

        # 2. FORWARD / BACKWARD
        for micro in range(args.grad_accum):
            ctx = (model.no_sync() if isinstance(model, DDP) and micro < args.grad_accum - 1 else contextlib.nullcontext())
            with ctx:
                batch = next(loader).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch, labels=batch).loss / args.grad_accum
                loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update Counters
        tokens_seen += tokens_per_step
        if is_l2_step:
            l2_tokens_seen += tokens_per_step
        step += 1

        # 3. CHECKPOINTING TRIGGERS
        # A. Phase 1 Final (Exactly at 50% of total training)
        if tokens_seen >= PHASE_2_START_TOKEN and "p1_final" not in milestones_hit:
            save_and_upload(rank, model, tokenizer, optimizer, step, tokens_seen, l2_tokens_seen, "phase1-final")
            milestones_hit.add("p1_final")

        # B. Phase 2 Curriculum (Based on L2 % seen in Phase 2)
        if tokens_seen > PHASE_2_START_TOKEN:
            l2_frac = l2_tokens_seen / TOTAL_L2_EXPECTED
            curriculum = [
                (0.25, "beginner"),
                (0.50, "intermediate"),
                (0.75, "advanced"),
                (0.99, "fluent") # Using 0.99 to catch the end of stream
            ]
            for threshold, label in curriculum:
                if l2_frac >= threshold and label not in milestones_hit:
                    save_and_upload(rank, model, tokenizer, optimizer, step, tokens_seen, l2_tokens_seen, label)
                    milestones_hit.add(label)

        if rank == 0 and step % 50 == 0:
            log(rank, f"Step {step} | L2 Tokens: {l2_tokens_seen/1e6:.1f}M | Total: {tokens_seen/1e9:.3f}B")

    log(rank, "✅ Training complete.")

if __name__ == "__main__":
    train()
