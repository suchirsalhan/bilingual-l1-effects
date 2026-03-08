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
# ENVIRONMENT SAFETY
# ------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

# ------------------------------------------------
# ARGUMENTS
# ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lang_l1", required=True)
parser.add_argument("--lang_l2", default="en")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--total_tokens", type=int, default=5_000_000_000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_frac", type=float, default=0.01)
args = parser.parse_args()

L1 = args.lang_l1
L2 = args.lang_l2
PAIR = f"{L1}-{L2}"

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = f"{HF_USER}/{PAIR}-bilingual-5B"
PHASE1_FINAL_REPO = f"{HF_USER}/{L1}-namefinal-phase-1"
TOKENIZER_ID = f"{HF_USER}/tokenizer-{PAIR}"

SEQ_LEN = args.seq_len
TOTAL_TOKENS = args.total_tokens
BATCH = args.batch_size
GRAD_ACCUM = args.grad_accum
LR_MAX = args.lr
WARMUP_FRAC = args.warmup_frac

PHASE_2_START = TOTAL_TOKENS // 2
CHECKPOINT_DIR = Path(f"checkpoints_{PAIR}")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_FILE = Path(f"{PAIR}_training.log")

DATASETS = {
    "l1": f"{HF_USER}/{L1}-{L2}-5B",
    "l2": f"{HF_USER}/{L2}-{L1}-5B",
}

# Curriculum Targets
P1_TARGETS = [0.0, 0.125, 0.25]
P2_CURRICULUM = {
    0.625: f"{PAIR}-l1-{L1}-l2-{L2}-beginner",
    0.75:  f"{PAIR}-l1-{L1}-l2-{L2}-intermediate",
    0.875: f"{PAIR}-l1-{L1}-l2-{L2}-advanced",
    1.0:   f"{PAIR}-l1-{L1}-l2-{L2}-fluent"
}
ALL_CHECKPOINTS = sorted(set(P1_TARGETS + [0.5] + list(P2_CURRICULUM.keys())))
saved_checkpoints = set()

# ------------------------------------------------
# DISTRIBUTED SETUP
# ------------------------------------------------
def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world > 1:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=2))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world, local_rank, device

# ------------------------------------------------
# LOGGING
# ------------------------------------------------
def log(rank, msg):
    if rank != 0: return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{PAIR}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ------------------------------------------------
# STREAMING DATASET
# ------------------------------------------------
class StreamingDataset(IterableDataset):
    def __init__(self, repo, rank, world):
        self.repo = repo
        self.rank = rank
        self.world = world

    def __iter__(self):
        ds = load_dataset(self.repo, split="train", streaming=True)
        ds = ds.shuffle(buffer_size=10000, seed=42)
        if self.world > 1:
            ds = ds.shard(self.world, self.rank)
        for row in ds:
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            if ids.shape[0] == SEQ_LEN:
                yield ids

def infinite_loader(repo, rank, world):
    dataset = StreamingDataset(repo, rank, world)
    loader = DataLoader(dataset, batch_size=BATCH, num_workers=0, pin_memory=True)
    while True:
        for batch in loader:
            yield batch

# ------------------------------------------------
# CHECKPOINT UTILS
# ------------------------------------------------
def latest_checkpoint():
    ckpts = glob.glob(str(CHECKPOINT_DIR / f"{PAIR}-bilingual-*"))
    if not ckpts: return None
    return Path(max(ckpts, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0))

def save_and_upload_checkpoint(rank, model, tokenizer, optimizer, step, tokens_seen, cp_frac):
    dist.barrier()
    if rank != 0:
        dist.barrier()
        return

    pct = int(cp_frac * 100)
    folder_name = f"{PAIR}-bilingual-{pct}"
    cp_path = CHECKPOINT_DIR / folder_name
    cp_path.mkdir(parents=True, exist_ok=True)

    # Local Save
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(cp_path, safe_serialization=False)
    tokenizer.save_pretrained(cp_path)
    torch.save({
        "step": step, 
        "tokens": tokens_seen,
        "optimizer": optimizer.state_dict()
    }, cp_path / "optimizer.pt")

    log(rank, f"💾 Checkpoint {pct}% saved locally.")

    if HF_TOKEN:
        try:
            # 1. Main Bilingual Rolling Repo
            create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)
            upload_folder(folder_path=str(cp_path), repo_id=REPO_ID, token=HF_TOKEN)
            
            # 2. End of Phase 1 (50% mark)
            if math.isclose(cp_frac, 0.5):
                log(rank, f"🌟 UPLOADING PHASE 1 FINAL TO: {PHASE1_FINAL_REPO}")
                create_repo(PHASE1_FINAL_REPO, token=HF_TOKEN, exist_ok=True)
                upload_folder(folder_path=str(cp_path), repo_id=PHASE1_FINAL_REPO, token=HF_TOKEN)

            # 3. Dedicated Curriculum Repos
            if cp_frac in P2_CURRICULUM:
                repo_name = P2_CURRICULUM[cp_frac]
                target_repo = f"{HF_USER}/{repo_name}"
                log(rank, f"🚀 Pushing to curriculum repo: {target_repo}")
                create_repo(target_repo, token=HF_TOKEN, exist_ok=True)
                upload_folder(folder_path=str(cp_path), repo_id=target_repo, token=HF_TOKEN)

        except Exception as e:
            log(rank, f"⚠️ Upload failed: {e}")
    
    dist.barrier()

# ------------------------------------------------
# TRAIN
# ------------------------------------------------
def train():
    rank, world, local_rank, device = setup_distributed()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    config = GPT2Config(
        vocab_size=50000, n_positions=SEQ_LEN, n_embd=896,
        n_layer=24, n_head=14, bos_token_id=0, eos_token_id=1, pad_token_id=2
    )

    step, tokens_seen = 0, 0
    cp = latest_checkpoint()

    if cp:
        log(rank, f"🔄 Resuming from {cp}")
        model = GPT2LMHeadModel.from_pretrained(cp).to(device)
        state = torch.load(cp / "optimizer.pt", map_location=device)
        step, tokens_seen = state["step"], state["tokens"]
        # Determine which checkpoints were already passed
        global saved_checkpoints
        for f in ALL_CHECKPOINTS:
            if (tokens_seen / TOTAL_TOKENS) >= f:
                saved_checkpoints.add(f)
    else:
        model = GPT2LMHeadModel(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)
    if cp: optimizer.load_state_dict(state["optimizer"])

    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    l1_loader = infinite_loader(DATASETS["l1"], rank, world)
    l2_loader = infinite_loader(DATASETS["l2"], rank, world)

    tokens_per_step = SEQ_LEN * BATCH * world * GRAD_ACCUM
    warmup_tokens = TOTAL_TOKENS * WARMUP_FRAC
    model.train()

    while tokens_seen < TOTAL_TOKENS:
        optimizer.zero_grad(set_to_none=True)

        # Logic: Phase 1 (10:1 ratio) vs Phase 2 (2:1 ratio)
        if tokens_seen < PHASE_2_START:
            loader = l1_loader if step % 10 != 0 else l2_loader
        else:
            loader = l1_loader if step % 3 == 0 else l2_loader

        for micro in range(GRAD_ACCUM):
            ctx = (model.no_sync() if isinstance(model, DDP) and micro < GRAD_ACCUM - 1 else contextlib.nullcontext())
            with ctx:
                batch = next(loader).to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch, labels=batch).loss / GRAD_ACCUM
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

        if rank == 0 and step % 20 == 0:
            log(rank, f"step {step} | {tokens_seen/1e9:.3f}B tokens | lr {lr:.2e}")

        # --- Checkpoint Trigger ---
        fraction = tokens_seen / TOTAL_TOKENS
        for cp_frac in ALL_CHECKPOINTS:
            if fraction >= cp_frac and cp_frac not in saved_checkpoints:
                save_and_upload_checkpoint(rank, model, tokenizer, optimizer, step, tokens_seen, cp_frac)
                saved_checkpoints.add(cp_frac)

    log(rank, "✅ Training complete")
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    train()
