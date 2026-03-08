#!/usr/bin/env python3
import os
import math
import time
import glob
import argparse
from pathlib import Path
import datetime
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from torch import optim

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import upload_folder, create_repo


# ==========================================================
# ARGUMENTS
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument("--lang_l1", required=True)
parser.add_argument("--lang_l2", default="en")

parser.add_argument("--total_tokens", type=int, default=5_000_000_000)
parser.add_argument("--seq_len", type=int, default=512)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=8)

parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_frac", type=float, default=0.01)

args, _ = parser.parse_known_args()

L1 = args.lang_l1
L2 = args.lang_l2
PAIR = f"{L1}-{L2}"

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_ID = f"{HF_USER}/{PAIR}-bilingual-5B"
TOKENIZER_ID = f"{HF_USER}/tokenizer-{PAIR}"

SEQ_LEN = args.seq_len
TOTAL_TOKENS = args.total_tokens

BATCH = args.batch_size
GRAD_ACCUM = args.grad_accum
LR_MAX = args.lr
WARMUP_FRAC = args.warmup_frac

CHECKPOINT_DIR = Path(f"checkpoints_{PAIR}")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_FILE = Path(f"{PAIR}_training.log")

DATASETS = {
    "l1": f"{HF_USER}/{L1}-{L2}-5B",
    "l2": f"{HF_USER}/{L2}-{L1}-5B",
}


# ==========================================================
# DISTRIBUTED SETUP
# ==========================================================

def setup_distributed():

    if "RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2)
        )

        rank = dist.get_rank()
        world = dist.get_world_size()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    else:

        rank = 0
        world = 1
        local_rank = 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    return rank, world, local_rank, device


# ==========================================================
# LOGGING
# ==========================================================

def log(rank, msg):

    if rank != 0:
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{PAIR}] {msg}"

    print(line, flush=True)

    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ==========================================================
# DATASET
# ==========================================================

class StreamingDataset(IterableDataset):

    def __init__(self, repo, rank, world):

        self.repo = repo
        self.rank = rank
        self.world = world

    def __iter__(self):

        ds = load_dataset(
            self.repo,
            split="train",
            streaming=True
        )

        ds = ds.shuffle(buffer_size=10_000, seed=42)

        ds = ds.shard(
            num_shards=self.world,
            index=self.rank,
            contiguous=True
        )

        for row in ds:

            ids = torch.tensor(row["input_ids"], dtype=torch.long)

            if ids.shape[0] == SEQ_LEN:
                yield ids


def infinite_loader(repo, rank, world):

    dataset = StreamingDataset(repo, rank, world)

    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        num_workers=0,
        pin_memory=True
    )

    while True:
        for batch in loader:
            yield batch


# ==========================================================
# CHECKPOINT UTILS
# ==========================================================

def latest_checkpoint():

    ckpts = glob.glob(str(CHECKPOINT_DIR / f"{PAIR}-step-*"))

    if not ckpts:
        return None

    return max(ckpts, key=lambda x: int(x.split("-")[-1]))


def save_checkpoint(rank, world, model, tokenizer, optimizer, step, tokens):

    # Barrier ensures all ranks finish the current step before rank 0 writes
    if world > 1:
        dist.barrier()

    if rank != 0:
        return

    folder = CHECKPOINT_DIR / f"{PAIR}-step-{step}"
    folder.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.save_pretrained(folder)
    tokenizer.save_pretrained(folder)

    torch.save(
        {
            "step": step,
            "tokens": tokens,
            "optimizer": optimizer.state_dict(),
        },
        folder / "optimizer.pt"
    )

    log(rank, f"checkpoint saved: step {step}")


# ==========================================================
# TRAIN
# ==========================================================

def train():

    rank, world, local_rank, device = setup_distributed()

    print("setup done")
    print(rank)
    quit()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    config = GPT2Config(
        vocab_size=50000,
        n_positions=SEQ_LEN,
        n_embd=896,
        n_layer=24,
        n_head=14,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2
    )

    step = 0
    tokens_seen = 0

    cp = latest_checkpoint()

    if cp:
        log(rank, f"resuming {cp}")
        model = GPT2LMHeadModel.from_pretrained(cp).to(device)
    else:
        model = GPT2LMHeadModel(config).to(device)

    # FIX 1: Wrap in DDP BEFORE creating optimizer so parameter references are stable
    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR_MAX,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # FIX 2: Load optimizer state AFTER DDP wrap so param groups match
    if cp:
        state = torch.load(Path(cp) / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        tokens_seen = state["tokens"]

    l1_loader = infinite_loader(DATASETS["l1"], rank, world)
    l2_loader = infinite_loader(DATASETS["l2"], rank, world)

    tokens_per_step = SEQ_LEN * BATCH * world * GRAD_ACCUM
    warmup_tokens = TOTAL_TOKENS * WARMUP_FRAC

    model.train()

    log(rank, f"tokens/step = {tokens_per_step}")

    while tokens_seen < TOTAL_TOKENS:

        optimizer.zero_grad(set_to_none=True)

        loader = l1_loader if step % 10 != 0 else l2_loader

        for micro in range(GRAD_ACCUM):

            ctx = (
                model.no_sync()
                if isinstance(model, DDP) and micro < GRAD_ACCUM - 1
                else contextlib.nullcontext()
            )

            with ctx:

                batch = next(loader).to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16):

                    loss = model(batch, labels=batch).loss / GRAD_ACCUM

                loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        tokens_seen += tokens_per_step
        step += 1

        if tokens_seen < warmup_tokens:
            lr = LR_MAX * (tokens_seen / warmup_tokens)
        else:
            progress = (tokens_seen - warmup_tokens) / (TOTAL_TOKENS - warmup_tokens)
            lr = LR_MAX * 0.5 * (1 + math.cos(math.pi * progress))

        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % 20 == 0:
            log(rank, f"step {step} | {tokens_seen/1e9:.2f}B tokens | lr {lr:.2e}")

        if step % 500 == 0:
            # FIX 3: Pass world to save_checkpoint so barrier is called inside
            save_checkpoint(rank, world, model, tokenizer, optimizer, step, tokens_seen)

    save_checkpoint(rank, world, model, tokenizer, optimizer, step, tokens_seen)

    log(rank, "training complete")

    # FIX 4: Always clean up process group
    if dist.is_initialized():
        dist.destroy_process_group()


# ==========================================================

if __name__ == "__main__":
    # FIX 5: Wrap in try/finally so process group is cleaned up even on crash
    try:
        train()
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
