#!/usr/bin/env python3
"""
End-to-End Bilingual Pretraining with Pre-Tokenized Datasets

Single-language CLI mode: --lang LANG
DDP-ready for multi-GPU (torchrun)
"""

import os, time, math, json
from pathlib import Path

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder

# ==========================================================
# CONFIG
# ==========================================================

HF_USER = "RA-ALTA"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEQ_LEN = 512
TOTAL_TOKENS = 5_000_000_000

BASE_CHECKPOINTS = [0.25, 0.5, 0.75, 1.0]
LOG_CHECKPOINTS = [2 ** i / 100 for i in range(0, 7)]
CHECKPOINT_PERCENTAGES = sorted(set(BASE_CHECKPOINTS + LOG_CHECKPOINTS))

GRAD_ACCUM_STEPS = 8
LR_MAX = 2e-4
WARMUP_FRACTION = 0.01

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

VOCAB_SIZE = 50_000  # large enough to cover all token IDs

# Pre-tokenized dataset paths
DATASETS = {
    "zh": "RA-ALTA/zh-5B",
    "en": "RA-ALTA/en-5B",
    "es": "RA-ALTA/es-5B",
    "de": "RA-ALTA/de-5B",
    "pl": "RA-ALTA/pl-5B",
    "fr": "RA-ALTA/fr-5B",
    "ar": "RA-ALTA/ar-5B",
    "tr": "RA-ALTA/tr-5B"
}

MODEL_REPOS = {
    "zh": "RA-ALTA/zh-bilingual-5B",
    "es": "RA-ALTA/es-bilingual-5B",
    "de": "RA-ALTA/de-bilingual-5B",
    "pl": "RA-ALTA/pl-bilingual-5B",
    "fr": "RA-ALTA/fr-bilingual-5B",
    "ar": "RA-ALTA/ar-bilingual-5B",
    "tr": "RA-ALTA/tr-bilingual-5B"
}

# ==========================================================
# LOGGING
# ==========================================================

def log(lang, msg, local_rank=0):
    """Only rank 0 writes logs to avoid duplication"""
    if local_rank != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR / f"{lang}-training.log", "a") as f:
        f.write(line + "\n")

# ==========================================================
# HELPERS
# ==========================================================

def compute_phase_tokens(total):
    return total // 2, total - total // 2

def cosine_lr(progress):
    return 0.5 * (1 + math.cos(math.pi * progress))

def stream_dataset(repo):
    ds = load_dataset(repo, split="train", streaming=True)
    block_idx = 0
    for item in ds:
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        if ids.size(0) == SEQ_LEN:
            block_idx += 1
            yield ids.unsqueeze(0), block_idx

def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f)

def load_state(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None

def push_model(repo_id, cp_path, local_rank=0):
    """Only rank 0 pushes to HF hub"""
    if local_rank != 0:
        return
    for attempt in range(5):
        try:
            create_repo(repo_id, exist_ok=True, token=HF_TOKEN)
            upload_folder(
                folder_path=str(cp_path),
                repo_id=repo_id,
                token=HF_TOKEN
            )
            log(cp_path.name.split('-')[0], f"✅ Pushed checkpoint to {repo_id}", local_rank)
            return
        except Exception as e:
            log(cp_path.name.split('-')[0], f"⚠️ Failed push attempt {attempt+1} to {repo_id}: {e}", local_rank)
            time.sleep(60)
    log(cp_path.name.split('-')[0], f"❌ Giving up on pushing {cp_path}", local_rank)

# ==========================================================
# TRAINING LOOP
# ==========================================================

def train_pair(l1, l2="en"):
    """DDP-aware training loop"""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    phase1_tokens, phase2_tokens = compute_phase_tokens(TOTAL_TOKENS)
    total_training_tokens = TOTAL_TOKENS
    warmup_tokens = int(total_training_tokens * WARMUP_FRACTION)

    # Model + DDP
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_positions=SEQ_LEN,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2
        )
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX)
    scaler = torch.cuda.amp.GradScaler()

    # Resume state
    state_path = CHECKPOINT_DIR / f"{l1}-state.json"
    resume = load_state(state_path)
    tokens_global = tokens_l1 = tokens_l2 = next_cp_idx = 0
    if resume:
        log(l1, "Resuming from checkpoint", local_rank)
        model.module.load_state_dict(torch.load(resume["model_path"], map_location=device))
        optimizer.load_state_dict(torch.load(resume["optim_path"], map_location=device))
        tokens_global = resume["tokens_global"]
        tokens_l1 = resume["tokens_l1"]
        tokens_l2 = resume["tokens_l2"]
        next_cp_idx = resume["next_cp_idx"]

    # Push old checkpoints
    for cp in CHECKPOINT_DIR.glob(f"{l1}-*"):
        if cp.is_dir():
            push_model(MODEL_REPOS[l1], cp, local_rank)

    start_time = time.time()
    accum = 0

    # ----------------------
    # Phase 1: 100% L1
    # ----------------------
    if tokens_l1 < phase1_tokens:
        for block, shard_idx in stream_dataset(DATASETS[l1]):
            block = block.to(device)
            block_tokens = block.numel()
            if tokens_l1 + block_tokens > phase1_tokens:
                block = block[:, :phase1_tokens - tokens_l1]
                block_tokens = block.numel()
            try:
                with torch.cuda.amp.autocast():
                    loss = model(block, labels=block).loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                accum += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log(l1, "⚠️ OOM — skipping batch", local_rank)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            if accum >= GRAD_ACCUM_STEPS:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
            tokens_l1 += block_tokens
            tokens_global += block_tokens
            elapsed = time.time() - start_time
            tok_sec = tokens_global / max(elapsed, 1)
            log(l1, f"Shard:{shard_idx} | {tokens_global/1e9:.2f}B tok | {tok_sec:,.0f} tok/s | LR {LR_MAX:.2e}", local_rank)
            if tokens_l1 >= phase1_tokens:
                break
    log(l1, "Phase 1 complete", local_rank)

    # ----------------------
    # Phase 2: Interleaved L1:L2
    # ----------------------
    total_l2_tokens = int(phase2_tokens * 2/3)
    total_l1_tokens = phase2_tokens - total_l2_tokens
    l1_gen = stream_dataset(DATASETS[l1])
    l2_gen = stream_dataset(DATASETS[l2])

    while tokens_l1 < total_l1_tokens or tokens_l2 < total_l2_tokens:
        for source, tag in [(l1_gen,"L1"), (l2_gen,"L2"), (l2_gen,"L2")]:
            if tag == "L1" and tokens_l1 >= total_l1_tokens: continue
            if tag == "L2" and tokens_l2 >= total_l2_tokens: continue
            try:
                block, shard_idx = next(source)
            except StopIteration:
                continue
            block = block.to(device)
            block_tokens = block.numel()
            try:
                with torch.cuda.amp.autocast():
                    loss = model(block, labels=block).loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                accum += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log(l1, "⚠️ OOM — skipping batch", local_rank)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            if tag=="L1": tokens_l1 += block_tokens
            else: tokens_l2 += block_tokens
            tokens_global += block_tokens

            if accum >= GRAD_ACCUM_STEPS:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            # Dynamic LR
            if tokens_global < warmup_tokens:
                lr = LR_MAX * (tokens_global / warmup_tokens)
            else:
                progress = (tokens_global - warmup_tokens) / (total_training_tokens - warmup_tokens)
                lr = LR_MAX * cosine_lr(progress)
            for g in optimizer.param_groups: g["lr"] = lr

            # Logging
            elapsed = time.time() - start_time
            tok_sec = tokens_global / max(elapsed, 1)
            log(l1, f"Shard:{shard_idx} | {tokens_global/1e9:.2f}B tok | {tok_sec:,.0f} tok/s | LR {lr:.2e}", local_rank)

            # Logarithmic checkpointing
            if next_cp_idx < len(CHECKPOINT_PERCENTAGES):
                target = CHECKPOINT_PERCENTAGES[next_cp_idx]*total_l2_tokens
                if tokens_l2 >= target:
                    cp_name = f"{l1}-en-{int(CHECKPOINT_PERCENTAGES[next_cp_idx]*100)}"
                    cp_path = CHECKPOINT_DIR / cp_name
                    cp_path.mkdir(parents=True, exist_ok=True)
                    if local_rank == 0:
                        torch.save(model.module.state_dict(), cp_path/"model.pt")
                        torch.save(optimizer.state_dict(), cp_path/"optim.pt")
                        save_state(state_path,{
                            "model_path": str(cp_path/"model.pt"),
                            "optim_path": str(cp_path/"optim.pt"),
                            "tokens_global": tokens_global,
                            "tokens_l1": tokens_l1,
                            "tokens_l2": tokens_l2,
                            "next_cp_idx": next_cp_idx+1
                        })
                        push_model(MODEL_REPOS[l1], cp_path, local_rank)
                    next_cp_idx += 1

    # Final model push
    if local_rank == 0:
        final_path = CHECKPOINT_DIR / f"{l1}-final"
        final_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), final_path/"model.pt")
        push_model(MODEL_REPOS[l1], final_path, local_rank)
        log(l1, "Training complete! Final model pushed.", local_rank)

# ==========================================================
# ENTRYPOINT
# ==========================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code for training")
    args = parser.parse_args()

    # Initialize DDP
    torch.distributed.init_process_group(backend="nccl")
    train_pair(args.lang)

if __name__ == "__main__":
    main()
