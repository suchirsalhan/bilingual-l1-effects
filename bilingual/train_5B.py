#!/usr/bin/env python3
"""
End-to-End Bilingual Pretraining with Pre-Tokenized Datasets

Features:
- Uses pre-tokenized datasets only
- Gradient accumulation
- Dynamic LR with warmup
- Logarithmic + fixed checkpointing
- Resume from checkpoint
- OOM recovery
- Shard-level logging
- HuggingFace branch pushes for intermediate checkpoints
- Final model pushed to main branch
- Persistent timestamped logs per language pair
"""

import os, time, math, json, subprocess
from pathlib import Path

import torch
from torch import optim
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
LOG_CHECKPOINTS = [2 ** i / 100 for i in range(0, 7)]  # 1%,2%,4%,8%,16%,32%,64%
CHECKPOINT_PERCENTAGES = sorted(set(BASE_CHECKPOINTS + LOG_CHECKPOINTS))

GRAD_ACCUM_STEPS = 8
LR_MAX = 2e-4
WARMUP_FRACTION = 0.01

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
STATE_FILE = "training_state.json"

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
    "ar": "RA-ALTA/ar-5B"
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

def log(l1, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR / f"{l1}-training.log", "a") as f:
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

def push_model(repo_id, cp_path):
    for attempt in range(5):
        try:
            create_repo(repo_id, exist_ok=True, token=HF_TOKEN)
            upload_folder(
                folder_path=str(cp_path),
                repo_id=repo_id,
                token=HF_TOKEN
            )
            log(cp_path.name.split('-')[0], f"✅ Pushed checkpoint to {repo_id}")
            return
        except Exception as e:
            log(cp_path.name.split('-')[0], f"⚠️ Failed push attempt {attempt+1} to {repo_id}: {e}")
            time.sleep(60)
    log(cp_path.name.split('-')[0], f"❌ Giving up on pushing {cp_path}")

# ==========================================================
# TRAINING LOOP
# ==========================================================

def train_pair(l1, l2, gpu):

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    phase1_tokens, phase2_tokens = compute_phase_tokens(TOTAL_TOKENS)
    total_training_tokens = TOTAL_TOKENS
    warmup_tokens = int(total_training_tokens * WARMUP_FRACTION)

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
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX)
    scaler = torch.cuda.amp.GradScaler()

    # Resume state if exists
    state_path = CHECKPOINT_DIR / f"{l1}-state.json"
    resume = load_state(state_path)
    tokens_global = tokens_l1 = tokens_l2 = next_cp_idx = 0
    if resume:
        log(l1, f"Resuming from checkpoint")
        model.load_state_dict(torch.load(resume["model_path"]))
        optimizer.load_state_dict(torch.load(resume["optim_path"]))
        tokens_global = resume["tokens_global"]
        tokens_l1 = resume["tokens_l1"]
        tokens_l2 = resume["tokens_l2"]
        next_cp_idx = resume["next_cp_idx"]

    # Push any old unpushed checkpoints
    for cp in CHECKPOINT_DIR.glob(f"{l1}-*"):
        if cp.is_dir():
            push_model(MODEL_REPOS[l1], cp)

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
                    log(l1, "⚠️ OOM — skipping batch")
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
            log(l1, f"Shard:{shard_idx} | {tokens_global/1e9:.2f}B tok | {tok_sec:,.0f} tok/s | LR {LR_MAX:.2e}")
            if tokens_l1 >= phase1_tokens:
                break
    log(l1, "Phase 1 complete")

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
                    log(l1, "⚠️ OOM — skipping batch")
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
            log(l1, f"Shard:{shard_idx} | {tokens_global/1e9:.2f}B tok | {tok_sec:,.0f} tok/s | LR {lr:.2e}")

            # Logarithmic checkpointing
            if next_cp_idx < len(CHECKPOINT_PERCENTAGES):
                target = CHECKPOINT_PERCENTAGES[next_cp_idx]*total_l2_tokens
                if tokens_l2 >= target:
                    cp_name = f"{l1}-en-{int(CHECKPOINT_PERCENTAGES[next_cp_idx]*100)}"
                    cp_path = CHECKPOINT_DIR / cp_name
                    cp_path.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), cp_path/"model.pt")
                    torch.save(optimizer.state_dict(), cp_path/"optim.pt")
                    save_state(state_path,{
                        "model_path": str(cp_path/"model.pt"),
                        "optim_path": str(cp_path/"optim.pt"),
                        "tokens_global": tokens_global,
                        "tokens_l1": tokens_l1,
                        "tokens_l2": tokens_l2,
                        "next_cp_idx": next_cp_idx+1
                    })
                    push_model(MODEL_REPOS[l1], cp_path)
                    next_cp_idx += 1

    # Final model push to main branch
    final_path = CHECKPOINT_DIR / f"{l1}-final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path/"model.pt")
    push_model(MODEL_REPOS[l1], final_path)
    log(l1, "Training complete! Final model pushed.")

# ==========================================================
# ORCHESTRATOR
# ==========================================================

def launch(l1,l2,gpu):
    return subprocess.Popen([
        "python", __file__,
        "--train_l1", l1,
        "--train_l2", l2,
        "--gpu", str(gpu)
    ])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_l1")
    parser.add_argument("--train_l2", default="en")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.train_l1:
        train_pair(args.train_l1, args.train_l2, args.gpu)
        return

    # Launch all bilingual pairs in parallel
    pairs = [("zh","en"),("es","en"),("de","en"),
             ("pl","en"),("fr","en"),("ar","en"),("tr","en")]

    procs = []
    for i,(l1,l2) in enumerate(pairs):
        gpu = i % torch.cuda.device_count()
        log(l1, f"[INFO] Launching {l1}-en on GPU {gpu}")
        procs.append(launch(l1,l2,gpu))

    for p in procs:
        p.wait()
    print("[INFO] All runs finished")

if __name__ == "__main__":
    main()
