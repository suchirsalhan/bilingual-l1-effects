#!/usr/bin/env python3
import os, json, random, time, logging, argparse
import torch
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from huggingface_hub import HfApi, create_repo, upload_folder

# ============================================================
# CONFIG
# ============================================================

LR = 5e-5
DEVICE = "cuda"
BATCH_SIZE = 16  # FP16 + A100 → good speed/memory balance
TOTAL_TOKENS = 5_000_000_000  # smaller for <6h training per lang
PHASE1_TOKENS = TOTAL_TOKENS // 2
L2_TOTAL = int(TOTAL_TOKENS * (1/3))
L1_PHASE2 = int(TOTAL_TOKENS * (1/6))

CHECKPOINT_FRACS = [0.25, 0.5, 0.75, 1.0]
CHECKPOINT_TOKENS = [int(L2_TOTAL*x) for x in CHECKPOINT_FRACS]

HF_USER = "suchirsalhan"
SAVE_DIR = "./runs"
STATE_FILE = "trainer_state.json"

LEVEL_NAMES = {0:"beginner",1:"intermediate",2:"advanced",3:"fluent"}
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SAVE_DIR, "training.log")),
        logging.StreamHandler()
    ]
)

# ============================================================
# TOKENIZER / MODEL
# ============================================================

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model_template = GPT2LMHeadModel.from_pretrained("gpt2")
MAX_POSITIONS = model_template.config.n_positions
SEQ_LEN = min(512, MAX_POSITIONS)
del model_template  # free memory

def load_or_create_model(run_dir):
    if os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        logging.info("Resuming model weights")
        model = GPT2LMHeadModel.from_pretrained(run_dir)
    else:
        logging.info("Initializing GPT2 base model")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.to(DEVICE)

# ============================================================
# TRAIN STATE
# ============================================================

def default_state():
    return dict(
        phase=1,
        l1_seen_phase1=0,
        l1_seen_phase2=0,
        l2_seen=0,
        checkpoint_idx=0,
        rng=random.getstate(),
        torch_rng=torch.get_rng_state().tolist(),
        cuda_rng=torch.cuda.get_rng_state().tolist()
    )

def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        logging.info("Loaded trainer state")
        return s
    return default_state()

def save_state(run_dir, state):
    state["rng"] = random.getstate()
    state["torch_rng"] = torch.get_rng_state().tolist()
    state["cuda_rng"] = torch.cuda.get_rng_state().tolist()
    with open(os.path.join(run_dir, STATE_FILE), "w") as f:
        json.dump(state, f)

# ============================================================
# DATA STREAMS (skip too long examples)
# ============================================================

def cultura_stream(lang, skip_tokens=0):
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    seen = 0
    for ex in ds:
        ids = tokenizer(ex["text"])["input_ids"]
        if len(ids) > MAX_POSITIONS:
            logging.warning(f"CulturaX example too long ({len(ids)}), skipping")
            continue
        if seen + len(ids) < skip_tokens:
            seen += len(ids)
            continue
        yield ids

def fineweb_stream(skip_tokens=0):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    seen = 0
    for ex in ds:
        ids = tokenizer(ex["text"])["input_ids"]
        if len(ids) > MAX_POSITIONS:
            logging.warning(f"FineWeb example too long ({len(ids)}), skipping")
            continue
        if seen + len(ids) < skip_tokens:
            seen += len(ids)
            continue
        yield ids

def pack_blocks(token_stream, seq_len=SEQ_LEN):
    buf = []
    for tokens in token_stream:
        buf.extend(tokens)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]

def batchify(stream, batch_size=BATCH_SIZE):
    batch = []
    for block in stream:
        batch.append(block)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# ============================================================
# HF PUSH
# ============================================================

def push_to_hub(path, lang, ckpt_idx):
    level = LEVEL_NAMES[ckpt_idx]
    repo_id = f"{HF_USER}/B-GPT-{lang}-fineweb-{level}"
    logging.info(f"Pushing checkpoint → {repo_id}")
    api = HfApi()
    create_repo(repo_id, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=path, commit_message=f"{level} learner checkpoint")

# ============================================================
# TRAIN STEP
# ============================================================

def step(model, optimizer, x_batch, y_batch, scaler):
    optimizer.zero_grad()
    loss_val = 0.0
    for x, y in zip(x_batch, y_batch):
        x = torch.tensor(x[:-1]).unsqueeze(0).to(DEVICE)
        y = torch.tensor(y[1:]).unsqueeze(0).to(DEVICE)
        with autocast():  # FP16
            loss = model(x, labels=y).loss
        scaler.scale(loss).backward()
        loss_val += loss.item()
    scaler.step(optimizer)
    scaler.update()
    return loss_val / len(x_batch)

# ============================================================
# TRAIN LOOP
# ============================================================

def train(lang):
    run_dir = os.path.join(SAVE_DIR, lang)
    os.makedirs(run_dir, exist_ok=True)
    model = load_or_create_model(run_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    state = load_state(run_dir)

    # restore RNG
    random.setstate(tuple(state["rng"]))
    torch.set_rng_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(state["cuda_rng"], dtype=torch.uint8))

    l1_stream = batchify(pack_blocks(cultura_stream(lang, state["l1_seen_phase1"]+state["l1_seen_phase2"])))
    l2_stream = batchify(pack_blocks(fineweb_stream(state["l2_seen"]))

    step_count = 0
    start_time = time.time()

    # ---------------- PHASE 1 ----------------
    if state["phase"] == 1:
        logging.info("Phase 1: L1 only")
        for batch in l1_stream:
            x_batch, y_batch = batch, batch
            loss_val = step(model, optimizer, x_batch, y_batch, scaler)
            step_count += 1
            state["l1_seen_phase1"] += sum(len(b) for b in batch)

            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Step {step_count} | Loss: {loss_val:.4f} | L1 tokens seen: {state['l1_seen_phase1']:,} | Elapsed: {elapsed/60:.2f} min")

            if state["l1_seen_phase1"] >= PHASE1_TOKENS:
                state["phase"] = 2
                save_state(run_dir, state)
                break
            if state["l1_seen_phase1"] % 50_000_000 == 0:
                save_state(run_dir, state)

    # ---------------- PHASE 2 ----------------
    logging.info("Phase 2: bilingual acquisition")
    l1_iter = iter(l1_stream)
    l2_iter = iter(l2_stream)

    while state["l2_seen"] < L2_TOTAL:
        for lang_id in ["l1","l2","l2"]:
            batch = next(l1_iter) if lang_id=="l1" else next(l2_iter)
            x_batch, y_batch = batch, batch
            loss_val = step(model, optimizer, x_batch, y_batch, scaler)
            step_count += 1

            if lang_id == "l2":
                state["l2_seen"] += sum(len(b) for b in batch)
            else:
                state["l1_seen_phase2"] += sum(len(b) for b in batch)

            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                logging.info(
                    f"Step {step_count} | Loss: {loss_val:.4f} | "
                    f"L1 seen phase2: {state['l1_seen_phase2']:,} | L2 seen: {state['l2_seen']:,} | "
                    f"ratio: {state['l1_seen_phase2']/max(1,state['l2_seen']):.3f} | "
                    f"Elapsed: {elapsed/60:.2f} min"
                )

            # CHECKPOINT
            while state["checkpoint_idx"] < len(CHECKPOINT_TOKENS) and state["l2_seen"] >= CHECKPOINT_TOKENS[state["checkpoint_idx"]]:
                pct = int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
                level = LEVEL_NAMES[state["checkpoint_idx"]]
                path = os.path.join(run_dir, f"checkpoint_{pct}")
                os.makedirs(path, exist_ok=True)
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                save_state(run_dir, state)
                logging.info(f"\n=== {level.upper()} CHECKPOINT ===")
                logging.info(f"L2 seen: {state['l2_seen']:,} | L1 seen: {state['l1_seen_phase2']:,}\n")
                push_to_hub(path, lang, state["checkpoint_idx"])
                state["checkpoint_idx"] += 1

        if state["l2_seen"] % 50_000_000 == 0:
            save_state(run_dir, state)

    logging.info("Training complete")

# ============================================================
# RUN
# ============================================================

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language to train")
    args = parser.parse_args()
    train(args.lang)

