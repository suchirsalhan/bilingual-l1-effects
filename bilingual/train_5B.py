#!/usr/bin/env python3
import os, json, math, logging
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Config
import torch.multiprocessing as mp
from huggingface_hub import HfApi, create_repo, upload_folder

# ================= CONFIG =================
HF_USER = "RA-ALTA"
SAVE_DIR = Path("./runs_bilingual_5B")
STATE_FILE = "trainer_state.json"

SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2
TOTAL_TOKENS = 5_000_000_000

# Curriculum
PHASE1_FRACTION = 0.5       # Phase 1 = L1 only
PHASE2_FRACTION = 0.5       # Phase 2 = interleaved
L1_RATIO_PHASE2 = 1/3       # Phase 2 L1:L2 ratio
L2_RATIO_PHASE2 = 2/3
CHECKPOINT_FRACS = [0.25, 0.5, 0.75, 1.0]

# Languages
L1_LANGS = ["es","fr","de","pl","tr","ar","zh"]
L2_LANG = "en"

# ================= LOGGING =================
def setup_logging(is_master, run_dir):
    if not is_master:
        logging.disable(logging.CRITICAL)
        return
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "training.log"),
            logging.StreamHandler()
        ]
    )

# ================= STATE =================
def default_state():
    return {"phase":1, "l1_seen":0, "l2_seen":0, "checkpoint_idx":0}

def load_state(run_dir):
    path = run_dir / STATE_FILE
    return json.load(open(path)) if path.exists() else default_state()

def save_state(run_dir, state):
    with open(run_dir / STATE_FILE, "w") as f:
        json.dump(state, f)

# ================= DATA HELPERS =================
def get_dataset_name(lang):
    return f"{HF_USER}/{lang}-5B"

def arrow_shard_iterator(dataset_name, max_tokens=None, is_master=True):
    """
    Stream tokens from a Hugging Face dataset while tracking shards consumed.
    Logs every time we pass approx one shard worth of tokens.
    """
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    # Estimate tokens per shard (assuming 650 shards)
    ds_size = 5_000_000_000  # total tokens in dataset (approx)
    num_shards = 650
    tokens_per_shard = ds_size // num_shards

    tokens_seen = 0
    shard_count = 0
    shard_tokens_seen = 0

    for row in ds:
        token_ids = row["input_ids"]
        if max_tokens is not None and tokens_seen + len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens - tokens_seen]
        tokens_seen += len(token_ids)
        shard_tokens_seen += len(token_ids)
        yield token_ids

        if shard_tokens_seen >= tokens_per_shard:
            shard_count += 1
            shard_tokens_seen = 0
            if is_master:
                logging.info(f"[{dataset_name}] Shards processed: {shard_count}/{num_shards} | Tokens seen: {tokens_seen:,}")
        
        if max_tokens is not None and tokens_seen >= max_tokens:
            if is_master:
                logging.info(f"[{dataset_name}] Reached max_tokens limit: {tokens_seen:,}")
            break

def interleave_stream(l1_stream, l2_stream, l1_ratio=1, l2_ratio=2):
    while True:
        for _ in range(l1_ratio):
            try:
                yield next(l1_stream), "L1"
            except StopIteration:
                return
        for _ in range(l2_ratio):
            try:
                yield next(l2_stream), "L2"
            except StopIteration:
                return

# ================= TRAIN STEP =================
def train_step(model, opt, batch, device, step):
    x = torch.tensor(batch, device=device)[:, :-1]
    y = torch.tensor(batch, device=device)[:, 1:]

    with autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, labels=y).loss / GRAD_ACCUM
    loss.backward()

    if (step + 1) % GRAD_ACCUM == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    return loss.item() * GRAD_ACCUM

# ================= HUGGINGFACE PUSH =================
def push_checkpoint_to_hf(lang, checkpoint_dir):
    repo_name = f"bilingual-{lang}-gpt2-5B"
    try:
        create_repo(repo_id=f"{HF_USER}/{repo_name}", exist_ok=True)
    except Exception:
        pass
    upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=f"{HF_USER}/{repo_name}",
        repo_type="model",
        path_in_repo=checkpoint_dir.name,
        ignore_patterns=["*.lock", "*.tmp"]
    )

# ================= TRAINING =================
def train_language(local_rank, world_size, lang):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    is_master = local_rank == 0
    run_dir = SAVE_DIR / lang
    setup_logging(is_master, run_dir)
    device = f"cuda:{local_rank}"

    # MODEL
    vocab_size = 50000
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    # STATE
    state = load_state(run_dir)

    # PHASE 1: L1 only
    phase1_tokens = int(TOTAL_TOKENS * PHASE1_FRACTION)
    l1_dataset_name = get_dataset_name(lang)
    if state["phase"] == 1:
        if is_master: logging.info(f"[{lang}] Phase 1: L1 only")
        l1_stream = arrow_shard_iterator(l1_dataset_name, max_tokens=phase1_tokens, is_master=is_master)
        step = 0
        tokens_seen = 0
        while True:
            batch = []
            try:
                for _ in range(BATCH_SIZE):
                    batch += next(l1_stream)[:SEQ_LEN]
            except StopIteration:
                break
            if not batch:
                break
            train_step(model, optimizer, [batch], device, step)
            tokens_seen += len(batch)
            step += 1
            state["l1_seen"] = tokens_seen
            # frequent checkpointing every 100k tokens
            if is_master and tokens_seen % 100_000 == 0:
                ckpt_path = run_dir / f"phase1_tokens_{tokens_seen//1000}k"
                ckpt_path.mkdir(exist_ok=True)
                model.module.save_pretrained(ckpt_path)
                push_checkpoint_to_hf(lang, ckpt_path)
                logging.info(f"[{lang}] Phase1 checkpoint at {tokens_seen} tokens")
            if is_master and step % 50 == 0:
                logging.info(f"[{lang}] Phase1 Step {step} | Tokens seen {tokens_seen:,}")
        state["phase"] = 2
        if is_master: save_state(run_dir, state)

    # PHASE 2: Interleaved L1:L2
    phase2_tokens = int(TOTAL_TOKENS * PHASE2_FRACTION)
    l1_tokens_phase2 = int(phase2_tokens * L1_RATIO_PHASE2)
    l2_tokens_phase2 = int(phase2_tokens * L2_RATIO_PHASE2)

    l1_dataset_name = get_dataset_name(lang)
    l2_dataset_name = get_dataset_name(L2_LANG)

    if is_master: logging.info(f"[{lang}] Phase 2: Interleaved L1:L2 1:2")
    l1_stream = arrow_shard_iterator(l1_dataset_name, max_tokens=l1_tokens_phase2, is_master=is_master)
    l2_stream = arrow_shard_iterator(l2_dataset_name, max_tokens=l2_tokens_phase2, is_master=is_master)
    interleaved = interleave_stream(l1_stream, l2_stream, l1_ratio=1, l2_ratio=2)

    step = 0
    state["checkpoint_idx"] = 0
    checkpoint_tokens = [int(frac * l2_tokens_phase2) for frac in CHECKPOINT_FRACS]

    l1_seen = state.get("l1_seen",0)
    l2_seen = state.get("l2_seen",0)

    for token_batch, tag in interleaved:
        batch = [token_batch[:SEQ_LEN] for _ in range(BATCH_SIZE)]
        train_step(model, optimizer, batch, device, step)
        if tag=="L2": l2_seen += BATCH_SIZE*SEQ_LEN
        else: l1_seen += BATCH_SIZE*SEQ_LEN
        step += 1

        # regular checkpointing every 100k tokens
        total_seen = l1_seen + l2_seen
        if is_master and total_seen % 100_000 == 0:
            ckpt_path = run_dir / f"phase2_tokens_{total_seen//1000}k"
            ckpt_path.mkdir(exist_ok=True)
            model.module.save_pretrained(ckpt_path)
            push_checkpoint_to_hf(lang, ckpt_path)
            logging.info(f"[{lang}] Phase2 checkpoint at {total_seen} tokens")

        # curriculum-based checkpoints
        if state["checkpoint_idx"] < len(checkpoint_tokens) and l2_seen >= checkpoint_tokens[state["checkpoint_idx"]]:
            pct = int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
            path = run_dir / f"checkpoint_{pct}"
            if is_master:
                path.mkdir(exist_ok=True)
                model.module.save_pretrained(path)
                push_checkpoint_to_hf(lang, path)
                logging.info(f"[{lang}] Saved checkpoint {pct}% L2 coverage")
            state["checkpoint_idx"] += 1

    dist.destroy_process_group()
    if is_master: logging.info(f"[{lang}] Training complete.")

# ================= MAIN =================
if __name__ == "__main__":
    world_size = min(len(L1_LANGS), torch.cuda.device_count())
    processes = []
    for local_rank, lang in enumerate(L1_LANGS):
        p = mp.Process(target=train_language, args=(local_rank, world_size, lang))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
