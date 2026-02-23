#!/usr/bin/env python3
import os, json, logging, argparse
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    PreTrainedTokenizerFast,
)
from huggingface_hub import create_repo, upload_folder

# ================= CONFIG =================

LR = 6e-4
SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2

TOTAL_TOKENS = 5_000_000_000
PHASE1_TOKENS = TOTAL_TOKENS // 2
L2_TOTAL = int(TOTAL_TOKENS * (1/3))

CHECKPOINT_FRACS = [0.25, 0.5, 0.75, 1.0]

HF_USER = "suchirsalhan"
TOKENIZER_ORG = "RA-ALTA"

SAVE_DIR = "./runs_finewebedu"
TOKENIZER_DIR = "./tokenizer_cache"
STATE_FILE = "trainer_state.json"

# ================= DISTRIBUTED =================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=120),
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        return (
            True,
            local_rank,
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )

    return False, 0, 0, 1


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
            logging.FileHandler(os.path.join(run_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


# ================= STATE =================

def default_state():
    return {"phase": 1, "l1_seen": 0, "l2_seen": 0, "checkpoint_idx": 0}


def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    return json.load(open(path)) if os.path.exists(path) else default_state()


def save_state(run_dir, state):
    json.dump(state, open(os.path.join(run_dir, STATE_FILE), "w"))


# ================= TOKENIZER =================

def load_hf_tokenizer(lang, is_ddp, global_rank):
    repo_id = f"{TOKENIZER_ORG}/tokenizer-fineweb_{lang}"

    if global_rank == 0:
        logging.info(f"Downloading tokenizer {repo_id}")
        tok = PreTrainedTokenizerFast.from_pretrained(repo_id)
        tok.save_pretrained(TOKENIZER_DIR)

    if is_ddp:
        dist.barrier()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    if global_rank == 0:
        logging.info("Tokenizer ready.")

    return tokenizer


# ================= STREAMING DATA =================

def streaming_blocks(dataset, config, rank, world_size, tokenizer, skip_tokens=0):
    ds = (
        load_dataset(dataset, config, split="train", streaming=True)
        .shard(num_shards=world_size, index=rank)
    )

    buf, seen = [], 0

    for ex in ds:
        text = ex.get("text")
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)

        if seen + len(ids) < skip_tokens:
            seen += len(ids)
            continue

        buf.extend(ids)

        while len(buf) >= SEQ_LEN:
            yield buf[:SEQ_LEN]
            buf = buf[SEQ_LEN:]


class ResilientStreamingIterator:
    def __init__(self, dataset, config, rank, world_size, tokenizer, skip_tokens=0):
        self.args = (dataset, config, rank, world_size, tokenizer, skip_tokens)
        self.iterator = streaming_blocks(*self.args)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = streaming_blocks(*self.args)
            return next(self.iterator)


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


# ================= HUB =================

def push_to_hub(local_path, lang, idx):
    repo_name = f"B-GPT-{lang}-fineweb-{idx}"

    create_repo(repo_name, exist_ok=True, private=False)

    upload_folder(
        folder_path=local_path,
        repo_id=f"{HF_USER}/{repo_name}",
        repo_type="model",
    )


# ================= TRAIN LOOP =================

def train(lang):

    is_ddp, local_rank, global_rank, world_size = setup_distributed()
    is_master = global_rank == 0

    run_dir = os.path.join(SAVE_DIR, lang)
    setup_logging(is_master, run_dir)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # -------- Tokenizer --------
    tokenizer = load_hf_tokenizer(lang, is_ddp, global_rank)

    # -------- Model --------
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -------- Resume State --------
    state = load_state(run_dir)

    l1_iter = ResilientStreamingIterator(
        "uonlp/CulturaX",
        lang,
        global_rank,
        world_size,
        tokenizer,
        state["l1_seen"],
    )

    l2_iter = ResilientStreamingIterator(
        "HuggingFaceFW/fineweb-edu",
        "default",
        global_rank,
        world_size,
        tokenizer,
        state["l2_seen"],
    )

    tokens_per_batch = BATCH_SIZE * SEQ_LEN
    checkpoint_tokens = [int(L2_TOTAL * x) for x in CHECKPOINT_FRACS]

    step = 0

    # ================= PHASE 1 =================
    if state["phase"] == 1:
        if is_master:
            logging.info("Starting Phase 1")

        while (state["l1_seen"] * world_size) < PHASE1_TOKENS:

            batch = [next(l1_iter) for _ in range(BATCH_SIZE)]
            loss = train_step(model, optimizer, batch, device, step)

            state["l1_seen"] += tokens_per_batch
            step += 1

            if is_master and step % 50 == 0:
                logging.info(
                    f"P1 Step {step} | Tokens {state['l1_seen']*world_size} | Loss {loss:.3f}"
                )

        state["phase"] = 2
        if is_master:
            save_state(run_dir, state)

    # ================= PHASE 2 =================
    if is_master:
        logging.info("Starting Phase 2")

    while (state["l2_seen"] * world_size) < L2_TOTAL:

        for src in ["l1", "l2", "l2"]:

            iterator = l1_iter if src == "l1" else l2_iter
            batch = [next(iterator) for _ in range(BATCH_SIZE)]

            loss = train_step(model, optimizer, batch, device, step)

            if src == "l2":
                state["l2_seen"] += tokens_per_batch
            else:
                state["l1_seen"] += tokens_per_batch

            step += 1

            # -------- Checkpoints --------
            if (
                state["checkpoint_idx"] < len(checkpoint_tokens)
                and (state["l2_seen"] * world_size)
                >= checkpoint_tokens[state["checkpoint_idx"]]
            ):

                pct = int(CHECKPOINT_FRACS[state["checkpoint_idx"]] * 100)
                path = os.path.join(run_dir, f"checkpoint_{pct}")

                if is_master:
                    os.makedirs(path, exist_ok=True)

                    (model.module if is_ddp else model).save_pretrained(path)
                    tokenizer.save_pretrained(path)

                    save_state(run_dir, state)

                    logging.info(f"Saved checkpoint {pct}%")

                    try:
                        push_to_hub(path, lang, state["checkpoint_idx"])
                    except Exception as e:
                        logging.error(f"Hub upload failed: {e}")

                state["checkpoint_idx"] += 1

                if is_ddp:
                    dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    if is_master:
        logging.info("Training complete.")


# ================= ENTRY =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    train(args.lang)
