#!/usr/bin/env python3
import os, json, logging, argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from datasets import load_dataset
import sentencepiece as spm

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    PreTrainedTokenizerFast,
)

from huggingface_hub import create_repo, upload_folder

# ============================================================
# CONFIG
# ============================================================

LR = 6e-4
SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2

TOTAL_TOKENS = 5_000_000_000
PHASE1_TOKENS = TOTAL_TOKENS // 2
L2_TOTAL = int(TOTAL_TOKENS * (1/3))

CHECKPOINT_FRACS = [0.25, 0.5, 0.75, 1.0]

HF_USER = "suchirsalhan"
SAVE_DIR = "./runs_finewebedu"
TOKENIZER_DIR = "./tokenizer-finewebedu"
STATE_FILE = "trainer_state.json"

LEVEL_NAMES = {
    0: "beginner",
    1: "intermediate",
    2: "advanced",
    3: "fluent",
}

# ============================================================
# DISTRIBUTED
# ============================================================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        return (
            True,
            local_rank,
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


# ============================================================
# LOGGING
# ============================================================

def setup_logging(is_master, run_dir):

    if not is_master:
        logging.disable(logging.CRITICAL)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(run_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


# ============================================================
# STATE
# ============================================================

def default_state():
    return {"phase": 1, "l1_seen": 0, "l2_seen": 0, "checkpoint_idx": 0}


def load_state(run_dir):
    p = os.path.join(run_dir, STATE_FILE)
    return json.load(open(p)) if os.path.exists(p) else default_state()


def save_state(run_dir, state):
    json.dump(state, open(os.path.join(run_dir, STATE_FILE), "w"))


# ============================================================
# TOKENIZER (FIXED)
# ============================================================

def build_tokenizer_corpus(lang, global_rank):

    corpus_file = os.path.join(TOKENIZER_DIR, "all_texts.txt")

    if global_rank != 0:
        return corpus_file

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    logging.info("Building tokenizer corpus...")

    with open(corpus_file, "w", encoding="utf-8") as f:

        # -------- CulturaX --------
        ds = load_dataset(
            "uonlp/CulturaX",
            lang,
            split="train",
            streaming=True,
        )

        for ex in ds:
            t = ex.get("text")
            if t:
                f.write(t.replace("\n", " ") + "\n")

        # -------- FineWeb-Edu --------
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "default",
            split="train",
            streaming=True,
            languages=[lang],
        )

        for ex in ds:
            t = ex.get("text")
            if t:
                f.write(t.replace("\n", " ") + "\n")

    if os.path.getsize(corpus_file) == 0:
        raise RuntimeError("Tokenizer corpus EMPTY")

    logging.info("Tokenizer corpus complete.")
    return corpus_file


def train_sentencepiece(lang, is_ddp, global_rank):

    model_path = os.path.join(TOKENIZER_DIR, "spm.model")

    if os.path.exists(model_path):
        return

    corpus = build_tokenizer_corpus(lang, global_rank)

    if is_ddp:
        dist.barrier()

    if global_rank == 0:

        logging.info("Training SentencePiece tokenizer")

        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=os.path.join(TOKENIZER_DIR, "spm"),
            vocab_size=50000,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=0,
            eos_id=1,
            pad_id=2,
            unk_id=3,
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=None,
            tokenizer_object=None,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            sp_model_kwargs={"model_file": model_path},
        )

        tokenizer.save_pretrained(TOKENIZER_DIR)

    if is_ddp:
        dist.barrier()

    logging.info("Tokenizer ready.")


# ============================================================
# STREAMING DATA
# ============================================================

def streaming_blocks(
    dataset,
    config,
    rank,
    world_size,
    tokenizer,
    skip_tokens=0,
):

    ds = load_dataset(
        dataset,
        config,
        split="train",
        streaming=True,
    ).shard(world_size, rank)

    buf, seen = [], 0

    for ex in ds:
        text = ex.get("text")
        if not text:
            continue

        ids = tokenizer.encode(text)

        if seen + len(ids) < skip_tokens:
            seen += len(ids)
            continue

        buf.extend(ids)

        while len(buf) >= SEQ_LEN:
            yield buf[:SEQ_LEN]
            buf = buf[SEQ_LEN:]


# ============================================================
# TRAIN STEP
# ============================================================

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


# ============================================================
# PUSH HUB
# ============================================================

def push_to_hub(path, lang, idx):

    repo = f"{HF_USER}/B-GPT-{lang}-finewebedu-{LEVEL_NAMES[idx]}"

    create_repo(repo, exist_ok=True)

    upload_folder(
        repo_id=repo,
        folder_path=path,
        commit_message="checkpoint",
    )


# ============================================================
# TRAIN LOOP
# ============================================================

def train(lang):

    is_ddp, local_rank, global_rank, world_size = setup_distributed()
    is_master = global_rank == 0

    run_dir = os.path.join(SAVE_DIR, lang)
    os.makedirs(run_dir, exist_ok=True)

    setup_logging(is_master, run_dir)

    device = f"cuda:{local_rank}"

    # ---------- TOKENIZER ----------
    train_sentencepiece(lang, is_ddp, global_rank)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    # ---------- MODEL ----------
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
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    state = load_state(run_dir)

    l1_iter = iter(
        streaming_blocks(
            "uonlp/CulturaX",
            lang,
            global_rank,
            world_size,
            tokenizer,
            state["l1_seen"],
        )
    )

    l2_iter = iter(
        streaming_blocks(
            "HuggingFaceFW/fineweb-edu",
            "default",
            global_rank,
            world_size,
            tokenizer,
            state["l2_seen"],
        )
    )

    step = 0
    checkpoint_tokens = [int(L2_TOTAL * x) for x in CHECKPOINT_FRACS]

    # ================= PHASE 1 =================

    if state["phase"] == 1:

        if is_master:
            logging.info("Phase1 L1")

        while state["l1_seen"] < PHASE1_TOKENS:

            batch = [next(l1_iter) for _ in range(BATCH_SIZE)]
            loss = train_step(model, optimizer, batch, device, step)

            state["l1_seen"] += BATCH_SIZE * SEQ_LEN * world_size
            step += 1

            if is_master and step % 50 == 0:
                logging.info(f"step {step} loss {loss:.3f}")

        state["phase"] = 2
        save_state(run_dir, state)

    # ================= PHASE 2 =================

    if is_master:
        logging.info("Phase2 bilingual")

    while state["l2_seen"] < L2_TOTAL:

        for src in ["l1", "l2", "l2"]:

            iterator = l1_iter if src == "l1" else l2_iter
            batch = [next(iterator) for _ in range(BATCH_SIZE)]

            loss = train_step(model, optimizer, batch, device, step)

            if src == "l2":
                state["l2_seen"] += BATCH_SIZE * SEQ_LEN * world_size
            else:
                state["l1_seen"] += BATCH_SIZE * SEQ_LEN * world_size

            step += 1

            while (
                state["checkpoint_idx"] < len(checkpoint_tokens)
                and state["l2_seen"]
                >= checkpoint_tokens[state["checkpoint_idx"]]
            ):

                pct = int(
                    CHECKPOINT_FRACS[state["checkpoint_idx"]] * 100
                )

                path = os.path.join(run_dir, f"checkpoint_{pct}")
                os.makedirs(path, exist_ok=True)

                if is_master:
                    (model.module if is_ddp else model).save_pretrained(
                        path
                    )
                    tokenizer.save_pretrained(path)
                    save_state(run_dir, state)
                    push_to_hub(path, lang, state["checkpoint_idx"])

                state["checkpoint_idx"] += 1

    if is_ddp:
        dist.destroy_process_group()

    if is_master:
        logging.info("Training complete.")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    train(args.lang)
