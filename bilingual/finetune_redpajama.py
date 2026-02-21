#!/usr/bin/env python3
import os, json, time, logging, argparse
import gzip
import requests
import itertools   # ✅ REQUIRED FIX
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from huggingface_hub import HfApi, create_repo, upload_folder

# ============================================================
# SPEED SETTINGS
# ============================================================
torch.set_float32_matmul_precision("high")

# ============================================================
# CONFIG
# ============================================================
LR = 5e-5
SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2
TOTAL_TOKENS = 5_000_000_000
PHASE1_TOKENS = TOTAL_TOKENS // 2
L2_TOTAL = int(TOTAL_TOKENS * (1/3))
CHECKPOINT_FRACS = [0.25, 0.5, 0.75, 1.0]

HF_USER = "suchirsalhan"

SAVE_DIR = "./runs_redpajama"   # ✅ CLEAN FIX
STATE_FILE = "trainer_state.json"

LEVEL_NAMES = {
    0: "beginner",
    1: "intermediate",
    2: "advanced",
    3: "fluent"
}

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
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
            logging.StreamHandler()
        ],
    )

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# TRAIN STATE
# ============================================================
def default_state():
    return dict(phase=1, l1_seen=0, l2_seen=0, checkpoint_idx=0)

def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    if os.path.exists(path):
        return json.load(open(path))
    return default_state()

def save_state(run_dir, state):
    json.dump(state, open(os.path.join(run_dir, STATE_FILE), "w"))

# ============================================================
# REDPAJAMA QUALITY FILTER
# ============================================================
def gopher_rules_pass(sample):
    if "quality_signals" not in sample:
        return True
    signals = json.loads(sample["quality_signals"])
    wc = signals["rps_doc_word_count"][0][2]
    if wc < 50 or wc > 100_000:
        return False
    ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if ratio > 0.1:
        return False
    return True

# ============================================================
# STREAMING DATA
# ============================================================
BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"

def streaming_blocks(dataset_name, config, split,
                     rank, world_size, skip_tokens=0):

    # ---------- L1 HF streaming ----------
    if dataset_name != "togethercomputer/RedPajama-Data-V2":

        if isinstance(config, dict):
            ds = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                **config
            )
        else:
            ds = load_dataset(
                dataset_name,
                config,
                split=split,
                streaming=True
            )

        ds = ds.shard(num_shards=world_size, index=rank)

        buffer = []
        seen = 0

        for ex in ds:
            ids = tokenizer(
                ex["text"],
                truncation=True,
                padding=False,
                max_length=SEQ_LEN,
                add_special_tokens=False,
            )["input_ids"]

            if seen + len(ids) < skip_tokens:
                seen += len(ids)
                continue

            buffer.extend(ids)

            while len(buffer) >= SEQ_LEN:
                yield buffer[:SEQ_LEN]
                buffer = buffer[SEQ_LEN:]

        return

    # ---------- RedPajama persistent streaming ----------
    snapshot = config.get("snapshots", ["2023-06"])[0]
    partition = config.get("partition", "head_middle")
    lang = config.get("languages", ["en"])[0]

    listing_url = f"{BASE_URL}/listings/{lang}-{snapshot}-{partition}.txt"

    session = requests.Session()
    listing = session.get(listing_url).text.splitlines()
    listing = listing[rank::world_size]

    buffer = []
    seen = 0

    for shard in itertools.cycle(listing):  # ✅ FIXED
        shard_url = f"{BASE_URL}/documents/{shard}.json.gz"

        try:
            r = session.get(shard_url, stream=True, timeout=60)

            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc = json.loads(line)

                    text = doc.get("raw_content")
                    if not text or not gopher_rules_pass(doc):
                        continue

                    ids = tokenizer(
                        text,
                        truncation=False,
                        padding=False,
                        add_special_tokens=False
                    )["input_ids"]

                    if seen + len(ids) < skip_tokens:
                        seen += len(ids)
                        continue

                    buffer.extend(ids)

                    while len(buffer) >= SEQ_LEN:
                        yield buffer[:SEQ_LEN]
                        buffer = buffer[SEQ_LEN:]

        except Exception:
            continue

# ============================================================
# TRAIN STEP
# ============================================================
def train_step(model, optimizer, batch, device, grad_accum, step_idx):
    x = torch.tensor(batch, device=device)[:, :-1]
    y = torch.tensor(batch, device=device)[:, 1:]

    with autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, labels=y).loss / grad_accum

    loss.backward()

    if (step_idx + 1) % grad_accum == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item() * grad_accum

# ============================================================
# PUSH TO HUB
# ============================================================
def push_to_hub(path, lang, ckpt_idx):
    level = LEVEL_NAMES[ckpt_idx]
    repo_id = f"{HF_USER}/B-GPT-{lang}-redpajama-{level}"

    api = HfApi()
    create_repo(repo_id, exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path=path,
        commit_message=f"{level} checkpoint"
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

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.gradient_checkpointing_enable()

    if is_ddp:
        model = DDP(model, device_ids=[local_rank],
                    gradient_as_bucket_view=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    state = load_state(run_dir)

    # Streams
    l1_stream = streaming_blocks(
        "uonlp/CulturaX", lang, "train",
        global_rank, world_size,
        skip_tokens=state["l1_seen"]
    )

    l2_stream = streaming_blocks(
        "togethercomputer/RedPajama-Data-V2",
        dict(
            name="default",
            partition="head_middle",
            snapshots=["2023-06"],
            languages=["en"]
        ),
        "train",
        global_rank,
        world_size,
        skip_tokens=state["l2_seen"]
    )

    l1_iter = iter(l1_stream)
    l2_iter = iter(l2_stream)

    checkpoint_tokens = [
        int(L2_TOTAL * x) for x in CHECKPOINT_FRACS
    ]

    step_idx = 0
    start = time.time()

    # ================= PHASE 1 =================
    if state["phase"] == 1:
        if is_master:
            logging.info("Phase 1: L1 only")

        while state["l1_seen"] < PHASE1_TOKENS:

            batch = [next(l1_iter) for _ in range(BATCH_SIZE)]

            loss = train_step(
                model, optimizer, batch,
                device, GRAD_ACCUM, step_idx
            )

            tokens = BATCH_SIZE * SEQ_LEN * world_size
            state["l1_seen"] += tokens
            step_idx += 1

            if is_master and step_idx % 50 == 0:
                elapsed = time.time() - start
                logging.info(
                    f"step {step_idx} | loss {loss:.3f} | "
                    f"L1 {state['l1_seen']/1e9:.2f}B | "
                    f"{state['l1_seen']/elapsed/1e6:.1f}M tok/s"
                )

        state["phase"] = 2
        save_state(run_dir, state)

    # ================= PHASE 2 =================
    if is_master:
        logging.info("Phase 2: bilingual (1 L1 : 2 L2)")

    while state["l2_seen"] < L2_TOTAL:

        for stream_name in ["l1", "l2", "l2"]:

            stream = l1_iter if stream_name == "l1" else l2_iter
            batch = [next(stream) for _ in range(BATCH_SIZE)]

            loss = train_step(
                model, optimizer, batch,
                device, GRAD_ACCUM, step_idx
            )

            tokens = BATCH_SIZE * SEQ_LEN * world_size
            step_idx += 1

            if stream_name == "l2":
                state["l2_seen"] += tokens
            else:
                state["l1_seen"] += tokens

            if is_master and step_idx % 50 == 0:
                logging.info(
                    f"step {step_idx} | loss {loss:.3f} | "
                    f"L2 {state['l2_seen']/1e9:.2f}B | "
                    f"ratio {state['l1_seen']/max(1,state['l2_seen']):.2f}"
                )

            while (
                state["checkpoint_idx"] < len(checkpoint_tokens)
                and state["l2_seen"] >= checkpoint_tokens[state["checkpoint_idx"]]
            ):

                pct = int(
                    CHECKPOINT_FRACS[state["checkpoint_idx"]] * 100
                )

                path = os.path.join(run_dir, f"checkpoint_{pct}")
                os.makedirs(path, exist_ok=True)

                if is_master:
                    (model.module if is_ddp else model).save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    save_state(run_dir, state)
                    push_to_hub(path, lang, state["checkpoint_idx"])

                state["checkpoint_idx"] += 1

    if is_ddp:
        dist.destroy_process_group()

    if is_master:
        logging.info("Training complete.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    train(args.lang)
