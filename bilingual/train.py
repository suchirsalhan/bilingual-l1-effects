#!/usr/bin/env python3
import os, json, time, logging, argparse, gzip, requests, itertools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datasets import load_dataset
import sentencepiece as spm
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AutoTokenizer
)
from huggingface_hub import HfApi, create_repo, upload_folder

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
STATE_FILE = "trainer_state.json"
TOKENIZER_DIR = "./tokenizer-finewebedu"
LEVEL_NAMES = {0: "beginner", 1: "intermediate", 2: "advanced", 3: "fluent"}

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        lrank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lrank)
        return True, lrank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1

# ============================================================
# LOGGING
# ============================================================
def setup_logging(is_master, run_dir):
    if is_master:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(run_dir, "training.log")),
                logging.StreamHandler()
            ],
        )
    else:
        logging.disable(logging.CRITICAL)

# ============================================================
# STATE
# ============================================================
def default_state():
    return {"phase":1,"l1_seen":0,"l2_seen":0,"checkpoint_idx":0}

def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    return json.load(open(path)) if os.path.exists(path) else default_state()

def save_state(run_dir, state):
    json.dump(state, open(os.path.join(run_dir, STATE_FILE),"w"))

# ============================================================
# STREAMING TOKENIZER TRAINING
# ============================================================
def train_sentencepiece(lang, is_ddp, global_rank, world_size):

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    corpus_name = os.path.join(TOKENIZER_DIR, "all_texts.txt")

    if (not os.path.exists(corpus_name)) and (not is_ddp or global_rank==0):
        with open(corpus_name,"w",encoding="utf-8") as f:
            for dname, config in [
                ("uonlp/CulturaX", lang),
                ("HuggingFaceFW/fineweb-edu",
                 dict(name="default",partition="train",languages=[lang]))
            ]:
                ds = load_dataset(
                    dname,
                    split="train",
                    streaming=True,
                    **(config if isinstance(config,dict) else {})
                )
                for ex in ds:
                    text = ex.get("text") or ex.get("raw_content")
                    if text:
                        f.write(text.replace("\n"," ")+"\n")

    if is_ddp:
        dist.barrier()

    spm.SentencePieceTrainer.train(
        input=corpus_name,
        model_prefix=os.path.join(TOKENIZER_DIR, "spm"),
        vocab_size=50000,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=0,eos_id=1,pad_id=2,unk_id=3
    )

    # ✅ CREATE HF TOKENIZER FILES
    if global_rank == 0:
        tokenizer_config = {
            "model_max_length": SEQ_LEN,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }

        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }

        json.dump(tokenizer_config,
                  open(os.path.join(TOKENIZER_DIR,"tokenizer_config.json"),"w"))

        json.dump(special_tokens_map,
                  open(os.path.join(TOKENIZER_DIR,"special_tokens_map.json"),"w"))

    if is_ddp:
        dist.barrier()

    logging.info("Tokenizer trained.")

# ============================================================
# STREAMING BLOCKS
# ============================================================
def streaming_blocks(dataset_name, config, split,
                     rank, world_size, tokenizer, skip_tokens=0):

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        **(config if isinstance(config,dict) else {})
    )

    ds = ds.shard(num_shards=world_size, index=rank)

    buf, seen = [],0

    for ex in ds:
        text = ex.get("text") or ex.get("raw_content")
        ids = tokenizer.encode(text)

        if seen + len(ids)<skip_tokens:
            seen += len(ids)
            continue

        buf.extend(ids)

        while len(buf)>=SEQ_LEN:
            yield buf[:SEQ_LEN]
            buf=buf[SEQ_LEN:]

# ============================================================
# TRAIN STEP
# ============================================================
def train_step(model, optimizer, batch, device, grad_accum, step_idx):

    x=torch.tensor(batch,device=device)[:,:-1]
    y=torch.tensor(batch,device=device)[:,1:]

    with autocast("cuda",dtype=torch.bfloat16):
        loss=model(x,labels=y).loss/grad_accum

    loss.backward()

    if (step_idx+1)%grad_accum==0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item()*grad_accum

# ============================================================
# PUSH TO HUB
# ============================================================
def push_to_hub(path, lang, ckpt_idx):
    level=LEVEL_NAMES[ckpt_idx]
    repo_id=f"{HF_USER}/B-GPT-{lang}-finewebedu-{level}"
    create_repo(repo_id,exist_ok=True)
    upload_folder(repo_id=repo_id,
                  folder_path=path,
                  commit_message=f"{level} checkpoint")

# ============================================================
# TRAIN LOOP
# ============================================================
def train(lang):

    is_ddp, local_rank, global_rank, world_size = setup_distributed()
    is_master = (global_rank==0)

    run_dir=os.path.join(SAVE_DIR,lang)
    os.makedirs(run_dir,exist_ok=True)

    setup_logging(is_master,run_dir)
    device=f"cuda:{local_rank}"

    # ========================================================
    # TOKENIZER FIX ✅
    # ========================================================
    if not os.path.exists(os.path.join(TOKENIZER_DIR,"spm.model")):
        train_sentencepiece(lang,is_ddp,global_rank,world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        use_fast=False
    )

    # ========================================================
    # MODEL
    # ========================================================
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    model.apply(model._init_weights)
    model.gradient_checkpointing_enable()

    if is_ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    gradient_as_bucket_view=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)

    state = load_state(run_dir)

    l1_iter=iter(streaming_blocks(
        "uonlp/CulturaX",
        lang,"train",
        global_rank,world_size,
        tokenizer,state["l1_seen"]
    ))

    l2_iter=iter(streaming_blocks(
        "HuggingFaceFW/fineweb-edu",
        dict(name="default",partition="train",languages=[lang]),
        "train",
        global_rank,world_size,
        tokenizer,state["l2_seen"]
    ))

    checkpoint_tokens=[int(L2_TOTAL*x) for x in CHECKPOINT_FRACS]
    step_idx=0

    # ========================================================
    # PHASE 1
    # ========================================================
    if state["phase"]==1:
        if is_master:
            logging.info("Phase1: L1 only")

        while state["l1_seen"]<PHASE1_TOKENS:
            batch=[next(l1_iter) for _ in range(BATCH_SIZE)]
            loss=train_step(model,optimizer,batch,device,GRAD_ACCUM,step_idx)

            state["l1_seen"] += BATCH_SIZE*SEQ_LEN*world_size
            step_idx+=1

            if is_master and step_idx%50==0:
                logging.info(
                    f"step {step_idx} loss {loss:.3f} "
                    f"L1 {state['l1_seen']/1e9:.2f}B"
                )

        state["phase"]=2
        save_state(run_dir,state)

    # ========================================================
    # PHASE 2
    # ========================================================
    if is_master:
        logging.info("Phase2 bilingual")

    while state["l2_seen"]<L2_TOTAL:

        for sname in ["l1","l2","l2"]:
            it = l1_iter if sname=="l1" else l2_iter

            batch=[next(it) for _ in range(BATCH_SIZE)]
            loss=train_step(model,optimizer,batch,device,GRAD_ACCUM,step_idx)

            if sname=="l2":
                state["l2_seen"] += BATCH_SIZE*SEQ_LEN*world_size
            else:
                state["l1_seen"] += BATCH_SIZE*SEQ_LEN*world_size

            step_idx+=1

            if is_master and step_idx%50==0:
                logging.info(
                    f"step {step_idx} loss {loss:.3f} "
                    f"L2 {state['l2_seen']/1e9:.2f}B"
                )

            while (
                state["checkpoint_idx"]<len(checkpoint_tokens)
                and state["l2_seen"]>=checkpoint_tokens[state["checkpoint_idx"]]
            ):
                pct=int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
                path=os.path.join(run_dir,f"checkpoint_{pct}")
                os.makedirs(path,exist_ok=True)

                if is_master:
                    (model.module if is_ddp else model).save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    save_state(run_dir,state)
                    push_to_hub(path,lang,state["checkpoint_idx"])

                state["checkpoint_idx"]+=1

    if is_ddp:
        dist.destroy_process_group()

    if is_master:
        logging.info("Training complete.")

# ============================================================
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lang",type=str,required=True)
    args=parser.parse_args()
    train(args.lang)
