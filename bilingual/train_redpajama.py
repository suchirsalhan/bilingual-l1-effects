#!/usr/bin/env python3
import os, json, logging, argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datasets import load_dataset
import sentencepiece as spm
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, LlamaTokenizer
from huggingface_hub import create_repo, upload_folder
import itertools, gzip, requests

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

SAVE_DIR = "./runs_redpajama"
STATE_FILE = "trainer_state.json"

LEVEL_NAMES = {
    0:"beginner",
    1:"intermediate",
    2:"advanced",
    3:"fluent"
}

# ================= DISTRIBUTED =================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
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
    return {"phase":1,"l1_seen":0,"l2_seen":0,"checkpoint_idx":0}

def load_state(run_dir):
    path=os.path.join(run_dir,STATE_FILE)
    return json.load(open(path)) if os.path.exists(path) else default_state()

def save_state(run_dir,state):
    json.dump(state,open(os.path.join(run_dir,STATE_FILE),"w"))


# ================= REDPAJAMA FILTER =================

def gopher_rules_pass(sample):
    if "quality_signals" not in sample:
        return True
    signals=json.loads(sample["quality_signals"])
    wc=signals["rps_doc_word_count"][0][2]
    if wc<50 or wc>100_000:
        return False
    ratio=signals["rps_doc_symbol_to_word_ratio"][0][2]
    if ratio>0.1:
        return False
    return True


# =========================================================
# TOKENIZER CHANGE (LOCAL LOAD FROM ADJACENT DIR)
# =========================================================

def load_local_tokenizer(lang, is_ddp, global_rank):
    """
    Loads the SentencePiece model from ../tokenizers/redpajama_{lang}/
    relative to the script location in /root/bilingual/
    """
    # Navigating from root/bilingual/ to root/tokenizers/
    base_script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_root = os.path.join(base_script_dir, "..", "tokenizers")
    local_path = os.path.join(tokenizer_root, f"redpajama_{lang}")
    model_path = os.path.join(local_path, "spm.model")

    if global_rank == 0:
        if not os.path.exists(model_path):
            # Fallback check for absolute path from root if script is moved
            logging.error(f"Tokenizer not found at {model_path}")
            raise FileNotFoundError(f"Missing spm.model at {model_path}")
        logging.info(f"Successfully located local tokenizer at {model_path}")

    # Initialize via LlamaTokenizer (best for raw .model files)
    tokenizer = LlamaTokenizer(vocab_file=model_path, legacy=False)

    # Standardize special tokens for GPT-2 Architecture
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_ddp:
        dist.barrier()

    return tokenizer


# ================= DATA STREAM =================

class ResilientStreamingIterator:

    def __init__(self,dataset,config,rank,world_size,tokenizer,skip_tokens=0):
        self.args=(dataset,config,rank,world_size,tokenizer,skip_tokens)
        self.iterator=self._build_iterator()

    def _build_iterator(self):
        return streaming_blocks(*self.args)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator=self._build_iterator()
            return next(self.iterator)


def streaming_blocks(dataset,config,rank,world_size,tokenizer,skip_tokens=0):

    # ---------- CulturaX ----------
    if dataset=="uonlp/CulturaX":

        ds=load_dataset(dataset,config,split="train",streaming=True)

        if world_size>1:
            ds=ds.shard(num_shards=world_size,index=rank)

        buf,seen=[],skip_tokens

        for ex in itertools.cycle(ds):
            text=ex.get("text")
            if not text:
                continue

            ids=tokenizer.encode(text,add_special_tokens=False)

            if seen+len(ids)<skip_tokens:
                seen+=len(ids)
                continue

            buf.extend(ids)

            while len(buf)>=SEQ_LEN:
                yield buf[:SEQ_LEN]
                buf=buf[SEQ_LEN:]

        return


    # ---------- RedPajama ----------
    BASE_URL="https://data.together.xyz/redpajama-data-v2/v1.0.0"

    snapshot=config.get("snapshots",["2023-06"])[0]
    partition=config.get("partition","head_middle")
    lang=config.get("languages",["en"])[0]

    listing_url=f"{BASE_URL}/listings/{lang}-{snapshot}-{partition}.txt"

    session=requests.Session()
    listing=session.get(listing_url).text.splitlines()
    listing=listing[rank::world_size]

    buf,seen=[],skip_tokens

    for shard in itertools.cycle(listing):
        try:
            r=session.get(
                f"{BASE_URL}/documents/{shard}.json.gz",
                stream=True,
                timeout=60,
            )

            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:

                    doc=json.loads(line)
                    text=doc.get("raw_content")

                    if not text or not gopher_rules_pass(doc):
                        continue

                    ids=tokenizer.encode(text,add_special_tokens=False)

                    if seen+len(ids)<skip_tokens:
                        seen+=len(ids)
                        continue

                    buf.extend(ids)

                    while len(buf)>=SEQ_LEN:
                        yield buf[:SEQ_LEN]
                        buf=buf[SEQ_LEN:]

        except Exception:
            continue


# ================= TRAIN STEP =================

def train_step(model,opt,batch,device,step):

    x=torch.tensor(batch,device=device)[:,:-1]
    y=torch.tensor(batch,device=device)[:,1:]

    with autocast("cuda",dtype=torch.bfloat16):
        loss=model(x,labels=y).loss/GRAD_ACCUM

    loss.backward()

    if (step+1)%GRAD_ACCUM==0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    return loss.item()*GRAD_ACCUM


# ================= HUB PUSH =================

def push_to_hub(local_path,lang,idx):

    level=LEVEL_NAMES[idx]
    repo_name=f"B-GPT-{lang}-redpajama-{level}"

    create_repo(repo_name,exist_ok=True,private=False)

    upload_folder(
        folder_path=local_path,
        repo_id=f"{HF_USER}/{repo_name}",
        repo_type="model",
    )


# ================= TRAIN LOOP =================

def train(lang):

    is_ddp,local_rank,global_rank,world_size=setup_distributed()
    is_master=global_rank==0

    run_dir=os.path.join(SAVE_DIR,lang)
    setup_logging(is_master,run_dir)

    device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # ---------- TOKENIZER ----------
    tokenizer=load_local_tokenizer(lang,is_ddp,global_rank)

    # ---------- MODEL ----------
    config=GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model=GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()

    if is_ddp:
        model=DDP(model,device_ids=[local_rank],output_device=local_rank)

    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)

    # ---------- STATE ----------
    state=load_state(run_dir)

    l1_iter=ResilientStreamingIterator(
        "uonlp/CulturaX",
        lang,
        global_rank,
        world_size,
        tokenizer,
        state["l1_seen"],
    )

    l2_iter=ResilientStreamingIterator(
        "togethercomputer/RedPajama-Data-V2",
        dict(partition="head_middle",snapshots=["2023-06"],languages=[lang]),
        global_rank,
        world_size,
        tokenizer,
        state["l2_seen"],
    )

    step=0
    tokens_per_batch=BATCH_SIZE*SEQ_LEN
    checkpoint_tokens=[int(L2_TOTAL*x) for x in CHECKPOINT_FRACS]

    # ---------- PHASE 1 ----------
    if state["phase"]==1:

        if is_master:
            logging.info("Starting Phase 1")

        while state["l1_seen"]<PHASE1_TOKENS:

            batch=[next(l1_iter) for _ in range(BATCH_SIZE)]
            loss=train_step(model,optimizer,batch,device,step)

            state["l1_seen"]+=tokens_per_batch
            step+=1

        state["phase"]=2
        if is_master:
            save_state(run_dir,state)

    # ---------- PHASE 2 ----------
    if is_master:
        logging.info("Starting Phase 2")

    while state["l2_seen"]<L2_TOTAL:

        for src in ["l1","l2","l2"]:

            try:
                iterator=l1_iter if src=="l1" else l2_iter
                batch=[next(iterator) for _ in range(BATCH_SIZE)]
            except StopIteration:
                continue

            loss=train_step(model,optimizer,batch,device,step)

            if src=="l2":
                state["l2_seen"]+=tokens_per_batch
            else:
                state["l1_seen"]+=tokens_per_batch

            step+=1

            if (
                state["checkpoint_idx"]<len(checkpoint_tokens)
                and state["l2_seen"]>=checkpoint_tokens[state["checkpoint_idx"]]
            ):

                pct=int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
                path=os.path.join(run_dir,f"checkpoint_{pct}")

                if is_master:
                    os.makedirs(path,exist_ok=True)

                    (model.module if is_ddp else model).save_pretrained(path)
                    
                    # This ensures the local tokenizer is saved into the checkpoint 
                    # folder before pushing to the hub
                    tokenizer.save_pretrained(path)

                    save_state(run_dir,state)

                    logging.info(f"Saved checkpoint {pct}% with local tokenizer files.")

                    try:
                        push_to_hub(path,lang,state["checkpoint_idx"])
                    except Exception as e:
                        logging.error(f"Hub upload failed: {e}")

                state["checkpoint_idx"]+=1

                if is_ddp:
                    dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    if is_master:
        logging.info("Training complete.")


# ================= ENTRY =================

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lang",required=True)
    args=parser.parse_args()

    train(args.lang)
