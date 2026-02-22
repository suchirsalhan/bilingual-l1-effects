#!/usr/bin/env python3
import os, json, time, logging, argparse, gzip, requests, itertools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datasets import load_dataset
import sentencepiece as spm
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from huggingface_hub import create_repo, upload_folder

# ============================================================
# SETTINGS
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
SAVE_DIR = "./runs_redpajama"
TOKENIZER_DIR = "./tokenizer-redpajama"
STATE_FILE = "trainer_state.json"
LEVEL_NAMES = {0:"beginner",1:"intermediate",2:"advanced",3:"fluent"}

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
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(run_dir,"training.log")), logging.StreamHandler()],
    )

# ============================================================
# STATE
# ============================================================
def default_state(): return {"phase":1,"l1_seen":0,"l2_seen":0,"checkpoint_idx":0}
def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    return json.load(open(path)) if os.path.exists(path) else default_state()
def save_state(run_dir, state): json.dump(state, open(os.path.join(run_dir, STATE_FILE),"w"))

# ============================================================
# REDPAJAMA FILTER
# ============================================================
def gopher_rules_pass(sample):
    if "quality_signals" not in sample: return True
    signals=json.loads(sample["quality_signals"])
    wc=signals["rps_doc_word_count"][0][2]
    if wc<50 or wc>100_000: return False
    ratio=signals["rps_doc_symbol_to_word_ratio"][0][2]
    if ratio>0.1: return False
    return True

# ============================================================
# TOKENIZER TRAINING (TEMP FILE)
# ============================================================
def build_tokenizer_corpus(lang, rank, world_size):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    corpus_file = os.path.join(TOKENIZER_DIR,"all_texts.txt")
    if rank != 0: return corpus_file
    logging.info("Building tokenizer corpus (temporary file)...")
    with open(corpus_file,"w",encoding="utf-8") as f:
        datasets_to_stream = [
            ("uonlp/CulturaX", {"split":"train","lang":lang}),
            ("togethercomputer/RedPajama-Data-V2", {"split":"train","name":"default","partition":"head_middle","snapshots":["2023-06"],"languages":[lang]})
        ]
        for dname, cfg in datasets_to_stream:
            ds = load_dataset(dname, **cfg, streaming=True)
            if world_size>1: ds = ds.shard(num_shards=world_size,index=rank)
            for ex in ds:
                text=ex.get("text") or ex.get("raw_content")
                if text:
                    f.write(text.replace("\n"," ")+"\n")
    logging.info("Tokenizer corpus done.")
    return corpus_file

def train_sentencepiece(lang, is_ddp, rank, world_size):
    model_prefix=os.path.join(TOKENIZER_DIR,"spm")
    model_file=model_prefix+".model"
    if os.path.exists(model_file): return
    corpus_file = build_tokenizer_corpus(lang, rank, world_size)
    if rank==0:
        logging.info("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=50000,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=0,
            eos_id=1,
            pad_id=2,
            unk_id=3
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=None,
            bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
            sp_model_kwargs={"model_file":model_file})
        tokenizer.save_pretrained(TOKENIZER_DIR)
    if is_ddp: dist.barrier()
    logging.info("Tokenizer ready.")

# ============================================================
# STREAMING BLOCKS RESILIENT
# ============================================================
BASE_URL="https://data.together.xyz/redpajama-data-v2/v1.0.0"
def streaming_blocks(dataset_name, config, rank, world_size, tokenizer, skip_tokens=0):
    buf, seen=[],0
    if dataset_name!="togethercomputer/RedPajama-Data-V2":
        ds=load_dataset(dataset_name, config, split="train", streaming=True)
        if world_size>1: ds=ds.shard(num_shards=world_size,index=rank)
        for ex in itertools.cycle(ds):
            ids=tokenizer.encode(ex.get("text") or "")
            if seen+len(ids)<skip_tokens: seen+=len(ids); continue
            buf.extend(ids)
            while len(buf)>=SEQ_LEN:
                yield buf[:SEQ_LEN]
                buf=buf[SEQ_LEN:]
        return

    snapshot=config.get("snapshots",["2023-06"])[0]
    partition=config.get("partition","head_middle")
    lang=config.get("languages",["en"])[0]
    listing_url=f"{BASE_URL}/listings/{lang}-{snapshot}-{partition}.txt"
    session=requests.Session()
    listing=session.get(listing_url).text.splitlines()
    listing=listing[rank::world_size]

    for shard in itertools.cycle(listing):
        try:
            r=session.get(f"{BASE_URL}/documents/{shard}.json.gz",stream=True,timeout=60)
            with gzip.GzipFile(fileobj=r.raw) as f:
                for line in f:
                    doc=json.loads(line)
                    text=doc.get("raw_content")
                    if not text or not gopher_rules_pass(doc): continue
                    ids=tokenizer.encode(text)
                    if seen+len(ids)<skip_tokens: seen+=len(ids); continue
                    buf.extend(ids)
                    while len(buf)>=SEQ_LEN:
                        yield buf[:SEQ_LEN]
                        buf=buf[SEQ_LEN:]
        except Exception: continue

# ============================================================
# TRAIN STEP
# ============================================================
def train_step(model,opt,batch,device,step_idx):
    x=torch.tensor(batch,device=device)[:,:-1]
    y=torch.tensor(batch,device=device)[:,1:]
    with autocast("cuda",dtype=torch.bfloat16):
        loss=model(x,labels=y).loss/GRAD_ACCUM
    loss.backward()
    if (step_idx+1)%GRAD_ACCUM==0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss.item()*GRAD_ACCUM

# ============================================================
# PUSH TO HUB
# ============================================================
def push_to_hub(path, lang, ckpt_idx):
    level=LEVEL_NAMES[ckpt_idx]
    repo_id=f"{HF_USER}/B-GPT-{lang}-redpajama-{level}"
    create_repo(repo_id,exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=path, commit_message=f"{level} checkpoint")

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

    # ---------- TOKENIZER ----------
    train_sentencepiece(lang,is_ddp,global_rank,world_size)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    # ---------- MODEL ----------
    config=GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model=GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()
    if is_ddp: model = DDP(model, device_ids=[local_rank])
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)

    state=load_state(run_dir)

    # ---------- DATA STREAMS ----------
    l1_iter=streaming_blocks("uonlp/CulturaX",lang,global_rank,world_size,tokenizer,state["l1_seen"])
    l2_iter=streaming_blocks("togethercomputer/RedPajama-Data-V2",
        dict(partition="head_middle",snapshots=["2023-06"],languages=[lang]),
        global_rank,world_size,tokenizer,state["l2_seen"]
    )
    l1_iter=iter(itertools.cycle(l1_iter))
    l2_iter=iter(itertools.cycle(l2_iter))

    checkpoint_tokens=[int(L2_TOTAL*x) for x in CHECKPOINT_FRACS]
    step_idx=0

    # ---------- PHASE 1 ----------
    if state["phase"]==1:
        if is_master: logging.info("Phase1: L1 only")
        while state["l1_seen"]<PHASE1_TOKENS:
            batch=[next(l1_iter) for _ in range(BATCH_SIZE)]
            loss=train_step(model,optimizer,batch,device,step_idx)
            state["l1_seen"]+=BATCH_SIZE*SEQ_LEN*world_size
            step_idx+=1
        state["phase"]=2
        save_state(run_dir,state)

    # ---------- PHASE 2 ----------
    if is_master: logging.info("Phase2 bilingual")
    while state["l2_seen"]<L2_TOTAL:
        for sname in ["l1","l2","l2"]:
            it=l1_iter if sname=="l1" else l2_iter
            batch=[next(it) for _ in range(BATCH_SIZE)]
            loss=train_step(model,optimizer,batch,device,step_idx)
            if sname=="l2": state["l2_seen"]+=BATCH_SIZE*SEQ_LEN*world_size
            else: state["l1_seen"]+=BATCH_SIZE*SEQ_LEN*world_size
            step_idx+=1
            while state["checkpoint_idx"]<len(checkpoint_tokens) and state["l2_seen"]>=checkpoint_tokens[state["checkpoint_idx"]]:
                pct=int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
                path=os.path.join(run_dir,f"checkpoint_{pct}")
                os.makedirs(path,exist_ok=True)
                if is_master:
                    (model.module if is_ddp else model).save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    save_state(run_dir,state)
                    push_to_hub(path,lang,state["checkpoint_idx"])
                state["checkpoint_idx"]+=1

    if is_ddp: dist.destroy_process_group()
    if is_master: logging.info("Training complete.")

# ============================================================
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lang",required=True)
    args=parser.parse_args()
    train(args.lang)
