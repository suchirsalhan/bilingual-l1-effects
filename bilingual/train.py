#!/usr/bin/env python3
import os, json, random
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from huggingface_hub import HfApi, create_repo, upload_folder

# ============================================================
# CONFIG
# ============================================================

SEQ_LEN=512
LR=5e-5
DEVICE="cuda"

TOTAL_TOKENS=20_000_000_000
PHASE1_TOKENS=TOTAL_TOKENS//2
L2_TOTAL=int(TOTAL_TOKENS*(1/3))
L1_PHASE2=int(TOTAL_TOKENS*(1/6))

CHECKPOINT_FRACS=[0.25,0.5,0.75,1.0]
CHECKPOINT_TOKENS=[int(L2_TOTAL*x) for x in CHECKPOINT_FRACS]

HF_USER="suchirsalhan"
SAVE_DIR="./runs"
STATE_FILE="trainer_state.json"

LEVEL_NAMES={
    0:"beginner",
    1:"intermediate",
    2:"advanced",
    3:"fluent"
}

os.makedirs(SAVE_DIR,exist_ok=True)

# ============================================================
# TOKENIZER / MODEL
# ============================================================

tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token

def load_or_create_model(run_dir):
    if os.path.exists(os.path.join(run_dir,"pytorch_model.bin")):
        print("Resuming model weights")
        model=GPT2LMHeadModel.from_pretrained(run_dir)
    else:
        model=GPT2LMHeadModel.from_pretrained("gpt2")
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
    path=os.path.join(run_dir,STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            s=json.load(f)
        print("Loaded trainer state")
        return s
    return default_state()

def save_state(run_dir,state):
    state["rng"]=random.getstate()
    state["torch_rng"]=torch.get_rng_state().tolist()
    state["cuda_rng"]=torch.cuda.get_rng_state().tolist()

    with open(os.path.join(run_dir,STATE_FILE),"w") as f:
        json.dump(state,f)

# ============================================================
# DATA STREAMS (resume-safe)
# ============================================================

def cultura_stream(lang,skip_tokens=0):
    ds=load_dataset("uonlp/CulturaX",lang,split="train",streaming=True)
    seen=0
    for ex in ds:
        ids=tokenizer(ex["text"])["input_ids"]
        if seen+len(ids)<skip_tokens:
            seen+=len(ids)
            continue
        yield ids

def fineweb_stream(skip_tokens=0):
    ds=load_dataset("HuggingFaceFW/fineweb-edu",split="train",streaming=True)
    seen=0
    for ex in ds:
        ids=tokenizer(ex["text"])["input_ids"]
        if seen+len(ids)<skip_tokens:
            seen+=len(ids)
            continue
        yield ids

def pack_blocks(token_stream):
    buf=[]
    for tokens in token_stream:
        buf.extend(tokens)
        while len(buf)>=SEQ_LEN:
            yield buf[:SEQ_LEN]
            buf=buf[SEQ_LEN:]

# ============================================================
# HF PUSH
# ============================================================

def push_to_hub(path,lang,ckpt_idx):
    level=LEVEL_NAMES[ckpt_idx]
    repo_id=f"{HF_USER}/B-GPT-{lang}-fineweb-{level}"

    print(f"Pushing → {repo_id}")

    api=HfApi()
    create_repo(repo_id,exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path=path,
        commit_message=f"{level} learner checkpoint"
    )

# ============================================================
# TRAIN STEP
# ============================================================

def step(model,optim,x,y):
    loss=model(x,labels=y).loss
    loss.backward()
    optim.step()
    optim.zero_grad()

# ============================================================
# TRAIN LOOP
# ============================================================

def train(lang):

    run_dir=f"{SAVE_DIR}/{lang}"
    os.makedirs(run_dir,exist_ok=True)

    model=load_or_create_model(run_dir)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)

    state=load_state(run_dir)

    # restore RNG
    random.setstate(tuple(state["rng"]))
    torch.set_rng_state(torch.tensor(state["torch_rng"],dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(state["cuda_rng"],dtype=torch.uint8))

    l1_stream=pack_blocks(cultura_stream(lang,state["l1_seen_phase1"]+state["l1_seen_phase2"]))
    l2_stream=pack_blocks(fineweb_stream(state["l2_seen"]))

    # ---------------- PHASE 1 ----------------
    if state["phase"]==1:
        print("Phase 1: L1 only")

        for block in l1_stream:
            x=torch.tensor(block[:-1]).unsqueeze(0).to(DEVICE)
            y=torch.tensor(block[1:]).unsqueeze(0).to(DEVICE)

            step(model,optimizer,x,y)

            state["l1_seen_phase1"]+=len(block)

            if state["l1_seen_phase1"]>=PHASE1_TOKENS:
                state["phase"]=2
                save_state(run_dir,state)
                break

            if state["l1_seen_phase1"]%(50_000_000)==0:
                save_state(run_dir,state)

    # ---------------- PHASE 2 ----------------
    print("Phase 2: bilingual acquisition")

    l1_iter=iter(l1_stream)
    l2_iter=iter(l2_stream)

    while state["l2_seen"]<L2_TOTAL:

        for lang_id in ["l1","l2","l2"]:

            block=next(l1_iter) if lang_id=="l1" else next(l2_iter)

            x=torch.tensor(block[:-1]).unsqueeze(0).to(DEVICE)
            y=torch.tensor(block[1:]).unsqueeze(0).to(DEVICE)

            step(model,optimizer,x,y)

            if lang_id=="l2":
                state["l2_seen"]+=len(block)
            else:
                state["l1_seen_phase2"]+=len(block)

            # -------- CHECKPOINT --------
            while state["checkpoint_idx"]<len(CHECKPOINT_TOKENS) and \
                  state["l2_seen"]>=CHECKPOINT_TOKENS[state["checkpoint_idx"]]:

                pct=int(CHECKPOINT_FRACS[state["checkpoint_idx"]]*100)
                level=LEVEL_NAMES[state["checkpoint_idx"]]

                path=f"{run_dir}/checkpoint_{pct}"
                os.makedirs(path,exist_ok=True)

                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                save_state(run_dir,state)

                print(f"\n=== {level.upper()} ===")
                print(f"L2 seen: {state['l2_seen']:,}")
                print(f"L1 seen: {state['l1_seen_phase2']:,}")
                print(f"ratio: {state['l1_seen_phase2']/state['l2_seen']:.3f}\n")

                push_to_hub(path,lang,state["checkpoint_idx"])

                state["checkpoint_idx"]+=1

        if state["l2_seen"]%(50_000_000)==0:
            save_state(run_dir,state)

    print("Training complete")

# ============================================================
# RUN ALL LANGUAGES
# ============================================================

if __name__=="__main__":
    for lang in ["es","fr","de","pl","tr","ar","zh"]:
        train(lang)

