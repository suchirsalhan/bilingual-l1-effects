!pip install transformers datasets torch tqdm

from datasets import get_dataset_config_names, load_dataset

# Get all BLiMP task names
configs = get_dataset_config_names("blimp")

print(f"Found {len(configs)} tasks")

# Load all tasks into a dictionary
dataset = {config: load_dataset("blimp", config) for config in configs}


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "lgb35/B-GPT-es-fineweb-beginner-alpacaEnglish" #change to the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

def score_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()

configs = get_dataset_config_names("blimp")

results = {}

for config in configs:
    dataset = load_dataset("blimp", config)
    correct = 0
    total = 0

    for example in tqdm(dataset["train"], desc=config):
        good_score = score_sentence(example["sentence_good"])
        bad_score = score_sentence(example["sentence_bad"])

        if good_score > bad_score:
            correct += 1
        total += 1

    acc = correct / total
    results[config] = acc
    print(f"{config}: {acc:.3f}")

overall = sum(results.values()) / len(results)
print("\nOverall BLiMP Accuracy:", overall)
