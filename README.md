# L2LM L1 effects
ACL CDL 2026 Submission 


# Installation
1. Clone the repository
```
git clone https://github.com/suchirsalhan/bilingual-l1-effects
cd bilingual-l1-effects
```
2. Set up the environment
```
python3 -m venv venvs/demo
source venvs/demo/bin/activate
pip install -r requirements.txt
```
3. Authentication
```
hf auth login
```
# Pretraining instructions

1. Train the tokenizer:
```
python ./pretraining/tokenizers/es-en-tok2.py
```
2. Pretokenize the data:
```
python ./pretraining/es-pretokenize.py
```
3. Model pretraining:
```
python ./pretraining/laura_train.py
```
# Instruction tuning
1. Ensure that you have been granted access to datasets.

2. Model instruction tuning:
```
python./instruction_tuning/instruction_tuning_learner_english.py
```
