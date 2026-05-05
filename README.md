# L2LM L1 effects
ACL Comp Dev Ling 2026 Submission 

```
# Installation
1. Clone the repository
git clone https://github.com/suchirsalhan/bilingual-l1-effects
cd bilingual-l1-effects

2. Set up the environment
python3 -m venv venvs/demo
source venvs/demo/bin/activate
pip install -r requirements.txt

3. Authentication
hf auth login
```
4. Pretraining instructions: 
```
1. train the tokenizer:
python ./pretraining/tokenizers/es-en-tok2.py

2. pretokenize the data:
python ./pretraining/es-pretokenize.py

3. model pretraining: 
python ./pretraining/laura_train.py

