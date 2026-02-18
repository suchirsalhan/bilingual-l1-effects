# bilingual-l1-effects
ACL Comp Dev Ling 2026 Submission 

```
git clone https://github.com/suchirsalhan/bilingual-l1-effects
pip install -r requirements.txt 
hf auth login
```

HPC 

```
cd scripts
```

```
for lang in es fr de pl tr ar zh; do sbatch launch_slurm.wilkes3 $lang; done
```
