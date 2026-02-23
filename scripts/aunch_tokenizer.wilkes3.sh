#!/bin/bash
#SBATCH -J tokenizer
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-00:35:00
#SBATCH --exclusive
#SBATCH --output=tokenizer_%j.log

# ---------------- Modules ----------------
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# ---------------- Paths ----------------
PROJECT=/rds/user/sas245/hpc-work/bilingual-l1-effects
VENV=$PROJECT/demo311

cd $PROJECT
source $VENV/bin/activate

echo "Host: $(hostname)"
echo "PWD: $(pwd)"
which python
python -V

# ---------------- NCCL (Wilkes3 safe) ----------------
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=true

# Run tokenizer training
python3 bilingual/train_bilingual_tokenizers.py --lang "$1" --source "$2"
