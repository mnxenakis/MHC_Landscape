#!/bin/bash
#SBATCH -o slurm_colabfold_%j.out
#SBATCH -e slurm_colabfold_%j.err
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 6:00:00

# Usage: sbatch colabfold_script.sh <num_models> <num_seeds> <num_recycle> <max_seq> <max_extra_seq> <FASTA_FILE>

# Parameter inputs (defaults if not provided)
NUM_MODELS=${1:-5}
NUM_SEEDS=${2:-200}
NUM_RECYCLE=${3:-1}
MAX_SEQ=${4:-256}
MAX_EXTRA_SEQ=${5:-512}
# FASTA input (default to what is found here)
if [ -z "$6" ]; then
    FASTA_FILE=$(ls *.fasta 2>/dev/null | head -n 1)
    if [ -z "$FASTA_FILE" ]; then
        echo "No .fasta file found in current directory."
        exit 1
    fi
else
    FASTA_FILE=$6
fi


echo "Using FASTA file: $FASTA_FILE"
echo "num-models: $NUM_MODELS"
echo "num-seeds: $NUM_SEEDS"
echo "num-recycle: $NUM_RECYCLE"
echo "max-seq: $MAX_SEQ"
echo "max-extra-seq: $MAX_EXTRA_SEQ"

# Directory setup
BASENAME=$(basename "$FASTA_FILE" .fasta)
RESULTS_DIR="${BASENAME}_results"
mkdir -p "$RESULTS_DIR"

# Improve speed: disable extra CPU threads fighting GPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Run ColabFold batch
colabfold_batch \
    --num-models "$NUM_MODELS" \
    --num-seeds "$NUM_SEEDS" \
    --num-recycle "$NUM_RECYCLE" \
    --templates \
    --max-seq "$MAX_SEQ" \
    --max-extra-seq "$MAX_EXTRA_SEQ" \
    --use-dropout \
    "$FASTA_FILE" "$RESULTS_DIR" \
    --use-gpu

