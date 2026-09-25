#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=48:00:00
#SBATCH --output=14.out

set -eo pipefail
if [[ -n "${COSMOCLASS_DIR:-}" ]]; then
    PROJECT_DIR="$COSMOCLASS_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -d "$PROJECT_DIR/class_public" && -d "$PROJECT_DIR/../class_public" ]]; then
    PROJECT_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
fi
if [[ ! -f "$PROJECT_DIR/call_class.py" || ! -d "$PROJECT_DIR/class_public" ]]; then
    echo "Set COSMOCLASS_DIR to the CosmoCLASS project root before submission." >&2
    exit 1
fi
export COSMOCLASS_DIR="$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd -- "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multinest

python call_class.py --start 26000 --end 28000 --params-file dataset/neff_train_param.npz
