#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:5:00
#SBATCH --output=start.out

set -euo pipefail
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

sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data1.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data2.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data3.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data4.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data5.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data6.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data7.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data8.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data9.sh"
sbatch --chdir="$PROJECT_DIR" "$PROJECT_DIR/data10.sh"
