#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=replacement_44141.out

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
if [[ ! -f "$PROJECT_DIR/call_class.py" || ! -f "$PROJECT_DIR/call_class_pk.py" || ! -d "$PROJECT_DIR/class_public" ]]; then
    echo "Set COSMOCLASS_DIR to the CosmoCLASS project root before submission." >&2
    exit 1
fi
export COSMOCLASS_DIR="$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd -- "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multinest

python - <<'PY'
from pathlib import Path
import numpy as np
from lhs_sampling import PARAM_RANGES, perform_lhs_sampling

path = Path("dataset/replacement_44141/params.npz")
if path.exists():
    print(f"Reusing parameters: {path}", flush=True)
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = perform_lhs_sampling(PARAM_RANGES, 1, seed=44141)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **samples)
    print(f"Generated parameters (seed=44141): {path}", flush=True)
PY

python -u call_class.py --resume --start 0 --end 1 --index-offset 44141 \
    --params-file dataset/replacement_44141/params.npz \
    --out-dir output/replacement_44141
