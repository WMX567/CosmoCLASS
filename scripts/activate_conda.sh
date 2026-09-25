#!/bin/bash
# Source this file from a job script. All settings are inherited by sbatch.

# Only load a cluster module when its actual name is explicitly provided.
if [[ -n "${COSMOCLASS_CONDA_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo "The module command is unavailable; initialize the cluster module system or set COSMOCLASS_CONDA_BASE." >&2
        return 1
    fi
    module load "$COSMOCLASS_CONDA_MODULE" || return 1
fi

if [[ -n "${COSMOCLASS_CONDA_BASE:-}" ]]; then
    cosmoclass_conda_exe="$COSMOCLASS_CONDA_BASE/bin/conda"
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    cosmoclass_conda_exe="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    cosmoclass_conda_exe=conda
else
    echo "Conda was not found. Export COSMOCLASS_CONDA_BASE=/path/to/conda before sbatch, or set COSMOCLASS_CONDA_MODULE to an available cluster module." >&2
    return 1
fi

cosmoclass_conda_hook="$("$cosmoclass_conda_exe" shell.bash hook)" || return 1
eval "$cosmoclass_conda_hook" || return 1
conda activate "${COSMOCLASS_CONDA_ENV:-multinest}" || return 1
unset cosmoclass_conda_exe cosmoclass_conda_hook
