#!/bin/bash
set -euo pipefail
if [[ -n "${COSMOCLASS_DIR:-}" ]]; then
    PROJECT_DIR="$COSMOCLASS_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -d "$PROJECT_DIR/class_public" && -d "$PROJECT_DIR/../class_public" ]]; then
    PROJECT_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
fi
if [[ ! -f "$PROJECT_DIR/call_class.py" || ! -d "$PROJECT_DIR/class_public" ]]; then
    echo "Set COSMOCLASS_DIR to the CosmoCLASS project root before submission." >&2
    exit 1
fi
export COSMOCLASS_DIR="$PROJECT_DIR"
cd -- "$PROJECT_DIR/class_public/doc/input"
 
doxygen doxyconf

cp doxygen.sty ../manual/latex/doxygen.sty
