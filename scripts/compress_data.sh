#!/usr/bin/env bash
#SBATCH --time=24:00:00

# Usage: bash scripts/compress_data.sh [--split train|test|val|all] [archive_prefix]
# Run after data-generation jobs finish so the archive contains stable files.
set -euo pipefail

SPLIT_SELECTION=all
if [[ "${1:-}" == "--split" ]]; then
    SPLIT_SELECTION="${2:-}"
    case "$SPLIT_SELECTION" in
        train|test|val|all) shift 2 ;;
        *) echo "Error: --split requires train, test, val, or all." >&2; exit 1 ;;
    esac
fi
if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/compress_data.sh [--split train|test|val|all] [archive_prefix]"
    echo "Create ZIP archives with matching outputs and parameter files (default: all splits)."
    exit 0
fi
if [[ "$SPLIT_SELECTION" == all ]]; then
    SPLITS=(train test val)
else
    SPLITS=("$SPLIT_SELECTION")
fi

# Slurm runs a copied script from its spool directory.
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
cd -- "$PROJECT_DIR"
ARCHIVE="${1:-cosmoclass_data_$(date +%Y%m%d_%H%M%S)}"
ARCHIVE="${ARCHIVE%.tar.gz}"
ARCHIVE="${ARCHIVE%.zip}"

if [[ ! -d output ]]; then
    echo "Error: output/ does not exist in $PWD." >&2
    echo "Set COSMOCLASS_DIR to the project directory containing output/ and dataset/ before submission." >&2
    exit 1
fi
ARCHIVE_DIR="$(cd -- "$(dirname -- "$ARCHIVE")" && pwd)"
ARCHIVE="$ARCHIVE_DIR/$(basename -- "$ARCHIVE")"
case "$ARCHIVE" in
    "$PWD/output/"*|"$PWD/dataset/"*)
        echo "Error: save the archive outside output/ and dataset/." >&2
        exit 1
        ;;
esac
# Prepare all file lists and check destinations before creating any archives.
TEMP_DIR="$(mktemp -d "$ARCHIVE_DIR/.cosmoclass_archives.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
for SPLIT in "${SPLITS[@]}"; do
    DESTINATION="${ARCHIVE}_${SPLIT}.zip"
    if [[ -e "$DESTINATION" || -L "$DESTINATION" ]]; then
        echo "Error: archive already exists: $DESTINATION" >&2
        exit 1
    fi
    PARAMS_FILE="dataset/neff_${SPLIT}_param.npz"
    if [[ ! -f "$PARAMS_FILE" ]]; then
        echo "Error: missing parameter file: $PARAMS_FILE" >&2
        exit 1
    fi
    PREFIX=classpt_sinu
    if [[ "$SPLIT" != train ]]; then
        PREFIX="classpt_sinu_${SPLIT}"
    fi
    # The digit after the prefix keeps train/test/val files separate.
    find output -maxdepth 1 -type f -name "${PREFIX}_[0-9]*.npz" -print0 > "$TEMP_DIR/$SPLIT.list"
    if [[ ! -s "$TEMP_DIR/$SPLIT.list" ]]; then
        echo "Error: no outputs found for $SPLIT ($PREFIX)." >&2
        exit 1
    fi
    printf '%s\0' "$PARAMS_FILE" >> "$TEMP_DIR/$SPLIT.list"
done

for SPLIT in "${SPLITS[@]}"; do
    DESTINATION="${ARCHIVE}_${SPLIT}.zip"
    echo "Compressing $SPLIT..."
    python3 - "$TEMP_DIR/$SPLIT.list" "$TEMP_DIR/$SPLIT.zip" <<'PY'
import os
import sys
import zipfile

with open(sys.argv[1], "rb") as manifest:
    paths = manifest.read().split(b"\0")
with zipfile.ZipFile(sys.argv[2], "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
    for path in paths:
        if path:
            name = os.fsdecode(path)
            archive.write(name, arcname=name)
PY
    # Publish only a completed archive; never overwrite an existing destination.
    ln -- "$TEMP_DIR/$SPLIT.zip" "$DESTINATION"
    echo "Archive created: $DESTINATION"
    du -h "$DESTINATION"
done
