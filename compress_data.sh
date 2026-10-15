#!/usr/bin/env bash
# Usage: bash compress_data.sh [archive.tar.gz]
# Run after data-generation jobs finish so the archive contains stable files.
set -euo pipefail

if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash compress_data.sh [archive.tar.gz]"
    echo "Archive output/ and dataset/ (if present), preserving the original files."
    exit 0
fi

PROJECT_DIR="${COSMOCLASS_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd -- "$PROJECT_DIR"
ARCHIVE="${1:-cosmoclass_data_$(date +%Y%m%d_%H%M%S).tar.gz}"

if [[ ! -d output ]]; then
    echo "Error: output/ does not exist in $PWD." >&2
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
if [[ -e "$ARCHIVE" || -L "$ARCHIVE" ]]; then
    echo "Error: archive already exists: $ARCHIVE" >&2
    exit 1
fi

DIRECTORIES=(output)
if [[ -d dataset ]]; then
    DIRECTORIES+=(dataset)
fi

# Build beside the destination, then publish only the completed archive.
TEMP_ARCHIVE="$(mktemp "$ARCHIVE_DIR/.cosmoclass_archive.XXXXXX")"
trap 'rm -f -- "$TEMP_ARCHIVE"' EXIT
echo "Compressing: ${DIRECTORIES[*]}"
tar -czf "$TEMP_ARCHIVE" "${DIRECTORIES[@]}"
# A hard link prevents overwriting an archive created by another invocation.
ln -- "$TEMP_ARCHIVE" "$ARCHIVE"
echo "Archive created: $ARCHIVE"
du -h "$ARCHIVE"
