#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

cd ../manual/latex

make

make

cp refman.pdf ../CLASS_MANUAL.pdf
cd ../../input
