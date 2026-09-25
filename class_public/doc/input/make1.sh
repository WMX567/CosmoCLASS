#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
 
doxygen doxyconf

cp doxygen.sty ../manual/latex/doxygen.sty
