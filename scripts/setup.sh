#!/usr/bin/env bash
set -euo pipefail

# Ensure a CUDA-capable GPU is available
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
PY

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" ..
make -j"$(nproc)"

# Move the generated extension into the Python package directory
mv chess_engine*.so ../src/python/
