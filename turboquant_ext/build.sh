#!/bin/bash
# Optional: CUDA fast path for TurboQuant 4bit_nc decode (vllm#53060). Not needed for the OSCAR KV formats.
# Usage: CUDA_HOME=/path/to/cuda-13.0 ./build.sh   (run with the venv's python on PATH)
set -e
cd "$(dirname "$0")"
: "${CUDA_HOME:?set CUDA_HOME to your CUDA 13.0 toolkit}"
export PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=1
python setup.py build_ext --inplace
SITE=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
cp _turboquant_C*.so "$SITE/"
echo "installed into $SITE"
