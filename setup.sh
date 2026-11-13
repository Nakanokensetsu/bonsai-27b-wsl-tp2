#!/usr/bin/env bash
# Install vLLM 0.32.0 + the Bonsai patches + the prism-ternary plugin fork into ./venv
# Needs: uv (https://docs.astral.sh/uv/), git, patch, an NVIDIA driver that supports CUDA 13.0, ~25 GB free disk.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

# 1. Python venv with vLLM 0.32.0, torch 2.13.0 (CUDA 13.0) and every other package pinned to the versions I run
[ -d venv ] || uv venv --python 3.12 venv
uv pip install --python venv/bin/python -r requirements-lock.txt

# 2. CUDA 13.0 compiler for the plugin's JIT kernels (pip wheels, no system CUDA install needed)
[ -d cuda130 ] || uv pip install --python venv/bin/python --link-mode=copy --target cuda130 \
  "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-cccl==13.0.85" \
  "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
# torch's headers also need cuBLAS/cuSPARSE/... headers: take them from the CUDA wheels torch already installed
NV=$(venv/bin/python -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13','include'))")
cp -rn "$NV"/. cuda130/nvidia/cu13/include/ 2>/dev/null   # add only the headers the toolkit lacks (never overwrite)
ln -sf libcudart.so.13 cuda130/nvidia/cu13/lib/libcudart.so
ln -sfn lib cuda130/nvidia/cu13/lib64      # flashinfer's JIT links from $CUDA_HOME/lib64
for d in /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu; do   # driver library (WSL2 first)
  [ -e "$d/libcuda.so" ] && { ln -sf "$d/libcuda.so" cuda130/nvidia/cu13/lib/libcuda.so; break; }
done

# 3. Patch vLLM in place (OSCAR KV, priority preemption, step prefill cap, SSD embeddings, WSL2 offload fixes, ...)
SITE=$(venv/bin/python -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
if patch -p1 -d "$SITE" --dry-run -R -s < vllm-patch/vllm-0.32.0-bonsai.patch >/dev/null 2>&1; then
  echo "vLLM patch already applied"
else
  patch -p1 -d "$SITE" -s < vllm-patch/vllm-0.32.0-bonsai.patch
  echo "vLLM patched: $SITE/vllm"
fi

# 4. The quantized-weight plugin (TP=2, int8 prefill, dp4a decode, int8 all-reduce)
[ -d bonsai-vllm ] || git clone -b bonsai-tp2-wsl "${FORK_URL:-https://github.com/Nakanokensetsu/bonsai-vllm.git}" bonsai-vllm
uv pip install --python venv/bin/python -e ./bonsai-vllm

venv/bin/python -c "import vllm, prism_ternary; print('vllm', vllm.__version__, '/ plugin ok')"
echo "done. next: ./model/prepare_model.sh"
