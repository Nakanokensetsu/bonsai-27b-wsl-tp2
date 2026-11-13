#!/usr/bin/env bash
# Download the vLLM-format Bonsai 2 27B (~9.6 GB) and add the vision tower (~0.9 GB) so images work.
# usage: ./model/prepare_model.sh [model_dir]   (default: ./models/bonsai-27b)
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-$PWD/models/bonsai-27b}
PY=venv/bin/python

venv/bin/hf download fraserprice/Ternary-Bonsai-2-27B-vllm --local-dir "$DIR"
cp model/preprocessor_config.json model/video_preprocessor_config.json "$DIR/"
grep -q '"model.visual.patch_embed.proj.weight"' "$DIR/model.safetensors.index.json" || $PY model/fetch_vision.py "$DIR"

# MTP head re-trained on Japanese (draft acceptance 65.9% -> 70.0%). Set JA_MTP=0 to keep the original head.
if [ "${JA_MTP:-1}" = 1 ] && [ ! -e "$DIR/model-00003-of-00003.safetensors.orig" ]; then
  venv/bin/hf download nakakennnn/Ternary-Bonsai-2-27B-MTP-ja model_mtp.safetensors --local-dir "$PWD/mtp_ja"
  $PY model/swap_mtp.py "$DIR" "$PWD/mtp_ja/model_mtp.safetensors"
fi

# Optional: token embeddings on SSD (frees ~0.9 GiB per GPU for the KV cache)
$PY ssd_embed/export_embed.py "$DIR" "$PWD/embed_ssd"
echo "model ready: $DIR"
