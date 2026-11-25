#!/usr/bin/env bash
# Download the vLLM-format Ternary-Bonsai-2.5-27B (~11 GB, vision tower and MTP head included).
# usage: ./model/prepare_model.sh [model_dir]   (default: ./models/bonsai-27b)
#   MODEL=2 ./model/prepare_model.sh   -> previous Bonsai 2 27B (+ vision tower + Japanese MTP head)
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-$PWD/models/bonsai-27b}
PY=venv/bin/python

if [ "${MODEL:-2.5}" = 2.5 ]; then
  # Converted from prism-ml/Ternary-Bonsai-2.5-27B-GGUF (PQ2_0-mtp). Its MTP head is used as is.
  venv/bin/hf download nakakennnn/Ternary-Bonsai-2.5-27B-vllm --local-dir "$DIR" --exclude "*.py"
else
  venv/bin/hf download fraserprice/Ternary-Bonsai-2-27B-vllm --local-dir "$DIR"
  cp model/preprocessor_config.json model/video_preprocessor_config.json "$DIR/"
  grep -q '"model.visual.patch_embed.proj.weight"' "$DIR/model.safetensors.index.json" || $PY model/fetch_vision.py "$DIR"
  # MTP head re-trained on Japanese (draft acceptance 65.9% -> 70.0%). Set JA_MTP=0 to keep the original head.
  if [ "${JA_MTP:-1}" = 1 ] && [ ! -e "$DIR/model-00003-of-00003.safetensors.orig" ]; then
    venv/bin/hf download nakakennnn/Ternary-Bonsai-2-27B-MTP-ja model_mtp.safetensors --local-dir "$PWD/mtp_ja"
    $PY model/swap_mtp.py "$DIR" "$PWD/mtp_ja/model_mtp.safetensors"
  fi
fi

# Optional: token embeddings on SSD (frees ~0.9 GiB per GPU for the KV cache). Re-run after switching models.
$PY ssd_embed/export_embed.py "$DIR" "$PWD/embed_ssd"
echo "model ready: $DIR"
