#!/usr/bin/env bash
# Serve Ternary-Bonsai-2-27B on 2x 12 GB GPUs (TP=2) with an OpenAI-compatible API.
# Every feature below is a plain environment variable; set it to 0/empty to turn it off.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

MODEL_DIR=${MODEL_DIR:-$ROOT/models/bonsai-27b}
PORT=${PORT:-8000}
MEM=${MEM:-0.72}          # lower it if the GPU also drives a display
SEQS=${SEQS:-24}          # requests decoded at the same time
export KV=${KV:-turboquant_k3v2_oscar}   # exported: bonsai_ext reads it when sizing the KV page. or turboquant_4bit_nc (vLLM's TurboQuant) / auto (bf16)
TEMPLATE=${TEMPLATE:-}    # optional chat template (.jinja)

C=$ROOT/cuda130/nvidia/cu13
export CUDA_HOME=$C PATH=$C/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TORCH_EXTENSIONS_DIR=$ROOT/torch_ext034 TORCH_CPP_LOG_LEVEL=ERROR
export MAX_JOBS=${MAX_JOBS:-1}   # first start compiles CUDA kernels; parallel nvcc can exhaust RAM

# plugin (prism-ternary fork)
export PRISM_INT8_PREFILL=${PRISM_INT8_PREFILL:-1}   # int8 GEMM for prefill
export PRISM_MTP_BF16=${PRISM_MTP_BF16:-0}           # 0 = MTP head in fp8 (Bonsai 2.5 ships it as Q4_0)
export PRISM_DP4A_DECODE=${PRISM_DP4A_DECODE:-1}     # dp4a GEMV when <=4 rows
export PRISM_AR_INT8=${PRISM_AR_INT8:-1}             # int8 all-reduce during prefill (TP=2)
export PRISM_BA_COLMAJOR=${PRISM_BA_COLMAJOR:-1}     # column-major GDN in_proj_ba (2-8 rows: 20us -> 4us)
export PRISM_FUSED_I8_MAX=${PRISM_FUSED_I8_MAX:-384} # int8 prefill without expanding the weight, 65..384 rows
# scheduler (bonsai_ext.scheduler)
export VLLM_SOLO_LPT=${VLLM_SOLO_LPT:-1584}          # prefill tokens/step when nobody is decoding
export VLLM_STEP_PREFILL_CAP=${VLLM_STEP_PREFILL_CAP:-256}  # prefill tokens/step while others decode
export VLLM_PREFILL_EVERY=${VLLM_PREFILL_EVERY:-3}   # mix prefill into 1 of every 3 steps
export VLLM_PRIO_PREEMPT=${VLLM_PRIO_PREEMPT:-1}     # preempt lower-priority requests when slots are full
export VLLM_PRIO_KV_THRESH=${VLLM_PRIO_KV_THRESH:-0.95}   # ...or when KV usage is above this
# memory
export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}  # vision tower weights in host RAM
[ -d "$ROOT/embed_ssd" ] && export VLLM_SSD_EMBED_DIR=${VLLM_SSD_EMBED_DIR-$ROOT/embed_ssd}
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export BONSAI_VRAM_LOG_DIR=${BONSAI_VRAM_LOG_DIR:-$ROOT/logs}; mkdir -p "$BONSAI_VRAM_LOG_DIR"  # per-GPU VRAM log once a minute
export PYTHONPATH=$ROOT/script_ban${PYTHONPATH:+:$PYTHONPATH}
export SCRIPT_BAN_TOKCLASS=$ROOT/script_ban/tokclass.json

GDN_PREFILL=${GDN_PREFILL:-triton}   # GDN prefill kernel. flashinfer adds a prefill checkpoint per request (-9% KV pool here, same speed)
ARGS=()
[ -n "$TEMPLATE" ] && ARGS+=(--chat-template "$TEMPLATE")
if [ "${KV_OFFLOAD:-1}" = 1 ]; then   # KV that falls out of the GPU goes to 2 GiB RAM, then SSD
  mkdir -p "$ROOT/kvcache_ssd"
  export BONSAI_KV_FS_TOUCH=${BONSAI_KV_FS_TOUCH:-1}   # bonsai_ext: refresh block mtime on use, so the prune below keeps hot blocks
  # vLLM never deletes SSD blocks: prune every 10 min while this process (vLLM after exec) lives.
  # Limits: KV_PRUNE_CAP_GB (default 100), KV_PRUNE_AGE_MIN (default 120). KV_PRUNE=0 to disable.
  if [ "${KV_PRUNE:-1}" = 1 ]; then
    ( me=$$; while kill -0 "$me" 2>/dev/null; do
        "$ROOT/kvcache_prune.sh" "$ROOT/kvcache_ssd"; sleep "${KV_PRUNE_INTERVAL_SEC:-600}"
      done ) >>"$ROOT/kvcache_prune.log" 2>&1 &
  fi
  ARGS+=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":2147483648,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$ROOT/kvcache_ssd\"}]}}")
fi

exec venv/bin/vllm serve "$MODEL_DIR" --served-model-name bonsai-27b \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}' \
  --tensor-parallel-size 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --kv-cache-dtype "$KV" --mamba-ssm-cache-dtype bfloat16 --mamba-backend flashinfer \
  --max-num-batched-tokens 3584 --long-prefill-token-threshold 256 \
  --prefix-match-unit 16 --prefix-caching-hash-algo xxhash --enable-mamba-shared-prefix-checkpoint \
  --attention-config '{"tq_max_kv_splits_for_cuda_graph":64}' \
  --logits-processors script_ban:ScriptBanProcessor \
  --scheduling-policy priority \
  --scheduler-cls bonsai_ext.scheduler.BonsaiScheduler --worker-cls bonsai_ext.worker.BonsaiWorker \
  --no-enable-extensible-kv-cache --gdn-prefill-backend "$GDN_PREFILL" \
  --max-model-len 131072 --gpu-memory-utilization "$MEM" --max-num-seqs "$SEQS" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" "${ARGS[@]}"
