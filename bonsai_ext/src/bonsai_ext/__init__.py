"""Bonsai additions for a stock vLLM 0.34 install (vllm.general_plugins entry point).

vLLM itself stays unmodified. What this plugin adds and through which hook:
  - OSCAR KV dtypes (turboquant_k*v*_oscar): register_kv_cache_dtype + the TurboQuant
    attention modules replaced by the vendored copies in bonsai_ext/vendor (they carry
    vllm#53060, vllm#53410, tiled continuation prefill and OSCAR).
  - WSL2 KV-offload device-pointer fix: vendored vllm.v1.kv_offload.cpu.gpu_worker.
  - Scheduler (step prefill cap / solo LPT / priority preemption): --scheduler-cls
    bonsai_ext.scheduler.BonsaiScheduler
  - VRAM logger + per-rank allocation cap: --worker-cls bonsai_ext.worker.BonsaiWorker
  - SSD embeddings (VLLM_SSD_EMBED_DIR), vision tower CPU offload
    (VLLM_VISION_CPU_OFFLOAD_GB), min_p / logit_bias with MTP: small wrappers below.
Every piece is a no-op unless its env var / CLI flag is set, except the module
replacement, which only matters when a turboquant_* KV dtype is used.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("bonsai_ext")
_VENDOR = Path(__file__).with_name("vendor")

# upstream module name -> vendored file
VENDORED = {
    "vllm.model_executor.layers.quantization.turboquant.config": "tq_config.py",
    "vllm.v1.attention.ops.triton_turboquant_decode": "triton_turboquant_decode.py",
    "vllm.v1.attention.ops.triton_turboquant_store": "triton_turboquant_store.py",
    "vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode": "soa_triton_turboquant_decode.py",
    "vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store": "soa_triton_turboquant_store.py",
    "vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention": "soa_triton_turboquant_unified_attention.py",
    "vllm.v1.attention.ops.oscar_kv": "oscar_kv.py",
    "vllm.v1.attention.backends.turboquant_attn": "turboquant_attn.py",
    "vllm.v1.kv_offload.cpu.gpu_worker": "kv_offload_gpu_worker.py",
}
# BONSAI_OFFLOAD_IMPL=pr61054: use upstream PR #61054's device-pointer fix instead of ours (to test it).
if os.environ.get("BONSAI_OFFLOAD_IMPL") == "pr61054":
    VENDORED["vllm.v1.kv_offload.cpu.gpu_worker"] = "kv_offload_gpu_worker_pr61054.py"

OSCAR_DTYPES = (
    "turboquant_k4v3_oscar",
    "turboquant_k4v2_oscar",
    "turboquant_k3v3_oscar",
    "turboquant_k3v2_oscar",
    "turboquant_k2v2_oscar",
)

# Our env vars, declared so vLLM does not warn about unknown VLLM_* variables.
OWN_ENVS = {
    "VLLM_SOLO_LPT": lambda: int(os.environ.get("VLLM_SOLO_LPT", "0")),
    "VLLM_STEP_PREFILL_CAP": lambda: int(os.environ.get("VLLM_STEP_PREFILL_CAP", "0")),
    "VLLM_PREFILL_EVERY": lambda: int(os.environ.get("VLLM_PREFILL_EVERY", "0")),
    "VLLM_PRIO_PREEMPT": lambda: int(os.environ.get("VLLM_PRIO_PREEMPT", "0")),
    "VLLM_PRIO_KV_THRESH": lambda: float(os.environ.get("VLLM_PRIO_KV_THRESH", "0")),
    "VLLM_PRIO_KV_GAP": lambda: float(os.environ.get("VLLM_PRIO_KV_GAP", "1.0")),
    "VLLM_SSD_EMBED_DIR": lambda: os.environ.get("VLLM_SSD_EMBED_DIR", ""),
    "VLLM_VISION_CPU_OFFLOAD_GB": lambda: float(os.environ.get("VLLM_VISION_CPU_OFFLOAD_GB", "0")),
    "VLLM_TQ_TILED_CONT": lambda: int(os.environ.get("VLLM_TQ_TILED_CONT", "0")),
    "VLLM_TQ_GROUPED_DECODE": lambda: os.environ.get("VLLM_TQ_GROUPED_DECODE", ""),
}

_done = False


def _install_vendored() -> None:
    for name, fname in VENDORED.items():
        if name in sys.modules:
            mod = sys.modules[name]
            if getattr(mod, "__bonsai_ext__", False):
                continue
            # Imported before the plugin ran: replacing it now would leave two copies alive.
            raise RuntimeError(f"bonsai_ext: {name} was imported before the plugin loaded")
        spec = importlib.util.spec_from_file_location(name, _VENDOR / fname)
        mod = importlib.util.module_from_spec(spec)
        mod.__bonsai_ext__ = True
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        spec.loader.exec_module(mod)
        setattr(importlib.import_module(parent), child, mod)


def _register_oscar_dtypes() -> None:
    import torch
    from vllm.config.cache import register_kv_cache_dtype
    from vllm.v1.kv_cache_interface import KVQuantMode

    for name in OSCAR_DTYPES:
        # The generic mode only has to say "TurboQuant packed slot"; the vendored backend
        # reads the real preset (bits, window) from cache_config.cache_dtype.
        cls = type(
            "OscarHandler_" + name,
            (),
            {
                "name": name,
                "torch_dtype": lambda self: torch.uint8,
                "quant_mode": lambda self: KVQuantMode.TURBOQUANT_4BIT_NC,
            },
        )
        register_kv_cache_dtype(name)(cls)


def _register_envs() -> None:
    import vllm.envs as envs

    for k, v in OWN_ENVS.items():
        envs.environment_variables.setdefault(k, v)


def _patch_min_p_with_spec() -> None:
    # Model Runner V2 (the only runner in 0.34) verifies drafts through
    # Sampler.apply_sampling_params with the per-token expanded idx mapping, so min_p and
    # logit_bias already apply at every draft position. Upstream still rejects them.
    from vllm.sampling_params import SamplingParams

    if getattr(SamplingParams._validate_spec_decode, "__bonsai_ext__", False):
        return

    def _validate_spec_decode(self, speculative_config):
        return None

    _validate_spec_decode.__bonsai_ext__ = True
    SamplingParams._validate_spec_decode = _validate_spec_decode


def register() -> None:
    global _done
    if _done:
        return
    _done = True
    _register_envs()
    _install_vendored()
    _register_oscar_dtypes()
    _patch_min_p_with_spec()
    from bonsai_ext import models

    models.install()
    logger.info("bonsai_ext: loaded (vendored %d modules, %d OSCAR dtypes)", len(VENDORED), len(OSCAR_DTYPES))
