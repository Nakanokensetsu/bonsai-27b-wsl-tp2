"""Model-side additions, applied by swapping constructor names inside the model modules.

- VLLM_SSD_EMBED_DIR: Qwen3.5 / Qwen3.5-MTP embed_tokens become SSDEmbedding (table on SSD,
  frequent rows on the GPU). Swapped at construction, so the full table is never allocated.
- VLLM_VISION_CPU_OFFLOAD_GB: the Qwen3-VL vision tower's blocks and final merger are moved
  to pinned CPU memory as each is built (bulk copy per forward, not UVA zero-copy).
Both are no-ops when their env var is unset.
"""

import logging
import os

_installed = False


def _ssd_embedding_factory(orig):
    def make(num_embeddings, embedding_dim, *args, **kwargs):
        from bonsai_ext.ssd_embedding import SSDEmbedding, ssd_embed_dir

        d = ssd_embed_dir()
        import logging
        logging.getLogger("vllm").info("bonsai_ext: embed factory pid=%s dir=%r n=%s", os.getpid(), d, num_embeddings)
        if d:
            return SSDEmbedding(num_embeddings, embedding_dim, d)
        return orig(num_embeddings, embedding_dim, *args, **kwargs)

    make.__bonsai_ext__ = True
    return make


_vision_offloader = None


def _vision_offloader_get():
    global _vision_offloader
    if _vision_offloader is None:
        from vllm.model_executor.offloader.uva import UVAOffloader

        gb = float(os.environ.get("VLLM_VISION_CPU_OFFLOAD_GB", "0") or 0)
        _vision_offloader = UVAOffloader(int(gb * 1024**3))
        # Bulk-copy each module for its forward. UVA zero-copy makes every GEMM reread
        # operand tiles over PCIe and is much slower here.
        _vision_offloader.uva_offloading = False
    return _vision_offloader


def _vision_factory(orig, only_suffix=None):
    def make(*args, **kwargs):
        m = orig(*args, **kwargs)
        if float(os.environ.get("VLLM_VISION_CPU_OFFLOAD_GB", "0") or 0) <= 0:
            return m
        prefix = kwargs.get("prefix", "")
        if only_suffix is not None and not prefix.endswith(only_suffix):
            return m
        return _vision_offloader_get().wrap_modules(iter([m]))[0]

    make.__bonsai_ext__ = True
    return make


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    logging.getLogger("vllm").info("bonsai_ext: models.install pid=%s ssd=%r", os.getpid(), os.environ.get("VLLM_SSD_EMBED_DIR"))
    if os.environ.get("VLLM_SSD_EMBED_DIR"):
        import importlib

        from vllm.model_executor.models import qwen3_5, qwen3_5_mtp

        mods = [qwen3_5, qwen3_5_mtp]
        # 0.34+: on CUDA, Qwen3.5 lives in vllm.models.qwen3_5.nvidia (new eager+breakable-graph code).
        for name in ("vllm.models.qwen3_5.nvidia.model", "vllm.models.qwen3_5.nvidia.mtp"):
            try:
                mods.append(importlib.import_module(name))
            except Exception as e:  # non-CUDA platforms do not ship it
                logging.getLogger("vllm").info("bonsai_ext: %s not patched (%s)", name, e)
        for mod in mods:
            if not getattr(mod.VocabParallelEmbedding, "__bonsai_ext__", False):
                mod.VocabParallelEmbedding = _ssd_embedding_factory(mod.VocabParallelEmbedding)
    if float(os.environ.get("VLLM_VISION_CPU_OFFLOAD_GB", "0") or 0) > 0:
        from vllm.model_executor.models import qwen3_vl

        if not getattr(qwen3_vl.Qwen3_VisionBlock, "__bonsai_ext__", False):
            qwen3_vl.Qwen3_VisionBlock = _vision_factory(qwen3_vl.Qwen3_VisionBlock)
            # Only the final merger, not the deepstack mergers (same as the 0.32 patch).
            qwen3_vl.Qwen3_VisionPatchMerger = _vision_factory(
                qwen3_vl.Qwen3_VisionPatchMerger, only_suffix=".merger"
            )
