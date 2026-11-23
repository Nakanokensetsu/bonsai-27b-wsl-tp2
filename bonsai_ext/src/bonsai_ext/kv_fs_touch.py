"""Keep SSD KV-block timestamps equal to "last used" (BONSAI_KV_FS_TOUCH=1).

vLLM's fs tier has no capacity limit, so the directory has to be pruned externally (e.g. a
timer that deletes blocks oldest-first by max(atime, mtime) above a size cap). Stock vLLM
never refreshes those timestamps for a block in use:
  - store of a block whose file already exists is skipped (io._store_block / fs_io_C),
    so mtime stays at the first write;
  - loads on a relatime mount update atime only once (atime > mtime afterwards).
So a hot block (shared system prompt, long document asked about again) looked old and was
deleted first, while one-shot blocks survived. Here every stored or loaded block gets
os.utime(now) after the transfer, on the pool worker thread (one syscall per block).
"""

import contextlib
import logging
import os

logger = logging.getLogger("bonsai_ext")


def _touch(paths) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            os.utime(p, None)


def install() -> None:
    if os.environ.get("BONSAI_KV_FS_TOUCH", "0") != "1":
        return
    from vllm.v1.kv_offload.tiering.fs import manager as fsm

    if getattr(fsm.batch_store_block, "__bonsai_ext__", False):
        return
    orig_store = fsm.batch_store_block
    orig_load = fsm.batch_load_block

    def batch_store_block(paths, *args, **kwargs):
        try:
            return orig_store(paths, *args, **kwargs)
        finally:
            # New files already have mtime=now; this refreshes the skipped existing ones.
            # On failure, touching an absent path is a no-op.
            _touch(paths)

    def batch_load_block(paths, *args, **kwargs):
        try:
            ret = orig_load(paths, *args, **kwargs)
        except OSError as exc:
            _touch(paths[: getattr(exc, "num_succeeded", 0)])
            raise
        _touch(paths)
        return ret

    batch_store_block.__bonsai_ext__ = True
    batch_load_block.__bonsai_ext__ = True
    fsm.batch_store_block = batch_store_block
    fsm.batch_load_block = batch_load_block
    logger.info("bonsai_ext: SSD KV blocks get mtime refreshed on store/load (BONSAI_KV_FS_TOUCH=1)")
