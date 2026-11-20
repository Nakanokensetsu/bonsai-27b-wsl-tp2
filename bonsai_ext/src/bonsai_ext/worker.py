"""GPU worker additions (use with --worker-cls bonsai_ext.worker.BonsaiWorker).

- VRAM breakdown logger: once a minute, torch allocated / cached / non-torch VRAM per rank to
  $BONSAI_VRAM_LOG_DIR/vram_breakdown_rank<N>.log (default ~/bonsai-vllm, skipped if the dir is missing) (2026-10-19).
- VRAM_CAP_FRAC_R<rank>: after warmup and CUDA graph capture (so KV sizing is unaffected),
  cap this rank's PyTorch allocations to that fraction of the GPU (2026-10-22).
"""

import os
import threading
import time

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)
_LOG_DIR = os.path.expanduser(os.environ.get("BONSAI_VRAM_LOG_DIR", "~/bonsai-vllm"))
_LOG = os.path.join(_LOG_DIR, "vram_breakdown_rank{}.log")


class BonsaiWorker(Worker):
    def _start_vram_logger(self) -> None:
        if getattr(self, "_vrambd_started", False) or not os.path.isdir(_LOG_DIR):
            return
        self._vrambd_started = True
        path, dev = _LOG.format(self.rank), self.device

        def run():
            while True:
                try:
                    free, total = torch.cuda.mem_get_info(dev)
                    al = torch.cuda.memory_allocated(dev)
                    rs = torch.cuda.memory_reserved(dev)
                    mx = torch.cuda.max_memory_allocated(dev)
                    used = total - free
                    with open(path, "a") as f:
                        f.write(
                            f"{time.strftime('%F %T')} used={used / 2**20:.0f} "
                            f"alloc={al / 2**20:.0f} cache={(rs - al) / 2**20:.0f} "
                            f"nontorch={(used - rs) / 2**20:.0f} max_alloc={mx / 2**20:.0f} MiB\n"
                        )
                except Exception:
                    pass
                time.sleep(60)

        threading.Thread(target=run, daemon=True, name="vrambd").start()

    def load_model(self, *args, **kwargs):
        out = super().load_model(*args, **kwargs)
        if os.environ.get("BONSAI_DUMP_PARAMS") == "1":
            try:
                from collections import defaultdict

                agg = defaultdict(int)
                for root, mod in (("model", self.model_runner.model),
                                  ("drafter", getattr(getattr(self.model_runner, "speculator", None), "model", None))):
                    if mod is None:
                        continue
                    seen = set()
                    for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                        if t.device.type != "cuda" or t.data_ptr() in seen:
                            continue
                        seen.add(t.data_ptr())
                        key = root + "." + ".".join(p for p in n.split(".")[:3] if not p.isdigit())
                        agg[key] += t.numel() * t.element_size()
                with open(_LOG.format(self.rank).replace(".log", "_params.log"), "w") as f:
                    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
                        f.write(f"{v / 2**20:10.1f} MiB  {k}\n")
            except Exception as e:
                logger.warning("param dump failed: %s", e)
        return out

    def compile_or_warm_up_model(self):
        self._start_vram_logger()
        out = super().compile_or_warm_up_model()
        cap = os.environ.get(f"VRAM_CAP_FRAC_R{self.rank}")
        if cap:
            try:
                torch.cuda.set_per_process_memory_fraction(float(cap), self.device)
                with open(_LOG.format(self.rank), "a") as f:
                    f.write(f"{time.strftime('%F %T')} VRAMCAP set fraction={cap}\n")
            except Exception as e:
                logger.warning("VRAMCAP failed: %s", e)
        return out
