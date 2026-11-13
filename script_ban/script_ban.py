"""vLLM v2 model-runner logits processor: ban whole writing systems per request.

Usage (server): --logits-processors script_ban:ScriptBanProcessor  (put this directory on PYTHONPATH)
Usage (request): "vllm_xargs": {"ban_scripts": "cyr,arab,thai,hangul,han_simp"}
Token classes come from tokclass.json (scripts/build_tokmask.py). Rows of requests without the arg are untouched.
The same apply() is used for the MTP draft-verification rows (Sampler.apply_sampling_params), so drafts are verified
against the masked distribution.
"""
import functools
import json
import os

import numpy as np

from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext, LogitsProcessor, LogitsProcRequestState)

TOKCLASS = os.environ.get("SCRIPT_BAN_TOKCLASS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokclass.json"))
MAX_PROFILES = 8   # distinct ban_scripts combinations alive at once (index 0 = no ban)
KEY = "ban_scripts"


@functools.lru_cache(maxsize=1)
def _known():
    return frozenset(json.load(open(TOKCLASS)))


def _parse(sp):
    v = (sp.extra_args or {}).get(KEY)
    if v is None or v == "":
        return ()
    names = v if isinstance(v, list) else str(v).split(",")
    return tuple(sorted({str(n).strip() for n in names if str(n).strip()}))


class ScriptBanProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states: LogitsProcRequestState):
        import torch
        self.torch = torch
        self.device = req_states.device
        self.vocab = req_states.vocab_size
        classes = json.load(open(TOKCLASS))
        self.cls_names = sorted(classes)
        self.cls_mask = {}
        for n in self.cls_names:
            m = torch.zeros(self.vocab, dtype=torch.bool, device=self.device)
            m[torch.tensor([i for i in classes[n] if i < self.vocab], dtype=torch.long, device=self.device)] = True
            self.cls_mask[n] = m
        self.table = torch.zeros(MAX_PROFILES, self.vocab, dtype=torch.bool, device=self.device)
        self.profiles = {(): 0}                       # key -> table row
        self.prof_np = np.zeros(req_states.max_num_reqs, dtype=np.int32)
        self.prof_gpu = torch.zeros(req_states.max_num_reqs, dtype=torch.int32, device=self.device)
        self.active = np.zeros(req_states.max_num_reqs, dtype=bool)
        self.dirty = False   # スロットの割り当てが変わったときだけ GPU へ転送する(毎ステップの同期コピーを避ける)

    @classmethod
    def validate_params(cls, sampling_params):
        key = _parse(sampling_params)
        if not key:
            return
        known = _known()
        bad = [k for k in key if k not in known]
        if bad:
            raise ValueError(f"{KEY}: unknown script class {bad}; known {sorted(known)}")

    def _profile(self, key):
        if key in self.profiles:
            return self.profiles[key]
        if len(self.profiles) >= MAX_PROFILES:
            raise ValueError(f"{KEY}: too many distinct combinations (max {MAX_PROFILES - 1})")
        row = len(self.profiles)
        m = self.table[row]
        m.zero_()
        for n in key:
            m |= self.cls_mask[n]
        self.profiles[key] = row
        return row

    def add_request(self, req_idx, sampling_params):
        key = _parse(sampling_params)
        self.prof_np[req_idx] = self._profile(key) if key else 0
        self.dirty = True
        self.active[req_idx] = bool(key)
        return bool(key)

    def apply_staged_writes(self):
        if self.dirty:
            self.prof_gpu.copy_(self.torch.from_numpy(self.prof_np), non_blocking=False)
            self.dirty = False

    def apply(self, logits, ctx: LogitsContext):
        if not self.active[ctx.idx_mapping_np].any():
            return logits
        prof = self.prof_gpu[ctx.expanded_idx_mapping.long()]          # [rows]
        logits.masked_fill_(self.table[prof.long()], float("-inf"))   # row 0 = all False
        return logits
