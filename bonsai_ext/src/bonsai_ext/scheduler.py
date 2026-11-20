"""Scheduler additions for vLLM 0.34 (use with --scheduler-cls bonsai_ext.scheduler.BonsaiScheduler).

Re-implements the 0.32 scheduler patch through overrides instead of edits:
  VLLM_STEP_PREFILL_CAP=N  while someone decodes, at most N prefill tokens per step, handed out
                           shortest-remaining-first (Sarathi-Serve style). 0 = off.
  VLLM_PREFILL_EVERY=N     with the cap on, only every N-th step carries prefill.
  VLLM_SOLO_LPT=N          a lone prefill (nothing else running or waiting) is also chunked to N
                           tokens/step, so a short request arriving mid-way waits <1 step. 0 = off.
  VLLM_PRIO_PREEMPT=1      policy=priority: when slots are full (or KV usage >= VLLM_PRIO_KV_THRESH)
                           and a higher-priority request waits, preempt the lowest-priority runner
                           (one per step), and do not let lower-priority KV-holding requests take
                           the freed slot first.
Per-request caps are applied where upstream itself clips a chunk: before Mamba block alignment when
the model needs it (so chunks still end on aligned positions), otherwise in _reserve_prefill_lookahead.
A cap of 0 makes upstream skip the request for this step (running: next request; waiting: requeue).
"""

import os
import time

from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import PauseState

_SOLO_LPT = int(os.environ.get("VLLM_SOLO_LPT", "0"))
_STEP_PREFILL_CAP = int(os.environ.get("VLLM_STEP_PREFILL_CAP", "0"))
_PREFILL_EVERY = int(os.environ.get("VLLM_PREFILL_EVERY", "0"))
_PRIO_PREEMPT = int(os.environ.get("VLLM_PRIO_PREEMPT", "0"))
_PRIO_KV_THRESH = float(os.environ.get("VLLM_PRIO_KV_THRESH", "0"))
_PRIO_KV_GAP = float(os.environ.get("VLLM_PRIO_KV_GAP", "1.0"))


class BonsaiScheduler(AsyncScheduler):
    _step_prefill_alloc: dict[str, int] | None = None
    _solo_prefill: bool = False
    _prio_kv_last: float = 0.0
    _prefill_step_ctr: int = -1

    # ---- per-step planning -------------------------------------------------
    def _plan_step_prefill(self) -> None:
        self._step_prefill_alloc = None
        num_running, num_waiting = self.get_request_counts()
        self._solo_prefill = _SOLO_LPT > 0 and num_running + num_waiting == 1
        if _STEP_PREFILL_CAP <= 0:
            return
        if not any(r.num_computed_tokens >= r.num_prompt_tokens for r in self.running):
            return  # nobody decoding: nothing to protect
        cands = [
            (r.num_prompt_tokens - r.num_computed_tokens, r.request_id)
            for r in self.running
            if r.num_computed_tokens < r.num_prompt_tokens
        ]
        for q in (self.kv_holding_waiting, self.waiting):
            for i, r in enumerate(q):
                if i >= 32:
                    break
                cands.append((max(r.num_prompt_tokens - r.num_computed_tokens, 1), r.request_id))
        if not cands:
            return
        if _PREFILL_EVERY > 1:
            self._prefill_step_ctr += 1
            if self._prefill_step_ctr % _PREFILL_EVERY:
                self._step_prefill_alloc = {}
                return
        left = _STEP_PREFILL_CAP
        alloc: dict[str, int] = {}
        for rem, rid in sorted(cands):
            alloc[rid] = min(rem, left)
            left -= alloc[rid]
        self._step_prefill_alloc = alloc

    def _prefill_cap(self, request, num_computed_tokens: int) -> int | None:
        """Max prefill tokens for this request in this step, None = no limit."""
        if num_computed_tokens >= request.num_prompt_tokens:
            return None
        cap = None
        if self._step_prefill_alloc is not None:
            cap = self._step_prefill_alloc.get(request.request_id, 0)
        if self._solo_prefill:
            cap = _SOLO_LPT if cap is None else min(cap, _SOLO_LPT)
        return cap

    # ---- where upstream clips a chunk ---------------------------------------
    def _mamba_block_aligned_split(
        self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0
    ):
        cap = self._prefill_cap(
            request, request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        )
        if cap is not None and cap > 0:
            num_new_tokens = min(num_new_tokens, cap)
        # cap == 0 is applied in _reserve_prefill_lookahead: a 0 here makes the waiting loop stop.
        return super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        num_new_tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        cap = self._prefill_cap(request, num_computed_tokens)
        if cap is None:
            return num_new_tokens
        if cap <= 0:
            return 0
        if self.need_mamba_block_aligned_split:
            return num_new_tokens  # already capped before alignment
        return min(num_new_tokens, cap)

    # ---- priority preemption ------------------------------------------------
    def _prio_active(self) -> bool:
        return bool(_PRIO_PREEMPT) and self.policy == SchedulingPolicy.PRIORITY and (
            self._pause_state == PauseState.UNPAUSED
        )

    def _prio_preempt_for_waiting(self, ts: float) -> None:
        if not self.running or not (self.waiting or self.kv_holding_waiting):
            return
        slots_full = len(self.running) + self.num_waiting_for_streaming_input >= self.max_num_active_reqs
        if not slots_full:
            if _PRIO_KV_THRESH <= 0 or self.kv_cache_manager.usage < _PRIO_KV_THRESH:
                return
            now = time.monotonic()
            if now - self._prio_kv_last < _PRIO_KV_GAP:
                return
            self._prio_kv_last = now
        head = min(
            (q.peek_request() for q in (self.kv_holding_waiting, self.waiting) if q),
            key=lambda r: (r.priority, r.arrival_time),
        )
        victim = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        if victim.priority <= head.priority:
            return
        self.running.remove(victim)
        self._preempt_request(victim, ts, drop_stale_output=self.requires_kv_delivery)

    def schedule(self, throttle_prefills: bool = False):
        self._plan_step_prefill()
        hidden = None
        if self._prio_active():
            self._prio_preempt_for_waiting(time.monotonic())
            # A higher-priority new request goes before KV-holding (preempted lower-priority)
            # ones; otherwise the freed slot is retaken and the preemption repeats.
            kvh = self.kv_holding_waiting
            if kvh and self.waiting and self.waiting.peek_request().priority < kvh.peek_request().priority:
                hidden = kvh
                self.kv_holding_waiting = create_request_queue(self.policy)
        try:
            return super().schedule(throttle_prefills)
        finally:
            if hidden is not None:
                added = self.kv_holding_waiting
                self.kv_holding_waiting = hidden
                for r in list(added):
                    hidden.add_request(r)
