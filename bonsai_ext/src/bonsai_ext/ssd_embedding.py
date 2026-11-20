# SPDX-License-Identifier: Apache-2.0
"""Token embedding kept in a file on SSD, with only frequent tokens on the GPU.

The input embedding is a pure row lookup, so most of its rows never need GPU
memory. The full table lives in a raw bf16 file read through the page cache
(np.memmap); a GPU table holds the rows of frequent tokens (`hot_ids.i32`).

- Eager calls (the target model's inputs_embeds, computed outside CUDA graphs)
  are exact: rows missing from the GPU table are read from the file.
- Calls inside CUDA graph capture or torch.compile (the MTP drafter) use the GPU
  table only; a missing row becomes zeros. Drafts are verified by the target,
  so this can only lower the draft acceptance rate, never change the output.

Files (written by ssd_embed/export_embed.py in the release repo):
  embed_tokens.bf16  [vocab, hidden] bf16, row major
  embed_tokens.json  {"vocab", "hidden", "dtype"}
  hot_ids.i32        int32 token ids kept on the GPU
"""

import json
import os

import numpy as np
import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs


def ssd_embed_dir() -> str | None:
    return os.environ.get("VLLM_SSD_EMBED_DIR") or None


class SSDEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, directory: str):
        super().__init__()
        meta = json.load(open(os.path.join(directory, "embed_tokens.json")))
        assert (meta["vocab"], meta["hidden"]) == (num_embeddings, embedding_dim), (
            meta,
            num_embeddings,
            embedding_dim,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.table = np.memmap(
            os.path.join(directory, "embed_tokens.bf16"),
            dtype=np.int16,
            mode="r",
            shape=(num_embeddings, embedding_dim),
        )
        hot = np.fromfile(os.path.join(directory, "hot_ids.i32"), dtype=np.int32)
        self.hot_ids = torch.from_numpy(hot.astype(np.int64))
        self.slot_cpu = np.full(num_embeddings, -1, dtype=np.int32)
        self.slot_cpu[hot] = np.arange(len(hot), dtype=np.int32)
        # Named `weight` so the checkpoint's embed_tokens.weight lands here; the
        # loader keeps only the frequent rows.
        self.weight = nn.Parameter(
            torch.empty(len(hot), embedding_dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        # token id -> row of the GPU table, -1 if the token is not on the GPU
        # (from_numpy ignores the default device; follow the weight's)
        self.register_buffer(
            "slot",
            torch.from_numpy(self.slot_cpu.copy()).to(self.weight.device),
            persistent=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self._load_hot_rows})

    def _load_hot_rows(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert tuple(loaded.shape) == (self.num_embeddings, self.embedding_dim)
        param.data.copy_(loaded[self.hot_ids.to(loaded.device)].to(param.dtype))

    def _gpu_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        s = self.slot[input_ids]
        rows = self.weight[s.clamp_min(0)]
        return torch.where((s >= 0).unsqueeze(-1), rows, torch.zeros_like(rows))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self._gpu_rows(input_ids)
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return out
        ids = input_ids.cpu().numpy()
        miss = np.nonzero(self.slot_cpu[ids] < 0)[0]
        if len(miss):
            rows = torch.from_numpy(np.ascontiguousarray(self.table[ids[miss]]))
            out[torch.from_numpy(miss).to(out.device)] = rows.view(torch.bfloat16).to(
                out.device
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, on_gpu={self.weight.shape[0]}"
        )
