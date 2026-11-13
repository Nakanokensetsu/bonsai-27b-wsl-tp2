"""Replace the 15 mtp.* tensors in the model's 3rd shard with another MTP head (same names/shapes/dtypes).

usage: python swap_mtp.py <model_dir> <mtp_head.safetensors>
The original shard is kept as <shard>.orig.
"""
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR, HEAD = sys.argv[1], sys.argv[2]
SHARD = os.path.join(MODEL_DIR, "model-00003-of-00003.safetensors")

new = {}
with safe_open(HEAD, "pt") as f:
    for k in f.keys():
        new[k] = f.get_tensor(k)

tensors, n = {}, 0
with safe_open(SHARD, "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        t = f.get_tensor(k)
        if k in new:
            assert new[k].shape == t.shape and new[k].dtype == t.dtype, (k, new[k].shape, t.shape)
            c = torch.nn.functional.cosine_similarity(new[k].float().flatten(), t.float().flatten(), 0).item()
            print(f"{k:55s} cos {c:.5f}")
            t = new[k].contiguous()
            n += 1
        tensors[k] = t
assert n == len(new) == 15, (n, len(new))

if not os.path.exists(SHARD + ".orig"):
    os.rename(SHARD, SHARD + ".orig")
save_file(tensors, SHARD, metadata=meta)
print("swapped", n, "tensors into", SHARD)
