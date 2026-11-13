"""Write the token-embedding table to a raw file on SSD (bf16, row-major [vocab, hidden]) for VLLM_SSD_EMBED_DIR.

usage: python export_embed.py <model_dir> <out_dir> [hot_ids.i32]

The GPU keeps only the rows listed in hot_ids.i32 (frequent tokens + all special tokens).
The bundled hot_ids.i32 = top 32,000 tokens of Japanese-heavy chat traffic + special tokens.
Rows not on the GPU are read from the file by the CPU, so the main model's output does not change.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR, OUT = sys.argv[1], sys.argv[2]
HOT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hot_ids.i32")
KEY = "model.language_model.embed_tokens.weight"
os.makedirs(OUT, exist_ok=True)

w = None
for shard in sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors"))):
    with safe_open(shard, "pt") as f:
        if KEY in f.keys():
            w = f.get_tensor(KEY)
            break
assert w is not None, f"{KEY} not found"
print("embed", tuple(w.shape), w.dtype)

w.contiguous().view(torch.int16).numpy().tofile(os.path.join(OUT, "embed_tokens.bf16"))
json.dump({"vocab": w.shape[0], "hidden": w.shape[1], "dtype": "bfloat16"},
          open(os.path.join(OUT, "embed_tokens.json"), "w"))
shutil.copy(HOT, os.path.join(OUT, "hot_ids.i32"))

m = np.memmap(os.path.join(OUT, "embed_tokens.bf16"), dtype=np.int16, mode="r", shape=tuple(w.shape))
for i in (0, 1234, w.shape[0] - 1):
    assert np.array_equal(m[i], w[i].view(torch.int16).numpy()), i
print("ok:", m.nbytes, "bytes,", np.fromfile(HOT, dtype=np.int32).size, "hot rows")
