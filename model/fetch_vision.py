"""Extract the vision tower from prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (HTTP Range, no full download)
and save it under HF names (model.visual.*) as model-vision.safetensors in the given model dir.
Also merges the weight_map into model.safetensors.index.json.

usage: python fetch_vision.py <model_dir>
"""
import concurrent.futures as cf
import json
import os
import struct
import sys
import time
import urllib.request

import torch
from safetensors.torch import save_file

URL = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/resolve/main/model.safetensors"
OUT_DIR = sys.argv[1]


def rng(a, b):
    for i in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(req, timeout=300).read()
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError("range fetch failed")


n = struct.unpack("<Q", rng(0, 7))[0]
h = json.loads(rng(8, 7 + n))
base = 8 + n
keys = sorted(k for k in h if k.startswith("vision_tower."))
DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def get(k):
    a, b = h[k]["data_offsets"]
    raw = bytearray(rng(base + a, base + b - 1))
    t = torch.frombuffer(raw, dtype=DT[h[k]["dtype"]]).reshape(h[k]["shape"]).clone()
    return k, t


out = {}
with cf.ThreadPoolExecutor(3) as ex:
    for i, (k, t) in enumerate(ex.map(get, keys)):
        name = "model.visual." + k.removeprefix("vision_tower.")
        if name == "model.visual.patch_embed.proj.weight" and t.shape[-1] == 3:
            # MLX stores conv weights channels-last [1152,2,16,16,3]; HF expects [1152,3,2,16,16]
            t = t.permute(0, 4, 1, 2, 3).contiguous()
        out[name] = t.to(torch.bfloat16)
        if i % 100 == 0:
            print(i, "/", len(keys), flush=True)

save_file(out, os.path.join(OUT_DIR, "model-vision.safetensors"),
          metadata={"format": "pt", "source": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit vision_tower.*"})

idx_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
idx = json.load(open(idx_path))
for k in out:
    idx["weight_map"][k] = "model-vision.safetensors"
json.dump(idx, open(idx_path, "w"), indent=1)
print("wrote", len(out), "tensors; patch_embed", tuple(out["model.visual.patch_embed.proj.weight"].shape))
