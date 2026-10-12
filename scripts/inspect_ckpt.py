import json
from safetensors import safe_open

path = "checkpoints/needle3.safetensors"
with safe_open(path, framework="np") as h:
    meta = h.metadata() or {}
    for k in list(h.keys()):
        print(k, h.get_slice(k).get_shape(), h.get_slice(k).get_dtype())
print("run:", meta.get("run"))
print("step:", meta.get("step"), "format_version:", meta.get("format_version"))
