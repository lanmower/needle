import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from needle.model.finetune import BOS_ID, render_example
from needle.model.tokenizer import get_tokenizer

import torch_finetune as T
import eval_torch as E

rows = [json.loads(l) for l in open("data/pref_heldout.jsonl", encoding="utf-8")][:1]
params, cfg, flat, paths, model = T.setup("cuda", 0, 16, 32.0)
model.grad_ckpt = False
tokenizer = get_tokenizer(cfg.vocab_size)
E.freeze_kernels(model)
device = torch.device("cuda")

prompt, _ = render_example({**rows[0], "answers": []})
ids = torch.tensor([[BOS_ID] + tokenizer.encode(prompt)], device=device)
print(f"tokens {ids.shape[1]}  engram {model.num_engram}  lanes {model.n_lanes}")

with torch.no_grad():
    for _ in range(2):
        model(ids)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        model(ids)
    torch.cuda.synchronize()
    print(f"forward {(time.time() - start) / 3 * 1000:.1f} ms")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model(ids)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof2:
        model(ids)
    print(prof2.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
