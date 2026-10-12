import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from needle.model.finetune import BOS_ID, render_example
from needle.model.tokenizer import get_tokenizer

import torch_finetune as T
import eval_torch as E


def timed(label, fn, n=5):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"  {label:<22} {(time.time() - start) / n * 1000:8.1f} ms", flush=True)


rows = [json.loads(l) for l in open("data/pref_heldout.jsonl", encoding="utf-8")][:1]
params, cfg, flat, paths, model = T.setup("cuda", 0, 16, 32.0)
model.grad_ckpt = False
tokenizer = get_tokenizer(cfg.vocab_size)
E.freeze_kernels(model)
device = torch.device("cuda")

prompt, _ = render_example({**rows[0], "answers": []})
ids = torch.tensor([[BOS_ID] + tokenizer.encode(prompt)], device=device)
t = ids.shape[1]
print(f"  prompt tokens          {t}", flush=True)
print(f"  num_engram             {model.num_engram}", flush=True)
print(f"  n_lanes                {model.n_lanes}", flush=True)

causal = model.causal(t)
timed("kernels()", lambda: model.kernels())
if model.num_engram:
    timed("engram_kv()", lambda: model.engram_kv(ids, causal))
timed("full forward", lambda: model(ids), n=3)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    steps = 8
    for i in range(steps):
        nxt = int(model(ids)[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    torch.cuda.synchronize()
    print(f"  decode step avg        {(time.time() - start) / steps * 1000:8.1f} ms",
          flush=True)
