"""Measure the network's own sensitivity: does fp32-level noise at the input
already produce O(1) logit differences by the output?  Runs JAX twice."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import jax
import jax.numpy as jnp
from needle.model.run import load_checkpoint
from needle.model.architecture import SimpleAttentionNetwork
from needle.model.tokenizer import get_tokenizer
from needle.model.finetune import load_jsonl
from needle.model.quantize import cq_ste_params, WEIGHT_BITS, configure_deploy
Q = os.environ.get("NOQ") != "1"

import torch_finetune as TF

configure_deploy(act_bits=8, kv_bits=8)
params, cfg = load_checkpoint(TF.BASE)
cfg.dtype = "float32"
params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
merged = cq_ste_params(params, WEIGHT_BITS) if Q else params

tok = get_tokenizer(cfg.vocab_size)
seqs, masks = load_jsonl(TF.DATA, tok, 64)
ids = jnp.asarray(seqs[:2].astype(np.int32))
m = SimpleAttentionNetwork(cfg)
base = np.asarray(m.apply({"params": merged}, ids, quant=Q))

for eps in ((1e-9,1e-7,1e-5) if Q else (1e-9,1e-7,1e-5)):
    rng = np.random.default_rng(0)
    emb = np.asarray(merged["embedding"]["embedding"])
    emb2 = emb + eps * rng.standard_normal(emb.shape).astype(np.float32)
    p2 = dict(merged)
    p2["embedding"] = {"embedding": jnp.asarray(emb2)}
    out = np.asarray(m.apply({"params": p2}, ids, quant=Q))
    d = np.abs(out - base)
    cos = float((out.reshape(-1) @ base.reshape(-1)) /
                (np.linalg.norm(out.reshape(-1)) * np.linalg.norm(base.reshape(-1))))
    print(f"eps {eps:.0e} (|emb| {np.abs(emb).max():.1f})  max|diff| {d.max():.3e}"
          f"  mean|diff| {d.mean():.3e}  cosine {cos:.10f}")
