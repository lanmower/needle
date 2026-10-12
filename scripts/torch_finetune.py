"""PyTorch port of needle's LoRA preference fine-tune, for the RTX 3060.

Only the forward/backward moves to torch: the tokenizer, prompt rendering,
sequence packing and LoRA target discovery all come from `needle.model.*`, so
token ids, masks, LoRA init and the shipped adapter format are identical to the
JAX reference loop in `needle/model/finetune.py::finetune_local`.

Modes:
  gate   compare torch logits against the JAX model on CPU (hard gate)
  bench  find the largest batch that fits on the GPU
  probe  cheap learnability probe (small subset, 1 epoch)
  train  the full run, writes the adapter
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from needle.model.architecture import (  # noqa: E402
    ENGRAM_CONV_TAPS, engram_geometry, ladder_layer_ranks,
)
from needle.model import quantize as Q  # noqa: E402
from needle.model.quantize import (  # noqa: E402
    CQ_GROUP_SIZE, WEIGHT_BITS, configure_deploy,
    _cq_codebook_np, _cq_hadamard_np,
)
from needle.model.checkpoints import write_adapter  # noqa: E402
from needle.model.finetune import (  # noqa: E402
    LORA_TARGETS, fit_max_len, init_lora, load_jsonl, lora_target_paths,
)
from needle.model.tokenizer import get_tokenizer  # noqa: E402
from needle.model.run import load_checkpoint  # noqa: E402

BASE = "checkpoints/needle3.safetensors"
DATA = "data/pref_train.jsonl"


# --------------------------------------------------------------------------
# quantization (CQ W4 STE + A8 activations), straight-through
# --------------------------------------------------------------------------

_CB = {}
_HH = {}


def _cb(device):
    if device not in _CB:
        _CB[device] = torch.from_numpy(np.ascontiguousarray(_cq_codebook_np(WEIGHT_BITS, CQ_GROUP_SIZE))).to(device)
    return _CB[device]


def _hh(device):
    if device not in _HH:
        _HH[device] = torch.from_numpy(np.ascontiguousarray(_cq_hadamard_np(CQ_GROUP_SIZE))).to(device)
    return _HH[device]


def _nearest(x, cb):
    flat = x.reshape(-1)
    pos = torch.searchsorted(cb, flat).clamp(1, cb.numel() - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = torch.where((flat - left).abs() <= (flat - right).abs(), pos - 1, pos)
    return cb[idx].reshape(x.shape)


def cq_quantize(w, bits=WEIGHT_BITS, group=CQ_GROUP_SIZE):
    cb, hh = _cb(w.device), _hh(w.device)
    d = w.shape[-1]
    pad = (-d) % group
    wp = F.pad(w, (0, pad)) if pad else w
    groups = wp.reshape(*wp.shape[:-1], -1, group).float()
    rot = groups @ hh
    norm = rot.pow(2).sum(-1, keepdim=True).sqrt()
    unit = rot / norm.clamp(min=1e-12)
    norm = norm.to(torch.float16).to(torch.float32)
    deq = (_nearest(unit, cb) * norm) @ hh
    deq = deq.reshape(wp.shape).to(w.dtype)
    return deq[..., :d] if pad else deq


def cq_ste(w, bits=WEIGHT_BITS, group=CQ_GROUP_SIZE):
    return w + (cq_quantize(w, bits, group) - w).detach()


def fake_quant(w, group_size, bits):
    qmax = 2 ** (bits - 1) - 1
    d = w.shape[-1]
    pad = (-d) % group_size
    wp = F.pad(w, (0, pad)) if pad else w
    g = wp.reshape(*wp.shape[:-1], -1, group_size).float()
    absmax = g.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax) * scale
    q = q.reshape(wp.shape).to(w.dtype)
    if pad:
        q = q[..., :d]
    return w + (q - w).detach()


QUANT_ON = True


def aq(x):
    """A8 activation fake-quant, per last-axis group (whole row here)."""
    if not QUANT_ON:
        return x
    return fake_quant(x, x.shape[-1], Q.ACT_BITS)


def quant_leaf(name, w):
    """Mirror quantize._map_quant_leaves for one flattened parameter leaf."""
    if not QUANT_ON:
        return w
    leaf = name.rsplit("/", 1)[-1]
    is_q = (leaf in ("kernel", "embedding") or leaf.startswith("mhc_phi")) and w.dim() >= 2
    if not is_q:
        return w
    second_last = leaf == "kernel" or leaf.startswith("mhc_phi")
    if second_last:
        return cq_ste(w.transpose(-1, -2)).transpose(-1, -2)
    return cq_ste(w)


# --------------------------------------------------------------------------
# model pieces
# --------------------------------------------------------------------------

def rms_unit(x, eps=1e-6):
    """architecture._rms_unit: computed in float32 and returned in float32."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def zc_norm(x, scale, eps=1e-6):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (1.0 + scale) * x / rms


def shift_right(x, offset):
    if offset == 0:
        return x
    zeros = torch.zeros((x.shape[0], offset, *x.shape[2:]), dtype=x.dtype, device=x.device)
    return torch.cat([zeros, x[:, :-offset]], dim=1)


def mask_diag(mask, offset):
    """mask: (1,1,T,T) -> (1,T): entry i is mask[0, i, i-offset], 0 for i < offset."""
    m = mask[:, 0]
    t = m.shape[-1]
    d = torch.diagonal(m, offset=-offset, dim1=-2, dim2=-1)
    if offset == 0:
        return d
    return F.pad(d, (offset, 0))


def rope_freqs(head_dim, seq_len, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, inv)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def sinkhorn(logits, iters=20):
    lk = logits
    for _ in range(iters):
        lk = lk - torch.logsumexp(lk, dim=-1, keepdim=True)
        lk = lk - torch.logsumexp(lk, dim=-2, keepdim=True)
    return torch.exp(lk)


def kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    z = torch.einsum("...ij,ik,jl->...kl", z, a, b)
    return z.reshape(*lead, a.shape[0] * b.shape[0])


def hada_perms(n, device):
    import numpy as _np
    return tuple(torch.from_numpy(_np.ascontiguousarray(
        _np.random.RandomState(s).permutation(n))).to(device) for s in (11, 13))


_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


_MASK32 = 0xFFFFFFFF


def engram_indices_np(tokens, orders, heads, slots, seed_heads=0):
    """Bit-exact port of architecture.engram_indices (uint32 arithmetic)."""
    u = np.asarray(tokens).astype(np.uint32)
    stride = seed_heads or heads
    b, t = u.shape
    out = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * stride + h + 1)) & _MASK32
            acc = np.full((b, t), seed, dtype=np.int64)
            for j in range(order):
                if j == 0:
                    sh = u
                else:
                    z = np.zeros((b, j), dtype=np.uint32)
                    sh = np.concatenate([z, u[:, :-j]], axis=1)
                acc = ((acc ^ sh.astype(np.int64)) * _ENGRAM_PRIME) & _MASK32
            acc = (acc ^ (acc >> 15)) & _MASK32
            out.append((acc % slots).astype(np.int32))
    return np.stack(out, axis=-1)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class NeedleTorch(nn.Module):
    def __init__(self, flat_params, cfg, device, rank=16, alpha=32.0, seed=0,
                 lora_paths=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.scale = alpha / rank
        d = cfg.d_model
        self.tdtype = torch.float64 if getattr(cfg, "dtype", "float32") == "float64" \
            else torch.float32

        # --- LoRA
        self.names = ["/".join(p) for p in (lora_paths or [])]
        self.base_k = {}
        import jax
        import jax.numpy as _jnp

        # --- fixed (non-LoRA) parameters, CQ-quantized once: cq() is a pure
        # function of the weights, so pre-quantizing them is bit-identical to
        # re-quantizing every step.
        self.fixed = {}
        for name, w in flat_params.items():
            if name in self.names:
                continue
            self.fixed[name] = quant_leaf(
                name, torch.as_tensor(np.asarray(w), dtype=self.tdtype, device=device))

        self.orders, self.heads, self.sub_dim = engram_geometry(cfg)
        self.num_engram = len(cfg.engram_layers)
        self.sites = len(cfg.engram_layers)

        a_list, b_list = [], []
        key = jax.random.PRNGKey(seed)
        for path in (lora_paths or []):
            name = "/".join(path)
            w = np.asarray(flat_params[name])
            in_dim, out_dim = w.shape[-2], w.shape[-1]
            lead = w.shape[:-2]
            key, sub = jax.random.split(key)
            a = np.asarray(jax.random.normal(sub, lead + (in_dim, rank), _jnp.float32)) / rank
            b = np.zeros(lead + (rank, out_dim), np.float32)
            self.base_k[name] = torch.as_tensor(
                w.astype(np.float64 if self.tdtype == torch.float64 else np.float32),
                device=device)
            a_list.append(nn.Parameter(torch.as_tensor(a, dtype=torch.float32, device=device)))
            b_list.append(nn.Parameter(torch.as_tensor(b, dtype=torch.float32, device=device)))
        self.A = nn.ParameterList(a_list)
        self.B = nn.ParameterList(b_list)

        # --- stack scaffolding
        self.n_lanes = cfg.mhc_lanes
        lanes = np.eye(cfg.mhc_lanes, dtype=np.float32)[np.arange(cfg.num_layers) % cfg.mhc_lanes]
        self.pre_off = torch.as_tensor(8.0 * lanes - 4.0, dtype=self.tdtype, device=device)
        self.post_off = torch.as_tensor(-4.0 * (1.0 - lanes), dtype=self.tdtype,
                                        device=device)
        flags = np.zeros((cfg.num_layers, self.sites), np.float32)
        for s, layer in enumerate(cfg.engram_layers):
            flags[layer, s] = 1.0
        self.site_flags = torch.as_tensor(flags, dtype=self.tdtype, device=device)
        self.global_flags = [1.0 if i in cfg.global_layers else 0.0
                             for i in range(cfg.num_layers)]
        self.layer_ranks = ladder_layer_ranks(cfg)
        self.embed_scale = math.sqrt(d)
        self.p1, self.p2 = hada_perms(1 << (d - 1).bit_length(), device)
        self._rope = {}
        self._band = {}
        self.grad_ckpt = True

    # -- helpers ---------------------------------------------------------
    def get_rope(self, t):
        if t not in self._rope:
            qk, _ = (self.cfg.qk_head_dim or self.cfg.d_model // self.cfg.num_heads, None)
            cos, sin = rope_freqs(qk, t, self.cfg.rope_theta, self.device)
            self._rope[t] = (cos, sin)
        return self._rope[t]

    def causal(self, t):
        key = ("c", t)
        if key not in self._band:
            self._band[key] = torch.tril(torch.ones(t, t, dtype=torch.bool,
                                                    device=self.device))[None, None]
        return self._band[key]

    def local(self, t, causal):
        key = ("l", t)
        if key not in self._band:
            pos = torch.arange(t, device=self.device)
            band = ((pos[:, None] - pos[None, :]) < self.cfg.sliding_window)[None, None]
            self._band[key] = causal & band
        return self._band[key]

    def kernels(self):
        out = {}
        for i, name in enumerate(self.names):
            w = self.base_k[name] + self.scale * (self.A[i] @ self.B[i])
            if not QUANT_ON:
                out[name] = w
                continue
            leaf = name.rsplit("/", 1)[-1]
            if leaf == "kernel" or leaf.startswith("mhc_phi"):
                w = cq_ste(w.transpose(-1, -2)).transpose(-1, -2)
            else:
                w = cq_ste(w)
            out[name] = w
        return out

    # -- engram ----------------------------------------------------------
    def engram_kv(self, ids, mask):
        cfg = self.cfg
        idx = engram_indices_np(ids.detach().cpu().numpy(), self.orders, self.heads,
                                cfg.engram_slots, getattr(cfg, "engram_seed_heads", 0))
        idx_t = torch.as_tensor(idx.astype(np.int64), device=self.device)
        ngram_ok = torch.stack([mask_diag(mask, o - 1)
                                for o in self.orders for _ in range(self.heads)], dim=-1)
        tap_ok = torch.stack([mask_diag(mask, j * max(self.orders))
                              for j in range(ENGRAM_CONV_TAPS)], dim=0)
        ar = torch.arange(len(self.orders) * self.heads, device=self.device)
        ks, vs = [], []
        for i in range(self.num_engram):
            tbl = self.fixed[f"engrams_{i}/embedding"]
            fetched = tbl[ar, idx_t]
            fetched = fetched * ngram_ok[..., None]
            e = fetched.reshape(*idx_t.shape[:2], -1)
            e = aq(e)
            k = e @ self.fixed[f"engrams_{i}/key_proj/kernel"]
            v = e @ self.fixed[f"engrams_{i}/value_proj/kernel"]
            taps = self.fixed[f"engrams_{i}/taps"]
            acc = None
            for j in range(ENGRAM_CONV_TAPS):
                term = taps[j] * shift_right(v, j * max(self.orders)) * tap_ok[j][..., None]
                acc = term if acc is None else acc + term
            ks.append(k)
            vs.append(acc)
        return torch.stack(ks), torch.stack(vs)

    # -- submodules ------------------------------------------------------
    def self_attn(self, l, x, mask, rope, K):
        cfg = self.cfg
        nh, nkv = cfg.num_heads, cfg.num_kv_heads
        qk_hd = cfg.qk_head_dim or cfg.d_model // nh
        v_hd = cfg.v_head_dim or cfg.d_model // nh
        out_dim = nh * v_hd
        b, t, _ = x.shape
        pre = f"stack/layers/block/self_attn"
        dbg = getattr(self, "debug", None)
        rec = (lambda k, v: dbg.setdefault(k, []).append(v.detach())) if dbg is not None \
            else (lambda k, v: None)
        xq = aq(x)
        rec("aq_x", xq)
        q = xq @ K[f"{pre}/q_proj/kernel"][l]
        k = xq @ K[f"{pre}/k_proj/kernel"][l]
        v = xq @ K[f"{pre}/v_proj/kernel"][l]
        rec("q_dense", q)
        rec("k_dense", k)
        rec("v_dense", v)
        ntaps = getattr(cfg, "qkv_conv_taps", 0)
        if ntaps:
            qt = self.fixed[f"{pre}/q_taps"][l]
            kt = self.fixed[f"{pre}/k_taps"][l]
            vt = self.fixed[f"{pre}/v_taps"][l]
            newq = newk = newv = None
            for j in range(ntaps):
                if j == 0:
                    sq, sk, sv = q, k, v
                else:
                    ok = mask_diag(mask, j)[..., None]
                    sq = shift_right(q, j) * ok
                    sk = shift_right(k, j) * ok
                    sv = shift_right(v, j) * ok
                newq = qt[j] * sq if newq is None else newq + qt[j] * sq
                newk = kt[j] * sk if newk is None else newk + kt[j] * sk
                newv = vt[j] * sv if newv is None else newv + vt[j] * sv
            q, k, v = newq, newk, newv
        rec("q_taps", q)
        rec("k_taps", k)
        rec("v_taps", v)

        q = q.reshape(b, t, nh, qk_hd).transpose(1, 2)
        k = k.reshape(b, t, nkv, qk_hd).transpose(1, 2)
        v = v.reshape(b, t, nkv, v_hd).transpose(1, 2)
        q = zc_norm(q, self.fixed[f"{pre}/q_norm/scale"][l])
        k = zc_norm(k, self.fixed[f"{pre}/k_norm/scale"][l])
        rec("q_norm", q)
        rec("k_norm", k)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rec("q_rope", q)
        rec("k_rope", k)
        if QUANT_ON and Q.ACT_BITS:
            q = fake_quant(q, q.shape[-1], 8)
        if QUANT_ON and Q.KV_BITS:
            k = fake_quant(k, k.shape[-1], 8)
            v = fake_quant(v, v.shape[-1], 8)
        rec("q_q", q)
        rec("k_q", k)
        rec("v_q", v)

        fused = max(qk_hd, v_hd)
        qf = q * math.sqrt(fused / qk_hd)
        pad_qk = fused - qk_hd
        if pad_qk:
            qf = F.pad(qf, (0, pad_qk))
            k = F.pad(k, (0, pad_qk))
        pad_v = fused - v_hd
        vf = F.pad(v, (0, pad_v)) if pad_v else v
        # GQA: jax reshapes query to (B,T,K,G,H), i.e. query head n uses kv head n//G
        rep = nh // nkv
        if rep > 1:
            k = torch.repeat_interleave(k, rep, dim=1)
            vf = torch.repeat_interleave(vf, rep, dim=1)
        scale = math.sqrt(fused)
        scores = (qf @ k.transpose(-1, -2)) / scale
        scores = scores.masked_fill(~mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out = (p @ vf)[..., :v_hd]
        rec("scores", scores)
        out = out.transpose(1, 2).reshape(b, t, out_dim)
        rec("attn_raw", out)
        gate = torch.sigmoid(xq @ K[f"{pre}/gate_proj/kernel"][l])
        rec("gate", gate)
        out = aq(out * gate)
        rec("gated", out)
        r = out @ K[f"{pre}/out_proj/kernel"][l]
        rec("attn_out", r)
        return r

    def hada(self, l, x):
        p = f"stack/layers/block/hadamard_mlp"
        f = self.fixed
        n = 1 << (self.cfg.d_model - 1).bit_length()
        cond = 1.0 + torch.softmax(x @ f[f"{p}/cond_v"][l], dim=-1) @ f[f"{p}/cond_u"][l]
        pad = n - x.shape[-1]
        z = F.pad(x, (0, pad)) if pad else x
        z = kron_apply(f[f"{p}/d1"][l] * z, f[f"{p}/w1a"][l], f[f"{p}/w1b"][l])[..., self.p1]
        z = kron_apply(F.silu(f[f"{p}/d2"][l] * cond * z + f[f"{p}/b2"][l]),
                       f[f"{p}/w2a"][l], f[f"{p}/w2b"][l])[..., self.p2]
        z = kron_apply(f[f"{p}/d3"][l] * z, f[f"{p}/w3a"][l], f[f"{p}/w3b"][l])
        return (f[f"{p}/d4"][l] * z)[..., :self.cfg.d_model]

    def block(self, l, x, mask, rope, K, ek, ev, sf):
        pre = "stack/layers/block"
        f = self.fixed
        dbg = getattr(self, "debug", None)
        if ek is not None:
            alpha = torch.sigmoid(
                torch.einsum("btd,sbtd->sbt", rms_unit(x), rms_unit(ek))
                / math.sqrt(self.cfg.d_model))
            # architecture.py runs this einsum in float32 and casts back
            x = x + torch.einsum("s,sbt,sbtd->btd", sf.float(), alpha,
                                 ev.float()).to(x.dtype)
        skip = x
        x = zc_norm(x, f[f"{pre}/ZCRMSNorm_0/scale"][l])
        if dbg is not None:
            dbg.setdefault("h1", []).append(x.detach())
        x = self.self_attn(l, x, mask, rope, K)
        if dbg is not None:
            dbg.setdefault("attn", []).append(x.detach())
        x = zc_norm(x, f[f"{pre}/post_attn_norm/scale"][l])
        if dbg is not None:
            dbg.setdefault("h2", []).append(x.detach())
        gate = torch.sigmoid(f[f"{pre}/attn_gate"][l])
        x = skip + gate * x
        skip = x
        x = zc_norm(x, f[f"{pre}/pre_hada_norm/scale"][l])
        if dbg is not None:
            dbg.setdefault("h3", []).append(x.detach())
        x = self.hada(l, x)
        if dbg is not None:
            dbg.setdefault("hada", []).append(x.detach())
        return skip + x

    def layer(self, l, stream, mask, rope, K, ek, ev, sf):
        b, t, n, c = stream.shape
        dbg = getattr(self, "debug", None)
        if dbg is not None:
            dbg.setdefault("stream", []).append(stream.detach())
        xf = stream.float()
        nx = aq(rms_unit(stream.reshape(b, t, n * c)))
        hc = f"stack/mhc"
        f = self.fixed
        # architecture.py casts the mHC projections to float32 before these matmuls
        hpre = torch.sigmoid(f[f"{hc}_a_pre"][l] * (nx @ f[f"{hc}_phi_pre"][l].float())
                             + f[f"{hc}_b_pre"][l] + self.pre_off[l])
        u = torch.einsum("btn,btnc->btc", hpre, xf.to(hpre.dtype))
        y = self.block(l, u, mask, rope, K, ek, ev, sf) - u
        if dbg is not None:
            dbg.setdefault("u", []).append(u.detach())
            dbg.setdefault("y", []).append(y.detach())
        hpost = 2.0 * torch.sigmoid(f[f"{hc}_a_post"][l] * (nx @ f[f"{hc}_phi_post"][l].float())
                                    + f[f"{hc}_b_post"][l] + self.post_off[l])
        res = nx @ f[f"{hc}_phi_res"][l].float()
        hres = sinkhorn(f[f"{hc}_a_res"][l] * res.reshape(b, t, n, n) + f[f"{hc}_b_res"][l])
        return (torch.einsum("btij,btjc->btic", hres, xf.to(hres.dtype))
                + hpost[..., None] * y.float()[:, :, None, :])

    def forward(self, ids):
        cfg = self.cfg
        b, t = ids.shape
        K = self.kernels()
        causal = self.causal(t)
        loc = self.local(t, causal) if cfg.sliding_window else causal
        cos, sin = self.get_rope(t)
        emb = self.fixed["embedding/embedding"]
        x_in = emb[ids] * self.embed_scale
        x = x_in
        if hasattr(self, "debug"):
            self.debug["emb_in"] = x_in.detach()
        ek, ev = (self.engram_kv(ids, causal) if self.num_engram else (None, None))
        if hasattr(self, "debug"):
            self.debug["ek"] = ek.detach()
            self.debug["ev"] = ev.detach()
        n = self.n_lanes
        x = x[:, :, None, :].expand(b, t, n, x.shape[-1])
        for l in range(cfg.num_layers):
            mask = causal if self.global_flags[l] else loc
            sf = self.site_flags[l]
            args = (l, x, mask, (cos, sin), K, ek, ev, sf)
            if self.grad_ckpt and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self.layer, *args, use_reentrant=False)
            else:
                x = self.layer(*args)
        x = x.mean(dim=2)
        x = zc_norm(x, self.fixed["stack/final_norm/scale"])
        if hasattr(self, "debug"):
            self.debug["final"] = x.detach()
            self.debug.setdefault("stream", []).append(x.detach())
        head = emb[:cfg.out_vocab] if cfg.out_vocab else emb
        return aq(x).float().to(head.dtype) @ head.t()


# --------------------------------------------------------------------------
# data / loss / schedule
# --------------------------------------------------------------------------

def load_data(path, tokenizer, max_len, limit=None):
    seqs, masks = load_jsonl(path, tokenizer, max_len)
    if limit:
        seqs, masks = seqs[:limit], masks[:limit]
    return (torch.as_tensor(np.ascontiguousarray(seqs)).long(),
            torch.as_tensor(np.ascontiguousarray(masks)).float())


def masked_ce(logits, ids, mask):
    logits, targets, m = logits[:, :-1], ids[:, 1:], mask[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                         targets.reshape(-1), reduction="none").reshape(m.shape)
    return (ce * m).sum() / torch.clamp(m.sum(), min=1.0)


def lr_at(step, total, warmup, peak):
    """optax.warmup_cosine_decay_schedule(count) with count starting at 0."""
    if step < warmup:
        return peak * step / warmup
    return 0.5 * peak * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# --------------------------------------------------------------------------
# setup shared by every mode
# --------------------------------------------------------------------------

def setup(device_str, seed=0, rank=16, alpha=32.0, dtype="float32"):
    import jax
    import jax.numpy as jnp
    params, cfg = load_checkpoint(BASE)
    cfg.dtype = dtype
    np_dtype = np.float64 if dtype == "float64" else np.float32
    params = jax.tree.map(lambda a: np.asarray(a).astype(np_dtype), params)
    configure_deploy(act_bits=getattr(cfg, "act_bits", 8),
                     kv_bits=getattr(cfg, "kv_bits", 8))
    flat = {}
    for name, v in sorted(_flatten(params).items()):
        flat[name] = np.asarray(v)
    paths = lora_target_paths(params)
    model = NeedleTorch(flat, cfg, torch.device(device_str), rank=rank, alpha=alpha,
                        seed=seed, lora_paths=paths)
    return params, cfg, flat, paths, model


def _flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, name))
        else:
            out[name] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "bench", "probe", "train"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=0, help="limit rows (0 = all)")
    ap.add_argument("--seq-len", type=int, default=0, help="truncate sequences")
    ap.add_argument("--out", default="checkpoints/pref_lora_gpu.safetensors")
    ap.add_argument("--log", default="data/ft_gpu.log")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--no-quant", action="store_true",
                    help="gate only: run both models with quant=False (removes the "
                         "A8/CQ step functions so fp32 noise is not amplified)")
    ap.add_argument("--fp64", action="store_true", help="gate only: compare in float64")
    ap.add_argument("--gate-rows", type=int, default=2)
    args = ap.parse_args()

    if args.no_quant:
        global QUANT_ON
        QUANT_ON = False
    if args.fp64:
        import jax
        jax.config.update("jax_enable_x64", True)
        import needle.model.architecture as _arch
        import jax.numpy as _jnp
        _arch.DTYPE_MAP["float64"] = _jnp.float64
    t0 = time.time()
    params, cfg, flat, paths, model = setup(args.device, args.seed,
                                            args.lora_rank, args.lora_alpha,
                                            dtype="float64" if args.fp64 else "float32")
    model.grad_ckpt = not args.no_ckpt
    tokenizer = get_tokenizer(cfg.vocab_size)
    max_len = fit_max_len(DATA, tokenizer, args.max_len)
    seqs, masks = load_data(DATA, tokenizer, max_len, args.rows or None)
    if args.seq_len:
        seqs, masks = seqs[:, :args.seq_len], masks[:, :args.seq_len]
    print(f"  {'setup':<9} {time.time()-t0:.1f}s  {len(paths)} lora groups  "
          f"seq_len {seqs.shape[1]}  rows {len(seqs)}", flush=True)
    for p in paths[:12]:
        print(f"  {'path':<9} {'/'.join(p)}  {np.asarray(flat['/'.join(p)]).shape}")
    seqs = seqs.to(args.device)
    masks = masks.to(args.device)

    if args.mode == "gate":
        return gate(params, cfg, flat, paths, model, seqs, masks, args)
    if args.mode == "bench":
        return bench(model, seqs, masks, args)
    return train(model, seqs, masks, args, paths)


# --------------------------------------------------------------------------
# gate: torch vs JAX logits
# --------------------------------------------------------------------------

def gate(params, cfg, flat, paths, model, seqs, masks, args):
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.finetune import merge_lora
    from needle.model.quantize import cq_ste_params

    quant = not args.no_quant
    rows = getattr(args, "gate_rows", 2) or 2
    width = min(args.seq_len or 64, seqs.shape[1])
    ids = seqs[:rows, :width].cpu().numpy().astype(np.int32)
    dev = torch.device("cpu")
    cpu = NeedleTorch(flat, cfg, dev, rank=model.rank,
                      alpha=model.rank * model.scale, seed=0, lora_paths=paths)
    # non-trivial B so the merge path is exercised too (off in fp64: the float32
    # A@B product is itself a source of fp noise, and merge is a plain matmul
    # that the smooth gate already covers)
    if not args.fp64:
        bkey = jax.random.PRNGKey(1234)
        for i in range(len(cpu.B)):
            bkey, sub = jax.random.split(bkey)
            with torch.no_grad():
                cpu.B[i].copy_(torch.as_tensor(
                    np.asarray(jax.random.normal(sub, tuple(cpu.B[i].shape),
                                                 jnp.float32)) * 0.01))
    lora = {}
    for i, p in enumerate(paths):
        lora[p] = {"A": jnp.asarray(cpu.A[i].detach().numpy()),
                   "B": jnp.asarray(cpu.B[i].detach().numpy())}
    scale = model.scale

    jax_model = SimpleAttentionNetwork(cfg)
    merged = merge_lora(params, lora, scale)
    if quant:
        merged = cq_ste_params(merged, WEIGHT_BITS)
    ref = np.asarray(jax_model.apply({"params": merged}, jnp.asarray(ids), quant=quant))

    with torch.no_grad():
        ids_t = torch.as_tensor(ids.astype(np.int64))
        got = cpu.forward(ids_t).numpy()

    # ---- self-calibration: how much does *JAX* disagree with itself when the
    # input is perturbed by one fp32 ULP worth of noise?  The quantized net is
    # a step function (A8/CQ round-to-nearest), so this is the noise floor any
    # correct implementation must sit at.
    rng = np.random.default_rng(0)

    def jitter(mult):
        out = {}
        for k, v in merged.items():
            out[k] = v
        def f(w):
            a = np.asarray(w)
            if not np.issubdtype(a.dtype, np.floating):
                return jnp.asarray(a)
            return jnp.asarray((a * (1.0 + mult * rng.standard_normal(a.shape))
                                ).astype(a.dtype))
        return jax.tree.map(f, merged)

    refs = [np.asarray(jax_model.apply({"params": jitter(6e-8 * (2 ** k))},
                                       jnp.asarray(ids), quant=quant))
            for k in range(3)]
    ref2 = refs[0]

    def stats(a, b, tag):
        a64 = np.asarray(a, np.float64)
        b64 = np.asarray(b, np.float64)
        d = np.abs(a64 - b64)
        num = float(np.dot(a64.reshape(-1), b64.reshape(-1)))
        den = float(np.linalg.norm(a64.reshape(-1))) * float(
            np.linalg.norm(b64.reshape(-1))) + 1e-300
        c = num / den
        print(f"  {tag:<13} max {d.max():.3e}  mean {d.mean():.3e}  "
              f"rms {np.sqrt((d ** 2).mean()):.3e}  cosine {c:.12f}")
        return d.max()

    def ce_of(logits, m, tgt):
        z = np.asarray(logits, np.float64)
        mx = z.max(axis=-1, keepdims=True)
        lse = mx[:, :, 0] + np.log(np.exp(z - mx).sum(axis=-1))
        ce = z - lse[:, :, None]
        b = np.arange(z.shape[0])[:, None]
        t = np.arange(z.shape[1] - 1)[None, :]
        picked = ce[b, t, np.asarray(tgt, np.int64)[:, 1:]]
        m = np.asarray(m, np.float64)
        return float(-(picked * m[:, 1:]).sum() / max(m[:, 1:].sum(), 1.0))

    m_np = masks[:rows, :width].cpu().numpy()
    print(f"  {'logits':<13} shape {ref.shape}  jax mean {ref.mean():.5f}  "
          f"torch mean {got.mean():.5f}  max|logit| {np.abs(ref).max():.1f}")
    dx = stats(ref, got, "torch vs jax")
    df = stats(ref, ref2, "jax vs jax*")
    print(f"  {'mask':<13} sum {m_np.sum():.0f} / {m_np.size}  rows {rows} width {width}")
    l_ref = ce_of(ref, m_np, ids)
    l_t = ce_of(got, m_np, ids)
    l_ref2 = ce_of(ref2, m_np, ids)
    l_refs = [ce_of(r, m_np, ids) for r in refs]
    l_tm = float(masked_ce(torch.as_tensor(got), ids_t,
                           torch.as_tensor(m_np, device=dev)))
    print(f"  {'loss':<13} jax {l_ref:.6f}  torch {l_t:.6f}  jax* {l_ref2:.6f}  "
          f"(torch fn {l_tm:.6f})")

    dl, dlf = abs(l_t - l_ref), abs(l_ref2 - l_ref)
    spread = max(abs(x - l_ref) for x in l_refs)
    print(f"  {'loss delta':<13} torch {dl:.2e}  jax self-noise "
          f"{' '.join(f'{abs(x-l_ref):.2e}' for x in l_refs)}  worst {spread:.2e}")
    ok = dx <= 3.0 * df and dl <= max(3.0 * spread, 1e-3)
    print(f"  {'gate':<13} {'PASS' if ok else 'FAIL'}  (torch error {dx/df:.2f}x the "
          f"jax-vs-jax max|diff| floor, loss {dl/max(spread,1e-12):.2f}x the loss floor)")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# train / probe / bench
# --------------------------------------------------------------------------

def bench(model, seqs, masks, args):
    for batch in (16, 12, 8, 4, 2):
        if batch > len(seqs):
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            opt = torch.optim.AdamW(list(model.A) + list(model.B), lr=args.lr)
            for rep in range(2):
                ids = seqs[:batch]
                m = masks[:batch]
                opt.zero_grad(set_to_none=True)
                loss = masked_ce(model(ids), ids, m)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.A) + list(model.B), 1.0)
                opt.step()
                if rep == 1:
                    torch.cuda.synchronize()
                    dt = time.time()
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True)
                        loss = masked_ce(model(ids), ids, m)
                        loss.backward()
                        opt.step()
                    torch.cuda.synchronize()
                    dt = (time.time() - dt) / 2
            print(f"  batch {batch:>3}  {dt:.3f} s/step  peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB  loss {float(loss):.4f}")
            del opt
        except RuntimeError as exc:
            print(f"  batch {batch:>3}  OOM: {str(exc).splitlines()[0]}")
            torch.cuda.empty_cache()
    return 0


def train(model, seqs, masks, args, paths):
    log = open(args.log, "w")

    def emit(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    n = len(seqs)
    batch = min(args.batch_size, n)
    steps_per_epoch = -(-n // batch)
    total = args.epochs * steps_per_epoch
    warmup = min(max(1, total // 20), total - 1)
    import optax
    wd = 1e-4
    try:
        import inspect
        wd = inspect.signature(optax.adamw).parameters["weight_decay"].default
    except Exception:
        pass
    opt = torch.optim.AdamW(list(model.A) + list(model.B), lr=0.0,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=float(wd))
    emit(f"  {'device':<9} {args.device}  {torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu'}")
    emit(f"  {'data':<9} {n} rows  seq_len {seqs.shape[1]}  batch {batch}  "
         f"{total} steps  warmup {warmup}  wd {wd}")
    emit(f"  {'numerics':<9} CQ W{WEIGHT_BITS} STE + A{Q.ACT_BITS}  grad-ckpt {model.grad_ckpt}")

    rng = np.random.default_rng(args.seed)
    every = max(1, total // 50)
    step = 0
    t_start = time.time()
    first_loss = None
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        idx_t = torch.as_tensor(order, device=seqs.device)
        last = 0.0
        for start in range(0, n, batch):
            sel = idx_t[start:start + batch]
            ids, m = seqs[sel], masks[sel]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, args.lr)
            opt.zero_grad(set_to_none=True)
            loss = masked_ce(model(ids), ids, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.A) + list(model.B), 1.0)
            opt.step()
            last = float(loss.detach())
            if first_loss is None:
                first_loss = last
            step += 1
            if step % every == 0 or step == 1:
                el = time.time() - t_start
                emit(f"  {'step':<9} {step}/{total}  loss {last:.4f}  "
                     f"lr {opt.param_groups[0]['lr']:.2e}  {el/max(1,step-1 if step>1 else 1):.2f}s/step")
        emit(f"  {'epoch':<9} {epoch+1}/{args.epochs}  loss {last:.4f}  "
             f"elapsed {time.time()-t_start:.0f}s")
    emit(f"  {'first':<9} {first_loss:.4f} -> {last:.4f}   total {time.time()-t_start:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_adapter(args.out, {
        "lora": {name: {"A": np.asarray(model.A[i].detach().cpu().numpy()),
                        "B": np.asarray(model.B[i].detach().cpu().numpy())}
                 for i, name in enumerate(model.names)},
        "scale": float(model.scale),
        "base": BASE,
        "rank": model.rank,
        "seed": args.seed,
    })
    emit(f"  {'adapter':<9} {args.out}  {os.path.getsize(args.out)/1e6:.2f} MB  "
         f"{len(model.names)} groups")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
