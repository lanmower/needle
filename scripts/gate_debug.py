"""Bisect the torch port against the JAX reference (debug only)."""
import os, sys, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import jax
import jax.numpy as jnp
FP64 = os.environ.get("FP64") == "1"
NOQ = os.environ.get("NOQ") == "1"
QFLAG = not NOQ
if FP64:
    jax.config.update("jax_enable_x64", True)
    import needle.model.architecture as _arch
    _arch.DTYPE_MAP["float64"] = jnp.float64
from needle.model.run import load_checkpoint
from needle.model.finetune import lora_target_paths, init_lora, merge_lora, load_jsonl
from needle.model.tokenizer import get_tokenizer
from needle.model.architecture import SimpleAttentionNetwork
from needle.model.quantize import cq_ste_params, WEIGHT_BITS, configure_deploy

import torch_finetune as TF

torch.set_grad_enabled(False)

params, cfg = load_checkpoint(TF.BASE)
cfg.dtype = "float64" if FP64 else "float32"
params = jax.tree.map(lambda a: np.asarray(a).astype(
    np.float64 if FP64 else np.float32), params)
TF.QUANT_ON = QFLAG
configure_deploy(act_bits=8, kv_bits=8)

flat = TF._flatten(params)
paths = lora_target_paths(params)
scale = 32.0 / 16
lora = init_lora(params, paths, 16, jax.random.PRNGKey(0))
bkey = jax.random.PRNGKey(1234)
for p in paths:
    bkey, sub = jax.random.split(bkey)
    lora[p]["B"] = jax.random.normal(sub, lora[p]["B"].shape, jnp.float32) * 0.01

merged = merge_lora(params, lora, scale)
if QFLAG:
    merged = cq_ste_params(merged, WEIGHT_BITS)
merged_flat = TF._flatten(merged)

# ---- 1. do the quantized params agree?
model = TF.NeedleTorch(flat, cfg, torch.device("cpu"), rank=16, alpha=32.0, seed=0,
                       lora_paths=paths)
for i, n in enumerate(model.names):
    with torch.no_grad():
        model.B[i].copy_(torch.as_tensor(np.asarray(lora[paths[i]]["B"])))
        da = float(np.abs(np.asarray(lora[paths[i]]["A"]) - model.A[i].numpy()).max())
    print(f"  A diff {da:.3e}  {n}")
K = model.kernels()
worst = []
for n in model.names:
    a = np.asarray(merged_flat[n])
    b = K[n].numpy()
    worst.append((float(np.abs(a - b).max()), n, float(np.abs(a).max())))
for n, v in model.fixed.items():
    a = np.asarray(merged_flat[n])
    b = v.numpy()
    worst.append((float(np.abs(a - b).max()), n, float(np.abs(a).max())))
worst.sort(reverse=True)
print("== quantized params: worst 6 ==")
for d, n, m in worst[:6]:
    print(f"  {d:>10.3e}  {n:<45} max|w| {m:.3f}")

print("== quantized params: mismatch counts ==")
for n in model.names:
    a = np.asarray(merged_flat[n]).reshape(-1)
    b = K[n].numpy().reshape(-1)
    bad = int((np.abs(a - b) > 1e-5).sum())
    print(f"  {n:<45} {bad}/{a.size} mismatch  max {np.abs(a-b).max():.3e}")

# ---- 2. forward intermediates
tok = get_tokenizer(cfg.vocab_size)
seqs, masks = load_jsonl(TF.DATA, tok, 64)
ids = seqs[:2].astype(np.int64)
ids_j = jnp.asarray(seqs[:2].astype(np.int32))

jax_model = SimpleAttentionNetwork(cfg)
out, state = jax_model.apply({"params": merged}, ids_j, quant=QFLAG,
                             capture_intermediates=True, mutable=["intermediates"])


def walk(tree, prefix=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from walk(v, prefix + (str(k),))
    else:
        yield prefix, tree


JI = {}
for p, v in walk(state.get("intermediates", {})):
    try:
        JI["/".join(p)] = np.asarray(v)
    except Exception:
        pass
print("== jax intermediates ==")
for k, v in JI.items():
    print("  ", k, v.shape, v.dtype)

model.debug = {}
logits = model.forward(torch.as_tensor(ids))
print("torch logits", logits.shape)

# embedding / engram / final-norm comparison
emb_j = JI["embedding/__call__"][0]
# embedding/__call__ is Embed output; SimpleAttentionNetwork scales by sqrt(d)
emb_t = model.debug.get("emb_in")
if emb_t is not None:
    d0 = np.abs(emb_j - emb_t.numpy() / math.sqrt(cfg.d_model)).max()
    print(f"embedding: {d0:.3e}  max|emb| {np.abs(emb_j).max():.3f}")
for i in range(5):
    kv_j = JI[f"engrams_{i}/__call__"][0]
    k_t = model.debug["ek"][i].numpy()
    v_t = model.debug["ev"][i].numpy()
    dk = np.abs(kv_j[0] - k_t).max()
    dv = np.abs(kv_j[1] - v_t).max()
    print(f"engram {i}: k {dk:.3e} ({dk/max(1e-9,np.abs(kv_j[0]).max()):.2e})"
          f"  v {dv:.3e} ({dv/max(1e-9,np.abs(kv_j[1]).max()):.2e})")
fn_j = JI["stack/final_norm/__call__"][0]
fn_t = model.debug["final"].numpy()
print("final_norm:", np.abs(fn_j - fn_t).max(), " rel-to-max",
      np.abs(fn_j - fn_t).max() / max(1e-6, np.abs(fn_j).max()),
      " rms", np.sqrt(((fn_j - fn_t) ** 2).mean()), " |x|rms", np.sqrt((fn_j ** 2).mean()))

# ------------------------------------------------------------------
# per-layer bisect: replicate Stack's mHC loop in JAX, using needle's Block
# ------------------------------------------------------------------
from needle.model.architecture import (Block, MultiHeadAttention as MHA,
                                       HadamardMLP as HADA, ZCRMSNorm as ZCN,
                                       _rms_unit, _sinkhorn, _aq, make_causal_mask,
                                       ladder_layer_ranks)

_D = jnp.float64 if FP64 else jnp.float32
L = cfg.num_layers
n_lane = cfg.mhc_lanes
T = ids_j.shape[1]
blk = Block(cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers, _D,
            cfg.flash, qk_head_dim=cfg.qk_head_dim, v_head_dim=cfg.v_head_dim,
            qkv_conv_taps=cfg.qkv_conv_taps)
mha = MHA(cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers, _D,
          cfg.flash, qk_head_dim=cfg.qk_head_dim, v_head_dim=cfg.v_head_dim,
          qkv_conv_taps=cfg.qkv_conv_taps)
hada = HADA(cfg.d_model, _D)
blkp = jax.tree.map(lambda a: a, merged["stack"]["layers"]["block"])
cos, sin = TF.rope_freqs(cfg.qk_head_dim, T, cfg.rope_theta, torch.device("cpu"))
rope_j = (jnp.asarray(cos.numpy()), jnp.asarray(sin.numpy()))
mask_j = make_causal_mask(T)
ek_j = jnp.asarray(np.asarray(model.debug["ek"]))
ev_j = jnp.asarray(np.asarray(model.debug["ev"]))
flags = np.zeros((L, 5), np.float32)
for s, layer in enumerate(cfg.engram_layers):
    flags[layer, s] = 1.0
lane = np.eye(n_lane, dtype=np.float32)[np.arange(L) % n_lane]
pre_off = 8 * lane - 4
post_off = -4 * (1 - lane)
hc = {k: np.asarray(merged["stack"][f"mhc_{k}"]) for k in
      ("phi_pre", "phi_post", "phi_res", "b_pre", "b_post", "b_res",
       "a_pre", "a_post", "a_res")}

emb_j_full = np.asarray(merged["embedding"]["embedding"])
x_j = jnp.asarray(emb_j_full[seqs[:2]]) * math.sqrt(cfg.d_model)
stream_j = jnp.broadcast_to(x_j[:, :, None, :], (*x_j.shape[:2], n_lane, x_j.shape[-1]))
ut = model.debug["u"]
yt = model.debug["y"]
st = model.debug["stream"]
for l in range(L):
    lp = jax.tree.map(lambda a: a[l], blkp)
    b_, t_, n_, c_ = stream_j.shape
    xf = stream_j.astype(jnp.float32)
    nx = _aq(_rms_unit(stream_j.reshape(b_, t_, n_ * c_)), QFLAG)
    hpre = jax.nn.sigmoid(hc["a_pre"][l] * (nx @ jnp.asarray(hc["phi_pre"][l]))
                          + jnp.asarray(hc["b_pre"][l]) + jnp.asarray(pre_off[l]))
    u_j = jnp.einsum("btn,btnc->btc", hpre, xf)
    du = float(np.abs(np.asarray(u_j) - ut[l].numpy()).max())
    # --- submodule bisect inside the block
    pj = {"params": lp}
    s0 = jnp.asarray(model.debug["h1"][l].numpy())
    h1_j = (1 + lp["ZCRMSNorm_0"]["scale"]) * u_j / jnp.sqrt(
        jnp.mean(u_j.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    dh1 = float(np.abs(np.asarray(h1_j) - model.debug["h1"][l].numpy()).max())
    attn_j = mha.apply({"params": lp["self_attn"]}, h1_j, mask=mask_j, rope=rope_j, quant=QFLAG)
    dattn = float(np.abs(np.asarray(attn_j) - model.debug["attn"][l].numpy()).max())
    h2_j = (1 + lp["post_attn_norm"]["scale"]) * attn_j / jnp.sqrt(
        jnp.mean(attn_j.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    dh2 = float(np.abs(np.asarray(h2_j) - model.debug["h2"][l].numpy()).max())
    xj = u_j + jax.nn.sigmoid(lp["attn_gate"]) * h2_j
    h3_j = (1 + lp["pre_hada_norm"]["scale"]) * xj / jnp.sqrt(
        jnp.mean(xj.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    dh3 = float(np.abs(np.asarray(h3_j) - model.debug["h3"][l].numpy()).max())
    hada_j = hada.apply({"params": lp["hadamard_mlp"]}, h3_j)
    dhada = float(np.abs(np.asarray(hada_j) - model.debug["hada"][l].numpy()).max())
    print(f"    norm {dh1:.2e}  attn {dattn:.2e} (|a| {np.abs(np.asarray(attn_j)).max():.2f})"
          f"  h2 {dh2:.2e}  h3 {dh3:.2e}  hada {dhada:.2e}"
          f" (|h| {np.abs(np.asarray(hada_j)).max():.2f})")
    if l == 0:
        from needle.model.architecture import apply_rope as jar, _shift_right, _mask_diag
        from needle.model import quantize as JQ
        attn_j2, st2 = mha.apply({"params": lp["self_attn"]}, h1_j, mask=mask_j,
                                 rope=rope_j, quant=QFLAG, capture_intermediates=True,
                                 mutable=["intermediates"])
        I2 = {}
        for p_, v_ in walk(st2.get("intermediates", {})):
            try:
                I2["/".join(p_)] = np.asarray(v_)
            except Exception:
                pass

        def cmp(key, arr, tl):
            a = np.asarray(arr)
            d = np.abs(a - tl)
            print(f"      {key:<12} {d.max():.3e}  rel {d.max()/max(1e-9,np.abs(a).max()):.2e}"
                  f"  |.| {np.abs(a).max():.3f}")

        for jk, tk in (("q_proj/__call__", "q_dense"), ("k_proj/__call__", "k_dense"),
                       ("v_proj/__call__", "v_dense"), ("q_norm/__call__", "q_norm"),
                       ("k_norm/__call__", "k_norm"), ("gate_proj/__call__", None),
                       ("out_proj/__call__", "attn_out")):
            if jk in I2 and tk:
                cmp(tk, I2[jk], model.debug[tk][l].numpy())
        qd = jnp.asarray(I2["q_proj/__call__"])
        kd = jnp.asarray(I2["k_proj/__call__"])
        vd = jnp.asarray(I2["v_proj/__call__"])
        qt = lp["self_attn"]["q_taps"] if False else lp["q_taps"]
        kt, vt = lp["k_taps"], lp["v_taps"]
        n_taps = cfg.qkv_conv_taps

        def jtapped(z, j):
            s = _shift_right(z, j)
            if j == 0:
                return s
            return s * _mask_diag(mask_j, j)[..., None].astype(z.dtype)

        qj = sum(qt[j] * jtapped(qd, j) for j in range(n_taps))
        kj = sum(kt[j] * jtapped(kd, j) for j in range(n_taps))
        vj = sum(vt[j] * jtapped(vd, j) for j in range(n_taps))
        cmp("q_taps", qj, model.debug["q_taps"][l].numpy())
        cmp("k_taps", kj, model.debug["k_taps"][l].numpy())
        cmp("v_taps", vj, model.debug["v_taps"][l].numpy())
        bsz = ids_j.shape[0]
        qh = qj.reshape(bsz, T, cfg.num_heads, cfg.qk_head_dim).transpose(0, 2, 1, 3)
        kh = kj.reshape(bsz, T, cfg.num_kv_heads, cfg.qk_head_dim).transpose(0, 2, 1, 3)
        vh = vj.reshape(bsz, T, cfg.num_kv_heads, cfg.v_head_dim).transpose(0, 2, 1, 3)
        qr = jar(qh, *rope_j)
        kr = jar(kh, *rope_j)
        cmp("q_rope", qr, model.debug["q_rope"][l].numpy())
        cmp("k_rope", kr, model.debug["k_rope"][l].numpy())
        cmp("q_q", JQ.fake_quant(qr, qr.shape[-1], 8), model.debug["q_q"][l].numpy())
        cmp("k_q", JQ.fake_quant(kr, kr.shape[-1], 8), model.debug["k_q"][l].numpy())
        cmp("v_q", JQ.fake_quant(vh, vh.shape[-1], 8), model.debug["v_q"][l].numpy())
        # scores
        fused = max(cfg.qk_head_dim, cfg.v_head_dim)
        qf = JQ.fake_quant(qr, qr.shape[-1], 8) * math.sqrt(fused / cfg.qk_head_dim)
        kk = JQ.fake_quant(kr, kr.shape[-1], 8)
        qf = jnp.pad(qf, ((0, 0), (0, 0), (0, 0), (0, fused - cfg.qk_head_dim)))
        kk = jnp.pad(kk, ((0, 0), (0, 0), (0, 0), (0, fused - cfg.qk_head_dim)))
        sc = jnp.einsum("bhts,bhks->bhtk", qf, kk) / math.sqrt(fused)
        cmp("scores", sc, model.debug["scores"][l].numpy())
        cmp("attn_raw", attn_j2, model.debug["attn_raw"][l].numpy())
    db = float(np.abs(np.asarray(blk.apply({"params": lp}, u_j, mask=mask_j, rope=rope_j,
                                           quant=True, engram_kv=(ek_j, ev_j),
                                           site_flags=jnp.asarray(flags[l])))
                      - (yt[l] + ut[l]).numpy()).max())
    y_j = blk.apply({"params": lp}, u_j, mask=mask_j, rope=rope_j, quant=QFLAG,
                    engram_kv=(ek_j, ev_j), site_flags=jnp.asarray(flags[l])) - u_j
    hpost = 2 * jax.nn.sigmoid(hc["a_post"][l] * (nx @ jnp.asarray(hc["phi_post"][l]))
                               + jnp.asarray(hc["b_post"][l]) + jnp.asarray(post_off[l]))
    res = nx @ jnp.asarray(hc["phi_res"][l])
    hres = _sinkhorn(hc["a_res"][l] * res.reshape(b_, t_, n_, n_) + jnp.asarray(hc["b_res"][l]))
    stream_j = (jnp.einsum("btij,btjc->btic", hres, xf)
                + hpost[..., None] * y_j.astype(jnp.float32)[:, :, None, :])
    ds_in = float(np.abs(np.asarray(st[l]) - np.asarray(model.debug["stream"][l])).max())
    ds = float(np.abs(np.asarray(stream_j) - np.asarray(
        model.debug["stream"][l + 1] if l + 1 < L else model.debug["final"])).max())
    print(f"layer {l:>2}: u {du:.2e}  blockout {db:.2e}  |stream| {np.abs(np.asarray(stream_j)).max():.3f}"
          f"  stream_out {ds:.2e}")
    if l == 2:
        break
