"""Check wsp/chunked_delta.py against mlx-lm's per-token reference (gated_delta_ops):
outputs, final state and gradients, on the real layer-0 decay parameters.

  python scripts/test_chunked_delta.py
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm import load
from mlx_lm.models.gated_delta import compute_g, gated_delta_ops, normalize_qk

from wsp.chunked_delta import chunked_gated_delta

model, _ = load("Qwen/Qwen3.5-0.8B")
lin = next(l.linear_attn for l in model.layers if hasattr(l, "linear_attn"))
A_log, dt_bias = lin.A_log, lin.dt_bias
H, D = A_log.shape[0], 128


def inputs(T, B=1, seed=0, dtype=mx.bfloat16, hard=False):
    """hard=True mimics real activations: nearly identical keys, beta ~ 1, decay ~ 1
    (the regime where an unstable triangular inverse blows up)."""
    mx.random.seed(seed)
    r = lambda *s: mx.random.normal(s).astype(dtype)
    kk = r(B, T, H, D)
    if hard:
        kk = (r(B, 1, H, D) * 4 + kk * 0.3).astype(dtype)
    q, k = normalize_qk(r(B, T, H, D), kk, inv_scale=D ** -0.5, eps=1e-6)
    a, b = r(B, T, H) * 2, r(B, T, H) * 2
    if hard:
        a, b = (a - 8).astype(dtype), (b + 6).astype(dtype)
    return q, k, r(B, T, H, D), a, b


def ref(q, k, v, a, b, state, mask=None):
    return gated_delta_ops(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, mask)


def new(q, k, v, a, b, state, mask=None):
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    return chunked_gated_delta(q, k, v, log_g, mx.sigmoid(b), state, mask)


def rel(x, y):
    x, y = x.astype(mx.float32), y.astype(mx.float32)
    return (mx.abs(x - y).max() / (mx.abs(y).max() + 1e-6)).item()


ok = True
for T, hard in ((1, 0), (7, 0), (32, 0), (33, 0), (200, 0), (700, 0), (300, 1), (700, 1)):
    for with_state in (False, True):
        q, k, v, a, b = inputs(T, seed=T, hard=bool(hard))
        s0 = mx.random.normal((1, H, D, D)) * 0.1 if with_state else None
        mask = None
        if T == 200:  # padding-style mask: last 30 steps masked out
            mask = mx.arange(T)[None] < T - 30
        (y1, s1), (y2, s2) = ref(q, k, v, a, b, s0, mask), new(q, k, v, a, b, s0, mask)
        sel = slice(None) if mask is None else slice(0, T - 30)
        ey, es = rel(y2[:, sel], y1[:, sel]), rel(s2, s1)
        good = ey < 2e-2 and es < 1e-3  # y is bf16 (~3 significant digits); state is float32
        ok &= good
        print(f"T={T:4d} hard={hard} state0={with_state!s:5} mask={mask is not None!s:5}  "
              f"y rel err {ey:.1e}  state rel err {es:.1e}  {'ok' if good else 'MISMATCH'}")

# gradients w.r.t. every input, float32 to make the comparison sharp
q, k, v, a, b = inputs(150, seed=1, dtype=mx.float32)
s0 = mx.random.normal((1, H, D, D)) * 0.1
mx.random.seed(2)
cot_y, cot_s = mx.random.normal((1, 150, H, D)), mx.random.normal((1, H, D, D))
for name, f in (("ref", ref), ("new", new)):
    loss = lambda *xs: (f(*xs)[0] * cot_y).sum() + (f(*xs)[1] * cot_s).sum()
    globals()[f"g_{name}"] = mx.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, a, b, s0)
for n, g1, g2 in zip("q k v a b state0".split(), g_ref, g_new):
    e = rel(g2, g1)
    ok &= e < 1e-3
    print(f"grad {n:6s} rel err {e:.1e}  {'ok' if e < 1e-3 else 'MISMATCH'}")

# End to end on real training data. The whole model runs in bf16, so even mlx-lm's own training
# path (per-token loop) differs from its Metal kernel by ~7% max logit error / KL ~1e-2 on long
# inputs. Requirement: ours is no further from the kernel than that reference is (+50%).
import json
from mlx_lm.models import qwen3_5
from wsp.chunked_delta import patch_qwen3_5


def kl(p, q):  # mean per-token KL(softmax p || softmax q)
    lp, lq = (x - mx.logsumexp(x, -1, keepdims=True) for x in (p, q))
    return (mx.exp(lp) * (lp - lq)).sum(-1).mean().item()


reference_loop = qwen3_5.gated_delta_update
patch_qwen3_5()
ours = qwen3_5.gated_delta_update
_, tok = load("Qwen/Qwen3.5-0.8B")
rows = [json.loads(l) for l in open("data/mix/train.jsonl")]
rows = sorted(rows, key=lambda r: len(json.dumps(r)))
for r in (rows[0], rows[len(rows) // 2], rows[-1]):
    ids = mx.array(tok.apply_chat_template(r["messages"], tools=r.get("tools"), return_dict=False))[None]
    model.eval()
    kernel = model(ids).astype(mx.float32)
    model.train()
    qwen3_5.gated_delta_update = reference_loop
    base = kl(kernel, model(ids).astype(mx.float32))
    qwen3_5.gated_delta_update = ours
    l_new = model(ids).astype(mx.float32)
    mine = kl(kernel, l_new)
    good = not mx.isnan(l_new).any().item() and mine <= 1.5 * base + 1e-4
    ok &= good
    print(f"real example, {ids.shape[1]} tokens: KL to kernel {mine:.2e} (mlx-lm reference loop "
          f"{base:.2e}), same top-1 {(kernel.argmax(-1) == l_new.argmax(-1)).mean().item():.1%}  "
          f"{'ok' if good else 'MISMATCH'}")
model.eval()
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
