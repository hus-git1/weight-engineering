"""Chunked gated-delta rule for TRAINING Qwen3.5's linear-attention layers in MLX.

In training mode mlx-lm runs the recurrence as a Python loop over every token
(`gated_delta_ops`); autograd then keeps ~5 MB of intermediates per token per layer, i.e. >10 GB
for one layer at 2.5k tokens. This computes the same recurrence chunk by chunk (C tokens at a
time) with matrix products, the standard way gated DeltaNet is trained: memory ~T/C states and
~50x fewer graph nodes. Verified against `gated_delta_ops` (outputs, final state, gradients) in
scripts/test_chunked_delta.py.

Recurrence (per head, state S in R^{Dv x Dk}, decay g_t in (0,1], beta_t in [0,1]):
    S_t = g_t S_{t-1} + u_t k_t^T,   u_t = beta_t (v_t - g_t S_{t-1} k_t),   y_t = S_t q_t
Within a chunk starting from S_0, with gamma_t = prod_{s<=t} g_s:
    (I + A) U = diag(beta) V - diag(beta*gamma) K S_0^T,   A_ts = beta_t (gamma_t/gamma_s) k_t.k_s (s<t)
    Y   = diag(gamma) Q S_0^T + M U,                       M_ts = (gamma_t/gamma_s) q_t.k_s   (s<=t)
    S_C = gamma_C S_0 + U^T diag(gamma_C/gamma) K
(I + A)^{-1} is computed by forward substitution (numerically stable). Note: the algebraically
exact Neumann product prod_j (I + (-A)^(2^j)) is NOT usable: on real Qwen activations (highly
correlated keys) it loses all precision and the model output overflows to NaN by layer 6.
"""
import mlx.core as mx
import mlx.nn as nn

CHUNK = 32  # forward substitution is C-1 sequential steps; autograd memory grows ~C^2 per layer


def _unit_lower_inverse(A):
    """(I + A)^{-1} for strictly lower-triangular A [..., C, C], by forward substitution:
    row i of X = e_i - A[i, :i] @ X[:i]."""
    C = A.shape[-1]
    eye = mx.eye(C, dtype=A.dtype)
    rows = [mx.broadcast_to(eye[0], A.shape[:-2] + (C,))]
    for i in range(1, C):
        prev = mx.stack(rows, axis=-2)                              # [..., i, C]
        rows.append(eye[i] - (A[..., i:i + 1, :i] @ prev)[..., 0, :])
    return mx.stack(rows, axis=-2)


def chunked_gated_delta(q, k, v, log_g, beta, state=None, mask=None, chunk=CHUNK):
    """q, k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; log_g, beta: [B,T,Hv]; state: [B,Hv,Dv,Dk].
    Returns y [B,T,Hv,Dv] (q.dtype) and the final state (float32), like gated_delta_ops."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (rep := Hv // Hk) > 1:
        q, k = mx.repeat(q, rep, -2), mx.repeat(k, rep, -2)
    f32 = mx.float32
    q, k, v, log_g, beta = (x.astype(f32) for x in (q, k, v, log_g, beta))
    if mask is not None:  # masked steps leave the state untouched: g = 1, beta = 0
        m = mask.astype(mx.bool_)[..., None]
        log_g, beta = mx.where(m, log_g, 0.0), mx.where(m, beta, 0.0)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=f32)
    C = chunk
    pad = (-T) % C
    if pad:  # padded steps are no-ops for the state (log_g = 0, beta = 0); outputs are dropped
        padT = lambda x: mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        q, k, v, log_g, beta = (padT(x) for x in (q, k, v, log_g, beta))
    N = (T + pad) // C
    # -> [B, H, N, C, ...]
    q, k, v = (x.reshape(B, N, C, Hv, -1).transpose(0, 3, 1, 2, 4) for x in (q, k, v))
    log_g, beta = (x.reshape(B, N, C, Hv).transpose(0, 3, 1, 2) for x in (log_g, beta))

    lg = mx.cumsum(log_g, axis=-1)                                  # log gamma_t within chunk
    decay = mx.exp(mx.minimum(lg[..., :, None] - lg[..., None, :], 0.0))  # gamma_t/gamma_s (s<=t)
    strict = mx.tril(mx.ones((C, C), dtype=f32), -1)
    incl = mx.tril(mx.ones((C, C), dtype=f32), 0)
    kT = k.swapaxes(-1, -2)
    A = beta[..., None] * (k @ kT) * decay * strict
    Tinv = _unit_lower_inverse(A)
    W = Tinv @ (k * (beta * mx.exp(lg))[..., None])                # [B,H,N,C,Dk]
    Ub = Tinv @ (v * beta[..., None])                              # [B,H,N,C,Dv]
    M = (q @ kT) * decay * incl                                    # [B,H,N,C,C]
    Qg = q * mx.exp(lg)[..., None]
    lg_last = lg[..., -1:]
    Kend = k * mx.exp(lg_last - lg)[..., None]                     # [B,H,N,C,Dk]
    g_chunk = mx.exp(lg_last)[..., None]                           # [B,H,N,1,1]

    S, ys = state.astype(f32), []
    for n in range(N):
        St = S.swapaxes(-1, -2)                                    # [B,H,Dk,Dv]
        U = Ub[:, :, n] - W[:, :, n] @ St                          # [B,H,C,Dv]
        ys.append(Qg[:, :, n] @ St + M[:, :, n] @ U)
        S = g_chunk[:, :, n] * S + U.swapaxes(-1, -2) @ Kend[:, :, n]
    y = mx.stack(ys, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, N * C, Hv, Dv)[:, :T]
    return y.astype(out_dtype), S


def patch_qwen3_5():
    """Use the chunked recurrence whenever mlx-lm would fall back to the per-token loop
    (training mode). Inference / eval mode keeps mlx-lm's Metal kernel."""
    from mlx_lm.models import gated_delta as G
    from mlx_lm.models import qwen3_5

    orig = G.gated_delta_update

    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *,
                           use_kernel=True, lower_bound=None, allow_neg_eigval=False):
        if use_kernel:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True,
                        lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        if lower_bound is None:  # log of G.compute_g, computed directly (no exp/log round trip)
            log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        else:
            log_g = mx.log(G.compute_lower_bound_g(A_log, a, dt_bias, lower_bound))
        return chunked_gated_delta(q, k, v, log_g, beta, state, mask)

    qwen3_5.gated_delta_update = gated_delta_update
    return orig
