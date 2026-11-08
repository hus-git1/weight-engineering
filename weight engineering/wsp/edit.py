"""Weight-space edits stored as small patch files instead of full model copies.

A patch records, per layer, which down_proj columns changed and their new values.
Column j of down_proj (shape [hidden, intermediate]) is neuron j's write vector.
"""
import torch
from safetensors.torch import load_file, save_file


def loop_direction(stats, layer):
    """Unit vector in residual space: what the MLP writes more during loops than healthy reasoning."""
    d = stats["loop"][1][layer] - stats["clean"][1][layer]
    return d / d.norm()


def rank_columns(model, stats, layer, d):
    """Score each neuron by how much extra it pushes along d during loops."""
    W = model.model.layers[layer].mlp.down_proj.weight.float().cpu()  # [hidden, inter]
    proj = d @ W                                                       # [inter]: column . d
    dh = stats["loop"][0][layer] - stats["clean"][0][layer]            # extra activation in loops
    return dh * proj                                                   # contribution to the d-shift


def ablate(model, layer, d, cols, strength=1.0):
    """Remove the d-component from the chosen columns. Returns (cols, new_values)."""
    lin = model.model.layers[layer].mlp.down_proj
    W = lin.weight.data
    d_ = d.to(W.device, torch.float32)
    sub = W[:, cols].float()
    sub = sub - strength * torch.outer(d_, d_ @ sub)
    W[:, cols] = sub.to(W.dtype)
    return cols, W[:, cols].detach().cpu().clone()


def save_patch(path, edits, meta=None):
    """edits: {layer: (cols LongTensor, new_values [hidden, k])}."""
    t = {}
    for layer, (cols, vals) in edits.items():
        t[f"L{layer}.cols"] = torch.as_tensor(cols, dtype=torch.long).cpu()
        t[f"L{layer}.vals"] = vals.contiguous()
    save_file(t, path, metadata={k: str(v) for k, v in (meta or {}).items()})


def apply_patch(model, path):
    t = load_file(path)
    for key in [k for k in t if k.endswith(".cols")]:
        layer = int(key[1:].split(".")[0])
        W = model.model.layers[layer].mlp.down_proj.weight.data
        W[:, t[key].to(W.device)] = t[f"L{layer}.vals"].to(W.device, W.dtype)
    return model


def count_changed(path):
    t = load_file(path)
    return sum(v.numel() for k, v in t.items() if k.endswith(".vals"))
