"""MLX generation backend (~5x faster than the PyTorch/MPS path on Apple Silicon).

Same interface as wsp.model: load() -> (model, tok); generate(...) -> [{text, n_tokens, finished}].
Supports LoRA adapters (mlx_lm format). Weight patches (wsp.edit) are PyTorch-only.
"""
import os

import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from wsp.model import DEFAULT_MODEL, GREEDY, THINKING_SAMPLING, render  # noqa: F401  (re-exported)

ENDOFTEXT = "<|endoftext|>"


MEMORY_LIMIT_GB = 5    # this Mac has 16 GB shared with the OS/apps; stay well clear of swap
CACHE_LIMIT_GB = 0.5   # MLX keeps freed buffers cached; cap it so memory is actually returned


def load(repo=DEFAULT_MODEL, adapter=None, patch=None, adapter_scale=1.0, adapter_file=None):
    """adapter_file: a checkpoint inside the adapter dir (e.g. 00111_adapters.safetensors);
    adapter_scale: multiply the LoRA update (0 = base model, 1 = as trained)."""
    if patch:
        raise ValueError("Weight patches are only supported by the torch backend.")
    mx.set_memory_limit(int(MEMORY_LIMIT_GB * 2**30))
    mx.set_cache_limit(int(CACHE_LIMIT_GB * 2**30))
    model, tok = mlx_load(repo, adapter_path=adapter)
    if adapter and adapter_file:
        model.load_weights(os.path.join(adapter, adapter_file), strict=False)
    if adapter and adapter_scale != 1.0:
        n = 0
        for _, m in model.named_modules():
            if hasattr(m, "lora_a") and hasattr(m, "scale"):
                m.scale = m.scale * adapter_scale
                n += 1
        assert n, "no LoRA layers found to scale"
    # Stop on <|endoftext|> too (MLX only stops on <|im_end|> by default).
    eot = tok.convert_tokens_to_ids(ENDOFTEXT)
    if hasattr(tok, "add_eos_token"):
        tok.add_eos_token(ENDOFTEXT)
    else:
        tok.eos_token_ids.add(eot)
    return model, tok


def generate(model, tok, conversations, thinking=True, max_new_tokens=4096, batch_size=32,
             sampling=THINKING_SAMPLING, seed=0, on_batch=None, tools=None):
    mx.random.seed(seed)
    temp = sampling.get("temperature", 0.0)
    sampler = make_sampler(temp=temp, top_p=sampling.get("top_p", 0.0) if temp > 0 else 0.0,
                           top_k=sampling.get("top_k", 0) if temp > 0 else 0)
    procs = make_logits_processors(presence_penalty=sampling.get("presence_penalty") or None,
                                   presence_context_size=max_new_tokens)
    results = []
    for i in range(0, len(conversations), batch_size):
        chunk = conversations[i:i + batch_size]
        prompts = [tok.encode(render(tok, c, thinking, tools), add_special_tokens=False) for c in chunk]
        r = batch_generate(model, tok, prompts, max_tokens=max_new_tokens, sampler=sampler,
                           logits_processors=procs, return_token_ids=True,
                           completion_batch_size=batch_size)
        for text, ids in zip(r.texts, r.token_ids):
            results.append(dict(text=text, n_tokens=len(ids), finished=len(ids) < max_new_tokens))
        if on_batch:
            on_batch(i + len(chunk), len(conversations))
    return results


def get(name):
    """Pick a backend module by name: 'mlx' or 'torch'."""
    if name == "mlx":
        import wsp.mlx_backend as b
    else:
        import wsp.model as b
    return b
