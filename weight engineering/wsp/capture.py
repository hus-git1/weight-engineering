"""Capture MLP activations on loop vs. healthy reasoning tokens (teacher-forced replay).

For every layer we hook mlp.down_proj and accumulate:
  - mean of its input  h   (3584-dim: which neurons fire)
  - mean of its output W h (1024-dim: what the MLP writes into the residual stream)
separately for tokens inside a loop region and tokens of healthy reasoning.
"""
import torch

from wsp.loops import loop_start
from wsp.model import render


class Accumulator:
    def __init__(self):
        self.sum_in, self.sum_out, self.n = {}, {}, 0

    def add(self, layer, h, y):
        h, y = h.float().sum(0).cpu(), y.float().sum(0).cpu()
        if layer in self.sum_in:
            self.sum_in[layer] += h
            self.sum_out[layer] += y
        else:
            self.sum_in[layer], self.sum_out[layer] = h, y

    def means(self):
        return ({l: v / self.n for l, v in self.sum_in.items()},
                {l: v / self.n for l, v in self.sum_out.items()})


def _layers(model):
    return model.model.layers


@torch.inference_mode()
def capture(model, tok, samples, thinking=True, max_tokens=6000):
    """samples: dicts with messages, text, looping. Returns {'loop': (in, out), 'clean': (in, out)}."""
    accs = {"loop": Accumulator(), "clean": Accumulator()}
    state = {}

    def hook(layer):
        def fn(mod, inp, out):
            mask = state["mask"]
            if mask.any():
                state["acc"].add(layer, inp[0][0][mask], out[0][mask])
        return fn

    handles = [L.mlp.down_proj.register_forward_hook(hook(i)) for i, L in enumerate(_layers(model))]
    try:
        for s in samples:
            prompt = render(tok, s["messages"], thinking)
            p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
            text = s["text"]
            if s["looping"]:
                start = loop_start(text)
                if start is None:
                    start = max(0, len(text) - 3000)  # repetitive tail of a truncated sample
                kind = "loop"
            else:
                # healthy reasoning: the thinking part only, so we compare like with like
                text = text.split("</think>")[0]
                start, kind = 0, "clean"
            head_ids = tok(text[:start], add_special_tokens=False)["input_ids"]
            tail_ids = tok(text[start:], add_special_tokens=False)["input_ids"]
            ids = torch.tensor(p_ids.tolist() + head_ids + tail_ids)[:max_tokens]
            mask = torch.zeros(len(ids), dtype=torch.bool)
            mask[len(p_ids) + len(head_ids):] = True
            if not mask.any():
                continue
            state.update(mask=mask.to(model.device), acc=accs[kind])
            accs[kind].n += int(mask.sum())
            model(input_ids=ids[None].to(model.device), use_cache=False)
            if model.device.type == "mps":
                torch.mps.empty_cache()
    finally:
        for h in handles:
            h.remove()
    return {k: a.means() for k, a in accs.items() if a.n > 0}
