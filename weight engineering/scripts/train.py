"""LoRA fine-tuning of Qwen3.5-0.8B on data/mix (own loop built from mlx-lm pieces).

  python scripts/train.py -c configs/lora_v1.yaml            # start (or --resume)
  python scripts/train.py -c configs/lora_v1.yaml --probe 3  # memory/speed probe on the longest examples

Why not `mlx_lm lora` (it OOMs on this model / data on a 16 GB Mac):
- its training-mode linear attention is a per-token Python loop (>10 GB/layer of autograd state at
  2.5k tokens) -> wsp/chunked_delta.py (verified by scripts/test_chunked_delta.py);
- its loss computes 248k-way logits for every position, even masked ones -> logits only at the
  trained positions;
- it trains only the LAST assistant turn, so an agent run must be split into one example per turn
  (6x the tokens) -> each example lists "train_turns"; every listed turn is trained, conditioned on
  its history (same gradient: each turn's prompt is an exact prefix of the run, asserted here);
- its prompt mask ends at the template's default `<think>\\n\\n`, so it would also train the
  `</think>\\n\\n` we supply at inference -> mask = exact inference prompt (asserted).
Every trained turn has equal weight (mean loss over its tokens), as with one example per turn.
Adapters are saved in mlx-lm's format, so `mlx_lm.load(..., adapter_path=...)` / our evals load them.
"""
import argparse
import json
import os
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten, tree_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

from wsp.chunked_delta import patch_qwen3_5

MEMORY_LIMIT_GB, CACHE_LIMIT_GB = 9, 1


def encode(tok, ex, im_end):
    """-> (tokens, segments). Each segment (start, end) is one trained assistant turn:
    tokens[start:end] is the answer the model must produce (ending at <|im_end|>) after
    tokens[:start], which is exactly the prompt it gets at inference."""
    msgs, tools = ex["messages"], ex.get("tools")
    render = lambda m, **kw: tok.apply_chat_template(m, tools=tools, return_dict=False, **kw)
    cut = lambda t: t[:len(t) - t[::-1].index(im_end)]  # drop the template's trailing "\n"
    full = cut(render(msgs[:ex["train_turns"][-1] + 1]))
    segs = []
    for i in ex["train_turns"]:
        prompt = render(msgs[:i], add_generation_prompt=True, enable_thinking=False)
        turn = cut(render(msgs[:i + 1]))
        if turn[:len(prompt)] != prompt or full[:len(turn)] != turn or len(turn) <= len(prompt):
            raise ValueError("a trained turn's inference prompt is not a prefix of the sequence")
        segs.append((len(prompt), len(turn)))
    return full, segs


def targets(segs):
    """Positions whose hidden state predicts a trained token, and per-token weights
    (1/len(turn): each turn counts once, like one example per turn)."""
    pos, w = [], []
    for a, b in segs:
        pos += range(a - 1, b - 1)
        w += [1.0 / (b - a)] * (b - a)
    return mx.array(pos), mx.array([p + 1 for p in pos]), mx.array(w)


def loss_fn(model, toks, pos, tgt, w):
    """Sum over trained turns of each turn's mean token loss."""
    lm = model.language_model
    h = lm.model(toks[None, :-1])[:, pos]  # only trained positions -> 248k-way logits
    logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), toks[tgt][None], reduction="none")
    return (ce[0] * w).sum()


def evaluate(model, data):
    """Mean loss per trained turn."""
    model.eval()  # eval mode: mlx-lm's fast Metal kernel for the linear attention
    total = n = 0
    for toks, segs in data:
        total += loss_fn(model, mx.array(toks), *targets(segs)).item()
        n += len(segs)
    model.train()
    mx.clear_cache()
    return total / n


def save(model, cfg, path, step):
    os.makedirs(path, exist_ok=True)
    w = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(path, "adapters.safetensors"), w)
    mx.save_safetensors(os.path.join(path, f"{step:05d}_adapters.safetensors"), w)
    json.dump({"fine_tune_type": "lora", "num_layers": cfg["num_layers"],
               "lora_parameters": cfg["lora_parameters"], "model": cfg["model"],
               "trainer": "scripts/train.py", "config": cfg},
              open(os.path.join(path, "adapter_config.json"), "w"), indent=1)
    json.dump({"step": step}, open(os.path.join(path, "state.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from adapter_path/state.json")
    ap.add_argument("--probe", type=int, help="time fwd+bwd on the N longest train examples, then exit")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mx.set_memory_limit(int(MEMORY_LIMIT_GB * 2**30))
    mx.set_cache_limit(int(CACHE_LIMIT_GB * 2**30))
    patch_qwen3_5()

    model, tok = load(cfg["model"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    data = {}
    for split in ("train", "valid"):
        rows = [json.loads(l) for l in open(os.path.join(cfg["data"], f"{split}.jsonl"))]
        enc = [encode(tok, r, im_end) for r in rows]
        too_long = sum(len(t) > cfg["max_seq_length"] for t, _ in enc)
        if too_long:
            raise ValueError(f"{too_long} {split} examples exceed max_seq_length (would lose answers)")
        data[split] = enc
    n_tr = len(data["train"])
    print(f"train {n_tr} sequences / {sum(len(sg) for _, sg in data['train'])} trained turns / "
          f"{sum(b - a for _, sg in data['train'] for a, b in sg)} trained tokens / "
          f"{sum(len(t) for t, _ in data['train'])} tokens; valid {len(data['valid'])}", flush=True)

    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    if cfg.get("grad_checkpoint"):
        grad_checkpoint(model.layers[0])
    model.train()
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable parameters: {n_params / 1e6:.2f}M", flush=True)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    if a.probe:
        tr = data["train"]  # longest sequences, then most trained tokens (logit memory)
        n_trained = lambda x: sum(b - a for a, b in x[1])
        worst = (sorted(tr, key=lambda x: -len(x[0]))[:a.probe]
                 + sorted(tr, key=lambda x: -n_trained(x))[:a.probe])
        for toks, segs in worst:
            mx.reset_peak_memory()
            t = time.time()
            loss, g = value_and_grad(model, mx.array(toks), *targets(segs))
            mx.eval(loss, g)
            print(f"len {len(toks)}, {len(segs)} turns, {n_trained((toks, segs))} trained tokens: "
                  f"loss/turn {loss.item() / len(segs):.3f}  {time.time() - t:.1f}s  "
                  f"peak {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
        return

    ga = cfg["sequences_per_step"]
    total_steps = cfg["epochs"] * n_tr // ga
    sched = dict(cfg["lr_schedule"], arguments=list(cfg["lr_schedule"]["arguments"]))
    if sched["arguments"][1] == "auto":  # decay over whatever is left after warmup
        sched["arguments"][1] = total_steps - sched.get("warmup", 0)
    print(f"{total_steps} optimizer steps; schedule {sched}", flush=True)
    opt = optim.Adam(learning_rate=build_schedule(sched))
    path, start = cfg["adapter_path"], 0
    if a.resume and os.path.exists(os.path.join(path, "state.json")):
        start = json.load(open(os.path.join(path, "state.json")))["step"]
        model.load_weights(os.path.join(path, "adapters.safetensors"), strict=False)
        opt.state["step"] = mx.array(start, mx.uint64)  # LR schedule resumes; Adam moments restart
        print(f"resumed at step {start}", flush=True)
    stream = []
    for e in range(cfg["epochs"]):
        order = list(range(n_tr))
        random.Random(cfg["seed"] + e).shuffle(order)
        stream += order
    log = open(cfg["log"], "a")
    if start == 0:
        v = evaluate(model, data["valid"])
        print(f"step 0: val loss {v:.4f}", flush=True)
        log.write(json.dumps({"step": 0, "val_loss": v}) + "\n")

    t0, losses, toks_seen = time.time(), [], 0
    for step in range(start, total_steps):
        acc, n_turns = None, 0
        for idx in stream[step * ga:(step + 1) * ga]:
            toks, segs = data["train"][idx]
            loss, g = value_and_grad(model, mx.array(toks), *targets(segs))
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(loss, acc)
            losses.append(loss.item() / len(segs))
            n_turns += len(segs)
            toks_seen += len(toks)
        opt.update(model, tree_map(lambda x: x / n_turns, acc))  # mean over trained turns
        mx.eval(model.trainable_parameters(), opt.state)
        s = step + 1
        if s % cfg["report_every"] == 0 or s == total_steps:
            el = time.time() - t0
            rec = {"step": s, "train_loss": round(sum(losses) / len(losses), 4),
                   "lr": float(opt.learning_rate), "tok_per_s": round(toks_seen / el),
                   "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
                   "eta_min": round(el / (s - start) * (total_steps - s) / 60)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            losses = []
        if s % cfg["eval_every"] == 0 or s == total_steps:
            v = evaluate(model, data["valid"])
            print(f"step {s}: val loss {v:.4f}", flush=True)
            log.write(json.dumps({"step": s, "val_loss": v}) + "\n")
            log.flush()
        if s % cfg["save_every"] == 0 or s == total_steps:
            save(model, cfg, path, s)
    print("training done", flush=True)


if __name__ == "__main__":
    main()
