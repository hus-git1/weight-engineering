"""Token-level diagnosis of the self-doubt loop and optional patch.

Contrast (per layer, at the MLP output = down_proj):
  A: positions whose NEXT token is a "Wait" token        (model decides to re-doubt)
  B: positions whose NEXT token starts a new line/sentence but is not "Wait"   (control)
The diff of means is the candidate "doubt direction". We report, per layer, how well
projecting onto it separates A from B (AUC on held-out samples), then optionally
ablate it from the top-k contributing down_proj columns of chosen layers.

  python scripts/wait_direction.py --results results/base_math.jsonl                 # report
  python scripts/wait_direction.py --results results/base_math.jsonl --layers 15 19 \
      --k 64 --out patches/waitfix.safetensors                                      # patch
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import model as M
from wsp.edit import ablate, count_changed, save_patch


def wait_token_ids(tok):
    ids = set()
    for s, i in tok.get_vocab().items():
        text = tok.convert_tokens_to_string([s])
        if re.fullmatch(r"[\s*_(]*Wait[\s,.:!*]*", text):
            ids.add(i)
    return ids


def boundary(text):
    return text.endswith("\n") or text.rstrip().endswith((".", ":", "?", "!", "*"))


# --mode doubt: contrast the start of any self-doubt line with the start of a committing line.
DOUBT = re.compile(r"(Wait|Is it|Is there|Is this|Maybe|What if|But |Hmm|Actually|Alternatively|Did I|"
                   r"Could it|Let me (re|double|check|verify)|Let's (re|double|check|verify))")
COMMIT = re.compile(r"(So the|So,? the answer|Therefore|Thus|Hence|The answer is|Final answer|"
                    r"The final|This confirms|Conclusion|.*\\boxed)")
LEAD = re.compile(r"^[\s*\-#>\d.()]*")


def doubt_commit_positions(tok, text, offset):
    """Token positions (in prompt+completion ids) just BEFORE the first word of doubt / commit lines."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [s for s, _ in enc["offset_mapping"]]
    A, B, pos = [], [], 0
    for line in text.split("\n"):
        lead = LEAD.match(line).end()
        body, char = line[lead:], pos + lead
        pos += len(line) + 1
        if not body:
            continue
        kind = "A" if DOUBT.match(body) else "B" if COMMIT.match(body) else None
        if kind is None:
            continue
        # first token starting at or after the line's first word; we read the position before it
        j = next((i for i, s in enumerate(starts) if s >= char), None)
        if j is None or j == 0:
            continue
        (A if kind == "A" else B).append(offset + j - 1)
    return A, B


@torch.inference_mode()
def collect(model, tok, rows, wait_ids, max_tokens=4400, max_control=60, mode="wait"):
    """Per sample and layer: per-position MLP outputs at A and (subsampled) B positions (fp16),
    and summed MLP inputs (neuron activations) for A and B. Selection happens on-device."""
    per_sample = []
    store, sel = {}, {}

    def hook(layer):
        def fn(mod, inp, out):
            h, y = inp[0][0], out[0]
            iA, iB = sel["A"], sel["B"]
            store[layer] = dict(A_out=y[iA].half().cpu(), B_out=y[iB].half().cpu(),
                                A_in=h[iA].float().sum(0).cpu(), B_in=h[iB].float().sum(0).cpu(),
                                nA=len(iA), nB=len(iB))
        return fn

    shared = None
    if mode == "doubt":
        # Global token matching: keep only positions whose CURRENT token precedes both doubt and
        # commit lines somewhere in the data, so the contrast is not just "what came before".
        before = {"A": set(), "B": set()}
        for r in rows:
            c = tok(r["text"], add_special_tokens=False)["input_ids"]
            A0, B0 = doubt_commit_positions(tok, r["text"], 0)
            before["A"].update(c[t] for t in A0)
            before["B"].update(c[t] for t in B0)
        shared = before["A"] & before["B"]
        print(f"{len(shared)} tokens precede both doubt and commit lines")

    hs = [L.mlp.down_proj.register_forward_hook(hook(i)) for i, L in enumerate(model.model.layers)]
    try:
        for n, r in enumerate(rows):
            prompt = M.render(tok, r["messages"], True)
            p = tok(prompt)["input_ids"]
            c = tok(r["text"], add_special_tokens=False)["input_ids"]
            ids = (p + c)[:max_tokens]
            pieces = [tok.decode([t]) for t in ids]
            span = range(len(p), len(ids) - 1)
            if mode == "doubt":
                A, B = doubt_commit_positions(tok, r["text"], len(p))
                A = [t for t in A if t < len(ids) - 1 and ids[t] in shared]
                B = [t for t in B if t < len(ids) - 1 and ids[t] in shared]
                B = [t for t in B if t < len(ids) - 1]
            else:
                A = [t for t in span if ids[t + 1] in wait_ids]
                # Token-matched control: same CURRENT token as some Wait position, different next
                # token. Removes the lexical confound (Wait tends to follow specific tokens).
                before_wait = {ids[t] for t in A}
                B = [t for t in span if ids[t + 1] not in wait_ids and ids[t] in before_wait
                     and pieces[t + 1].strip() and not pieces[t + 1].strip().startswith("Wait")]
            if not A or not B:
                continue
            g = torch.Generator().manual_seed(n)
            B = [B[i] for i in torch.randperm(len(B), generator=g)[:max_control].sort().values]
            sel.update(A=torch.tensor(A, device=model.device), B=torch.tensor(B, device=model.device))
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            per_sample.append(dict(store))
            store.clear()
            if model.device.type == "mps":
                torch.mps.empty_cache()
            print(f"  sample {n+1}/{len(rows)}: {len(A)} wait / {len(B)} control positions", flush=True)
    finally:
        for h in hs:
            h.remove()
    return per_sample


def auc(pos, neg):
    """Probability a random A-position projects higher than a random B-position."""
    pos, neg = pos.flatten(), neg.flatten()
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
    return ((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--mode", choices=["wait", "doubt"], default="wait")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.cache = a.cache or ("results/wait_acts.pt" if a.mode == "wait" else "results/doubt_acts.pt")

    model, tok = M.load()
    if os.path.exists(a.cache):
        data = torch.load(a.cache)
        print(f"reusing {a.cache}")
    else:
        rows = [json.loads(l) for l in open(a.results)][:a.samples]
        wait_ids = wait_token_ids(tok)
        print(f"{len(wait_ids)} 'Wait' token variants")
        data = collect(model, tok, rows, wait_ids, mode=a.mode)
        torch.save(data, a.cache + ".tmp")
        os.replace(a.cache + ".tmp", a.cache)  # atomic: never leave a half-written cache

    train, test = data[: len(data) * 2 // 3], data[len(data) * 2 // 3:]
    cat = lambda part, l, k: torch.cat([s[l][k] for s in part]).float()
    mean_in = lambda part, l, k: sum(s[l][k + "_in"] for s in part) / sum(s[l]["n" + k] for s in part)
    nl = len(model.model.layers)
    dirs, report = {}, []
    print(f"\n{len(train)} train / {len(test)} held-out samples")
    print("layer | held-out AUC of doubt direction | ||mean diff||")
    for l in range(nl):
        diff = cat(train, l, "A_out").mean(0) - cat(train, l, "B_out").mean(0)
        d = diff / diff.norm()
        dirs[l] = d
        s = auc(cat(test, l, "A_out") @ d, cat(test, l, "B_out") @ d)
        report.append((l, s, diff.norm().item()))
        print(f"{l:5d} | {s:.3f} | {diff.norm().item():.3f}")
    torch.save(dict(dirs=dirs, report=report), a.cache.replace("_acts", "_dirs"))

    if not (a.layers and a.out):
        return
    edits = {}
    for l in a.layers:
        d = dirs[l]
        W = model.model.layers[l].mlp.down_proj.weight.float().cpu()
        dh = mean_in(data, l, "A") - mean_in(data, l, "B")
        contrib = dh * (d @ W)
        cols = contrib.topk(a.k).indices.sort().values
        share = contrib.topk(a.k).values.clamp(min=0).sum() / contrib.clamp(min=0).sum()
        print(f"layer {l}: top-{a.k} neurons carry {share:.0%} of the positive push along the direction")
        edits[l] = ablate(model, l, d, cols.to(model.device), a.strength)
    save_patch(a.out, edits, meta=dict(method="wait-direction", layers=a.layers, k=a.k, strength=a.strength))
    n = count_changed(a.out)
    total = sum(p.numel() for p in model.parameters())
    print(f"wrote {a.out}: {n:,} weights changed ({100*n/total:.4f}%)")


if __name__ == "__main__":
    main()
