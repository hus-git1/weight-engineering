"""Diagnose the loop direction from eval results and write a weight patch.

  python scripts/find_and_edit.py --results results/base_math.jsonl --layers 15 19 --k 64 \
      --out patches/loopfix.safetensors

Without --layers it prints a per-layer report so you can choose layers yourself.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import model as M
from wsp.capture import capture
from wsp.edit import ablate, count_changed, loop_direction, rank_columns, save_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=24, help="per class")
    ap.add_argument("--stats", default="results/loop_stats.pt")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = [json.loads(l) for p in a.results for l in open(p)]
    loop = [r for r in rows if r["looping"]][:a.max_samples]
    clean = [r for r in rows if not r["looping"] and r["finished"]][:a.max_samples]
    print(f"{len(loop)} looping and {len(clean)} healthy samples")
    if not loop or not clean:
        sys.exit("Need both looping and healthy samples to contrast.")

    model, tok = M.load()
    if os.path.exists(a.stats):
        stats = torch.load(a.stats)
        print(f"reusing cached activation stats from {a.stats}")
    else:
        stats = capture(model, tok, loop + clean)
        torch.save(stats, a.stats)

    print("\nlayer | loop-vs-clean shift in MLP output | top-64 neuron share of that shift")
    for l in range(len(model.model.layers)):
        shift = (stats["loop"][1][l] - stats["clean"][1][l]).norm().item()
        contrib = rank_columns(model, stats, l, loop_direction(stats, l))
        top = contrib.topk(a.k).values.clamp(min=0).sum() / contrib.clamp(min=0).sum()
        print(f"{l:5d} | {shift:8.3f} | {top:.2f}")

    if not (a.layers and a.out):
        return
    edits = {}
    for l in a.layers:
        d = loop_direction(stats, l)
        cols = rank_columns(model, stats, l, d).topk(a.k).indices.sort().values
        edits[l] = ablate(model, l, d, cols.to(model.device), a.strength)
    save_patch(a.out, edits, meta=dict(layers=a.layers, k=a.k, strength=a.strength, results=a.results))
    n = count_changed(a.out)
    print(f"\nwrote {a.out}: {n:,} weights changed ({100*n/sum(p.numel() for p in model.parameters()):.4f}% of the text model)")


if __name__ == "__main__":
    main()
