"""Run an evaluation and write per-sample results to JSONL.

  python scripts/eval.py --task math --n 50 --out results/base_math.jsonl
  python scripts/eval.py --task code --n 50 --patch patches/loopfix.safetensors --out results/fix_code.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import model as M
from wsp.evals import code_eval, math_eval
from wsp.loops import is_looping

EVALS = {"math": math_eval, "code": code_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=EVALS, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--patch", help="weight patch (torch backend only)")
    ap.add_argument("--adapter", help="LoRA adapter dir (mlx backend only)")
    ap.add_argument("--adapter-scale", type=float, default=1.0, help="scale the LoRA update (0 = base)")
    ap.add_argument("--adapter-file", help="checkpoint file inside --adapter, e.g. 00111_adapters.safetensors")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ev = EVALS[a.task]
    tasks = ev.load_tasks(a.n, a.offset)
    if a.backend == "mlx":
        from wsp import mlx_backend as B
        model, tok = B.load(adapter=a.adapter, patch=a.patch, adapter_scale=a.adapter_scale,
                            adapter_file=a.adapter_file)
    else:
        B = M
        model, tok = M.load(patch=a.patch)
    t0 = time.time()
    outs = B.generate(model, tok, [t["messages"] for t in tasks], thinking=not a.no_thinking,
                      max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, seed=a.seed,
                      sampling=M.GREEDY if a.greedy else M.THINKING_SAMPLING,
                      on_batch=lambda d, n: print(f"  {d}/{n} done ({time.time()-t0:.0f}s)", flush=True))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    correct = loops = toks = 0
    with open(a.out, "w") as f:
        for t, o in zip(tasks, outs):
            ok = ev.score(t, o["text"])
            loop = is_looping(o["text"], o["finished"])
            correct, loops, toks = correct + ok, loops + loop, toks + o["n_tokens"]
            f.write(json.dumps(dict(id=t["id"], messages=t["messages"], correct=ok, looping=loop, **o)) + "\n")

    n = len(tasks)
    summary = dict(task=a.task, n=n, backend=a.backend, patch=a.patch, adapter=a.adapter, adapter_scale=a.adapter_scale, adapter_file=a.adapter_file, accuracy=round(correct / n, 4),
                   loop_rate=round(loops / n, 4), unfinished=sum(not o["finished"] for o in outs),
                   avg_tokens=round(toks / n), minutes=round((time.time() - t0) / 60, 1),
                   think_closed=sum("</think>" in o["text"] for o in outs),
                   avg_waits=round(sum(o["text"].count("Wait") for o in outs) / n, 1))
    print(json.dumps(summary))
    with open(a.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
