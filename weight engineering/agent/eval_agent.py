"""Agentic benchmark: run the agent on every task in a fresh copy, grade with hidden tests.

  python agent/eval_agent.py --out results/agent_base.json
  python agent/eval_agent.py --patch patches/loopfix.safetensors --out results/agent_fix.json
  python agent/eval_agent.py --tasks agent/tasks_he --adapter adapters/v1 --out results/he_agent_v1.json

Each run uses a fresh, empty memory directory so results are comparable. Session tasks
(ledger_s1 -> ledger_s2) share that memory, which is what measures memory use.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from harness import run_agent
from wsp import model as M

TASKS = os.path.join(HERE, "tasks")


def grade(ws):
    for d in glob.glob(os.path.join(ws, "**", "__pycache__"), recursive=True):
        shutil.rmtree(d, ignore_errors=True)  # never grade against stale bytecode
    try:
        r = subprocess.run([sys.executable, "-E", "-s", "_hidden_test.py"], cwd=ws, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:  # e.g. the agent left an infinite loop
        return False
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--patch")
    ap.add_argument("--adapter")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--adapter-file")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tasks", default=TASKS, help="task directory (agent/tasks: 6 hand-made; agent/tasks_he: 80 HumanEval)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = sorted(os.listdir(a.tasks))
    metas = {n: json.load(open(os.path.join(a.tasks, n, "task.json"))) for n in names}
    # run session prerequisites first
    order = [n for n in names if "after" not in metas[n]] + [n for n in names if "after" in metas[n]]
    if a.only:
        order = [n for n in order if n in a.only]
    # resumable: finished rows are kept in <out>.partial.jsonl; a session-2 task that still has to
    # run also re-runs its session-1 prerequisite (the memory directory is fresh each invocation)
    partial = a.out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        done = {r["task"]: r for r in map(json.loads, open(partial))}
    needed = {metas[n]["after"] for n in order if n not in done and "after" in metas[n]}
    todo = [n for n in order if n not in done or n in needed]
    print(f"{len(order)} tasks, {len(order) - len(todo)} already done", flush=True)

    from wsp.mlx_backend import get
    B = get(a.backend)
    model, tok = B.load(patch=a.patch) if a.backend == "torch" else B.load(adapter=a.adapter, adapter_scale=a.adapter_scale, adapter_file=a.adapter_file)
    tmp = tempfile.mkdtemp(prefix="agent_eval_")
    memory = os.path.join(tmp, "memory")
    rows = []
    for n in todo:
        ws = os.path.join(tmp, n)
        shutil.copytree(os.path.join(a.tasks, n), ws,
                        ignore=shutil.ignore_patterns("_hidden_test.py", "task.json"))
        rec = run_agent(metas[n]["prompt"], ws, metas[n]["project"], memory, model, tok,
                        a.max_steps, a.thinking, backend=B, verbose=False)
        shutil.copy(os.path.join(a.tasks, n, "_hidden_test.py"), ws)
        ok = grade(ws)
        used = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
        row = dict(task=n, kind=metas[n].get("kind"), passed=ok, status=rec["status"], steps=rec["steps"],
                   edited_tests=any("<parameter=path>\ntest_" in u and "write_file" in u for u in used),
                   memory_reads=sum("memory_search" in u or "memory_read" in u for u in used),
                   memory_writes=sum("memory_write" in u for u in used))
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(partial, "a") as f:
            f.write(json.dumps(row) + "\n")
    fresh = {r["task"] for r in rows}
    rows = [done[n] for n in order if n in done and n not in fresh] + rows

    kinds = sorted({r["kind"] for r in rows if r["kind"]})
    by_kind = {k: round(sum(r["passed"] for r in rows if r["kind"] == k) / sum(r["kind"] == k for r in rows), 3) for k in kinds}
    summary = dict(backend=a.backend, patch=a.patch, adapter=a.adapter, adapter_scale=a.adapter_scale,
                   adapter_file=a.adapter_file, tasks_dir=a.tasks, by_kind=by_kind,
                   pass_rate=round(sum(r["passed"] for r in rows) / len(rows), 3),
                   avg_steps=round(sum(r["steps"] for r in rows) / len(rows), 1), tasks=rows,
                   memory_dir=memory)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"pass rate {summary['pass_rate']} {by_kind}  avg steps {summary['avg_steps']}  (memory kept at {memory})")


if __name__ == "__main__":
    main()
