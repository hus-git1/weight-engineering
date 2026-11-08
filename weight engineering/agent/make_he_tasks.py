"""Held-out agent benchmark from HumanEval (never used for training): implement + bug-fix tasks.

  python agent/make_he_tasks.py            # -> agent/tasks_he/ (40 implement + 40 bugfix)
  python agent/eval_agent.py --tasks agent/tasks_he --out results/he_agent_base.json

Same shape as the MBPP training tasks: a module with the function, a visible test with a couple of
the official asserts, and the hidden test = the full official HumanEval check. Each problem is
used once. Every task is verified: reference passes the hidden test, the start state fails it, and
the visible test fails at the start (so the agent sees the failure).
"""
import ast
import os
import random
import re
import shutil
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_train_tasks import MODULES, MUTATIONS, run_hidden, write_task

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks_he")


def visible_asserts(test, entry, k):
    out = []
    for line in test.splitlines():
        s = line.strip()
        if s.startswith("assert") and "candidate(" in s:
            s = s.replace("candidate(", f"{entry}(")
            try:
                ast.parse(s)
            except SyntaxError:  # multi-line assert
                continue
            out.append(s)
        if len(out) == k:
            break
    return out


def main(n_impl=40, n_fix=40, seed=0):
    rng = random.Random(seed)
    rows = list(load_dataset("openai/openai_humaneval", split="test"))
    rng.shuffle(rows)
    shutil.rmtree(ROOT, ignore_errors=True)
    made = {"implement": 0, "bugfix": 0}
    for r in rows:
        if made["implement"] >= n_impl and made["bugfix"] >= n_fix:
            break
        entry, mod = r["entry_point"], rng.choice(MODULES)
        hidden = f"from {mod} import *\n{r['test']}\ncheck({entry})\n"
        ref_code = r["prompt"] + r["canonical_solution"]
        if not run_hidden({mod + ".py": ref_code}, hidden):
            continue
        kind = "implement" if made["implement"] < n_impl and (made["bugfix"] >= n_fix or rng.random() < 0.5) else "bugfix"
        n_vis = 1 if kind == "implement" else 2
        vis = visible_asserts(r["test"], entry, n_vis)
        if not vis:
            continue
        visible = f"from {mod} import *\n" + "\n".join(vis) + "\nprint('ok')\n"
        if kind == "implement":
            start = {mod + ".py": r["prompt"].rstrip() + "\n    raise NotImplementedError\n"}
            prompt = (f"Implement `{entry}` in {mod}.py as described in its docstring. "
                      f"Run `python test_{mod}.py` to check, and make sure your solution is general.")
        else:
            start = None
            body = r["canonical_solution"]
            for a, b in rng.sample(MUTATIONS, len(MUTATIONS)):
                if a in body:
                    cand = {mod + ".py": r["prompt"] + body.replace(a, b, 1)}
                    try:
                        ast.parse(cand[mod + ".py"])
                    except SyntaxError:
                        continue
                    # must fail the hidden test AND the visible one (the agent has to see it)
                    if run_hidden(cand, hidden) is False and run_hidden(cand, visible) is False:
                        start = cand
                        break
            if start is None:
                continue
            prompt = (f"The tests in test_{mod}.py fail. Find and fix the bug in {mod}.py "
                      f"(`{entry}`). Run `python test_{mod}.py` to check.")
        files = {**start, f"test_{mod}.py": visible}
        if run_hidden(files, hidden) is not False or run_hidden(files, visible) is not False:
            continue
        if not run_hidden({mod + ".py": ref_code, f"test_{mod}.py": visible}, visible):
            continue  # visible test must pass with the reference
        n = r["task_id"].split("/")[1]
        made[kind] += 1
        write_task(os.path.join(ROOT, f"{kind}_he{n}"), files, hidden,
                   dict(prompt=prompt, project=f"he{n}", kind=kind))
    print(made)


if __name__ == "__main__":
    main()
