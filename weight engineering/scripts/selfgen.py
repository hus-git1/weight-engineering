"""Self-taught training data (rejection sampling): the model attempts training problems
several times; we keep, per problem, the SHORTEST verified-correct attempt.
Keeping the shortest correct answer directly targets the model's habit of rambling.

  python scripts/selfgen.py --task code --k 4                 # MBPP (all 974), tests executed
  python scripts/selfgen.py --task math --n 4000 --k 2        # MATH train split (not MATH-500)
  python scripts/selfgen.py --task code_he --k 4              # MBPP as HumanEval-style stubs (v2)

Resumable: processed problem ids are logged, so re-running continues where it stopped.
Output: data/self/<task>.jsonl with {"id", "source", "messages", "n_tokens", "n_correct"}.
"""
import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_dataset_config_names, load_dataset

from wsp import mlx_backend as B
from wsp.evals.code_eval import PROMPT as HE_PROMPT
from wsp.evals.code_eval import extract_code
from wsp.evals.math_eval import PROMPT as MATH_PROMPT
from wsp.evals.math_eval import score as math_score

SAMPLING = dict(temperature=0.7, top_p=0.9, top_k=20, presence_penalty=0.0)
OUT = "data/self"


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    depth = 0
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1:k]
    return None


def math_problems():
    rows = []
    for c in get_dataset_config_names("EleutherAI/hendrycks_math"):
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", c, split="train")):
            ans = last_boxed(r["solution"])
            if ans:
                rows.append(dict(id=f"math/{c}/{i}", answer=ans, level=r["level"],
                                 messages=[{"role": "user", "content": MATH_PROMPT.format(problem=r["problem"])}]))
    return rows


CODE_PROMPT = ("{text}\nYour function must satisfy this example:\n```python\n{example}\n```\n"
               "Reply with the complete solution in a single ```python code block.")


def code_problems():
    rows = []
    for split, ds in load_dataset("google-research-datasets/mbpp", "full").items():
        for r in ds:
            rows.append(dict(id=f"mbpp/{r['task_id']}", tests=r["test_list"] + r["challenge_test_list"],
                             setup=r["test_setup_code"], ref=r["code"],
                             messages=[{"role": "user", "content": CODE_PROMPT.format(
                                 text=r["text"], example=r["test_list"][0])}]))
    return rows


def doctest_example(test):
    """`assert f(x) == y` -> '>>> f(x)\\n    y' (None for other assert shapes)."""
    try:
        node = ast.parse(test.strip()).body[0]
    except SyntaxError:
        return None
    t = getattr(node, "test", None)
    if isinstance(node, ast.Assert) and isinstance(t, ast.Compare) and len(t.ops) == 1 \
            and isinstance(t.ops[0], ast.Eq) and isinstance(t.left, ast.Call):
        return f">>> {ast.unparse(t.left)}\n    {ast.unparse(t.comparators[0])}"
    return None


def code_he_problems():
    """v2 code data: MBPP problems in HumanEval's format (stub = signature + docstring with the
    description and a >>> example, same prompt as the HumanEval eval). v1's MBPP prompt format
    taught the model to drop docstrings and append its own asserts."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
    from gen_train_tasks import main_function
    rows = []
    for split, ds in load_dataset("google-research-datasets/mbpp", "full").items():
        for r in ds:
            tests = r["test_list"] + r["challenge_test_list"]
            try:
                fn = main_function(r["code"], tests)
            except SyntaxError:
                continue
            ex = doctest_example(r["test_list"][0])
            if fn is None or ex is None:
                continue
            stub = f'def {fn.name}({ast.unparse(fn.args)}):\n    """ {r["text"].strip()}\n    {ex}\n    """\n'
            rows.append(dict(id=f"mbpp/{r['task_id']}", tests=tests, setup=r["test_setup_code"],
                             ref=r["code"], stub=stub,
                             messages=[{"role": "user", "content": HE_PROMPT.format(prompt=stub)}]))
    return rows


def keeps_stub(p, text):
    """code_he answers must echo the stub (signature + docstring) and add no test code."""
    code = extract_code(text)
    head = p["stub"].splitlines()
    top_tests = any(l.startswith(("assert ", "print(")) for l in code.splitlines())
    return head[0] in code and head[1].strip() in code and not top_tests


def code_score(task, text, timeout=10):
    program = f"{task['setup']}\n{extract_code(text)}\n\n" + "\n".join(task["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        open(path, "w").write(program)
        try:
            return subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                                  timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade_all(task, scorer, pool, pairs):
    """Grade (problem, output) pairs. math-verify uses signal.alarm for its timeout, which only
    works on the main thread (in a worker thread every check silently fails) -> math is graded
    here; code runs subprocesses, so it is graded in the thread pool."""
    check = lambda po: scorer(po[0], po[1]["text"]) and po[1]["finished"]
    return [check(po) for po in pairs] if task == "math" else list(pool.map(check, pairs))


def self_test(task, scorer, pool, p):
    """The grader must accept a known-correct answer through the exact same code path."""
    gold = (f"The answer is \\boxed{{{p['answer']}}}." if task == "math"
            else f"```python\n{p['ref']}\n```")
    wrong = "The answer is \\boxed{nope}." if task == "math" else "```python\npass\n```"
    got = grade_all(task, scorer, pool, [(p, {"text": gold, "finished": True}),
                                         (p, {"text": wrong, "finished": True})])
    if got != [True, False]:
        sys.exit(f"grader self-test FAILED on {p['id']}: expected [True, False], got {got}")
    print("grader self-test passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["math", "code", "code_he"], required=True)
    ap.add_argument("--n", type=int, help="max problems (default: all)")
    ap.add_argument("--k", type=int, default=4, help="attempts per problem")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--chunk", type=int, default=8, help="problems per generation round (x k sequences in parallel)")
    ap.add_argument("--max-waits", type=int, default=2, help="reject attempts that second-guess more")
    a = ap.parse_args()
    max_tokens = a.max_new_tokens or (1024 if a.task == "math" else 768)

    problems = {"math": math_problems, "code": code_problems, "code_he": code_he_problems}[a.task]()
    random.Random(0).shuffle(problems)  # mix subjects/levels so partial runs are representative
    problems = problems[:a.n] if a.n else problems
    os.makedirs(OUT, exist_ok=True)
    out_path, done_path = f"{OUT}/{a.task}.jsonl", f"{OUT}/{a.task}.done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    todo = [p for p in problems if p["id"] not in done]
    print(f"{len(problems)} problems, {len(done)} already done, {len(todo)} to go; k={a.k}", flush=True)

    scorer = math_score if a.task == "math" else code_score
    pool = ThreadPoolExecutor(8)
    self_test(a.task, scorer, pool, problems[0])
    model, tok = B.load()
    kept = solved_any = 0
    t0 = time.time()
    for c in range(0, len(todo), a.chunk):
        chunk = todo[c:c + a.chunk]
        convs = [p["messages"] for p in chunk for _ in range(a.k)]
        outs = B.generate(model, tok, convs, thinking=False, max_new_tokens=max_tokens,
                          batch_size=len(convs), sampling=SAMPLING, seed=len(done) + c)
        oks = grade_all(a.task, scorer, pool,
                        [(p, outs[i * a.k + j]) for i, p in enumerate(chunk) for j in range(a.k)])
        with open(out_path, "a") as f:
            for i, p in enumerate(chunk):
                atts = [(outs[i * a.k + j], oks[i * a.k + j]) for j in range(a.k)]
                good = [o for o, ok in atts if ok and o["text"].count("Wait") <= a.max_waits
                        and (a.task != "code_he" or keeps_stub(p, o["text"]))]
                solved_any += any(ok for _, ok in atts)
                if good:
                    best = min(good, key=lambda o: o["n_tokens"])
                    f.write(json.dumps(dict(id=p["id"], source=a.task, n_tokens=best["n_tokens"],
                                            n_correct=sum(ok for _, ok in atts),
                                            messages=p["messages"] + [{"role": "assistant", "content": best["text"]}])) + "\n")
                    kept += 1
        with open(done_path, "a") as f:
            f.write("\n".join(p["id"] for p in chunk) + "\n")
        n = c + len(chunk)
        rate = (time.time() - t0) / n
        print(f"  {n}/{len(todo)}: solved {solved_any/n:.0%} (any of {a.k}), kept {kept} "
              f"| {rate:.1f}s/problem, ~{rate*(len(todo)-n)/60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
