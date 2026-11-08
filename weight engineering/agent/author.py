"""Authored teacher: trajectories written by Claude (in the Claude Code session) as a policy,
executed for real in the harness. Every tool output in the data is real (files are really
read, tests really run); the reasoning notes, memory notes, quick tests and recovery steps
are authored here. Only runs whose hidden tests pass are kept.

  python agent/author.py [--limit N] [--recovery 0.3]

Writes agent/teacher_runs_authored/*.json in the same format as agent/teacher.py.
"""
import argparse
import ast
import difflib
import glob
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_train_tasks as G
import teacher as T

MUTATIONS = G.MUTATIONS

# ------------------------------------------------------------------ per-family authored content
FAM = {
    "money": dict(
        s1=("The README says money is always integer cents, but order_total divides by 100 and returns dollars. "
            "I'll remove the division so it returns cents."),
        note="# {P} conventions\n- Money is integer cents everywhere (e.g. $12.50 -> 1250). Never use floats or divide by 100.\n- Tests are plain scripts: `python test_pricing.py`.",
        summary="{P}: money is integer cents, never floats",
        s2="My note for {P} says money is integer cents, never floats. So apply_tax must take cents and return a rounded int in cents.",
        quick=("test_tax.py", "from pricing import apply_tax\nr = apply_tax(500, 10)\nassert r == 550 and isinstance(r, int), r\nassert apply_tax(1, 50) == 2\nprint('ok')\n")),
    "errors": dict(
        s1="The README requires invalid input to raise {E} from errors.py, but check_port raises ValueError. I'll import {E} and raise it instead.",
        note="# {P} conventions\n- Invalid input raises `{E}` (defined in errors.py), never ValueError or TypeError.\n- Validators return the value when it is valid.",
        summary="{P}: invalid input raises {E} from errors.py",
        s2="My note for {P} says invalid input must raise {E} from errors.py. So check_username returns the name when valid and raises {E} otherwise.",
        quick=("test_username.py", "from validate import check_username\nfrom errors import {E}\nassert check_username('alice_01') == 'alice_01'\ntry:\n    check_username('no spaces')\n    raise SystemExit('accepted a bad name')\nexcept {E}:\n    print('ok')\n")),
    "dates": dict(
        s1="The README says dates are DD/MM/YYYY strings, but parse() uses '%Y-%m-%d'. I'll change the format to '%d/%m/%Y'.",
        note="# {P} conventions\n- Dates are strings in DD/MM/YYYY format (e.g. 05/11/2026). Parse and format with '%d/%m/%Y'.",
        summary="{P}: dates are DD/MM/YYYY strings",
        s2="My note for {P} says dates are DD/MM/YYYY strings. So add_days parses with '%d/%m/%Y', adds the days, and formats back the same way.",
        quick=("test_add_days.py", "from dates import add_days\nassert add_days('28/02/2026', 1) == '01/03/2026'\nassert add_days('05/11/2026', 0) == '05/11/2026'\nprint('ok')\n")),
    "ids": dict(
        s1="The README says IDs are {X}- plus the number zero-padded to 6 digits, but make_id doesn't pad. I'll format the number with :06d.",
        note="# {P} conventions\n- Record IDs are `{X}-NNNNNN`: prefix {X}, a dash, number zero-padded to 6 digits (e.g. {X}-000042).",
        summary="{P}: record IDs are {X}-NNNNNN (6-digit zero-padded)",
        s2="My note for {P} says IDs look like {X}-000042. So next_id takes the highest number, adds one, and formats it as {X}- plus 6 digits.",
        quick=("test_next_id.py", "from records import next_id\nassert next_id(['{X}-000009']) == '{X}-000010'\nassert next_id([]) == '{X}-000001'\nprint('ok')\n")),
    "seconds": dict(
        s1="The README says all durations are integer seconds, so minutes(n) must return n * 60, not n.",
        note="# {P} conventions\n- All durations are integers in seconds (minutes * 60, hours * 3600).",
        summary="{P}: durations are integer seconds",
        s2="My note for {P} says durations are integer seconds. So parse_duration converts hours and minutes to seconds.",
        quick=("test_parse_duration.py", "from timeparse import parse_duration\nassert parse_duration('3h') == 10800\nassert parse_duration('10m') == 600\nassert parse_duration('1h1m') == 3660\nprint('ok')\n")),
    "registry": dict(
        s1="The README says every command must be registered in COMMANDS, but cmd_echo is never registered. I'll add COMMANDS['echo'] = cmd_echo.",
        note="# {P} conventions\n- Commands are functions `cmd_<name>(args) -> str` in commands.py.\n- Every command must be registered: `COMMANDS['<name>'] = cmd_<name>` (COMMANDS lives in registry.py).",
        summary="{P}: commands are cmd_<name>(args) registered in COMMANDS",
        s2="My note for {P} says commands are cmd_<name>(args) functions registered in COMMANDS. So I'll add cmd_shout and register it.",
        quick=("test_shout.py", "import commands\nfrom registry import COMMANDS\nassert COMMANDS['shout'](['a', 'b']) == 'A B'\nprint('ok')\n")),
    "percent": dict(
        s1="The README says percentages are 0-100 rounded to 1 decimal, but pct returns a fraction. I'll multiply by 100 and round to 1 decimal.",
        note="# {P} conventions\n- Percentages are floats from 0 to 100 rounded to 1 decimal place (33.3, not 0.333).",
        summary="{P}: percentages are 0-100 floats rounded to 1 decimal",
        s2="My note for {P} says percentages are 0-100 rounded to 1 decimal. So growth returns round(100 * (new - old) / old, 1).",
        quick=("test_growth.py", "from stats import growth\nassert growth(100, 150) == 50.0\nassert growth(4, 3) == -25.0\nprint('ok')\n")),
}


def fmt(s, P):
    return s.format(P=P, E=P.capitalize() + "Error", X=P[:3].upper())


def run_py(files, test, timeout=10):
    """Pre-check a candidate in a scratch dir (used to pick realistic wrong first attempts)."""
    with tempfile.TemporaryDirectory() as d:
        for fn, body in files.items():
            open(os.path.join(d, fn), "w").write(body)
        try:
            return subprocess.run([sys.executable, "-E", "-s", test], cwd=d, capture_output=True,
                                  timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return None


def first_change(a, b):
    """(old_line, new_line) of the first differing line between two versions of a file."""
    al, bl = a.splitlines(), b.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, al, bl).get_opcodes():
        if tag != "equal":
            old = al[i1].strip() if i1 < i2 else ""
            new = bl[j1].strip() if j1 < j2 else ""
            return old, new
    return "", ""


def last_line(out):
    """The failing source line plus the exception, e.g. `assert f(3, 1) == 4` -> AssertionError."""
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if not lines:
        return out.strip()[:160]
    exc = lines[-1].strip()
    code = ""
    for i, l in enumerate(lines):
        if l.strip().startswith("File ") and i + 1 < len(lines) and not lines[i + 1].strip().startswith("File "):
            code = lines[i + 1].strip()
    return (f"{code}` -> `{exc}" if code and code != exc else exc)[:200]


def wrong_attempt(ref_body, files, mod_file, test_file, rng):
    """A plausible buggy version of the reference that FAILS the visible test (for recovery demos)."""
    for a, b in rng.sample(MUTATIONS, len(MUTATIONS)):
        if a in ref_body:
            cand = ref_body.replace(a, b, 1)
            if cand != ref_body and run_py({**files, mod_file: cand}, test_file) is False:
                return cand
    return None


def implement_solution(stub, ref, fn):
    """v2: keep the stub (signature + docstring) and replace `raise NotImplementedError` with the
    reference body, normalized by ast.unparse; the reference's other top-level code (imports,
    helpers) goes first. (v1 overwrote the stub with raw MBPP code, and the trained model learned
    to drop docstrings -- which cost it ~10 points on single-shot HumanEval.)"""
    i = stub.index(f"def {fn}(")
    setup, stub_fn = stub[:i], stub[i:]
    tree = ast.parse(ref[len(setup):] if ref.startswith(setup) else ref)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    body = main.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # the reference's own docstring: the stub's is kept
    body_src = textwrap.indent("\n".join(ast.unparse(st) for st in body), "    ")
    others = "\n".join(ast.unparse(n) for n in tree.body if n is not main)
    head = setup + (others + "\n\n\n" if others else "")
    return head + stub_fn.replace("    raise NotImplementedError\n", body_src + "\n")


def failing_call(out):
    """From the failing `assert f(x) == y` in a traceback -> ("f(x)", "y"), else None."""
    code = last_line(out).split("` -> `")[0]
    try:
        node = ast.parse(code.strip()).body[0]
    except Exception:
        return None
    t = getattr(node, "test", None)
    if isinstance(node, ast.Assert) and isinstance(t, ast.Compare) and len(t.ops) == 1 \
            and isinstance(t.ops[0], ast.Eq) and isinstance(t.left, ast.Call):
        return ast.unparse(t.left), ast.unparse(t.comparators[0])
    return None


def probe(mod_file, out):
    """v2 grounded debugging: run the failing call and look at what it really returns.
    Yields the probe step; returns a sentence comparing actual with expected (or None)."""
    fc = failing_call(out)
    if not fc:
        return None
    call, expected = fc
    code = f"from {mod_file[:-3]} import *; print(repr({call}))"
    res = yield dict(kind="tool", name="run", args={"command": f"python -c {shlex.quote(code)}"},
                     rationale=f"Let me see what `{call}` actually returns.")
    head, _, body = res.partition("\n")
    if head.strip() == "exit code 0":
        actual = body.strip()
        actual = actual if len(actual) <= 120 else actual[:117] + "..."
        return f"`{call}` returns `{actual}`, but the test expects `{expected}`."
    return f"`{call}` raises `{last_line(body)}`; the test expects `{expected}`."


# ------------------------------------------------------------------ policies
def single_policy(meta, task_dir, rng, recovery):
    mod_file, ref_body = next(iter(meta["reference"].items()))
    test_file = next(f for f in os.listdir(task_dir) if f.startswith("test_"))
    start = {f: open(os.path.join(task_dir, f)).read() for f in os.listdir(task_dir)
             if f.endswith(".py") and not f.startswith("_")}
    fn = meta["prompt"].split("`")[1]
    yield dict(kind="tool", name="memory_search", args={"query": f"{fn} {meta['project']}"},
               rationale="First I'll check memory for anything relevant to this task.")
    yield dict(kind="tool", name="list_files", args={}, rationale="Nothing relevant in memory. Let me see the files.")
    yield dict(kind="tool", name="read_file", args={"path": test_file}, rationale=f"I'll read the test to see what `{fn}` must return.")
    yield dict(kind="tool", name="read_file", args={"path": mod_file}, rationale=f"Now the current code in {mod_file}.")
    if meta["kind"] == "bugfix":
        out = yield dict(kind="tool", name="run", args={"command": f"python {test_file}"},
                         rationale="Let me run the test to see how it fails.")
        seen = yield from probe(mod_file, out)
        old, new = first_change(start[mod_file], ref_body)
        why = seen or f"The test fails: `{last_line(out)}`."
        reason = (f"{why} In `{fn}`, `{old}` is wrong; it should be `{new}`."
                  if old and new else f"{why} I'll fix the logic in `{fn}`.")
        yield dict(kind="tool", name="write_file", args={"path": mod_file, "content": ref_body}, rationale=reason)
    else:
        text = meta["prompt"].split(":", 1)[1].split("Run `")[0].strip()
        hidden = {**start, "_hidden_test.py": open(os.path.join(task_dir, "_hidden_test.py")).read()}
        try:
            sol = implement_solution(start[mod_file], ref_body, fn)
            if run_py({**hidden, mod_file: sol}, "_hidden_test.py") is not True:
                sol = ref_body
        except Exception:
            sol = ref_body
        wrong = wrong_attempt(sol, start, mod_file, test_file, rng) if rng.random() < recovery else None
        if wrong:
            yield dict(kind="tool", name="write_file", args={"path": mod_file, "content": wrong},
                       rationale=f"I'll implement `{fn}`: {text}", no_train=True)
            out = yield dict(kind="tool", name="run", args={"command": f"python {test_file}"}, rationale="Now run the test.")
            seen = yield from probe(mod_file, out)
            old, new = first_change(wrong, sol)
            why = seen or f"The test fails: `{last_line(out)}`."
            yield dict(kind="tool", name="write_file", args={"path": mod_file, "content": sol},
                       rationale=f"{why} My code is wrong, not the test: `{old}` should be `{new}`.")
        else:
            yield dict(kind="tool", name="write_file", args={"path": mod_file, "content": sol},
                       rationale=f"I'll implement `{fn}`: {text}")
    out = yield dict(kind="tool", name="run", args={"command": f"python {test_file}"}, rationale="Run the test again to verify.")
    if "exit code 0" not in out:
        return  # never demonstrate finishing on a failing test
    yield dict(kind="tool", name="finish", rationale="The test passes. Nothing here is worth saving to memory, so I'm done.",
               args={"summary": f"{'Fixed' if meta['kind'] == 'bugfix' else 'Implemented'} `{fn}` in {mod_file}; `python {test_file}` passes."})


def memory_policy(meta, task_dir, rng, recovery):
    fam_name = meta["pair"].split("_")[0]
    P, F = meta["project"], FAM[fam_name]
    fam = next(f for f in G.FAMILIES if f.__name__[4:] == fam_name)
    spec = fam(P)
    session = meta["kind"][-2:]
    files, _, _, ref = spec[session]
    code_files = [f for f in ref if f != "README.md"]
    if session == "s1":
        test_file = next(f for f in files if f.startswith("test_"))
        yield dict(kind="tool", name="memory_search", args={"query": f"{P} conventions"},
                   rationale=f"I'll check memory for notes about {P} first.")
        yield dict(kind="tool", name="list_files", args={}, rationale=f"No notes about {P} yet. Let me look at the files.")
        yield dict(kind="tool", name="read_file", args={"path": "README.md"}, rationale="The task points to the README, so I'll read it first.")
        for f in code_files:
            if f in files and not f.startswith("test_"):
                yield dict(kind="tool", name="read_file", args={"path": f}, rationale=f"Now {f}.")
        out = yield dict(kind="tool", name="run", args={"command": f"python {test_file}"}, rationale="Let me run the failing test.")
        changed = [f for f in code_files if ref[f] != files.get(f)]
        for i, f in enumerate(changed):
            yield dict(kind="tool", name="write_file", args={"path": f, "content": ref[f]},
                       rationale=(f"The test fails: `{last_line(out)}`. " + fmt(F["s1"], P)) if i == 0 else f"Also update {f}.")
        out = yield dict(kind="tool", name="run", args={"command": f"python {test_file}"}, rationale="Run the test again.")
        if "exit code 0" not in out:
            return
        yield dict(kind="tool", name="memory_write",
                   args={"path": f"projects/{P}/conventions.md", "content": fmt(F["note"], P), "summary": fmt(F["summary"], P)},
                   rationale="The test passes. This convention will matter in future sessions, so I'll save it to memory.")
        yield dict(kind="tool", name="finish", rationale="Saved. Done.",
                   args={"summary": f"Fixed the convention bug in {', '.join(changed)}; test passes; saved the {P} convention to memory."})
    else:
        target = code_files[0] if len(code_files) == 1 else next(f for f in code_files if ref[f] != files.get(f))
        qname, qbody = F["quick"]
        res = yield dict(kind="tool", name="memory_search", args={"query": f"{P} conventions"},
                         rationale=f"The task says to follow {P}'s conventions. I'll check memory first.")
        note = f"projects/{P}/conventions.md"
        if note not in res:
            return  # session 1 did not leave a note: nothing honest to demonstrate
        yield dict(kind="tool", name="memory_read", args={"path": note}, rationale=f"There's a note for {P}. Let me read it.")
        yield dict(kind="tool", name="list_files", args={}, rationale="Now the files in this workspace.")
        yield dict(kind="tool", name="read_file", args={"path": target}, rationale=f"I'll read {target} before editing it.")
        wrong = None
        if rng.random() < recovery:
            wrong = wrong_attempt(ref[target], {**files, qname: fmt(qbody, P)}, target, qname, rng)
        yield dict(kind="tool", name="write_file", args={"path": target, "content": wrong or ref[target]},
                   rationale=fmt(F["s2"], P), no_train=bool(wrong))
        yield dict(kind="tool", name="write_file", args={"path": qname, "content": fmt(qbody, P)},
                   rationale="Now a quick test that checks the convention.")
        out = yield dict(kind="tool", name="run", args={"command": f"python {qname}"}, rationale="Run it.")
        if wrong:
            old, new = first_change(wrong, ref[target])
            yield dict(kind="tool", name="write_file", args={"path": target, "content": ref[target]},
                       rationale=f"It fails: `{last_line(out)}`. The bug is mine: `{old}` should be `{new}`.")
            out = yield dict(kind="tool", name="run", args={"command": f"python {qname}"}, rationale="Run the test again.")
        if "exit code 0" not in out:
            return
        yield dict(kind="tool", name="finish", rationale="The test passes and the code follows the project's convention.",
                   args={"summary": f"Added the new function to {target} following the {P} convention from memory; {qname} passes."})


class AuthorTeacher(T.Teacher):
    def __init__(self, recovery):
        super().__init__(budget=0, dry_run=True)
        self.recovery = recovery

    def over_budget(self):
        return False

    def scripted(self, meta, task_dir):
        rng = random.Random(task_dir)
        if meta["kind"].startswith("memory"):
            return memory_policy(meta, task_dir, rng, self.recovery)
        return single_policy(meta, task_dir, rng, self.recovery)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--recovery", type=float, default=0.3, help="share of runs that show recovering from a failed attempt")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default="teacher_runs_authored_v2")
    a = ap.parse_args()
    T.RUNS = os.path.join(HERE, a.out_dir)
    os.makedirs(T.RUNS, exist_ok=True)
    rules = json.load(open(f"{T.TASKS}/memory/rules.json"))
    teacher = AuthorTeacher(a.recovery)

    def save(rec):
        name = rec["task"].replace("/", "__").strip("_")
        json.dump(rec, open(os.path.join(T.RUNS, name + ".json"), "w"), indent=1)
        return rec

    def work(unit):
        kind, path = unit
        rng = random.Random(path)
        mem_dir = tempfile.mkdtemp(prefix="authmem_")
        sessions = [os.path.join(path, "s1"), os.path.join(path, "s2")] if kind == "pair" else [path]
        T.seed_memory(mem_dir, rules, json.load(open(os.path.join(sessions[0], "task.json")))["project"], rng)
        return [save(teacher.run(s, mem_dir)) for s in sessions]

    recs = [r for rs in ThreadPoolExecutor(a.workers).map(work, T.units(a.limit)) for r in rs]
    by = {}
    for r in recs:
        k = r["kind"]
        by.setdefault(k, [0, 0])
        by[k][0] += r["passed"]
        by[k][1] += 1
    recov = sum(any("My code is wrong" in m["content"] or "The bug is mine" in m["content"]
                    for m in r["messages"] if m["role"] == "assistant") for r in recs if r["passed"])
    print(json.dumps({k: f"{p}/{n} passed" for k, (p, n) in by.items()}, indent=1))
    print(f"recovery demonstrations among passing runs: {recov}")


if __name__ == "__main__":
    main()
