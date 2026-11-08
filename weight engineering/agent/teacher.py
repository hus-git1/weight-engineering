"""Teacher trajectories: Claude solves the training tasks inside OUR harness (same tools, same
system prompt, same memory directory), and each run is converted into the small model's
tool-call format. Only runs whose hidden tests pass are kept for fine-tuning.

  python agent/teacher.py --dry-run --limit 4         # no API: replays reference solutions
  python agent/teacher.py --budget-usd 25             # real run (needs ANTHROPIC_API_KEY or `ant auth login`)

Resumable: finished runs are saved in agent/teacher_runs/ and skipped next time; the budget
counts the cost already spent by earlier runs.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import SYSTEM, TOOLS, Workspace
from memory_store import Memory

TASKS = os.path.join(HERE, "train_tasks")
RUNS = os.path.join(HERE, "teacher_runs")
MODEL = "claude-opus-5-5"
# USD per million tokens for claude-opus-5-5 (input, output, cache write 5m, cache read)
PRICE = dict(input=4.0, output=20.0, cache_write=5.0, cache_read=0.20)
BETAS = ["server-side-fallback-2026-07-01", "thinking-display-updates-2026-08-18"]

TEACHER_STYLE = """

How to work in this session (you are demonstrating ideal behaviour that a much smaller model will learn from):
- Start by calling memory_search with keywords from the task and project name; read a note only if it looks relevant.
- Call exactly one tool per turn, and keep your reasoning short and concrete.
- Read the relevant files before editing. After editing, run the tests. If they fail, read the error and fix your own code; the tests are correct.
- Before finishing, if you learned a durable fact about this project (a convention, a gotcha, how to run tests), save it with memory_write in a short note under projects/<project>/. Do not save trivia or anything specific to this one task.
- Call finish only after the tests pass."""

NUDGE = ("You did not call a tool. If the task is not done, continue by calling a tool (e.g. write_file, "
         "then run the tests). If it is done and verified, call finish.")


def anthropic_tools():
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in TOOLS]


def to_xml(name, args):
    params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                     for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def seed_memory(mem_dir, rules, project, rng):
    """Distractor notes from other projects, so the agent learns to search and pick the right one."""
    mem = Memory(mem_dir)
    for r in rng.sample([r for r in rules if r["project"] != project], 3):
        mem.write(f"projects/{r['project']}/conventions.md",
                  f"# {r['project']} conventions\n- {r['rule']}\n", f"{r['project']}: {r['rule']}")
    mem.write("skills/python-tests.md",
              "# Running tests\n- Test files are plain scripts: `python test_x.py`. Exit code 0 means pass.\n",
              "how to run the plain-script tests used in these projects")
    return mem


def grade(ws):
    for d in glob.glob(os.path.join(ws, "**", "__pycache__"), recursive=True):
        shutil.rmtree(d, ignore_errors=True)  # never grade against stale bytecode
    r = subprocess.run([sys.executable, "-E", "-s", "_hidden_test.py"], cwd=ws, capture_output=True, timeout=30)
    return r.returncode == 0


class Teacher:
    def __init__(self, budget, dry_run):
        self.budget, self.dry_run = budget, dry_run
        self.spent = sum(json.load(open(p)).get("cost_usd", 0) for p in glob.glob(f"{RUNS}/*.json"))
        self.lock = threading.Lock()
        if not dry_run:
            import anthropic
            self.client = anthropic.Anthropic()
            self.anthropic = anthropic

    def cost(self, usage):
        u = lambda k: getattr(usage, k, 0) or 0
        return (u("input_tokens") * PRICE["input"] + u("output_tokens") * PRICE["output"]
                + u("cache_creation_input_tokens") * PRICE["cache_write"]
                + u("cache_read_input_tokens") * PRICE["cache_read"]) / 1e6

    def over_budget(self):
        with self.lock:
            return self.spent >= self.budget

    # ------------------------------------------------------------------ one task
    def run(self, task_dir, mem_dir, max_turns=25):
        meta = json.load(open(os.path.join(task_dir, "task.json")))
        tmp = tempfile.mkdtemp(prefix="teach_")
        ws_path = os.path.join(tmp, "ws")
        shutil.copytree(task_dir, ws_path, ignore=shutil.ignore_patterns("_hidden_test.py", "task.json"))
        ws, mem = Workspace(ws_path), Memory(mem_dir)
        handlers = {"list_files": ws.list_files, "read_file": ws.read_file, "write_file": ws.write_file,
                    "run": ws.run, "memory_search": mem.search, "memory_read": mem.read, "memory_write": mem.write}
        system = SYSTEM.format(project=meta["project"], index=mem.index())
        qwen = [{"role": "system", "content": system}, {"role": "user", "content": meta["prompt"]}]
        steps = (self.scripted(meta, task_dir) if self.dry_run
                 else self.claude_steps(system + TEACHER_STYLE, meta["prompt"], max_turns))
        status, cost, result = "max_turns", 0.0, None
        try:
            while True:
                step = steps.send(result) if result is not None else next(steps)
                result = None
                cost += step.get("cost", 0.0)
                if step["kind"] == "stop":
                    status = step["status"]
                    break
                if step["kind"] == "text":  # no tool call: harness nudge, same as at inference
                    qwen += [{"role": "assistant", "content": step["text"]}, {"role": "user", "content": NUDGE}]
                    result = NUDGE
                    continue
                name, args = step["name"], step["args"]
                body = (step["rationale"].strip() + "\n\n" if step.get("rationale", "").strip() else "") + to_xml(name, args)
                qwen.append({"role": "assistant", "content": body, **({"no_train": True} if step.get("no_train") else {})})
                if name == "finish":
                    status = "finished"
                    break
                try:
                    out = handlers[name](**args) if name in handlers else f"Unknown tool {name}."
                except Exception as e:
                    out = f"Error: {type(e).__name__}: {e}"
                out = str(out)[:6000]
                qwen.append({"role": "tool", "content": out})
                result = out
        except StopIteration:
            pass
        shutil.copy(os.path.join(task_dir, "_hidden_test.py"), ws_path)
        passed = grade(ws_path)
        shutil.rmtree(tmp, ignore_errors=True)
        with self.lock:
            self.spent += cost
        return dict(task=os.path.relpath(task_dir, TASKS), project=meta["project"], kind=meta.get("kind"),
                    status=status, passed=passed, cost_usd=round(cost, 4), messages=qwen,
                    memory_writes=sum("<function=memory_write>" in m["content"] for m in qwen if m["role"] == "assistant"),
                    memory_reads=sum(("<function=memory_search>" in m["content"]) or ("<function=memory_read>" in m["content"])
                                     for m in qwen if m["role"] == "assistant"))

    # ------------------------------------------------------------------ real teacher
    def claude_steps(self, system, prompt, max_turns):
        """Generator yielding tool calls; receives each tool result via send()."""
        a = self.anthropic
        msgs = [{"role": "user", "content": prompt}]
        nudged, carry = 0, 0.0
        for _ in range(max_turns):
            if self.over_budget():
                yield dict(kind="stop", status="budget")
                return
            try:
                resp = self.client.beta.messages.create(
                    model=MODEL, max_tokens=16000, system=system, tools=anthropic_tools(),
                    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                    thinking={"type": "adaptive", "display": "updates"},
                    output_config={"effort": "medium"},
                    cache_control={"type": "ephemeral"},
                    betas=BETAS, fallbacks="default", messages=msgs)
            except a.RateLimitError:
                import time
                time.sleep(30)
                continue
            except a.APIStatusError as e:
                yield dict(kind="stop", status=f"api_error_{e.status_code}")
                return
            cost = self.cost(resp.usage) + carry
            carry = 0.0
            if resp.stop_reason in ("refusal", "max_tokens"):
                yield dict(kind="stop", status=resp.stop_reason, cost=cost)
                return
            msgs.append({"role": "assistant", "content": resp.content})  # append-only, unchanged
            if resp.stop_reason == "pause_turn":
                carry = cost  # resend to continue; bill this call with the next step
                continue
            notes = [getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking"]
            texts = [b.text for b in resp.content if b.type == "text"]
            uses = [b for b in resp.content if b.type == "tool_use"]
            rationale = " ".join(t.strip() for t in notes + texts if t and t.strip())
            if not uses:
                if nudged >= 2:
                    yield dict(kind="stop", status="no_tool_call", cost=cost)
                    return
                nudged += 1
                result = yield dict(kind="text", text=rationale or "(no reply)", cost=cost)
                msgs.append({"role": "user", "content": NUDGE})
                continue
            tu = uses[0]
            result = yield dict(kind="tool", name=tu.name, args=dict(tu.input), rationale=rationale, cost=cost)
            if tu.name == "finish":
                return
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id,
                                                      "content": result or ""}]})

    # ------------------------------------------------------------------ dry run (no API)
    def scripted(self, meta, task_dir):
        """Replays a plausible ideal trajectory using the reference solution (pipeline test only)."""
        ref = meta.get("reference")
        if ref is None:  # memory tasks: reconstruct from gen_train_tasks families
            import gen_train_tasks as G
            fam = next(f for f in G.FAMILIES if f.__name__[4:] == meta["pair"].split("_")[0])
            ref = fam(meta["project"])[meta["kind"][-2:]][3]
        yield dict(kind="tool", name="memory_search", args={"query": meta["project"]}, rationale="Check memory for this project.")
        yield dict(kind="tool", name="list_files", args={}, rationale="See what files exist.")
        for fn, body in ref.items():
            if fn != "README.md":
                yield dict(kind="tool", name="write_file", args={"path": fn, "content": body}, rationale=f"Write the fix to {fn}.")
        tests = [f for f in os.listdir(task_dir) if f.startswith("test_")]
        if tests:
            yield dict(kind="tool", name="run", args={"command": f"python {tests[0]}"}, rationale="Run the tests.")
        if meta.get("rule"):
            yield dict(kind="tool", name="memory_write", rationale="Save the project convention.",
                       args={"path": f"projects/{meta['project']}/conventions.md",
                             "content": f"# {meta['project']}\n- {meta['rule']}", "summary": f"{meta['project']}: {meta['rule']}"})
        yield dict(kind="tool", name="finish", args={"summary": "Fixed and verified."}, rationale="Tests pass.")


def units(limit):
    """Independent work units: a memory pair (s1 then s2 sharing memory) or a single task."""
    pairs = sorted(glob.glob(f"{TASKS}/memory/*/"))
    singles = sorted(glob.glob(f"{TASKS}/single/*/"))
    rng = random.Random(0)
    rng.shuffle(singles)
    out = [("pair", p) for p in pairs]
    out += [("single", s) for s in singles]
    rng.shuffle(out)
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--limit", type=int, help="max work units (pairs count as one)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    global RUNS
    if a.dry_run:
        RUNS = os.path.join(HERE, "teacher_runs_dryrun")
    os.makedirs(RUNS, exist_ok=True)
    rules = json.load(open(f"{TASKS}/memory/rules.json"))
    T = Teacher(a.budget_usd, a.dry_run)
    print(f"already spent ${T.spent:.2f} of ${a.budget_usd:.2f}", flush=True)

    def save(rec):
        name = rec["task"].replace("/", "__").strip("_")
        json.dump(rec, open(os.path.join(RUNS, name + ".json"), "w"), indent=1)
        print(f"  {rec['task']:<45} passed={rec['passed']!s:<5} {rec['status']:<12} "
              f"mem r/w={rec['memory_reads']}/{rec['memory_writes']} ${rec['cost_usd']:.3f} (total ${T.spent:.2f})", flush=True)

    def done(task_dir):
        name = os.path.relpath(task_dir, TASKS).replace("/", "__").strip("_")
        return os.path.exists(os.path.join(RUNS, name + ".json"))

    def work(unit):
        kind, path = unit
        rng = random.Random(path)
        if kind == "pair":
            s1, s2 = os.path.join(path, "s1"), os.path.join(path, "s2")
            if done(s1) and done(s2):
                return
            mem_dir = tempfile.mkdtemp(prefix="teachmem_")
            seed_memory(mem_dir, rules, json.load(open(os.path.join(s1, "task.json")))["project"], rng)
            for s in (s1, s2):
                if T.over_budget():
                    return
                save(T.run(s, mem_dir))
        else:
            if done(path) or T.over_budget():
                return
            mem_dir = tempfile.mkdtemp(prefix="teachmem_")
            seed_memory(mem_dir, rules, json.load(open(os.path.join(path, "task.json")))["project"], rng)
            save(T.run(path, mem_dir))

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, units(a.limit)))
    recs = [json.load(open(p)) for p in glob.glob(f"{RUNS}/*.json")]
    print(f"\n{len(recs)} runs, {sum(r['passed'] for r in recs)} passed, total ${sum(r['cost_usd'] for r in recs):.2f}")


if __name__ == "__main__":
    main()
