"""A minimal coding agent loop with file, shell, and long-term memory tools.

  python agent/harness.py --workspace agent/tasks/fizzbuzz --project fizzbuzz "Make the tests pass"

Every run is saved to agent/trajectories/ — successful ones become fine-tuning data later.
NOTE: `run` executes shell commands in the workspace with a timeout. It is NOT a security
sandbox; only point the agent at throwaway directories.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from memory_store import Memory
from wsp import model as M


def tool(name, desc, **params):
    req = [k for k, v in params.items() if not v.startswith("(optional)")]
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
        "required": req}}}


TOOLS = [
    tool("list_files", "List files in the workspace.", path="(optional) subdirectory, default '.'"),
    tool("read_file", "Read a workspace file.", path="file path relative to the workspace"),
    tool("write_file", "Create or overwrite a workspace file.", path="file path", content="full file content"),
    tool("run", "Run a shell command in the workspace (e.g. tests). 60s timeout.", command="shell command"),
    tool("memory_search", "Search long-term memory notes from past work.", query="keywords"),
    tool("memory_read", "Read one memory note.", path="note path from the index or search"),
    tool("memory_write", "Save a durable lesson or project fact for future sessions. Keep it short and specific.",
         path="e.g. projects/<project>/gotchas.md or skills/<topic>.md", content="the note",
         summary="one-line description for the index"),
    tool("finish", "Call when the task is complete.", summary="what you did and how you verified it"),
]

SYSTEM = """You are a careful coding agent working in a project workspace.
Work in small steps: inspect files, make a change, run the tests, repeat until they pass.
You have long-term memory that persists across sessions. Your context is small, so do not
try to remember everything: search memory at the start, and before finishing save anything
a future session would need (project structure, decisions, bugs you hit and how you fixed them).
Current project: {project}

Memory index:
{index}"""

CALL = re.compile(r"<tool_call>\s*<function=([\w_]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([\w_]+)>\n?(.*?)\n?</parameter>", re.S)
BLOCKED = re.compile(r"\brm\s+-rf\s+[/~]|\bsudo\b|\bcurl\b|\bwget\b|>\s*/dev/|\bmkfs\b")


class Workspace:
    def __init__(self, root):
        self.root = os.path.abspath(root)

    def _p(self, rel):
        p = os.path.abspath(os.path.join(self.root, rel))
        if not (p == self.root or p.startswith(self.root + os.sep)):
            raise ValueError("path escapes workspace")
        return p

    def list_files(self, path="."):
        out = []
        for d, ds, fs in os.walk(self._p(path)):
            ds[:] = [x for x in ds if not x.startswith(".") and x != "__pycache__"]
            out += [os.path.relpath(os.path.join(d, f), self.root) for f in fs]
        return "\n".join(sorted(out)) or "(empty)"

    def read_file(self, path):
        return open(self._p(path)).read()

    def write_file(self, path, content):
        p = self._p(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content if content.endswith("\n") else content + "\n")
        return f"Wrote {path} ({len(content.splitlines())} lines)."

    def run(self, command):
        if BLOCKED.search(command):
            return "Command blocked by harness policy."
        try:
            # PYTHONDONTWRITEBYTECODE: a same-size edit within the same second would otherwise run a
            # stale cached .pyc (pyc invalidation uses mtime in whole seconds + file size).
            env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
                       PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                               timeout=60, env=env)
            out = (r.stdout + r.stderr)[-4000:]
            return f"exit code {r.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return "Timed out after 60s."


def parse_calls(text):
    return [(name, {k: v for k, v in PARAM.findall(body)}) for name, body in CALL.findall(text)]


def run_agent(task, workspace, project, memory_dir, model=None, tok=None, max_steps=25, thinking=False,
              backend=None,
              log_dir=os.path.join(HERE, "trajectories"), verbose=True):
    B = backend or M
    if model is None:
        model, tok = B.load()
    mem, ws = Memory(memory_dir), Workspace(workspace)
    handlers = {
        "list_files": ws.list_files, "read_file": ws.read_file, "write_file": ws.write_file, "run": ws.run,
        "memory_search": mem.search, "memory_read": mem.read, "memory_write": mem.write,
    }
    msgs = [{"role": "system", "content": SYSTEM.format(project=project, index=mem.index())},
            {"role": "user", "content": task}]
    status, summary = "max_steps", ""
    nudges, max_nudges = 0, 2
    for step in range(max_steps):
        out = B.generate(model, tok, [msgs], thinking=thinking, max_new_tokens=2048, batch_size=1,
                         sampling=M.GREEDY if not thinking else M.THINKING_SAMPLING, tools=TOOLS, seed=step)[0]
        text = out["text"]
        reply = text.split("</think>")[-1].strip()
        msgs.append({"role": "assistant", "content": reply})
        calls = parse_calls(reply)
        if verbose:
            print(f"\n--- step {step} ---\n{reply[:1500]}")
        if not calls:
            nudges += 1
            if nudges > max_nudges:
                status = "no_tool_call"
                break
            msgs.append({"role": "user", "content": "You did not call a tool. If the task is not done, "
                         "continue by calling a tool (e.g. write_file, then run the tests). If it is done "
                         "and verified, call finish."})
            continue
        results = []
        for name, args in calls:
            if name == "finish":
                status, summary = "finished", args.get("summary", "")
                break
            try:
                res = handlers[name](**args) if name in handlers else f"Unknown tool {name}."
            except Exception as e:
                res = f"Error: {type(e).__name__}: {e}"
            results.append(str(res)[:6000])
        if status == "finished":
            break
        for r in results:
            msgs.append({"role": "tool", "content": r})
            if verbose:
                print(f"[tool] {r[:800]}")

    os.makedirs(log_dir, exist_ok=True)
    rec = dict(task=task, project=project, workspace=workspace, status=status, summary=summary,
               steps=len([m for m in msgs if m["role"] == "assistant"]), messages=msgs, tools=TOOLS)
    path = os.path.join(log_dir, f"{project}-{int(time.time())}.json")
    json.dump(rec, open(path, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--memory", default=os.path.join(HERE, "memory"))
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--patch")
    ap.add_argument("--adapter")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-steps", type=int, default=25)
    a = ap.parse_args()
    from wsp.mlx_backend import get
    B = get(a.backend)
    model, tok = B.load(patch=a.patch) if a.backend == "torch" else B.load(adapter=a.adapter)
    rec = run_agent(a.task, a.workspace, a.project, a.memory, model, tok, a.max_steps, a.thinking, backend=B)
    print(f"\n== {rec['status']} in {rec['steps']} steps: {rec['summary']}")


if __name__ == "__main__":
    main()
