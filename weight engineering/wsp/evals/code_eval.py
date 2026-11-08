"""HumanEval evaluation: run the model's function against the official tests.

Generated code runs in a separate isolated Python process with a timeout, in a temp dir.
"""
import re
import subprocess
import sys
import tempfile

from datasets import load_dataset

PROMPT = ("Complete the following Python function. Reply with the complete function "
          "(including the signature and any imports) in a single ```python code block.\n\n"
          "```python\n{prompt}```")


def load_tasks(n=None, offset=0):
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = list(ds)[offset:offset + n if n else None]
    return [dict(id=r["task_id"], prompt=r["prompt"], test=r["test"], entry_point=r["entry_point"],
                 messages=[{"role": "user", "content": PROMPT.format(prompt=r["prompt"])}])
            for r in rows]


def extract_code(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def score(task, text, timeout=10):
    code = extract_code(text)
    if f"def {task['entry_point']}" not in code:
        code = task["prompt"] + code
    program = f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/prog.py"
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
