"""Generate the starter agentic task suite under agent/tasks/.

Each task dir has the starting files, task.json (prompt, test command), and a hidden test
in _hidden_test.py that the grader runs after the agent finishes (the agent also sees a
visible test it can run).
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")

TASKS = {
    "median_bug": {
        "prompt": "The tests in test_stats.py fail. Find and fix the bug in stats.py. Run `python test_stats.py` to check.",
        "files": {
            "stats.py": "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    if n % 2 == 1:\n        return s[n // 2]\n    return (s[n // 2] + s[n // 2 + 1]) / 2\n",
            "test_stats.py": "from stats import median\nassert median([3, 1, 2]) == 2\nassert median([4, 1, 3, 2]) == 2.5\nprint('ok')\n",
        },
        "hidden": "from stats import median\nassert median([1, 2]) == 1.5\nassert median([5]) == 5\nassert median([7, 1, 3, 9, 5, 11]) == 6\n",
    },
    "slugify": {
        "prompt": "Implement slugify(text) in textutils.py: lowercase, keep only a-z and 0-9, turn every run of other characters into a single '-', and strip leading/trailing '-'. Run `python test_textutils.py`.",
        "files": {
            "textutils.py": "def slugify(text):\n    raise NotImplementedError\n",
            "test_textutils.py": "from textutils import slugify\nassert slugify('Hello, World!') == 'hello-world'\nprint('ok')\n",
        },
        "hidden": "from textutils import slugify\nassert slugify('  A  B  ') == 'a-b'\nassert slugify('C++ & Rust 2026') == 'c-rust-2026'\nassert slugify('---') == ''\n",
    },
    "brackets": {
        "prompt": "Write is_balanced(s) in brackets.py that returns True if every (, [ and { in s is closed in the correct order, ignoring other characters. Run `python test_brackets.py`.",
        "files": {
            "brackets.py": "",
            "test_brackets.py": "from brackets import is_balanced\nassert is_balanced('(a[b]{c})')\nassert not is_balanced('(]')\nprint('ok')\n",
        },
        "hidden": "from brackets import is_balanced\nassert is_balanced('')\nassert not is_balanced('((')\nassert not is_balanced('([)]')\nassert is_balanced('x{y(z)[w]}')\nassert not is_balanced('}')\n",
    },
    "inventory_multi_file": {
        "prompt": "report.total_value() returns the wrong number. Investigate the package (models.py, store.py, report.py), fix the bug, and run `python test_report.py`.",
        "files": {
            "models.py": "from dataclasses import dataclass\n\n@dataclass\nclass Item:\n    name: str\n    unit_price: float\n    quantity: int\n",
            "store.py": "from models import Item\n\ndef load():\n    return [Item('bolt', 0.25, 100), Item('nut', 0.10, 250), Item('gear', 12.5, 4)]\n",
            "report.py": "from store import load\n\ndef total_value():\n    return round(sum(i.unit_price for i in load()), 2)\n",
            "test_report.py": "from report import total_value\nassert total_value() == 100.0, total_value()\nprint('ok')\n",
        },
        "hidden": "from report import total_value\nassert total_value() == 100.0\n",
    },
}

# Two sessions on the same project. Session 1 reveals a project convention; session 2 needs it.
# With good memory use, session 2 should succeed in fewer steps.
LEDGER_FILES = {
    "ledger.py": ("# Convention: all money is stored as integer cents. Never use floats for money.\n"
                  "def add(entries, desc, cents):\n    entries.append({'desc': desc, 'cents': cents})\n\n"
                  "def balance(entries):\n    return sum(e['amount'] for e in entries)\n"),
    "test_ledger.py": ("from ledger import add, balance\ne = []\nadd(e, 'coffee', -450)\nadd(e, 'salary', 100000)\n"
                       "assert balance(e) == 99550\nprint('ok')\n"),
}
SESSIONS = {
    "ledger_s1": {
        "prompt": "test_ledger.py fails. Fix ledger.py and run `python test_ledger.py`.",
        "files": LEDGER_FILES,
        "hidden": "from ledger import add, balance\ne=[]\nadd(e,'a',5)\nassert balance(e)==5\n",
    },
    "ledger_s2": {
        "prompt": ("Add split(entries, desc, total, n) to ledger.py: it records n entries for desc that together "
                   "sum exactly to total, as evenly as possible. Follow the project's conventions. "
                   "Write your own quick test and run it."),
        "files": {"ledger.py": LEDGER_FILES["ledger.py"].replace("e['amount']", "e['cents']")},
        "hidden": ("from ledger import split, balance\ne=[]\nsplit(e,'dinner',1000,3)\n"
                   "assert balance(e)==1000\nassert all(isinstance(x['cents'], int) for x in e)\n"
                   "assert sorted(x['cents'] for x in e)==[333,333,334]\n"),
        "after": "ledger_s1",
    },
}


def main():
    for name, t in {**TASKS, **SESSIONS}.items():
        d = os.path.join(ROOT, name)
        os.makedirs(d, exist_ok=True)
        for fn, body in t["files"].items():
            open(os.path.join(d, fn), "w").write(body)
        open(os.path.join(d, "_hidden_test.py"), "w").write(t["hidden"])
        meta = {"prompt": t["prompt"], "project": name.split("_s")[0] if "after" in t or name.endswith("_s1") else name}
        if "after" in t:
            meta["after"] = t["after"]
        json.dump(meta, open(os.path.join(d, "task.json"), "w"), indent=1)
    print("wrote", len(TASKS) + len(SESSIONS), "tasks to", ROOT)


if __name__ == "__main__":
    main()
