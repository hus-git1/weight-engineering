"""Generate TRAINING tasks for the agent (separate from the 6 eval tasks in agent/tasks/).

  python agent/gen_train_tasks.py            -> agent/train_tasks/{single,memory}/...

single/  MBPP-based: implement a stubbed function, or fix an injected bug.
memory/  two-session projects. Session 1's workspace has a README stating a project rule and
         a bug related to it. Session 2's workspace has NO README and the rule is (mostly) not
         inferable from code, so passing needs the memory saved in session 1.

Every task is verified: the starting files fail the hidden test, a reference solution passes.
"""
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

from datasets import load_dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_tasks")
PY = sys.executable


def run_hidden(files, hidden, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        for fn, body in files.items():
            open(os.path.join(d, fn), "w").write(body)
        open(os.path.join(d, "_hidden_test.py"), "w").write(hidden)
        try:
            return subprocess.run([PY, "-E", "-s", "_hidden_test.py"], cwd=d, capture_output=True,
                                  timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return None


def write_task(path, files, hidden, meta):
    os.makedirs(path, exist_ok=True)
    for fn, body in files.items():
        open(os.path.join(path, fn), "w").write(body)
    open(os.path.join(path, "_hidden_test.py"), "w").write(hidden)
    json.dump(meta, open(os.path.join(path, "task.json"), "w"), indent=1)


# ---------------------------------------------------------------- single tasks (MBPP)
MUTATIONS = [("<=", "<"), ("<", "<="), (">=", ">"), (">", ">="), ("+ 1", ""), ("- 1", ""),
             ("max(", "min("), ("min(", "max("), (" and ", " or "), (" or ", " and "),
             ("==", "!="), ("[0]", "[1]"), ("+=", "-="), ("return True", "return False"),
             ("range(1,", "range(0,"), ("sorted(", "list("), ("reverse=True", "reverse=False")]
MODULES = ["solution", "utils", "core", "helpers", "tools", "funcs"]


def main_function(code, tests):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(node.name + "(" in t for t in tests):
            return node
    return None


def single_tasks(rng, n_impl, n_fix):
    rows = [r for _, ds in load_dataset("google-research-datasets/mbpp", "full").items() for r in ds]
    rng.shuffle(rows)
    made = {"implement": 0, "bugfix": 0}
    for r in rows:
        if made["implement"] >= n_impl and made["bugfix"] >= n_fix:
            break
        tests = r["test_list"] + r["challenge_test_list"]
        try:
            fn = main_function(r["code"], tests)
        except SyntaxError:
            continue
        if fn is None:
            continue
        mod = rng.choice(MODULES)
        setup = (r["test_setup_code"] + "\n") if r["test_setup_code"].strip() else ""
        hidden = f"{setup}from {mod} import *\n" + "\n".join(tests) + "\n"
        kind = "implement" if made["implement"] < n_impl and (made["bugfix"] >= n_fix or rng.random() < 0.5) else "bugfix"
        ref = {mod + ".py": setup + r["code"] + "\n"}
        if not run_hidden(ref, hidden):
            continue
        if kind == "implement":
            sig = f"def {fn.name}({ast.unparse(fn.args)}):"
            start = {mod + ".py": f'{setup}{sig}\n    """{r["text"]}"""\n    raise NotImplementedError\n'}
            visible = f"{setup}from {mod} import *\n{tests[0]}\nprint('ok')\n"
            prompt = (f"Implement `{fn.name}` in {mod}.py: {r['text']} "
                      f"Run `python test_{mod}.py` to check, and make sure your solution is general.")
        else:
            start = None
            for a, b in rng.sample(MUTATIONS, len(MUTATIONS)):
                if a in r["code"]:
                    mutated = r["code"].replace(a, b, 1)
                    try:
                        ast.parse(mutated)
                    except SyntaxError:
                        continue
                    cand = {mod + ".py": setup + mutated + "\n"}
                    if run_hidden(cand, hidden) is False:  # fails, but does not hang
                        start = cand
                        break
            if start is None:
                continue
            visible = f"{setup}from {mod} import *\n" + "\n".join(tests[:2]) + "\nprint('ok')\n"
            prompt = (f"The tests in test_{mod}.py fail. Find and fix the bug in {mod}.py "
                      f"(`{fn.name}`: {r['text']}). Run `python test_{mod}.py` to check.")
        files = {**start, f"test_{mod}.py": visible}
        if run_hidden(files, hidden) is not False:
            continue
        made[kind] += 1
        name = f"{kind}_mbpp{r['task_id']}"
        write_task(os.path.join(ROOT, "single", name), files, hidden,
                   dict(prompt=prompt, project=f"mbpp{r['task_id']}", kind=kind, reference=ref))
    return made


# ---------------------------------------------------------------- memory (two-session) tasks
NAMES = ["atlas", "beacon", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
         "kestrel", "lumen", "meridian", "nimbus", "onyx", "pylon", "quartz", "ridge", "sable", "tundra",
         "umber", "vertex", "willow", "xenon", "yarrow", "zephyr", "anchor", "bramble", "cinder", "drift",
         "eddy", "falcon", "glacier", "hollow", "iris", "jasper", "kelp", "lagoon"]


def fam_money(P):
    readme = f"# {P}\n\nConventions:\n- All money amounts are integer cents (e.g. $12.50 is 1250). Never use floats for money.\n"
    s1 = {"README.md": readme,
          "pricing.py": "def line_total(unit, qty):\n    return unit * qty\n\n\ndef order_total(lines):\n    return sum(line_total(u, q) for u, q in lines) / 100\n",
          "test_pricing.py": "from pricing import order_total\nassert order_total([(250, 2), (1000, 1)]) == 1500\nprint('ok')\n"}
    s1_ref = {**s1, "pricing.py": s1["pricing.py"].replace(" / 100", "")}
    s2 = {"pricing.py": s1_ref["pricing.py"]}
    s2_ref = {"pricing.py": s2["pricing.py"] + "\n\ndef apply_tax(amount, rate_percent):\n    return int(amount * (100 + rate_percent) / 100 + 0.5)\n"}
    return dict(rule="money is integer cents, never floats",
                s1=(s1, "from pricing import order_total\nassert order_total([(5, 3)]) == 15\n",
                    "test_pricing.py fails. Read the project README, fix pricing.py, and run `python test_pricing.py`.", s1_ref),
                s2=(s2, "from pricing import apply_tax\nr = apply_tax(1000, 8)\nassert r == 1080 and isinstance(r, int), r\nassert apply_tax(999, 7.5) == 1074\n",
                    "Add `apply_tax(amount, rate_percent)` to pricing.py: it returns the amount with tax added, rounded to the nearest unit, following this project's money conventions. Write a quick test and run it.", s2_ref))


def fam_errors(P):
    E = P.capitalize() + "Error"
    readme = f"# {P}\n\nConventions:\n- Invalid input must raise `{E}` from errors.py, never ValueError or TypeError.\n"
    errors = f"class {E}(Exception):\n    pass\n"
    s1 = {"README.md": readme, "errors.py": errors,
          "validate.py": "def check_port(p):\n    if not 0 < p < 65536:\n        raise ValueError(f'bad port {p}')\n    return p\n",
          "test_validate.py": f"from validate import check_port\nfrom errors import {E}\ntry:\n    check_port(70000)\n    raise SystemExit('no error raised')\nexcept {E}:\n    print('ok')\n"}
    s1_ref = {**s1, "validate.py": f"from errors import {E}\n\n\ndef check_port(p):\n    if not 0 < p < 65536:\n        raise {E}(f'bad port {{p}}')\n    return p\n"}
    s2 = {"errors.py": errors, "validate.py": "def check_port(p):\n    return p\n"}
    s2_ref = {**s2, "validate.py": f"import re\nfrom errors import {E}\n\n\ndef check_port(p):\n    return p\n\n\ndef check_username(name):\n    if not re.fullmatch(r'[A-Za-z0-9_]{{3,16}}', name):\n        raise {E}(name)\n    return name\n"}
    hidden2 = (f"from validate import check_username\nfrom errors import {E}\nassert check_username('ok_name') == 'ok_name'\n"
               f"for bad in ['a!', 'ab', 'x' * 17]:\n    try:\n        check_username(bad)\n        raise SystemExit('accepted ' + bad)\n    except {E}:\n        pass\n")
    return dict(rule=f"invalid input raises {E} from errors.py",
                s1=(s1, f"from validate import check_port\nfrom errors import {E}\nassert check_port(80) == 80\ntry:\n    check_port(0)\n    raise SystemExit(1)\nexcept {E}:\n    pass\n",
                    "test_validate.py fails. Check the project conventions in the README, fix validate.py, and run `python test_validate.py`.", s1_ref),
                s2=(s2, hidden2, "Add `check_username(name)` to validate.py: valid names are 3-16 characters of letters, digits or underscore; return the name if valid. Follow the project's error-handling conventions. Write a quick test and run it.", s2_ref))


def fam_dates(P):
    readme = f"# {P}\n\nConventions:\n- Dates are always strings formatted DD/MM/YYYY (e.g. 05/11/2026).\n"
    s1 = {"README.md": readme,
          "dates.py": "from datetime import datetime\n\n\ndef parse(s):\n    return datetime.strptime(s, '%Y-%m-%d').date()\n\n\ndef fmt(d):\n    return d.strftime('%d/%m/%Y')\n",
          "test_dates.py": "from dates import parse, fmt\nassert fmt(parse('05/11/2026')) == '05/11/2026'\nprint('ok')\n"}
    s1_ref = {**s1, "dates.py": s1["dates.py"].replace("'%Y-%m-%d'", "'%d/%m/%Y'")}
    s2 = {"dates.py": "from datetime import datetime, timedelta\n"}
    s2_ref = {"dates.py": s2["dates.py"] + "\n\ndef add_days(date_str, n):\n    d = datetime.strptime(date_str, '%d/%m/%Y') + timedelta(days=n)\n    return d.strftime('%d/%m/%Y')\n"}
    return dict(rule="dates are DD/MM/YYYY strings",
                s1=(s1, "from dates import parse\nassert parse('31/01/2026').month == 1\n",
                    "test_dates.py fails. Read the README, fix dates.py, and run `python test_dates.py`.", s1_ref),
                s2=(s2, "from dates import add_days\nassert add_days('30/12/2026', 3) == '02/01/2027'\nassert add_days('01/03/2026', -1) == '28/02/2026'\n",
                    "Add `add_days(date_str, n)` to dates.py that returns the date n days later, using this project's date format. Write a quick test and run it.", s2_ref))


def fam_ids(P):
    X = P[:3].upper()
    readme = f"# {P}\n\nConventions:\n- Record IDs look like `{X}-000123`: the prefix `{X}`, a dash, and the number zero-padded to 6 digits.\n"
    s1 = {"README.md": readme, "ids.py": f"def make_id(n):\n    return f'{X}-{{n}}'\n",
          "test_ids.py": f"from ids import make_id\nassert make_id(42) == '{X}-000042'\nprint('ok')\n"}
    s1_ref = {**s1, "ids.py": f"def make_id(n):\n    return f'{X}-{{n:06d}}'\n"}
    s2 = {"records.py": "# record helpers\n"}
    s2_ref = {"records.py": f"def next_id(existing_ids):\n    nums = [int(i.split('-')[1]) for i in existing_ids]\n    return f'{X}-{{max(nums, default=0) + 1:06d}}'\n"}
    return dict(rule=f"record IDs are {X}-NNNNNN (6-digit zero-padded)",
                s1=(s1, f"from ids import make_id\nassert make_id(7) == '{X}-000007'\n",
                    "test_ids.py fails. Read the README, fix ids.py, and run `python test_ids.py`.", s1_ref),
                s2=(s2, f"from records import next_id\nassert next_id(['{X}-000041', '{X}-000007']) == '{X}-000042'\nassert next_id([]) == '{X}-000001'\n",
                    "Add `next_id(existing_ids)` to records.py: given a list of existing record IDs, return the next ID (one more than the highest number) in the project's ID format. Write a quick test and run it.", s2_ref))


def fam_seconds(P):
    readme = f"# {P}\n\nConventions:\n- All durations are integers in seconds.\n"
    s1 = {"README.md": readme, "timeparse.py": "def minutes(n):\n    return n\n",
          "test_timeparse.py": "from timeparse import minutes\nassert minutes(3) == 180\nprint('ok')\n"}
    s1_ref = {**s1, "timeparse.py": "def minutes(n):\n    return n * 60\n"}
    s2 = {"timeparse.py": "# duration helpers\n"}
    s2_ref = {"timeparse.py": "import re\n\n\ndef parse_duration(text):\n    h = re.search(r'(\\d+)h', text)\n    m = re.search(r'(\\d+)m', text)\n    return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60\n"}
    return dict(rule="durations are integer seconds",
                s1=(s1, "from timeparse import minutes\nassert minutes(1) == 60\n",
                    "test_timeparse.py fails. Read the README, fix timeparse.py, and run `python test_timeparse.py`.", s1_ref),
                s2=(s2, "from timeparse import parse_duration\nassert parse_duration('1h30m') == 5400\nassert parse_duration('45m') == 2700\nassert parse_duration('2h') == 7200\n",
                    "Add `parse_duration(text)` to timeparse.py that parses strings like '1h30m', '45m' or '2h' and returns the duration in the project's standard unit. Write a quick test and run it.", s2_ref))


def fam_registry(P):
    readme = f"# {P}\n\nConventions:\n- Every command is a function `cmd_<name>(args) -> str` in commands.py and must be registered in `COMMANDS` (registry.py) under its name.\n"
    s1 = {"README.md": readme, "registry.py": "COMMANDS = {}\n",
          "commands.py": "from registry import COMMANDS\n\n\ndef cmd_echo(args):\n    return ' '.join(args)\n",
          "test_commands.py": "import commands\nfrom registry import COMMANDS\nassert COMMANDS['echo'](['a', 'b']) == 'a b'\nprint('ok')\n"}
    s1_ref = {**s1, "commands.py": s1["commands.py"] + "\n\nCOMMANDS['echo'] = cmd_echo\n"}
    s2 = {"registry.py": "COMMANDS = {}\n", "commands.py": s1_ref["commands.py"]}
    s2_ref = {**s2, "commands.py": s2["commands.py"] + "\n\ndef cmd_shout(args):\n    return ' '.join(args).upper()\n\n\nCOMMANDS['shout'] = cmd_shout\n"}
    return dict(rule="commands are cmd_<name>(args) registered in COMMANDS",
                s1=(s1, "import commands\nfrom registry import COMMANDS\nassert COMMANDS['echo']([]) == ''\n",
                    "test_commands.py fails. Read the README for the project's conventions, fix it, and run `python test_commands.py`.", s1_ref),
                s2=(s2, "import commands\nfrom registry import COMMANDS\nassert COMMANDS['shout'](['hi', 'you']) == 'HI YOU'\n",
                    "Add a `shout` command that returns its arguments joined by spaces in upper case. Follow the project's conventions for commands. Write a quick test and run it.", s2_ref))


def fam_percent(P):
    readme = f"# {P}\n\nConventions:\n- Percentages are floats from 0 to 100, rounded to 1 decimal place (e.g. 33.3, not 0.333).\n"
    s1 = {"README.md": readme, "stats.py": "def pct(part, whole):\n    return part / whole\n",
          "test_stats.py": "from stats import pct\nassert pct(1, 3) == 33.3\nprint('ok')\n"}
    s1_ref = {**s1, "stats.py": "def pct(part, whole):\n    return round(100 * part / whole, 1)\n"}
    s2 = {"stats.py": "# statistics helpers\n"}
    s2_ref = {"stats.py": "def growth(old, new):\n    return round(100 * (new - old) / old, 1)\n"}
    return dict(rule="percentages are 0-100 floats rounded to 1 decimal",
                s1=(s1, "from stats import pct\nassert pct(1, 2) == 50.0\n",
                    "test_stats.py fails. Read the README, fix stats.py, and run `python test_stats.py`.", s1_ref),
                s2=(s2, "from stats import growth\nassert growth(200, 250) == 25.0\nassert growth(3, 2) == -33.3\n",
                    "Add `growth(old, new)` to stats.py returning the percentage change from old to new, following the project's conventions for percentages. Write a quick test and run it.", s2_ref))


FAMILIES = [fam_money, fam_errors, fam_dates, fam_ids, fam_seconds, fam_registry, fam_percent]


def memory_tasks(rng, per_family):
    names = NAMES[:]
    rng.shuffle(names)
    rules, made = [], 0
    for fam in FAMILIES:
        for _ in range(per_family):
            P = names.pop()
            spec = fam(P)
            pair = f"{fam.__name__[4:]}_{P}"
            for s in ("s1", "s2"):
                files, hidden, prompt, ref = spec[s]
                assert run_hidden(files, hidden) is False, (pair, s, "start should fail")
                assert run_hidden({k: v for k, v in ref.items() if k != "README.md"} | ({"README.md": ref["README.md"]} if "README.md" in ref else {}), hidden), (pair, s, "reference should pass")
                write_task(os.path.join(ROOT, "memory", pair, s), files, hidden,
                           dict(prompt=prompt, project=P, kind=f"memory_{s}", pair=pair, rule=spec["rule"]))
            rules.append(dict(project=P, rule=spec["rule"]))
            made += 1
    json.dump(rules, open(os.path.join(ROOT, "memory", "rules.json"), "w"), indent=1)
    return made


def main():
    rng = random.Random(0)
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    pairs = memory_tasks(rng, per_family=5)
    singles = single_tasks(rng, n_impl=100, n_fix=100)
    print(f"memory pairs: {pairs} ({2*pairs} sessions); single tasks: {singles}")


if __name__ == "__main__":
    main()
