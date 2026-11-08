"""Merge self-taught data and teacher agent runs into an mlx-lm chat dataset.

  python scripts/build_dataset.py --out data/mix [--teacher-dir agent/teacher_runs]

- self data (data/self/*.jsonl): one example per problem (prompt -> shortest correct answer)
- teacher runs (passed only): one example PER RUN; "train_turns" lists the assistant messages
  that are training targets (scripts/train.py trains each of them, conditioned on its history).
  Equivalent to one example per turn -- every turn's prompt is an exact prefix of the run
  (checked for all 2147 turns) -- at 1/6 of the compute.
- turns without a tool call (nudged replies) and no_train turns (deliberately wrong first
  attempts in recovery demos) stay in the history as context but are never trained on
Every example has "train_turns" (self data: the final answer).
Writes train.jsonl / valid.jsonl (5% held out, by source) and prints a summary.
"""
import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "agent"))
from harness import TOOLS, parse_calls


TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def train_turns(rec):
    """Indices of the assistant messages to train on."""
    out = []
    for i, m in enumerate(rec["messages"]):
        if m["role"] != "assistant" or m.get("no_train"):
            continue
        calls = parse_calls(m["content"])
        if not calls:
            continue
        for name, args in calls:  # round-trip check: parsed call must be a known tool
            assert name in TOOL_NAMES, name
        out.append(i)
    return out


def agent_example(rec):
    turns = train_turns(rec)
    msgs = [{k: v for k, v in m.items() if k != "no_train"} for m in rec["messages"][:turns[-1] + 1]]
    return {"messages": msgs, "tools": TOOLS, "train_turns": turns}


def agent_examples(rec):
    """Per-turn view (one example per trained turn), used to check prefix equivalence."""
    ex = agent_example(rec)
    return [{"messages": ex["messages"][:i + 1], "tools": TOOLS} for i in ex["train_turns"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mix")
    ap.add_argument("--self-dir", default="data/self")
    ap.add_argument("--teacher-dir", default="agent/teacher_runs_authored")
    ap.add_argument("--max-math", type=int, help="cap math examples to balance the mix")
    a = ap.parse_args()
    rng = random.Random(0)

    groups = {}
    for p in sorted(glob.glob(f"{a.self_dir}/*.jsonl")):
        src = os.path.basename(p).split(".")[0]
        rows = [json.loads(l) for l in open(p)]
        groups[f"self_{src}"] = [{"messages": r["messages"], "train_turns": [len(r["messages"]) - 1]}
                                 for r in rows]
    if a.max_math and "self_math" in groups:
        rng.shuffle(groups["self_math"])
        groups["self_math"] = groups["self_math"][:a.max_math]

    runs = [json.load(open(p)) for p in glob.glob(f"{a.teacher_dir}/*.json")]
    passed = [r for r in runs if r["passed"]]
    # split by RUN (not by turn) so validation turns come from unseen runs
    rng.shuffle(passed)
    n_val_runs = max(1, len(passed) // 20) if passed else 0
    agent_train = [agent_example(r) for r in passed[n_val_runs:]]
    agent_valid = [agent_example(r) for r in passed[:n_val_runs]]

    train, valid = list(agent_train), list(agent_valid)
    for name, rows in groups.items():
        rng.shuffle(rows)
        k = max(1, len(rows) // 20)
        valid += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    os.makedirs(a.out, exist_ok=True)
    for split, rows in (("train", train), ("valid", valid)):
        with open(f"{a.out}/{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({**{k: len(v) for k, v in groups.items()},
                      "teacher_runs": len(runs), "teacher_passed": len(passed),
                      "agent_trained_turns": sum(len(e["train_turns"]) for e in agent_train + agent_valid),
                      "train": len(train), "valid": len(valid)}, indent=1))


if __name__ == "__main__":
    main()
