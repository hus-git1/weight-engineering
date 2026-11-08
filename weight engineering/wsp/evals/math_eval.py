"""MATH-500 evaluation: \\boxed{} answer extraction checked with math-verify."""
from datasets import load_dataset
from math_verify import parse, verify

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def load_tasks(n=None, offset=0):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = list(ds)[offset:offset + n if n else None]
    return [dict(id=r["unique_id"], level=r["level"], answer=r["answer"],
                 messages=[{"role": "user", "content": PROMPT.format(problem=r["problem"])}])
            for r in rows]


def final_answer_text(text):
    """Only grade what comes after the thinking block, if there is one."""
    return text.split("</think>")[-1] if "</think>" in text else text


def score(task, text):
    try:
        gold = parse("$" + task["answer"] + "$")
        pred = parse(final_answer_text(text))
        return bool(pred) and verify(gold, pred)
    except Exception:
        return False
