"""File-based long-term memory for the coding agent.

Layout:
  memory/MEMORY.md                 index: one line per memory, always shown to the agent
  memory/projects/<project>/*.md   architecture, decisions, gotchas for one codebase
  memory/skills/*.md               reusable lessons that apply across projects

The agent keeps a small context window and pulls notes in on demand with search/read.
"""
import math
import os
import re
from collections import Counter

INDEX = "MEMORY.md"
WORD = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def _tokens(text):
    return [w.lower() for w in WORD.findall(text)]


class Memory:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self._p(INDEX)):
            with open(self._p(INDEX), "w") as f:
                f.write("# Memory index\n")

    def _p(self, rel):
        path = os.path.abspath(os.path.join(self.root, rel))
        if not path.startswith(self.root + os.sep) and path != self.root:
            raise ValueError("path escapes memory directory")
        return path

    def files(self):
        for d, _, fs in os.walk(self.root):
            for f in fs:
                if f.endswith(".md") and f != INDEX:
                    yield os.path.relpath(os.path.join(d, f), self.root)

    def index(self):
        return open(self._p(INDEX)).read()

    def search(self, query, k=5):
        """BM25-style keyword search over all memory files."""
        docs = {rel: open(self._p(rel)).read() for rel in self.files()}
        if not docs:
            return "No memories yet."
        toks = {rel: Counter(_tokens(rel + " " + t)) for rel, t in docs.items()}
        avg = sum(sum(c.values()) for c in toks.values()) / len(toks)
        q = set(_tokens(query))
        scores = {}
        for rel, c in toks.items():
            n, s = sum(c.values()), 0.0
            for w in q:
                df = sum(1 for cc in toks.values() if w in cc)
                if c[w]:
                    idf = math.log(1 + (len(toks) - df + 0.5) / (df + 0.5))
                    s += idf * c[w] * 2.2 / (c[w] + 1.2 * (0.25 + 0.75 * n / avg))
            if s > 0:
                scores[rel] = s
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        if not best:
            return "No matching memories."
        out = []
        for rel in best:
            first = docs[rel].strip().splitlines()
            out.append(f"- {rel}: {' '.join(first[:3])[:200]}")
        return "\n".join(out)

    def read(self, path):
        p = self._p(path)
        return open(p).read() if os.path.exists(p) else f"No memory at {path}."

    def write(self, path, content, summary):
        if not path.endswith(".md"):
            path += ".md"
        p = self._p(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content.strip() + "\n")
        lines = [l for l in self.index().splitlines() if not l.startswith(f"- [{path}]")]
        lines.append(f"- [{path}] {summary.strip()}")
        with open(self._p(INDEX), "w") as f:
            f.write("\n".join(lines) + "\n")
        return f"Saved {path}."
