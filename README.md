# A Bias in Disguise: tiny gradient edits that stop a small model's thinking loops

**Changing 140 of the 873 million weights in Qwen3.5-0.8B takes its thinking-mode MATH-500 score from 15% to 39%.** Changing 1,398 weights takes it to 43%. Training takes 2–9 minutes on one GPU, and 50 of the model's own answers are enough data.

The edit doesn't teach the model new math. It stops the model from looping, so it finishes answers it could already work out. Mechanically, the edit is a constant vector added to the network, written through the most constant channel in each of the first three layers.



## The problem

With thinking on, Qwen3.5-0.8B almost never stops reasoning. At a 4,096-token budget on MATH-500, only 1 of 500 answers closes its `</think>` block; the rest repeat themselves until they run out of tokens. It scores 15.4% this way, against 41.0% with thinking turned off.

In November 2026, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL) had GPT-6 Astra read this model's weights and edit 131,072 of them by hand, with no gradients. MATH-500 rose from 7.9% to 48.1%. Nobody had checked what ordinary gradient descent does on the same problem. This repo does.

## What we did

1. **Data from the model itself.** The base model answers MATH *training* problems with thinking off. We keep its correct, finished answers and rewrite them in thinking format: the reasoning, then `</think>`, then a one-line final answer. No test problems, and no other model.
2. **Choose which weights may change.** We measure how strongly each weight's gradient pulls on that data, and keep only the top *k* weights across the whole model (140 to 139,750 of them).
3. **Train only those weights**, then write them back into the original checkpoint. We count how many bf16 values actually changed.
4. **Test against controls.** We compare with random weights of the same count, a dense LoRA, and the base model with thinking off. Paired significance tests are run on the same problems.

## Results

MATH-500 with thinking on, 4,096 tokens, mean of two samples:

| Edit | Weights changed | Accuracy | Answers that finished thinking (of 500) |
|---|---|---|---|
| None (base model) | 0 | 15.4% | 1–2 |
| Gradient-chosen | 140 | 39.0% | ~280 |
| Gradient-chosen | 1,398 | 42.9% | ~342 |
| Gradient-chosen | 139,750 | 45.5% (48.0% at 8k tokens) | ~356 |
| Random positions (control) | 1,398 | 16.3% | 2–3 |
| Random positions (control) | 139,750 | 43.4% | ~345 |
| Dense LoRA, rank 16 | 409M | 41.0% | 345 |
| Base model, **thinking off** (greedy) | 0 | 41.0% | n/a |

- **Small and cheap.** 140 weights recover most of the gap. Training 1,398 weights on just 50 examples (1.9 minutes) gives 39.6%.
- **Which weights matter at small sizes.** At 1,398 weights, random positions do nothing. At the size of the Astra edit, random positions work too, so 0.016% of the weights is a generous budget for this fix.
- **It's a loop fix, not new skill.** Every problem an edit fixed was one the base model never finished. The edited model roughly matches the base model with thinking off; it beats it only at 8k tokens (48.0% vs 41.0%, p = 0.002).
- **Transfer.** It carries over to GSM8K, which it never trained on (13.6% → 56.1% with thinking on). It does not carry over to code.

All runs, with paired tests: [`results/EDITS_REPORT.md`](results/EDITS_REPORT.md).

## Why it works: a bias in disguise

The gradient puts most of the edit on four input columns of `linear_attn.out_proj` in layers 0–2. Each of these is **the single most constant channel in its layer**; layer 2's sits at −2.69 with a standard deviation of only 0.067.

Qwen3.5 has no bias terms. Changing how an always-on channel writes into the network therefore adds nearly the same vector at every position: it's a bias in disguise.

| Test | MATH-500, thinking 4k |
|---|---|
| Train only those 4 columns (4,096 weights) | 45.8% / 44.0% |
| Train 4 random columns in the same matrices | 20.4% / 20.6% |
| Base model + 3 fixed vectors added by hand, no weight change (96 problems) | 45.8% (edit: 42.7%, base: 11.5%) |
| All 3 vectors packed into a single column | 21.2% / 20.4% (fails: placement matters) |

## Compared with the GPT-6 Astra edit

The Astra edit is [public](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming), and we ran it through our evaluation. It works differently from ours:

| | Our edit | Astra edit |
|---|---|---|
| Found by | Gradient descent, minutes on one GPU | GPT-6 Astra reading the weights, no gradients |
| Where | Layers 0–2, linear-attention output | Layers 15 and 19, MLP output |
| What it does | Adds a near-constant vector | Flips one direction, multiplied by −5 on 64 columns per layer; the effect depends on the input |
| MATH-500, thinking 4k | 41.0–46.8% | 18.6% / 19.2% |

The comparison is **not** on their protocol. Their results use 32,768 tokens and a presence penalty of 1.5; ours use 4k–8k tokens and no penalty, and at those budgets their edit rarely finishes. We never ran their protocol, so we don't claim ours is better there. Stacking both edits adds nothing over ours alone.

## Also in this repo: coding adapters that didn't transfer

Before the thinking-mode work, we fine-tuned LoRA adapters on the Mac (MLX) to make the model better at coding and agent tasks. They improved our own development tasks a lot: the agent's bug-fix rate went from 22.5% to 57.5%. **None of that held up on public held-out benchmarks.** We report it because negative results are useful.

| Benchmark (held out) | Base | v4 adapter |
|---|---|---|
| HumanEval+ | 28.7% | 28.7% |
| HumanEvalFix, single reply | 8.5% | 3.0% (significantly worse) |
| HumanEvalFix, as agent tasks | 13.4% | 14.6% |
| LiveCodeBench v6 | 8.6% | 6.9% |
| GSM8K | 56.0% | 56.3% |
| MATH-500 | 41.8% | 42.6% |

The adapter learned the *form* of the agent workflow: it calls `finish` and never edits test files. It did not learn the substance: in 105 of its 109 wrong `finish` calls, the last test had just failed. Its development-set gains came from bugs made the same way as its training bugs. The full tables are in [`results/REPORT.md`](results/REPORT.md).

An earlier attempt to copy the Astra approach by hand also failed. We removed a "Wait"/self-doubt direction from the model; that halved the number of "Wait"s, but the model just doubted in other words and accuracy didn't change (`scripts/wait_direction.py`).

### Try the adapters locally (MLX, Apple silicon)

Turn thinking off and use temperature 0; that's how the adapters were trained and evaluated.

```sh
source .venv/bin/activate
python -m mlx_lm generate --model Qwen/Qwen3.5-0.8B --adapter-path adapters/v4 \
  --chat-template-config '{"enable_thinking": false}' --temp 0 --max-tokens 1024 --prompt "..."
# coding agent on a folder (it edits files and runs shell commands there: use a COPY)
python agent/harness.py "The tests in test_x.py fail. Fix x.py." --workspace /path/to/copy \
  --project myproj --adapter adapters/v4 --max-steps 15
# OpenAI-compatible server at http://localhost:8080/v1
python -m mlx_lm server --model models/qwen3.5-0.8b-v4 --temp 0 \
  --chat-template-args '{"enable_thinking": false}'
```

MBPP and the MATH training split were used for training, so scores on them aren't a fair test. HumanEval, MATH-500 and the HumanEval agent tasks were never trained on.

## Layout

| Path | What it is |
|---|---|
| `paper/` | The paper: LaTeX source and PDF |
| `cloud/gen.py` | vLLM generation on a Modal GPU (base model, our checkpoints, or any Hugging Face repo) |
| `cloud/train.py` | Sparse edits (gradient, random or chosen columns) and LoRA, merged into the original checkpoint |
| `cloud/edit.py` | Checkpoints built from part of an existing edit (ablations, one-column edit) |
| `cloud/probe.py` | Activation probe and the bias-equivalence test |
| `scripts/cloud_eval.py`, `scripts/cloud_suite.py` | Evaluations on Modal, graded locally |
| `scripts/cloud_train.py`, `scripts/build_sft.py` | Training data and training on Modal |
| `scripts/compare_astra.py`, `scripts/probe_run.py`, `scripts/probe_report.py`, `scripts/bias_run.py` | Mechanism analysis |
| `scripts/report_edits.py` → `results/EDITS_REPORT.md` | All weight-edit results with paired tests |
| `scripts/report.py` → `results/REPORT.md` | Held-out results for the coding adapters |
| `scripts/compare.py`, `scripts/fixed_analysis.py` | Paired tests between two result files |
| `wsp/evals/` | Graders: MATH (math-verify), GSM8K, HumanEval, HumanEvalFix, LiveCodeBench |
| `wsp/mlx_backend.py`, `scripts/train.py`, `wsp/chunked_delta.py` | Local MLX generation and memory-safe LoRA training on a 16 GB Mac |
| `agent/` | The coding agent (files, shell, memory tools), its tasks and its evaluation |
| `results/edits/` | The sparse edit files |

## Known quirks

- **Missing generation config.** The model repo ships no `generation_config.json`, so torch stops only on `<|endoftext|>` and MLX only on `<|im_end|>`. Both backends here add the missing stop token; without it the model invents tool responses.
- **`</think>` is not a special token.** Count an answer as finished only if it ends normally *and* contains `</think>`. Graders can find a correct-looking number inside an unfinished trace.
- **Batched greedy decoding is not reproducible.** Batch composition changes the bf16 numerics. The same 100 MATH problems scored 40% in one run and 47% in another, so small evals can't detect gains of a few points.
- **`mlx_lm lora` runs out of memory on this model.** `scripts/train.py` uses a chunked linear-attention recurrence (`wsp/chunked_delta.py`) and computes logits only where needed; peak memory is about 4.8 GB.
- **math-verify fails in worker threads.** Its timeout uses `signal.alarm`, so in a worker thread it marks every answer wrong. Grade math on the main thread.
- **One model job at a time on a 16 GB Mac.** Start long local jobs with `nohup caffeinate -i ...` so the Mac doesn't sleep.
- **The agent's `run` tool is not a sandbox.** Use it only on throwaway copies.

## Acknowledgments

Experiment code and drafting were assisted by Claude (Anthropic). The author designed the study, checked the results and takes responsibility for the content. Base model: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
