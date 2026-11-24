# A Bias in Disguise: Tiny Gradient Edits Stop a Small Model's Thinking Loops and Complement a Frontier Model's Edit

**Abstract.** With thinking on, Qwen3.5-0.8B almost never finishes reasoning: 1 of 500 MATH-500 answers closes its think block, and accuracy is 15.4%. MIT CSAIL recently had GPT-6 Astra edit 131,072 of its weights without gradients and reported MATH-500 rising from 7.9% to 48.1%. We test the obvious baseline that work did not run: plain gradient descent on a handful of weights. Training only the 140 weights with the largest gradient, on the model's own answers, raises thinking-mode MATH-500 to 39.0%; 1,398 weights reach 42.9%. Training takes minutes on one GPU, and 50 examples suffice. The gain is a termination fix, not new ability, and mechanically the edit is a constant vector hidden in the model's most constant channels. Under MIT's own 32k-token protocol their edit beats ours (52.1% vs 45.7%), but ours wins at every budget up to 16k. **Combined, the two edits beat both: MIT's edit plus ours at half strength scores 62.2%, 10 points above MIT's, with half the tokens.**

## 1. Problem

Qwen3.5-0.8B has a thinking mode: it reasons inside a `<think>` block, then answers. In practice it rarely closes the block. It repeats itself ("wait, let me reconsider…") until the token budget runs out.

| Qwen3.5-0.8B on MATH-500 | Accuracy | Answers that finish |
|---|---|---|
| Thinking on, 4,096 tokens | 15.4% | 1 of 500 |
| Thinking off | 41.0% | n/a |

Thinking is supposed to help, but here it costs 26 points. The model can solve these problems; it cannot stop.

## 2. Our focus

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, November 2026) showed that a frontier model can read this model's weights and repair the loop by hand. It did not show whether that needs a frontier model. We ask four questions:

1. Can ordinary gradient descent find an equally small fix, and how small, and from how little data?
2. Does the choice of weights matter, or would any weights do?
3. What does the fix actually do inside the network, and how does it relate to MIT's?
4. Under MIT's own protocol, which edit wins, and can the two be combined?

## 3. Methodology

1. **Self-generated data.** The base model answers MATH *training* problems with thinking off. We keep its correct, finished answers and rewrite each in thinking format: the same reasoning, then `</think>`, then a one-line answer. No test problems and no stronger model are used, so any gain must come from ability the model already has.
2. **Weight selection.** We pass this data through the frozen model and add up each weight's absolute gradient. Only the top *k* weights across the whole model may change (*k* = 140 to 139,750).
3. **Training.** Only those *k* values are trained, and the result is written back into the original bf16 checkpoint.
4. **Controls.** Random weights of the same count, random weight columns, a dense LoRA, and the base model with thinking off.
5. **Evaluation.** MATH-500 (and GSM8K, HumanEval) with thinking on, at 4,096 and 7,600 tokens, temperature 0.6, two samples. We use paired exact tests on the same problems, and we count an answer as finished only if it closes `</think>`.
6. **MIT's protocol.** For the head-to-head (Section 7) we use MIT's settings: 32,768 tokens, presence penalty 1.5, same vLLM version, two samples (MIT uses four). "Strict" means finished and correct. Scores at smaller budgets are read off the same answers (an answer counts if it finished within the budget), which matches direct runs within about 1 point.

## 4. Results

MATH-500, thinking on, 4,096 tokens, mean of two samples:

| Edit | Weights changed | Accuracy | Answers that finish (of 500) |
|---|---|---|---|
| None (base model) | 0 | 15.4% | 1–2 |
| Gradient-chosen | 140 | 39.0% | ~280 |
| Gradient-chosen | 1,398 | 42.9% | ~342 |
| Gradient-chosen | 139,750 | 45.5% (48.0% at 8k tokens) | ~356 |
| Random positions | 1,398 | 16.3% | 2–3 |
| Random positions | 139,750 | 43.4% | ~345 |
| Dense LoRA | 409M | 41.0% | 345 |
| Base model, thinking off | 0 | 41.0% | n/a |

- **Small and cheap.** 140 weights close most of the gap. 1,398 weights trained on only 50 examples (1.9 minutes) reach 39.6%.
- **Selection matters at small sizes.** 1,398 random weights do nothing. At 140k weights even random ones work, so a 0.016% budget is generous for this fix.
- **Termination, not ability.** Every problem an edit fixed was one the base model never finished. The edited model roughly matches the base model with thinking off; it beats it only at 8k tokens (48.0% vs 41.0%, p = 0.002).
- **Transfer.** It carries over to GSM8K (13.6% → 56.1%), which was never trained on, but not to code.

## 5. Why it works: a bias in disguise

The gradient places most of the edit on four weight columns in layers 0–2 (`linear_attn.out_proj`). Each column belongs to **the most constant channel in its layer**; one sits at −2.69 with a standard deviation of 0.067. The model has no bias terms. Changing how an always-on channel writes into the network therefore adds nearly the same vector at every token, which is a bias in disguise. The training signal asks for one uniform shift (finish sooner), and an always-on channel is the cheapest place to put a constant.

| Test | MATH-500, thinking on |
|---|---|
| Train only those 4 columns (4,096 weights) | 45.8% / 44.0% |
| Train 4 random columns, same matrices | 20.4% / 20.6% |
| No weight change: add 3 fixed vectors to the base model | 45.8% (edit 42.7%, base 11.5%; 96 problems) |
| All 3 vectors in a single column | 21.2% / 20.4%: placement matters |

## 6. Comparison with MIT CSAIL

**Similar.** Same model, same failure, same goal. Both edits are tiny and sparse, change how a few units write into the network, and are stored directly in the weights. Both teams conclude the gain unlocks existing ability rather than adding new ability, and both see it transfer to GSM8K.

**Different.**

| | MIT CSAIL | This work |
|---|---|---|
| Who finds the edit | GPT-6 Astra reasoning over the weights | Gradient descent |
| Signal | 50 forward passes, no gradients | Gradients on the model's own answers |
| Weights changed | 131,072 | 140 to 139,750 |
| Location | Layers 15 and 19, MLP output | Layers 0–2, linear-attention output |
| What the edit does | Flips one direction (×−5); effect depends on the input | Adds a near-constant vector |
| Controls | None reported for this edit | Random weights, random columns, LoRA, thinking off, ablations |
| Protocol | 32,768 tokens, presence penalty 1.5, 4 samples | 4,096 / 7,600 tokens, no penalty, 2 samples |

The two edits point in unrelated directions (|cos| ≤ 0.10, about chance). The loop can be switched off from more than one place.

**Before and after.** The MIT column uses their protocol; the other three use ours. Compare only within the same protocol.

| Benchmark (thinking on unless noted) | MIT CSAIL, their protocol | Base, our protocol | Astra edit, our protocol | Our 139,750-weight edit, our protocol |
|---|---|---|---|---|
| MATH-500 | 7.9% → 48.1% | 15.4% | 18.9% | **45.5%** |
| MATH-500, 8k tokens | — | 15.6% | 24.0% | **48.0%** |
| MATH-500 answers out of tokens | 93% → 33% | 99.8% | 97.2% | **30.2%** |
| GSM8K | 13.8% → 67.1% | 13.6% | 21.5% | **56.1%** |
| HumanEval | — | 27.4% | **32.3%** | 23.8% |
| MATH-500, thinking off | — | **41.0%** | 37.2% | 40.4% |
| IFEval | 21.0% → 62.3% | — | — | — |

Our smaller edits reach 39.0% (140 weights) and 42.9% (1,398) on MATH-500. At these short budgets the Astra edit rarely finishes; it was built for 32k-token runs. Section 7 runs MIT's protocol.

## 7. Under MIT's protocol: each edit wins in its own regime, and together they win

MATH-500, 32,768 tokens, presence penalty 1.5, two samples per problem:

| Model | Weights changed | Strict | Out of tokens | Avg tokens | vs MIT's edit |
|---|---|---|---|---|---|
| MIT's edit (their report / re-run) | 131,072 | 48.1% / 51.0% (any answer) | 33% / 32% | — | |
| MIT's edit (our run) | 131,072 | 52.1% | 32.0% | 21,966 | |
| Ours, 1,398 | 1,398 | 45.7% | 11.5% | 6,861 | −6.4, p = 2×10⁻⁶ |
| Ours, 139,750 at half strength | 139,750 | 46.7% | 42.9% | 16,230 | −5.4, p = 6×10⁻⁴ |
| MIT + ours, full strength | ~270,000 | 49.7% | 0.5% | 4,382 | −2.4, p = 0.11 |
| **MIT + ours, half strength** | ~270,000 | **62.2%** | 8.2% | **11,523** | **+10.1, p = 3×10⁻⁹** |
| MIT + ours, quarter strength | ~270,000 | 59.8% | 20.9% | 17,483 | +7.7, p = 10⁻⁸ |

Strict accuracy by token budget:

| Budget | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| MIT's edit | 2.7% | 6.7% | 13.4% | 27.5% | 52.1% |
| Ours, 1,398 | 36.2% | 39.5% | 42.2% | 45.2% | 45.7% |
| MIT + ours, full strength | **40.2%** | **43.6%** | **46.4%** | 48.9% | 49.7% |
| MIT + ours, half strength | 27.2% | 35.3% | 43.8% | **54.0%** | **62.2%** |

- **Our run reproduces MIT's.** 32.0% of answers run out of tokens, against 31.6% in their re-run.
- **Each edit wins in its own regime.** Ours stops early: half its answers use under 1,200 tokens, and it is near its best by 4k. MIT's thinks long: its median answer is 23,000 tokens. Ours wins at every budget up to 16k; MIT's passes it between 16k and 32k.
- **They fail in opposite ways.** When MIT's edit finishes, it is right 77% of the time, but a third of its answers run out. Ours finishes 88% of the time but is right on only 53% of those.
- **Together they win.** At full strength our edit cuts MIT's thinking short (median 1,068 tokens). At half strength the combined model keeps the long thinking and stops when done: 92% of answers finish, and it scores 62.2%. Our half-strength edit alone scores only 46.7%, so the gain needs both edits.

## 8. Limitations

One model and one training domain (math). Two samples per setting (MIT uses four). The 32k comparison covers MATH-500 only. The strength of our edit in the combination was chosen from three values (1, ½, ¼) on MATH-500 itself, with no held-out set; both weaker strengths beat MIT's edit by 7.7 points or more, but 62.2% is the best of three. The bias account is tested on the four-column edit, and we cannot yet say why per-layer placement matters. Total cloud cost was about $49.

## 9. Conclusion

A small model's thinking loop can be fixed by gradient descent on as few as 140 weights, in minutes, from its own answers. No frontier model is needed for that. In this bias-free architecture the fix is a constant vector hidden in always-on channels. The frontier model's edit adds something ours lacks: given 32k tokens, it reasons better. The two are not rivals. Stacked, with ours at half strength, they reach 62.2%, 10 points above MIT's edit and 15 above ours, using half the tokens. Weight edits found in different ways can be combined, and claims about any of them should be measured at more than one token budget, against random-weight controls and the thinking-off score.

*Experiment code and drafting were assisted by Claude (Anthropic); the author designed the study, checked the results and takes responsibility for the content. Base model: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).*
