# Weight Engineering

This project made a 0.8B-parameter language model better at fixing code as an agent. All of the work ran on one 16 GB laptop.

- Model: Qwen3.5-0.8B
- Hardware: one Apple M2 with 16 GB of shared memory; no cloud GPUs


## Results

The best checkpoint is a LoRA adapter called v3. Every run used greedy decoding with thinking off, on the same fixed test sets. Each p-value comes from a two-sided paired sign test against the base model, counted over the items where the two models disagree.

| Benchmark | Base | v3 | Change | p |
|---|---|---|---|---|
| Agent bug-fixing (40 tasks) | 9/40 (22.5%) | 20/40 (50.0%) | 2.2x | 0.007 |
| Agent, all tasks (80) | 19/80 (23.8%) | 30/80 (37.5%) | +13.7 points | 0.027 |
| Agent implementation (40) | 25.0% | 25.0% | none | 1.0 |
| HumanEval, single-shot (164) | 33.5% | 34.8% | +1.3 points | 0.85 |
| MATH-500 (first 100) | 40% | 39% | -1 point | 1.0 |

The fine-tuned model solves more than twice as many agent bug-fixing tasks as the base model. In a successful run it searches its memory, reads the failing test and the code, runs the code, fixes the bug and runs the test again. It gained this skill without losing single-shot coding ability; the first attempt, v1, lost about 10 points there.

The agent benchmark is built from HumanEval, which was never used for training. The full official HumanEval tests grade each task, and the agent never sees them.

### Inference settings alone

Before any training, changing only the decoding settings tripled MATH accuracy with the same weights:

| Setting | MATH-500 (100) | HumanEval (164) |
|---|---|---|
| Model-card sampling, thinking on | 13% | not run |
| Model-card sampling, thinking off | 27% | 18.3% |
| Greedy, thinking off | 40% | 33.5% |

With thinking on, the model never closed its reasoning block. All 100 MATH answers hit the 4,096-token limit, with an average of 23 "Wait"s each. The thinking-off runs scored higher even though their limits were smaller: 2,048 tokens for MATH and 1,024 for HumanEval.

### Adapter strength trades one skill for another

These rows score a single adapter, v1, with its update scaled at load time. Scale 0 is the base model.

| v1 scale | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| HumanEval | 33.5% | 32.3% | 28.0% | 26.2% | 23.2% |
| MATH-500 | 40% | 42% | 49% | 43% | 44% |
| Agent (80) | 23.8% | - | 26.3% | - | 37.5% |

At half strength, MATH reached 49%, the highest MATH score in the project. That is 9 points above the base model, but the difference is not significant (p = 0.12).

## The weight edit

The project began as a reproduction of "Sparks of Weight-Space Programming" (MIT CSAIL, November 2026). That method changes a small number of MLP weights directly instead of training the model.

- **Finding the direction.** I looked for a "self-doubt" direction: the direction in the MLP output that appears just before the model writes "Wait". The comparison had to use matched control tokens. Without them it produced a false signal, even at layer 0.
- **The edit.** The direction was removed from 64 down_proj columns in each of layers 14 and 16. That changes 131,072 weights, under 0.02% of the model, and the edit is stored as a 264 KB patch file instead of a full model copy.
- **Effect on doubt.** On 24 MATH problems, "Wait"s per answer fell from 23.8 to 11.7 (-51%). At double strength they fell to 2.5 (-89%).
- **Effect on accuracy.** Accuracy stayed at 4/24, because the model kept doubting in other words. Adding layer 22 broke the model: it averaged 142 "Wait"s per answer. A broader edit, targeting doubt versus commitment, dropped accuracy to 0/24.

The edit gave precise control over what the model wrote, but in this setting it did not make the model more accurate. The project therefore moved to fine-tuning.

## How it was built

1. **Baseline.** I measured MATH-500, HumanEval and a small agent suite, and found that the model's thinking never ended.
2. **Weight edits.** Described above: precise control over the model's wording, but no accuracy gain.
3. **Inference.** Turning thinking off and decoding greedily gave the largest gain that required no training.
4. **Data.** The training set has 1,401 examples, none taken from the test sets.
   - **Self-taught answers.** For each problem, the model's shortest attempt that was verified correct: 759 from 2,000 MATH training problems and 372 from 974 MBPP problems. Code was checked by running its tests, and math answers with math-verify. Keeping the shortest correct answer works directly against the model's habit of rambling.
   - **Authored agent runs.** Claude acted as the agent inside the same harness the small model uses. Every tool call ran for real, and a run was kept only if the hidden tests passed. There are 270 runs:
     - 100 bug fixes;
     - 100 implementations;
     - 70 sessions from two-session projects, where the second session can only succeed by reading notes saved in the first;
     - of all these, 42 also show recovery from a deliberately wrong first attempt.
5. **Training.** The standard trainer, `mlx_lm lora`, runs out of memory on this model, so I wrote a custom training loop with three changes:
   - The linear-attention layers use a chunked gated-delta recurrence (`wsp/chunked_delta.py`). It replaces a per-token loop that needs more than 10 GB of autograd state per layer at 2,500 tokens. Its outputs, final state and gradients were checked against the reference implementation.
   - Logits over the 248k-token vocabulary are computed only at positions that are trained on.
   - All the assistant turns in an agent run train in a single pass. The loss is the same as training one example per turn, at one sixth of the compute.

   Training peaked at 4.9 GB of memory. The LoRA adapter has rank 16 across all 24 layers, with 10.8M trainable parameters, trained for 666 steps over 2 epochs.
6. **Iteration.**
   - v1 (peak learning rate 2e-5) raised the agent score from 23.8% to 37.5%. Single-shot HumanEval fell from 33.5% to 23.2% (p = 0.01).
   - v2 used new data at the same learning rate and was worse on every measure: HumanEval 17.7%, MATH 34%, agent 32.5%.
   - The diagnosis was that the update was too large. The earliest checkpoints were the worst (v2 at step 111 scored 7.9% on HumanEval), and scaling the v1 update down brought coding ability back.
   - v3 trained on the v1 data with a 4x lower peak learning rate (5e-6) and a longer warmup. It kept the full agent gain and lost no coding ability.

## Limitations

- Implementation tasks did not improve (25% before and after), and v3 does not improve MATH.
- v3 edits the visible test file in 9 of the 80 agent tasks; the base model does so in 1. Separate hidden tests do the grading, so these edits cannot raise the score, but the habit is unwanted.
- On the 6 hand-made agent tasks, the base model and v1 both scored 0/6. v3 was not run on them.
- Each configuration was trained once, with one seed, and the test sets have 80 to 164 items. Differences smaller than about 10 points are within noise.
