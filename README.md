# Weight Engineering

This project made a 0.8B-parameter language model better at fixing code as an agent. All of the work ran on one 16 GB laptop.

- Model: Qwen3.5-0.8B
- Hardware: one Apple M2 with 16 GB of shared memory; no cloud GPUs
- Built: 4 to 7 November 2026

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

## Code

| Path | Purpose |
|---|---|
| `wsp/model.py`, `wsp/mlx_backend.py` | Generation with PyTorch (MPS) and MLX; the MLX path is about 5x faster, supports LoRA adapters and caps memory use |
| `wsp/evals/` | MATH-500 scoring with math-verify; HumanEval scoring by running the tests |
| `wsp/loops.py` | Detects repetitive output by its compression ratio |
| `wsp/capture.py`, `wsp/edit.py` | Activation capture; weight edits stored as small patch files |
| `scripts/wait_direction.py`, `scripts/find_and_edit.py` | Find the doubt direction and the repetition direction, and write patches |
| `scripts/eval.py` | Run one evaluation; write per-item results and a summary |
| `scripts/selfgen.py` | Self-taught data by rejection sampling (resumable) |
| `scripts/build_dataset.py` | Merge the self-taught data and the agent runs into a training set |
| `scripts/train.py`, `wsp/chunked_delta.py` | LoRA training that fits in memory |
| `scripts/test_chunked_delta.py` | Correctness test for the chunked recurrence |
| `agent/harness.py` | Coding agent with file, shell and memory tools |
| `agent/memory_store.py` | Memory directory of Markdown notes, with an index file and BM25 search |
| `agent/eval_agent.py` | Agent benchmark graded by hidden tests |
| `agent/make_tasks.py`, `agent/make_he_tasks.py`, `agent/gen_train_tasks.py` | Build the hand-made, HumanEval and MBPP task sets; each task is checked to fail at the start and to pass with a reference solution |
| `agent/author_v1.py` | Produced the authored runs that trained v3 |
| `agent/author.py` | Later version that produced the v2 runs |
| `agent/teacher.py` | The same job done through the Claude API instead (optional) |

## Running

Requirements: Apple Silicon and Python 3.11, with these packages:

- mlx 0.32
- mlx-lm 0.32
- torch 2.15
- transformers 5.22
- datasets 5.1
- math-verify 0.9
- safetensors 0.8
- anthropic 2.5 (only for `agent/teacher.py`)

```sh
python agent/make_he_tasks.py                      # build the 80-task agent benchmark
python scripts/eval.py --task code --n 164 --no-thinking --greedy --max-new-tokens 1024 --out results/base_code.jsonl
python agent/eval_agent.py --tasks agent/tasks_he --out results/he_agent_base.json

python scripts/selfgen.py --task code --k 4        # self-taught data
python scripts/selfgen.py --task math --n 2000 --k 2
python agent/gen_train_tasks.py && python agent/author_v1.py
python scripts/build_dataset.py --out data/mix
python scripts/train.py -c lora_v3.yaml
python agent/eval_agent.py --tasks agent/tasks_he --adapter adapters/v3 --out results/he_agent_v3.json
```

Contents of `lora_v3.yaml`:

```yaml
model: Qwen/Qwen3.5-0.8B
data: data/mix
num_layers: -1
lora_parameters: {rank: 16, dropout: 0.0, scale: 20.0}
epochs: 2
sequences_per_step: 4
lr_schedule: {name: cosine_decay, warmup: 60, warmup_init: 1.0e-7, arguments: [5.0e-6, auto, 5.0e-7]}
max_seq_length: 2560
grad_checkpoint: true
report_every: 10
eval_every: 111
save_every: 111
adapter_path: adapters/v3
log: logs/train_v3.jsonl
seed: 0
```

On a 16 GB machine, run one model job at a time. Generation is capped at 5 GB of memory, and training peaks at 4.9 GB.

This folder contains no API keys or credentials. If you use `agent/teacher.py`, it reads `ANTHROPIC_API_KEY` from your environment.
