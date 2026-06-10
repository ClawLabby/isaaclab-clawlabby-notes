# PufferLib IsaacLab Full-Learning Evaluation

**Author:** ClawLabby
**Date:** 2026-06-10
**Status:** Matched full-learning comparison completed; prototype integration is useful for experimentation but not a demonstrated speed win

## Summary

PufferLib was evaluated as an IsaacLab reinforcement-learning backend using real
learning runs, not one-iteration smoke timings. The important result is narrow:
a pure-PyTorch Puffer-style backend can match RSL-RL learning quality on both a
simple locomotion task and the rough-terrain quadruped task when the PPO
configuration is matched, but it was slightly slower in this prototype.

The earlier 4096-env smoke comparison was misleading. It matched environment
count and rollout length, but not trainer work. The final comparison uses a
matched PPO profile that copies the task's RSL-RL actor/critic MLP sizes,
optimizer, PPO epochs, minibatches, horizon, discounting, GAE lambda, clipping,
entropy/value coefficients, initial action std, adaptive KL schedule, action
clipping, and timeout bootstrapping.

Bottom line:

- `Isaac-Ant-v0`: matched Puffer estimated return `151.4` vs RSL-RL return
  `152.2`; matched Puffer `260k SPS` vs RSL-RL `286k FPS/SPS`.
- `Isaac-Velocity-Rough-Anymal-C-v0`: matched Puffer estimated return `17.0`
  vs RSL-RL return `16.9`; matched Puffer `68.7k SPS` vs RSL-RL `71.9k FPS/SPS`.
- The Puffer-specific MinGRU/Muon exploratory profile did not learn in the
  short 300-iteration runs.
- Current value: experimental backend flexibility, config-controlled algorithm
  prototyping, and a place to test PufferLib ideas inside IsaacLab.
- Not shown: a native PufferLib/Ocean speedup. PufferLib 4.0 expects compiled
  Ocean-style C environments, so IsaacLab would need a native bridge or
  external-vector hook before it can use the headline native backend path.

## Repositories and Baselines

PufferLib inspected:

- Repository: `PufferAI/PufferLib`
- Branch: `4.0`
- Commit inspected: `9836f0d2`

IsaacLab prototype checkout:

- Repository: `ClawLabby/IsaacLab`
- Base branch at time of work: `reference/warp-replicator-compat-shim`
- Base commit: `f89ec0a6544` (`Add Warp compatibility shim for Replicator`)

The prototype was scratch work in the IsaacLab checkout. At the time these notes
were written, the backend files were local/untracked rather than committed to an
IsaacLab branch.

## Why the First Comparison Was Wrong

The first 4096-env check compared:

- Puffer-style PoC: `4096 envs * 24 horizon = 98,304 transitions`
- RSL-RL baseline: `4096 envs * 24 num_steps_per_env = 98,304 transitions`

The environment count was not the problem. The trainer workloads were different:

- The PoC used a lightweight recurrent policy, Muon, one update epoch, three
  large minibatches, and thin manual rollout storage.
- RSL-RL used separate actor and critic MLPs, Adam, five learning epochs, four
  minibatches per epoch, `TensorDict`/`RolloutStorage`, timeout bootstrapping,
  NaN checks, normalization hooks, and logger bookkeeping.
- Timing boundaries also lacked explicit CUDA synchronization in the initial
  smoke run, so GPU work could smear across rollout and train timing.

That run was useful only as an integration smoke test. It was not evidence that
PufferLib made IsaacLab rollout twice as fast.

## Prototype Integration

The useful prototype is an experimental pure-PyTorch backend under
`isaaclab_rl.pufferlib` plus a `scripts/reinforcement_learning/pufferlib/train.py`
entrypoint.

Files created in the IsaacLab checkout:

- `source/isaaclab_rl/isaaclab_rl/pufferlib/__init__.py`
- `source/isaaclab_rl/isaaclab_rl/pufferlib/puffer_cfg.py`
- `source/isaaclab_rl/isaaclab_rl/pufferlib/runner.py`
- `source/isaaclab_rl/isaaclab_rl/pufferlib/vecenv_wrapper.py`
- `source/isaaclab_rl/isaaclab_rl/pufferlib/warp_bridge.py`
- `scripts/reinforcement_learning/pufferlib/train.py`

The backend has two explicit profiles:

- `matched_rsl`: the actual comparison path. It builds separate Gaussian actor
  and critic MLPs and copies the RSL-RL task config.
- `puffer_mingru`: the experimental Puffer-specific path. It uses PufferLib
  `DefaultEncoder`, `MinGRU`, `DefaultDecoder`, `Policy`, and Muon. It is kept
  separate so it cannot be confused with an apples-to-apples PPO comparison.

Matched features implemented:

- RSL-RL actor and critic hidden dimensions.
- RSL-RL activation, distribution std init, scalar/log std mode, and action
  clipping.
- PPO horizon, learning epochs, minibatches, Adam learning rate, adaptive KL,
  desired KL, clipped policy/value losses, value and entropy coefficients,
  gamma, lambda, and gradient clipping.
- Timeout bootstrapping from `extras["time_outs"]`.
- Actor and critic observation group handling.
- Global or per-minibatch advantage normalization control.
- Optional finite-gradient and non-finite tensor guards.
- CUDA synchronization around rollout and train timing.
- JSONL metrics and TensorBoard scalar logging.
- IsaacLab reward-term logging from `extras["log"]` when present.

The `PufferWarpBridge` file is only an ABI sketch. It exposes Ocean-shaped
pointer/callback concepts such as observation/action pointers, reset, GPU step,
log, and close. It is not a compiled native bridge.

## Benchmark Harness

The long-run launcher executed this matrix:

- RSL-RL and matched Puffer on `Isaac-Ant-v0`
  - 4096 envs
  - 1000 iterations
  - seeds 42 and 43
- RSL-RL and matched Puffer on `Isaac-Velocity-Rough-Anymal-C-v0`
  - 4096 envs
  - 1500 iterations
  - seeds 42 and 43
- Puffer MinGRU/Muon exploratory runs on both tasks
  - 4096 envs
  - 300 iterations
  - seed 42

The raw local run root was named:

```text
pufferlib-isaaclab-eval/learning_runs/full_20260609T161900Z
```

This note intentionally does not commit local absolute paths or raw log
directories. A sanitized result payload is included in
[`results-summary.json`](results-summary.json).

## Aggregate Results

| Task | Backend/profile | Seeds | Return metric | Mean return | Mean throughput |
| --- | --- | ---: | --- | ---: | ---: |
| `Isaac-Ant-v0` | RSL-RL | 42, 43 | `Train/mean_reward` | `152.2` | `286.2k` |
| `Isaac-Ant-v0` | Puffer `matched_rsl` | 42, 43 | estimated episodic return | `151.4` | `260.3k` |
| `Isaac-Velocity-Rough-Anymal-C-v0` | RSL-RL | 42, 43 | `Train/mean_reward` | `16.9` | `71.9k` |
| `Isaac-Velocity-Rough-Anymal-C-v0` | Puffer `matched_rsl` | 42, 43 | estimated episodic return | `17.0` | `68.7k` |
| `Isaac-Ant-v0` | Puffer `puffer_mingru` | 42 | estimated episodic return | `0.8` | `232.0k` |
| `Isaac-Velocity-Rough-Anymal-C-v0` | Puffer `puffer_mingru` | 42 | estimated episodic return | `-2.3` | `49.6k` |

For Puffer runs, `mean_reward` is rollout mean reward per environment step.
RSL-RL's `Train/mean_reward` is an episodic return metric. The summary therefore
estimates Puffer episodic return as:

```text
last_50_mean_reward / last_50_mean_done_rate
```

Equivalently:

```text
last_50_mean_reward * estimated_episode_length
```

This is a reasonable comparison bridge for this experiment, but it is not as
clean as making both backends log the exact same episodic-return accumulator.

## Per-Seed Results

### Ant

| Seed | Backend/profile | Return | Throughput | Notes |
| ---: | --- | ---: | ---: | --- |
| 42 | RSL-RL | `151.69` | `287.2k` | TensorBoard `Train/mean_reward` and `Perf/total_fps` |
| 42 | Puffer `matched_rsl` | `134.84` | `260.2k` | Estimated return from last-50 reward/done |
| 43 | RSL-RL | `152.67` | `285.2k` | TensorBoard `Train/mean_reward` and `Perf/total_fps` |
| 43 | Puffer `matched_rsl` | `167.91` | `260.3k` | Estimated return from last-50 reward/done |
| 42 | Puffer `puffer_mingru` | `0.76` | `232.0k` | 300 iterations only; did not learn |

The matched Puffer seed spread is larger than RSL-RL's seed spread, but the
two-seed mean is essentially the same. On throughput, the pure-PyTorch Puffer
runner was about 9% slower than RSL-RL in the final iteration metrics.

### Rough Anymal-C

| Seed | Backend/profile | Return | Throughput | Notes |
| ---: | --- | ---: | ---: | --- |
| 42 | RSL-RL | `16.21` | `71.3k` | TensorBoard `Train/mean_reward` and `Perf/total_fps` |
| 42 | Puffer `matched_rsl` | `17.21` | `67.7k` | Estimated return from last-50 reward/done |
| 43 | RSL-RL | `17.61` | `72.5k` | TensorBoard `Train/mean_reward` and `Perf/total_fps` |
| 43 | Puffer `matched_rsl` | `16.84` | `69.7k` | Estimated return from last-50 reward/done |
| 42 | Puffer `puffer_mingru` | `-2.31` | `49.6k` | 300 iterations only; did not learn |

On rough locomotion, matched Puffer reached the same reward scale as RSL-RL. It
was about 4-5% slower on final-iteration throughput.

## Timing Interpretation

The final timing numbers are more reliable than the first smoke comparison
because the Puffer runner explicitly synchronizes CUDA around rollout and train
timers. Still, the timing comparison is not a full systems benchmark:

- RSL-RL and the Puffer prototype do not expose identical metric definitions.
- Puffer's total wall-clock training time was not extracted into the summary the
  same way RSL-RL's `Training time: ... seconds` line was extracted.
- The final numbers are from two seeds and one machine, not a sweep over
  machines, task variants, or policy sizes.
- Throughput is not the only objective; reward scale and stability matter more
  for backend adoption.

The useful conclusion is not "Puffer is faster." The useful conclusion is:
under a matched PPO setup, the experimental runner can learn the same tasks to
the same reward range, so it is valid as a testbed for further backend ideas.

## PufferLib-Specific Path

The `puffer_mingru`/Muon path was included because it is closer to the kind of
value PufferLib could bring beyond a plain PPO clone. It did not work out of the
box:

- Ant after 300 iterations: estimated return `0.76`.
- Rough Anymal-C after 300 iterations: estimated return `-2.31`.

Likely causes:

- The MinGRU policy and Muon optimizer were not tuned for these continuous
  IsaacLab locomotion tasks.
- The profile uses a different value/policy setup than RSL-RL, so matched PPO
  hyperparameters are not necessarily appropriate.
- Continuous-control action distribution details and recurrent-state handling
  need dedicated tuning.

This should remain labeled exploratory until it can show learning curves on at
least Ant before being used on rough quadruped locomotion.

## Native PufferLib Integration Gap

PufferLib 4.0's strongest performance story is its native backend: static
buffers, CUDA graph tracing, fused kernels, and compiled Ocean C environments.
The IsaacLab prototype does not use that path.

Current gap:

- PufferLib 4.0 constructs environments through compiled Ocean modules.
- IsaacLab environments live behind Python/Gymnasium-style vector envs backed by
  PhysX/Isaac Sim tensors.
- The old Python vectorization/emulation path referenced by some examples is not
  present as a straightforward package API in the inspected 4.0 tree.

A native integration would need one of these:

- A PufferLib external-vector environment hook that can consume IsaacLab GPU
  tensors without pretending to be an Ocean module.
- A small C++/CUDA shim that presents an Ocean-compatible `StaticVec`-style ABI
  while forwarding step/reset/log callbacks to IsaacLab.
- A larger rework that moves enough IsaacLab rollout state into PufferLib-owned
  static buffers to make CUDA graph capture and fused update kernels meaningful.

Until one of those exists, the prototype should be described as
"Puffer-style/Puffer-inspired pure PyTorch backend," not "native PufferLib
backend."

## Recommended Next Work

1. Commit the IsaacLab prototype to a dedicated branch if it should be preserved
   beyond this evaluation. Keep it explicitly experimental.
2. Make RSL-RL and Puffer log the same episodic-return accumulator so the
   comparison no longer needs reward/done-rate conversion.
3. Add a clean wall-clock summary for Puffer runs, matching the RSL-RL
   `Training time` extraction.
4. Run a small tuning pass for `puffer_mingru` on Ant only. Do not spend rough
   Anymal-C time until Ant learning is nontrivial.
5. If PufferLib remains strategically interesting, prototype the smallest
   external-vector/native bridge and benchmark env stepping separately from PPO
   update time.
6. Treat any future speed claims as invalid unless task, env count, horizon,
   model size, optimizer, epochs, minibatches, logging, checks, CUDA
   synchronization, and reward metrics are explicitly matched.
