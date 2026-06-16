# PufferLib Native Integration Path for IsaacLab

**Author:** ClawLabby
**Date:** 2026-06-11
**Status:** Investigation complete; smallest IsaacLab-side bridge is clear, but PufferLib needs an external-vector hook before this can become a fully native backend

## Summary

The lowest-change IsaacLab path is not to port IsaacLab tasks into PufferLib
Ocean `binding.c` files. It is to keep IsaacLab owning the environment and expose
an Ocean-like GPU vector interface: stable observation, reward, terminal, and
action buffers plus `reset`, `gpu_step`, `log`, and `close` callbacks.

That path is especially reasonable for direct pure-Warp IsaacLab environments,
because those environments already maintain flat GPU buffers that are much closer
to PufferLib's preferred layout than manager-based Isaac Sim environments.

The blocker is on the PufferLib side. In the inspected PufferLib 4.0 tree, the
native CUDA trainer creates and owns a compiled Ocean `StaticVec` internally.
There is no public hook for an externally-owned GPU vector environment. Until
that hook exists, IsaacLab can provide a small adapter and can use PufferLib's
Python/PyTorch-facing runner, but it cannot use the fully native PufferLib CUDA
trainer without modifying PufferLib.

## Branches and Local Experiment

Local branches created for this follow-up:

- IsaacLab: `investigation/pufferlib-native-warp`
- PufferLib: `investigation/isaaclab-native-warp`
- Notes: existing branch `investigation/pufferlib-isaaclab-learning`

IsaacLab-side prototype edits:

- `source/isaaclab_rl/isaaclab_rl/pufferlib/warp_bridge.py`
  - Tightened the bridge to look more like PufferLib's vector protocol:
    `gpu=True`, `action_mask_size=0`, `gpu_action_mask_ptr=0`, and `render()`.
  - Corrected continuous-action metadata to PufferLib's convention:
    `act_sizes=[1] * action_dim`, where action size `1` marks a continuous
    action head.
  - Added persistent Warp-buffer detection for manager/direct Warp envs. The
    bridge now uses persistent observation and reward tensors directly when the
    env exposes them, keeps a stable action buffer, and reports buffer-source
    diagnostics.
- `source/isaaclab_rl/isaaclab_rl/pufferlib/native.py`
  - Added `build_pufferlib_args(...)` to synthesize the PufferLib config dict for
    an externally-owned IsaacLab vector environment.
  - Added `probe_native_backend(...)` to report whether local `pufferlib._C` is
    importable, whether it exposes the native trainer, and whether it exposes an
    external-vector hook. For IsaacLab, `native_train_available` now requires
    an external-vector hook, not merely a native Ocean `create_pufferl`.
  - Added `create_torch_pufferl_from_isaaclab(...)` for the currently possible
    hybrid path: PufferLib's Python `PuffeRL` wrapper consuming IsaacLab's
    external vector protocol.
  - Added `create_native_pufferl_from_isaaclab(...)`, which deliberately fails
    with a precise missing-hook error unless PufferLib grows
    `create_external_pufferl` or an equivalent.
- `source/isaaclab_rl/isaaclab_rl/pufferlib/__init__.py`
  - Exported the new helpers.

Validation performed:

- `py_compile` on the touched IsaacLab PufferLib files.
- Import/probe from the IsaacLab venv with `PYTHONPATH=source/isaaclab_rl:/home/horde/claw/research/PufferLib`.
- CUDA fake-env check of `PufferWarpBridge` using persistent manager-like tensors:
  - observation pointer remained stable,
  - reward pointer remained stable,
  - action pointer remained stable,
  - continuous action sizes reported as `[1, 1, 1]` for a 3D fake action space,
  - `zero_copy_obs=True`,
  - `zero_copy_rewards=True`.

PufferLib build/probe:

```bash
source /home/horde/claw/git/IsaacLab/.venv/bin/activate
CC=gcc CXX=g++ ./build.sh squared_continuous --cpu
```

Local changes to make that CPU build possible:

- `build.sh` now drops clang-only warning flags when `CC=gcc`.
- `build.sh` uses `-lgomp` instead of LLVM `-lomp5` when `CC=gcc`.

The CPU extension now builds and imports:

- `_C.env_name`: `squared_continuous`
- `_C.gpu`: `0`
- `_C.precision_bytes`: `4`
- `_C.create_vec`: present
- `_C.create_pufferl`: absent
- external-vector hook: absent

Probe result after installing `pybind11` and `rich_argparse` into the IsaacLab
venv:

```python
{
  "can_import_c": True,
  "can_import_torch_pufferl": True,
  "c_env_name": "squared_continuous",
  "c_gpu": 0,
  "c_precision_bytes": 4,
  "has_create_vec": True,
  "has_create_pufferl": False,
  "has_external_vec_hook": False,
  "external_vec_hook": None,
  "native_train_available": False,
  "error": None,
}
```

Attempted native CUDA build remains blocked because the compiler toolchain is
not available on `PATH`:

- `clang` was not available.
- `nvcc` was not available on `PATH` or under `/usr/local/cuda/bin`.

The installed runtime stack is enough for IsaacLab training (`torch
2.10.0+cu128`, CUDA available with 4 devices), but not enough to compile
PufferLib's native CUDA extension here.

## What PufferLib Exposes Today

PufferLib has a useful vector protocol already visible from Python. Its
`VecEnv` wrapper exposes:

- `total_agents`
- `obs_size`
- `num_atns`
- `act_sizes`
- `obs_dtype`
- `gpu_obs_ptr`
- `gpu_rewards_ptr`
- `gpu_terminals_ptr`
- `reset()`
- `gpu_step(actions_ptr)`
- `log()`
- `close()`

PufferLib's Python `PuffeRL` wrapper can consume an object with that shape. That
is why a small IsaacLab `PufferWarpBridge` is viable as a near-term experiment.

The fully native path is different. `_C.create_pufferl(args)` calls into the CUDA
trainer, and that trainer currently creates its own Ocean `StaticVec`. The
relevant coupling is:

- `src/bindings.cu`: parses `args` and calls `create_pufferl_impl(...)`.
- `src/pufferlib.cu`: `create_pufferl_impl(...)` calls `create_environments(...)`.
- `src/pufferlib.cu`: `create_environments(...)` calls `create_static_vec(...)`.
- `src/vecenv.h`: `StaticVec` owns env buffers, streams, worker threads, reset,
  step, render, and log functions.

That design is excellent for compiled Ocean environments, but it does not yet
have an attachment point for IsaacLab-owned GPU buffers and callbacks.

## Minimum PufferLib Hook Needed

The smallest useful PufferLib-side hook would look conceptually like:

```text
create_external_pufferl(args, external_vec)
```

Where `external_vec` provides:

- Metadata:
  - `total_agents`
  - `obs_size`
  - `num_atns`
  - `act_sizes`
  - `obs_dtype`
  - optional action mask size
- GPU pointers:
  - observations
  - rewards
  - terminals
  - actions
  - optional action mask
- Callbacks:
  - `reset()`
  - `gpu_step(actions_ptr)`
  - `log()`
  - `close()`
- Ownership rules:
  - PufferLib must not free IsaacLab-owned buffers.
  - PufferLib must know whether it may CUDA-graph-capture rollout callbacks.
  - The hook needs a clear stream synchronization contract.

The important implementation detail is to refactor native trainer setup so it
can receive an already-populated `EnvBuf`/external vector instead of always
constructing a `StaticVec`.

A pure `StaticVec*` injection is not enough by itself. The native rollout path
currently steps compiled Ocean environments through PufferLib-owned worker
threads; IsaacLab needs an external rollout branch that lets native PufferLib
write actions, synchronize, then call back into `external_vec.gpu_step(action_ptr)`.

## Pure-Warp IsaacLab Scope

Restricting this to direct pure-Warp environments is the right first scope.

Why it helps:

- Direct Warp tasks already organize observations, actions, rewards, resets, and
  episode state as GPU-resident buffers.
- Flat `policy` observations map naturally to PufferLib's single observation
  tensor.
- Continuous `Box` actions with one agent per env match the current locomotion
  tasks and avoid action-mask or multi-agent complications.
- IsaacLab-side changes can stay near the adapter boundary instead of reaching
  into manager/event/recorder paths.

What it does not solve:

- PufferLib still lacks the external-vector hook.
- Native PufferLib still assumes a single flat observation stream. It does not
  have a public asymmetric actor/critic observation API.
- IsaacLab distinguishes termination and timeouts; PufferLib's native protocol
  has one terminal float buffer.
- Native PufferLib reward handling includes clamping behavior that must be
  audited before comparing locomotion reward curves.

## Value Assessment

There is a tangible potential value, but it is not yet proven by a runnable
native IsaacLab benchmark.

Possible value:

- Reuse PufferLib native rollout/training kernels, CUDA graph capture, and
  compiled optimizer path against IsaacLab's Warp environments.
- Avoid duplicating IsaacLab task physics or task logic in Ocean C.
- Keep IsaacLab integration small enough to maintain as an experimental backend.
- Give pure-Warp IsaacLab tasks a non-RSL-RL/non-PyTorch-heavy backend path if
  PufferLib external vectors become supported.

Current non-value:

- The already-tested pure-PyTorch Puffer-style runner learned, but was slightly
  slower than RSL-RL in matched state-observation locomotion runs.
- The Puffer MinGRU/Muon profile did not learn in the short locomotion runs.
- No compiled native PufferLib IsaacLab run exists yet.

Update from the June 16 camera probes: rendered-pixel RL changes the practical
value assessment. On `Isaac-Cartpole-Camera-Direct` with Newton/Warp RGB camera
observations, the optimized Puffer-style PyTorch backend was materially faster
than RSL-RL because PPO update time dropped enough to offset slower
rollout/render time. That does not remove the native integration blocker, but it
does make a tensor-native Hermes/IsaacLab backend worth prototyping for
pixel-heavy tasks.

## Recommendation

Do not implement an IsaacLab Ocean `binding.c` port. That would duplicate task
logic, cut around IsaacLab's main value, and still fight PufferLib's
compile-one-env model.

The best next experiment is:

1. Add a local PufferLib `create_external_pufferl` hook that accepts externally
   owned GPU buffers and callbacks.
2. Keep IsaacLab support limited to direct pure-Warp, single-agent,
   continuous-action tasks.
3. Expose existing Warp buffers directly where possible, rather than copying
   observations, rewards, and done flags into new Torch buffers.
4. Benchmark env stepping separately from PPO update time before making any
   end-to-end speed claim.
5. Only after the external-vector hook works, compare against RSL-RL with matched
   task, env count, horizon, model size, PPO epochs, minibatches, logging,
   synchronization, timeout handling, and reward metrics.

This is worth a small PufferLib-side prototype. It is not yet worth a large
IsaacLab-side integration.
