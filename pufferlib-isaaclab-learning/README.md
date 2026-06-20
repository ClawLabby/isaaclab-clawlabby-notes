# PufferLib / IsaacLab Native External-Vector Evaluation

Date: 2026-06-20

## Bottom Line

The previous 2026-06-18 conclusion that PufferLib lacked the required compiled
external-vector hook is now superseded. The hook has been implemented locally and
validated against both synthetic CUDA tensors and a real IsaacLab Newton/MJWarp
environment.

What is now true:

- `pufferlib._C` exposes `create_external_pufferl`.
- `pufferlib._C` exposes `external_rollouts`.
- `ExternalGPUVec.create_native_pufferl(args)` routes an externally owned CUDA
  vector environment into the compiled PufferLib backend.
- The compiled backend can borrow external CUDA observation, reward, terminal,
  and action-mask buffers without owning or freeing them.
- The compiled backend forwards its CUDA action buffer pointer into a Python
  callback, allowing IsaacLab to apply native PufferLib actions without copying
  through the host.
- IsaacLab's experimental `PufferWarpBridge` can now keep the bridge and callback
  alive through a `NativeExternalPuffeRL` wrapper and run native rollout, train,
  log, and close.

What this still does not prove:

- It does not yet prove that compiled PufferLib-native IsaacLab training is faster
  than RSL-RL or the tensor-native Puffer-style runner.
- The successful IsaacLab run was a scoped smoke test: `Isaac-Ant-v0`,
  Newton/MJWarp, 16 environments, horizon 4, one PPO iteration, small network.
- A fair speed claim still needs a real benchmark harness with matched
  hyperparameters, long enough runs, and comparable policy sizes.

The honest claim is now narrower but important: the missing ABI no longer blocks
evaluation. PufferLib can consume IsaacLab-owned CUDA buffers through a compiled
external-vector path, and that path runs through native rollout and native train
on an IsaacLab Newton/MJWarp task.

## Implementation Summary

PufferLib checkout:
`/home/horde/claw/research/PufferLib`

Branch:
`investigation/isaaclab-native-warp`

Base HEAD during this work:
`9836f0d2e78889c1aaf189c04d161b6fc61a9386`

Key PufferLib changes:

- Added external-vector ownership state to `StaticVec` in `src/vecenv.h`.
- Made `static_vec_close` skip internally owned Ocean env teardown for external
  vectors.
- Made external-vector close free only PufferLib's internally allocated action
  buffer, not externally supplied IsaacLab buffers.
- Added `ExternalVecSpec` in `src/pufferlib.cu`.
- Added `create_external_environments`.
- Refactored PufferLib creation through a common implementation path that can use
  either internally owned Ocean envs or externally supplied CUDA vector buffers.
- Added `create_external_pufferl_impl`.
- Added Python binding `_C.create_external_pufferl(args, external_vec)`.
- Added Python binding `_C.external_rollouts(pufferl, step_callback)`.
- Added readonly `PuffeRL.external_actions_ptr`.
- Added `ExternalGPUVec.create_native_pufferl(args)`.
- Added `ExternalGPUVec.native_rollouts(pufferl)`.
- Added protocol tests covering native helper routing.
- Fixed a float build overload collision in `src/kernels.cu` where
  `precision_t == float`.
- Fixed native close for `cudagraphs=-1` by guarding null CUDA graph handles.

Important implementation detail:

The compiled Ocean environment is currently used as a kernel and dtype carrier,
not as the owner of the IsaacLab environment shape. The external hook takes
`obs_size`, `num_atns`, `action_mask_size`, `act_sizes`, and CUDA buffer pointers
from the external vector. This matters because IsaacLab tasks do not share Ocean
environment observation or action dimensions.

IsaacLab checkout:
`/home/horde/claw/git/IsaacLab`

Branch:
`investigation/pufferlib-native-warp`

Base HEAD during this work:
`f89ec0a6544c4545bf747ff0ea1561dacb8065d9`

Key IsaacLab changes:

- Updated `isaaclab_rl.pufferlib.warp_bridge.PufferWarpBridge` from an ABI sketch
  into the external-vector bridge used by the native hook.
- Added `obs_elem_size`.
- Updated `isaaclab_rl.pufferlib.native` to probe and use
  `_C.create_external_pufferl`.
- Added `NativeExternalPuffeRL`, which keeps the native `_C` module, raw PufferLib
  object, and IsaacLab bridge alive together.
- Added wrapper methods for `rollouts`, `train`, `log`, `close`, `num_params`,
  and `global_step`.
- Exported `NativeExternalPuffeRL` from `isaaclab_rl.pufferlib`.

## Build Notes

The local machine initially lacked enough CUDA development tooling to build the
compiled PufferLib extension. I installed the missing CUDA 12.6 development
packages and used the Python wheel cuDNN already present in the PufferLib venv.

Local linker shims were created under:

`/home/horde/claw/research/PufferLib/build/liblinks`

Those shims point at:

- `libnccl.so.2`
- `libnvidia-ml.so.1`
- Python wheel `libcudnn.so.9`

Successful build command:

```bash
cd /home/horde/claw/research/PufferLib
CUDA_HOME=/usr/local/cuda-12.6 \
PATH=/usr/local/cuda-12.6/bin:$PATH \
CC=gcc \
CXX=g++ \
LIBRARY_PATH=/home/horde/claw/research/PufferLib/build/liblinks:$LIBRARY_PATH \
LD_LIBRARY_PATH=/home/horde/claw/research/PufferLib/build/liblinks:/home/horde/claw/research/PufferLib/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
uv run ./build.sh drone --float
```

Result:

`Built: pufferlib/_C.cpython-312-x86_64-linux-gnu.so`

The `drone --float` build was used because its compiled tensor dtype is float.
The external hook supplies IsaacLab's real observation and action metadata at
runtime.

## Validation

### 1. PufferLib external-vector tests

Command:

```bash
cd /home/horde/claw/research/PufferLib
uv run --with pytest pytest tests/test_external_gpu_vec.py -q
```

Result:

`5 passed in 3.77s`

### 2. Python compile checks

PufferLib:

```bash
cd /home/horde/claw/research/PufferLib
python -m py_compile pufferlib/external.py
```

IsaacLab:

```bash
cd /home/horde/claw/git/IsaacLab
python -m py_compile \
  source/isaaclab_rl/isaaclab_rl/pufferlib/native.py \
  source/isaaclab_rl/isaaclab_rl/pufferlib/warp_bridge.py \
  source/isaaclab_rl/isaaclab_rl/pufferlib/__init__.py
```

Result: both compile checks passed.

### 3. Standalone native external CUDA smoke

Artifact:

`/home/horde/claw/pufferlib-isaaclab-eval/native_external_hook_20260620/native_external_smoke.log`

Setup:

- Synthetic Torch CUDA buffers.
- `total_agents = 8`
- `obs_size = 13`
- `num_atns = 3`
- `num_buffers = 2`
- `horizon = 4`
- Compiled carrier env: `drone`

Result:

```json
{
  "action_ptr_forwarded": true,
  "compiled_env_name": "drone",
  "global_step_after_rollout": 32,
  "has_create_external_pufferl": true,
  "has_external_rollouts": true,
  "loss_keys": ["clipfrac", "entropy", "kl", "old_kl", "policy", "total", "value"],
  "num_callback_steps": 4,
  "perf_keys": ["eval_env", "eval_gpu", "rollout", "train", "train_forward", "train_misc"],
  "precision_bytes": 4,
  "status": "passed"
}
```

Interpretation: the compiled backend created a native PufferLib object from
external CUDA buffers, ran rollout, called the external step callback once per
horizon step, forwarded the action pointer unchanged, ran native train, logged,
and closed.

### 4. IsaacLab native capability probe

Artifact:

`/home/horde/claw/pufferlib-isaaclab-eval/native_external_hook_20260620/isaaclab_native_probe.log`

Result:

```json
{
  "c_env_name": "drone",
  "c_gpu": 1,
  "c_precision_bytes": 4,
  "can_import_c": true,
  "can_import_torch_pufferl": true,
  "error": null,
  "external_vec_hook": "create_external_pufferl",
  "has_create_pufferl": true,
  "has_create_vec": true,
  "has_external_gpu_vec": true,
  "has_external_vec_hook": true,
  "native_train_available": true
}
```

Interpretation: IsaacLab can import the locally built PufferLib extension, see
the native external-vector hook, and detect that native train is available.

### 5. Real IsaacLab native external-vector smoke

Artifact:

`/home/horde/claw/pufferlib-isaaclab-eval/native_external_hook_20260620/isaaclab_ant_native_external_smoke.log`

Task:

- `Isaac-Ant-v0`
- Physics: `newton_mjwarp`
- Device: `cuda:0`
- Headless: yes
- Environments: 16
- Seed: 45
- Observation group: `policy`, shape `(60,)`
- Action shape: 8
- Horizon: 4
- PPO iterations: 1
- PPO epochs: 1
- Minibatches: 2
- Hidden size: 32
- Hidden layers: 1

Result:

```json
{
  "global_step": 64,
  "loss_keys": ["clipfrac", "entropy", "kl", "old_kl", "policy", "total", "value"],
  "num_envs": 16,
  "num_params": 5288,
  "perf": {
    "eval_env": 0.06238913908600807,
    "eval_gpu": 0.01009225845336914,
    "rollout": 0.07248950004577637,
    "train": 0.0013161280658096075,
    "train_forward": 0.0011839360231533647,
    "train_misc": 0.00013219199900049716
  },
  "status": "passed",
  "task": "Isaac-Ant-v0"
}
```

Interpretation: the compiled PufferLib native external-vector path ran end to end
against a real IsaacLab Newton/MJWarp task: environment creation, bridge setup,
native rollout, callback stepping into IsaacLab, native PPO train, native log,
and close.

Residual note: the run emitted an ignored `SensorBase.__del__` shutdown warning
after the JSON success record. It did not prevent the native rollout/train smoke
from passing.

## Performance Context from Earlier Runs

The previous performance conclusions remain useful, but they evaluate the
tensor-native IsaacLab/Puffer-style runner, not the newly implemented compiled
external-vector hook.

State-observation PPO:

- Run root:
  `/home/horde/claw/pufferlib-isaaclab-eval/learning_runs/full_20260609T161900Z`
- Tasks:
  - `Isaac-Ant-v0`, 4096 envs, 1000 iterations, seeds 42 and 43.
  - `Isaac-Velocity-Rough-Anymal-C-v0`, 4096 envs, 1500 iterations, seeds 42 and
    43.
- Conclusion: matched Puffer-style PyTorch PPO learned in the same broad range
  as RSL-RL but was slower on throughput. This was not a state-task speed win.

Representative state results:

| Task | Seed | RSL reward/return | Puffer est. return | RSL FPS | Puffer FPS | FPS delta |
|---|---:|---:|---:|---:|---:|---:|
| Ant | 42 | 151.69 | 134.84 | 287,157 | 265,179 | -7.7% |
| Ant | 43 | 152.67 | 167.91 | 285,237 | 263,434 | -7.6% |
| Anymal-C Rough | 42 | 16.21 | 17.21 | 71,332 | 68,403 | -4.1% |
| Anymal-C Rough | 43 | 17.61 | 16.84 | 72,456 | 69,955 | -3.5% |

Fresh seed-44 confirmation, 4096 envs and 200 iterations:

| Task | RSL reward/return | Puffer est. return | RSL FPS | Puffer FPS | FPS delta |
|---|---:|---:|---:|---:|---:|
| Ant | 87.36 | 93.66 | 292,573 | 257,434 | -12.0% |
| Anymal-C Rough | 11.34 | 11.02 | 73,722 | 67,170 | -8.9% |

Newton/Warp camera PPO:

- Run root:
  `/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now`
- Task: `Isaac-Cartpole-Camera-Direct`
- Presets: `newton_mjwarp,newton_renderer,rgb`
- Device: `cuda:0`, headless.
- Puffer profile: `camera_cnn`.
- Conclusion: pixel observations were the performance-positive case. The
  Puffer-style backend was slower in rollout/render but saved enough PPO update
  time to improve total iteration SPS.

Fresh seed-44 camera validation:

| Camera case | RSL SPS | Puffer SPS | Puffer vs RSL | RSL collect/learn | Puffer rollout/train |
|---|---:|---:|---:|---:|---:|
| 100x100, 512 envs, 25 iters | 13,370 | 16,954 | +26.8% | 0.148/0.465s | 0.171/0.312s |
| 160x160, 256 envs, 15 iters | 5,110 | 6,518 | +27.5% | 0.168/0.634s | 0.198/0.430s |
| 224x224, 128 envs, 10 iters | 2,305 | 2,884 | +25.1% | 0.165/0.723s | 0.194/0.516s |

Those earlier camera results support the tensor-native Puffer-style path for
pixel-heavy workloads. They should not be used as evidence that the new compiled
external-vector path is faster until the native path is benchmarked directly.

## Current Answer to the Original Question

The confusion was justified. The report said the proper native external-vector
hook was still missing, while the intended engineering task was to build exactly
that hook.

That is now fixed:

- PufferLib has a compiled external-vector hook.
- IsaacLab has a bridge that can feed externally owned CUDA buffers into it.
- The hook has been run through tests and a real IsaacLab Newton/MJWarp smoke.

The next step is no longer "add the ABI." The next step is "benchmark the native
external-vector path fairly."

## Recommended Next Benchmark

To make a defensible speed claim, run a native external-vector benchmark matrix:

- `Isaac-Ant-v0`, 4096 envs, matched RSL policy, at least 200 iterations for
  throughput and a longer 1000-iteration learning check.
- `Isaac-Velocity-Rough-Anymal-C-v0`, 4096 envs, matched RSL policy, same
  iteration structure.
- `Isaac-Cartpole-Camera-Direct`, Newton/MJWarp RGB camera presets, using a native
  visual policy path if PufferLib's compiled trainer is extended to support the
  required image encoder cleanly.

Report these separately:

- RSL-RL baseline.
- Tensor-native Puffer-style runner.
- Compiled PufferLib native external-vector runner.

Only the third line answers whether the new native hook is faster.
