# PufferLib / IsaacLab Native-Path Evaluation

Date: 2026-06-18

## Bottom Line

The current edit does not prove that PufferLib's compiled native Ocean trainer is
faster for IsaacLab, because that trainer still cannot consume an externally
owned IsaacLab/Warp vector environment. The native probe now makes that explicit:
`pufferlib._C` imports and exposes `create_vec`, but it has no external-vector
hook and `native_train_available` is `false`.

What is actually tested and useful is the tensor-native IsaacLab/Puffer-style
path:

- For state-observation PPO on realistic 4096-env IsaacLab tasks, the matched
  Puffer-style PyTorch backend learns in the same range as RSL-RL but is slower
  on throughput. This is not a performance win.
- For Newton/Warp rendered RGB camera observations, the corrected Puffer-style
  camera backend is consistently faster than RSL-RL in the tested Cartpole
  camera workload. Fresh seed-44 runs show +25% to +28% iteration SPS across
  100x100, 160x160, and 224x224 camera cases.
- The camera win is not from faster simulation or rendering. Puffer rollout is
  still slower. The win comes from lower PPO update time after avoiding duplicate
  shared-CNN actor/critic work.

The practical recommendation is to continue with a tensor-native Hermes/IsaacLab
backend prototype for pixel-heavy workloads. A true compiled PufferLib-native
IsaacLab path only becomes meaningful after PufferLib grows a native external
vector hook that accepts IsaacLab-owned CUDA buffers and callbacks.

## What Changed

PufferLib checkout:

- Added `pufferlib.external.ExternalGPUVec`.
- Exported `ExternalGPUVec` from `pufferlib/__init__.py`.
- Added protocol tests in `tests/test_external_gpu_vec.py`.
- Adjusted `build.sh` so the local GCC build path can use `-lgomp` instead of
  Clang-only OpenMP assumptions.

IsaacLab checkout:

- Added experimental `isaaclab_rl.pufferlib` package:
  - `puffer_cfg.py`: PPO config used by the experimental runner.
  - `vecenv_wrapper.py`: IsaacLab Gym/vector-env tensor wrapper with actor and
    critic observation group handling.
  - `runner.py`: pure-PyTorch PPO runner with three explicit profiles:
    `matched_rsl`, `camera_cnn`, and `puffer_mingru`.
  - `warp_bridge.py`: pointer/callback ABI sketch for IsaacLab CUDA buffers.
  - `native.py`: native availability probe plus explicit failure path for the
    missing compiled external-vector hook.
- Added `scripts/reinforcement_learning/pufferlib/train.py`, using the normal
  IsaacLab launcher, preset resolution, task registry, log dumping, and RSL-RL
  config matching.
- Patched `NewtonWarpRenderer` to adapt to installed Newton `1.2.0rc2`, where
  `SensorTiledCamera.update()` does not accept `hdr_color_image`. PPISP still
  fails fast if HDR output is required but unsupported.

## Native-Adapter Validation

Validated command family:

```bash
cd /home/horde/claw/git/IsaacLab
PYTHONPATH=/home/horde/claw/research/PufferLib ./isaaclab.sh -p -m pytest \
  /home/horde/claw/research/PufferLib/tests/test_external_gpu_vec.py
```

Result: 4 tests passed.

Real Warp CUDA pointer smoke:

- Warp version: 1.13.0
- Device: NVIDIA L40 on `cuda:0`
- `ExternalGPUVec.from_warp(...)` preserved observation, reward, terminal, and
  action-mask CUDA pointers.
- `gpu_step(123456)` forwarded the action pointer to the callback unchanged.

Native PufferLib probe after installing the missing local `rich_argparse`
dependency:

```json
{
  "c_env_name": "squared_continuous",
  "c_gpu": 0,
  "c_precision_bytes": 4,
  "can_import_c": true,
  "can_import_torch_pufferl": true,
  "error": null,
  "external_vec_hook": null,
  "has_create_pufferl": false,
  "has_create_vec": true,
  "has_external_gpu_vec": true,
  "has_external_vec_hook": false,
  "native_train_available": false
}
```

Interpretation: the PufferLib-side external buffer adapter is real and tested,
including against real Warp arrays. The compiled native trainer still cannot be
used for IsaacLab-owned environments because there is no `_C` hook such as
`create_external_pufferl` or `create_pufferl_from_vec`.

Native-adapter log:
`/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now/native_probe_after_dependency.log`

## State-Observation Learning: Realistic 4096-Env Tasks

These are the best state-observation learning comparisons because they run
longer than the fresh confirmation pass:

- Run root:
  `/home/horde/claw/pufferlib-isaaclab-eval/learning_runs/full_20260609T161900Z`
- `Isaac-Ant-v0`: 4096 envs, 1000 iterations, seeds 42 and 43.
- `Isaac-Velocity-Rough-Anymal-C-v0`: 4096 envs, 1500 iterations, seeds 42 and
  43.
- Puffer profile: `matched_rsl`, copying the RSL-RL actor/critic MLP sizes,
  rollout horizon, PPO epochs, minibatches, Adam learning rate, KL schedule,
  gamma/lambda, value/entropy coefficients, clipping, timeout bootstrapping, and
  action std initialization.

| Task | Seed | RSL reward/return | Puffer est. return | Return delta | RSL FPS | Puffer FPS | FPS delta | RSL train | Puffer train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ant | 42 | 151.69 | 134.84 | -11.1% | 287,157 | 265,179 | -7.7% | 0.095s | 0.117s |
| Ant | 43 | 152.67 | 167.91 | +10.0% | 285,237 | 263,434 | -7.6% | 0.094s | 0.116s |
| Anymal-C Rough | 42 | 16.21 | 17.21 | +6.2% | 71,332 | 68,403 | -4.1% | 0.100s | 0.125s |
| Anymal-C Rough | 43 | 17.61 | 16.84 | -4.4% | 72,456 | 69,955 | -3.5% | 0.097s | 0.121s |

Fresh seed-44 confirmation, 4096 envs and 200 iterations:

| Task | RSL reward/return | Puffer est. return | RSL FPS | Puffer FPS | FPS delta |
|---|---:|---:|---:|---:|---:|
| Ant | 87.36 | 93.66 | 292,573 | 257,434 | -12.0% |
| Anymal-C Rough | 11.34 | 11.02 | 73,722 | 67,170 | -8.9% |

State-task conclusion: the matched Puffer-style runner is close enough to be a
valid experimental backend, but it is not faster than RSL-RL for state
observations. RSL-RL remains ahead on FPS and PPO update time in the realistic
state tasks tested here.

## Newton/Warp Camera Learning

Fresh seed-44 camera validation:

- Run root:
  `/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now`
- Task: `Isaac-Cartpole-Camera-Direct`
- Presets: `newton_mjwarp,newton_renderer,rgb`
- Device: `cuda:0`, headless.
- Puffer profile: `camera_cnn`.
- Warmup handling: summaries exclude iteration 0.

| Camera case | RSL SPS | Puffer SPS | Puffer vs RSL | RSL collect/learn | Puffer rollout/train |
|---|---:|---:|---:|---:|---:|
| 100x100, 512 envs, 25 iters | 13,370 | 16,954 | +26.8% | 0.148/0.465s | 0.171/0.312s |
| 160x160, 256 envs, 15 iters | 5,110 | 6,518 | +27.5% | 0.168/0.634s | 0.198/0.430s |
| 224x224, 128 envs, 10 iters | 2,305 | 2,884 | +25.1% | 0.165/0.723s | 0.194/0.516s |

Artifacts:

- Summary:
  `/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now/CAMERA_SUMMARY.md`
- JSON:
  `/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now/camera_summary.json`
- Follow-up validation log:
  `/home/horde/claw/pufferlib-isaaclab-eval/native_validation_20260618T_now/validation_followup_20260618T134149Z.log`

Exit-marker note: all RSL camera markers are zero. The 160x160 and 224x224
Puffer markers are zero. The 100x100 Puffer run has complete JSONL metrics,
TensorBoard event output, `model_final.pt`, and the final "wrote logs" message,
but the earlier interrupted wrapper did not leave a `puffer.exit` marker for
that one run.

Prior camera sweeps remain consistent with the fresh seed-44 result:

| Camera case | Seed | RSL SPS | Puffer SPS | Puffer vs RSL |
|---|---:|---:|---:|---:|
| 100x100, 256 envs | 42 | 11,634 | 14,144 | +21.6% |
| 100x100, 512 envs | 42 | 13,651 | 16,873 | +23.6% |
| 100x100, 1024 envs | 42 | 14,278 | 19,101 | +33.8% |
| 100x100, 2048 envs | 42 | 14,055 | 20,033 | +42.5% |
| 100x100, 512 envs | 43 | 13,581 | 17,087 | +25.8% |
| 224x224, 128 envs | 43 | 2,323 | 2,895 | +24.6% |

Camera-task conclusion: pixel observations are the real performance-positive
case. The Puffer-style backend is slower in rollout/render by roughly 23-30 ms
per iteration in the fresh camera cases, but it saves roughly 153-207 ms per
PPO update, so total iteration SPS improves by about 25-28%.

## What This Means for "Native"

There are three distinct paths:

1. Fully compiled PufferLib native Ocean trainer.
   This is not currently usable for IsaacLab-owned environments. The compiled
   extension creates its own Ocean vecs and exposes no external-vector hook.

2. PufferLib Python wrapper over external GPU buffers.
   `ExternalGPUVec` now supplies the pointer/callback surface and is validated
   against real Warp arrays. This is a useful bridge component, but it is not
   itself the full compiled trainer.

3. Tensor-native IsaacLab/Puffer-style backend.
   This is the path actually evaluated for performance. It is not a state-task
   speed win, but it is a consistent rendered-camera speed win because it
   reduces trainer overhead and avoids duplicate shared CNN work.

## Recommendation

Do not claim a general PufferLib native speedup for IsaacLab yet.

The evidence supports a narrower and more useful claim: a tensor-native
Hermes/IsaacLab backend is worth pursuing for pixel-heavy Newton/Warp workloads,
especially if it preserves IsaacLab-owned CUDA tensors and keeps visual encoder
work shared between actor and critic. State-observation locomotion should not be
the selling case unless a later native hook or trainer rewrite changes the
throughput profile.

The next engineering step, if the goal is true native PufferLib integration, is
small but specific PufferLib-side API work: add a compiled external-vector hook
that accepts externally owned CUDA observation/reward/done/action buffers plus
reset/step/log callbacks. Without that hook, the native Ocean path cannot be
honestly benchmarked on IsaacLab use cases.
