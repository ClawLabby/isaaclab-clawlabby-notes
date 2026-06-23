# DexSuite Pixel Loop Lab-Scale Analysis

Date: 2026-06-23

## Executive Summary

The Lab-scale DexSuite pixel data is sufficient for a narrow performance conclusion:

**For DexSuite RGB64 single-camera pixels in the loop, the current IsaacLab/PufferLib `mixed_camera_cnn` path does not beat RSL-RL CNN at serious Isaac Lab environment counts.**

Across the completed decision-scale cases, RSL-RL is faster than PufferLib by 4.8% to 11.4%. The Warp frontend is not a consistent throughput win in this workload: it is roughly neutral on Lift at 2048/4096 envs and negative on Reorient at 4096 envs.

This does not close the question of a future no-PyTorch or zero-copy pixel trainer. The native external-vector work validated a state-observation bridge, but the decision-scale pixel comparison here is between RSL-RL CNN and the current PufferLib PyTorch `mixed_camera_cnn` runner.

## Code Under Review

IsaacLab integration branch:

- Repo: `ClawLabby/IsaacLab`
- Branch: `pr/dexsuite-pufferlib-warp-frontend`
- Commit: `b6e186dcea395467b3a6b6b076b35e82c03db93a`
- URL: <https://github.com/ClawLabby/IsaacLab/tree/pr/dexsuite-pufferlib-warp-frontend>

PufferLib external-vector branch:

- Repo: `ClawLabby/PufferLib`
- Branch: `isaaclab-external-gpu-vec`
- Commit: `3085221faa0f67b436aae9fb52152e8707f2c385`
- URL: <https://github.com/ClawLabby/PufferLib/tree/isaaclab-external-gpu-vec>

The IsaacLab branch adds:

- Experimental PufferLib-style PPO runner under `source/isaaclab_rl/isaaclab_rl/pufferlib`.
- `scripts/reinforcement_learning/pufferlib/train.py`.
- DexSuite `Lift-Warp-v0` and `Reorient-Warp-v0` task registrations.
- Stable-manager fallback dispatch for `ManagerBasedRLEnvWarp`.
- A Newton renderer compatibility guard for `SensorTiledCamera.update()` signatures with and without `hdr_color_image`.
- A unified RL dispatcher path fix so `scripts/reinforcement_learning/rsl_rl` does not shadow the installed `rsl_rl` package.

The PufferLib branch adds:

- `pufferlib.external.ExternalGPUVec`, an adapter for externally-owned CUDA observation, reward, terminal, and optional action-mask buffers.
- Native C++/CUDA hooks for `create_external_pufferl` and `external_rollouts`.
- Lifetime handling so external buffers are not freed by PufferLib.
- A focused unit test file for the Python external-vector adapter.

## Test Definition

All decision-scale runs used:

- Tasks:
  - `Isaac-Dexsuite-Kuka-Allegro-Lift-v0`
  - `Isaac-Dexsuite-Kuka-Allegro-Lift-Warp-v0`
  - `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0`
  - `Isaac-Dexsuite-Kuka-Allegro-Reorient-Warp-v0`
- Presets: `single_camera,newton_mjwarp,newton_renderer,rgb64`
- Pixel observation: real `base_image` camera group at `(3, 64, 64)`.
- Vector observations: DexSuite policy/proprio groups as configured by the task and runner.
- Backends:
  - PufferLib `mixed_camera_cnn`
  - RSL-RL CNN
- Metric: mean iteration SPS after the first iteration.
- Seed: 42.

Artifacts:

- Combined summary JSON: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_labscale_summary.json`
- Local report: `/home/horde/claw/pufferlib-isaaclab-eval/DEXSUITE_PIXEL_LABSCALE_REPORT.md`
- Lift 1024 envs, 100 iterations: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_realtest_20260622_lift1024_100it_serial`
- Lift 2048 envs, 50 iterations: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_realtest_20260622_lift2048_50it_serial`
- Lift 2048 fill cases, 50 iterations: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_realtest_20260622_lift2048_fill_cuda0_50it_serial`
- Lift 4096 envs, 50 iterations: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_realtest_20260622_lift4096_50it_parallel`
- Reorient 4096 envs, 50 iterations: `/home/horde/claw/pufferlib-isaaclab-eval/dexsuite_realtest_20260622_reorient4096_50it_parallel`

All 16 decision-scale exit markers were `0`.

## Results

### Lift

| Envs | Frontend | Puffer SPS | RSL-RL SPS | Puffer vs RSL-RL | Puffer rollout s | Puffer train s | RSL rollout s | RSL train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | stable | 29031.43 | 30793.11 | -5.7% | 0.749 | 0.380 | 0.704 | 0.360 |
| 1024 | Warp | 28960.80 | 30408.26 | -4.8% | 0.751 | 0.381 | 0.714 | 0.364 |
| 2048 | stable | 39357.51 | 43207.43 | -8.9% | 0.957 | 0.708 | 0.895 | 0.622 |
| 2048 | Warp | 39924.66 | 43376.22 | -8.0% | 0.945 | 0.696 | 0.896 | 0.615 |
| 4096 | stable | 46420.45 | 52229.57 | -11.1% | 1.424 | 1.400 | 1.296 | 1.214 |
| 4096 | Warp | 46582.35 | 51905.98 | -10.3% | 1.399 | 1.416 | 1.306 | 1.220 |

Lift interpretation:

- RSL-RL wins all tested Lift env counts and frontend variants.
- The Puffer gap grows with scale in this dataset: -5.7% at 1024 stable, -8.9% at 2048 stable, and -11.1% at 4096 stable.
- Warp has little net effect on Lift throughput at serious scale.

### Reorient

| Envs | Frontend | Puffer SPS | RSL-RL SPS | Puffer vs RSL-RL | Puffer rollout s | Puffer train s | RSL rollout s | RSL train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | stable | 47199.31 | 51775.78 | -8.8% | 1.418 | 1.360 | 1.323 | 1.209 |
| 4096 | Warp | 45190.42 | 50987.47 | -11.4% | 1.457 | 1.444 | 1.330 | 1.241 |

Reorient interpretation:

- RSL-RL wins both 4096-env Reorient cases.
- Warp reduces Reorient throughput for both backends in this dataset.

## Warp Frontend Effect

The Warp frontend is not a reliable win for this pixel loop:

- Lift, PufferLib: -0.2% at 1024, +1.4% at 2048, +0.3% at 4096.
- Lift, RSL-RL: -1.2% at 1024, +0.4% at 2048, -0.6% at 4096.
- Reorient 4096, PufferLib: -4.3%.
- Reorient 4096, RSL-RL: -1.5%.

For this workload, the current bottleneck does not appear to be solved by switching only the manager frontend to Warp.

## Device Binding Finding

Newton/Warp multi-GPU runs must isolate each process with `CUDA_VISIBLE_DEVICES=N` and pass `--device cuda:0` inside that process.

Using physical ordinals such as `--device cuda:1` inside the process produced `CUDA error 101: invalid device ordinal`, later surfacing as small Warp allocation failures. The corrected harness pattern completed all 4096-env four-way Lift and Reorient cases.

Updated local harness:

- `/home/horde/claw/pufferlib-isaaclab-eval/run_dexsuite_realtest_parallel.sh`

## Validation Notes

Completed validation:

- IsaacLab Python syntax check passed for the touched PufferLib runner, Warp env, and DexSuite config modules.
- PufferLib Python syntax check passed for `pufferlib/external.py` and `tests/test_external_gpu_vec.py`.
- All decision-scale benchmark exit markers were `0`.
- Logs confirm real `base_image` `(3, 64, 64)` camera observations and RSL-RL `CNNModel` usage.

Validation gap:

- The local PufferLib checkout did not have `pytest` installed in either system Python or `.venv`, so `tests/test_external_gpu_vec.py` was syntax-checked but not executed in this packaging pass.

## What This Does Not Prove

This is throughput evidence, not a learning-quality result. The runs are short and single-seed:

- Lift: 1024 envs for 100 iterations, 2048/4096 envs for 50 iterations.
- Reorient: 4096 envs for 50 iterations.

This also does not prove that a native no-PyTorch or zero-copy pixel trainer would lose. The native external-vector path is currently state-only in the available validation artifacts and reports `zero_copy_obs=false` and `zero_copy_rewards=false` for the DexSuite native checks. A final no-PyTorch pixel-loop answer would require a native CNN path or equivalent pixel policy path using the external-vector bridge.

## Conclusion

There is enough evidence to make the current implementation decision:

**Do not claim a PufferLib throughput win over RSL-RL for DexSuite RGB64 pixels in the loop at Lab scale.**

The current PufferLib mixed-CNN path is useful as an integration and experimentation surface, and the external-vector branch is a reasonable substrate for the next no-PyTorch direction. But the measured Lab-scale pixel-loop comparison favors RSL-RL today.
