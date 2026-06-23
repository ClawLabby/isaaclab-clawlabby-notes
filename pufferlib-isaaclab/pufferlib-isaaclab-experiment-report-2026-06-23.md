# PufferLib for Isaac Lab Robotics: Experiment Report

Date: 2026-06-23

## Executive Summary

This report evaluates whether the current PufferLib integration work is likely to be valuable for future Isaac Lab robotics training.

The answer is mixed:

- **For current DexSuite RGB64 pixel training at Lab-scale environment counts, PufferLib does not beat RSL-RL.** The current PufferLib PyTorch `mixed_camera_cnn` runner is slower than the RSL-RL CNN runner by 4.8% to 11.4% across the completed 1024-4096 environment single-camera DexSuite cases.
- **The Warp manager frontend is not a standalone throughput win for this workload.** Comparing standard Isaac Lab manager tasks to Warp-manager task variants, both with and without PufferLib, shows neutral-to-negative results at serious scale.
- **The native/no-PyTorch PufferLib direction remains unproven for pixels.** The native external-vector bridge was validated only on state observations, not RGB image policies. It also reported `zero_copy_obs=false` and `zero_copy_rewards=false` in the DexSuite checks, so it is not yet the final zero-copy path.
- **PufferLib still has a plausible future value proposition, but not from the current DexSuite pixel data.** Prior smaller camera diagnostics showed PufferLib can reduce policy update time in image workloads, and the external-vector work is a reasonable substrate for a future tensor-native backend. The present evidence does not justify claiming a robotics throughput win over RSL-RL today.

Decision recommendation: **keep this as an experiment branch, not a PR branch.** The experiment is useful for learning and for future backend work, but the current implementation should not be presented as an upstream-ready performance improvement.

## What PufferLib Is

PufferLib is a reinforcement-learning framework focused on high-throughput training across many vectorized environments. In this experiment, it was used in two different ways:

1. **PufferLib PyTorch runner inside Isaac Lab:** an alternate PPO training path that consumes Isaac Lab observations and trains a PyTorch policy. For image observations, the tested model was `mixed_camera_cnn`, which combines vector observations with one or more RGB camera tensors.
2. **PufferLib native/external-vector path:** experimental C++/CUDA-side machinery intended to let externally owned CUDA buffers feed a PufferLib-native training path without forcing Isaac Lab to become a PufferLib-owned environment.

Those are separate claims. A PyTorch PufferLib runner can be evaluated against RSL-RL today. A future native/no-PyTorch pixel trainer cannot be evaluated from the current pixel numbers because that path has not been implemented and benchmarked for RGB image policies.

## Experiment Scope

This was a PufferLib-for-Isaac-Lab experiment, not a DexSuite-only experiment. DexSuite was used because it is a realistic dexterous manipulation workload with state and camera observation variants, but it is only one part of the evidence.

The experiment covered:

- State-observation learning checks on existing Isaac Lab tasks.
- A Cartpole camera diagnostic that exercises image observations at small scale.
- DexSuite state-only native external-vector checks.
- DexSuite RGB64 single-camera Lab-scale pixel benchmarks.
- DexSuite RGB64 dual-camera small-scale diagnostics.
- Standard Isaac Lab manager frontend versus Warp-manager frontend comparisons, both with PufferLib and with RSL-RL.

The experiment did **not** cover:

- Native/no-PyTorch pixel training with RGB image policies.
- A zero-copy DexSuite observation/reward bridge in the reported native checks.
- Learning-quality conclusions for DexSuite pixel tasks. The DexSuite pixel runs were throughput measurements, not long-horizon policy-quality studies.

## Code and Branches

Isaac Lab integration:

- Repo: `ClawLabby/IsaacLab`
- Branch: `experiment/pufferlib-isaaclab-warp-frontend-20260623`
- Commit tested: `b6e186dcea395467b3a6b6b076b35e82c03db93a`
- URL: <https://github.com/ClawLabby/IsaacLab/tree/experiment/pufferlib-isaaclab-warp-frontend-20260623>

PufferLib external-vector work:

- Repo: `ClawLabby/PufferLib`
- Branch: `isaaclab-external-gpu-vec`
- Commit tested: `3085221faa0f67b436aae9fb52152e8707f2c385`
- URL: <https://github.com/ClawLabby/PufferLib/tree/isaaclab-external-gpu-vec>

Public summary data in this notes repo:

- [DexSuite Lab-scale single-camera pixel summary](data/dexsuite_labscale_single_camera_summary.json)
- [DexSuite native state external-vector summary](data/dexsuite_native_state_external_summary.json)
- [DexSuite dual-camera diagnostic summary](data/dexsuite_dual_camera_diagnostic_summary.json)
- [Cartpole camera diagnostic summary](data/cartpole_camera_diagnostic_summary.json)
- [State learning summary](data/state_learning_summary.json)

The Isaac Lab branch adds:

- Experimental PufferLib PPO runner under `source/isaaclab_rl/isaaclab_rl/pufferlib`.
- `scripts/reinforcement_learning/pufferlib/train.py`.
- DexSuite `Lift-Warp-v0` and `Reorient-Warp-v0` task registrations.
- Standard-manager fallback dispatch for `ManagerBasedRLEnvWarp`.
- Newton renderer compatibility handling for `SensorTiledCamera.update()`.
- Unified RL dispatcher path handling so `scripts/reinforcement_learning/rsl_rl` does not shadow the installed `rsl_rl` package.

The PufferLib branch adds:

- `pufferlib.external.ExternalGPUVec`, an adapter for externally owned CUDA observation, reward, terminal, and optional action-mask buffers.
- Native C++/CUDA hooks for `create_external_pufferl` and `external_rollouts`.
- Lifetime handling so external buffers are not freed by PufferLib.
- Focused unit-test coverage for the Python external-vector adapter.

## Terminology

**State-based workflow:** the policy receives vector observations such as joint states, object state, target information, and proprioception. No camera tensor is used by the policy.

**Image-based or pixel workflow:** the policy receives RGB camera tensors, optionally combined with vector observations. The DexSuite Lab-scale decision runs used `base_image` at `(3, 64, 64)` plus the task's vector observation groups.

**Standard manager frontend:** the regular Isaac Lab manager-based RL task path.

**Warp manager frontend:** the experimental `ManagerBasedRLEnvWarp` task path, using Warp-oriented manager dispatch where available and falling back to standard managers where needed.

**PufferLib backend:** the experimental PufferLib runner and model path.

**RSL-RL backend:** the existing Isaac Lab RSL-RL runner used as the baseline.

## Decision-Grade DexSuite Pixel Results

These are the most important results because they use RGB64 camera observations at 1024-4096 environments.

Configuration:

- Tasks: DexSuite Kuka Allegro Lift and Reorient.
- Observations: single `base_image` RGB64 camera plus vector groups.
- Presets: `single_camera,newton_mjwarp,newton_renderer,rgb64`.
- Backends: PufferLib `mixed_camera_cnn` and RSL-RL CNN.
- Frontends: standard manager and Warp manager variants.
- Metric: mean iteration SPS after the first iteration.
- Seed: 42.
- Exit status: all 16 decision-scale cases exited successfully.

### Lift, Single-Camera RGB64

| Envs | Frontend | PufferLib SPS | RSL-RL SPS | PufferLib vs RSL-RL | Puffer rollout s | Puffer train s | RSL rollout s | RSL train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | standard manager | 29031.43 | 30793.11 | -5.7% | 0.749 | 0.380 | 0.704 | 0.360 |
| 1024 | Warp manager | 28960.80 | 30408.26 | -4.8% | 0.751 | 0.381 | 0.714 | 0.364 |
| 2048 | standard manager | 39357.51 | 43207.43 | -8.9% | 0.957 | 0.708 | 0.895 | 0.622 |
| 2048 | Warp manager | 39924.66 | 43376.22 | -8.0% | 0.945 | 0.696 | 0.896 | 0.615 |
| 4096 | standard manager | 46420.45 | 52229.57 | -11.1% | 1.424 | 1.400 | 1.296 | 1.214 |
| 4096 | Warp manager | 46582.35 | 51905.98 | -10.3% | 1.399 | 1.416 | 1.306 | 1.220 |

Interpretation:

- RSL-RL is faster in every Lift pixel case.
- The PufferLib gap grows with environment count in this dataset.
- The gap is visible in both rollout and train/update time at 4096 environments.

### Reorient, Single-Camera RGB64

| Envs | Frontend | PufferLib SPS | RSL-RL SPS | PufferLib vs RSL-RL | Puffer rollout s | Puffer train s | RSL rollout s | RSL train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | standard manager | 47199.31 | 51775.78 | -8.8% | 1.418 | 1.360 | 1.323 | 1.209 |
| 4096 | Warp manager | 45190.42 | 50987.47 | -11.4% | 1.457 | 1.444 | 1.330 | 1.241 |

Interpretation:

- RSL-RL is faster in both Reorient pixel cases.
- Warp manager reduces throughput for both backends in the 4096-env Reorient data.

## Warp Manager Frontend Comparison Without PufferLib

The RSL-RL rows isolate the frontend question without PufferLib.

| Task | Envs | Backend | Standard manager SPS | Warp manager SPS | Warp vs standard |
| --- | ---: | --- | ---: | ---: | ---: |
| Lift | 1024 | RSL-RL CNN | 30793.11 | 30408.26 | -1.2% |
| Lift | 2048 | RSL-RL CNN | 43207.43 | 43376.22 | +0.4% |
| Lift | 4096 | RSL-RL CNN | 52229.57 | 51905.98 | -0.6% |
| Reorient | 4096 | RSL-RL CNN | 51775.78 | 50987.47 | -1.5% |

This does not support a claim that the Warp manager frontend alone improves DexSuite RGB64 pixel throughput. It is roughly neutral for Lift and negative for Reorient in the completed Lab-scale matrix.

For completeness, the same frontend comparison with PufferLib:

| Task | Envs | Backend | Standard manager SPS | Warp manager SPS | Warp vs standard |
| --- | ---: | --- | ---: | ---: | ---: |
| Lift | 1024 | PufferLib mixed CNN | 29031.43 | 28960.80 | -0.2% |
| Lift | 2048 | PufferLib mixed CNN | 39357.51 | 39924.66 | +1.4% |
| Lift | 4096 | PufferLib mixed CNN | 46420.45 | 46582.35 | +0.3% |
| Reorient | 4096 | PufferLib mixed CNN | 47199.31 | 45190.42 | -4.3% |

Again, there is no consistent Warp-manager speedup.

## State-Based Results

The state-observation evidence answers a different question from the pixel evidence.

### Existing Isaac Lab State Tasks

The state learning summary includes PufferLib matched-RSL and RSL-RL runs on:

- `Isaac-Ant-v0` at 4096 envs, 1000 iterations, seeds 42 and 43.
- `Isaac-Velocity-Rough-Anymal-C-v0` at 4096 envs, 1500 iterations, seeds 42 and 43.

Selected final throughput and learning signals:

| Task | Seed | Backend | Final throughput | Final reward metric |
| --- | ---: | --- | ---: | ---: |
| Ant | 42 | PufferLib matched-RSL | 260215 SPS | estimated episode return 134.84 |
| Ant | 43 | PufferLib matched-RSL | 260328 SPS | estimated episode return 167.91 |
| Ant | 42 | RSL-RL | 287157 FPS | Train/mean_reward 151.69 |
| Ant | 43 | RSL-RL | 285237 FPS | Train/mean_reward 152.67 |
| Anymal-C Rough | 42 | PufferLib matched-RSL | 67676 SPS | estimated episode return 17.21 |
| Anymal-C Rough | 43 | PufferLib matched-RSL | 69698 SPS | estimated episode return 16.84 |
| Anymal-C Rough | 42 | RSL-RL | 71332 FPS | Train/mean_reward 16.21 |
| Anymal-C Rough | 43 | RSL-RL | 72456 FPS | Train/mean_reward 17.61 |

These runs do not show a decisive state-observation performance win for PufferLib over RSL-RL. They do show that the PufferLib runner can train real Isaac Lab state tasks, but the value proposition is not established on state tasks alone.

### DexSuite Native External-Vector State Checks

The DexSuite native checks exercised the experimental external-vector path on state observations at 32 and 128 envs for Lift and Reorient. These are compatibility and bridge checks, not Lab-scale decision runs.

| Task | Envs | Frontend | Wall SPS | Native SPS | Rollout s | Train s | Zero-copy obs | Zero-copy rewards |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| Lift | 32 | standard manager | 2623.87 | 2628.44 | 0.394 | 0.015 | false | false |
| Lift | 32 | Warp manager | 2510.40 | 2511.62 | 0.411 | 0.016 | false | false |
| Lift | 128 | standard manager | 10649.43 | 10672.35 | 0.368 | 0.016 | false | false |
| Lift | 128 | Warp manager | 10562.40 | 10585.25 | 0.372 | 0.016 | false | false |
| Reorient | 32 | standard manager | 2339.09 | 2343.94 | 0.442 | 0.016 | false | false |
| Reorient | 32 | Warp manager | 2416.17 | 2420.81 | 0.428 | 0.015 | false | false |
| Reorient | 128 | standard manager | 9703.01 | 9720.93 | 0.404 | 0.018 | false | false |
| Reorient | 128 | Warp manager | 10123.38 | 10144.72 | 0.388 | 0.016 | false | false |

Interpretation:

- The native external-vector bridge can run DexSuite state observations through the experimental native path.
- This is not a pixel result.
- This is not a zero-copy result as measured in the run metadata.
- These env counts are too small to support Lab-scale performance claims.

## Image-Based Diagnostic Results Outside the Lab-Scale DexSuite Matrix

### Cartpole Camera Diagnostic

A smaller Cartpole camera diagnostic showed a PufferLib advantage:

| Task | Envs | Backend | Mean SPS | Rollout/collection s | Train/learning s |
| --- | ---: | --- | ---: | ---: | ---: |
| Cartpole camera | 512 | PufferLib camera CNN | 17086.62 | 0.171 | 0.309 |
| Cartpole camera | 512 | RSL-RL camera CNN | 13580.71 | 0.149 | 0.454 |

PufferLib was +25.8% on mean SPS in this diagnostic. The win came from lower train/update time, not from faster rollout/collection.

This result is useful because it shows the kind of image-workload advantage PufferLib might provide. It is not sufficient to override the DexSuite Lab-scale result because the task, observation pipeline, and scale are different.

### DexSuite Dual-Camera Diagnostic

The dual-camera DexSuite runs used RGB64 `base_image` plus `wrist_image`, but only at 32 and 128 envs. They are diagnostic, not decision-grade Lab-scale evidence.

| Task | Envs | Frontend | PufferLib SPS | RSL-RL SPS | PufferLib vs RSL-RL |
| --- | ---: | --- | ---: | ---: | ---: |
| Lift | 32 | standard manager | 1680.91 | 1587.50 | +5.9% |
| Lift | 32 | Warp manager | 1583.14 | 1511.88 | +4.7% |
| Lift | 128 | standard manager | 5241.47 | 4846.04 | +8.2% |
| Lift | 128 | Warp manager | 5251.95 | 5129.25 | +2.4% |
| Reorient | 32 | standard manager | 1577.89 | 1527.04 | +3.3% |
| Reorient | 32 | Warp manager | 1539.11 | 1507.88 | +2.1% |
| Reorient | 128 | standard manager | 4947.78 | 4980.13 | -0.6% |
| Reorient | 128 | Warp manager | 4975.58 | 4862.46 | +2.3% |

Interpretation:

- PufferLib often wins these small dual-camera diagnostics.
- These env counts should not be treated as serious Isaac Lab throughput evidence.
- The result suggests image-heavy PufferLib work is still worth investigating, but it does not establish a Lab-scale win.

## Native Pixel Path Status

The native/no-PyTorch pixel path has not been tested because it does not yet exist in the benchmarked form.

The completed native external-vector checks are state-only. They validate some integration mechanics, but they do not answer whether a PufferLib-native CNN or other no-PyTorch pixel policy can outperform RSL-RL on DexSuite camera observations.

Therefore:

- The current DexSuite pixel report is sufficient to reject the claim that the **current PufferLib PyTorch mixed-CNN path** beats RSL-RL at Lab scale.
- The current DexSuite pixel report is **not** sufficient to reject a future native/no-PyTorch pixel backend.
- It would be wrong to use the current data as proof that native Warp pixel tasks are fast, because those tasks were not run.

## Validation and Limitations

Completed:

- All 16 Lab-scale DexSuite single-camera pixel benchmark cases exited successfully.
- Logs confirmed real `base_image` camera tensors at `(3, 64, 64)` and RSL-RL `CNNModel` usage.
- The RSL-RL rows provide a direct standard-manager versus Warp-manager comparison without PufferLib.
- The PufferLib rows provide the same frontend comparison with PufferLib.
- DexSuite native external-vector checks ran successfully for state observations.

Limitations:

- DexSuite pixel results are short throughput runs: Lift used 100 iterations at 1024 envs and 50 iterations at 2048/4096 envs; Reorient used 50 iterations at 4096 envs.
- The DexSuite pixel results are single-seed throughput measurements, not final learning-quality comparisons.
- Native external-vector DexSuite runs are state-only and small-scale.
- Native external-vector DexSuite runs reported non-zero-copy observation and reward handling.
- No native/no-PyTorch pixel policy benchmark was completed.

## Conclusion

The experiment does not support upstreaming the current PufferLib Isaac Lab path as a performance improvement over RSL-RL for robotics pixels.

The strongest decision-grade data is DexSuite RGB64 at 1024-4096 environments. In that matrix, RSL-RL is consistently faster than the current PufferLib `mixed_camera_cnn` runner. The Warp manager frontend does not change that conclusion and does not independently improve RSL-RL throughput.

PufferLib may still be worth future work if the goal is a genuinely tensor-native or native/no-PyTorch image backend, because smaller camera diagnostics and update-time behavior suggest there may be room to win in image-heavy workloads. But that future claim needs new data from actual native pixel tasks. The current report should be read as a negative result for the present PyTorch mixed-CNN integration and an incomplete result for the native pixel direction.
