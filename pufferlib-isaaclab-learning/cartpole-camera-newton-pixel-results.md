# PufferLib Camera-Mode Probe: Newton/Warp Pixel Observations

**Author:** ClawLabby
**Date:** 2026-06-16
**Status:** Rendered-pixel follow-up completed; tensor-native Puffer-style backend shows tangible value on this workload

## Executive Summary

The earlier state-observation learning matrix did not show a Puffer speed win
over RSL-RL. The rendered-pixel case is different.

On `Isaac-Cartpole-Camera-Direct` using Newton/Warp rendering and RGB camera
observations, an optimized Puffer-style PyTorch backend outperformed RSL-RL by
roughly 22-43% at 100x100 camera resolution, and by roughly 24-27% on larger
160x160 and 224x224 camera probes.

The win does not come from faster environment stepping or faster rendering.
RSL-RL still collects/render-rolls out faster in the paired measurements. The
win comes from lower PPO update time once image batches and CNN encoder work
dominate enough of the iteration. That makes rendered-pixel RL the first tested
case where the Puffer/Hermes direction has a clear performance reason beyond
being a useful experimental trainer.

Important limitation: this is evidence for a tensor-native Puffer-style
Hermes/IsaacLab backend, not evidence that the current PufferLib 4.0 native
Ocean path can plug into IsaacLab directly. PufferLib still lacks the
external-vector/native-tensor hook needed for IsaacLab-owned Warp/torch tensors.

## Workload

Task and renderer:

- Task: `Isaac-Cartpole-Camera-Direct`
- Presets: `newton_mjwarp,newton_renderer,rgb`
- Device: `cuda:0`
- Default camera observation: Newton/Warp RGB, `frame_stack: 2`, resolved
  observation space `[6, 100, 100]`
- Main baseline run: 512 envs, 50 iterations, seed 42

Prototype changes:

- Added a `camera_cnn` profile to the experimental IsaacLab Puffer runner so
  camera observations stay `[B, C, H, W]` instead of being flattened.
- Added CNN actor/critic support matching the Cartpole camera RSL-RL CNN shape.
- Used IsaacLab `resolve_task_config()` so preset tokens are applied before Kit
  launch.
- Added camera-width and camera-height overrides to the probe launcher, including
  matching `env.observation_space` overrides for larger-resolution tests.
- Patched the local Newton renderer compatibility issue for installed Newton
  `1.2.0rc2`: `SensorTiledCamera.update()` does not accept `hdr_color_image`, so
  the renderer now only passes that kwarg when supported.

## Initial Probe and Correction

The first camera probe was misleading:

- RSL-RL CNN, 512 envs, seed 42: post-warmup mean `13,653` SPS.
- Initial Puffer `camera_cnn`, same workload: post-warmup mean `12,282` SPS.

At face value this suggested that even pixel input did not make the Puffer-style
backend valuable. The root cause was then found in the prototype model: actor
and critic used the same camera observation with `share_cnn_encoders=True`, but
the Puffer runner still executed the shared CNN twice, once for the actor path
and once for the critic path. RSL-RL's camera config is a shared-CNN setup, so
that was not an apples-to-apples implementation.

After fixing `CameraCNNActorCritic` to reuse the shared feature tensor when
actor and critic observation groups and shapes match, the conclusion flipped.

## Corrected Default-Resolution Results

Post-warmup means exclude iteration 0.

| Envs | RSL-RL SPS | Initial Puffer SPS | Optimized Puffer SPS | Optimized vs RSL | RSL collection / learning | Puffer rollout / train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 11,634 | 11,119 | 14,144 | +21.6% | 0.123 / 0.229 s | 0.127 / 0.162 s |
| 512 | 13,651 | 12,283 | 16,873 | +23.6% | 0.150 / 0.450 s | 0.175 / 0.311 s |
| 1024 | 14,278 | 12,796 | 19,101 | +33.8% | 0.210 / 0.937 s | 0.259 / 0.599 s |
| 2048 | 14,055 | 13,203 | 20,033 | +42.5% | 0.353 / 1.979 s | 0.439 / 1.197 s |

Interpretation:

- Puffer rollout/render is consistently slower than RSL-RL collection.
- Puffer train/update is substantially faster once duplicate CNN work is removed.
- The train/update savings grow with batch size and dominate end-to-end
  iteration throughput at larger env counts.

## Larger Pixel Inputs

To test whether the result persists with more image data, the probe script was
extended to override camera resolution and matching observation shape.

Post-warmup means exclude iteration 0.

| Camera | Envs | Iterations | RSL-RL SPS | Optimized Puffer SPS | Optimized vs RSL | RSL collection / learning | Puffer rollout / train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160x160 | 256 | 15 | 5,126 | 6,521 | +27.2% | 0.166 / 0.633 s | 0.201 / 0.427 s |
| 224x224 | 128 | 10 | 2,330 | 2,892 | +24.1% | 0.167 / 0.712 s | 0.195 / 0.513 s |

These runs support the same conclusion as the default-resolution sweep: larger
pixel observations preserve the Puffer-style advantage, again through faster PPO
update time rather than faster rendering.

## Seed-43 Repeats

A second seed was run for the default 100x100 case and the 224x224 high-res
case. `summarize_camera_runs.py` parses RSL stdout and Puffer `metrics.jsonl`
files into comparable post-warmup means.

| Camera | Envs | Iterations | Seed | RSL-RL SPS | Optimized Puffer SPS | Optimized vs RSL | RSL collection / learning | Puffer rollout / train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100x100 | 512 | 25 | 43 | 13,581 | 17,087 | +25.8% | 0.149 / 0.454 s | 0.171 / 0.309 s |
| 224x224 | 128 | 10 | 43 | 2,323 | 2,895 | +24.6% | 0.164 / 0.717 s | 0.198 / 0.510 s |

The repeats increase confidence that the corrected result is not a single-seed
or single-run artifact.

## Value Assessment

For state observations, the previous conclusion still stands: matched PPO
learning quality was comparable, but RSL-RL was slightly faster. That alone
would not justify a full PufferLib backend as a performance project.

For rendered-pixel RL, there is tangible value:

- Pixel observations make CNN/update cost important enough that trainer
  efficiency changes the total iteration rate.
- The optimized Puffer-style runner reduces PPO update time on large image
  batches.
- The observed end-to-end speedup is material in this Cartpole camera probe:
  about 22-43% at default resolution, about 27% at 160x160, and about 24% at
  224x224 across two seeds.

The practical direction is not to port IsaacLab tasks into PufferLib Ocean C
environments. The useful direction is a Hermes/IsaacLab backend that keeps
IsaacLab owning simulation/rendering and feeds GPU-resident tensor observations
into a lean trainer path.

## Remaining Caveats

- This is Cartpole camera, not dexterous manipulation or quadruped vision.
- The runs are throughput probes, not long reward-quality learning curves.
- Puffer rollout/render is slower than RSL-RL in the paired numbers, so the next
  prototype should profile why collection is slower before assuming all wins
  will transfer.
- The current PufferLib 4.0 native Ocean path still has no public
  external-vector hook for IsaacLab-owned tensors.
- Native-Puffer claims remain unproven until PufferLib can consume an external
  vector environment without forcing an Ocean/C-env rewrite.

## Recommendation

Proceed with a targeted Hermes/IsaacLab tensor-native backend prototype for
pixel-heavy RL. Treat this as a trainer/backend design experiment, not as a
PufferLib Ocean port.

Recommended next steps:

1. Preserve the camera-aware `isaaclab_rl.pufferlib` prototype on a dedicated
   branch.
2. Run one larger rendered-pixel task with real learning curves, not just
   Cartpole throughput.
3. Profile collection/render overhead separately from PPO update overhead.
4. Keep the native PufferLib investigation focused on an external-vector hook;
   do not duplicate IsaacLab task logic in Ocean `binding.c` files.
5. Make future comparisons log identical episodic-return and wall-clock metrics
   across backends.

## Local Artifacts

Primary local report:

- `/home/horde/claw/pufferlib-isaaclab-eval/CARTPOLE_CAMERA_REPORT.md`

Summary artifacts:

- `/home/horde/claw/pufferlib-isaaclab-eval/camera_repeats_seed43_summary.md`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_repeats_seed43_summary.json`

Representative run roots:

- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_probe_512x50_20260616T041708Z`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_puffer_sharedcnn_512x25_20260616T043000Z`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_160px_256x15_20260616T043456Z`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_224px_128x10_20260616T043618Z`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_repeat_seed43_512x25_20260616T044114Z`
- `/home/horde/claw/pufferlib-isaaclab-eval/camera_runs/cartpole_newton_repeat_seed43_224px_128x10_20260616T044225Z`
