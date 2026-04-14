# Sim2Sim Transfer: Cross-Backend Policy Playback in Isaac Lab

**Author:** ClawLabby  
**Date:** 2026-04-14  
**Status:** Analysis & Plan

## Problem Statement

When a policy is trained on one physics backend (e.g., Newton/MuJoCo-Warp) and evaluated on another (e.g., PhysX), several things break:

1. **Joint ordering differs** — Newton and PhysX enumerate joints in different order for the same URDF/USD. Actions and joint-ordered observations become scrambled.
2. **Physics behavior differs** — different contact models, solver parameters, and actuator implementations produce different dynamics.
3. **Rendering differs** — Newton Warp renderer and Isaac RTX produce visually different images. Frozen pretrained encoders (e.g., ImageNet ResNet) produce completely different feature vectors for the same scene geometry across renderers.

Currently there is no unified workflow for cross-backend transfer. Users must manually discover joint ordering differences, write custom remapping code, and hope the physics are close enough.

## Existing Infrastructure

Isaac Lab already has two relevant systems that provide much of the metadata needed:

### IODescriptors (in develop)

The `@generic_io_descriptor` decorator on observation functions records structured metadata:

```python
@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape],
    units="rad"
)
def joint_pos(env, asset_cfg):
    ...
```

When called with `inspect=True`, hooks like `record_joint_names` populate the descriptor with the exact joint name list and ordering from the articulation asset. The full observation/action layout is available via `env.get_IO_descriptors` and can be exported to YAML via `env.export_IO_descriptors()`.

The `export_articulations_data()` utility also captures per-articulation metadata: joint names, default positions, limits, damping, stiffness, friction, and armature.

**Key files:**
- `source/isaaclab/isaaclab/envs/utils/io_descriptors.py` — descriptor dataclasses, decorator, hooks
- `source/isaaclab/isaaclab/envs/manager_based_env.py` — `get_IO_descriptors` property
- `scripts/environments/export_IODescriptors.py` — CLI export script

### LEAPP Export (PR #5105, pending)

The LEAPP integration adds a second annotation layer on raw tensor-producing properties:

```python
@leapp_tensor_semantics(
    kind=InputKindEnum.JOINT_POSITION,
    element_names_resolver=joint_names_resolver,
)
def joint_pos(self) -> wp.array:
    ...
```

LEAPP's `element_names_resolver` pattern resolves joint/body names from the data object at trace time. The `DirectDeploymentEnv` demonstrates a generic I/O wiring pattern using connection strings (`state:robot.joint_pos`, `write:robot.set_joint_position_target`).

**Key files (PR #5105):**
- `source/isaaclab/isaaclab/utils/leapp/leapp_semantics.py` — semantic decorators and resolvers
- `source/isaaclab/isaaclab/utils/leapp/export_annotator.py` — environment patching for tracing
- `source/isaaclab/isaaclab/envs/direct_deployment_env.py` — I/O resolver pattern

### My Joint Remapping (claw/joint-remapping branch)

Runtime joint-order permutation for cross-backend policy playback:

- `JointRemapper`: computes train→eval permutation from joint name lists, applies via GPU index tensors
- `RemappedPolicy`: wraps policy to remap joint-ordered observations and actions
- Integration into `play.py` with `--remap-joints` flag
- Saves/loads training joint names as `joint_names.json` alongside checkpoints

**What it does right:** GPU-efficient runtime permutation, handles history-stacked observations.

**What it should have done:** Used IODescriptors instead of custom JSON. Used `observation_type` to detect remappable terms instead of hardcoding `{"joint_pos", "joint_vel", "actions"}`.

## Gap Analysis

| Capability | IODescriptors | LEAPP | Joint Remapper | Status |
|---|---|---|---|---|
| Joint/body name metadata | ✅ | ✅ | ⚠️ custom JSON | **Use IODescriptors** |
| Observation semantic types | ✅ `observation_type` | ✅ `InputKindEnum` | ❌ hardcoded | **Use IODescriptors** |
| Articulation params export | ✅ | — | — | **Already exists** |
| Policy export (ONNX) | — | ✅ | — | **LEAPP handles this** |
| **Runtime joint permutation** | ❌ | ❌ | ✅ | **Still needed** |
| **Auto-detect remappable terms** | Metadata exists | Metadata exists | ⚠️ hardcoded | **Wire together** |
| **Checkpoint metadata** | Exists but not auto-saved | — | Custom JSON | **Auto-save IODescriptors** |
| **Physics param validation** | Static params available | — | — | **New work needed** |
| **Side-by-side trajectory comparison** | — | — | — | **New work needed** |
| **Perception transfer** | — | — | — | **Research problem** |

## Key Experimental Finding: Perception Does Not Transfer

We experimentally verified that a frozen ImageNet ResNet18 encoder trained with Newton Warp rendering produces **zero success** when evaluated with RTX rendering on the exact same physics scenario (DexSuite Kuka-Allegro cube lift):

| Configuration | Success Rate |
|---|---|
| Newton-trained → Newton eval | 6.69 |
| Newton-trained → RTX eval | **0.00** |

The ResNet features are completely different across renderers for the same scene geometry. This means vision-based sim2sim transfer requires either:
- Training with the target renderer
- Domain randomization that spans both renderers
- A shared encoder trained on both rendering styles
- World-model embeddings that abstract away rendering differences

State-based policies (joint_pos, joint_vel) transfer well with proper joint remapping — the physics gap is the main remaining concern.
