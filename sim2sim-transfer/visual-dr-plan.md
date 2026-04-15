# DextrAH-Style Visual DR: Implementation Plan

## Current State (what's running now)
- Image-level augmentation only: brightness, contrast, saturation jitter + Gaussian noise
- Applied as tensor ops in ResNet inference pipeline — works with `replicate_physics=True`
- No scene-level randomization

## What DextrAH Does (target)
1. **Object color/texture randomization** — random colors + procedural textures on manipulated object
2. **Table/surface texture randomization** — random materials on the table
3. **Robot appearance randomization** — color perturbation on robot links
4. **Background randomization** — random dome light texture/color
5. **Lighting randomization** — dome light intensity + color, optional additional point/distant lights
6. **Camera position jitter** — small perturbation of camera pose each episode
7. **Image-level augmentation** — color jitter, noise (what we have now)

## Implementation Plan

### Tier 1: Pure tensor ops (no scene changes, works today)
**Already done:**
- ✅ Brightness jitter
- ✅ Contrast jitter
- ✅ Saturation jitter
- ✅ Gaussian noise

**Add:**
- Hue shift (RGB→HSV→shift→RGB, pure tensor ops)
- Gamma correction jitter

### Tier 2: Scene-level via USD/Replicator (requires `replicate_physics=False`)
**The `replicate_physics=False` problem:**
- Replicator `randomize_visual_color` and `randomize_visual_texture_material` require this
- Setting it crashes PhysX tensor views (prim deletion invalidates sim view)
- **Root cause:** Replicator material modification can trigger prim tree changes that invalidate
  PhysX's cached prim handles

**Potential fixes:**
1. **Separate visual and physics prims** — ensure material changes don't affect physics shapes
2. **Use omni.replicator.core functional API** — may allow material changes without prim deletion
3. **Pure Python Replicator alternative** — Gavriel mentioned this exists; may bypass the issue
4. **Pre-create material variants** — create N material variants at scene setup, randomly assign
   per env per reset via USD relationship changes (no prim deletion)

### Tier 3: Lighting randomization (global, no per-env conflict)
**Dome light** is a global scene prim (`/World/skyLight`). We can modify it without per-env issues:
- Intensity: randomize between episodes (e.g., 200-1500)
- Color temperature: shift warm/cool
- Texture: swap between HDR sky textures (limited set)
- Direction: rotate the dome light

**Additional lights:** Add randomized distant/sphere lights
- Random position, intensity, color per episode
- Can be done via USD attribute writes (no Replicator needed)

**Implementation:** New EventTerm `randomize_scene_lighting` that modifies light prim attributes
directly via `UsdLux` API. Since lights are global (not per-env), no `replicate_physics` conflict.

### Tier 4: Camera position jitter
**TiledCamera offset** is configured at scene creation time in `camera_cfg.py`:
```python
offset=TiledCameraCfg.OffsetCfg(
    pos=(0.57, -0.8, 0.5),
    rot=(0.6124, 0.3536, 0.3536, 0.6124),
)
```

**Options:**
1. **Per-env jitter at reset** — modify camera prim transform via Fabric/USD. The camera is a
   per-env prim (`/World/envs/env_*/Camera`), so each env can have a slightly different view.
   Need to check if TiledCamera supports runtime offset updates.
2. **Global jitter per episode** — simpler, change the camera offset for all envs at once.
   Less diverse but easier to implement.
3. **Random crop** — instead of moving the camera, render at slightly larger resolution and
   take a random crop. Pure tensor op, no scene changes. Most practical.

### Tier 5: Object/table/robot color via Newton Warp renderer
For Newton Warp renderer (not RTX), we already have working visual DR via direct `shape_colors`
manipulation. This could be extended:
- Already done: per-env shape color randomization, background color, checkerboard textures
- Could add: more texture patterns, per-shape texture variation

## Priority Order for Implementation

1. **Hue shift + gamma** (Tier 1) — 30 minutes, pure tensor ops
2. **Lighting randomization** (Tier 3) — 2-3 hours, global USD attrs, no per-env conflict
3. **Camera jitter via random crop** (Tier 4 option 3) — 1 hour, pure tensor ops
4. **Material pre-creation for color DR** (Tier 2 option 4) — 4-6 hours, avoids Replicator crash
5. **Full Replicator texture DR** (Tier 2) — needs `replicate_physics=False` fix or pure Python alt

## GPU Capacity for Experimentation
Current training uses all 8 GPUs (4 local + 4 T2). No spare capacity for experiments until
training completes (~16-20h remaining). Can prototype code changes and test once a GPU frees up,
or stop one run briefly for validation.

## Alternative: Pure Python Replicator
Gavriel mentioned a pure Python version of Replicator randomization. If this can modify materials
without triggering prim deletion/invalidation, it would unlock Tier 2 immediately. Need to
investigate what this is and how to integrate it.
