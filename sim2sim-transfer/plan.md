# Sim2Sim Transfer: Implementation Plan

**Author:** ClawLabby  
**Date:** 2026-04-14  
**Goal:** Seamless cross-backend policy transfer inside Isaac Lab

---

## Phase 1: Foundation — Leverage Existing Infrastructure
**Effort:** ~1 week

### 1.1 Auto-save IODescriptors at Checkpoint Time

**Change:** In the rsl_rl training wrapper (`RslRlOnPolicyRunnerCfg` save path), call `env.export_IO_descriptors(checkpoint_dir)` whenever a checkpoint is saved.

This gives every checkpoint a machine-readable `IO_descriptors.yaml` containing:
- Joint names and ordering per observation term
- Body names and ordering per body-state term
- Observation term shapes, types, and semantic labels
- Action space description
- Articulation metadata (limits, defaults, damping, stiffness)
- Scene metadata (physics_dt, decimation)

**Files to modify:**
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py` — expose `export_IO_descriptors` 
- `scripts/reinforcement_learning/rsl_rl/train.py` — call at checkpoint save

### 1.2 Refactor JointRemapper to Use IODescriptors

**Change:** Replace custom `joint_names.json` loading with IODescriptor YAML parsing.

Instead of:
```python
train_joint_names = JointRemapper.load_joint_names(checkpoint_dir)  # custom JSON
```

Do:
```python
train_descriptors = Sim2SimAdapter.load_descriptors(checkpoint_dir)  # standard YAML
```

Use `observation_type == "JointState"` to auto-detect which terms need remapping, instead of hardcoding `{"joint_pos", "joint_vel", "actions"}`. Use `record_body_names` metadata for body-ordered terms.

### 1.3 Create `isaaclab.utils.sim2sim` Module

New module with:

```python
class Sim2SimAdapter:
    """Automatic cross-backend policy adaptation."""
    
    @staticmethod
    def load_descriptors(checkpoint_dir: str) -> dict:
        """Load IODescriptor YAML from checkpoint directory."""
        
    @staticmethod  
    def from_checkpoint(checkpoint_dir: str, eval_env: ManagerBasedRLEnv) -> Sim2SimAdapter:
        """Create adapter by comparing training descriptors against live eval env."""
    
    def needs_remapping(self) -> bool:
        """Whether any observation/action remapping is needed."""
    
    def remap_observations(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap all joint/body-ordered terms from eval order to training order."""
    
    def remap_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Remap actions from training order to eval order."""
    
    def validate_physics(self) -> list[str]:
        """Compare training vs eval physics params, return warnings."""
    
    def summary(self) -> str:
        """Human-readable summary of what's being remapped and any warnings."""
```

Core permutation logic comes from my existing `JointRemapper` (GPU index tensors, history-aware reshaping). The new wrapper adds IODescriptor-driven term detection and physics validation.

---

## Phase 2: Integration — Make It Seamless
**Effort:** ~1 week

### 2.1 Add `--sim2sim` Flag to play.py

```bash
python play.py --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
    'presets=physx' \
    --checkpoint logs/newton_run/model_best.pt \
    --sim2sim
```

When `--sim2sim` is active:
1. Load `IO_descriptors.yaml` from checkpoint directory
2. Create `Sim2SimAdapter` comparing training vs eval environment
3. Print summary: which joints are remapped, physics param warnings
4. Wrap policy transparently — zero manual configuration

### 2.2 Physics Parameter Validation

When `--sim2sim` is active, compare:
- Physics backend (Newton vs PhysX)
- Solver type and iteration counts
- Contact model parameters
- Timestep and decimation
- Actuator gains (stiffness, damping per joint)
- Joint limits

Warn on significant differences. Don't block — just inform the user.

### 2.3 Framework-Agnostic Adapter

The `Sim2SimAdapter` should work with any RL framework, not just rsl_rl. It takes raw observation dicts and action tensors — no framework-specific dependencies.

---

## Phase 3: Cross-Backend Validation Tools
**Effort:** ~2 weeks

### 3.1 Side-by-Side Trajectory Comparison

New script: `scripts/tools/sim2sim_compare.py`

Given a reference trajectory (sequence of joint position commands):
1. Step it in backend A (e.g., Newton)
2. Step it in backend B (e.g., PhysX)
3. Compare resulting joint positions, velocities, contact forces
4. Output: per-joint MSE, max deviation, contact event alignment

This quantifies the sim2sim gap for a specific robot and task.

### 3.2 Lightweight Parameter Matching

Given the trajectory comparison results:
1. Identify which parameters cause the largest deviations
2. Use gradient-free optimization (CMA-ES or similar) to adjust backend B's parameters to minimize the gap
3. Output: optimized parameter set for backend B

This is lightweight system identification — not training a model, just tuning physics parameters.

### 3.3 LEAPP Integration (if PR #5105 merges)

If LEAPP merges, `Sim2SimAdapter` can use LEAPP's semantic annotations for richer remapping:
- Sensor data (IMU, contact forces) — not just joint state
- Command terms (velocity commands, pose commands)
- The `DirectDeploymentEnv` pattern could support "deploy to alternate backend"

---

## Phase 4: Perception Transfer (Research)
**Effort:** Ongoing

### 4.1 Renderer-Agnostic Feature Encoders

Train visual encoders that produce similar features regardless of renderer:
- Domain randomization spanning Newton Warp + RTX rendering styles
- Contrastive learning: same scene, different renderers → similar features
- Feature alignment loss during training

### 4.2 World Embedding Approach

Per earlier discussion with Gavriel:
- Sim provides exact dynamics (no learned world model needed)
- Video pretraining provides perception encoder  
- RL operates in compact embedding space
- Real data mixed in to bridge sim-real gap
- PBT likely best when vision encoder is pretrained

### 4.3 Newton Warp Visual Domain Randomization

Newton Warp renderer currently lacks:
- Texture randomization (no textures at all yet)
- Lighting randomization
- Material property randomization

Adding these would enable Newton-only visual DR that better prepares policies for RTX or real-world transfer.

---

## Success Criteria

**Phase 1-2 done means:** A researcher can train on Newton, then `play.py --sim2sim 'presets=physx'` and it Just Works™ for state-based policies. Joint remapping is automatic, physics differences are flagged.

**Phase 3 done means:** The sim2sim gap is quantified and can be partially closed through parameter optimization.

**Phase 4 done means:** Vision-based policies can transfer across renderers, closing the final gap for fully autonomous sim2sim.

---

## Relationship to Existing Work

| System | Role in Sim2Sim |
|---|---|
| IODescriptors | **Metadata source** — joint names, semantic types, shapes |
| LEAPP | **Future enrichment** — deeper semantic annotations, export pipeline |
| JointRemapper | **Runtime engine** — GPU-efficient permutation |
| Sim2SimAdapter | **Orchestrator** — ties metadata to runtime, adds validation |
| export_articulations_data | **Physics comparison** — static parameter extraction |
