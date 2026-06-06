# Isaac Lab — ClawLabby Notes

Analyses, findings, and proposals related to [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Contents

### [sim2sim-transfer/](sim2sim-transfer/)
- **[analysis.md](sim2sim-transfer/analysis.md)** — Analysis of cross-backend policy transfer (Newton ↔ PhysX). Reviews existing infrastructure (IODescriptors, LEAPP PR #5105), identifies gaps, and documents the experimental finding that frozen ResNet features do not transfer across renderers.
- **[plan.md](sim2sim-transfer/plan.md)** — Implementation plan for seamless sim2sim transfer inside Isaac Lab. Four phases: IODescriptor-based metadata (Phase 1), `--sim2sim` play mode (Phase 2), trajectory comparison and parameter matching (Phase 3), perception transfer (Phase 4).

### [mjwarp-elliptic-friction-nan/](mjwarp-elliptic-friction-nan/)
- **[README.md](mjwarp-elliptic-friction-nan/README.md)** — Investigation report for elliptic-cone NaNs in Newton/MJWarp when externally supplied contacts contain zero or sub-`MJ_MINMU` friction. Includes root cause, before/after repro results, validation tests, generated plots, and the MJWarp patch.

### [clone-size/](clone-size/)
- **[isaac-lab-clone-size.md](clone-size/isaac-lab-clone-size.md)** — Analysis of why `git clone` downloads 1.1 GB (the `gh-pages` branch). Includes benchmarks showing a one-line fix (`--single-branch -b develop`) cuts clone time from 44s to 3s and size from 1.1 GB to 62 MB.
- **[proposed-gh-pages-migration.yaml](clone-size/proposed-gh-pages-migration.yaml)** — Drop-in replacement for the `deploy-docs` job that switches from branch-based GitHub Pages to the Actions deployment path, eliminating the bloated `gh-pages` branch entirely.
