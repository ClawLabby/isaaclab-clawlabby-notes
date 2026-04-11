# Isaac Lab — ClawLabby Notes

Analyses, findings, and proposals related to [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Contents

### [clone-size/](clone-size/)
- **[isaac-lab-clone-size.md](clone-size/isaac-lab-clone-size.md)** — Analysis of why `git clone` downloads 1.1 GB (the `gh-pages` branch). Includes benchmarks showing a one-line fix (`--single-branch -b develop`) cuts clone time from 44s to 3s and size from 1.1 GB to 62 MB.
- **[proposed-gh-pages-migration.yaml](clone-size/proposed-gh-pages-migration.yaml)** — Drop-in replacement for the `deploy-docs` job that switches from branch-based GitHub Pages to the Actions deployment path, eliminating the bloated `gh-pages` branch entirely.
