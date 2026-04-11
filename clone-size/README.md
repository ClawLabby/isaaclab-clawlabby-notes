# Isaac Lab Clone Size — Analysis & Fix

## Problem

`git clone https://github.com/isaac-sim/IsaacLab.git` downloads **1.1 GB** and takes ~44 seconds.
The actual source code is ~71 MB. The bloat comes from the `gh-pages` branch, which accumulates
built HTML documentation across every deployment.

**[Full analysis →](isaac-lab-clone-size.md)**

## Immediate Fix (Documentation)

Recommend `--single-branch` in the install docs. This alone gives a **95% reduction**:

```bash
# Before: 44s, 1.1 GB
git clone https://github.com/isaac-sim/IsaacLab.git

# After: 3s, 62 MB
git clone --single-branch --branch develop https://github.com/isaac-sim/IsaacLab.git
```

Users only need `develop` (or `main`). The `gh-pages` branch is never checked out — it's
only used by GitHub Pages internally.

## Permanent Fix: Migrate to GitHub Actions Pages Deployment

The root cause is that `peaceiris/actions-gh-pages` pushes built HTML to a git branch on
every deployment. Over time, this branch accumulates gigabytes of history.

The fix: switch to GitHub's native Actions-based Pages deployment, which uploads an artifact
directly — no branch involved.

### What changes

| | Current | Proposed |
|---|---|---|
| **Deploy action** | `peaceiris/actions-gh-pages@v3` | `actions/deploy-pages@v4` |
| **Mechanism** | Pushes HTML to `gh-pages` branch | Uploads artifact, no branch |
| **History accumulation** | Every deploy adds to git history | No git history at all |
| **URLs** | `isaac-sim.github.io/IsaacLab/{version}/` | Identical — no change |

### What does NOT change

- The **build process** (`make multi-docs`) — identical, still produces versioned subdirectories
- The **URL structure** — `main/`, `v2.3.2/`, `v2.0.0/`, `v1.0.0/`, etc. all preserved
- The **trigger conditions** — same branches (main, devel, release/*)
- The **multi-version assembly** — `build-docs` still runs `git fetch --prune --unshallow --tags`
  and `make multi-docs`, which builds all versions into `docs/_build/`

The artifact deployment is a **full replacement** each time (not incremental), but that's
fine because `make multi-docs` already assembles the complete site with all versions on every run.

### Migration Steps

1. **Update the workflow** — replace the `deploy-docs` job in `.github/workflows/docs.yaml`
   with the version in [`proposed-gh-pages-migration.yaml`](proposed-gh-pages-migration.yaml).
   Only the `deploy-docs` job changes; `check-secrets` and `build-docs` stay identical.

2. **Change repo Pages settings** — go to the repo **Settings → Pages → Source** and
   select **"GitHub Actions"** instead of "Deploy from a branch".

3. **Verify** — push to `main` or a `release/*` branch, confirm the docs deploy correctly
   and all version URLs still work:
   - https://isaac-sim.github.io/IsaacLab/main/index.html
   - https://isaac-sim.github.io/IsaacLab/v2.3.2/index.html
   - https://isaac-sim.github.io/IsaacLab/v2.0.0/index.html
   - https://isaac-sim.github.io/IsaacLab/v1.0.0/index.html

4. **Delete the `gh-pages` branch** — once verified, delete it. This is what frees the
   1+ GB from future clones. Existing clones can run `git remote prune origin` to clean up.

### Current vs Proposed Workflow (diff)

The only job that changes is `deploy-docs`. Here's a side-by-side:

**Current (`docs.yaml` lines 53–65):**
```yaml
  deploy-docs:
    name: Deploy Docs
    runs-on: ubuntu-latest
    needs: [check-secrets, build-docs]
    if: needs.check-secrets.outputs.trigger-deploy == 'true'

    steps:
    - name: Download docs artifact
      uses: actions/download-artifact@v4
      with:
        name: docs-html
        path: ./docs/_build

    - name: Deploy to gh-pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build
```

**Proposed ([`proposed-gh-pages-migration.yaml`](proposed-gh-pages-migration.yaml)):**
```yaml
  deploy-docs:
    name: Deploy Docs
    runs-on: ubuntu-latest
    needs: [check-secrets, build-docs]
    if: needs.check-secrets.outputs.trigger-deploy == 'true'
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
    - name: Download docs artifact
      uses: actions/download-artifact@v4
      with:
        name: docs-html
        path: ./docs/_build

    - name: Upload Pages artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: ./docs/_build

    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v4
```

Key additions:
- **`permissions`** — `pages: write` and `id-token: write` are required for the Actions deployment token
- **`environment`** — links the deployment to the `github-pages` environment in GitHub's UI (shows deployment status, URL)
- **`upload-pages-artifact`** — packages the build output in the format `deploy-pages` expects
- **`deploy-pages`** — replaces the branch push with a direct artifact deployment
