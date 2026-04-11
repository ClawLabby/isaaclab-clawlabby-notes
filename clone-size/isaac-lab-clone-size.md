# Isaac Lab Clone Size Analysis

## Problem
Default `git clone https://github.com/isaac-sim/IsaacLab.git` takes ~44 seconds and produces a 1.1GB `.git` directory.
The actual source code is only ~71MB.

## Root Cause: `gh-pages` branch
The `gh-pages` branch contains **1.3 GB of generated HTML documentation** (versioned API docs for every release).
Across all historical commits, it accounts for **~94 GB of uncompressed blob data** (over 1 million blobs).
Git's delta compression squashes this to 1.1GB in the pack file, but it still dominates clone time and disk usage.

## Benchmark Results (same machine, same network)

| Clone Strategy | Time | .git Size | Working Tree | Total | Notes |
|---|---|---|---|---|---|
| `git clone` (default) | **44s** | **1.1 GB** | 58 MB | 1.2 GB | What most users do |
| `git clone --single-branch -b develop` | **3s** | **62 MB** | 65 MB | 127 MB | Single branch, full history |
| `git clone --depth 1 --single-branch -b develop` | **2.5s** | **36 MB** | 65 MB | 101 MB | Shallow, single branch |
| `git clone --filter=blob:none --single-branch -b develop` | **4s** | **42 MB** | 65 MB | 107 MB | Blobless (lazy fetch) |
| `git clone --filter=tree:0 --single-branch -b develop` | **4s** | **38 MB** | 65 MB | 103 MB | Treeless (lazier) |
| Fetch all branches **except** gh-pages | **4s** | **83 MB** | - | - | Negative refspec |

## Key Findings

1. **`gh-pages` is the entire problem.** Excluding it drops the clone from 1.1GB/44s to 83MB/4s (93% reduction).
2. **`--single-branch` is almost as good** — 62MB/3s because it naturally excludes gh-pages and other feature branches.
3. **Shallow clone saves very little on top of single-branch** — the history for develop is small; it's gh-pages bloat, not deep history.
4. **Partial clone (blobless/treeless) adds no meaningful benefit** when gh-pages is already excluded.

## Recommendations

### Quick Win (Documentation Change)
Update the install docs to recommend:
```bash
git clone --single-branch --branch develop https://github.com/isaac-sim/IsaacLab.git
```
This alone cuts clone from 44s/1.1GB to 3s/62MB — a **15x speedup and 95% size reduction**.

### Medium-Term (Repo Hygiene)
1. **Move gh-pages to a separate repo** (e.g., `isaac-sim/IsaacLab-docs`). GitHub Pages supports serving from a different repo.
2. Alternatively, **orphan the gh-pages branch** and use GitHub Actions to deploy from a clean orphan branch with no history. This won't fix existing clones but prevents further growth.

### Long-Term (Preventive)
1. Add `.gitattributes` with `*.html diff=html` or mark doc output paths as generated.
2. Consider using GitHub's built-in doc hosting (GitHub Pages from a separate deployment) rather than committing built HTML to the source repo.
3. For large binary assets (STL meshes, wheels), consider Git LFS — though currently these are small enough (~26MB total) to not matter.

## Impact
Most users only need the `develop` branch. The current docs don't mention `--single-branch`, so users unknowingly download 1GB of HTML documentation history they'll never use.
