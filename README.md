# Smooth graph error analysis

Reproducibility code for **When Accurate Reachability States Give Inaccurate
Centrality**, by Walid Guettala and László Gulyás.

This repository studies approximation errors in a specified sigmoid reachability
recurrence and finite-horizon soft shortest paths. It contains reference
implementations, numerical checks, recorded results, and figure scripts. It is
not a trained graph model or a claim that all differentiable graph algorithms fail.

## Main results

- Neighborhood sums amplify small false activations. Connected barbells exhibit
  an exact critical window and matching leading temperature thresholds under
  bounded preactivation perturbations.
- Replicated path components have vanishing maximum state error but persistent
  error in an unmasked mean-distance output. Exact component support removes
  this accumulation mechanism.
- Weighted triangles with shrinking costs have vanishing distance error but
  persistent absolute reciprocal-closeness error. A stronger temperature
  schedule restores convergence. Relative error already vanishes.

## Setup and tests

Use Python 3.11. The recorded runs used NumPy 2.0.1 and NetworkX 3.5.
PyTorch is required for the gradient test, not for the NumPy sweeps.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

The tests compare Boolean execution with BFS, symmetry reductions with explicit
graphs, soft paths with independent exact targets, and automatic gradients with
finite differences. All calculations can run on CPUs.

## Recorded results

`results/sofsem2027_theory/` contains 242 scaling cases, 1,210 critical-window
cases, 270 perturbation cases, precision checks, and explicit masking/exact-zero
controls. Very large graph orders are represented by exact symmetry reductions,
not by materialized graphs or runtime benchmarks.

`results/sofsem2027_shortest_paths/` contains 56 unweighted, 324 weighted, and
176 controlled cases, together with example full matrices. Twelve weighted
settings retain invalid-closeness flags for nonpositive predicted means.
These cases have not been dropped or clipped.

Original JSON manifests and SHA-256 hashes are preserved. The theory runner's
current hash is recorded as `render_script_sha256`, separately from the earlier
calculation-run hash. Numerical records were not regenerated for this release.
PDFs under `results/` are numerical plots, not manuscript drafts.

## Reproduce calculations

Run from the repository root. Use fresh output directories to keep the recorded
results intact. Full sweeps use these commands without `--quick`.

```sh
python src/run_reachability_theory.py --output runs/theory
python src/run_shortest_path_extension.py --output runs/paths
```

The explicit control runner writes to the canonical
`results/sofsem2027_theory/escape_controls.json`. Run the following only in a
disposable copy if you want to preserve that original file unchanged.

```sh
python -m src.run_reachability_escape_controls
```

## Reproduce figures

These scripts use the recorded results, not `runs/`. They generate numerical
plots only and do not require LaTeX or build the manuscript.

```sh
mkdir -p docs/figures
python docs/plot_sofsem.py
python docs/plot_shortest_paths.py
```

The second script also regenerates the tracked preview
`results/sofsem2027_shortest_paths/shortest_path_extension.png`.

## Repository contents

- `src/` — three numerical implementations and three experiment runners.
- `tests/` — unit and numerical consistency tests.
- `results/` — recorded numerical evidence and original manifests.
- `docs/plot_*.py` — figure reproduction scripts only.

## Citation and reuse

Author and manuscript metadata are in `CITATION.cff`. No acceptance or publication
is implied. Licensing is pending author and rights-holder confirmation. No
open-source license is granted by this snapshot. See `RIGHTS.md`.

Contact: guettalawalid@inf.elte.hu.
