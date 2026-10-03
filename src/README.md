# Source code

The `experiments/` directory contains the final public-facing analysis scripts from the research workflow.

Key entry points:

- `run_map_predictor_with_elevation.py` — core terrain/elevation map predictor and compliance-state cost model.
- `run_corrected_2d_spatial_ensemble_fast.py` — corrected 8-neighbor spatial ensemble with cached cost evaluation.
- `run_fast_vs_original_regression.py` — numerical equivalence check between accelerated and original solvers.
- `run_obstacle_height_regression.py` — exact tested-height obstacle-feasibility regression.
- `run_terrain_interface_convergence.py` — grid-refinement / terrain-interface convergence audit.
- `run_connectivity_anisotropy_audit.py` and `run_angular_anisotropy_audit.py` — routing-geometry audits.
- `run_physical_resolution_invariance.py` — physical-resolution consistency check.

The scripts expect the two derived input tables documented in `../data/README.md`.
