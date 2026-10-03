# Derived model inputs

The public portfolio repository does **not** redistribute the upstream third-party source dataset used during model development.

The final solver expects two derived tables in this directory:

## `continuous_q_deformation_energy_curve.csv`

Required columns used by the public code:

- `state_s` — normalized wheel compliance state in `[0, 1]`
- `q` — normalized loading-work feature
- `deformation_increment_mm` — deformation increment associated with the state

## `continuous_soil_cost_surface.csv`

Required columns used by the public code:

- `terrain`
- `theta_deg`
- `state_s`
- `mobility_cost_J_per_m`
- `feasible`

The derived tables were generated in the original research workflow from the cited Lee et al. wheel data and literature/inherited terrain parameters. They should be regenerated from appropriately obtained source materials rather than treated as independent experimental measurements produced by this repository.

Compact final result tables and plots are included in `../results/` so the reported study outputs remain reviewable without redistributing the upstream dataset.
