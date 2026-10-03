# Variable-Compliance Rover Mobility

**Reduced-order terramechanics and route optimization for adaptive rover wheels on heterogeneous terrain**

The project develops a source-informed computational framework for asking a specific engineering question:

> **When can changing wheel compliance along a route provide a mobility advantage over the best globally fixed compliance state?**

The model couples a continuous wheel-compliance state with reduced Bekker–Wong terramechanics, terrain-dependent feasibility, obstacle constraints, and graph-based route optimization. It deliberately reports an **ideal mobility benefit**, not a mission-calibrated energy prediction.

[**Read the research report →**](report/complete_research_report.pdf)  
[**Read the technical formulation →**](report/technical_formulation.pdf)

---

## Project at a glance

The computational chain is:

```text
wheel state s
    ↓
force–deformation response D(F, s)
    ↓
contact geometry
    ↓
terrain / soil response
    ↓
state-dependent edge cost
    ↓
fixed vs. adaptive route optimization
```

The comparison is intentionally conservative. The fixed-wheel baseline is allowed to choose both its **best constant compliance state** and its **best route**. The adaptive solution may vary compliance edge by edge and may also reroute.

The central metric is

```text
B_ideal = (J*_fixed − J*_adaptive) / J*_fixed
```

where `J*_fixed` is the globally optimized fixed-compliance route cost and `J*_adaptive` is the optimized adaptive-compliance route cost.

---

## What I built

- A continuous compliance representation rather than a binary “soft/stiff” wheel.
- A reduced contact model that maps wheel deformation into contact length, pressure, sinkage, and terrain interaction.
- Terrain-dependent feasibility using reduced Bekker–Wong pressure–sinkage and traction limits.
- Directed obstacle constraints based only on experimentally represented obstacle heights.
- An 8-neighbor graph planner with physical cardinal/diagonal edge lengths and signed slope.
- Separate optimization for the globally best fixed state/route and the adaptive state sequence/route.
- A decomposition of adaptive benefit into **state adaptation** and **rerouting** contributions.
- Numerical audits for solver equivalence, grid resolution, routing anisotropy, obstacle thresholds, and negative controls.

---

## Representative results

### State adaptation versus rerouting

For the corrected representative heterogeneous map, the ideal benefit remained positive across all four rolling-loss sensitivity forms. Most of the modeled advantage came from changing compliance state along the route; rerouting was a smaller secondary contribution.

### Spatial organization matters

The same terrain composition can produce different outcomes depending on spatial arrangement. In the controlled hard-ground / JLU Mars-1 loose ensemble, clustered and random layouts did not show a single monotonic ordering across all terrain fractions.

### Obstacle feasibility is treated as a constraint, not invented energy

Only exact experimentally represented obstacle heights are used. Obstacle measurements define the minimum admissible compliance state; they are not converted into an unsupported impact- or obstacle-energy penalty.

---

## Validation philosophy

A major part of this project was identifying and removing numerical artifacts rather than tuning the model to produce a desired adaptive benefit.

Examples include:

- replacing 4-neighbor Manhattan routing after quantifying its 45° path-length bias;
- using correct diagonal edge lengths in the 8-neighbor grid;
- checking terrain-interface convergence under grid refinement;
- verifying the accelerated/cached solver against the original implementation to floating-point roundoff;
- testing an all-hard negative control where the adaptive benefit should be zero;
- separating source-derived quantities, literature-derived terrain parameters, numerical choices, and uncalibrated sensitivity assumptions.

See [`docs/`](docs/) for the corresponding modeling and numerical-boundary notes.

---

## Scientific boundary

The current framework is **not** a validated mission-energy predictor. The physical calibration remains tied to the source wheel geometry/load domain, and several quantities are intentionally treated as sensitivity assumptions rather than measured inputs—most importantly compliance-dependent cyclic rolling loss.

The project is therefore most useful for studying **mechanisms, tradeoffs, numerical behavior, and conditions under which adaptive compliance may be advantageous**, rather than predicting exact rover energy savings.

---

## Repository structure

```text
.
├── README.md
├── report/
│   ├── complete_research_report.pdf
│   ├── technical_formulation.pdf
│   └── source/                  # LaTeX source and report figures
├── src/
│   └── experiments/             # final solver, ensemble, and validation scripts
├── examples/                    # small terrain/elevation/obstacle examples
├── results/                     # final figures and compact result tables
├── docs/                        # numerical/modeling boundary notes
└── data/
    └── derived/                 # expected locally regenerated derived inputs
```

---

## Code and data note

The final research code originally consumed two **derived** input tables generated from the source-paper data during the research workflow:

- `continuous_q_deformation_energy_curve.csv`
- `continuous_soil_cost_surface.csv`

They are intentionally not bundled in this public portfolio package because the upstream experimental/source data are third-party materials. The public scripts have been cleaned so they look for these files under `data/derived/` rather than a personal-machine path. See [`data/README.md`](data/README.md) for the expected schemas and provenance note.

The final figures and compact result summaries used in the report are included under [`results/`](results/).

---

## Running the code

Python dependencies are minimal:

```bash
python -m pip install -r requirements.txt
```

After regenerating or supplying the two derived input tables described in `data/README.md`, representative checks can be run from the repository root, for example:

```bash
python src/experiments/run_fast_vs_original_regression.py
python src/experiments/run_obstacle_height_regression.py
python src/experiments/run_terrain_interface_convergence.py
python src/experiments/run_angular_anisotropy_audit.py
```

The corrected spatial ensemble is in:

```bash
python src/experiments/run_corrected_2d_spatial_ensemble_fast.py
```

---

## References

The wheel constitutive/obstacle calibration is based on:

J.-Y. Lee et al., **“Variable-stiffness–morphing wheel inspired by the surface tension of a liquid droplet,”** *Science Robotics*, 9(93), eadl2067, 2024. DOI: `10.1126/scirobotics.adl2067`.

The reduced terrain model follows classical Bekker–Wong terramechanics; full references and model provenance are provided in the research report and technical formulation.
