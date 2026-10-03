
from __future__ import annotations

from pathlib import Path
import math, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src" / "experiments"))
import run_map_predictor_with_elevation as pred


# Physical 8 m corridor:
# 0-2 m hard
# 2-4 m JLU loose
# 4-6 m JLU dense
# 6-8 m hard
#
# Elevation rises linearly by 0.08 m over 8 m:
# dz/dx = 0.01 -> slope atan(0.01) = 0.57294 deg.
#
# Each grid represents the same underlying piecewise-constant physical field.
L_TOTAL = 8.0
BREAKS = [0.0, 2.0, 4.0, 6.0, 8.0]
TERRAINS = [
    "HARD_GROUND",
    "JLU_MARS1_LOOSE",
    "JLU_MARS1_DENSE",
    "HARD_GROUND",
]
RESOLUTIONS = [1.0, 0.5, 0.25]
CRR = 0.002
LAM = 0.01


def terrain_at_node(x):
    # Node label uses the terrain immediately to the right, except at x=L.
    # This convention is held identical across resolutions.
    if x >= L_TOTAL - 1e-12:
        return TERRAINS[-1]
    for i in range(len(TERRAINS)):
        if BREAKS[i] - 1e-12 <= x < BREAKS[i+1] - 1e-12:
            return TERRAINS[i]
    raise RuntimeError(x)


def build_case(dx):
    n_edges = int(round(L_TOTAL / dx))
    xs = np.linspace(0.0, L_TOTAL, n_edges + 1)
    grid = np.array([[terrain_at_node(float(x)) for x in xs]], dtype=object)
    # linear elevation field
    z = np.array([[0.01 * float(x) for x in xs]], dtype=float)
    return xs, grid, z


def solve_case(dx, shape, qU, qD, soil_cache):
    xs, grid, z = build_case(dx)
    start = (0, 0)
    goal = (0, grid.shape[1]-1)
    res = pred.solve(
        grid, z, start, goal, shape, CRR, LAM,
        qU, qD, soil_cache, dx, {}
    )
    if res is None:
        raise RuntimeError(f"No feasible route for dx={dx}, shape={shape}")
    return xs, grid, z, res


def main():
    print("=== Physical grid-resolution invariance audit v2.77 ===")
    print(f"physical_length={L_TOTAL} m")
    print(f"resolutions={RESOLUTIONS} m")
    print("same terrain boundaries at x=2,4,6 m")
    print("same linear elevation gradient dz/dx=0.01")
    print(f"Crr_ref={CRR}, lambda={LAM}")
    print("No obstacles in this audit.")
    print("Purpose: test discretization, not physical calibration.")

    qU, qD = pred.features()
    soil_cache = pred.load_soil_cache()

    rows = []

    for shape in pred.SHAPES:
        print(f"SHAPE {shape}")
        for dx in RESOLUTIONS:
            xs, grid, z, res = solve_case(dx, shape, qU, qD, soil_cache)
            slope_expected = math.degrees(math.atan(0.01))

            # Every edge should have same signed slope.
            slopes = []
            route = eval(res["adaptive_route"])
            for u, v in pred.pedges(route):
                slopes.append(pred.slope_deg(z, u, v, dx))

            slope_err = max(abs(s - slope_expected) for s in slopes)

            rows.append(dict(
                shape=shape,
                cell_length_m=dx,
                physical_length_m=L_TOTAL,
                number_of_edges=len(route)-1,
                J_fixed_J=float(res["J_fixed"]),
                J_adaptive_J=float(res["J_adaptive"]),
                delta_state_J=float(res["delta_state"]),
                delta_route_J=float(res["delta_route"]),
                B_total=float(res["B_total"]),
                s_fixed=float(res["s_fixed"]),
                max_edge_slope_error_deg=float(slope_err),
            ))

            print(
                f"  dx={dx:.2f} m edges={len(route)-1:2d} "
                f"Jf={res['J_fixed']:.8f} "
                f"Ja={res['J_adaptive']:.8f} "
                f"B={res['B_total']:.8f} "
                f"s_fixed={res['s_fixed']:.2f}"
            )

    df = pd.DataFrame(rows)
    detail_path = OUT / "physical_resolution_invariance.csv"
    df.to_csv(detail_path, index=False)

    summary_rows = []
    for shape, g in df.groupby("shape"):
        g = g.sort_values("cell_length_m", ascending=False)
        jfr = float(g.J_fixed_J.max() - g.J_fixed_J.min())
        jar = float(g.J_adaptive_J.max() - g.J_adaptive_J.min())
        br = float(g.B_total.max() - g.B_total.min())

        # Relative to 1 m reference
        ref = g.iloc[(g.cell_length_m - 1.0).abs().argmin()]
        rel_jf = float(
            np.max(np.abs(g.J_fixed_J - ref.J_fixed_J))
            / max(abs(ref.J_fixed_J), 1e-15)
        )
        rel_ja = float(
            np.max(np.abs(g.J_adaptive_J - ref.J_adaptive_J))
            / max(abs(ref.J_adaptive_J), 1e-15)
        )

        summary_rows.append(dict(
            shape=shape,
            fixed_cost_span_J=jfr,
            adaptive_cost_span_J=jar,
            B_span=br,
            max_relative_fixed_cost_change_vs_1m=rel_jf,
            max_relative_adaptive_cost_change_vs_1m=rel_ja,
            fixed_state_unique_count=int(g.s_fixed.nunique()),
            max_slope_error_deg=float(g.max_edge_slope_error_deg.max()),
        ))

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / "physical_resolution_invariance_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("SUMMARY")
    for _, r in summary.iterrows():
        print(
            f"  {r['shape']}: "
            f"Jf_rel_change={r.max_relative_fixed_cost_change_vs_1m:.3e} "
            f"Ja_rel_change={r.max_relative_adaptive_cost_change_vs_1m:.3e} "
            f"B_span={r.B_span:.3e} "
            f"fixed_state_unique={int(r.fixed_state_unique_count)}"
        )

    print("INTERPRETATION")
    print(
        "  If results converge or remain nearly invariant as dx is refined, "
        "cell length is acting as physical map resolution rather than an "
        "arbitrary energy-tuning parameter."
    )
    print(
        "  Any residual differences are caused by terrain-boundary representation "
        "and endpoint-averaged edge discretization and must be reported."
    )
    print(
        "  This audit does not identify the correct mission map resolution; that "
        "must come from the spatial resolution of the actual terrain data."
    )
    print("OUTPUTS")
    print(" ", detail_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
