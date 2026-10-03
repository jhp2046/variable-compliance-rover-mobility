
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

L_TOTAL = 8.0
BREAKS = [0.0, 2.0, 4.0, 6.0, 8.0]
TERRAINS = [
    "HARD_GROUND",
    "JLU_MARS1_LOOSE",
    "JLU_MARS1_DENSE",
    "HARD_GROUND",
]
RESOLUTIONS = [1.0, 0.5, 0.25, 0.125, 0.0625]
CRR = 0.002
LAM = 0.01
GRADIENT = 0.01
THETA = math.degrees(math.atan(GRADIENT))


def terrain_at_node(x):
    if x >= L_TOTAL - 1e-12:
        return TERRAINS[-1]
    for i, t in enumerate(TERRAINS):
        if BREAKS[i] - 1e-12 <= x < BREAKS[i+1] - 1e-12:
            return t
    raise RuntimeError(x)


def build_case(dx):
    n_edges = int(round(L_TOTAL / dx))
    xs = np.linspace(0.0, L_TOTAL, n_edges + 1)
    grid = np.array([[terrain_at_node(float(x)) for x in xs]], dtype=object)
    z = np.array([[GRADIENT * float(x) for x in xs]], dtype=float)
    return xs, grid, z


def local_cost_density(terrain, s, shape, qU, qD, soil_cache):
    rr = pred.W * CRR + pred.W * LAM * pred.hshape(shape, s, qU, qD)
    return pred.terrain_directional_cost(
        soil_cache, terrain, s, THETA, rr
    )


def continuum_reference(shape, qU, qD, soil_cache):
    """
    Piecewise-constant continuum benchmark for this special forced corridor.

    Fixed:
      one state minimizes the exact segment-length-weighted integral.

    Adaptive:
      with zero transition cost, each terrain segment independently uses its
      local optimal state. Since slope is constant, order inside the segment
      does not matter.

    This benchmark is specific to this audit geometry; it is not a replacement
    for graph routing on general 2-D maps.
    """
    seg_lengths = np.diff(BREAKS)

    fixed_candidates = []
    for s0 in pred.STATE_GRID:
        s = float(s0)
        total = 0.0
        feasible = True
        for L, t in zip(seg_lengths, TERRAINS):
            c = local_cost_density(t, s, shape, qU, qD, soil_cache)
            if not math.isfinite(c):
                feasible = False
                break
            total += float(L) * c
        if feasible:
            fixed_candidates.append((total, s))

    if not fixed_candidates:
        raise RuntimeError(f"No continuum fixed solution for {shape}")

    Jf, sf = min(fixed_candidates, key=lambda x: (x[0], x[1]))

    Ja = 0.0
    adaptive_states = []
    for L, t in zip(seg_lengths, TERRAINS):
        cand = []
        for s0 in pred.STATE_GRID:
            s = float(s0)
            c = local_cost_density(t, s, shape, qU, qD, soil_cache)
            if math.isfinite(c):
                cand.append((c, s))
        if not cand:
            raise RuntimeError(f"No local state for {t}, {shape}")
        cmin, smin = min(cand, key=lambda x: (x[0], x[1]))
        Ja += float(L) * cmin
        adaptive_states.append(smin)

    B = (Jf - Ja) / Jf
    return dict(
        continuum_J_fixed=Jf,
        continuum_s_fixed=sf,
        continuum_J_adaptive=Ja,
        continuum_B=B,
        continuum_adaptive_segment_states=repr(adaptive_states),
    )


def solve_grid(dx, shape, qU, qD, soil_cache):
    xs, grid, z = build_case(dx)
    res = pred.solve(
        grid, z, (0,0), (0,grid.shape[1]-1),
        shape, CRR, LAM, qU, qD, soil_cache, dx, {}
    )
    if res is None:
        raise RuntimeError(f"No grid solution dx={dx}, shape={shape}")
    return res


def main():
    print("=== Terrain-interface convergence audit v2.78 ===")
    print(f"physical corridor length={L_TOTAL} m")
    print(f"terrain boundaries={BREAKS}")
    print(f"resolutions={RESOLUTIONS} m")
    print(f"constant slope={THETA:.6f} deg")
    print(f"Crr_ref={CRR}, lambda={LAM}")
    print("Continuum benchmark uses exact physical segment lengths and zero transition cost.")
    print("Purpose: quantify numerical interface-discretization error.")

    qU, qD = pred.features()
    soil_cache = pred.load_soil_cache()

    rows = []
    refs = {}

    for shape in pred.SHAPES:
        ref = continuum_reference(shape, qU, qD, soil_cache)
        refs[shape] = ref

        print(f"SHAPE {shape}")
        print(
            f"  CONTINUUM: Jf={ref['continuum_J_fixed']:.10f} "
            f"Ja={ref['continuum_J_adaptive']:.10f} "
            f"B={ref['continuum_B']:.10f} "
            f"s_fixed={ref['continuum_s_fixed']:.2f}"
        )

        for dx in RESOLUTIONS:
            res = solve_grid(dx, shape, qU, qD, soil_cache)

            err_jf = float(res["J_fixed"] - ref["continuum_J_fixed"])
            err_ja = float(res["J_adaptive"] - ref["continuum_J_adaptive"])
            err_b = float(res["B_total"] - ref["continuum_B"])

            rows.append(dict(
                shape=shape,
                cell_length_m=dx,
                J_fixed_J=float(res["J_fixed"]),
                J_adaptive_J=float(res["J_adaptive"]),
                B_total=float(res["B_total"]),
                s_fixed=float(res["s_fixed"]),
                continuum_J_fixed_J=ref["continuum_J_fixed"],
                continuum_J_adaptive_J=ref["continuum_J_adaptive"],
                continuum_B=ref["continuum_B"],
                fixed_error_J=err_jf,
                adaptive_error_J=err_ja,
                B_error=err_b,
                relative_fixed_error=abs(err_jf)/max(abs(ref["continuum_J_fixed"]),1e-15),
                relative_adaptive_error=abs(err_ja)/max(abs(ref["continuum_J_adaptive"]),1e-15),
                absolute_B_error=abs(err_b),
            ))

            print(
                f"  dx={dx:7.4f}: "
                f"Jf={res['J_fixed']:.10f} "
                f"Ja={res['J_adaptive']:.10f} "
                f"B={res['B_total']:.10f} "
                f"|err_Ja|={abs(err_ja):.6e} "
                f"|err_B|={abs(err_b):.6e}"
            )

    df = pd.DataFrame(rows)
    detail_path = OUT / "terrain_interface_convergence.csv"
    df.to_csv(detail_path, index=False)

    summary_rows = []
    for shape, g in df.groupby("shape"):
        g = g.sort_values("cell_length_m", ascending=False).reset_index(drop=True)

        # observed error ratios E(dx)/E(dx/2), where available
        adaptive_ratios = []
        b_ratios = []
        for i in range(len(g)-1):
            e1 = abs(float(g.loc[i, "adaptive_error_J"]))
            e2 = abs(float(g.loc[i+1, "adaptive_error_J"]))
            if e2 > 1e-15:
                adaptive_ratios.append(e1/e2)

            b1 = abs(float(g.loc[i, "B_error"]))
            b2 = abs(float(g.loc[i+1, "B_error"]))
            if b2 > 1e-15:
                b_ratios.append(b1/b2)

        finest = g.iloc[-1]
        summary_rows.append(dict(
            shape=shape,
            continuum_J_fixed_J=float(finest.continuum_J_fixed_J),
            continuum_J_adaptive_J=float(finest.continuum_J_adaptive_J),
            continuum_B=float(finest.continuum_B),
            finest_dx_m=float(finest.cell_length_m),
            finest_relative_fixed_error=float(finest.relative_fixed_error),
            finest_relative_adaptive_error=float(finest.relative_adaptive_error),
            finest_absolute_B_error=float(finest.absolute_B_error),
            median_adaptive_error_halving_ratio=(
                float(np.median(adaptive_ratios)) if adaptive_ratios else math.nan
            ),
            median_B_error_halving_ratio=(
                float(np.median(b_ratios)) if b_ratios else math.nan
            ),
        ))

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / "terrain_interface_convergence_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("SUMMARY")
    for _, r in summary.iterrows():
        print(
            f"  {r['shape']}: finest dx={r.finest_dx_m:.4f} m "
            f"rel_err_Jf={r.finest_relative_fixed_error:.3e} "
            f"rel_err_Ja={r.finest_relative_adaptive_error:.3e} "
            f"abs_err_B={r.finest_absolute_B_error:.3e} "
            f"median_error_halving_ratio_Ja="
            f"{r.median_adaptive_error_halving_ratio:.3f}"
        )

    print("INTERPRETATION")
    print(
        "  Exact fixed-cost invariance is expected here because one global state "
        "is integrated over a piecewise-constant field with aligned boundaries."
    )
    print(
        "  Adaptive error comes from mixed-terrain boundary edges sharing one "
        "state. If the error approximately halves when dx halves, that is "
        "first-order convergence of the interface discretization."
    )
    print(
        "  The continuum benchmark is exact only for this forced 1-D, "
        "piecewise-constant, constant-slope, zero-transition-cost audit."
    )
    print(
        "  General 2-D mission maps should use a spatial resolution fine enough "
        "that further refinement changes the reported B negligibly relative to "
        "the much larger physical/model uncertainty."
    )
    print("OUTPUTS")
    print(" ", detail_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
