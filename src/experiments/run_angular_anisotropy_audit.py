
from __future__ import annotations

from pathlib import Path
import math, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src" / "experiments"))
import run_connectivity_anisotropy_audit as conn


def octile_distance(dx_cells, dy_cells, h):
    a = min(abs(dx_cells), abs(dy_cells))
    b = max(abs(dx_cells), abs(dy_cells))
    return h * (math.sqrt(2) * a + (b - a))


def main():
    print("=== Angular anisotropy audit v2.80 ===")
    print("Quantifies residual path-length error of 8-neighbor routing.")
    print("Angles are measured from a grid axis over 0 to 45 degrees.")
    print("No wheel physics enters this audit.")

    R = 100.0
    h = 0.1
    angles = np.linspace(0.0, 45.0, 181)

    rows = []

    for deg in angles:
        th = math.radians(float(deg))
        x = R * math.cos(th)
        y = R * math.sin(th)

        # Snap endpoint to nearest grid node.
        ix = int(round(x / h))
        iy = int(round(y / h))

        xs = ix * h
        ys = iy * h
        euclid_to_snapped = math.hypot(xs, ys)

        d4 = h * (abs(ix) + abs(iy))
        d8 = octile_distance(ix, iy, h)

        err4 = (d4 - euclid_to_snapped) / euclid_to_snapped
        err8 = (d8 - euclid_to_snapped) / euclid_to_snapped

        rows.append(dict(
            angle_deg=float(deg),
            snapped_x_m=xs,
            snapped_y_m=ys,
            euclidean_distance_m=euclid_to_snapped,
            four_neighbor_distance_m=d4,
            eight_neighbor_distance_m=d8,
            four_neighbor_relative_error=err4,
            eight_neighbor_relative_error=err8,
        ))

    df = pd.DataFrame(rows)
    detail = OUT / "angular_anisotropy_sweep.csv"
    df.to_csv(detail, index=False)

    i4 = df.four_neighbor_relative_error.idxmax()
    i8 = df.eight_neighbor_relative_error.idxmax()

    summary = pd.DataFrame([dict(
        radius_m=R,
        grid_spacing_m=h,
        angles_tested=len(df),
        four_neighbor_max_relative_error=float(df.loc[i4, "four_neighbor_relative_error"]),
        four_neighbor_max_error_angle_deg=float(df.loc[i4, "angle_deg"]),
        eight_neighbor_max_relative_error=float(df.loc[i8, "eight_neighbor_relative_error"]),
        eight_neighbor_max_error_angle_deg=float(df.loc[i8, "angle_deg"]),
        eight_neighbor_error_at_0_deg=float(
            df.iloc[(df.angle_deg - 0.0).abs().argmin()].eight_neighbor_relative_error
        ),
        eight_neighbor_error_at_45_deg=float(
            df.iloc[(df.angle_deg - 45.0).abs().argmin()].eight_neighbor_relative_error
        ),
    )])

    summ = OUT / "angular_anisotropy_summary.csv"
    summary.to_csv(summ, index=False)

    r = summary.iloc[0]
    print("SUMMARY")
    print(
        f"  4-neighbor max relative path-length error="
        f"{r.four_neighbor_max_relative_error:.6f} "
        f"at ~{r.four_neighbor_max_error_angle_deg:.2f} deg"
    )
    print(
        f"  8-neighbor max relative path-length error="
        f"{r.eight_neighbor_max_relative_error:.6f} "
        f"at ~{r.eight_neighbor_max_error_angle_deg:.2f} deg"
    )
    print(
        f"  8-neighbor error at 0 deg="
        f"{r.eight_neighbor_error_at_0_deg:.6e}"
    )
    print(
        f"  8-neighbor error at 45 deg="
        f"{r.eight_neighbor_error_at_45_deg:.6e}"
    )
    print("INTERPRETATION")
    print(
        "  8-neighbor removes the severe Manhattan-grid bias but does not "
        "make the square grid rotationally invariant."
    )
    print(
        "  The remaining angular error is a numerical route-geometry limitation, "
        "not wheel or terrain physics."
    )
    print(
        "  No arbitrary pass/fail tolerance is imposed here; the magnitude is "
        "reported explicitly and can be compared with physical/model uncertainty."
    )
    print("OUTPUTS")
    print(" ", detail)
    print(" ", summ)


if __name__ == "__main__":
    main()
