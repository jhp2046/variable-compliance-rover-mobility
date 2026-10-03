
from __future__ import annotations

from pathlib import Path
import argparse, ast, heapq, math, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src" / "experiments"))
import run_map_predictor_with_elevation as base


def neighbors(grid, u, connectivity):
    nr, nc = grid.shape
    r, c = u

    if connectivity == 4:
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        dirs = [
            (-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(-1,1),(1,-1),(1,1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for dr, dc in dirs:
        v = (r+dr, c+dc)
        if 0 <= v[0] < nr and 0 <= v[1] < nc and grid[v] != "BLOCKED":
            # Prevent diagonal corner cutting through two blocked orthogonal neighbors.
            if abs(dr) == 1 and abs(dc) == 1:
                a = (r+dr, c)
                b = (r, c+dc)
                if grid[a] == "BLOCKED" and grid[b] == "BLOCKED":
                    continue
            yield v


def geometric_edge_length(u, v, cell_length):
    dr = abs(v[0] - u[0])
    dc = abs(v[1] - u[1])
    return cell_length * math.hypot(dr, dc)


def slope_deg(z, u, v, cell_length):
    L = geometric_edge_length(u, v, cell_length)
    dz = float(z[v] - z[u])
    return math.degrees(math.atan2(dz, L))


def obstacle_state_feasible(obstacles, u, v, s):
    # Obstacles remain explicit directed edges. If no exact obstacle edge is
    # specified, no obstacle constraint is inferred.
    h = obstacles.get((u, v))
    if h is None:
        return True
    return s + 1e-12 >= base.OBSTACLE_MIN_STATE[h]


def edge_cost(
    grid, z, u, v, s, shape, crr, lam,
    qU, qD, soil_cache, cell_length, obstacles
):
    if grid[u] == "BLOCKED" or grid[v] == "BLOCKED":
        return math.inf

    if not obstacle_state_feasible(obstacles, u, v, s):
        return math.inf

    L = geometric_edge_length(u, v, cell_length)
    theta = slope_deg(z, u, v, cell_length)

    h = base.hshape(shape, s, qU, qD)
    rr = base.W * crr + base.W * lam * h

    a = base.terrain_directional_cost(
        soil_cache, grid[u], s, theta, rr
    )
    b = base.terrain_directional_cost(
        soil_cache, grid[v], s, theta, rr
    )

    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf

    return L * 0.5 * (a + b)


def dijkstra(grid, start, goal, costfn, connectivity):
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue
        if u == goal:
            break

        for v in neighbors(grid, u, connectivity):
            c = costfn(u, v)
            if not math.isfinite(c):
                continue
            nd = d + c
            if nd < dist.get(v, math.inf) - 1e-15:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return math.inf, []

    p = [goal]
    while p[-1] != start:
        p.append(prev[p[-1]])
    p.reverse()
    return float(dist[goal]), p


def pedges(path):
    return list(zip(path[:-1], path[1:]))


def adaptive_path_cost(
    grid, z, path, shape, crr, lam,
    qU, qD, soil_cache, cell_length, obstacles
):
    total = 0.0
    states = []

    for u, v in pedges(path):
        vals = [
            (
                edge_cost(
                    grid, z, u, v, float(s), shape, crr, lam,
                    qU, qD, soil_cache, cell_length, obstacles
                ),
                float(s),
            )
            for s in base.STATE_GRID
        ]

        finite = [x for x in vals if math.isfinite(x[0])]
        if not finite:
            return math.inf, []

        c, s = min(finite, key=lambda x: (x[0], x[1]))
        total += c
        states.append(s)

    return total, states


def solve(
    grid, z, start, goal, shape, crr, lam,
    qU, qD, soil_cache, cell_length, obstacles,
    connectivity
):
    best = (math.inf, None, [])

    for s0 in base.STATE_GRID:
        s = float(s0)
        J, p = dijkstra(
            grid,
            start,
            goal,
            lambda u, v, ss=s: edge_cost(
                grid, z, u, v, ss, shape, crr, lam,
                qU, qD, soil_cache, cell_length, obstacles
            ),
            connectivity,
        )

        if J < best[0] - 1e-12:
            best = (J, s, p)

    Jf, sf, pf = best
    if not math.isfinite(Jf):
        return None

    Jaf, seqf = adaptive_path_cost(
        grid, z, pf, shape, crr, lam,
        qU, qD, soil_cache, cell_length, obstacles
    )

    edge_best = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            u = (r, c)
            if grid[u] == "BLOCKED":
                continue

            for v in neighbors(grid, u, connectivity):
                vals = [
                    (
                        edge_cost(
                            grid, z, u, v, float(s), shape, crr, lam,
                            qU, qD, soil_cache, cell_length, obstacles
                        ),
                        float(s),
                    )
                    for s in base.STATE_GRID
                ]
                finite = [x for x in vals if math.isfinite(x[0])]
                edge_best[(u, v)] = (
                    min(finite, key=lambda x: (x[0], x[1]))
                    if finite else (math.inf, math.nan)
                )

    Ja, pa = dijkstra(
        grid,
        start,
        goal,
        lambda u, v: edge_best[(u, v)][0],
        connectivity,
    )

    if not math.isfinite(Ja):
        return None

    seqa = [edge_best[(u, v)][1] for u, v in pedges(pa)]

    ds = Jf - Jaf
    dr = Jaf - Ja
    dt = Jf - Ja
    tol = 1e-8

    if ds < -tol or dr < -tol or Ja > Jf + tol or abs(ds + dr - dt) > tol:
        raise RuntimeError("routing/decomposition invariant failure")

    total_fixed_length = sum(
        geometric_edge_length(u, v, cell_length) for u, v in pedges(pf)
    )
    total_adaptive_length = sum(
        geometric_edge_length(u, v, cell_length) for u, v in pedges(pa)
    )

    return dict(
        J_fixed=Jf,
        s_fixed=sf,
        J_adaptive_on_fixed_route=Jaf,
        J_adaptive=Ja,
        delta_state=max(0.0, ds),
        delta_route=max(0.0, dr),
        delta_total=max(0.0, dt),
        B_total=(dt / Jf if Jf > 0 else math.nan),
        route_share=(dr / dt if dt > tol else math.nan),
        route_changed=(pf != pa),
        fixed_route=repr(pf),
        adaptive_route=repr(pa),
        adaptive_states_on_fixed_route=repr(seqf),
        adaptive_states=repr(seqa),
        fixed_route_length_m=total_fixed_length,
        adaptive_route_length_m=total_adaptive_length,
    )


def open_map_distance_audit():
    """
    Homogeneous hard-ground distance audit.

    Start=(n-1,0), goal=(0,n-1). Physical bounding square is 10m x 10m.
    Compare graph route length with exact Euclidean diagonal 10*sqrt(2).
    """
    print("OPEN_MAP_DISTANCE_AUDIT")
    rows = []

    for n in [11, 21, 41]:
        physical_side = 10.0
        dx = physical_side / (n - 1)
        grid = np.full((n, n), "HARD_GROUND", dtype=object)
        z = np.zeros((n, n), float)
        start = (n-1, 0)
        goal = (0, n-1)
        exact = physical_side * math.sqrt(2)

        for connectivity in [4, 8]:
            # Pure geometric shortest path.
            J, p = dijkstra(
                grid, start, goal,
                lambda u, v: geometric_edge_length(u, v, dx),
                connectivity
            )
            error = J - exact
            rows.append(dict(
                n=n,
                cell_length_m=dx,
                connectivity=connectivity,
                graph_distance_m=J,
                euclidean_distance_m=exact,
                absolute_error_m=error,
                relative_error=error/exact,
            ))

            print(
                f"  n={n:2d} dx={dx:.3f} conn={connectivity}: "
                f"graph={J:.8f} exact={exact:.8f} "
                f"rel_error={error/exact:.6f}"
            )

    df = pd.DataFrame(rows)
    path = OUT / "open_map_distance_audit.csv"
    df.to_csv(path, index=False)
    return path


def homogeneous_energy_audit(qU, qD, soil_cache):
    """
    On hard ground, energy per meter is constant for a fixed state.
    Therefore route energy should be exactly proportional to graph path length.
    """
    print("HOMOGENEOUS_ENERGY_AUDIT")
    rows = []

    n = 21
    side = 10.0
    dx = side/(n-1)
    grid = np.full((n,n), "HARD_GROUND", dtype=object)
    z = np.zeros((n,n), float)
    start=(n-1,0)
    goal=(0,n-1)

    crr=0.002
    lam=0.01

    for shape in base.SHAPES:
        for conn in [4,8]:
            res = solve(
                grid,z,start,goal,shape,crr,lam,
                qU,qD,soil_cache,dx,{},conn
            )
            rows.append(dict(
                shape=shape,
                connectivity=conn,
                J_fixed_J=res["J_fixed"],
                J_adaptive_J=res["J_adaptive"],
                B_total=res["B_total"],
                fixed_route_length_m=res["fixed_route_length_m"],
                adaptive_route_length_m=res["adaptive_route_length_m"],
            ))
            print(
                f"  {shape} conn={conn}: "
                f"L={res['fixed_route_length_m']:.8f} "
                f"Jf={res['J_fixed']:.8f} Ja={res['J_adaptive']:.8f} "
                f"B={res['B_total']:.3e}"
            )

    df = pd.DataFrame(rows)
    path = OUT / "homogeneous_connectivity_energy_audit.csv"
    df.to_csv(path,index=False)
    return path


def heterogeneous_demo(qU, qD, soil_cache):
    """
    Re-run the existing forced heterogeneous 8x8 example with both
    connectivities to quantify the numerical route-geometry effect.
    """
    print("HETEROGENEOUS_CONNECTIVITY_COMPARISON")

    map_path = ROOT / "examples" / "example_forced_heterogeneous.csv"
    elev_path = ROOT / "examples" / "elevation_flat_zero.csv"

    grid = base.read_terrain(map_path)
    z = base.read_elevation(elev_path, grid.shape)
    start=(7,0)
    goal=(0,7)
    rows=[]

    for shape in base.SHAPES:
        for conn in [4,8]:
            res=solve(
                grid,z,start,goal,shape,0.002,0.01,
                qU,qD,soil_cache,1.0,{},conn
            )
            rows.append(dict(
                shape=shape,
                connectivity=conn,
                J_fixed_J=res["J_fixed"],
                J_adaptive_J=res["J_adaptive"],
                B_total=res["B_total"],
                delta_state_J=res["delta_state"],
                delta_route_J=res["delta_route"],
                fixed_route_length_m=res["fixed_route_length_m"],
                adaptive_route_length_m=res["adaptive_route_length_m"],
            ))
            print(
                f"  {shape} conn={conn}: "
                f"Jf={res['J_fixed']:.6f} Ja={res['J_adaptive']:.6f} "
                f"B={res['B_total']:.4f} "
                f"Lf={res['fixed_route_length_m']:.3f} "
                f"La={res['adaptive_route_length_m']:.3f}"
            )

    df=pd.DataFrame(rows)
    path=OUT/"heterogeneous_connectivity_comparison.csv"
    df.to_csv(path,index=False)
    return path


def main():
    print("=== Grid-connectivity / anisotropy audit v2.79 ===")
    print("4-neighbor is retained as regression mode.")
    print("8-neighbor uses cardinal length dx and diagonal length sqrt(2)*dx.")
    print("Diagonal slope uses dz / actual diagonal edge length.")
    print("No diagonal corner-cutting through two blocked orthogonal cells.")

    qU,qD=base.features()
    soil_cache=base.load_soil_cache()

    p1=open_map_distance_audit()
    p2=homogeneous_energy_audit(qU,qD,soil_cache)
    p3=heterogeneous_demo(qU,qD,soil_cache)

    print("INTERPRETATION")
    print(
        "  In an unobstructed square, 4-neighbor routing overestimates diagonal "
        "distance because it uses Manhattan geometry."
    )
    print(
        "  8-neighbor routing exactly represents 45-degree diagonal distance "
        "for this audit and reduces grid-orientation bias."
    )
    print(
        "  8-neighbor still does not provide full rotational invariance: only "
        "eight travel directions are represented. General mission maps may "
        "require finer grids or a different graph/mesh if angular discretization "
        "becomes important."
    )
    print(
        "  Obstacle edges remain explicit directed measured constraints; no "
        "diagonal obstacle behavior is inferred."
    )
    print("OUTPUTS")
    print(" ",p1)
    print(" ",p2)
    print(" ",p3)


if __name__=="__main__":
    main()
