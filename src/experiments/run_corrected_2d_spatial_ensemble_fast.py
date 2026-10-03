
from __future__ import annotations

from pathlib import Path
import argparse, heapq, math, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src" / "experiments"))
import run_map_predictor_with_elevation as base

SIZE = 20
FRACTIONS = [0.25, 0.50, 0.75]
ARRANGEMENTS = ["RANDOM", "CLUSTERED"]
SEEDS = list(range(10))
DEFAULT_CRR = [0.0005, 0.001, 0.002, 0.005]
DEFAULT_LAMBDA = [0.003, 0.01, 0.02]
CONNECTIVITY = 8
CELL_LENGTH = 1.0

PAIRS = [
    ("HARD_GROUND", "JLU_MARS1_LOOSE"),
    ("JLU_MARS1_DENSE", "JLU_MARS1_LOOSE"),
]

CHECKPOINT = OUT / "corrected_2d_spatial_ensemble_checkpoint.csv"


def parse_floats(text, default):
    if text is None:
        return list(default)
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def make_random_map(a, b, frac_b, seed):
    rng = np.random.default_rng(seed)
    arr = np.full((SIZE, SIZE), a, dtype=object)
    n = SIZE * SIZE
    k = int(round(frac_b * n))
    ids = rng.choice(n, size=k, replace=False)
    flat = arr.ravel()
    flat[ids] = b
    return arr


def make_clustered_map(a, b, frac_b, seed):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(SIZE, SIZE))
    for _ in range(6):
        f = (
            f
            + np.roll(f, 1, axis=0)
            + np.roll(f, -1, axis=0)
            + np.roll(f, 1, axis=1)
            + np.roll(f, -1, axis=1)
        ) / 5.0

    n = SIZE * SIZE
    k = int(round(frac_b * n))
    order = np.argsort(f.ravel())
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        mask[order[-k:]] = True

    arr = np.full(n, a, dtype=object)
    arr[mask] = b
    return arr.reshape(SIZE, SIZE)


def ensure_endpoints(grid, a):
    grid = grid.copy()
    grid[SIZE-1, 0] = a
    grid[0, SIZE-1] = a
    return grid


DIRS8 = [
    (-1,0), (1,0), (0,-1), (0,1),
    (-1,-1), (-1,1), (1,-1), (1,1),
]


def build_adjacency(grid):
    nr, nc = grid.shape
    adj = {}
    for r in range(nr):
        for c in range(nc):
            u = (r,c)
            if grid[u] == "BLOCKED":
                continue
            lst = []
            for dr,dc in DIRS8:
                rr,cc = r+dr,c+dc
                if not (0 <= rr < nr and 0 <= cc < nc):
                    continue
                v = (rr,cc)
                if grid[v] == "BLOCKED":
                    continue
                if abs(dr)==1 and abs(dc)==1:
                    a=(r+dr,c)
                    b=(r,c+dc)
                    if grid[a]=="BLOCKED" and grid[b]=="BLOCKED":
                        continue
                mult = math.sqrt(2.0) if abs(dr)==1 and abs(dc)==1 else 1.0
                lst.append((v, mult))
            adj[u] = lst
    return adj


def dijkstra_lookup(grid, adj, start, goal, pair_cost):
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]

    while pq:
        d,u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue
        if u == goal:
            break
        tu = str(grid[u])
        for v,mult in adj[u]:
            tv = str(grid[v])
            c = pair_cost[(tu,tv)] * mult * CELL_LENGTH
            if not math.isfinite(c):
                continue
            nd = d+c
            if nd < dist.get(v, math.inf) - 1e-15:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq,(nd,v))

    if goal not in dist:
        return math.inf, []

    p=[goal]
    while p[-1] != start:
        p.append(prev[p[-1]])
    p.reverse()
    return float(dist[goal]), p


def path_edges(path):
    return list(zip(path[:-1],path[1:]))


def path_length(path):
    total=0.0
    for u,v in path_edges(path):
        dr=abs(v[0]-u[0]); dc=abs(v[1]-u[1])
        total += CELL_LENGTH * math.hypot(dr,dc)
    return total


def make_density_table(terrains, shape, crr, lam, qU, qD, soil_cache):
    """
    Exact flat-map cost-density lookup for the same reduced model used by v2.82.
    The expensive soil interpolation is evaluated once per (terrain,state),
    rather than once per graph-edge visit.
    """
    table={}
    for t in terrains:
        for s0 in base.STATE_GRID:
            s=float(s0)
            h=base.hshape(shape,s,qU,qD)
            rr=base.W*crr + base.W*lam*h
            c=base.terrain_directional_cost(soil_cache,t,s,0.0,rr)
            table[(t,s)] = float(c)
    return table


def make_pair_tables(terrains, density):
    fixed_by_state={}
    adaptive_cost={}
    adaptive_state={}

    for s0 in base.STATE_GRID:
        s=float(s0)
        pc={}
        for a in terrains:
            for b in terrains:
                ca=density[(a,s)]
                cb=density[(b,s)]
                pc[(a,b)] = (
                    math.inf
                    if not (math.isfinite(ca) and math.isfinite(cb))
                    else 0.5*(ca+cb)
                )
        fixed_by_state[s]=pc

    for a in terrains:
        for b in terrains:
            vals=[]
            for s0 in base.STATE_GRID:
                s=float(s0)
                vals.append((fixed_by_state[s][(a,b)],s))
            finite=[x for x in vals if math.isfinite(x[0])]
            if finite:
                c,s=min(finite,key=lambda x:(x[0],x[1]))
                adaptive_cost[(a,b)] = c
                adaptive_state[(a,b)] = s
            else:
                adaptive_cost[(a,b)] = math.inf
                adaptive_state[(a,b)] = math.nan

    return fixed_by_state, adaptive_cost, adaptive_state


def solve_fast(grid, adj, start, goal, fixed_by_state, adaptive_cost, adaptive_state):
    best=(math.inf,None,[])
    for s0 in base.STATE_GRID:
        s=float(s0)
        J,p=dijkstra_lookup(grid,adj,start,goal,fixed_by_state[s])
        if J < best[0]-1e-12:
            best=(J,s,p)

    Jf,sf,pf=best
    if not math.isfinite(Jf):
        return None

    # Adaptive states on the best fixed route.
    Jaf=0.0
    for u,v in path_edges(pf):
        mult=math.hypot(abs(v[0]-u[0]),abs(v[1]-u[1]))
        Jaf += adaptive_cost[(str(grid[u]),str(grid[v]))] * mult * CELL_LENGTH

    # Globally adaptive route.
    Ja,pa=dijkstra_lookup(grid,adj,start,goal,adaptive_cost)
    if not math.isfinite(Ja):
        return None

    ds=Jf-Jaf
    dr=Jaf-Ja
    dt=Jf-Ja
    tol=1e-8
    if ds < -tol or dr < -tol or Ja > Jf+tol or abs(ds+dr-dt) > tol:
        raise RuntimeError(
            f"routing/decomposition invariant failure: "
            f"ds={ds}, dr={dr}, dt={dt}"
        )

    return dict(
        J_fixed=Jf,
        s_fixed=sf,
        J_adaptive_on_fixed_route=Jaf,
        J_adaptive=Ja,
        delta_state=max(0.0,ds),
        delta_route=max(0.0,dr),
        delta_total=max(0.0,dt),
        B_total=(dt/Jf if Jf>0 else math.nan),
        route_share=(dr/dt if dt>tol else math.nan),
        route_changed=(pf!=pa),
        fixed_route_length_m=path_length(pf),
        adaptive_route_length_m=path_length(pa),
    )


def key_tuple(a,b,frac,arr,seed,crr,lam,shape):
    return (
        str(a),str(b),round(float(frac),8),str(arr),int(seed),
        round(float(crr),12),round(float(lam),12),str(shape)
    )


def load_checkpoint():
    if not CHECKPOINT.exists():
        return pd.DataFrame(), set()

    df=pd.read_csv(CHECKPOINT)
    done=set()
    for _,r in df.iterrows():
        done.add(key_tuple(
            r.terrain_A,r.terrain_B,r.target_fraction_B,
            r.arrangement,r.seed,r.Crr_ref,r["lambda"],r["shape"]
        ))
    return df,done


def append_checkpoint(rows):
    if not rows:
        return
    df=pd.DataFrame(rows).rename(columns={"lambda_":"lambda"})
    header=not CHECKPOINT.exists()
    df.to_csv(CHECKPOINT,mode="a",header=header,index=False)


def finalize():
    df=pd.read_csv(CHECKPOINT)
    detail_path=OUT/"corrected_2d_spatial_ensemble.csv"
    df.to_csv(detail_path,index=False)

    f=df[df.feasible==True].copy()
    if f.empty:
        raise RuntimeError("No feasible ensemble cases")

    aggregate=(
        f.groupby(
            ["terrain_A","terrain_B","target_fraction_B","arrangement"]
        )
        .agg(
            cases=("B_total","size"),
            B_min=("B_total","min"),
            B_q25=("B_total",lambda x:x.quantile(.25)),
            B_median=("B_total","median"),
            B_q75=("B_total",lambda x:x.quantile(.75)),
            B_max=("B_total","max"),
            positive_B_fraction=("B_total",lambda x:float((x>1e-12).mean())),
            route_value_case_fraction=("route_change_has_energy_value","mean"),
            route_share_median=("route_share","median"),
            route_share_max=("route_share","max"),
            delta_state_median_J=("delta_state_J","median"),
            delta_route_median_J=("delta_route_J","median"),
        )
        .reset_index()
    )
    aggregate_path=OUT/"corrected_2d_spatial_ensemble_summary.csv"
    aggregate.to_csv(aggregate_path,index=False)

    arrangement_rows=[]
    for (a,b,frac),g in f.groupby(
        ["terrain_A","terrain_B","target_fraction_B"]
    ):
        vals={}
        for arrangement in ARRANGEMENTS:
            ga=g[g.arrangement==arrangement]
            vals[arrangement]=dict(
                B_median=float(ga.B_total.median()),
                B_min=float(ga.B_total.min()),
                B_max=float(ga.B_total.max()),
                route_share_median=(
                    float(ga.route_share.dropna().median())
                    if ga.route_share.notna().any() else math.nan
                ),
            )

        arrangement_rows.append(dict(
            terrain_A=a,
            terrain_B=b,
            target_fraction_B=frac,
            random_B_median=vals["RANDOM"]["B_median"],
            clustered_B_median=vals["CLUSTERED"]["B_median"],
            clustered_minus_random_B_median=(
                vals["CLUSTERED"]["B_median"]-vals["RANDOM"]["B_median"]
            ),
            random_B_min=vals["RANDOM"]["B_min"],
            random_B_max=vals["RANDOM"]["B_max"],
            clustered_B_min=vals["CLUSTERED"]["B_min"],
            clustered_B_max=vals["CLUSTERED"]["B_max"],
            random_route_share_median=vals["RANDOM"]["route_share_median"],
            clustered_route_share_median=vals["CLUSTERED"]["route_share_median"],
        ))

    arrangement_df=pd.DataFrame(arrangement_rows)
    arrangement_path=OUT/"corrected_2d_arrangement_comparison.csv"
    arrangement_df.to_csv(arrangement_path,index=False)

    print("SUMMARY")
    for _,r in aggregate.iterrows():
        print(
            f"  {r.terrain_A}/{r.terrain_B} "
            f"fracB={r.target_fraction_B:.2f} {r.arrangement}: "
            f"B=[{r.B_min:.3f},{r.B_max:.3f}] "
            f"median={r.B_median:.3f} "
            f"positive={r.positive_B_fraction:.3f} "
            f"route_case_frac={r.route_value_case_fraction:.3f} "
            f"route_share_med={r.route_share_median:.3f}"
        )

    print("ARRANGEMENT EFFECT")
    for _,r in arrangement_df.iterrows():
        print(
            f"  {r.terrain_A}/{r.terrain_B} "
            f"fracB={r.target_fraction_B:.2f}: "
            f"median_B random={r.random_B_median:.3f}, "
            f"clustered={r.clustered_B_median:.3f}, "
            f"delta={r.clustered_minus_random_B_median:+.3f}"
        )

    print("OUTPUTS")
    print(" ",detail_path)
    print(" ",aggregate_path)
    print(" ",arrangement_path)
    print(" ",CHECKPOINT)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--crr-values",default=None)
    ap.add_argument("--lambda-values",default=None)
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Delete an existing checkpoint and restart from zero."
    )
    args=ap.parse_args()

    crr_values=parse_floats(args.crr_values,DEFAULT_CRR)
    lambda_values=parse_floats(args.lambda_values,DEFAULT_LAMBDA)

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()

    old,done=load_checkpoint()

    qU,qD=base.features()
    soil_cache=base.load_soil_cache()

    print("=== Corrected 2-D spatial ensemble v2.82.1 FAST/CHECKPOINT ===")
    print(f"grid={SIZE}x{SIZE}, connectivity={CONNECTIVITY}")
    print(f"Crr_ref={crr_values}")
    print(f"lambda={lambda_values}")
    print(f"shapes={base.SHAPES}")
    print(f"checkpoint={CHECKPOINT}")
    print(f"completed sensitivity rows already found={len(done)}")
    print("Flat-map terrain/state costs are cached exactly once per parameter case.")
    print("Results are checkpointed after each map realization.")
    print("Safe to stop with Ctrl+C and rerun; completed rows will be skipped.")

    # Precompute flat physics tables once for every terrain pair and parameter case.
    tables={}
    print("PRECOMPUTING FLAT COST TABLES")
    for a,b in PAIRS:
        terrains=tuple(sorted(set([a,b])))
        for crr in crr_values:
            for lam in lambda_values:
                for shape in base.SHAPES:
                    density=make_density_table(
                        terrains,shape,crr,lam,qU,qD,soil_cache
                    )
                    tables[(a,b,crr,lam,shape)] = make_pair_tables(
                        terrains,density
                    )
    print("  done")

    total_maps=len(PAIRS)*len(FRACTIONS)*len(ARRANGEMENTS)*len(SEEDS)
    completed_maps=0
    t_all=time.time()

    for a,b in PAIRS:
        for frac_b in FRACTIONS:
            for arrangement in ARRANGEMENTS:
                for seed in SEEDS:
                    if arrangement=="RANDOM":
                        grid=make_random_map(a,b,frac_b,seed)
                    else:
                        grid=make_clustered_map(a,b,frac_b,seed)
                    grid=ensure_endpoints(grid,a)
                    actual_frac_b=float((grid==b).mean())

                    needed=[]
                    for crr in crr_values:
                        for lam in lambda_values:
                            for shape in base.SHAPES:
                                k=key_tuple(a,b,frac_b,arrangement,seed,crr,lam,shape)
                                if k not in done:
                                    needed.append((crr,lam,shape,k))

                    if not needed:
                        completed_maps += 1
                        print(
                            f"  SKIP already complete "
                            f"{a}/{b} fracB={frac_b:.2f} {arrangement} seed={seed}"
                        )
                        continue

                    t0=time.time()
                    adj=build_adjacency(grid)
                    start=(SIZE-1,0); goal=(0,SIZE-1)
                    rows=[]

                    for crr,lam,shape,k in needed:
                        fixed_by_state,adaptive_cost,adaptive_state = tables[
                            (a,b,crr,lam,shape)
                        ]
                        res=solve_fast(
                            grid,adj,start,goal,
                            fixed_by_state,adaptive_cost,adaptive_state
                        )

                        if res is None:
                            row=dict(
                                terrain_A=a,terrain_B=b,
                                target_fraction_B=frac_b,
                                actual_fraction_B=actual_frac_b,
                                arrangement=arrangement,seed=seed,
                                Crr_ref=crr,lambda_=lam,shape=shape,
                                feasible=False,
                            )
                        else:
                            dt=float(res["delta_total"])
                            dr=float(res["delta_route"])
                            row=dict(
                                terrain_A=a,terrain_B=b,
                                target_fraction_B=frac_b,
                                actual_fraction_B=actual_frac_b,
                                arrangement=arrangement,seed=seed,
                                Crr_ref=crr,lambda_=lam,shape=shape,
                                feasible=True,
                                J_fixed_J=float(res["J_fixed"]),
                                J_adaptive_on_fixed_route_J=float(
                                    res["J_adaptive_on_fixed_route"]
                                ),
                                J_adaptive_J=float(res["J_adaptive"]),
                                delta_state_J=float(res["delta_state"]),
                                delta_route_J=dr,
                                delta_total_J=dt,
                                B_total=float(res["B_total"]),
                                route_share=(dr/dt if dt>1e-12 else math.nan),
                                s_fixed=float(res["s_fixed"]),
                                fixed_route_length_m=float(
                                    res["fixed_route_length_m"]
                                ),
                                adaptive_route_length_m=float(
                                    res["adaptive_route_length_m"]
                                ),
                                route_change_has_energy_value=bool(dr>1e-10),
                            )
                        rows.append(row)
                        done.add(k)

                    append_checkpoint(rows)
                    completed_maps += 1
                    elapsed=time.time()-t0
                    print(
                        f"  completed {a}/{b} fracB={frac_b:.2f} "
                        f"{arrangement} seed={seed} "
                        f"({len(rows)} cases, {elapsed:.1f}s) "
                        f"maps={completed_maps}/{total_maps}"
                    )

    print(f"ALL MAPS COMPLETE in {(time.time()-t_all)/60:.1f} min")
    finalize()


if __name__=="__main__":
    main()
