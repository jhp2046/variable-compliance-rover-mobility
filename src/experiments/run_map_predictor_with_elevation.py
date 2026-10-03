
from __future__ import annotations
from pathlib import Path
import argparse, ast, heapq, math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)

QFILE=ROOT/"data"/"derived"/"continuous_q_deformation_energy_curve.csv"
SOIL=ROOT/"data"/"derived"/"continuous_soil_cost_surface.csv"

W=5.3*9.81
STATE_GRID=np.round(np.linspace(0,1,101),2)
SHAPES=["LINEAR_STATE","SQRT_LOADING_WORK","LOADING_WORK","DEFLECTION"]

SUPPORTED_TERRAINS=[
    "HARD_GROUND",
    "DRY_SAND",
    "LUNAR_REGOLITH",
    "JLU_MARS1_LOOSE",
    "JLU_MARS1_DENSE",
    "MMS",
]


# Lee Fig. 6A / source-data observed climbing feasibility proxy.
# Exact tested obstacle heights only; no height interpolation.
# min hub gap [mm] with positive measured climbing stability:
# 90->4, 95->6, 100->6, 105->6, 110->8, 115/120/125->12.
LEE_STATE_BY_GAP_MM={
    0:0.0,
    2:0.149537,
    4:0.422427,
    6:0.572765,
    8:0.746899,
    10:0.916081,
    12:1.0,
}
OBSTACLE_MIN_GAP_MM={
    90:4,
    95:6,
    100:6,
    105:6,
    110:8,
    115:12,
    120:12,
    125:12,
}
OBSTACLE_MIN_STATE={
    h:LEE_STATE_BY_GAP_MM[g] for h,g in OBSTACLE_MIN_GAP_MM.items()
}

def features():
    qdf=pd.read_csv(QFILE).sort_values("state_s")
    qU={}; qD={}
    D0=float(qdf.iloc[(qdf.state_s.astype(float)-0).abs().argmin()].deformation_increment_mm)
    for s0 in STATE_GRID:
        s=float(s0)
        r=qdf.iloc[(qdf.state_s.astype(float)-s).abs().argmin()]
        qU[s]=float(r.q)
        qD[s]=float(r.deformation_increment_mm)/D0
    return qU,qD

def feat(shape,s,qU,qD):
    if shape=="LINEAR_STATE": return s
    if shape=="SQRT_LOADING_WORK": return math.sqrt(qU[s])
    if shape=="LOADING_WORK": return qU[s]
    if shape=="DEFLECTION": return qD[s]
    raise ValueError(shape)

def hshape(shape,s,qU,qD):
    f0=feat(shape,0.,qU,qD); f1=feat(shape,1.,qU,qD); fs=feat(shape,s,qU,qD)
    return (fs-f0)/(f1-f0)

def read_terrain(path):
    df=pd.read_csv(path,header=None,dtype=str)
    arr=df.to_numpy(object)
    for x in np.unique(arr):
        if x not in SUPPORTED_TERRAINS and x!="BLOCKED":
            raise ValueError(f"Unsupported cell '{x}'")
    return arr

def read_elevation(path,shape):
    if path is None:
        return np.zeros(shape,float)
    z=pd.read_csv(path,header=None).to_numpy(float)
    if z.shape!=shape:
        raise ValueError(f"Elevation grid shape {z.shape} != terrain grid shape {shape}")
    if not np.isfinite(z).all():
        raise ValueError("Elevation grid contains non-finite values")
    return z


def load_soil_cache():
    """
    Pre-cache every (terrain,state) slope curve once.

    This removes repeated pandas filtering from the innermost routing loop.
    """
    df=pd.read_csv(SOIL)
    required={"terrain","theta_deg","state_s","mobility_cost_J_per_m","feasible"}
    missing=required-set(df.columns)
    if missing:
        raise RuntimeError(f"Soil surface missing columns: {sorted(missing)}")

    cache={}
    for t in SUPPORTED_TERRAINS:
        if t=="HARD_GROUND":
            continue
        gt=df[df.terrain==t]
        for s0 in STATE_GRID:
            s=float(s0)
            g=gt[np.isclose(gt.state_s.astype(float),s)].sort_values("theta_deg")
            if g.empty:
                raise RuntimeError(f"No soil rows for {t}, s={s}")
            cache[(t,s)]={
                "theta":g.theta_deg.astype(float).to_numpy(),
                "mobility":g.mobility_cost_J_per_m.astype(float).to_numpy(),
                "feasible":g.feasible.astype(bool).to_numpy(),
            }
    return cache

def soil_at_positive_slope(cache,t,s,theta_abs_deg):
    g=cache[(t,s)]
    th=g["theta"]; mob=g["mobility"]; feas=g["feasible"]

    if theta_abs_deg < th.min()-1e-12 or theta_abs_deg > th.max()+1e-12:
        return math.inf,False

    idx=np.where(np.isclose(th,theta_abs_deg,atol=1e-10))[0]
    if len(idx):
        i=int(idx[0])
        if not bool(feas[i]):
            return math.inf,False
        comp=float(mob[i])-W*math.sin(math.radians(theta_abs_deg))
        return comp,True

    hi=int(np.searchsorted(th,theta_abs_deg))
    lo=hi-1
    if not (bool(feas[lo]) and bool(feas[hi])):
        return math.inf,False

    t0=float(th[lo]); t1=float(th[hi])
    m0=float(mob[lo])-W*math.sin(math.radians(t0))
    m1=float(mob[hi])-W*math.sin(math.radians(t1))
    a=(theta_abs_deg-t0)/(t1-t0)
    return (1-a)*m0+a*m1,True

def flat_soil_component(cache,t,s):
    if t=="HARD_GROUND":
        return 0.0,True
    g=cache[(t,s)]
    th=g["theta"]; mob=g["mobility"]; feas=g["feasible"]
    idx=np.where(np.isclose(th,0.0,atol=1e-10))[0]
    if not len(idx):
        raise RuntimeError(f"No flat soil row for {t}, s={s}")
    i=int(idx[0])
    if not bool(feas[i]):
        return math.inf,False
    return float(mob[i]),True

def terrain_directional_raw(cache,t,s,theta_deg,rr):
    """
    Return raw (unclipped) propulsion cost density and feasibility.
    """
    if t=="HARD_GROUND":
        return rr + W*math.sin(math.radians(theta_deg)), True

    if theta_deg>=0:
        comp,ok=soil_at_positive_slope(cache,t,s,theta_deg)
    else:
        comp,ok=flat_soil_component(cache,t,s)

    if not ok or not math.isfinite(comp):
        return math.inf,False

    return comp + rr + W*math.sin(math.radians(theta_deg)), True

def terrain_directional_cost(cache,t,s,theta_deg,rr):
    raw,ok=terrain_directional_raw(cache,t,s,theta_deg,rr)
    if not ok:
        return math.inf
    return max(0.0,raw)


def read_obstacles(path,grid_shape):
    """
    Directed obstacle-edge CSV with header:
      from_row,from_col,to_row,to_col,height_mm

    Only exact Lee-tested obstacle heights are accepted.
    The file describes climbing direction only; reverse descent is not inferred.
    """
    if path is None:
        return {}

    df=pd.read_csv(path)
    required={"from_row","from_col","to_row","to_col","height_mm"}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"Obstacle CSV missing columns: {sorted(missing)}")

    nr,nc=grid_shape
    out={}
    for _,r in df.iterrows():
        u=(int(r.from_row),int(r.from_col))
        v=(int(r.to_row),int(r.to_col))
        h=int(r.height_mm)

        for q in [u,v]:
            if not (0<=q[0]<nr and 0<=q[1]<nc):
                raise ValueError(f"Obstacle edge endpoint outside map: {q}")

        if abs(u[0]-v[0])+abs(u[1]-v[1])!=1:
            raise ValueError(f"Obstacle edge must connect 4-neighbor cells: {u}->{v}")

        if h not in OBSTACLE_MIN_STATE:
            raise ValueError(
                f"Obstacle height {h} mm is not one of the exact tested heights "
                f"{sorted(OBSTACLE_MIN_STATE)}. No interpolation is used."
            )

        out[(u,v)]=h
    return out

def obstacle_state_feasible(obstacles,u,v,s):
    """
    Source-data feasibility proxy.

    For a directed climbing edge with tested height h, state must satisfy
    s >= the minimum Lee-derived state corresponding to the minimum tested
    feasible hub gap.

    This assumes monotone passability above the observed minimum within the
    tested wheel family. It is a feasibility proxy only, not an energy model.
    """
    h=obstacles.get((u,v))
    if h is None:
        return True
    return s+1e-12 >= OBSTACLE_MIN_STATE[h]

def neighbors(grid,u):
    nr,nc=grid.shape
    r,c=u
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        v=(r+dr,c+dc)
        if 0<=v[0]<nr and 0<=v[1]<nc and grid[v]!="BLOCKED":
            yield v

def slope_deg(z,u,v,cell_length):
    dz=float(z[v]-z[u])
    return math.degrees(math.atan2(dz,cell_length))

def make_edge_lib(shape,crr,lam,qU,qD,soil_cache):
    h={float(s):hshape(shape,float(s),qU,qD) for s in STATE_GRID}
    return h

def edge_cost(grid,z,u,v,s,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles):
    if grid[u]=="BLOCKED" or grid[v]=="BLOCKED":
        return math.inf
    if not obstacle_state_feasible(obstacles,u,v,s):
        return math.inf

    theta=slope_deg(z,u,v,cell_length)
    h=hshape(shape,s,qU,qD)
    rr=W*crr+W*lam*h

    # Edge-consistent average of endpoint terrain costs evaluated at the SAME signed edge slope.
    a=terrain_directional_cost(soil_cache,grid[u],s,theta,rr)
    b=terrain_directional_cost(soil_cache,grid[v],s,theta,rr)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return cell_length*0.5*(a+b)

def dijkstra(grid,start,goal,costfn):
    dist={start:0.0}; prev={}; pq=[(0.0,start)]
    while pq:
        d,u=heapq.heappop(pq)
        if d!=dist.get(u,math.inf): continue
        if u==goal: break
        for v in neighbors(grid,u):
            c=costfn(u,v)
            if not math.isfinite(c): continue
            nd=d+c
            if nd<dist.get(v,math.inf)-1e-15:
                dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    if goal not in dist:
        return math.inf,[]
    p=[goal]
    while p[-1]!=start:
        p.append(prev[p[-1]])
    p.reverse()
    return float(dist[goal]),p

def pedges(p):
    return list(zip(p[:-1],p[1:]))

def adaptive_path_cost(grid,z,path,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles):
    total=0.; states=[]
    for u,v in pedges(path):
        vals=[(edge_cost(grid,z,u,v,float(s),shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles),float(s))
              for s in STATE_GRID]
        f=[x for x in vals if math.isfinite(x[0])]
        if not f: return math.inf,[]
        c,s=min(f,key=lambda x:(x[0],x[1]))
        total+=c; states.append(s)
    return total,states

def solve(grid,z,start,goal,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles):
    best=(math.inf,None,[])
    for s0 in STATE_GRID:
        s=float(s0)
        J,p=dijkstra(grid,start,goal,
            lambda u,v,ss=s:edge_cost(grid,z,u,v,ss,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles))
        if J<best[0]-1e-12:
            best=(J,s,p)
    Jf,sf,pf=best
    if not math.isfinite(Jf): return None

    Jaf,seqf=adaptive_path_cost(grid,z,pf,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles)

    eb={}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            u=(r,c)
            if grid[u]=="BLOCKED": continue
            for v in neighbors(grid,u):
                vals=[(edge_cost(grid,z,u,v,float(s),shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles),float(s))
                      for s in STATE_GRID]
                f=[x for x in vals if math.isfinite(x[0])]
                eb[(u,v)]=min(f,key=lambda x:(x[0],x[1])) if f else (math.inf,math.nan)

    Ja,pa=dijkstra(grid,start,goal,lambda u,v:eb[(u,v)][0])
    if not math.isfinite(Ja): return None
    seqa=[eb[(u,v)][1] for u,v in pedges(pa)]

    ds=Jf-Jaf; dr=Jaf-Ja; dt=Jf-Ja
    tol=1e-8
    if ds<-tol or dr<-tol or Ja>Jf+tol or abs(ds+dr-dt)>tol:
        raise RuntimeError("routing/decomposition invariant failure")

    return dict(
        J_fixed=Jf,s_fixed=sf,
        J_adaptive_on_fixed_route=Jaf,J_adaptive=Ja,
        delta_state=max(0.,ds),delta_route=max(0.,dr),delta_total=max(0.,dt),
        B_total=(dt/Jf if Jf>0 else math.nan),
        route_share=(dr/dt if dt>tol else math.nan),
        route_changed=(pf!=pa),
        fixed_route=repr(pf),adaptive_route=repr(pa),
        adaptive_states_on_fixed_route=repr(seqf),adaptive_states=repr(seqa)
    )


def route_obstacle_diag(path,states,obstacles):
    used=[]
    for i,(u,v) in enumerate(pedges(path)):
        if (u,v) in obstacles:
            h=obstacles[(u,v)]
            s=states[i] if i < len(states) else math.nan
            used.append({
                "edge":(u,v),
                "height_mm":h,
                "required_min_s":OBSTACLE_MIN_STATE[h],
                "used_s":s,
            })
    return used

def route_diag(grid,z,path,cell_length):
    cells={}
    slopes=[]
    for u in path:
        t=grid[u]; cells[t]=cells.get(t,0)+1
    for u,v in pedges(path):
        slopes.append(slope_deg(z,u,v,cell_length))
    return cells,slopes


def clipping_audit(grid,z,path,shape,crr,lam,qU,qD,soil_cache,cell_length):
    """
    Audit clipping on the chosen path across all 101 compliance states.

    Reports state-endpoint evaluations whose raw downhill propulsion cost <= 0.
    This is a numerical/model diagnostic, not a physical threshold.
    """
    n_eval=0
    n_clipped=0
    edge_any_clip=0
    edge_all_states_both_endpoints_clip=0

    for u,v in pedges(path):
        theta=slope_deg(z,u,v,cell_length)
        edge_clip=False
        edge_all=True
        for s0 in STATE_GRID:
            s=float(s0)
            rr=W*crr+W*lam*hshape(shape,s,qU,qD)
            both=True
            for t in [grid[u],grid[v]]:
                raw,ok=terrain_directional_raw(soil_cache,t,s,theta,rr)
                if not ok or not math.isfinite(raw):
                    both=False
                    continue
                n_eval+=1
                if raw<=0:
                    n_clipped+=1
                    edge_clip=True
                else:
                    both=False
            if not both:
                edge_all=False
        if edge_clip:
            edge_any_clip+=1
        if edge_all:
            edge_all_states_both_endpoints_clip+=1

    return {
        "endpoint_state_evaluations":n_eval,
        "clipped_endpoint_state_evaluations":n_clipped,
        "clipped_fraction":(n_clipped/n_eval if n_eval else math.nan),
        "edges_with_any_clipping":edge_any_clip,
        "edges_all_states_both_endpoints_clipped":edge_all_states_both_endpoints_clip,
        "path_edges":len(path)-1,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--map",required=True)
    ap.add_argument("--elevation",default=None,help="CSV meters, same shape as terrain map")
    ap.add_argument("--obstacles",default=None,help="Directed obstacle-edge CSV")
    ap.add_argument("--start",default=None)
    ap.add_argument("--goal",default=None)
    ap.add_argument("--cell-length",type=float,default=1.0)
    ap.add_argument("--crr",type=float,default=0.002)
    ap.add_argument("--lambda-value",type=float,default=0.01,dest="lam")
    args=ap.parse_args()

    grid=read_terrain(Path(args.map))
    z=read_elevation(Path(args.elevation) if args.elevation else None,grid.shape)
    obstacles=read_obstacles(Path(args.obstacles) if args.obstacles else None,grid.shape)
    nr,nc=grid.shape
    start=(nr-1,0) if args.start is None else tuple(map(int,args.start.split(",")))
    goal=(0,nc-1) if args.goal is None else tuple(map(int,args.goal.split(",")))

    if args.cell_length<=0: raise ValueError("cell-length must be positive")
    if grid[start]=="BLOCKED" or grid[goal]=="BLOCKED": raise ValueError("start/goal blocked")

    print("=== Terrain + slope + measured-obstacle predictor v2.73.1 ===")
    print(f"map={args.map} elevation={args.elevation or 'FLAT_ZERO'} size={nr}x{nc}")
    print(f"obstacles={args.obstacles or 'NONE'}")
    print(f"start={start} goal={goal} cell_length={args.cell_length} m")
    print(f"Crr_ref={args.crr} lambda={args.lam}")
    print("Signed slope is computed edgewise from elevation difference.")
    print("Downhill propulsion is clipped at zero; no regeneration.")
    print("Downhill deformable-soil resistance uses the flat soil component as a reduced approximation.")
    print("Hard-ground slope traction is not constrained in this version.")
    print("v2.73.1: adds forced-obstacle diagnostics and a bottleneck regression case.")

    qU,qD=features()
    soil_cache=load_soil_cache()
    rows=[]

    for shape in SHAPES:
        res=solve(grid,z,start,goal,shape,args.crr,args.lam,qU,qD,soil_cache,args.cell_length,obstacles)
        if res is None:
            print(f"{shape}: NO_FEASIBLE_ROUTE")
            continue
        rows.append(dict(shape=shape,Crr_ref=args.crr,lambda_=args.lam,**res))
        print(
            f"{shape}: J_fixed={res['J_fixed']:.6f} J "
            f"J_adaptive={res['J_adaptive']:.6f} J "
            f"B={res['B_total']:.4f} "
            f"dJ_state={res['delta_state']:.6f} "
            f"dJ_route={res['delta_route']:.6f} "
            f"route_changed={res['route_changed']}"
        )
        pf=ast.literal_eval(res["fixed_route"]); pa=ast.literal_eval(res["adaptive_route"])
        fc,fs=route_diag(grid,z,pf,args.cell_length)
        ac,ass=route_diag(grid,z,pa,args.cell_length)
        print(f"  fixed route cells: {fc}")
        print(f"  fixed edge slopes deg: {[round(x,3) for x in fs]}")
        print(f"  adaptive route cells: {ac}")
        print(f"  adaptive edge slopes deg: {[round(x,3) for x in ass]}")
        seqf=ast.literal_eval(res["adaptive_states_on_fixed_route"])
        seqa=ast.literal_eval(res["adaptive_states"])
        print(f"  obstacle edges on fixed route under adaptive states: {route_obstacle_diag(pf,seqf,obstacles)}")
        print(f"  obstacle edges on adaptive route: {route_obstacle_diag(pa,seqa,obstacles)}")
        clip=clipping_audit(
            grid,z,pa,shape,args.crr,args.lam,qU,qD,soil_cache,args.cell_length
        )
        print(
            f"  clipping audit: fraction={clip['clipped_fraction']:.3f} "
            f"edges_any={clip['edges_with_any_clipping']}/{clip['path_edges']} "
            f"edges_all_states_both_endpoints="
            f"{clip['edges_all_states_both_endpoints_clipped']}/{clip['path_edges']}"
        )

    if not rows:
        raise RuntimeError("No feasible route under any rolling-loss shape")

    df=pd.DataFrame(rows).rename(columns={"lambda_":"lambda"})
    stem=Path(args.map).stem
    detail=OUT/f"{stem}_slope_prediction_detail.csv"
    df.to_csv(detail,index=False)

    Bs=df.B_total.to_numpy(float); dS=df.delta_state.to_numpy(float)
    dR=df.delta_route.to_numpy(float); dT=df.delta_total.to_numpy(float)
    summary=pd.DataFrame([dict(
        map_name=stem,Crr_ref=args.crr,lambda_=args.lam,
        elevation_file=(args.elevation or "FLAT_ZERO"),
        models=len(df),
        B_min=float(Bs.min()),B_median=float(np.median(Bs)),B_max=float(Bs.max()),
        delta_total_min_J=float(dT.min()),delta_total_max_J=float(dT.max()),
        delta_state_min_J=float(dS.min()),delta_state_max_J=float(dS.max()),
        delta_route_min_J=float(dR.min()),delta_route_max_J=float(dR.max()),
        route_changed_fraction=float(df.route_changed.mean()),
        all_models_adaptive_nonworse=bool((df.J_adaptive<=df.J_fixed+1e-8).all())
    )]).rename(columns={"lambda_":"lambda"})
    summ=OUT/f"{stem}_slope_prediction_summary.csv"
    summary.to_csv(summ,index=False)

    print("UNCERTAINTY_ENVELOPE")
    r=summary.iloc[0]
    print(f"  B=[{r.B_min:.4f},{r.B_max:.4f}] median={r.B_median:.4f}")
    print(f"  DeltaJ_state=[{r.delta_state_min_J:.6f},{r.delta_state_max_J:.6f}] J")
    print(f"  DeltaJ_route=[{r.delta_route_min_J:.6f},{r.delta_route_max_J:.6f}] J")
    print("SCIENTIFIC_STATUS")
    print("  Signed elevation/slope is now integrated edge-consistently.")
    print("  Uphill deformable-soil feasibility comes from the v2.53 positive-slope surface.")
    print("  Downhill soil and hard-ground traction remain reduced assumptions.")
    print("  Rolling-loss law remains sensitivity-only, not calibrated.")
    print("  Obstacle data affect feasibility only; no obstacle energy is invented.")
    print("OUTPUTS")
    print(" ",detail)
    print(" ",summ)

if __name__=="__main__":
    main()
