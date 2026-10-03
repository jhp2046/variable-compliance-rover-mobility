from __future__ import annotations
from pathlib import Path
import sys, ast
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'/'experiments'))
import run_map_predictor_with_elevation as pred

HEIGHTS=[90,95,100,105,110,115,120,125]
EXPECTED_MIN_STATE={90:0.422427,95:0.572765,100:0.572765,105:0.572765,110:0.746899,115:1.0,120:1.0,125:1.0}
EXPECTED_GRID_STATE={90:0.43,95:0.58,100:0.58,105:0.58,110:0.75,115:1.0,120:1.0,125:1.0}
MAP=ROOT/'examples'/'example_forced_obstacle_corridor.csv'
ELEV=ROOT/'examples'/'example_forced_obstacle_elevation.csv'

def main():
    print('=== Lee obstacle-height feasibility regression v2.74 ===')
    print('Single unavoidable obstacle edge on a 1x8 hard-ground corridor.')
    print('Only exact Lee-tested heights are evaluated; no obstacle energy is added.')
    grid=pred.read_terrain(MAP); z=pred.read_elevation(ELEV,grid.shape)
    qU,qD=pred.features(); soil_cache=pred.load_soil_cache()
    start=(0,0); goal=(0,7); crr=0.002; lam=0.01; cell_length=1.0
    baseline={shape:pred.solve(grid,z,start,goal,shape,crr,lam,qU,qD,soil_cache,cell_length,{}) for shape in pred.SHAPES}
    rows=[]
    for h in HEIGHTS:
        obstacles={((0,3),(0,4)):h}
        print(f'HEIGHT {h} mm source_min={EXPECTED_MIN_STATE[h]:.6f} expected_grid={EXPECTED_GRID_STATE[h]:.2f}')
        for shape in pred.SHAPES:
            res=pred.solve(grid,z,start,goal,shape,crr,lam,qU,qD,soil_cache,cell_length,obstacles)
            if res is None: raise RuntimeError(f'No feasible route: h={h}, shape={shape}')
            pa=ast.literal_eval(res['adaptive_route']); seq=ast.literal_eval(res['adaptive_states'])
            obs=pred.route_obstacle_diag(pa,seq,obstacles)
            if len(obs)!=1: raise RuntimeError(f'Expected one obstacle edge, got {obs}')
            used=float(obs[0]['used_s']); required=float(obs[0]['required_min_s']); fixed_s=float(res['s_fixed'])
            if used+1e-12 < required: raise RuntimeError(f'Adaptive below threshold h={h}: {used} < {required}')
            if fixed_s+1e-12 < required: raise RuntimeError(f'Fixed below threshold h={h}: {fixed_s} < {required}')
            expected=EXPECTED_GRID_STATE[h]
            if abs(used-expected)>1e-12: raise RuntimeError(f'Unexpected obstacle state h={h}, shape={shape}: used={used}, expected={expected}')
            rows.append(dict(height_mm=h,shape=shape,required_min_state=required,expected_grid_state=expected,
                adaptive_obstacle_state=used,fixed_state=fixed_s,J_fixed=res['J_fixed'],J_adaptive=res['J_adaptive'],
                B_total=res['B_total'],delta_state=res['delta_state'],delta_route=res['delta_route'],
                route_changed=res['route_changed'],no_obstacle_J=baseline[shape]['J_fixed']))
            print(f"  {shape}: fixed_s={fixed_s:.2f} obstacle_s={used:.2f} Jf={res['J_fixed']:.6f} Ja={res['J_adaptive']:.6f} B={res['B_total']:.4f}")
    df=pd.DataFrame(rows); df.to_csv(OUT/'obstacle_height_regression.csv',index=False)
    summary=(df.groupby('height_mm').agg(required_min_state=('required_min_state','first'),expected_grid_state=('expected_grid_state','first'),adaptive_state_min=('adaptive_obstacle_state','min'),adaptive_state_max=('adaptive_obstacle_state','max'),fixed_state_min=('fixed_state','min'),fixed_state_max=('fixed_state','max'),B_min=('B_total','min'),B_max=('B_total','max')).reset_index())
    summary.to_csv(OUT/'obstacle_height_summary.csv',index=False)
    print('SANITY_CHECKS')
    print(f'  rows={len(df)}')
    print(f"  adaptive_threshold_all_pass={bool((df.adaptive_obstacle_state+1e-12>=df.required_min_state).all())}")
    print(f"  fixed_threshold_all_pass={bool((df.fixed_state+1e-12>=df.required_min_state).all())}")
    print(f"  exact_expected_grid_state_all_pass={bool(((df.adaptive_obstacle_state-df.expected_grid_state).abs()<1e-12).all())}")
    print(f"  max_delta_route={df.delta_route.max():.3e}")
    print('HEIGHT_SUMMARY')
    for _,r in summary.iterrows():
        print(f"  h={int(r.height_mm)} mm: source_min={r.required_min_state:.6f} grid_state={r.expected_grid_state:.2f} B_across_shapes=[{r.B_min:.4f},{r.B_max:.4f}]")
    print('INTERPRETATION')
    print('  Passing validates implementation of the measured obstacle-state feasibility mapping only.')
    print('  It does not validate the rolling-loss model or calibrate the B magnitudes.')
    print('OUTPUTS')
    print(' ',OUT/'obstacle_height_regression.csv')
    print(' ',OUT/'obstacle_height_summary.csv')

if __name__=='__main__': main()
