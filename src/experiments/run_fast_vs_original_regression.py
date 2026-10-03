from __future__ import annotations
from pathlib import Path
import math, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'src' / 'experiments'))

import run_connectivity_anisotropy_audit as original_model
import run_map_predictor_with_elevation as base
import run_corrected_2d_spatial_ensemble_fast as fastmod

SIZE=20
CELL_LENGTH=1.0
CONNECTIVITY=8

CASES=[
 dict(terrain_A='HARD_GROUND',terrain_B='JLU_MARS1_LOOSE',frac=0.25,arrangement='RANDOM',seed=0,crr=0.0005,lam=0.003,shape='LINEAR_STATE'),
 dict(terrain_A='HARD_GROUND',terrain_B='JLU_MARS1_LOOSE',frac=0.50,arrangement='CLUSTERED',seed=4,crr=0.002,lam=0.01,shape='LOADING_WORK'),
 dict(terrain_A='HARD_GROUND',terrain_B='JLU_MARS1_LOOSE',frac=0.75,arrangement='RANDOM',seed=7,crr=0.005,lam=0.02,shape='DEFLECTION'),
 dict(terrain_A='JLU_MARS1_DENSE',terrain_B='JLU_MARS1_LOOSE',frac=0.25,arrangement='CLUSTERED',seed=2,crr=0.001,lam=0.003,shape='SQRT_LOADING_WORK'),
 dict(terrain_A='JLU_MARS1_DENSE',terrain_B='JLU_MARS1_LOOSE',frac=0.50,arrangement='RANDOM',seed=6,crr=0.002,lam=0.02,shape='LOADING_WORK'),
 dict(terrain_A='JLU_MARS1_DENSE',terrain_B='JLU_MARS1_LOOSE',frac=0.75,arrangement='CLUSTERED',seed=9,crr=0.005,lam=0.01,shape='DEFLECTION'),
]

FIELDS=['J_fixed','J_adaptive_on_fixed_route','J_adaptive','delta_state','delta_route','delta_total','B_total','fixed_route_length_m','adaptive_route_length_m']

def make_grid(c):
    if c['arrangement']=='RANDOM':
        g=fastmod.make_random_map(c['terrain_A'],c['terrain_B'],c['frac'],c['seed'])
    else:
        g=fastmod.make_clustered_map(c['terrain_A'],c['terrain_B'],c['frac'],c['seed'])
    return fastmod.ensure_endpoints(g,c['terrain_A'])

def run_original(grid,c,qU,qD,soil_cache):
    z=np.zeros(grid.shape,float)
    return original_model.solve(grid,z,(SIZE-1,0),(0,SIZE-1),c['shape'],c['crr'],c['lam'],qU,qD,soil_cache,CELL_LENGTH,{},CONNECTIVITY)

def run_fast(grid,c,qU,qD,soil_cache):
    terrains=tuple(sorted(set([c['terrain_A'],c['terrain_B']])))
    density=fastmod.make_density_table(terrains,c['shape'],c['crr'],c['lam'],qU,qD,soil_cache)
    fixed_by_state,adaptive_cost,_=fastmod.make_pair_tables(terrains,density)
    adj=fastmod.build_adjacency(grid)
    return fastmod.solve_fast(grid,adj,(SIZE-1,0),(0,SIZE-1),fixed_by_state,adaptive_cost,{})

def main():
    print('=== Fast-vs-original solver regression v2.82.2 ===')
    print(f'representative_cases={len(CASES)}')
    print('connectivity=8, flat elevation, no obstacles')
    print('Purpose: verify caching changed speed, not model outputs.')
    qU,qD=base.features(); soil_cache=base.load_soil_cache()
    rows=[]; gabs=0.0; grel=0.0; flags_match=True
    t_all=time.time()
    for i,c in enumerate(CASES,1):
        grid=make_grid(c)
        t0=time.time(); orig=run_original(grid,c,qU,qD,soil_cache); t_orig=time.time()-t0
        t0=time.time(); fast=run_fast(grid,c,qU,qD,soil_cache); t_fast=time.time()-t0
        if orig is None or fast is None: raise RuntimeError(f'case {i} unexpectedly infeasible')
        cabs=0.0; crel=0.0
        for field in FIELDS:
            a=float(orig[field]); b=float(fast[field]); ae=abs(a-b); re=ae/max(abs(a),1e-15)
            rows.append(dict(case=i,terrain_A=c['terrain_A'],terrain_B=c['terrain_B'],fraction_B=c['frac'],arrangement=c['arrangement'],seed=c['seed'],Crr_ref=c['crr'],lambda_=c['lam'],shape=c['shape'],field=field,original_value=a,fast_value=b,absolute_error=ae,relative_error=re,original_seconds=t_orig,fast_seconds=t_fast))
            cabs=max(cabs,ae); crel=max(crel,re)
        smatch=abs(float(orig['s_fixed'])-float(fast['s_fixed']))<=1e-12
        rmatch=bool(orig['route_changed'])==bool(fast['route_changed'])
        flags_match &= smatch and rmatch
        gabs=max(gabs,cabs); grel=max(grel,crel)
        print(f"CASE {i}: {c['terrain_A']}/{c['terrain_B']} fracB={c['frac']:.2f} {c['arrangement']} seed={c['seed']} {c['shape']}")
        print(f'  max_abs_error={cabs:.3e} max_rel_error={crel:.3e}')
        print(f"  s_fixed original={orig['s_fixed']:.2f} fast={fast['s_fixed']:.2f} match={smatch}")
        print(f"  route_changed original={orig['route_changed']} fast={fast['route_changed']} match={rmatch}")
        print(f'  time original={t_orig:.2f}s fast={t_fast:.2f}s speedup={t_orig/max(t_fast,1e-12):.1f}x')
    df=pd.DataFrame(rows).rename(columns={'lambda_':'lambda'})
    detail=OUT/'fast_vs_original_regression.csv'; df.to_csv(detail,index=False)
    tol=1e-10; passed=(gabs<=tol and flags_match)
    summ=pd.DataFrame([dict(representative_cases=len(CASES),compared_numeric_fields=len(FIELDS),global_max_absolute_error=gabs,global_max_relative_error=grel,fixed_state_and_route_change_flags_match=flags_match,implementation_regression_tolerance=tol,numeric_regression_pass=passed)])
    summary=OUT/'fast_vs_original_regression_summary.csv'; summ.to_csv(summary,index=False)
    print('SANITY_CHECKS')
    print(f'  cases={len(CASES)}')
    print(f'  global_max_absolute_error={gabs:.3e}')
    print(f'  global_max_relative_error={grel:.3e}')
    print(f'  fixed_state_and_route_change_flags_match={flags_match}')
    print(f'  numeric_regression_pass={passed}')
    print(f'  total_runtime={(time.time()-t_all)/60:.1f} min')
    print('INTERPRETATION')
    print('  PASS means the cached solver reproduces the original solver for the audited outputs to the stated implementation-regression tolerance.')
    print('  This validates the optimization/caching implementation only; it does not add physical calibration.')
    print('OUTPUTS')
    print(' ',detail)
    print(' ',summary)

if __name__=='__main__': main()
