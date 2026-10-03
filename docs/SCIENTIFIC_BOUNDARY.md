# Scientific boundary for v2.71

v2.71 is a **predictive framework under explicit parameter assumptions**, not a calibrated predictor of a specific Lee wheel in an arbitrary mission.

Measured/source-supported inputs:
- Lee force-deformation data and state map;
- published/project soil parameter sets.

Derived:
- continuous compliance interpolation;
- reduced terramechanics soil cost;
- graph-routing solution.

Assumed/sensitivity:
- mapping from compliance state to rolling loss;
- `Crr_ref`;
- `lambda`;
- zero state-transition energy.

Not included yet:
- signed slope/elevation;
- obstacle feasibility;
- calibrated slip/shear energy;
- measured cyclic rolling loss.

Therefore report cross-model envelopes and conditional statements, not a single universal benefit percentage.
