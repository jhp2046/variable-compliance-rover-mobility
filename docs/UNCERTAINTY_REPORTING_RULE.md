# Uncertainty reporting rule

The current rolling-loss uncertainty study is deterministic sensitivity analysis.

It varies:
- baseline rolling coefficient `Crr_ref`,
- state-dependent amplitude `lambda`,
- normalized rolling-loss shape.

Therefore report:
- tested parameter ranges,
- min/median/max across tested cases,
- cross-shape envelopes at fixed parameter pairs,
- whether conclusions such as `B>0` survive all tested cases.

Do not report:
- probability of benefit,
- confidence interval,
- standard statistical uncertainty,
- calibrated prediction error.

Those require measured rolling-loss data and an uncertainty distribution that the current project does not have.
