# Obstacle model boundary

Source-supported:
- Lee small-wheel obstacle climbing experiment;
- exact tested obstacle heights 90–125 mm;
- minimum observed feasible hub gap at each tested height.

Derived:
- map hub gap to recovered compliance state `s`;
- treat states at or above the observed minimum as passable within the tested family.

Assumption:
- monotone passability above the observed minimum state. No tested case showed a higher-gap state failing where a lower-gap state succeeded.

Not claimed:
- obstacle traversal energy;
- continuous interpolation between untested obstacle heights;
- reverse-direction descent behavior;
- obstacle dynamics at other wheel geometries, loads, or speeds.

Therefore obstacle data are used only to prune infeasible state/edge combinations.
