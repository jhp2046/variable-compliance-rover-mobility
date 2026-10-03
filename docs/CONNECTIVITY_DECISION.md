# Connectivity decision

v2.79 showed:

- 4-neighbor open-square diagonal distance error: +41.42%.
- The error does not shrink with grid refinement.
- 8-neighbor represents the 45-degree diagonal exactly.
- Homogeneous hard-ground controls remain B=0.
- Heterogeneous map B changed materially when the route geometry was corrected.

Therefore 8-neighbor is the default from v2.80 onward.

This is a numerical routing decision, not a physical wheel parameter.

Residual anisotropy remains because only eight headings are permitted. v2.80 quantifies that remaining angular error explicitly rather than assuming rotational invariance.
