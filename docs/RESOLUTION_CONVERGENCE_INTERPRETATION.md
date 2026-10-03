# Resolution convergence interpretation

v2.77 demonstrated that the fixed route cost is exactly resolution-invariant in the controlled corridor, while the adaptive result contains finite terrain-interface discretization error.

Why:

For an edge joining terrain A and terrain B, the planner uses

`C_edge(s) = L/2 * [c_A(s) + c_B(s)]`

with one state `s` on that edge.

A continuously adaptive idealization with zero transition cost would switch state exactly at the physical terrain boundary. A finite mixed edge cannot do that. Its error is proportional to the physical width of the mixed boundary edge.

Therefore refinement should shrink that error.

v2.78 compares the grid result against the exact piecewise-segment limit for a special 1-D audit and estimates the observed convergence rate.

For actual maps, edge length should be chosen from map resolution and then checked by refinement until the remaining numerical change is small compared with physical/model uncertainty.
