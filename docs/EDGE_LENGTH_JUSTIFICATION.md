# Edge length / map resolution justification

`cell_length` is not a wheel parameter.

It is the physical distance represented by one horizontal or vertical graph edge.

It enters the model in two places:

1. Edge energy
   `C_edge = cell_length * average(cost density at the two endpoint terrains)`

2. Signed slope
   `theta = atan2(delta_z, cell_length)`

Therefore changing the number of grid cells while keeping the same physical map requires changing both the node elevations and `cell_length` consistently.

The v2.77 resolution audit tests whether the same physical terrain/elevation field gives convergent results when represented at 1.0, 0.5, and 0.25 m spacing.

For mission use, the chosen edge length must be tied to the map's actual spatial resolution; it is not calibrated to make adaptive benefit larger or smaller.
