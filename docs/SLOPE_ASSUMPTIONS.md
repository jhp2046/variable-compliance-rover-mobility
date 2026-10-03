# v2.72 slope assumptions

Measured / inherited surface:
- positive-slope deformable-soil costs and feasibility from v2.53.

Derived:
- signed edge slope from elevation;
- endpoint-averaged edge cost;
- interpolation between available positive-slope samples.

Reduced assumptions:
- remove `W sin(theta)` from the positive-slope mobility surface to estimate the non-gravity soil component;
- for downhill edges, use the flat soil component because uphill traction failure should not automatically be transferred downhill;
- add signed gravity `W sin(theta_edge)`;
- clip total propulsion cost at zero, representing no regenerative credit;
- no hard-ground slope traction constraint until a measured/declared friction coefficient is available.

These assumptions must remain explicit in reporting.
