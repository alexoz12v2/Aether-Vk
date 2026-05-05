# `imex_integrator.comp` (p1, p2, p3-4, p5)

## Purpose
This collection of shaders drives the main physics numerical integration using a robust IMEX (Implicit-Explicit) simulation loop.

## Mathematical Foundation
The system employs the Implicit Midpoint Rule (IMR) for complex, sparse rigid-body systems, and the explicit Velocity Verlet integration for massively parallel particles (Chapter 7.2 & 7.3 for integration fundamentals, adapted for IMEX). The explicit integration evaluates $f_{n}$ and $v_{n}$ to drift particles to a midpoint, allows the implicit solver to compute the global rigid-body state at the midpoint, and then performs a final particle drift and velocity kick using $f_{n+1}$.

## Implementation Details
- **`p1-2_imex_particles.comp`**: The first half-step. Applies an explicit velocity kick to the particles using the forces accumulated at the end of the *previous* frame ($v_{mid} = v_n + \frac{\Delta t}{2m}F_n$), then drifts them by half a time step to $q_{mid}$.
- **`p3-4_imex_rigidbody_imr.comp`**: Executes the Newton-Raphson nonlinear solver for the Implicit Midpoint Rule. Uses the previously evaluated $q_{mid}$ of particles/emitters to evaluate interactions (e.g. gravitational constraints) implicitly over $\Delta t$.
- **`p5_imex_particles.comp`**: Completes the explicit loop. Drifts particles to their final $q_{n+1}$ state, triggers a new force accumulation $F_{n+1}$ evaluation based on the new topology, and executes the final velocity kick.
- **AOSOA Layouts**: To maximize caching and memory-controller bandwidth, particles are flattened in an Array of Structures of Arrays layout optimally matching the hardware `SUBGROUP_SIZE`.
