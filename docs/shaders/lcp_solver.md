# `lcp_solver.comp`

## Purpose
Solves for the exact impulses required to resolve multiple simultaneous collisions without causing artificial energy gains or jitter.

## Mathematical Foundation
When multiple objects collide at the same time, computing their impulsive responses sequentially can lead to infinite loops or incorrect final states (Chapter 3.5.2 & 4.11). The system must be solved simultaneously as a Linear Complementarity Problem (LCP): $A x + b \ge 0, x \ge 0, x^T(Ax+b) = 0$. 
- The matrix $A$ represents the mass and normal coupling between shared contacts.
- The vector $b$ maps the target post-collision velocity (including restitution).
- For resting contacts (where relative velocity $\approx 0$), restitution is clamped to $0.0$, effectively solving for the continuous contact forces needed to prevent interpenetration (Chapter 3.5.3).

## Implementation Details
- **Projected Gauss-Seidel (PGS)**: Rather than pivoting (Dantzig/Baraff), a highly parallelizable PGS iterative solver is used.
- **Island Grouping**: Operates on densely packed islands of collisions.
- **Matrix Assembly**: Populates the off-diagonal elements of $A$ by detecting shared particles between collision pairs $i$ and $j$, appropriately adjusting the sign and inverse mass weighting based on the dot product of their respective collision normals.
