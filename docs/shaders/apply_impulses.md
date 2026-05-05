# `apply_impulses.comp`

## Purpose
This shader maps the computed impulses from the LCP solver back to the linear velocities of the particles.

## Mathematical Foundation
According to the principle of impulse and momentum (Chapter 3.5.1 of the PDF), when a collision is resolved, an impulse $\vec{P}$ is applied positively to Particle A and negatively to Particle B. The change in velocity is given by $\Delta \vec{v} = \frac{\pm \vec{P}}{m}$.

## Implementation Details
- **Race Condition Prevention**: A single particle might be involved in multiple simultaneous contacts (e.g., in a dense cluster). Naively adding the velocity changes would lead to thread race conditions.
- **Lock-Free Atomic Floats**: Because core Vulkan does not universally support atomic operations on floats (`VK_EXT_shader_atomic_float`), this shader implements a custom Compare-And-Swap (CAS) loop. It uses `atomicCompSwap` by coercing the IEEE-754 float bits to `uint` to accumulate velocity changes safely across all subgroups and workgroups.
