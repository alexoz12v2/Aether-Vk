# IMEX Explaination and documentation

## Physics Formulas for particles and rigidbodies

TODO ADD HERE all formulas

### Aside: Rotation Representation

TODO 

## Principles

We want to partition a step of the and ODE solver iteration by explicitly partitioning our differential equations

- *Velocity Vertlet* or *Leapfrog* method for Particles in $\mathbb{R}^{3N}$, which are represented by their euclidean space position
  $q$
- *Implicit Midpoint Rule* for the $\mathbb{R}^{3N}\times \mathrm{SO}_3$ rigid bodies.

With this choice of algoritms we 

- avoid monolithic jacobian inversion (TODO ADD EXPLAINATION HERE) stiffness matrix = 0 for dust. explain
- velocity vertle requires evaluating $F(q_{n+1})$ to finish phase 5. To avoid computing forces in 2 steps (which we might to in future as it can be reused)

If we don't want to compute forces for the next position in the nth step, we turn for particles towards *Leapfrog Stagger*,
for which, instead of storing $v_n$, we store $v_{n-1/2}$

- Phase 1, 2: Kick by $h F_n$ to reach $v_{n+1/2}$, then drift to $q_{\mathrm{mid}}$.
- Phase 5: Complete the drift to $q_{n+1}$ and stop

The velocity left in the buffer is $v_{n+1/2}$, which is the exact average velocity over the interval. This makes
*Continuous Collision Detection* math perfectly linear (Linear intra-step motion assumption)

```math
q(t) = q_n + t v_{n+1/2}
```

Kernels implementation example

```rust
impl Kernels for VulkanPhysicsEngine {
    // ... setup and other trait methods ...

    fn step_ode(
        &self,
        cmd: &mut Self::Cmd,
        dynamics: &mut Self::Buffer<DynamicBody>,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let vk_cmd = cmd.get_raw(); // Retrieve underlying ash vk::CommandBuffer
        let h_sec = dt.as_seconds_f32();
        
        let pc = ImexPushConstants { h: h_sec, count: dynamics.capacity() as u32 };

        // Zero-overhead memory barrier configuration
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            // ==========================================
            // DISPATCH 1: Particle Phase 1 & 2 
            // ==========================================
            self.device.cmd_bind_pipeline(vk_cmd, vk::PipelineBindPoint::COMPUTE, self.pipe_imex_p12);
            self.device.cmd_push_constants(vk_cmd, self.pipe_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
            self.device.cmd_dispatch(vk_cmd, (pc.count + 255) / 256, 1, 1);

            // BARRIER A: Freeze `q_mid`
            self.device.cmd_pipeline_barrier(
                vk_cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER, 
                vk::PipelineStageFlags::COMPUTE_SHADER, 
                vk::DependencyFlags::empty(),
                core::slice::from_ref(&memory_barrier), &[], &[]
            );

            // ==========================================
            // DISPATCH 2: Rigid Body Phase 3 & 4 (IMR)
            // ==========================================
            self.device.cmd_bind_pipeline(vk_cmd, vk::PipelineBindPoint::COMPUTE, self.pipe_imex_rigidbody);
            self.device.cmd_dispatch(vk_cmd, (self.rigidbody_count + 31) / 32, 1, 1);

            // BARRIER B: Freeze Rigid Bodies at t_{n+1}
            self.device.cmd_pipeline_barrier(
                vk_cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                core::slice::from_ref(&memory_barrier), &[], &[]
            );

            // ==========================================
            // DISPATCH 3: Particle Phase 5
            // ==========================================
            self.device.cmd_bind_pipeline(vk_cmd, vk::PipelineBindPoint::COMPUTE, self.pipe_imex_p5);
            self.device.cmd_dispatch(vk_cmd, (pc.count + 255) / 256, 1, 1);

            // BARRIER C: Prepare t_{n+1} geometries for CCD & BVH Builders
            self.device.cmd_pipeline_barrier(
                vk_cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                core::slice::from_ref(&memory_barrier), &[], &[]
            );
        }

        Ok(())
    }
}
```
