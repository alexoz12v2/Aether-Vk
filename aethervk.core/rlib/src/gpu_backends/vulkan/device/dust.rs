//! Dust system v3 device resources and command recording. See [`crate::scene::dust`].
//!
//! Per particle system:
//! - `clusters`: ring of immutable [`DustCluster`] records (device local). Written only by
//!   `dust_emit.comp` on the compute queue, read by `dust_propagate.comp` on the graphics queue.
//!   Append-only + FIFO expiry, so the only synchronization needed is the graphics submit waiting
//!   on the compute timeline value of the newest emission it draws (no ownership transfer: the
//!   buffer is `CONCURRENT` when the queue families differ).
//! - `render`: per-frame [`DustRenderCluster`]s, written by propagate and read by `dust.vert`, both
//!   on the graphics queue (pipeline barriers only).
//! - `batches`: host-visible ring of [`DUST_BATCH_SLOTS`] [`DustBatch`] descriptors read by
//!   `dust_emit.comp` through its buffer device address.
//!
//! CPU particle mode (`AETHERVK_PARTICLES_CPU=1`) keeps the ring in host memory and runs
//! [`dust::emit_cluster`] / [`dust::evaluate_cluster`] (the reference implementation) instead of
//! the compute shaders; the evaluated clusters go to the per-frame staging arena, which `dust.vert`
//! reads through its device address. Same layouts, same draw path.
use super::*;
use crate::scene::dust::{
  self, DustBatch, DustCluster, DustDrawPushConstants, DustEmitPushConstants, DustFrame,
  DustPropagatePushConstants, DustRenderCluster,
};

/// batch descriptor slots per system; a slot is reused after this many emissions
pub const DUST_BATCH_SLOTS: u32 = 1024;
/// ring capacity in CPU particle mode (evaluation is single threaded)
pub const DUST_RING_CAPACITY_CPU: u32 = 8192;
/// local size of `dust_emit.comp` / `dust_propagate.comp`
const DUST_WG: u32 = 64;

pub struct DustBuffer {
  pub buffer: vk::Buffer,
  pub alloc: vk_mem::Allocation,
  pub address: u64,
}

pub struct DustGpuSystem {
  pub capacity: u32,
  pub clusters: Option<DustBuffer>,
  pub render: Option<DustBuffer>,
  pub batches: Option<DustBuffer>,
  batches_mapped: *mut u8,
  /// CPU particle mode ring
  pub cpu_ring: Option<spin::RwLock<alloc::vec::Vec<DustCluster>>>,
}

// SAFETY: `batches_mapped` points into a persistently mapped VMA allocation owned by this struct
unsafe impl Send for DustGpuSystem {}
unsafe impl Sync for DustGpuSystem {}

pub struct DustManager {
  pub systems: dashmap::DashMap<u64, DustGpuSystem>,
  allocator_view: vk_mem::AllocatorView,
}

impl DustManager {
  pub fn new(allocator_view: vk_mem::AllocatorView) -> Self {
    Self {
      systems: dashmap::DashMap::new(),
      allocator_view,
    }
  }

  fn destroy_system(allocator: &vk_mem::AllocatorView, mut sys: DustGpuSystem) {
    for b in [sys.clusters.take(), sys.render.take(), sys.batches.take()]
      .into_iter()
      .flatten()
    {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
  }
}

impl Drop for DustManager {
  fn drop(&mut self) {
    let ids: alloc::vec::Vec<u64> = self.systems.iter().map(|e| *e.key()).collect();
    for id in ids {
      if let Some((_, sys)) = self.systems.remove(&id) {
        Self::destroy_system(&self.allocator_view, sys);
      }
    }
  }
}

impl Device {
  /// Ring capacity for this device: high tier on discrete GPUs, low tier otherwise, small in CPU
  /// particle mode. `AETHERVK_DUST_RING=<power of two>` overrides it.
  pub fn dust_ring_capacity(&self) -> u32 {
    let env = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_RING")
      .and_then(|s| s.trim().parse::<u32>().ok());
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return dust::ring_capacity(false, env.or(Some(DUST_RING_CAPACITY_CPU)));
    }
    let high_end = self.query_result.physical_device_properties.device_type
      == vk::PhysicalDeviceType::DISCRETE_GPU;
    dust::ring_capacity(high_end, env)
  }

  fn dust_create_buffer(
    &self,
    allocator: &vk_mem::AllocatorView,
    size: u64,
    host_visible: bool,
    concurrent: bool,
    name: &str,
  ) -> GpuResult<(DustBuffer, *mut u8)> {
    let mut info = vk::BufferCreateInfo::default()
      .size(size)
      // TRANSFER_SRC: debug / test readbacks
      .usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
          | vk::BufferUsageFlags::TRANSFER_SRC,
      );
    let families = [
      self.get_compute_queue().family_index,
      self.get_graphics_queue().family_index,
    ];
    if concurrent && families[0] != families[1] {
      info = info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families);
    }
    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(alloc_info);
    if host_visible {
      alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
      alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED;
    } else {
      alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
      alloc_info.priority = 1.0;
    }
    let (buffer, alloc, alloc_res) =
      unsafe { allocator.create_buffer_get_info(&info, &alloc_info) }
        .with_name(&self.device, name)?;
    let address = unsafe {
      self
        .device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };
    Ok((
      DustBuffer {
        buffer,
        alloc,
        address,
      },
      alloc_res.mapped_data.cast(),
    ))
  }

  /// Allocates the dust resources of particle system `id`. Returns the ring capacity.
  pub fn create_particle_system(&self, id: u64) -> GpuResult<u32> {
    let capacity = self.dust_ring_capacity();
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    let sys = if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      DustGpuSystem {
        capacity,
        clusters: None,
        render: None,
        batches: None,
        batches_mapped: core::ptr::null_mut(),
        cpu_ring: Some(spin::RwLock::new(
          alloc::vec![bytemuck::Zeroable::zeroed(); capacity as usize],
        )),
      }
    } else {
      let size_c = capacity as u64 * core::mem::size_of::<DustCluster>() as u64;
      let size_r = capacity as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
      let size_b = DUST_BATCH_SLOTS as u64 * core::mem::size_of::<DustBatch>() as u64;
      let (clusters, _) = self.dust_create_buffer(
        &allocator,
        size_c,
        false,
        true,
        &alloc::format!("DustClusters_{id}"),
      )?;
      let render = match self.dust_create_buffer(
        &allocator,
        size_r,
        false,
        false,
        &alloc::format!("DustRender_{id}"),
      ) {
        Ok((b, _)) => b,
        Err(e) => {
          DustManager::destroy_system(
            &allocator,
            DustGpuSystem {
              capacity,
              clusters: Some(clusters),
              render: None,
              batches: None,
              batches_mapped: core::ptr::null_mut(),
              cpu_ring: None,
            },
          );
          return Err(e);
        }
      };
      let (batches, mapped) = match self.dust_create_buffer(
        &allocator,
        size_b,
        true,
        false,
        &alloc::format!("DustBatches_{id}"),
      ) {
        Ok(x) => x,
        Err(e) => {
          DustManager::destroy_system(
            &allocator,
            DustGpuSystem {
              capacity,
              clusters: Some(clusters),
              render: Some(render),
              batches: None,
              batches_mapped: core::ptr::null_mut(),
              cpu_ring: None,
            },
          );
          return Err(e);
        }
      };
      DustGpuSystem {
        capacity,
        clusters: Some(clusters),
        render: Some(render),
        batches: Some(batches),
        batches_mapped: mapped,
        cpu_ring: None,
      }
    };
    if let Some(old) = mgr.systems.insert(id, sys) {
      DustManager::destroy_system(&allocator, old);
    }
    aethervk_oshal_rlib::log!("[Dust] system {id}: ring capacity {capacity}");
    Ok(capacity)
  }

  /// Discards the resources of particle system `id`, once both queues are past the given values.
  pub fn discard_particle_system(
    &self,
    id: u64,
    gfx_timeline: u64,
    comp_timeline: u64,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let (_, mut sys) = mgr
      .systems
      .remove(&id)
      .ok_or(gpu_err!("particle system with id {} not found", id))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    // clusters: written by compute, read by graphics. Wait (bounded) for the last compute submit,
    // then retire it with the graphics frames that may still read it. Discards are rare.
    if let Some(b) = sys.clusters.take() {
      let last_compute = comp_timeline.saturating_sub(1);
      if last_compute > 0 {
        let _ =
          self
            .device
            .wait_for_semaphore_value(self.kernels.timeline, last_compute, 1_000_000_000);
      }
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
    if let Some(b) = sys.batches.take() {
      self.kernels.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        comp_timeline,
      );
    }
    if let Some(b) = sys.render.take() {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
    Ok(())
  }

  /// Records the emission of `batch` (its `first_index`, `count`, `ring_mask` already set) into
  /// the sub-ring starting at slot `ring_base` (age tier, see `dust::DustSystemState`).
  /// `seq` is the monotonic batch sequence number of the system (selects the descriptor slot).
  /// CPU particle mode: emits on the host immediately, records nothing.
  pub fn cmd_dust_emit(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    batch: &DustBatch,
    seq: u64,
  ) -> GpuResult<()> {
    if batch.count == 0 {
      return Ok(());
    }
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    if ring_base as u64 + batch.ring_mask as u64 + 1 > sys.capacity as u64 {
      return Err(gpu_err!(
        "dust sub-ring {}+{} exceeds capacity {}",
        ring_base,
        batch.ring_mask + 1,
        sys.capacity
      ));
    }
    if let Some(ring) = sys.cpu_ring.as_ref() {
      let mut ring = ring.write();
      for j in 0..batch.count {
        let slot = (batch.first_index.wrapping_add(j) & batch.ring_mask) as usize;
        ring[ring_base as usize + slot] = dust::emit_cluster(batch, j);
      }
      return Ok(());
    }
    let (Some(clusters), Some(batches)) = (sys.clusters.as_ref(), sys.batches.as_ref()) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", id));
    };
    let slot = (seq % DUST_BATCH_SLOTS as u64) as usize;
    let stride = core::mem::size_of::<DustBatch>();
    unsafe {
      core::ptr::copy_nonoverlapping(
        bytemuck::bytes_of(batch).as_ptr(),
        sys.batches_mapped.add(slot * stride),
        stride,
      );
    }
    res.allocator.allocator.as_allocator_view().flush_allocation(
      &batches.alloc,
      (slot * stride) as u64,
      stride as u64,
    )?;
    let cluster_size = core::mem::size_of::<DustCluster>() as u64;
    let pc = DustEmitPushConstants {
      // the shaders index the sub-ring from its own base address
      clusters: clusters.address + ring_base as u64 * cluster_size,
      batch: batches.address + (slot * stride) as u64,
    };
    let pipeline = self.kernels.pipelines.dust_emit;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, batch.count.div_ceil(DUST_WG), 1, 1);
    }
    Ok(())
  }

  /// Barrier before the first propagate of a graphics command buffer: previous draws' vertex
  /// reads of the render buffers (WAR) and host writes of the batch slots.
  pub fn cmd_dust_pre_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier after all propagates: compute writes → vertex shader reads.
  pub fn cmd_dust_post_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Records (GPU) or performs (CPU mode) the per-frame evaluation of the live range
  /// `[first_slot, first_slot + live_count)` of the sub-ring `[ring_base, ring_base + capacity)`
  /// (one age tier) of system `id`. Must be outside a render pass.
  /// Returns the device address of the compact render buffer to draw from.
  pub fn cmd_dust_propagate(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
  ) -> GpuResult<u64> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    if !capacity.is_power_of_two() || ring_base as u64 + capacity as u64 > sys.capacity as u64 {
      return Err(gpu_err!(
        "dust sub-ring {}+{} exceeds capacity {}",
        ring_base,
        capacity,
        sys.capacity
      ));
    }
    let live_count = live_count.min(capacity);
    let mask = capacity - 1;
    if let Some(ring) = sys.cpu_ring.as_ref() {
      // CPU mode: evaluate into the frame staging arena (host visible, device addressable)
      let arena_guard = utils::RwLockable::read(&res.frame_staging_arena);
      let arena = arena_guard.as_ref().ok_or(gpu_err!("no frame staging arena"))?;
      let bytes = (live_count.max(1) as usize) * core::mem::size_of::<DustRenderCluster>();
      let (offset, ptr) = arena.allocate(bytes, 16).ok_or(GpuError::OutOfMemory)?;
      let ring = ring.read();
      let out = unsafe {
        core::slice::from_raw_parts_mut(ptr.cast::<DustRenderCluster>(), live_count as usize)
      };
      for (i, o) in out.iter_mut().enumerate() {
        let slot = first_slot.wrapping_add(i as u32) & mask;
        *o = dust::evaluate_cluster(&ring[(ring_base + slot) as usize], slot, frame);
      }
      let base = unsafe {
        self
          .device
          .buffer_device_address
          .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
      };
      return Ok(base + offset as u64);
    }
    let (Some(clusters), Some(render)) = (sys.clusters.as_ref(), sys.render.as_ref()) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", id));
    };
    // each tier evaluates into its own part of the render buffer (drawn by its own draw call)
    let render_offset = ring_base as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    if live_count > 0 {
      let pc = DustPropagatePushConstants {
        clusters: clusters.address + ring_base as u64 * core::mem::size_of::<DustCluster>() as u64,
        render: render.address + render_offset,
        first_slot,
        live_count,
        ring_mask: mask,
        _pad0: 0,
        frame: *frame,
      };
      let pipeline = self.kernels.pipelines.dust_propagate;
      self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&pc),
        );
        self.device.cmd_dispatch(cmd, live_count.div_ceil(DUST_WG), 1, 1);
      }
    }
    Ok(render.address + render_offset)
  }

  /// Draws `live_count × children` instanced quads. The dust pipeline must be bound.
  pub fn cmd_dust_draw(&self, cmd: vk::CommandBuffer, pc: &DustDrawPushConstants) -> GpuResult<()> {
    if pc.live_count == 0 {
      return Ok(());
    }
    let layout = {
      let res = self.res.read();
      let arena_lock = res.dust_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena_arc = arena_lock.read();
      arena_arc.pipeline_layout.get()
    };
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        bytemuck::bytes_of(pc),
      );
      let children = pc.children.clamp(1, dust::MAX_CHILDREN_PER_CLUSTER);
      self.device.cmd_draw(cmd, 6, pc.live_count * children, 0, 0);
    }
    Ok(())
  }
}
