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
//! - `lod`: per-frame render LOD (`dust_lod.comp`, graphics queue): per-tier indirect draw headers,
//!   the white-point tile grid and the per-tier instance lists (see `dust::LOD_*`).
//! - `lod_readback`: host-visible ring of [`LOD_READBACK_SLOTS`] copies of the headers and tiles;
//!   the host reads a slot [`LOD_READBACK_SLOTS`] frames later to steer the budget share λ and the
//!   white point (eye adaptation).
//!
//! CPU particle mode (`AETHERVK_PARTICLES_CPU=1`) keeps the ring in host memory and runs
//! [`dust::emit_cluster`] / [`dust::evaluate_cluster`] (the reference implementation) instead of
//! the compute shaders; the evaluated clusters go to the per-frame staging arena, which `dust.vert`
//! reads through its device address. The LOD runs on the host too ([`dust::lod_evaluate`]), its
//! instance list goes to the arena and the draw is direct. Same layouts, same shaders.
use super::*;
use crate::scene::dust::{
  self, DustBatch, DustCluster, DustDrawPushConstants, DustEmitPushConstants, DustFrame,
  DustLodPushConstants, DustPropagatePushConstants, DustRenderCluster,
};

/// batch descriptor slots per system; a slot is reused after this many emissions
pub const DUST_BATCH_SLOTS: u32 = 1024;
/// ring capacity in CPU particle mode (evaluation is single threaded)
pub const DUST_RING_CAPACITY_CPU: u32 = 8192;
/// local size of `dust_emit.comp` / `dust_propagate.comp` / `dust_lod.comp`
const DUST_WG: u32 = 64;
/// frames between an LOD readback copy and its host read (≥ frames in flight)
pub const LOD_READBACK_SLOTS: usize = 4;

/// Memory of a dust buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DustMem {
  Device,
  /// host writes, device reads
  HostWrite,
  /// device writes (transfer), host reads
  HostRead,
}

/// How to draw one tier this frame (filled by [`Device::cmd_dust_lod`]).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DustLodDraw {
  /// render buffer (flux per child)
  pub render: u64,
  /// instance list (`cluster | child << LOD_CLUSTER_BITS`)
  pub list: u64,
  /// the tier's LOD header (list address, flow clock, view flags for `dust.vert`)
  pub header: u64,
  /// GPU: `VkDrawIndirectCommand` (buffer, offset)
  pub indirect: Option<(vk::Buffer, u64)>,
  /// CPU particle mode: instance count of a direct draw
  pub instances: u32,
}

/// The flow uniform and view flags in a tier's LOD header (read by `dust.vert`)
fn write_flow_words(header: &mut [u32], flags: u32, flow: &dust::DustFlowUniform) {
  header[dust::LOD_HEADER_FLOW_SPEED as usize] = flow.speed.to_bits();
  header[dust::LOD_HEADER_FLAGS as usize] = flags;
  header[dust::LOD_HEADER_T_HI as usize] = flow.t_hi.to_bits();
  header[dust::LOD_HEADER_T_LO as usize] = flow.t_lo.to_bits();
}

/// Host side of the LOD of one system: the white point feedback (the budget share is computed by
/// the LOD itself in the same frame, `dust::lod_lambda_from`).
pub struct DustLodHost {
  /// largest budget share per tier (1; tests lower it to force a split)
  pub lambda: [f32; dust::LOD_MAX_TIERS as usize],
  /// white point (exposure-scaled optical depth, raw exposure), 0 = not measured yet
  pub white: f32,
  frame: u64,
  /// per readback slot: the tile unit when it was recorded
  slot_unit: [f32; LOD_READBACK_SLOTS],
  slot_valid: [bool; LOD_READBACK_SLOTS],
  /// this frame's view aids (`dust::DUST_VIEW_*`) and flow uniform, set by `cmd_dust_lod_begin`
  pub view_flags: u32,
  pub flow: dust::DustFlowUniform,
  /// solar gravity at the jet (m/s²): the β extent of the LOD footprints (`dust::dust_extent`)
  pub sun_g: f32,
}

impl Default for DustLodHost {
  fn default() -> Self {
    Self {
      lambda: [1.0; dust::LOD_MAX_TIERS as usize],
      white: 0.0,
      frame: 0,
      slot_unit: [0.0; LOD_READBACK_SLOTS],
      slot_valid: [false; LOD_READBACK_SLOTS],
      view_flags: 0,
      flow: Default::default(),
      sun_g: 0.0,
    }
  }
}

impl DustLodHost {
  /// tile fixed-point unit: τ per count, relative to the current white point
  pub fn tile_unit(&self) -> f32 {
    let w = if self.white > 0.0 { self.white } else { 1.0 };
    w * dust::WHITE_TILE_UNIT_REL
  }

  /// Folds one frame's tiles (recorded with tile unit `unit`) into the white point.
  pub fn absorb(&mut self, words: &[u32], unit: f32) {
    let t0 = dust::LOD_TILE_WORD0 as usize;
    let tiles = &words[t0..t0 + dust::DUST_TILE_COUNT as usize];
    if let Some(w) = dust::white_point_from_tiles(tiles, unit) {
      self.white = dust::adapt_white(self.white, w);
    }
  }
}

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
  pub lod: Option<DustBuffer>,
  pub lod_readback: Option<DustBuffer>,
  lod_readback_mapped: *mut u8,
  pub lod_host: spin::Mutex<DustLodHost>,
  /// dust trace (`gpu::dust_trace`): host-visible copies of the traced render clusters per
  /// readback slot and tier, created on the first traced frame
  trace_rb: spin::Mutex<Option<(DustBuffer, *mut u8)>>,
  /// what each readback slot's traced frame drew with (the line is written when it is read back)
  trace_meta: spin::Mutex<[Option<crate::gpu::dust_trace::TraceFrameMeta>; LOD_READBACK_SLOTS]>,
  /// CPU particle mode ring
  pub cpu_ring: Option<spin::RwLock<alloc::vec::Vec<DustCluster>>>,
  /// CPU particle mode: this frame's evaluated clusters per tier (keyed by ring base)
  cpu_render: spin::Mutex<alloc::collections::BTreeMap<u32, alloc::vec::Vec<DustRenderCluster>>>,
}

impl DustGpuSystem {
  fn empty(capacity: u32) -> Self {
    Self {
      capacity,
      clusters: None,
      render: None,
      batches: None,
      batches_mapped: core::ptr::null_mut(),
      lod: None,
      lod_readback: None,
      lod_readback_mapped: core::ptr::null_mut(),
      lod_host: spin::Mutex::new(DustLodHost::default()),
      trace_rb: spin::Mutex::new(None),
      trace_meta: spin::Mutex::new([const { None }; LOD_READBACK_SLOTS]),
      cpu_ring: None,
      cpu_render: spin::Mutex::new(alloc::collections::BTreeMap::new()),
    }
  }
}

/// bytes of the LOD buffer of a system of ring `capacity`
pub fn lod_buffer_bytes(capacity: u32) -> u64 {
  (dust::LOD_LIST_WORD0 as u64 + capacity as u64 * dust::CHILDREN_PER_CLUSTER as u64) * 4
}
/// bytes of the traced clusters of one readback slot (all tiers)
pub const TRACE_SLOT_BYTES: u64 = dust::LOD_MAX_TIERS as u64
  * crate::gpu::dust_trace::TRACE_SAMPLES as u64
  * core::mem::size_of::<DustRenderCluster>() as u64;
/// bytes of one LOD readback slot
pub const LOD_READBACK_SLOT_BYTES: u64 = dust::LOD_READBACK_WORDS as u64 * 4;

// SAFETY: `batches_mapped` / `lod_readback_mapped` point into persistently mapped VMA allocations
// owned by this struct
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
    let trace = sys.trace_rb.get_mut().take().map(|(b, _)| b);
    for b in [
      trace,
      sys.clusters.take(),
      sys.render.take(),
      sys.batches.take(),
      sys.lod.take(),
      sys.lod_readback.take(),
    ]
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
  /// Micro-layer color target of the compositing pass (`R16G16B16A16_SFLOAT`, or `R8G8B8A8_UNORM`
  /// where unsupported): dust drawn into an 8-bit target is stochastically rounded.
  pub fn micro_color_format(&self) -> vk::Format {
    self.micro_color_format
  }

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
    mem: DustMem,
    concurrent: bool,
    extra_usage: vk::BufferUsageFlags,
    name: &str,
  ) -> GpuResult<(DustBuffer, *mut u8)> {
    let mut info = vk::BufferCreateInfo::default()
      .size(size)
      // TRANSFER_SRC: debug / test readbacks
      .usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
          | vk::BufferUsageFlags::TRANSFER_SRC
          | extra_usage,
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
    match mem {
      DustMem::HostWrite => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
          | vk_mem::AllocationCreateFlags::MAPPED;
      }
      DustMem::HostRead => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags =
          vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
      }
      DustMem::Device => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        alloc_info.priority = 1.0;
      }
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
      let mut sys = DustGpuSystem::empty(capacity);
      sys.cpu_ring = Some(spin::RwLock::new(
        alloc::vec![bytemuck::Zeroable::zeroed(); capacity as usize],
      ));
      sys
    } else {
      let size_c = capacity as u64 * core::mem::size_of::<DustCluster>() as u64;
      let size_r = capacity as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
      let size_b = DUST_BATCH_SLOTS as u64 * core::mem::size_of::<DustBatch>() as u64;
      let none = vk::BufferUsageFlags::empty();
      // on failure, everything created so far is released with the partial system
      let mut sys = DustGpuSystem::empty(capacity);
      let mut make = |size, mem, concurrent, usage, what: &str| {
        self.dust_create_buffer(
          &allocator,
          size,
          mem,
          concurrent,
          usage,
          &alloc::format!("{what}_{id}"),
        )
      };
      let res: GpuResult<()> = (|| {
        sys.clusters = Some(make(size_c, DustMem::Device, true, none, "DustClusters")?.0);
        sys.render = Some(make(size_r, DustMem::Device, false, none, "DustRender")?.0);
        let (b, mapped) = make(size_b, DustMem::HostWrite, false, none, "DustBatches")?;
        sys.batches = Some(b);
        sys.batches_mapped = mapped;
        sys.lod = Some(
          make(
            lod_buffer_bytes(capacity),
            DustMem::Device,
            false,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "DustLod",
          )?
          .0,
        );
        let (b, mapped) = make(
          LOD_READBACK_SLOTS as u64 * LOD_READBACK_SLOT_BYTES,
          DustMem::HostRead,
          false,
          vk::BufferUsageFlags::TRANSFER_DST,
          "DustLodReadback",
        )?;
        sys.lod_readback = Some(b);
        sys.lod_readback_mapped = mapped;
        Ok(())
      })();
      if let Err(e) = res {
        DustManager::destroy_system(&allocator, sys);
        return Err(e);
      }
      sys
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
    let trace = sys.trace_rb.get_mut().take().map(|(b, _)| b);
    for b in [
      sys.render.take(),
      sys.lod.take(),
      sys.lod_readback.take(),
      trace,
    ]
    .into_iter()
    .flatten()
    {
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

  /// Barrier before the first dust command of a graphics command buffer: previous frames' vertex
  /// and indirect reads and LOD readback copies (WAR), and host writes of the batch slots.
  pub fn cmd_dust_pre_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::VERTEX_SHADER
          | vk::PipelineStageFlags2::DRAW_INDIRECT
          | vk::PipelineStageFlags2::TRANSFER,
      )
      .src_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::TRANSFER_READ,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier between the propagates (and LOD resets) and the LOD passes: compute / transfer writes
  /// → compute reads and atomics.
  pub fn cmd_dust_pre_lod_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier after all LOD passes: compute writes → vertex shader reads, indirect draws and the
  /// readback copies.
  pub fn cmd_dust_post_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(
        vk::PipelineStageFlags2::VERTEX_SHADER
          | vk::PipelineStageFlags2::DRAW_INDIRECT
          | vk::PipelineStageFlags2::TRANSFER,
      )
      .dst_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::TRANSFER_READ,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Starts the LOD of system `id` for this frame: folds the readback slot written
  /// [`LOD_READBACK_SLOTS`] frames ago into λ and the white point, then (GPU) records the reset of
  /// the tier headers (`vertexCount = 6`, counters 0, the view aids `flags` and the `flow` uniform
  /// for `dust.vert`) and of the tile grid. Before [`Self::cmd_dust_pre_lod_barrier`].
  /// Returns the system's white point (0 = not measured yet).
  pub fn cmd_dust_lod_begin(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    flags: u32,
    flow: dust::DustFlowUniform,
    sun_g: f32,
  ) -> GpuResult<f32> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    sys.cpu_render.lock().clear();
    {
      let mut host = sys.lod_host.lock();
      host.view_flags = flags;
      host.flow = flow;
      host.sun_g = sun_g;
    }
    let (Some(lod), Some(rb)) = (sys.lod.as_ref(), sys.lod_readback.as_ref()) else {
      // CPU mode: the LOD folds its result in immediately (cmd_dust_lod)
      return Ok(sys.lod_host.lock().white);
    };
    let mut host = sys.lod_host.lock();
    let slot = (host.frame % LOD_READBACK_SLOTS as u64) as usize;
    if host.slot_valid[slot] {
      res.allocator.allocator.as_allocator_view().invalidate_allocation(
        &rb.alloc,
        slot as u64 * LOD_READBACK_SLOT_BYTES,
        LOD_READBACK_SLOT_BYTES,
      )?;
      let words = unsafe {
        core::slice::from_raw_parts(
          sys
            .lod_readback_mapped
            .add(slot * LOD_READBACK_SLOT_BYTES as usize)
            .cast::<u32>(),
          dust::LOD_READBACK_WORDS as usize,
        )
      };
      let unit = host.slot_unit[slot];
      host.absorb(words, unit);
    }
    let folded = host.slot_valid[slot];
    host.slot_valid[slot] = false;
    let white = host.white;
    drop(host);
    // the traced frame of this slot (if any) is complete too: write its line
    if let Some(meta) = sys.trace_meta.lock()[slot].take()
      && folded
      && let Some((tb, tptr)) = sys.trace_rb.lock().as_ref()
    {
      res.allocator.allocator.as_allocator_view().invalidate_allocation(
        &tb.alloc,
        slot as u64 * TRACE_SLOT_BYTES,
        TRACE_SLOT_BYTES,
      )?;
      let per_tier = crate::gpu::dust_trace::TRACE_SAMPLES;
      let all = unsafe {
        core::slice::from_raw_parts(
          tptr.add(slot * TRACE_SLOT_BYTES as usize).cast::<DustRenderCluster>(),
          dust::LOD_MAX_TIERS as usize * per_tier,
        )
      };
      let clusters: alloc::vec::Vec<alloc::vec::Vec<DustRenderCluster>> = meta
        .tiers
        .iter()
        .map(|t| {
          let base = t.tier as usize * per_tier;
          all[base..base + t.samples.len()].to_vec()
        })
        .collect();
      crate::gpu::dust_trace::append(&crate::gpu::dust_trace::record_line(&meta, &clusters));
    }
    let mut header = [0u32; (dust::LOD_MAX_TIERS * dust::LOD_HEADER_WORDS) as usize];
    for t in 0..dust::LOD_MAX_TIERS as usize {
      let h = t * dust::LOD_HEADER_WORDS as usize;
      header[h] = 6;
      write_flow_words(
        &mut header[h..h + dust::LOD_HEADER_WORDS as usize],
        flags,
        &flow,
      );
      header[h + dust::LOD_HEADER_SUN_G as usize] = sun_g.to_bits();
    }
    unsafe {
      self.device.cmd_update_buffer(cmd, lod.buffer, 0, bytemuck::cast_slice(&header));
      self.device.cmd_fill_buffer(
        cmd,
        lod.buffer,
        dust::LOD_TILE_WORD0 as u64 * 4,
        dust::DUST_TILE_COUNT as u64 * 4,
        0,
      );
    }
    Ok(white)
  }

  /// Records (GPU) or performs (CPU mode) the LOD of one tier of system `id` (see
  /// `dust_lod.comp`): `render` is the tier's render buffer from [`Self::cmd_dust_propagate`],
  /// `pc` the push constants without the buffer addresses, budget, λ and tile scale, which are
  /// filled here (`exposure` = raw dust exposure, without the white point).
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_lod(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    tier: u32,
    ring_base: u32,
    capacity: u32,
    render: u64,
    live_count: u32,
    exposure: f32,
    mvp: [f32; 16],
    params: [f32; 4],
  ) -> GpuResult<DustLodDraw> {
    if tier >= dust::LOD_MAX_TIERS {
      return Err(gpu_err!("dust tier {} has no LOD header", tier));
    }
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    if ring_base as u64 + capacity as u64 > sys.capacity as u64 {
      return Err(gpu_err!(
        "dust sub-ring {}+{} exceeds capacity {}",
        ring_base,
        capacity,
        sys.capacity
      ));
    }
    let budget = capacity * dust::CHILDREN_PER_CLUSTER;
    let (lambda, unit, flags, flow, sun_g) = {
      let host = sys.lod_host.lock();
      (
        host.lambda[tier as usize],
        host.tile_unit(),
        host.view_flags,
        host.flow,
        host.sun_g,
      )
    };
    let mut pc = DustLodPushConstants {
      render,
      header: 0,
      tiles: 0,
      list: 0,
      live_count,
      budget,
      lambda,
      tile_scale: exposure / unit,
      mvp,
      params,
    };
    let Some(lod) = sys.lod.as_ref() else {
      // CPU mode: run the mirror on the clusters propagate evaluated, upload render + list
      let mut cpu = sys.cpu_render.lock();
      let Some(clusters) = cpu.get_mut(&ring_base) else {
        return Ok(DustLodDraw::default());
      };
      let mut tiles = alloc::vec![0u32; dust::DUST_TILE_COUNT as usize];
      let mut list = alloc::vec::Vec::new();
      pc.live_count = pc.live_count.min(clusters.len() as u32);
      let out = dust::lod_evaluate(clusters, &pc, flags, sun_g, &mut tiles, &mut list);
      {
        let mut host = sys.lod_host.lock();
        let mut words = alloc::vec![0u32; dust::LOD_READBACK_WORDS as usize];
        words[dust::LOD_TILE_WORD0 as usize..].copy_from_slice(&tiles);
        host.absorb(&words, unit);
      }
      let arena_guard = utils::RwLockable::read(&res.frame_staging_arena);
      let arena = arena_guard.as_ref().ok_or(gpu_err!("no frame staging arena"))?;
      let base = unsafe {
        self
          .device
          .buffer_device_address
          .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
      };
      let rbytes = clusters.len().max(1) * core::mem::size_of::<DustRenderCluster>();
      let (roff, rptr) = arena.allocate(rbytes, 16).ok_or(GpuError::OutOfMemory)?;
      let lbytes = list.len().max(1) * 4;
      let (loff, lptr) = arena.allocate(lbytes, 16).ok_or(GpuError::OutOfMemory)?;
      let (hoff, hptr) = arena
        .allocate(dust::LOD_HEADER_WORDS as usize * 4, 16)
        .ok_or(GpuError::OutOfMemory)?;
      let list_addr = base + loff as u64;
      let mut header = [0u32; dust::LOD_HEADER_WORDS as usize];
      header[dust::LOD_HEADER_LIST as usize] = list_addr as u32;
      header[dust::LOD_HEADER_LIST as usize + 1] = (list_addr >> 32) as u32;
      write_flow_words(&mut header, flags, &flow);
      header[dust::LOD_HEADER_LAMBDA as usize] = out.lambda.to_bits();
      header[dust::LOD_HEADER_SUN_G as usize] = sun_g.to_bits();
      unsafe {
        core::ptr::copy_nonoverlapping(
          clusters.as_ptr().cast::<u8>(),
          rptr,
          clusters.len() * core::mem::size_of::<DustRenderCluster>(),
        );
        core::ptr::copy_nonoverlapping(list.as_ptr().cast::<u8>(), lptr, list.len() * 4);
        core::ptr::copy_nonoverlapping(header.as_ptr().cast::<u8>(), hptr, header.len() * 4);
      }
      return Ok(DustLodDraw {
        render: base + roff as u64,
        list: list_addr,
        header: base + hoff as u64,
        indirect: None,
        instances: out.instances,
      });
    };
    let header_off = tier as u64 * dust::LOD_HEADER_WORDS as u64 * 4;
    let list_off =
      (dust::LOD_LIST_WORD0 as u64 + ring_base as u64 * dust::CHILDREN_PER_CLUSTER as u64) * 4;
    pc.header = lod.address + header_off;
    pc.tiles = lod.address + dust::LOD_TILE_WORD0 as u64 * 4;
    pc.list = lod.address + list_off;
    {
      let mut host = sys.lod_host.lock();
      let slot = (host.frame % LOD_READBACK_SLOTS as u64) as usize;
      host.slot_unit[slot] = unit;
    }
    if live_count > 0 {
      let pipeline = self.kernels.pipelines.dust_lod;
      self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        // pass A: this frame's demand; pass B: the share that fits the budget, then the LOD
        // (`dust::lod_lambda_from`: no feedback lag, no ramp after a view change)
        for lambda in [dust::LOD_DEMAND_PASS, lambda] {
          let pc = DustLodPushConstants { lambda, ..pc };
          if lambda != dust::LOD_DEMAND_PASS {
            let b = vk::MemoryBarrier2::default()
              .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
              .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
              .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
              .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
              );
            let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
            self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
          }
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
    }
    Ok(DustLodDraw {
      render,
      list: pc.list,
      header: pc.header,
      indirect: Some((lod.buffer, header_off)),
      instances: 0,
    })
  }

  /// Traces this frame of system `id` (`gpu::dust_trace`): between the LOD passes'
  /// post barrier and [`Self::cmd_dust_lod_end`]. `ring_bases[k]` is the sub-ring base of
  /// `meta.tiers[k]`. GPU: copies the traced render clusters into this frame's readback slot (the
  /// line is written when the slot is read back); CPU mode: writes the line now.
  pub fn cmd_dust_trace(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    meta: crate::gpu::dust_trace::TraceFrameMeta,
    ring_bases: &[u32],
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let stride = core::mem::size_of::<DustRenderCluster>() as u64;
    let Some(render) = sys.render.as_ref() else {
      // CPU mode: the evaluated (and LOD-rewritten) clusters are still on the host
      let cpu = sys.cpu_render.lock();
      let clusters: alloc::vec::Vec<alloc::vec::Vec<DustRenderCluster>> = meta
        .tiers
        .iter()
        .zip(ring_bases)
        .map(|(t, base)| {
          let v = cpu.get(base);
          t.samples
            .iter()
            .map(|&(i, _)| {
              v.and_then(|v| v.get(i as usize).copied())
                .unwrap_or(bytemuck::Zeroable::zeroed())
            })
            .collect()
        })
        .collect();
      crate::gpu::dust_trace::append(&crate::gpu::dust_trace::record_line(&meta, &clusters));
      return Ok(());
    };
    let mut trace = sys.trace_rb.lock();
    if trace.is_none() {
      let allocator = res.allocator.allocator.as_allocator_view();
      *trace = Some(self.dust_create_buffer(
        &allocator,
        LOD_READBACK_SLOTS as u64 * TRACE_SLOT_BYTES,
        DustMem::HostRead,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustTrace_{id}"),
      )?);
    }
    let (tb, _) = trace.as_ref().unwrap();
    let slot = (sys.lod_host.lock().frame % LOD_READBACK_SLOTS as u64) as usize;
    let per_tier = crate::gpu::dust_trace::TRACE_SAMPLES as u64;
    let mut regions = alloc::vec::Vec::new();
    for (t, &base) in meta.tiers.iter().zip(ring_bases) {
      if t.tier >= dust::LOD_MAX_TIERS {
        continue;
      }
      for (k, &(i, _)) in t.samples.iter().enumerate().take(per_tier as usize) {
        regions.push(
          vk::BufferCopy::default()
            .src_offset((base as u64 + i as u64) * stride)
            .dst_offset(
              slot as u64 * TRACE_SLOT_BYTES + (t.tier as u64 * per_tier + k as u64) * stride,
            )
            .size(stride),
        );
      }
    }
    if !regions.is_empty() {
      unsafe { self.device.cmd_copy_buffer(cmd, render.buffer, tb.buffer, &regions) };
    }
    sys.trace_meta.lock()[slot] = Some(meta);
    Ok(())
  }

  /// Ends the LOD of system `id` for this frame (after [`Self::cmd_dust_post_propagate_barrier`]):
  /// copies the headers and tiles into this frame's readback slot.
  pub fn cmd_dust_lod_end(&self, cmd: vk::CommandBuffer, id: u64) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let (Some(lod), Some(rb)) = (sys.lod.as_ref(), sys.lod_readback.as_ref()) else {
      return Ok(());
    };
    let mut host = sys.lod_host.lock();
    let slot = (host.frame % LOD_READBACK_SLOTS as u64) as usize;
    host.slot_valid[slot] = true;
    host.frame += 1;
    let region = vk::BufferCopy::default()
      .src_offset(0)
      .dst_offset(slot as u64 * LOD_READBACK_SLOT_BYTES)
      .size(LOD_READBACK_SLOT_BYTES);
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::HOST)
      .dst_access_mask(vk::AccessFlags2::HOST_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe {
      self
        .device
        .cmd_copy_buffer(cmd, lod.buffer, rb.buffer, core::slice::from_ref(&region));
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
    }
    Ok(())
  }

  /// White point of system `id` (0 = not measured yet), see [`DustLodHost`].
  pub fn dust_white_point(&self, id: u64) -> f32 {
    let res = self.res.read();
    res
      .dust_manager
      .as_ref()
      .and_then(|m| m.systems.get(&id).map(|s| s.lod_host.lock().white))
      .unwrap_or(0.0)
  }

  /// Test / diagnostic: the readback words of the newest completed slot of system `id`, read on
  /// the host (call after the frame's submit finished).
  pub fn dust_lod_readback_latest(&self, id: u64) -> GpuResult<alloc::vec::Vec<u32>> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let rb = sys.lod_readback.as_ref().ok_or(gpu_err!("CPU mode: no readback"))?;
    let host = sys.lod_host.lock();
    let slot = (host.frame.wrapping_sub(1) % LOD_READBACK_SLOTS as u64) as usize;
    res.allocator.allocator.as_allocator_view().invalidate_allocation(
      &rb.alloc,
      slot as u64 * LOD_READBACK_SLOT_BYTES,
      LOD_READBACK_SLOT_BYTES,
    )?;
    let words = unsafe {
      core::slice::from_raw_parts(
        sys
          .lod_readback_mapped
          .add(slot * LOD_READBACK_SLOT_BYTES as usize)
          .cast::<u32>(),
        dust::LOD_READBACK_WORDS as usize,
      )
    };
    Ok(words.to_vec())
  }

  /// Test / diagnostic: the tier instance list (`count` words) of system `id`.
  pub fn dust_lod_list_buffer(&self, id: u64) -> Option<(vk::Buffer, u64)> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let sys = mgr.systems.get(&id)?;
    sys.lod.as_ref().map(|b| (b.buffer, lod_buffer_bytes(sys.capacity)))
  }

  /// Records (GPU) or performs (CPU mode) the per-frame evaluation of the live range
  /// `[first_slot, first_slot + live_count)` of the sub-ring `[ring_base, ring_base + capacity)`
  /// (one age tier) of system `id`. Must be outside a render pass.
  /// Returns the device address of the compact render buffer (0 in CPU mode, where the clusters
  /// stay on the host until [`Self::cmd_dust_lod`] uploads them).
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
      // CPU mode: evaluate on the host; cmd_dust_lod uploads the result with its instance list
      let ring = ring.read();
      let out: alloc::vec::Vec<DustRenderCluster> = (0..live_count)
        .map(|i| {
          let slot = first_slot.wrapping_add(i) & mask;
          dust::evaluate_cluster(&ring[(ring_base + slot) as usize], slot, frame)
        })
        .collect();
      sys.cpu_render.lock().insert(ring_base, out);
      return Ok(0);
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

  /// Draws the tier's LOD instance list (indirect on the GPU path). The dust pipeline must be
  /// bound.
  pub fn cmd_dust_draw(
    &self,
    cmd: vk::CommandBuffer,
    pc: &DustDrawPushConstants,
    draw: &DustLodDraw,
  ) -> GpuResult<()> {
    if draw.indirect.is_none() && draw.instances == 0 {
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
      match draw.indirect {
        Some((buffer, offset)) => self.device.cmd_draw_indirect(cmd, buffer, offset, 1, 16),
        None => self.device.cmd_draw(cmd, 6, draw.instances, 0, 0),
      }
    }
    Ok(())
  }
}
