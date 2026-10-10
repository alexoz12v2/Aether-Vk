//! Dust system v3/v4 device resources and command recording. See [`crate::scene::dust`] and
//! [`crate::scene::dust::splat`].
//!
//! Per particle system:
//! - `clusters`: ring of immutable [`DustCluster`] records (device local). Written only by
//!   `dust_emit.comp` on the compute queue, read by `dust_propagate.comp` on the graphics queue.
//!   Append-only + FIFO expiry, so the only synchronization needed is the graphics submit waiting
//!   on the compute timeline value of the newest emission it draws (no ownership transfer: the
//!   buffer is `CONCURRENT` when the queue families differ).
//! - `render`: per-frame [`DustRenderCluster`]s (positions, predecessor words), written by
//!   propagate, read by `dust_splat.comp` and the dust trace.
//! - `moments`: per-frame [`DustMoments`] (the packets' second moments), written by propagate,
//!   read by `dust_splat.comp`.
//! - `batches`: host-visible ring of [`DUST_BATCH_SLOTS`] [`DustBatch`] descriptors read by
//!   `dust_emit.comp` through its buffer device address.
//!
//! Per presentation engine ([`DustView`]): the fixed-point splat pyramid (`dust::PyramidLayout`)
//! every system of the frame scatters into, zeroed per frame, read by `composite.frag` through
//! its device address; a host-visible readback ring of the header and the white-point level
//! ([`dust::DUST_WHITE_LEVEL`]), read [`DUST_READBACK_SLOTS`] frames later for the white point,
//! the auto softening and the fixed-point unit (eye adaptation).
//!
//! CPU particle mode (`AETHERVK_PARTICLES_CPU=1`) keeps the ring in host memory and runs
//! [`dust::emit_cluster`] / [`dust::packet_moments`] / [`dust::splat_tier`] (the reference
//! implementation) instead of the compute shaders; the pyramid is then a host-visible buffer the
//! host writes directly and the composite reads through the same address. Same layouts, same
//! composite.
use super::*;
use crate::scene::dust::{
  self, DustBatch, DustCluster, DustEmitPushConstants, DustFrame, DustMoments,
  DustPropagatePushConstants, DustRenderCluster, DustSplatPushConstants, PyramidLayout,
};

/// batch descriptor slots per system; a slot is reused after this many emissions (a tick records
/// at most `logic_thread::utils::DUST_BATCHES_PER_TICK` before it waits for its own submission)
pub const DUST_BATCH_SLOTS: u32 = 4096;
/// ring capacity in CPU particle mode (evaluation is single threaded): the low GPU tier, whose
/// 16 / 8 / 8 streams keep the image within a few % of the 262 144 ring
/// (`dust_tau_image_vs_ring_capacity`); 8 192 (4 / 2 / 2) was ~15 % off in extent
pub const DUST_RING_CAPACITY_CPU: u32 = 32_768;
/// local size of `dust_emit.comp` / `dust_propagate.comp` / `dust_splat.comp` / `dust_measure.comp`
const DUST_WG: u32 = 64;
/// frames between a readback copy and its host read (≥ frames in flight)
pub const DUST_READBACK_SLOTS: usize = 4;
/// alias kept for the dust trace
pub const LOD_READBACK_SLOTS: usize = DUST_READBACK_SLOTS;

/// Memory of a dust buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DustMem {
  Device,
  /// host writes, device reads
  HostWrite,
  /// device writes (transfer), host reads
  HostRead,
}

pub struct DustBuffer {
  pub buffer: vk::Buffer,
  pub alloc: vk_mem::Allocation,
  pub address: u64,
}

/// Per-frame host state of one system (the dust trace ring)
pub struct DustSystemHost {
  pub frame: u64,
  /// this frame's view aids (`dust::DUST_VIEW_*`) and flow uniform, set by `cmd_dust_system_begin`
  pub view_flags: u32,
  pub flow: dust::DustFlowUniform,
  slot_valid: [bool; DUST_READBACK_SLOTS],
}

pub struct DustGpuSystem {
  pub capacity: u32,
  pub clusters: Option<DustBuffer>,
  pub render: Option<DustBuffer>,
  pub moments: Option<DustBuffer>,
  pub batches: Option<DustBuffer>,
  batches_mapped: *mut u8,
  pub host: spin::Mutex<DustSystemHost>,
  /// dust trace (`gpu::dust_trace`): host-visible copies of the traced render clusters per
  /// readback slot and tier, created on the first traced frame
  trace_rb: spin::Mutex<Option<(DustBuffer, *mut u8)>>,
  /// what each readback slot's traced frame drew with (the line is written when it is read back)
  trace_meta: spin::Mutex<[Option<crate::gpu::dust_trace::TraceFrameMeta>; DUST_READBACK_SLOTS]>,
  /// CPU particle mode ring
  pub cpu_ring: Option<spin::RwLock<alloc::vec::Vec<DustCluster>>>,
  /// CPU particle mode: this frame's evaluated clusters and moments per tier (keyed by ring base)
  cpu_render: spin::Mutex<
    alloc::collections::BTreeMap<
      u32,
      (
        alloc::vec::Vec<DustRenderCluster>,
        alloc::vec::Vec<DustMoments>,
      ),
    >,
  >,
}

impl DustGpuSystem {
  fn empty(capacity: u32) -> Self {
    Self {
      capacity,
      clusters: None,
      render: None,
      moments: None,
      batches: None,
      batches_mapped: core::ptr::null_mut(),
      host: spin::Mutex::new(DustSystemHost {
        frame: 0,
        view_flags: 0,
        flow: Default::default(),
        slot_valid: [false; DUST_READBACK_SLOTS],
      }),
      trace_rb: spin::Mutex::new(None),
      trace_meta: spin::Mutex::new([const { None }; DUST_READBACK_SLOTS]),
      cpu_ring: None,
      cpu_render: spin::Mutex::new(alloc::collections::BTreeMap::new()),
    }
  }
}

/// bytes of the traced clusters of one readback slot (all tiers)
pub const TRACE_SLOT_BYTES: u64 = dust::LOD_MAX_TIERS as u64
  * crate::gpu::dust_trace::TRACE_SAMPLES as u64
  * core::mem::size_of::<DustRenderCluster>() as u64;

// SAFETY: `batches_mapped` points into a persistently mapped VMA allocation owned by this struct
unsafe impl Send for DustGpuSystem {}
unsafe impl Sync for DustGpuSystem {}

/// Host side of one view's splat pyramid: the white point feedback (`dust::white_point_from_level`),
/// the auto softening and the fixed-point unit.
pub struct DustViewHost {
  /// white point (exposure-scaled optical depth per px², raw exposure), 0 = not measured yet
  pub white: f32,
  /// median dust texel of the last measured frame (same units), 0 = none
  pub p50: f32,
  /// largest packet peak optical depth per px² of the last measured frame (raw exposure units,
  /// `dust::PYR_TAU_MAX`): the fixed-point unit follows it, so no view measures as empty
  pub tau_max: f32,
  pub frame: u64,
  /// per readback slot: the unit when it was recorded
  slot_unit: [f32; DUST_READBACK_SLOTS],
  slot_valid: [bool; DUST_READBACK_SLOTS],
}

impl Default for DustViewHost {
  fn default() -> Self {
    Self {
      white: 0.0,
      p50: 0.0,
      tau_max: 0.0,
      frame: 0,
      slot_unit: [0.0; DUST_READBACK_SLOTS],
      slot_valid: [false; DUST_READBACK_SLOTS],
    }
  }
}

impl DustViewHost {
  /// fixed-point unit: exposure-scaled optical depth × px² per count, relative to the largest
  /// packet peak of the last measured frame (1 before any: the first frames may saturate or read
  /// empty, the maximum is measured regardless, so the unit is right from the next readback on)
  pub fn unit(&self) -> f32 {
    let m = if self.tau_max > 0.0 && self.tau_max.is_finite() {
      self.tau_max
    } else {
      1.0
    };
    m * dust::WHITE_TILE_UNIT_REL
  }

  /// Folds one frame's header and white-point level (recorded with `unit`) into the white
  /// point, the median and the unit.
  pub fn absorb(&mut self, header: &[u32], level: &[u32], texel_px: u32, unit: f32) {
    if let Some((p99, p50)) = dust::white_point_from_level(level, texel_px, unit) {
      self.white = dust::adapt_white(self.white, p99);
      self.p50 = dust::adapt_white(self.p50, p50);
    }
    let v = f32::from_bits(header[dust::PYR_TAU_MAX as usize]);
    if v.is_finite() && v > 0.0 {
      self.tau_max = v;
    }
  }

  /// what the composite needs: `(white, auto softening)`
  pub fn display(&self) -> (f32, f32) {
    (
      dust::display_white_v4(self.white),
      dust::auto_softening(self.p50, self.white),
    )
  }
}

/// The splat pyramid of one presentation engine.
pub struct DustView {
  pub layout: PyramidLayout,
  pub pyramid: Option<DustBuffer>,
  /// CPU particle mode: the pyramid is host-visible and written directly; one per frame in
  /// flight (`DUST_READBACK_SLOTS`), since the GPU may still read the previous frames' while the
  /// host zeroes and fills the next (the first entry is `pyramid`'s)
  cpu_pyramids: alloc::vec::Vec<(DustBuffer, *mut u8)>,
  pyramid_mapped: *mut u8,
  pub readback: Option<DustBuffer>,
  readback_mapped: *mut u8,
  pub host: spin::Mutex<DustViewHost>,
  /// this frame's view words (flags, flow, tracer counts), written into the header
  frame_words: spin::Mutex<[u32; dust::PYRAMID_HEADER_WORDS as usize]>,
}

// SAFETY: the mapped pointers point into persistently mapped VMA allocations owned by this struct
unsafe impl Send for DustView {}
unsafe impl Sync for DustView {}

impl DustView {
  /// CPU mode: the host pyramid (buffer, mapped words) of frame `frame`
  fn cpu_pyramid(&self, frame: u64) -> Option<(&DustBuffer, *mut u8)> {
    let k = (frame % DUST_READBACK_SLOTS as u64) as usize;
    if k == 0 {
      self.pyramid.as_ref().map(|b| (b, self.pyramid_mapped))
    } else {
      self.cpu_pyramids.get(k - 1).map(|(b, m)| (b, *m))
    }
  }

  /// words copied back per frame: the header and the white-point measurement grid
  fn readback_words(layout: &PyramidLayout) -> u32 {
    let (_, w, h) = layout.measure;
    dust::PYRAMID_HEADER_WORDS + w * h * dust::PYRAMID_TEXEL_WORDS
  }
  /// `(level, offset, width, height)` of the measurement grid
  fn white_level(layout: &PyramidLayout) -> (u32, u32, u32, u32) {
    let (off, w, h) = layout.measure;
    (dust::DUST_WHITE_LEVEL, off, w, h)
  }
}

pub struct DustManager {
  pub systems: dashmap::DashMap<u64, DustGpuSystem>,
  pub views: dashmap::DashMap<gpu::PresentationEngineHandle, DustView>,
  allocator_view: vk_mem::AllocatorView,
}

impl DustManager {
  pub fn new(allocator_view: vk_mem::AllocatorView) -> Self {
    Self {
      systems: dashmap::DashMap::new(),
      views: dashmap::DashMap::new(),
      allocator_view,
    }
  }

  fn destroy_system(allocator: &vk_mem::AllocatorView, mut sys: DustGpuSystem) {
    let trace = sys.trace_rb.get_mut().take().map(|(b, _)| b);
    for b in [
      trace,
      sys.clusters.take(),
      sys.render.take(),
      sys.moments.take(),
      sys.batches.take(),
    ]
    .into_iter()
    .flatten()
    {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
  }

  fn destroy_view(allocator: &vk_mem::AllocatorView, mut view: DustView) {
    let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
    for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
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
    let pes: alloc::vec::Vec<_> = self.views.iter().map(|e| *e.key()).collect();
    for pe in pes {
      if let Some((_, v)) = self.views.remove(&pe) {
        Self::destroy_view(&self.allocator_view, v);
      }
    }
  }
}

/// What the composite needs of a view's dust this frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DustViewParams {
  /// pyramid device address (0: no dust)
  pub pyramid: u64,
  pub levels: u32,
  /// exposure-scaled optical depth × px² per count
  pub unit: f32,
  /// white point (1 before any measurement)
  pub white: f32,
  /// auto softening (`dust::auto_softening`)
  pub softening: f32,
}

impl Device {
  /// Micro-layer color target of the compositing pass (`R16G16B16A16_SFLOAT`, or `R8G8B8A8_UNORM`
  /// where unsupported).
  pub fn micro_color_format(&self) -> vk::Format {
    self.micro_color_format
  }

  /// Dust accumulation target of the compositing pass (unused by the v4 renderer, kept until the
  /// attachment is removed; see `choose_dust_accum_format`).
  pub fn dust_accum_format(&self) -> vk::Format {
    self.dust_accum_format
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
    super::fault::register_address_range(name, address, size);
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
      let size_m = capacity as u64 * core::mem::size_of::<DustMoments>() as u64;
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
        sys.moments = Some(make(size_m, DustMem::Device, false, none, "DustMoments")?.0);
        let (b, mapped) = make(size_b, DustMem::HostWrite, false, none, "DustBatches")?;
        sys.batches = Some(b);
        sys.batches_mapped = mapped;
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
    for b in [sys.render.take(), sys.moments.take(), trace].into_iter().flatten() {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
    Ok(())
  }

  /// Discards the dust view of presentation engine `pe` (its pyramid), once the graphics queue is
  /// past `gfx_timeline`. No-op without one.
  pub fn discard_dust_view(&self, pe: gpu::PresentationEngineHandle, gfx_timeline: u64) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else {
      return;
    };
    let Some((_, mut view)) = mgr.views.remove(&pe) else {
      return;
    };
    let allocator = res.allocator.allocator.as_allocator_view();
    let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
    for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
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

  /// Barrier before the first dust command of a graphics command buffer: the previous frame's
  /// composite (fragment reads of the pyramid), trace / readback copies (WAR) and host writes of
  /// the batch slots, before this frame's compute writes and the pyramid zeroing.
  pub fn cmd_dust_pre_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::FRAGMENT_SHADER
          | vk::PipelineStageFlags2::COMPUTE_SHADER
          | vk::PipelineStageFlags2::TRANSFER,
      )
      .src_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE
          | vk::AccessFlags2::TRANSFER_READ,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier between the propagates (and the pyramid reset) and the splat passes: compute /
  /// transfer writes → compute reads and atomics.
  pub fn cmd_dust_pre_splat_barrier(&self, cmd: vk::CommandBuffer) {
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

  /// Barrier after all splat passes: compute writes → the composite's fragment reads and the
  /// readback / trace copies.
  pub fn cmd_dust_post_splat_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Starts the frame of system `id`: stores the view aids and the flow uniform, writes the
  /// trace line of the slot read back this frame. Before [`Self::cmd_dust_pre_splat_barrier`].
  pub fn cmd_dust_system_begin(
    &self,
    id: u64,
    flags: u32,
    flow: dust::DustFlowUniform,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    sys.cpu_render.lock().clear();
    let (slot, folded) = {
      let mut host = sys.host.lock();
      host.view_flags = flags;
      host.flow = flow;
      let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
      let folded = host.slot_valid[slot];
      host.slot_valid[slot] = false;
      (slot, folded)
    };
    // the traced frame of this slot (if any) is complete: write its line
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
    Ok(())
  }

  /// Ends the frame of system `id`: the trace slot of this frame becomes readable in
  /// [`DUST_READBACK_SLOTS`] frames.
  pub fn cmd_dust_system_end(&self, id: u64) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let mut host = sys.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    host.slot_valid[slot] = sys.render.is_some();
    host.frame += 1;
    Ok(())
  }

  /// Starts the dust frame of presentation engine `pe` of `extent`: creates (or re-creates on a
  /// size change) the view's pyramid, folds the readback slot written [`DUST_READBACK_SLOTS`]
  /// frames ago into the white point, median and unit, then records the header write (level
  /// table, `flags`, `flow`, the tracer height) and the zeroing of the levels. Returns this
  /// frame's [`DustViewParams`] (the unit the splats must use). Before the propagates.
  pub fn cmd_dust_frame_begin(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    extent: [u32; 2],
    flags: u32,
    flow: dust::DustFlowUniform,
  ) -> GpuResult<DustViewParams> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    let cpu = crate::gpu_backends::vulkan::physics::is_cpu_particles_mode();
    let layout = PyramidLayout::new(extent[0], extent[1]);
    // (re)create on first use or a size change
    let stale = mgr.views.get(&pe).is_some_and(|v| v.layout != layout);
    if stale {
      let timeline = res.timeline_manager.get_next_submit_value();
      if let Some((_, mut view)) = mgr.views.remove(&pe) {
        let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
        for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
          res.discard_pool.discard_buffer(
            allocator.as_allocator_view(),
            b.buffer,
            b.alloc,
            timeline,
          );
        }
      }
    }
    if !mgr.views.contains_key(&pe) {
      let bytes = layout.total_words as u64 * 4;
      let (pyramid, pmapped) = self.dust_create_buffer(
        &allocator,
        bytes,
        if cpu {
          DustMem::HostWrite
        } else {
          DustMem::Device
        },
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustPyramid_{}", pe.0),
      )?;
      let mut cpu_pyramids = alloc::vec::Vec::new();
      if cpu {
        for k in 1..DUST_READBACK_SLOTS {
          match self.dust_create_buffer(
            &allocator,
            bytes,
            DustMem::HostWrite,
            false,
            vk::BufferUsageFlags::TRANSFER_DST,
            &alloc::format!("DustPyramid_{}_{k}", pe.0),
          ) {
            Ok(x) => cpu_pyramids.push(x),
            Err(e) => {
              let mut a = pyramid.alloc;
              unsafe { allocator.destroy_buffer(pyramid.buffer, &mut a) };
              for (b, _) in cpu_pyramids {
                let mut a = b.alloc;
                unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
              }
              return Err(e);
            }
          }
        }
      }
      let rb_bytes = DUST_READBACK_SLOTS as u64 * DustView::readback_words(&layout) as u64 * 4;
      let (readback, rmapped) = match self.dust_create_buffer(
        &allocator,
        rb_bytes,
        DustMem::HostRead,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustPyramidReadback_{}", pe.0),
      ) {
        Ok(x) => x,
        Err(e) => {
          let mut a = pyramid.alloc;
          unsafe { allocator.destroy_buffer(pyramid.buffer, &mut a) };
          return Err(e);
        }
      };
      aethervk_oshal_rlib::log!(
        "[Dust] view {}: pyramid {}x{} ({} levels, {} MB)",
        pe.0,
        layout.width,
        layout.height,
        layout.level_count(),
        bytes / (1024 * 1024)
      );
      mgr.views.insert(
        pe,
        DustView {
          layout: layout.clone(),
          pyramid: Some(pyramid),
          cpu_pyramids,
          pyramid_mapped: pmapped,
          readback: Some(readback),
          readback_mapped: rmapped,
          host: spin::Mutex::new(DustViewHost::default()),
          frame_words: spin::Mutex::new([0; dust::PYRAMID_HEADER_WORDS as usize]),
        },
      );
    }
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view absent"))?;
    let frame_no = view.host.lock().frame;
    let (pyramid, pyramid_mapped) = if cpu {
      view.cpu_pyramid(frame_no).ok_or(gpu_err!("dust view has no pyramid"))?
    } else {
      (
        view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?,
        core::ptr::null_mut(),
      )
    };
    // fold the readback of DUST_READBACK_SLOTS frames ago
    let (lvl, lvl_off, lw, lh) = DustView::white_level(&view.layout);
    let rb_words = DustView::readback_words(&view.layout) as usize;
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    if host.slot_valid[slot] && !cpu {
      if let Some(rb) = view.readback.as_ref() {
        res.allocator.allocator.as_allocator_view().invalidate_allocation(
          &rb.alloc,
          (slot * rb_words * 4) as u64,
          (rb_words * 4) as u64,
        )?;
        let words = unsafe {
          core::slice::from_raw_parts(
            view.readback_mapped.add(slot * rb_words * 4).cast::<u32>(),
            rb_words,
          )
        };
        let unit = host.slot_unit[slot];
        let (header, level) = words.split_at(dust::PYRAMID_HEADER_WORDS as usize);
        host.absorb(header, level, 1 << lvl, unit);
      }
    }
    host.slot_valid[slot] = false;
    let unit = host.unit();
    host.slot_unit[slot] = unit;
    let (white, softening) = host.display();
    drop(host);
    let _ = (lvl_off, lw, lh);
    // the header: level table + this frame's view words
    let mut header = view.layout.header();
    header[dust::PYR_FLAGS as usize] = flags;
    header[dust::PYR_T_HI as usize] = flow.t_hi.to_bits();
    header[dust::PYR_T_LO as usize] = flow.t_lo.to_bits();
    // tracer dot: a fixed level of the white point, in counts (area 1 px²)
    let tracer_counts = if white > 0.0 {
      (dust::TRACER_LEVEL * white / unit).min(dust::DUST_COUNT_MAX_PER_ADD as f32)
    } else {
      0.0
    };
    header[dust::PYR_TRACER_COUNTS as usize] = tracer_counts.to_bits();
    // the unit this frame's counts are written with, for dump readers (`download_dust_pyramid`)
    header[dust::PYR_UNIT as usize] = unit.to_bits();
    *view.frame_words.lock() = header;
    if cpu {
      // host-written pyramid: zero it and write the header now; the splats follow on the host
      let words = unsafe {
        core::slice::from_raw_parts_mut(
          pyramid_mapped.cast::<u32>(),
          view.layout.total_words as usize,
        )
      };
      words.fill(0);
      words[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&header);
    } else {
      unsafe {
        self
          .device
          .cmd_update_buffer(cmd, pyramid.buffer, 0, bytemuck::cast_slice(&header));
        self.device.cmd_fill_buffer(
          cmd,
          pyramid.buffer,
          dust::PYRAMID_HEADER_WORDS as u64 * 4,
          (view.layout.total_words - dust::PYRAMID_HEADER_WORDS) as u64 * 4,
          0,
        );
      }
    }
    Ok(DustViewParams {
      pyramid: pyramid.address,
      levels: view.layout.level_count(),
      unit,
      white,
      softening,
    })
  }

  /// Records (GPU) or performs (CPU mode) the splat pass of one tier of system `id` into the
  /// pyramid of `pe` (see `dust_splat.comp`): `render` / `moments` are the tier's buffers from
  /// [`Self::cmd_dust_propagate`], `pc` the push constants without the buffer addresses, which
  /// are filled here. After [`Self::cmd_dust_pre_splat_barrier`].
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_splat(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    render: u64,
    moments: u64,
    mut pc: DustSplatPushConstants,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
    let frame_no = view.host.lock().frame;
    let (pyramid, pyramid_mapped) = if sys.cpu_ring.is_some() {
      view.cpu_pyramid(frame_no).ok_or(gpu_err!("dust view has no pyramid"))?
    } else {
      (
        view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?,
        core::ptr::null_mut(),
      )
    };
    pc.render = render;
    pc.moments = moments;
    pc.pyramid = pyramid.address;
    if sys.cpu_ring.is_some() {
      // CPU mode: the reference splat on the clusters propagate evaluated, straight into the
      // host-visible pyramid of this frame
      let cpu = sys.cpu_render.lock();
      let Some((clusters, moms)) = cpu.get(&ring_base) else {
        return Ok(());
      };
      let flow = sys.host.lock().flow;
      let tracer_counts = f32::from_bits(view.frame_words.lock()[dust::PYR_TRACER_COUNTS as usize]);
      let words = unsafe {
        core::slice::from_raw_parts_mut(
          pyramid_mapped.cast::<u32>(),
          view.layout.total_words as usize,
        )
      };
      pc.live_count = pc.live_count.min(clusters.len() as u32);
      dust::splat_tier(
        moms,
        clusters,
        &pc,
        &flow,
        (tracer_counts + 0.5) as u32,
        &view.layout,
        words,
      );
      res.allocator.allocator.as_allocator_view().flush_allocation(
        &pyramid.alloc,
        0,
        view.layout.total_words as u64 * 4,
      )?;
      return Ok(());
    }
    if pc.live_count == 0 {
      return Ok(());
    }
    let pipeline = self.kernels.pipelines.dust_splat;
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
      self.device.cmd_dispatch(cmd, pc.live_count.div_ceil(DUST_WG), 1, 1);
    }
    Ok(())
  }

  /// Test only (fault injection): the splat dispatch of one tier with an explicit pyramid
  /// address, bypassing the view lookup.
  #[cfg(test)]
  pub fn cmd_dust_splat_with_pyramid(
    &self,
    cmd: vk::CommandBuffer,
    _id: u64,
    render: u64,
    moments: u64,
    mut pc: DustSplatPushConstants,
    pyramid: u64,
  ) -> GpuResult<()> {
    pc.render = render;
    pc.moments = moments;
    pc.pyramid = pyramid;
    let pipeline = self.kernels.pipelines.dust_splat;
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, pc.live_count.div_ceil(DUST_WG), 1, 1);
    }
    Ok(())
  }

  /// Ends the dust frame of `pe` (after [`Self::cmd_dust_post_splat_barrier`]): copies the header
  /// and the white-point level into this frame's readback slot. CPU mode folds the measurement
  /// in at once.
  pub fn cmd_dust_frame_end(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
    let (Some(pyramid), Some(rb)) = (view.pyramid.as_ref(), view.readback.as_ref()) else {
      return Ok(());
    };
    let (lvl, lvl_off, lw, lh) = DustView::white_level(&view.layout);
    let rb_words = DustView::readback_words(&view.layout) as u64;
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      let (_, mapped) = view.cpu_pyramid(host.frame).ok_or(gpu_err!("dust view has no pyramid"))?;
      let words = unsafe {
        core::slice::from_raw_parts_mut(mapped.cast::<u32>(), view.layout.total_words as usize)
      };
      // the measurement grid from the finished pyramid (every tier is splatted by now)
      dust::measure_from_pyramid(&view.layout, words);
      let unit = host.slot_unit[slot];
      let n = (lw * lh * dust::PYRAMID_TEXEL_WORDS) as usize;
      host.absorb(
        &words[..dust::PYRAMID_HEADER_WORDS as usize],
        &words[lvl_off as usize..lvl_off as usize + n],
        1 << lvl,
        unit,
      );
      host.frame += 1;
      return Ok(());
    }
    host.slot_valid[slot] = true;
    host.frame += 1;
    drop(host);
    // the measurement grid from the finished pyramid (`dust_measure.comp`): the splats' writes →
    // this pass, then its writes → the readback copy
    let (_, mw, mh) = view.layout.measure;
    let pipeline = self.kernels.pipelines.dust_measure;
    let mpc = dust::DustMeasurePushConstants {
      pyramid: pyramid.address,
      pad: [0; 2],
    };
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&mpc));
    let to_compute = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE);
    let to_copy = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_compute)),
      );
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&mpc),
      );
      self.device.cmd_dispatch(cmd, (mw * mh).div_ceil(DUST_WG), 1, 1);
      self.device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_copy)),
      );
    }
    let regions = [
      vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(slot as u64 * rb_words * 4)
        .size(dust::PYRAMID_HEADER_WORDS as u64 * 4),
      vk::BufferCopy::default()
        .src_offset(lvl_off as u64 * 4)
        .dst_offset(slot as u64 * rb_words * 4 + dust::PYRAMID_HEADER_WORDS as u64 * 4)
        .size((lw * lh * dust::PYRAMID_TEXEL_WORDS) as u64 * 4),
    ];
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::HOST)
      .dst_access_mask(vk::AccessFlags2::HOST_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe {
      self.device.cmd_copy_buffer(cmd, pyramid.buffer, rb.buffer, &regions);
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
    }
    Ok(())
  }

  /// The view's current display parameters (`None` without a view).
  pub fn dust_view_params(&self, pe: gpu::PresentationEngineHandle) -> Option<DustViewParams> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let view = mgr.views.get(&pe)?;
    let host = view.host.lock();
    let (white, softening) = host.display();
    Some(DustViewParams {
      pyramid: view.pyramid.as_ref().map(|b| b.address).unwrap_or(0),
      levels: view.layout.level_count(),
      unit: host.unit(),
      white,
      softening,
    })
  }

  /// Diagnostic: the pyramid layout of `pe`'s view.
  pub fn dust_view_layout(&self, pe: gpu::PresentationEngineHandle) -> Option<PyramidLayout> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    mgr.views.get(&pe).map(|v| v.layout.clone())
  }

  /// Diagnostic / observer: downloads the whole pyramid of `pe`'s view as written by the last
  /// completed frame (waits for the graphics queue). The first [`dust::PYRAMID_HEADER_WORDS`]
  /// words are the header. Expensive: not for every frame.
  pub fn download_dust_pyramid(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<alloc::vec::Vec<u32>> {
    let (buffer, words, mapped_cpu) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      let last = view.host.lock().frame.wrapping_sub(1);
      let cpu_mapped = view.cpu_pyramid(last).map(|(_, m)| m).filter(|m| !m.is_null());
      (pyramid.buffer, view.layout.total_words, cpu_mapped)
    };
    if let Some(ptr) = mapped_cpu {
      // CPU mode: the last completed frame's host pyramid (frames are in flight: wait for the
      // graphics queue through a serialized empty submission, never `vkQueueWaitIdle`, which the
      // render thread's submissions would race)
      let (handle, _) = self.get_command_buffer_and_native()?;
      self.begin_command_buffer_all(handle, QueueRole::Graphics)?;
      let (sem, value) =
        self.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Graphics)?;
      self.device.wait_for_semaphore_value(sem, value, 5_000_000_000)?;
      let v = unsafe { core::slice::from_raw_parts(ptr.cast::<u32>(), words as usize) }.to_vec();
      return Ok(v);
    }
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    let (staging, mapped) = self.dust_create_buffer(
      &allocator,
      words as u64 * 4,
      DustMem::HostRead,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      "DustPyramidDownload",
    )?;
    let out: GpuResult<alloc::vec::Vec<u32>> = (|| {
      let (handle, cmd) = self.get_command_buffer_and_native()?;
      self.begin_command_buffer_all(handle, QueueRole::Graphics)?;
      let region = vk::BufferCopy::default().size(words as u64 * 4);
      // in-order on the graphics queue after the last frame's splats and composite
      let pre = vk::MemoryBarrier2::default()
        .src_stage_mask(
          vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        )
        .src_access_mask(
          vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
        )
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
      let b = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ);
      unsafe {
        let dep0 = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&pre));
        self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep0);
        self
          .device
          .cmd_copy_buffer(cmd, buffer, staging.buffer, core::slice::from_ref(&region));
        let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
        self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
      }
      let (sem, value) =
        self.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Graphics)?;
      self.device.wait_for_semaphore_value(sem, value, 5_000_000_000)?;
      allocator.invalidate_allocation(&staging.alloc, 0, words as u64 * 4)?;
      Ok(unsafe { core::slice::from_raw_parts(mapped.cast::<u32>(), words as usize) }.to_vec())
    })();
    let mut a = staging.alloc;
    unsafe { allocator.destroy_buffer(staging.buffer, &mut a) };
    out
  }

  /// Traces this frame of system `id` (`gpu::dust_trace`): between the splat passes' post barrier
  /// and [`Self::cmd_dust_system_end`]. `ring_bases[k]` is the sub-ring base of `meta.tiers[k]`.
  /// GPU: copies the traced render clusters into this frame's readback slot (the line is written
  /// when the slot is read back); CPU mode: writes the line now.
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
      // CPU mode: the evaluated clusters are still on the host
      let cpu = sys.cpu_render.lock();
      let clusters: alloc::vec::Vec<alloc::vec::Vec<DustRenderCluster>> = meta
        .tiers
        .iter()
        .zip(ring_bases)
        .map(|(t, base)| {
          let v = cpu.get(base).map(|(r, _)| r);
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
        DUST_READBACK_SLOTS as u64 * TRACE_SLOT_BYTES,
        DustMem::HostRead,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustTrace_{id}"),
      )?);
    }
    let (tb, _) = trace.as_ref().unwrap();
    let slot = (sys.host.lock().frame % DUST_READBACK_SLOTS as u64) as usize;
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

  /// Records (GPU) or performs (CPU mode) the per-frame evaluation of the live range
  /// `[first_slot, first_slot + live_count)` of the sub-ring `[ring_base, ring_base + capacity)`
  /// (one age tier) of system `id`. Must be outside a render pass.
  /// Returns the device addresses of the compact render and moments buffers (0 in CPU mode,
  /// where the clusters stay on the host for [`Self::cmd_dust_splat`]).
  pub fn cmd_dust_propagate(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
  ) -> GpuResult<(u64, u64)> {
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
      // CPU mode: evaluate on the host; cmd_dust_splat scatters the result
      let ring = ring.read();
      let mut rs = alloc::vec::Vec::with_capacity(live_count as usize);
      let mut ms = alloc::vec::Vec::with_capacity(live_count as usize);
      for i in 0..live_count {
        let slot = first_slot.wrapping_add(i) & mask;
        let (r, m) = dust::packet_moments(&ring[(ring_base + slot) as usize], slot, frame);
        rs.push(r);
        ms.push(m);
      }
      sys.cpu_render.lock().insert(ring_base, (rs, ms));
      return Ok((0, 0));
    }
    let (Some(clusters), Some(render), Some(moments)) = (
      sys.clusters.as_ref(),
      sys.render.as_ref(),
      sys.moments.as_ref(),
    ) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", id));
    };
    // each tier evaluates into its own part of the render / moments buffers
    let render_offset = ring_base as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    let moments_offset = ring_base as u64 * core::mem::size_of::<DustMoments>() as u64;
    if live_count > 0 {
      let skip_moments = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_DEBUG_SKIP")
        .is_some_and(|v| v.split(',').any(|p| p.trim() == "moments"));
      let pc = DustPropagatePushConstants {
        clusters: clusters.address + ring_base as u64 * core::mem::size_of::<DustCluster>() as u64,
        render: render.address + render_offset,
        moments: if skip_moments {
          0
        } else {
          moments.address + moments_offset
        },
        first_slot,
        live_count,
        ring_mask: mask,
        _pad0: 0,
        _pad1: [0; 2],
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
    Ok((
      render.address + render_offset,
      moments.address + moments_offset,
    ))
  }
}
