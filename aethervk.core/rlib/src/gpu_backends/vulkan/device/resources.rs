use core::ptr;
use core::hash::{Hash, Hasher};
use aethervk_oshal_rlib::hash::FnvHasher;
use ash::vk;
use alloc::{boxed::Box, collections::VecDeque, sync, vec::Vec};
use spirv_reflect::{ffi::SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS, types::ReflectShaderStageFlags};
use vk_mem::Alloc;

use crate::gpu::PipelineKeyable;
use crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo;
use crate::{
  gpu::{PipelineKey, frame::Frame},
  gpu_backends::vulkan::{
    device::{
      DeviceResource, DeviceResourceJanitor, FunctionalDeviceResource, VertexIn,
      descriptors::{self, DescriptorPools},
      shader_manager::Shader,
    },
    utils::NonZeroHandle,
  },
  types::{GpuError, GpuResult},
};

pub struct TimelineQueue<T> {
  items: VecDeque<(u64, T)>,
}

impl<T> TimelineQueue<T> {
  pub fn with_capacity(cap: usize) -> Self {
    Self {
      items: VecDeque::with_capacity(cap),
    }
  }

  pub fn push(&mut self, timeline: u64, item: T) {
    self.items.push_back((timeline, item));
  }

  pub fn drain_ready<F>(&mut self, current: u64, mut f: F)
  where
    F: FnMut(T),
  {
    while let Some((t, _)) = self.items.front() {
      if *t > current {
        break;
      }

      let (_, item) = self.items.pop_front().unwrap();
      f(item);
    }
  }
}

enum DiscardItem {
  Buffer(BufferDiscard),
  Image(ImageDiscard),
  ImageView(vk::ImageView),
  Pipeline(vk::Pipeline),
  DescriptorPool(vk::DescriptorPool, sync::Arc<descriptors::DescriptorPools>),
  RenderPass(vk::RenderPass),
  Framebuffer(vk::Framebuffer),
  // TODO other types of resources as needed
  /// Placeholder to use any cleanable resource. Slower than having a specialized type
  GenericHandle(Box<dyn DeviceResource>),
}

struct BufferDiscard {
  buffer: vk::Buffer,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}
struct ImageDiscard {
  image: vk::Image,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}

pub trait DiscardableResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64);
}

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub(super) struct DiscardPool {
  items: spin::Mutex<TimelineQueue<DiscardItem>>,
}

unsafe impl Sync for DiscardPool {}
unsafe impl Send for DiscardPool {}

impl DiscardPool {
  /// Safety: device and allocator should outlive Self
  pub unsafe fn new(cap: usize) -> Self {
    Self {
      items: spin::Mutex::new(TimelineQueue::with_capacity(cap)),
    }
  }

  pub fn discard_type_erased<T: DeviceResource + 'static>(&self, item: T, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::GenericHandle(Box::new(item)));
  }

  // TODO all other types of resources as needed
  pub fn discard_render_pass(&self, render_pass: vk::RenderPass, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::RenderPass(render_pass));
  }

  pub fn discard_framebuffer(&self, framebuffer: vk::Framebuffer, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::Framebuffer(framebuffer));
  }

  pub fn discard_buffer(
    &self,
    allocator: vk_mem::ffi::VmaAllocator,
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    timeline: u64,
  ) {
    let mut q = self.items.lock();
    q.push(
      timeline,
      DiscardItem::Buffer(BufferDiscard {
        buffer,
        alloc,
        allocator,
      }),
    );
  }

  pub fn discard_image(
    &self,
    allocator: vk_mem::ffi::VmaAllocator,
    image: vk::Image,
    alloc: vk_mem::Allocation,
    timeline: u64,
  ) {
    let mut q = self.items.lock();
    q.push(
      timeline,
      DiscardItem::Image(ImageDiscard {
        image,
        alloc,
        allocator,
      }),
    );
  }

  pub fn discard_image_view(&self, image_view: vk::ImageView, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::ImageView(image_view));
  }

  pub fn discard_descriptor_pool(
    &self,
    pool: vk::DescriptorPool,
    manager: sync::Arc<descriptors::DescriptorPools>,
    timeline: u64,
  ) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::DescriptorPool(pool, manager));
  }

  pub fn discard_pipeline(&self, pipeline: vk::Pipeline, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::Pipeline(pipeline));
  }

  pub fn destroy_discarded_resources_all(&self, device: &ash::Device) {
    self.destroy_discarded_resources_internal(device, u64::MAX);
  }

  /// safety: `sem` needs to be a valid timeline semaphore
  pub unsafe fn destroy_discarded_resources_timeline(
    &self,
    device: &ash::Device,
    sem: vk::Semaphore,
  ) -> ash::prelude::VkResult<()> {
    let timeline = unsafe { device.get_semaphore_counter_value(sem) }?;
    self.destroy_discarded_resources_internal(device, timeline);
    Ok(())
  }

  fn destroy_discarded_resources_internal(&self, device: &ash::Device, timeline: u64) {
    let mut items = self.items.lock();
    items.drain_ready(timeline, |item| match item {
      DiscardItem::Buffer(BufferDiscard {
        buffer,
        alloc,
        allocator,
      }) => unsafe {
        vk_mem::ffi::vmaDestroyBuffer(allocator, buffer, alloc.get_raw());
      },
      DiscardItem::Image(ImageDiscard {
        image,
        alloc,
        allocator,
      }) => unsafe {
        vk_mem::ffi::vmaDestroyImage(allocator, image, alloc.get_raw());
      },
      DiscardItem::Pipeline(pipeline) => {
        unsafe { device.destroy_pipeline(pipeline, None) };
      }
      DiscardItem::DescriptorPool(pool, manager) => {
        // return the pool to the manager for recycling
        manager.recycle(device, pool);
      }
      DiscardItem::ImageView(image_view) => unsafe {
        device.destroy_image_view(image_view, None);
      },
      DiscardItem::RenderPass(render_pass) => unsafe {
        device.destroy_render_pass(render_pass, None);
      },
      DiscardItem::Framebuffer(framebuffer) => unsafe {
        device.destroy_framebuffer(framebuffer, None);
      },
      DiscardItem::GenericHandle(mut handle) => {
        handle.cleanup(device);
      }
    });
  }
}

impl super::DeviceResource for DiscardPool {
  fn cleanup(&mut self, device: &ash::Device) {
    self.destroy_discarded_resources_all(device);
  }
}

/// Note: Caller should also provide its own timeline value
pub(super) trait DiscardPoolCaller {
  fn discard_buffer(&self, buffer: vk::Buffer, alloc: vk_mem::Allocation);
  fn discard_image(&self, image: vk::Image, alloc: vk_mem::Allocation);
  fn discard_image_view(&self, image_view: vk::ImageView);
  fn discard_pipeline(&self, pipeline: vk::Pipeline);
  fn discard_descriptor_pool(&self, pool: vk::DescriptorPool);
  fn destroy_discarded_resources(&self);
  fn destroy_discarded_resources_all(&self);
}

pub(super) struct Buffer {
  buffer: NonZeroHandle<vk::Buffer>,
  allocation: vk_mem::Allocation,
}

impl Hash for Buffer {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.buffer.hash(state);
  }
}

pub(super) struct Image {
  image: NonZeroHandle<vk::Image>,
  allocation: vk_mem::Allocation,
}

impl Hash for Image {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.image.hash(state);
  }
}

// TODO: move up to frame.rs
bitflags::bitflags! {
  pub(super) struct TextureFlags: u32 {
    const FlagAlbedo = 1u32 << 0;
    const FlagNormal = 1u32 << 1;
    const FlagRoughness = 1u32 << 2;
    const FlagAo  = 1u32 << 3;
  }
}

#[repr(C)]
pub(super) struct ForwardMeshRenderResourcePushData {
  model_view_projection: [f32; 16],
  model: [f32; 16],
  sun_dir: [f32; 3],
  texture_flags: TextureFlags,
  sun_color: [f32; 4],
}
sa::const_assert!(core::mem::size_of::<ForwardMeshRenderResourcePushData>() == 160);

impl Default for ForwardMeshRenderResourcePushData {
  fn default() -> Self {
    Self {
      model_view_projection: Default::default(),
      model: Default::default(),
      sun_dir: Default::default(),
      texture_flags: TextureFlags::empty(),
      sun_color: Default::default(),
    }
  }
}

pub(super) struct ForwardMeshRenderResource {
  allocator: vk_mem::ffi::VmaAllocator, // necessary evil. TODO: Edit DeviceResource trait and remove this.
  position_vertex_buffer: Buffer,
  attributes_vertex_buffer: Buffer,
  index_buffer: Buffer,
  /// Each frame, this is copied and then overwritten with [`crate::simulation::comet::PushConstants`]
  push_data: ForwardMeshRenderResourcePushData,
  /// layout(binding = 0) uniform sampler2D albedoMap;
  albedo_image: Option<Image>,
  /// layout(binding = 1) uniform sampler2D normalMap;
  normal_image: Option<Image>,
  /// layout(binding = 2) uniform sampler2D roughnessMap;
  roughness_image: Option<Image>,
  /// layout(binding = 3) uniform sampler2D aoMap;
  ao_image: Option<Image>,
}

unsafe impl Sync for ForwardMeshRenderResource {}
unsafe impl Send for ForwardMeshRenderResource {}

impl DiscardableResource for ForwardMeshRenderResource {
  fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_buffer(
      self.allocator,
      self.position_vertex_buffer.buffer.get(),
      self.position_vertex_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.attributes_vertex_buffer.buffer.get(),
      self.attributes_vertex_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.index_buffer.buffer.get(),
      self.index_buffer.allocation,
      timeline,
    );
    if let Some(albedo_image) = &self.albedo_image {
      discard_pool.discard_image(
        self.allocator,
        albedo_image.image.get(),
        albedo_image.allocation,
        timeline,
      );
    }
    if let Some(normal_image) = &self.normal_image {
      discard_pool.discard_image(
        self.allocator,
        normal_image.image.get(),
        normal_image.allocation,
        timeline,
      );
    }
    if let Some(roughness_image) = &self.roughness_image {
      discard_pool.discard_image(
        self.allocator,
        roughness_image.image.get(),
        roughness_image.allocation,
        timeline,
      );
    }
    if let Some(ao_image) = &self.ao_image {
      discard_pool.discard_image(
        self.allocator,
        ao_image.image.get(),
        ao_image.allocation,
        timeline,
      );
    }
  }
}

impl ForwardMeshRenderResource {
  pub fn buffers_hash(&self) -> u64 {
    let mut hasher = FnvHasher::new();
    self.position_vertex_buffer.hash(&mut hasher);
    self.attributes_vertex_buffer.hash(&mut hasher);
    self.index_buffer.hash(&mut hasher);
    hasher.finish()
  }

  #[allow(clippy::too_many_arguments)]
  pub(super) fn new(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    command_buffer: vk::CommandBuffer,
    discard_pool: &DiscardPool,
    timeline: u64,
    position_data: &[f32],
    attribute_data: &[f32],
    index_data: &[u32],
    albedo_image: Option<Image>, // Image creation is complex, pass them in for now
    normal_image: Option<Image>, //
    roughness_image: Option<Image>, //
    ao_image: Option<Image>,     //
  ) -> GpuResult<Self> {
    // Reusable helper function to perform the explicit staging buffer upload pattern.
    fn create_buffer_with_staging<T: Copy>(
      device: &ash::Device,
      allocator: &vk_mem::Allocator,
      command_buffer: vk::CommandBuffer,
      discard_pool: &DiscardPool,
      timeline: u64,
      data: &[T],
      usage: vk::BufferUsageFlags,
    ) -> GpuResult<Buffer> {
      let buffer_size = (core::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
      if buffer_size == 0 {
        return Err(GpuError::InvalidArgument);
      }

      let vma_allocator = allocator.get_raw();

      // 1. Create staging buffer (CPU-visible)
      let staging_buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
      let staging_alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Auto,
        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
          | vk_mem::AllocationCreateFlags::MAPPED,
        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
        ..Default::default()
      };
      let (staging_buffer, staging_allocation, staging_alloc_info) =
        unsafe { allocator.create_buffer_get_info(&staging_buffer_info, &staging_alloc_info) }?;

      // 2. Create device buffer (GPU-local). In case of failure, we clean up the staging buffer.
      let (device_buffer, device_allocation) = {
        let device_buffer_info = vk::BufferCreateInfo::default()
          .size(buffer_size)
          .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
        let device_alloc_info = vk_mem::AllocationCreateInfo {
          usage: vk_mem::MemoryUsage::Auto,
          flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
          preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
          ..Default::default()
        };
        match unsafe { allocator.create_buffer(&device_buffer_info, &device_alloc_info) } {
          Ok(result) => result,
          Err(err) => {
            unsafe {
              vk_mem::ffi::vmaDestroyBuffer(
                vma_allocator,
                staging_buffer,
                staging_allocation.get_raw(),
              );
            }
            return Err(err.into());
          }
        }
      };

      // 3. Copy data to staging buffer
      unsafe {
        core::ptr::copy_nonoverlapping(
          data.as_ptr(),
          staging_alloc_info.mapped_data as *mut T,
          data.len(),
        );
      }
      if !unsafe { allocator.get_allocation_memory_properties(&staging_allocation) }
        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
      {
        allocator.flush_allocation(&staging_allocation, 0, vk::WHOLE_SIZE)?;
      }

      // 4. Record copy command
      let copy_region = vk::BufferCopy::default().size(buffer_size);
      unsafe {
        device.cmd_copy_buffer(
          command_buffer,
          staging_buffer,
          device_buffer,
          &[copy_region],
        );
      }

      // 5. Insert a pipeline barrier to synchronize
      let (dst_stage, dst_access) = if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        (
          vk::PipelineStageFlags::VERTEX_INPUT,
          vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        )
      } else if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        (
          vk::PipelineStageFlags::VERTEX_INPUT,
          vk::AccessFlags::INDEX_READ,
        )
      } else {
        (
          vk::PipelineStageFlags::TOP_OF_PIPE,
          vk::AccessFlags::empty(),
        )
      };

      if dst_access != vk::AccessFlags::empty() {
        let buffer_barrier = vk::BufferMemoryBarrier::default()
          .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .dst_access_mask(dst_access)
          .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
          .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
          .buffer(device_buffer)
          .offset(0)
          .size(buffer_size);

        unsafe {
          device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier],
            &[],
          );
        }
      }

      // 6. Schedule staging buffer for destruction.
      discard_pool.discard_buffer(vma_allocator, staging_buffer, staging_allocation, timeline);

      Ok(Buffer {
        buffer: unsafe { NonZeroHandle::new_unchecked(device_buffer) },
        allocation: device_allocation,
      })
    }

    let mut janitor = DeviceResourceJanitor::<'_, 7>::new(device);
    let vma_allocator = allocator.get_raw();

    // Create position buffer
    let position_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      discard_pool,
      timeline,
      position_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let pos_alloc = position_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        position_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, pos_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create attributes buffer
    let attributes_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      discard_pool,
      timeline,
      attribute_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let attr_alloc = attributes_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        attributes_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, attr_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create index buffer
    let index_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      discard_pool,
      timeline,
      index_data,
      vk::BufferUsageFlags::INDEX_BUFFER,
    )?;
    let idx_alloc = index_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        index_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, idx_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // For images, we are still passing them in, but if they were created here,
    // they would also be pushed to the janitor.
    if let Some(image) = &albedo_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    // ... repeat for other optional images ...

    // Everything was created successfully. Defuse the janitor.
    janitor.clear();

    Ok(Self {
      allocator: vma_allocator,
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      push_data: Default::default(),
      albedo_image,
      normal_image,
      roughness_image,
      ao_image,
    })
  }
}

/// Structure which is built up per frame and then discarded on submission
/// It holds the vulkan-backend specific draw call data
/// Each frame end, all [`FrameResource`]s are discarded through the [`DiscardableResource`] trait
pub(super) enum FrameResource {
  ForwardMeshRenderResource(ForwardMeshRenderResource),
}

impl DiscardableResource for FrameResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    match self {
      Self::ForwardMeshRenderResource(resource) => {
        resource.discard(device, discard_pool, timeline);
      }
    }
  }
}

/// To be destroyed before descriptor pool
pub(super) struct ForwardMeshRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>>,
  pub descriptor_sets: Vec<NonZeroHandle<vk::DescriptorSet>>,
  pub push_contant_ranges: Vec<vk::PushConstantRange>,
  // 0 = vertex, 1 = fragment
  pub specialization_constants: [Vec<vk::SpecializationMapEntry>; 2],
  // 0 = vertex, 1 = fragment
  pub specialization_constants_values: [Vec<u8>; 2],
  /// Populated after with_graphics_info
  pub graphics_info: Option<GraphicsInfo>,
  /// Populated after with_pipeline_key
  pub pipeline_key: Option<PipelineKey>,
}

impl ForwardMeshRenderResourceArchetype {
  pub fn with_graphics_info(self, graphics_info: GraphicsInfo) -> Self {
    let pipeline_key = graphics_info.pipeline_key();
    Self {
      graphics_info: Some(graphics_info),
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }

  /// Safety:
  /// - `pipeline_key` must refer to a pipeline created with `vertex_shader` and `fragment_shader`,
  pub unsafe fn new(
    descriptor_pools: &sync::Arc<DescriptorPools>,
    device: &ash::Device,
    discard_pool: &DiscardPool,
    vertex_shader: &Shader,
    fragment_shader: &Shader,
  ) -> GpuResult<Self> {
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;
    let mut janitor = DeviceResourceJanitor::<'_, 16>::new(device);

    if !vertex_shader
      .spv_module
      .get_shader_stage()
      .contains(ReflectShaderStageFlags::VERTEX)
      || !fragment_shader
        .spv_module
        .get_shader_stage()
        .contains(ReflectShaderStageFlags::FRAGMENT)
    {
      return Err(GpuError::InvalidShader);
    }
    // --------------------------- 1. Descriptor Sets -------------------------------------------
    // Helper to map descriptor types from spirv-reflect to ash and handle unsupported cases.
    let map_descriptor_type =
      |reflect_type: spirv_reflect::types::ReflectDescriptorType| -> GpuResult<vk::DescriptorType> {
        use spirv_reflect::types::ReflectDescriptorType;
        Ok(match reflect_type {
          ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
          ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
          ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
          ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
          ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
          ReflectDescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
          ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
          ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
          ReflectDescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
          ReflectDescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
          ReflectDescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
          ReflectDescriptorType::AccelerationStructureKHR => {
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
          }
          _ => {
            return Err(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
              "Unsupported descriptor type: {:?}",
              reflect_type
            ))));
          }
        })
      };

    // This will hold the merged layout information.
    // Map<set_number, Map<binding_number, vk::DescriptorSetLayoutBinding>>
    let mut merged_sets: hashbrown::HashMap<
      u32,
      hashbrown::HashMap<u32, vk::DescriptorSetLayoutBinding>,
    > = hashbrown::HashMap::new();

    for shader in [vertex_shader, fragment_shader] {
      let shader_stage = shader.shader_stage;
      let sets = shader
        .spv_module
        .enumerate_descriptor_sets(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for set in sets {
        let bindings_map = merged_sets.entry(set.set).or_default();

        for binding in &set.bindings {
          let reflect_binding = binding;
          let new_descriptor_type = map_descriptor_type(reflect_binding.descriptor_type)?;

          if let Some(existing_binding) = bindings_map.get_mut(&reflect_binding.binding) {
            // Binding already exists in another shader stage, check for conflicts.
            if existing_binding.descriptor_type != new_descriptor_type
              || existing_binding.descriptor_count != reflect_binding.count
            {
              return Err(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
                "Descriptor set binding conflict at (set={}, binding={}). Mismatch in descriptor type or count across shader stages.",
                set.set, reflect_binding.binding
              ))));
            }

            // No conflict, so merge the stage flags.
            existing_binding.stage_flags |= shader_stage;
          } else {
            // First time seeing this binding, create a new one.
            let new_binding = vk::DescriptorSetLayoutBinding::default()
              .binding(reflect_binding.binding)
              .descriptor_type(new_descriptor_type)
              .descriptor_count(reflect_binding.count)
              .stage_flags(shader_stage);
            bindings_map.insert(reflect_binding.binding, new_binding);
          }
        }
      }
    }

    // Convert the map of maps into the final structure needed for layout creation.
    // Map<set_number, Vec<vk::DescriptorSetLayoutBinding>>
    let set_layouts: hashbrown::HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = merged_sets
      .into_iter()
      .map(|(set_number, bindings_map)| {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
          bindings_map.into_values().collect();
        // Sort by binding number for consistency.
        bindings.sort_by_key(|b| b.binding);
        (set_number, bindings)
      })
      .collect();
    // Sort by set number to ensure the final layouts have a deterministic order.
    let mut sorted_layouts: Vec<_> = set_layouts.into_iter().collect();
    sorted_layouts.sort_by_key(|(set, _)| *set);

    let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = sorted_layouts
      .into_iter()
      .map(|(_, bindings)| {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
        janitor
          .push(FunctionalDeviceResource::new(layout, |h, d| unsafe {
            d.destroy_descriptor_set_layout(h, None)
          }))
          .map_err(|_| GpuError::InvalidState)?;
        Ok(layout)
      })
      .collect::<GpuResult<Vec<_>>>()?;

    // no need for janitor as pool will be destroyed when discard pool is destroyed
    let descriptor_sets = descriptor_set_layouts
      .iter()
      .map(|layout| {
        descriptor_pools.allocate(device, *layout, discard_pool, NEVER_DISCARD_TIMELINE)
      })
      .collect::<GpuResult<Vec<_>>>()?;

    // --------------------------- 2. Push Constants --------------------------------------------
    let mut push_constant_ranges = Vec::<vk::PushConstantRange>::new();
    for shader in [vertex_shader, fragment_shader] {
      let blocks = shader
        .spv_module
        .enumerate_push_constant_blocks(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for block in blocks {
        // Find a range with the same offset and size to merge stage flags.
        if let Some(range) = push_constant_ranges
          .iter_mut()
          .find(|r| r.offset == block.offset && r.size == block.size)
        {
          // Merge shader stages into the existing range.
          range.stage_flags |= shader.shader_stage;
        } else {
          // Add a new range for this push constant block.
          push_constant_ranges.push(
            vk::PushConstantRange::default()
              .stage_flags(shader.shader_stage)
              .offset(block.offset)
              .size(block.size),
          );
        }
      }
    }

    // --------------------------- 3. Pipeline Layout -------------------------------------------
    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(&descriptor_set_layouts)
      .push_constant_ranges(&push_constant_ranges);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }?;
    janitor
      .push(FunctionalDeviceResource::new(
        pipeline_layout,
        |h, d| unsafe { d.destroy_pipeline_layout(h, None) },
      ))
      .map_err(|_| GpuError::InvalidState)?;

    // --------------------------- 4. Specialization Infos --------------------------------------
    let mut specialization_constants = [Vec::new(), Vec::new()];
    let mut specialization_constants_values = [Vec::new(), Vec::new()];

    for (i, shader) in [vertex_shader, fragment_shader].iter().enumerate() {
      // NOTE: `spirv-reflect` does not provide the size of specialization constants
      // directly. We are assuming here that all specialization constants are 32-bit (4-byte)
      // values like int, float, or bool. This may need adjustment if other types are used.
      const ASSUMED_SPEC_CONST_SIZE: usize = 4;
      let spv_specialization_constants = unsafe {
        let mut count: u32 = 0;
        let mut res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          ptr::null_mut(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec = Vec::new();
        the_vec.resize(
          count as usize,
          ptr::null_mut::<spirv_reflect::ffi::SpvReflectSpecializationConstant>(),
        );
        res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          the_vec.as_mut_ptr(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec: Vec<_> = the_vec
          .iter()
          .map(|c| c.as_ref().unwrap_unchecked())
          .collect();
        the_vec.sort_by_key(|&c| c.constant_id);
        Ok::<Vec<_>, GpuError>(the_vec)
      }?;
      specialization_constants[i].reserve(spv_specialization_constants.len());

      for spec_const in spv_specialization_constants {
        let offset = specialization_constants_values[i].len() as u32;

        specialization_constants[i].push(
          vk::SpecializationMapEntry::default()
            .constant_id(spec_const.constant_id)
            .offset(offset)
            .size(ASSUMED_SPEC_CONST_SIZE),
        );

        // The reflection does not provide the default value from the shader. We populate
        // the data blob with a default value based on its name.
        let name = unsafe {
          if spec_const.name.is_null() {
            ""
          } else {
            core::ffi::CStr::from_ptr(spec_const.name)
              .to_str()
              .unwrap_or("")
          }
        };

        let default_value_bytes = match name {
          "BASE_ALBEDO_R" => 0.04f32.to_ne_bytes(),
          "BASE_ALBEDO_G" => 0.04f32.to_ne_bytes(),
          "BASE_ALBEDO_B" => 0.04f32.to_ne_bytes(),
          "BASE_ROUGHNESS" => 0.9f32.to_ne_bytes(),
          "BASE_AO" => 1.0f32.to_ne_bytes(),
          _ => [0u8; ASSUMED_SPEC_CONST_SIZE],
        };
        specialization_constants_values[i].extend_from_slice(&default_value_bytes);
      }
    }

    janitor.clear();
    Ok(Self {
      pipeline_layout: NonZeroHandle::new(pipeline_layout).unwrap(),
      descriptor_set_layouts: descriptor_set_layouts
        .into_iter()
        .map(|l| NonZeroHandle::new(l).unwrap())
        .collect(),
      descriptor_sets,
      push_contant_ranges: push_constant_ranges,
      specialization_constants,
      specialization_constants_values,
      pipeline_key: None,
      graphics_info: None,
    })
  }
}

impl DiscardableResource for ForwardMeshRenderResourceArchetype {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(
        self.pipeline_layout.get(),
        |pipeline_layout, device| unsafe {
          device.destroy_pipeline_layout(pipeline_layout, None);
        },
      ),
      timeline,
    );
    for layout in &self.descriptor_set_layouts {
      unsafe {
        device.destroy_descriptor_set_layout(layout.get(), None);
      }
    }
  }
}

/// Structure which holds vulkan resources which are common to all frame instances of a given
/// render resource type
/// These are destroyed when the [`super::Device`] instance is dropped, ie when the [`DiscardPool`]
/// is dropped, through the [`DiscardableResource`] trait
pub(super) enum FrameResourceArchetype {
  ForwardMeshRenderResource(ForwardMeshRenderResourceArchetype),
}

impl DiscardableResource for FrameResourceArchetype {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    match self {
      FrameResourceArchetype::ForwardMeshRenderResource(forward_mesh_render_resource_archetype) => {
        forward_mesh_render_resource_archetype.discard(device, discard_pool, timeline);
      }
    }
  }
}
