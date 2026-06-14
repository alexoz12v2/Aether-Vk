//! renderpasses module.

#[cfg(test)]
use crate::gpu_backends::vulkan::utils::create_test_attachment;
use crate::{
  gpu::{PresentationEngineHandle, vulkan::device::swapchain},
  gpu_backends::vulkan::{
    device::{DeviceResource, resources::DiscardPool, swapchain::PresentationState},
    utils::{NonZeroHandle, create_transient_attachment},
  },
  types::GpuResult,
};
use ash::vk;
use core::slice;
use function_name::named;

enum RenderPassAttachment {
  SwapchainColorImage,
  DepthStencilAttachment(
    NonZeroHandle<vk::Image>,
    vk_mem::Allocation,
    NonZeroHandle<vk::ImageView>,
  ),
  /// Transient color attachment (used for compositing intermediate buffers).
  ColorAttachment(
    NonZeroHandle<vk::Image>,
    vk_mem::Allocation,
    NonZeroHandle<vk::ImageView>,
  ),
}

pub(super) const MAX_ATTACHMENTS: usize = 8;
const VK_SUBPASS_EXTERNAL: u32 = 0xFFFFFFFF;

struct RenderPassBundle {
  render_pass: NonZeroHandle<vk::RenderPass>,
  /// Clear values, one per attachment.
  clear_value: heapless::Vec<vk::ClearValue, MAX_ATTACHMENTS>,
  // keep track of swapchain recreation
  swapchain_generation: u64,
  // VkFramebufferCreateInfo
  /// 1-1 correspondance with swapchain_image
  framebuffer: heapless::Vec<NonZeroHandle<vk::Framebuffer>, { swapchain::MAX_FRAMES }>,
  width: u32,
  height: u32,
  /// attachments handle: Note that they are 1 per graphics queue, which is just one per device in our setup
  attachments: heapless::Vec<RenderPassAttachment, MAX_ATTACHMENTS>,
  /// Depth-only image views for compositing input attachment descriptors.
  /// Vulkan requires input attachment descriptors for D32S8 images to use views
  /// with only DEPTH aspect, not DEPTH|STENCIL. These are created for attachments
  /// [3] (macroDepth) and [5] (microDepth) when compositing is active.
  /// Index 0 = macroDepth depth-only view, Index 1 = microDepth depth-only view.
  depth_only_views: heapless::Vec<NonZeroHandle<vk::ImageView>, 2>,
  /// Compositing resources — only populated when using ColorDepthCompositing.
  composite: Option<CompositeResources>,
}

/// Resources needed for the depth-compositing fullscreen pass (subpass 2).
/// Created alongside the compositing RenderPassBundle.
struct CompositeResources {
  descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  descriptor_set: vk::DescriptorSet,
  pipeline_key: crate::gpu::PipelineKey,
}

/// TODO: Document this item
#[derive(Clone)]
pub(super) enum RenderPassSpecification {
  ColorDepthSingleSubpass {
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_layout: vk::ImageLayout,
    extent: (u32, u32),
    swapchain_generation: u64,
    image_views: heapless::Vec<NonZeroHandle<vk::ImageView>, { swapchain::MAX_FRAMES }>,
  },
  /// 3-subpass compositing render pass: macro → micro → composite.
  ///
  /// Subpass 0 (macro):     renders to transient macroColor/macroDepth
  /// Subpass 1 (micro):     renders to transient microColor/microDepth
  /// Subpass 2 (composite): reads all 4 transient images as input attachments,
  ///                        outputs to swapchain color + depth
  ColorDepthCompositing {
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_layout: vk::ImageLayout,
    extent: (u32, u32),
    swapchain_generation: u64,
    image_views: heapless::Vec<NonZeroHandle<vk::ImageView>, { swapchain::MAX_FRAMES }>,
  },
}

impl RenderPassSpecification {
  /// TODO: Document this item
  pub fn single_pass(presentation_engine: &PresentationState, d: vk::Format) -> Self {
    let final_layout = match presentation_engine {
      PresentationState::Windowed(_) => vk::ImageLayout::PRESENT_SRC_KHR,
      PresentationState::Windowless(_) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let mut image_views = heapless::Vec::new();
    presentation_engine
      .for_each_swapchain_image(|image_view| {
        unsafe { image_views.push_unchecked(image_view) };
        Ok(())
      })
      .unwrap();

    Self::ColorDepthSingleSubpass {
      color_format: presentation_engine.format(),
      depth_stencil_format: d,
      final_layout,
      extent: presentation_engine.extent(),
      swapchain_generation: presentation_engine.swapchain_generation(),
      image_views,
    }
  }

  /// Construct a 3-subpass compositing render pass specification.
  pub fn compositing_pass(presentation_engine: &PresentationState, d: vk::Format) -> Self {
    let final_layout = match presentation_engine {
      PresentationState::Windowed(_) => vk::ImageLayout::PRESENT_SRC_KHR,
      PresentationState::Windowless(_) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let mut image_views = heapless::Vec::new();
    presentation_engine
      .for_each_swapchain_image(|image_view| {
        unsafe { image_views.push_unchecked(image_view) };
        Ok(())
      })
      .unwrap();

    Self::ColorDepthCompositing {
      color_format: presentation_engine.format(),
      depth_stencil_format: d,
      final_layout,
      extent: presentation_engine.extent(),
      swapchain_generation: presentation_engine.swapchain_generation(),
      image_views,
    }
  }

  /// Extract common fields regardless of variant.
  fn fields(
    &self,
  ) -> (
    vk::Format,
    vk::Format,
    vk::ImageLayout,
    (u32, u32),
    u64,
    &heapless::Vec<NonZeroHandle<vk::ImageView>, { swapchain::MAX_FRAMES }>,
  ) {
    match self {
      Self::ColorDepthSingleSubpass {
        color_format,
        depth_stencil_format,
        final_layout,
        extent,
        swapchain_generation,
        image_views,
      }
      | Self::ColorDepthCompositing {
        color_format,
        depth_stencil_format,
        final_layout,
        extent,
        swapchain_generation,
        image_views,
      } => (
        *color_format,
        *depth_stencil_format,
        *final_layout,
        *extent,
        *swapchain_generation,
        image_views,
      ),
    }
  }

  /// Returns the render area extent.
  pub fn extent(&self) -> (u32, u32) {
    match self {
      Self::ColorDepthSingleSubpass { extent, .. } | Self::ColorDepthCompositing { extent, .. } => {
        *extent
      }
    }
  }

  /// Returns the number of attachments (and clear values) for this specification.
  pub fn num_attachments(&self) -> usize {
    match self {
      Self::ColorDepthSingleSubpass { .. } => 2,
      Self::ColorDepthCompositing { .. } => 6,
    }
  }
}

/// TODO: Document this item
pub(super) struct RenderPasses {
  render_passes: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    hashbrown::HashMap<PresentationEngineHandle, RenderPassBundle>,
  >,
  pipeline_render_passes: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    hashbrown::HashMap<(vk::Format, vk::Format), NonZeroHandle<vk::RenderPass>>,
  >,
  render_pass_device: ash::khr::create_renderpass2::Device,
  // this is bad but I've got no other clue
  allocator: vk_mem::ffi::VmaAllocator,
}

impl RenderPassBundle {
  fn discard(
    &mut self,
    discard_pool: &DiscardPool,
    allocator: vk_mem::ffi::VmaAllocator,
    timeline: u64,
  ) {
    for attachment in self.attachments.iter() {
      match attachment {
        RenderPassAttachment::SwapchainColorImage => {}
        RenderPassAttachment::DepthStencilAttachment(image, allocation, image_view)
        | RenderPassAttachment::ColorAttachment(image, allocation, image_view) => {
          discard_pool.discard_image_view(image_view.get(), timeline);
          discard_pool.discard_image(allocator, image.get(), *allocation, timeline);
        }
      }
    }

    // Discard depth-only views for compositing
    for view in self.depth_only_views.iter() {
      discard_pool.discard_image_view(view.get(), timeline);
    }

    for framebuffer in self.framebuffer.iter() {
      discard_pool.discard_framebuffer(framebuffer.get(), timeline);
    }

    discard_pool.discard_render_pass(self.render_pass.get(), timeline);

    // Clean up compositing pipeline resources if present
    if let Some(ref comp) = self.composite {
      discard_pool.discard_descriptor_set_layout(comp.descriptor_set_layout.get(), timeline);
      discard_pool.discard_pipeline_layout(comp.pipeline_layout.get(), timeline);
      struct CompositeDescriptorPoolDiscard(vk::DescriptorPool);
      impl super::DeviceResource for CompositeDescriptorPoolDiscard {
        fn cleanup(&mut self, device: &ash::Device) {
          unsafe { device.destroy_descriptor_pool(self.0, None) };
        }
      }
      discard_pool.discard_type_erased(
        CompositeDescriptorPoolDiscard(comp.descriptor_pool.get()),
        timeline,
      );
    }
  }

  fn clean(&mut self, device: &ash::Device, allocator: vk_mem::ffi::VmaAllocator) {
    unsafe { device.destroy_render_pass(self.render_pass.get(), None) };

    for framebuffer in self.framebuffer.iter() {
      unsafe { device.destroy_framebuffer(framebuffer.get(), None) };
    }
    self.framebuffer.clear();

    for attachment in self.attachments.iter() {
      match attachment {
        RenderPassAttachment::DepthStencilAttachment(image, allocation, view)
        | RenderPassAttachment::ColorAttachment(image, allocation, view) => {
          unsafe { device.destroy_image_view(view.get(), None) };
          unsafe { vk_mem::ffi::vmaDestroyImage(allocator, image.get(), allocation.get_raw()) };
        }
        RenderPassAttachment::SwapchainColorImage => {}
      }
    }

    // Clean up depth-only views for compositing
    for view in self.depth_only_views.iter() {
      unsafe { device.destroy_image_view(view.get(), None) };
    }

    // Clean up compositing pipeline resources if present
    if let Some(ref comp) = self.composite {
      unsafe {
        device.destroy_descriptor_set_layout(comp.descriptor_set_layout.get(), None);
        device.destroy_pipeline_layout(comp.pipeline_layout.get(), None);
        device.destroy_descriptor_pool(comp.descriptor_pool.get(), None);
      }
    }
  }
}

impl DeviceResource for RenderPasses {
  fn cleanup(&mut self, device: &ash::Device) {
    for (_, mut bundle) in
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&self.render_passes)
        .drain()
    {
      bundle.clean(&device, self.allocator);
    }
    for (_, rp) in crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
      &self.pipeline_render_passes,
    )
    .drain()
    {
      unsafe { device.destroy_render_pass(rp.get(), None) };
    }
  }
}

unsafe impl Sync for RenderPasses {}
unsafe impl Send for RenderPasses {}

/// Thin abstraction over render pass creation and management.
/// Note: It Implicitly requires to not outlive the VmaAllocator
impl RenderPasses {
  /// TODO: Document this item
  pub fn new(
    instance: &ash::Instance,
    device: &ash::Device,
    allocator: vk_mem::AllocatorView,
  ) -> Self {
    Self {
      render_passes: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(
        hashbrown::HashMap::with_capacity(8),
      ),
      pipeline_render_passes: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(
        hashbrown::HashMap::with_capacity(8),
      ),
      render_pass_device: ash::khr::create_renderpass2::Device::new(instance, device),
      allocator: allocator.get_raw(),
    }
  }

  /// Returns or creates a render pass suitable for pipeline creation.
  ///
  /// Creates a single-subpass render pass for pipeline compatibility.
  /// When compositing mode is used at render time, the render_frame code
  /// is responsible for rebinding pipelines compatible with the compositing
  /// render pass.
  #[named]
  pub fn get_pipeline_render_pass(
    &self,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
  ) -> GpuResult<NonZeroHandle<vk::RenderPass>> {
    let key = (color_format, depth_stencil_format);
    if let Some(&rp) = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
      &self.pipeline_render_passes,
    )
    .get(&key)
    {
      return Ok(rp);
    }

    let rp = Self::create_color_depth_single_render_pass(
      &self.render_pass_device,
      color_format,
      depth_stencil_format,
      vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    )?;

    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
      &self.pipeline_render_passes,
    )
    .insert(key, rp);
    Ok(rp)
  }

  /// Returns the number of clear values copied into `out_values`.
  #[named]
  pub fn get_clear_values_render_pass(
    &self,
    pe_handle: PresentationEngineHandle,
    out_values: &mut [vk::ClearValue],
  ) -> GpuResult<usize> {
    let read_render_passes =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes);
    if !read_render_passes.contains_key(&pe_handle) {
      return Err(crate::gpu_err_device!());
    }
    let bundle = unsafe { read_render_passes.get(&pe_handle).unwrap_unchecked() };
    let count = out_values.len().min(bundle.clear_value.len());
    out_values[..count].copy_from_slice(&bundle.clear_value[..count]);
    Ok(count)
  }

  /// TODO: Document this item
  #[named]
  pub fn get_or_create_render_pass(
    &self,
    pe_handle: PresentationEngineHandle,
    ty: RenderPassSpecification,
    image_index: u32,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    discard_pool: &DiscardPool,
    timeline: u64,
  ) -> GpuResult<(
    NonZeroHandle<vk::RenderPass>,
    NonZeroHandle<vk::Framebuffer>,
  )> {
    let (
      color_format,
      depth_stencil_format,
      final_layout,
      extent,
      swapchain_generation,
      image_views,
    ) = ty.fields();
    let image_views = image_views.clone();
    let is_compositing = matches!(ty, RenderPassSpecification::ColorDepthCompositing { .. });

    if let Some(bundle) =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes)
        .get(&pe_handle)
    {
      let (width, height) = extent;
      if bundle.swapchain_generation == swapchain_generation
        && bundle.width == width
        && bundle.height == height
      {
        return Ok((bundle.render_pass, bundle.framebuffer[image_index as usize]));
      }
    }

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&self.render_passes, device)
      .prepare_write((), |state, _| {
        if let Some(mut bundle) = state.remove(&pe_handle) {
          bundle.discard(discard_pool, self.allocator, timeline);
        }
        Ok::<(), crate::types::GpuError>(())
      })?
      .execute(|_, rollback| {
        if is_compositing {
          Self::create_compositing_bundle(
            &self.render_pass_device,
            device,
            allocator,
            self.allocator,
            rollback,
            color_format,
            depth_stencil_format,
            final_layout,
            extent,
            swapchain_generation,
            &image_views,
          )
        } else {
          Self::create_single_subpass_bundle(
            &self.render_pass_device,
            device,
            allocator,
            self.allocator,
            rollback,
            color_format,
            depth_stencil_format,
            final_layout,
            extent,
            swapchain_generation,
            &image_views,
          )
        }
      })
      .commit(|state, execute_result| {
        let bundle = execute_result?;
        let rp = bundle.render_pass;
        let fb = bundle.framebuffer[image_index as usize];
        unsafe { state.insert_unique_unchecked(pe_handle, bundle) };
        Ok((rp, fb))
      })
  }

  /// Initialize composite pipeline resources for a compositing render pass.
  /// Call this after `get_or_create_render_pass` when using `ColorDepthCompositing`.
  /// Creates the descriptor set layout, pipeline layout, descriptor pool/set,
  /// and writes the input attachment descriptors.
  #[named]
  pub fn init_composite_pipeline(
    &self,
    pe_handle: PresentationEngineHandle,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    pipeline_pool: &crate::gpu_backends::vulkan::device::pipelines::PipelinePool,
  ) -> GpuResult<()> {
    use crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock;

    let mut write_guard = DebugTrackedRwLock::write(&self.render_passes);
    let bundle = write_guard.get_mut(&pe_handle).ok_or(crate::gpu_err_device!())?;

    // Already initialized?
    if bundle.composite.is_some() {
      return Ok(());
    }

    // Only compositing bundles should have this — verify we have 6 attachments
    if bundle.attachments.len() < 6 {
      return Err(crate::gpu_err!(
        "init_composite_pipeline called on non-compositing bundle"
      ));
    }

    // Create descriptor set layout with 4 INPUT_ATTACHMENT bindings (set 0, bindings 0-3)
    let bindings = [
      vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
      vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
      vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
      vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let ds_layout = unsafe {
      NonZeroHandle::new_unchecked(device.create_descriptor_set_layout(&layout_ci, None)?)
    };

    // Push constant range for CompositePushConstants (16 bytes, fragment stage)
    let push_constant_range = vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::CompositePushConstants>() as u32);

    let ds_layout_raw = ds_layout.get();
    let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(core::slice::from_ref(&ds_layout_raw))
      .push_constant_ranges(core::slice::from_ref(&push_constant_range));
    let pipeline_layout = unsafe {
      NonZeroHandle::new_unchecked(device.create_pipeline_layout(&pipeline_layout_ci, None)?)
    };

    // Create descriptor pool with capacity for 4 INPUT_ATTACHMENT descriptors
    let pool_size = vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::INPUT_ATTACHMENT)
      .descriptor_count(4);
    let pool_ci = vk::DescriptorPoolCreateInfo::default()
      .max_sets(1)
      .pool_sizes(core::slice::from_ref(&pool_size));
    let descriptor_pool =
      unsafe { NonZeroHandle::new_unchecked(device.create_descriptor_pool(&pool_ci, None)?) };

    // Allocate descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool.get())
      .set_layouts(core::slice::from_ref(&ds_layout_raw));
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

    // Write descriptor set to point at the 4 transient attachment image views
    // Attachments: [0]=swapColor, [1]=swapDepth, [2]=macroColor, [3]=macroDepth,
    //              [4]=microColor, [5]=microDepth
    // Input attachment indices: 0=macroColor(att 2), 1=macroDepth(att 3),
    //                          2=microColor(att 4), 3=microDepth(att 5)
    //
    // For depth attachments, we MUST use depth-only views (DEPTH aspect only),
    // not the DEPTH|STENCIL views used for framebuffer attachments.
    let mut image_views = [vk::ImageView::null(); 4];

    // [0] macroColor — use the color attachment view directly
    match &bundle.attachments[2] {
      RenderPassAttachment::ColorAttachment(_, _, view) => image_views[0] = view.get(),
      _ => return Err(crate::gpu_err!("expected ColorAttachment at index 2")),
    }
    // [1] macroDepth — use depth-only view
    if bundle.depth_only_views.len() < 2 {
      return Err(crate::gpu_err!("depth_only_views not populated"));
    }
    image_views[1] = bundle.depth_only_views[0].get();

    // [2] microColor — use the color attachment view directly
    match &bundle.attachments[4] {
      RenderPassAttachment::ColorAttachment(_, _, view) => image_views[2] = view.get(),
      _ => return Err(crate::gpu_err!("expected ColorAttachment at index 4")),
    }
    // [3] microDepth — use depth-only view
    image_views[3] = bundle.depth_only_views[1].get();

    let image_infos: [vk::DescriptorImageInfo; 4] = core::array::from_fn(|i| {
      vk::DescriptorImageInfo::default()
        .image_layout(if i % 2 == 0 {
          // Even indices are color attachments
          vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
          // Odd indices are depth attachments
          vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        })
        .image_view(image_views[i])
    });

    let writes: [vk::WriteDescriptorSet; 4] = core::array::from_fn(|i| {
      vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(i as u32)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(core::slice::from_ref(&image_infos[i]))
    });

    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // Create composite graphics pipeline
    let composite_pipeline_key = {
      use crate::{
        gpu::PipelineKeyable,
        gpu_backends::vulkan::device::pipelines::{
          FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization, VertexIn,
        },
      };

      // Load composite shader modules
      let vert_spv = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/composite.vert.spv"
      ));
      let frag_spv = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/composite.frag.spv"
      ));

      // SPIR-V requires 4-byte alignment. include_bytes! may not guarantee
      // this, so we use an aligned intermediate buffer if needed.
      let vert_code: alloc::vec::Vec<u32> = match bytemuck::try_cast_slice(vert_spv) {
        Ok(aligned) => aligned.to_vec(),
        Err(_) => {
          let mut buf = alloc::vec![0u32; (vert_spv.len() + 3) / 4];
          unsafe {
            core::ptr::copy_nonoverlapping(
              vert_spv.as_ptr(),
              buf.as_mut_ptr() as *mut u8,
              vert_spv.len(),
            );
          }
          buf
        }
      };
      let frag_code: alloc::vec::Vec<u32> = match bytemuck::try_cast_slice(frag_spv) {
        Ok(aligned) => aligned.to_vec(),
        Err(_) => {
          let mut buf = alloc::vec![0u32; (frag_spv.len() + 3) / 4];
          unsafe {
            core::ptr::copy_nonoverlapping(
              frag_spv.as_ptr(),
              buf.as_mut_ptr() as *mut u8,
              frag_spv.len(),
            );
          }
          buf
        }
      };

      let vert_ci = vk::ShaderModuleCreateInfo::default().code(&vert_code);
      let frag_ci = vk::ShaderModuleCreateInfo::default().code(&frag_code);

      let vert_module = unsafe { device.create_shader_module(&vert_ci, None)? };
      let frag_module = unsafe { device.create_shader_module(&frag_ci, None)? };

      // Build the pipeline info — no vertex input (fullscreen triangle), subpass 2
      let mut graphics_info = GraphicsInfo::default()
        .with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST))
        .with_pre_rasterization(PreRasterization::default().with_vertex_module(vert_module))
        .with_fragment_shader(FragmentShader::default().with_fragment_module(frag_module))
        .with_pipeline_flags(
          PipelineFlags::NO_DEPTH_TEST | PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_BLEND,
        );

      // Manually set fields that apply_presentation_defaults would set, but with subpass=2
      graphics_info.fragment_shader.viewports.push(vk::Viewport::default());
      graphics_info.fragment_shader.scissors.push(vk::Rect2D::default());
      graphics_info
        .fragment_out
        .color_attachment_formats
        .push(vk::Format::B8G8R8A8_SRGB);
      graphics_info.pipeline_layout = pipeline_layout.get();
      graphics_info.render_pass = bundle.render_pass.get();
      graphics_info.subpass = 2; // Composite subpass

      let key = graphics_info.pipeline_key();

      // Create a temporary rollback context for pipeline creation
      let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(device);
      pipeline_pool.get_or_create_graphics_pipeline(device, &graphics_info, &mut rollback)?;
      rollback.defuse(); // Prevent rollback on success

      // Clean up shader modules — they're no longer needed after pipeline creation
      unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
      }

      key
    };

    bundle.composite = Some(CompositeResources {
      descriptor_set_layout: ds_layout,
      pipeline_layout,
      descriptor_pool,
      descriptor_set,
      pipeline_key: composite_pipeline_key,
    });

    Ok(())
  }

  /// Get composite resources for a PE handle. Returns None if not a compositing bundle.
  pub fn get_composite_resources(
    &self,
    pe_handle: PresentationEngineHandle,
  ) -> Option<(
    vk::DescriptorSet,
    vk::PipelineLayout,
    crate::gpu::PipelineKey,
  )> {
    let read_guard =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes);
    let bundle = read_guard.get(&pe_handle)?;
    let comp = bundle.composite.as_ref()?;
    Some((
      comp.descriptor_set,
      comp.pipeline_layout.get(),
      comp.pipeline_key,
    ))
  }

  /// Returns the VkRenderPass handle for the compositing bundle of the given PE.
  /// Returns None if no compositing bundle exists for this PE.
  #[allow(dead_code)]
  pub fn get_compositing_render_pass(
    &self,
    pe_handle: PresentationEngineHandle,
  ) -> Option<vk::RenderPass> {
    let read_guard =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes);
    let bundle = read_guard.get(&pe_handle)?;
    // Only return the handle if this bundle has composite resources
    // (meaning it was created as a compositing bundle)
    bundle.composite.as_ref()?;
    Some(bundle.render_pass.get())
  }

  // ---------------------------------------------------------------------------
  // Single-subpass bundle creation (unchanged logic, extracted into helper)
  // ---------------------------------------------------------------------------

  fn create_single_subpass_bundle(
    render_pass_device: &ash::khr::create_renderpass2::Device,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    vma: vk_mem::ffi::VmaAllocator,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_layout: vk::ImageLayout,
    extent: (u32, u32),
    swapchain_generation: u64,
    image_views: &heapless::Vec<NonZeroHandle<vk::ImageView>, { swapchain::MAX_FRAMES }>,
  ) -> GpuResult<RenderPassBundle> {
    let render_pass = Self::create_color_depth_single_render_pass(
      render_pass_device,
      color_format,
      depth_stencil_format,
      final_layout,
    )?;
    let rp_h = render_pass.get();
    rollback.defer(move |dev| unsafe { dev.destroy_render_pass(rp_h, None) });

    let (width, height) = extent;

    let black_value = vk::ClearValue {
      color: vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 0.0],
      },
    };
    let depth_clear = vk::ClearValue {
      depth_stencil: vk::ClearDepthStencilValue {
        depth: 0.0,
        stencil: 0,
      },
    };

    let mut attachments = heapless::Vec::new();
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::SwapchainColorImage);
    }

    let usage = if cfg!(test) {
      vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::SAMPLED
    } else {
      vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    };

    let (image, alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          vk::Extent2D { width, height },
          depth_stencil_format,
          usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          vk::Extent2D { width, height },
          depth_stencil_format,
          usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };

    let img_h = image.get();
    rollback.defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc.get_raw()) });

    let view_create_info = vk::ImageViewCreateInfo::default()
      .image(image.get())
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(depth_stencil_format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
          .level_count(1)
          .layer_count(1),
      );
    let view =
      unsafe { NonZeroHandle::new_unchecked(device.create_image_view(&view_create_info, None)?) };
    let view_h = view.get();
    rollback.defer(move |dev| unsafe { dev.destroy_image_view(view_h, None) });

    unsafe {
      attachments.push_unchecked(RenderPassAttachment::DepthStencilAttachment(
        image, alloc, view,
      ));
    }

    let mut framebuffer = heapless::Vec::new();
    for image_view in image_views {
      let fb_attachments = [image_view.get(), view.get()];
      let framebuffer_create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass.get())
        .width(width)
        .height(height)
        .layers(1)
        .attachments(&fb_attachments);
      let fb = unsafe {
        NonZeroHandle::new_unchecked(device.create_framebuffer(&framebuffer_create_info, None)?)
      };
      let fb_h = fb.get();
      rollback.defer(move |dev| unsafe { dev.destroy_framebuffer(fb_h, None) });
      unsafe {
        framebuffer.push_unchecked(fb);
      };
    }

    let mut clear_value = heapless::Vec::new();
    unsafe {
      clear_value.push_unchecked(black_value);
      clear_value.push_unchecked(depth_clear);
    }

    let bundle = RenderPassBundle {
      render_pass,
      clear_value,
      swapchain_generation,
      framebuffer,
      width,
      height,
      attachments,
      depth_only_views: heapless::Vec::new(),
      composite: None,
    };

    Ok(bundle)
  }

  // ---------------------------------------------------------------------------
  // Compositing bundle creation (3-subpass, 6-attachment)
  // ---------------------------------------------------------------------------

  #[allow(clippy::too_many_arguments)]
  fn create_compositing_bundle(
    render_pass_device: &ash::khr::create_renderpass2::Device,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    vma: vk_mem::ffi::VmaAllocator,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_layout: vk::ImageLayout,
    extent: (u32, u32),
    swapchain_generation: u64,
    image_views: &heapless::Vec<NonZeroHandle<vk::ImageView>, { swapchain::MAX_FRAMES }>,
  ) -> GpuResult<RenderPassBundle> {
    let render_pass = Self::create_compositing_render_pass(
      render_pass_device,
      color_format,
      depth_stencil_format,
      final_layout,
    )?;
    let rp_h = render_pass.get();
    rollback.defer(move |dev| unsafe { dev.destroy_render_pass(rp_h, None) });

    let (width, height) = extent;
    let ext2d = vk::Extent2D { width, height };

    let mut attachments: heapless::Vec<RenderPassAttachment, MAX_ATTACHMENTS> =
      heapless::Vec::new();

    // [0] swapchainColor — managed by presentation engine
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::SwapchainColorImage);
    }

    // [1] swapchainDepth
    let depth_usage_swapchain = if cfg!(test) {
      vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::SAMPLED
    } else {
      vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    };

    let (sc_depth_image, sc_depth_alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_usage_swapchain,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_usage_swapchain,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };

    {
      let img_h = sc_depth_image.get();
      let alloc_copy = sc_depth_alloc;
      rollback
        .defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_copy.get_raw()) });
    }

    let sc_depth_view =
      Self::create_depth_stencil_view(device, sc_depth_image, depth_stencil_format)?;
    {
      let vh = sc_depth_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }

    unsafe {
      attachments.push_unchecked(RenderPassAttachment::DepthStencilAttachment(
        sc_depth_image,
        sc_depth_alloc,
        sc_depth_view,
      ));
    }

    // --- Transient intermediate attachments ---
    // Color usage: COLOR_ATTACHMENT | INPUT_ATTACHMENT
    let color_transient_usage =
      vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT;
    // Depth usage: DEPTH_STENCIL_ATTACHMENT | INPUT_ATTACHMENT
    let depth_transient_usage =
      vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT;

    // Helper closure: allocate one transient image + view, register rollback
    // Returns (image, alloc, view) — must be stored in `attachments`.

    // [2] macroColor — RGBA8, transient color
    let (macro_color_img, macro_color_alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          ext2d,
          vk::Format::R8G8B8A8_UNORM,
          color_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          ext2d,
          vk::Format::R8G8B8A8_UNORM,
          color_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };
    {
      let ih = macro_color_img.get();
      let ac = macro_color_alloc;
      rollback.defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, ih, ac.get_raw()) });
    }
    let macro_color_view =
      Self::create_color_view(device, macro_color_img, vk::Format::R8G8B8A8_UNORM)?;
    {
      let vh = macro_color_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::ColorAttachment(
        macro_color_img,
        macro_color_alloc,
        macro_color_view,
      ));
    }

    // [3] macroDepth — D32S8, transient depth
    let (macro_depth_img, macro_depth_alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };
    {
      let ih = macro_depth_img.get();
      let ac = macro_depth_alloc;
      rollback.defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, ih, ac.get_raw()) });
    }
    let macro_depth_view =
      Self::create_depth_stencil_view(device, macro_depth_img, depth_stencil_format)?;
    {
      let vh = macro_depth_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::DepthStencilAttachment(
        macro_depth_img,
        macro_depth_alloc,
        macro_depth_view,
      ));
    }

    // [4] microColor — RGBA8, transient color
    let (micro_color_img, micro_color_alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          ext2d,
          vk::Format::R8G8B8A8_UNORM,
          color_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          ext2d,
          vk::Format::R8G8B8A8_UNORM,
          color_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };
    {
      let ih = micro_color_img.get();
      let ac = micro_color_alloc;
      rollback.defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, ih, ac.get_raw()) });
    }
    let micro_color_view =
      Self::create_color_view(device, micro_color_img, vk::Format::R8G8B8A8_UNORM)?;
    {
      let vh = micro_color_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::ColorAttachment(
        micro_color_img,
        micro_color_alloc,
        micro_color_view,
      ));
    }

    // [5] microDepth — D32S8, transient depth
    let (micro_depth_img, micro_depth_alloc) = {
      #[cfg(test)]
      {
        create_test_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
      #[cfg(not(test))]
      {
        create_transient_attachment(
          allocator,
          ext2d,
          depth_stencil_format,
          depth_transient_usage,
          vk::SampleCountFlags::TYPE_1,
        )?
      }
    };
    {
      let ih = micro_depth_img.get();
      let ac = micro_depth_alloc;
      rollback.defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, ih, ac.get_raw()) });
    }
    let micro_depth_view =
      Self::create_depth_stencil_view(device, micro_depth_img, depth_stencil_format)?;
    {
      let vh = micro_depth_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    unsafe {
      attachments.push_unchecked(RenderPassAttachment::DepthStencilAttachment(
        micro_depth_img,
        micro_depth_alloc,
        micro_depth_view,
      ));
    }

    // Create depth-only image views for input attachment descriptors.
    // Vulkan requires that input attachment descriptors for D32S8 images use
    // views with a single aspect (DEPTH), not DEPTH|STENCIL.
    let macro_depth_only_view =
      Self::create_depth_only_view(device, macro_depth_img, depth_stencil_format)?;
    {
      let vh = macro_depth_only_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    let micro_depth_only_view =
      Self::create_depth_only_view(device, micro_depth_img, depth_stencil_format)?;
    {
      let vh = micro_depth_only_view.get();
      rollback.defer(move |dev| unsafe { dev.destroy_image_view(vh, None) });
    }
    let mut depth_only_views = heapless::Vec::new();
    unsafe {
      depth_only_views.push_unchecked(macro_depth_only_view);
      depth_only_views.push_unchecked(micro_depth_only_view);
    }

    // --- Framebuffers ---
    // The swapchain color view varies per-framebuffer; the other 5 views are shared.
    let shared_views = [
      sc_depth_view.get(),    // [1]
      macro_color_view.get(), // [2]
      macro_depth_view.get(), // [3]
      micro_color_view.get(), // [4]
      micro_depth_view.get(), // [5]
    ];

    let mut framebuffer = heapless::Vec::new();
    for image_view in image_views {
      let fb_attachments = [
        image_view.get(), // [0] swapchain color (varies per frame)
        shared_views[0],  // [1] swapchain depth
        shared_views[1],  // [2] macroColor
        shared_views[2],  // [3] macroDepth
        shared_views[3],  // [4] microColor
        shared_views[4],  // [5] microDepth
      ];
      let framebuffer_create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass.get())
        .width(width)
        .height(height)
        .layers(1)
        .attachments(&fb_attachments);
      let fb = unsafe {
        NonZeroHandle::new_unchecked(device.create_framebuffer(&framebuffer_create_info, None)?)
      };
      let fb_h = fb.get();
      rollback.defer(move |dev| unsafe { dev.destroy_framebuffer(fb_h, None) });
      unsafe {
        framebuffer.push_unchecked(fb);
      };
    }

    // --- Clear values (6 attachments) ---
    let black_opaque = vk::ClearValue {
      color: vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
      },
    };
    let black_transparent = vk::ClearValue {
      color: vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 0.0],
      },
    };
    let depth_clear = vk::ClearValue {
      depth_stencil: vk::ClearDepthStencilValue {
        depth: 0.0, // reverse-Z
        stencil: 0,
      },
    };

    let mut clear_value = heapless::Vec::new();
    unsafe {
      clear_value.push_unchecked(black_opaque); // [0] swapchainColor
      clear_value.push_unchecked(depth_clear); // [1] swapchainDepth
      clear_value.push_unchecked(black_opaque); // [2] macroColor
      clear_value.push_unchecked(depth_clear); // [3] macroDepth
      clear_value.push_unchecked(black_transparent); // [4] microColor (alpha=0)
      clear_value.push_unchecked(depth_clear); // [5] microDepth
    }

    Ok(RenderPassBundle {
      render_pass,
      clear_value,
      swapchain_generation,
      framebuffer,
      width,
      height,
      attachments,
      depth_only_views,
      composite: None, // populated by create_composite_pipeline_resources()
    })
  }

  // ---------------------------------------------------------------------------
  // Image view creation helpers
  // ---------------------------------------------------------------------------

  fn create_color_view(
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    image: NonZeroHandle<vk::Image>,
    format: vk::Format,
  ) -> GpuResult<NonZeroHandle<vk::ImageView>> {
    let ci = vk::ImageViewCreateInfo::default()
      .image(image.get())
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .level_count(1)
          .layer_count(1),
      );
    unsafe {
      Ok(NonZeroHandle::new_unchecked(
        device.create_image_view(&ci, None)?,
      ))
    }
  }

  fn create_depth_stencil_view(
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    image: NonZeroHandle<vk::Image>,
    format: vk::Format,
  ) -> GpuResult<NonZeroHandle<vk::ImageView>> {
    let ci = vk::ImageViewCreateInfo::default()
      .image(image.get())
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
          .level_count(1)
          .layer_count(1),
      );
    unsafe {
      Ok(NonZeroHandle::new_unchecked(
        device.create_image_view(&ci, None)?,
      ))
    }
  }

  /// Create an image view with only DEPTH aspect for use in input attachment descriptors.
  /// Vulkan requires that input attachment descriptors for depth/stencil images use
  /// a view with either DEPTH or STENCIL aspect, but not both.
  fn create_depth_only_view(
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    image: NonZeroHandle<vk::Image>,
    format: vk::Format,
  ) -> GpuResult<NonZeroHandle<vk::ImageView>> {
    let ci = vk::ImageViewCreateInfo::default()
      .image(image.get())
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::DEPTH)
          .level_count(1)
          .layer_count(1),
      );
    unsafe {
      Ok(NonZeroHandle::new_unchecked(
        device.create_image_view(&ci, None)?,
      ))
    }
  }

  // ---------------------------------------------------------------------------
  // VkRenderPass creation — single subpass (original)
  // ---------------------------------------------------------------------------

  #[named]
  fn create_color_depth_single_render_pass(
    render_pass_device: &ash::khr::create_renderpass2::Device,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_color_layout: vk::ImageLayout,
  ) -> GpuResult<NonZeroHandle<vk::RenderPass>> {
    // In production, we don't care about memory after rendering (tiled GPU optimization).
    // In tests, we MUST STORE it back to memory so the copy command can read it.
    let (depth_store_op, stencil_store_op) = if cfg!(test) {
      (vk::AttachmentStoreOp::STORE, vk::AttachmentStoreOp::STORE)
    } else {
      (
        vk::AttachmentStoreOp::DONT_CARE,
        vk::AttachmentStoreOp::DONT_CARE,
      )
    };

    let attachments = [
      vk::AttachmentDescription2::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(final_color_layout),
      vk::AttachmentDescription2::default()
        .format(depth_stencil_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(depth_store_op) // configured dynamically
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(stencil_store_op) // configured dynamically
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    let subpass_0_output_attachment_refs = [
      vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR),
      vk::AttachmentReference2::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL),
    ];

    let subpass_0 = vk::SubpassDescription2::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(slice::from_ref(&subpass_0_output_attachment_refs[0]))
      .depth_stencil_attachment(&subpass_0_output_attachment_refs[1]);

    let mut external_0_memory_barrier = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
      .dst_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .src_access_mask(vk::AccessFlags2::empty())
      .dst_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
          | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      );

    let mut dst_stage_mask = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    let mut dst_access_mask = vk::AccessFlags2::empty();

    if final_color_layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL || cfg!(test) {
      dst_stage_mask = vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
      dst_access_mask = vk::AccessFlags2::TRANSFER_READ;
    }

    let mut _0_external_memory_barrier = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .dst_stage_mask(dst_stage_mask)
      .src_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
          | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      )
      .dst_access_mask(dst_access_mask);

    let subpass_dependencies = [
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(VK_SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .push_next(&mut external_0_memory_barrier),
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(0)
        .dst_subpass(VK_SUBPASS_EXTERNAL)
        .push_next(&mut _0_external_memory_barrier),
    ];

    let render_pass_create_info = vk::RenderPassCreateInfo2::default()
      .attachments(&attachments)
      .subpasses(slice::from_ref(&subpass_0))
      .dependencies(&subpass_dependencies);

    let render_pass = unsafe {
      NonZeroHandle::new_unchecked(
        render_pass_device.create_render_pass2(&render_pass_create_info, None)?,
      )
    };

    Ok(render_pass)
  }

  // ---------------------------------------------------------------------------
  // VkRenderPass creation — 3-subpass compositing
  // ---------------------------------------------------------------------------

  /// Creates a 3-subpass compositing render pass:
  ///
  /// ```text
  /// Attachments:
  ///   [0] swapchainColor  — final output (PRESENT_SRC_KHR or COLOR_ATTACHMENT_OPTIMAL)
  ///   [1] swapchainDepth  — D32S8
  ///   [2] macroColor      — RGBA8, transient
  ///   [3] macroDepth      — D32S8, transient
  ///   [4] microColor      — RGBA8, transient
  ///   [5] microDepth      — D32S8, transient
  ///
  /// Subpass 0 (macro):     color=[2], depth=[3]
  /// Subpass 1 (micro):     color=[4], depth=[5]
  /// Subpass 2 (composite): color=[0], depth=[1], input=[2,3,4,5]
  /// ```
  #[named]
  fn create_compositing_render_pass(
    render_pass_device: &ash::khr::create_renderpass2::Device,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_color_layout: vk::ImageLayout,
  ) -> GpuResult<NonZeroHandle<vk::RenderPass>> {
    // --- Attachment descriptions ---

    // Transient store ops: DONT_CARE in production (never read back).
    // In tests the compositing pass' own transients still don't need STORE.
    let attachments = [
      // [0] swapchainColor
      vk::AttachmentDescription2::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(final_color_layout),
      // [1] swapchainDepth
      vk::AttachmentDescription2::default()
        .format(depth_stencil_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(if cfg!(test) {
          vk::AttachmentStoreOp::STORE
        } else {
          vk::AttachmentStoreOp::DONT_CARE
        })
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(if cfg!(test) {
          vk::AttachmentStoreOp::STORE
        } else {
          vk::AttachmentStoreOp::DONT_CARE
        })
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
      // [2] macroColor — transient
      vk::AttachmentDescription2::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
      // [3] macroDepth — transient
      vk::AttachmentDescription2::default()
        .format(depth_stencil_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
      // [4] microColor — transient
      vk::AttachmentDescription2::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
      // [5] microDepth — transient
      vk::AttachmentDescription2::default()
        .format(depth_stencil_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    // --- Subpass 0 (macro): color=[2], depth=[3] ---

    let sp0_color_ref = vk::AttachmentReference2::default()
      .attachment(2)
      .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::COLOR);

    let sp0_depth_ref = vk::AttachmentReference2::default()
      .attachment(3)
      .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL);

    let subpass_0 = vk::SubpassDescription2::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(slice::from_ref(&sp0_color_ref))
      .depth_stencil_attachment(&sp0_depth_ref);

    // --- Subpass 1 (micro): color=[4], depth=[5] ---

    let sp1_color_ref = vk::AttachmentReference2::default()
      .attachment(4)
      .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::COLOR);

    let sp1_depth_ref = vk::AttachmentReference2::default()
      .attachment(5)
      .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL);

    let subpass_1 = vk::SubpassDescription2::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(slice::from_ref(&sp1_color_ref))
      .depth_stencil_attachment(&sp1_depth_ref);

    // --- Subpass 2 (composite): color=[0], depth=[1], input=[2,3,4,5] ---

    let sp2_color_ref = vk::AttachmentReference2::default()
      .attachment(0)
      .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::COLOR);

    let sp2_depth_ref = vk::AttachmentReference2::default()
      .attachment(1)
      .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
      .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL);

    // Input attachment references: color inputs use COLOR aspect,
    // depth inputs use DEPTH aspect only (not STENCIL) for subpassInput reads.
    let sp2_input_refs = [
      vk::AttachmentReference2::default()
        .attachment(2) // macroColor
        .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR),
      vk::AttachmentReference2::default()
        .attachment(3) // macroDepth
        .layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH),
      vk::AttachmentReference2::default()
        .attachment(4) // microColor
        .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR),
      vk::AttachmentReference2::default()
        .attachment(5) // microDepth
        .layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH),
    ];

    let subpass_2 = vk::SubpassDescription2::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(slice::from_ref(&sp2_color_ref))
      .depth_stencil_attachment(&sp2_depth_ref)
      .input_attachments(&sp2_input_refs);

    let subpasses = [subpass_0, subpass_1, subpass_2];

    // --- Subpass dependencies ---

    // EXTERNAL → 0: acquire → first subpass attachment writes
    let mut dep_ext_0 = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
      .dst_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .src_access_mask(vk::AccessFlags2::empty())
      .dst_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ
          | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      );

    // 0 → 2: macro outputs → composite input attachment reads
    let mut dep_0_2 = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
      .src_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      )
      .dst_access_mask(vk::AccessFlags2::INPUT_ATTACHMENT_READ);

    // 1 → 2: micro outputs → composite input attachment reads
    let mut dep_1_2 = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
      .src_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      )
      .dst_access_mask(vk::AccessFlags2::INPUT_ATTACHMENT_READ);

    // 2 → EXTERNAL: composite output → present/transfer
    let mut dst_stage_mask = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    let mut dst_access_mask = vk::AccessFlags2::empty();
    if final_color_layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL || cfg!(test) {
      dst_stage_mask = vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
      dst_access_mask = vk::AccessFlags2::TRANSFER_READ;
    }

    let mut dep_2_ext = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .dst_stage_mask(dst_stage_mask)
      .src_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ
          | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      )
      .dst_access_mask(dst_access_mask);

    let subpass_dependencies = [
      // EXTERNAL → 0
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(VK_SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
        .dst_stage_mask(
          vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(
          vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .push_next(&mut dep_ext_0),
      // 0 → 2
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(0)
        .dst_subpass(2)
        .src_stage_mask(
          vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .src_access_mask(
          vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        .push_next(&mut dep_0_2),
      // 1 → 2
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(1)
        .dst_subpass(2)
        .src_stage_mask(
          vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .src_access_mask(
          vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        .push_next(&mut dep_1_2),
      // 2 → EXTERNAL
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(2)
        .dst_subpass(VK_SUBPASS_EXTERNAL)
        .src_stage_mask(
          vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::from_raw(
          dst_stage_mask.as_raw() as u32
        ))
        .src_access_mask(
          vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::from_raw(dst_access_mask.as_raw() as u32))
        .push_next(&mut dep_2_ext),
    ];

    let render_pass_create_info = vk::RenderPassCreateInfo2::default()
      .attachments(&attachments)
      .subpasses(&subpasses)
      .dependencies(&subpass_dependencies);

    let render_pass = unsafe {
      NonZeroHandle::new_unchecked(
        render_pass_device.create_render_pass2(&render_pass_create_info, None)?,
      )
    };

    Ok(render_pass)
  }

  #[cfg(test)]
  /// TODO: Document this item
  pub(crate) fn get_test_depth_stencil_image(
    &self,
    pe_handle: PresentationEngineHandle,
  ) -> Option<NonZeroHandle<vk::Image>> {
    let read_render_passes =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes);
    let bundle = read_render_passes.get(&pe_handle)?;
    for attachment in bundle.attachments.iter() {
      if let RenderPassAttachment::DepthStencilAttachment(image, _, _) = attachment {
        return Some(*image);
      }
    }
    None
  }
}