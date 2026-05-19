//! renderpasses module.

#[cfg(test)]
use crate::gpu_backends::vulkan::utils::create_test_attachment;
use crate::{
  gpu::{PresentationEngineHandle, vulkan::device::swapchain},
  gpu_backends::vulkan::{
    device::{
      DeviceResource, locks::DebugTrackedRwLock, resources::DiscardPool,
      swapchain::PresentationState,
    },
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
}

const MAX_ATTACHMENTS: usize = 8;
const VK_SUBPASS_EXTERNAL: u32 = 0xFFFFFFFF;

struct RenderPassBundle {
  render_pass: NonZeroHandle<vk::RenderPass>,
  // VkRenderPassBeginInfo (first -> color, second -> depth)
  clear_value: [vk::ClearValue; 2],
  // keep track of swapchain recreation
  swapchain_generation: u64,
  // VkFramebufferCreateInfo
  /// 1-1 correspondance with swapchain_image
  framebuffer: heapless::Vec<NonZeroHandle<vk::Framebuffer>, { swapchain::MAX_FRAMES }>,
  width: u32,
  height: u32,
  /// attachments handle: Note that they are 1 per graphics queue, which is just one per device in our setup
  attachments: heapless::Vec<RenderPassAttachment, MAX_ATTACHMENTS>,
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
        RenderPassAttachment::DepthStencilAttachment(image, allocation, image_view) => {
          discard_pool.discard_image_view(image_view.get(), timeline);
          discard_pool.discard_image(allocator, image.get(), *allocation, timeline);
        }
      }
    }

    for framebuffer in self.framebuffer.iter() {
      discard_pool.discard_framebuffer(framebuffer.get(), timeline);
    }

    discard_pool.discard_render_pass(self.render_pass.get(), timeline);
  }

  fn clean(&mut self, device: &ash::Device, allocator: vk_mem::ffi::VmaAllocator) {
    unsafe { device.destroy_render_pass(self.render_pass.get(), None) };

    for framebuffer in self.framebuffer.iter() {
      unsafe { device.destroy_framebuffer(framebuffer.get(), None) };
    }
    self.framebuffer.clear();

    for attachment in self.attachments.iter() {
      match attachment {
        RenderPassAttachment::DepthStencilAttachment(image, allocation, view) => {
          unsafe { device.destroy_image_view(view.get(), None) };
          unsafe { vk_mem::ffi::vmaDestroyImage(allocator, image.get(), allocation.get_raw()) };
        }
        RenderPassAttachment::SwapchainColorImage => {}
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

  /// TODO: Document this item
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

  /// TODO: Document this item
  #[named]
  pub fn get_clear_values_render_pass(
    &self,
    pe_handle: PresentationEngineHandle,
    out_values: &mut [vk::ClearValue],
  ) -> GpuResult<()> {
    let read_render_passes =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.render_passes);
    if !read_render_passes.contains_key(&pe_handle) {
      return Err(crate::gpu_err_device!());
    }
    if out_values.len() != 2 {
      return Err(crate::gpu_invalid_arg!("invalid argument"));
    }
    let bundle = unsafe { read_render_passes.get(&pe_handle).unwrap_unchecked() };
    out_values[0] = bundle.clear_value[0];
    out_values[1] = bundle.clear_value[1];
    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn get_or_create_render_pass(
    &self,
    pe_handle: PresentationEngineHandle,
    ty: RenderPassSpecification,
    image_index: u32,
    device: &ash::Device,
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
    ) = match ty {
      RenderPassSpecification::ColorDepthSingleSubpass {
        color_format,
        depth_stencil_format,
        final_layout,
        extent,
        swapchain_generation,
        image_views,
      } => (
        color_format,
        depth_stencil_format,
        final_layout,
        extent,
        swapchain_generation,
        image_views,
      ),
    };

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
        let render_pass = Self::create_color_depth_single_render_pass(
          &self.render_pass_device,
          color_format,
          depth_stencil_format,
          final_layout,
        )?;
        let rp_h = render_pass.get();
        rollback.defer(move |dev| unsafe { dev.destroy_render_pass(rp_h, None) });

        let (width, height) = extent;

        let black_value = vk::ClearValue {
          color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
          },
        };
        let white_value = vk::ClearValue {
          depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
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
        let vma = self.allocator;
        rollback
          .defer(move |_| unsafe { vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc.get_raw()) });

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
        let view = unsafe {
          NonZeroHandle::new_unchecked(device.create_image_view(&view_create_info, None)?)
        };
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

        let bundle = RenderPassBundle {
          render_pass,
          clear_value: [black_value, white_value],
          swapchain_generation,
          framebuffer,
          width,
          height,
          attachments,
        };

        Ok(bundle)
      })
      .commit(|state, execute_result| {
        let bundle = execute_result?;
        let rp = bundle.render_pass;
        let fb = bundle.framebuffer[image_index as usize];
        unsafe { state.insert_unique_unchecked(pe_handle, bundle) };
        Ok((rp, fb))
      })
  }

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
