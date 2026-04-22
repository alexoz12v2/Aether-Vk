use core::{ptr, slice};

use aethervk_oshal_rlib::panic_handler_impl;
use ash::{khr::create_renderpass2, vk};
use spin::RwLock;

use crate::{
  gpu::frame,
  gpu_backends::vulkan::{
    device::{
      DeviceResource,
      resources::DiscardPool,
      swapchain::{MAX_FRAMES_IN_FLIGHT, PresentationState},
    },
    utils::{NonZeroHandle, create_transient_attachment},
  },
  types::GpuResult,
};

/// Struct which encapsulates what we are interested about a render pass, namely
/// - color attachment format and depth/stencil format
pub(super) struct RenderPassInfo {
  pub color_format: vk::Format,
  pub depth_stencil_format: vk::Format,
}

impl RenderPassInfo {
  pub fn new(color_format: vk::Format, depth_stencil_format: vk::Format) -> Self {
    Self {
      color_format,
      depth_stencil_format,
    }
  }
}

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
  framebuffer: heapless::Vec<NonZeroHandle<vk::Framebuffer>, { MAX_FRAMES_IN_FLIGHT }>,
  width: u32,
  height: u32,
  /// attachments handle: Note that they are 1 per graphics queue, which is just one per device in our setup
  attachments: heapless::Vec<RenderPassAttachment, MAX_ATTACHMENTS>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum RenderPassType {
  ColorDepthSingleSubpass,
}

pub(super) enum RenderPassSpecification<'a> {
  ColorDepthSingleSubpass {
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    swapchain: &'a PresentationState,
  },
}

impl<'a> RenderPassSpecification<'a> {
  pub fn single_pass(presentation_engine: &'a PresentationState, d: vk::Format) -> Self {
    Self::ColorDepthSingleSubpass {
      color_format: presentation_engine.format(),
      depth_stencil_format: d,
      swapchain: presentation_engine,
    }
  }
}

pub(super) struct RenderPasses {
  render_passes: RwLock<hashbrown::HashMap<RenderPassType, RenderPassBundle>>,
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
    for (_, mut bundle) in self.render_passes.write().drain() {
      bundle.clean(&device, self.allocator);
    }
  }
}

unsafe impl Sync for RenderPasses {}
unsafe impl Send for RenderPasses {}

/// Thin abstraction over render pass creation and management.
/// Note: It Implicitly requires to not outlive the VmaAllocator
impl RenderPasses {
  pub fn new(
    instance: &ash::Instance,
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
  ) -> Self {
    Self {
      render_passes: RwLock::new(hashbrown::HashMap::with_capacity(8)),
      render_pass_device: ash::khr::create_renderpass2::Device::new(instance, device),
      allocator: allocator.get_raw(),
    }
  }

  pub fn get_clear_values_render_pass(
    &self,
    ty: RenderPassType,
    out_values: &mut [vk::ClearValue],
  ) -> GpuResult<()> {
    match ty {
      RenderPassType::ColorDepthSingleSubpass => {
        let read_render_passes = self.render_passes.read();
        if !read_render_passes.contains_key(&RenderPassType::ColorDepthSingleSubpass) {
          return Err(crate::types::GpuError::InvalidState);
        }
        if out_values.len() != 2 {
          return Err(crate::types::GpuError::InvalidArgument);
        }
        let bundle = unsafe {
          read_render_passes
            .get(&RenderPassType::ColorDepthSingleSubpass)
            .unwrap_unchecked()
        };
        out_values[0] = bundle.clear_value[0];
        out_values[1] = bundle.clear_value[1];

        Ok(())
      }
    }
  }

  pub fn get_or_create_render_pass(
    &self,
    ty: RenderPassSpecification,
    image_index: u32,
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    discard_pool: &DiscardPool,
    timeline: u64,
  ) -> GpuResult<(
    NonZeroHandle<vk::RenderPass>,
    NonZeroHandle<vk::Framebuffer>,
  )> {
    match ty {
      RenderPassSpecification::ColorDepthSingleSubpass {
        color_format,
        depth_stencil_format,
        swapchain,
      } => {
        if let Some(bundle) = self
          .render_passes
          .read()
          .get(&RenderPassType::ColorDepthSingleSubpass)
        {
          if bundle.swapchain_generation == swapchain.swapchain_generation() {
            return Ok((bundle.render_pass, bundle.framebuffer[image_index as usize]));
          }
        }

        let mut write_render_passes = self.render_passes.write();
        if let Some(mut bundle) =
          write_render_passes.remove(&RenderPassType::ColorDepthSingleSubpass)
        {
          bundle.discard(discard_pool, self.allocator, timeline);
        }

        let final_layout = match swapchain {
          PresentationState::Windowed(_) => vk::ImageLayout::PRESENT_SRC_KHR,
          PresentationState::Windowless(_) => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        };

        let render_pass = Self::create_color_depth_single_render_pass(
          &self.render_pass_device,
          color_format,
          depth_stencil_format,
          final_layout,
        )
        .or_else(|e| {
          let _ = (&mut write_render_passes).remove(&RenderPassType::ColorDepthSingleSubpass);
          Err(e)
        })?;
        let (width, height) = swapchain.extent();

        // Safety: We removed everything with that key above.
        let black_value = vk::ClearValue {
          color: vk::ClearColorValue {
            float32: [0.0, 0.0, 1.0, 1.0],
          },
        };
        // TODO: zero our stencil
        let white_value = vk::ClearValue {
          depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
          },
        };
        unsafe {
          write_render_passes.insert_unique_unchecked(
            RenderPassType::ColorDepthSingleSubpass,
            RenderPassBundle {
              render_pass,
              clear_value: [black_value, white_value],
              swapchain_generation: swapchain.swapchain_generation(),
              framebuffer: heapless::Vec::new(),
              width,
              height,
              attachments: heapless::Vec::new(),
            },
          )
        };

        // create attachments
        // Safety: This is empty and cap is 8
        unsafe {
          write_render_passes
            .get_mut(&mut RenderPassType::ColorDepthSingleSubpass)
            .unwrap_unchecked()
            .attachments
            .push_unchecked(RenderPassAttachment::SwapchainColorImage)
        };

        let (image, alloc) = create_transient_attachment(
          allocator,
          vk::Extent2D { width, height },
          depth_stencil_format,
          vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
        )
        .or_else(|e| {
          let _ = write_render_passes.remove(&RenderPassType::ColorDepthSingleSubpass);
          Err(e)
        })?;
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
        let view = unsafe { device.create_image_view(&view_create_info, None) }.or_else(|e| {
          let _ = write_render_passes.remove(&RenderPassType::ColorDepthSingleSubpass);
          Err(e)
        })?;
        let view = unsafe { NonZeroHandle::new_unchecked(view) };
        unsafe {
          write_render_passes
            .get_mut(&RenderPassType::ColorDepthSingleSubpass)
            .unwrap_unchecked()
            .attachments
            .push_unchecked(RenderPassAttachment::DepthStencilAttachment(
              image, alloc, view,
            ));
        }

        // create framebuffers
        swapchain
          .for_each_swapchain_image(|image_view| {
            let attachments = [image_view.get(), view.get()];
            let framebuffer_create_info = vk::FramebufferCreateInfo::default()
              .render_pass(render_pass.get())
              .width(width)
              .height(height)
              .layers(1)
              .attachments(&attachments);
            let framebuffer = unsafe {
              NonZeroHandle::new_unchecked(
                device.create_framebuffer(&framebuffer_create_info, None)?,
              )
            };
            unsafe {
              write_render_passes
                .get_mut(&RenderPassType::ColorDepthSingleSubpass)
                .unwrap_unchecked()
                .framebuffer
                .push_unchecked(framebuffer);
            };
            Ok(())
          })
          .or_else(|e| {
            let _ = write_render_passes.remove(&RenderPassType::ColorDepthSingleSubpass);
            Err(e)
          })?;
        drop(write_render_passes);
        let read_render_passes = self.render_passes.read();
        let bundle = unsafe {
          read_render_passes
            .get(&RenderPassType::ColorDepthSingleSubpass)
            .unwrap_unchecked()
        };

        Ok((bundle.render_pass, bundle.framebuffer[image_index as usize]))
      }
    }
  }

  fn create_color_depth_single_render_pass(
    render_pass_device: &ash::khr::create_renderpass2::Device,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
    final_color_layout: vk::ImageLayout,
  ) -> GpuResult<NonZeroHandle<vk::RenderPass>> {
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
        .store_op(vk::AttachmentStoreOp::DONT_CARE) // combined with TRANSIENT usage, lets tiled GPUs not store depth image explicitly
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    // subpass 0
    let subpass_0_output_attachment_refs = [
      vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) // required: External -> 0: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL transition
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
      // anything that can write to depth buffer or color buffer
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
    let mut _0_external_memory_barrier = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
          | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
          | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
      .src_access_mask(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
          | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
          | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
      )
      // note: VK_ACCESS_MEMORY_READ is allowed on exiting dependency, but needed only if there's a
      // consumer for this memory within the render-pass
      .dst_access_mask(vk::AccessFlags2::empty());

    let subpass_dependencies = [
      // EXTERNAL -> 0
      vk::SubpassDependency2::default()
        .dependency_flags(vk::DependencyFlags::BY_REGION)
        .src_subpass(VK_SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .push_next(&mut external_0_memory_barrier),
      // 0 -> EXTERNAL
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
}
