//! Sky image overlay (HUD) device resources: the `R16_SFLOAT` texture and the registration mesh of
//! each scene's telescope image ([`crate::scene::sky_image`]), and the pipeline that draws it in
//! the composite subpass (2) between the composite and the UI ([`crate::gpu::sky_image_draw`]).
//!
//! - [`Device::sky_overlay_prepare`] (outside the render pass, once per frame): uploads the
//!   scene's image when its store generation changed (staging copy recorded on `cmd`), records
//!   which image and display parameters this presentation engine draws, retires replaced or
//!   removed images once the graphics timeline passes the next submit value.
//! - [`Device::cmd_draw_sky_overlay`] (subpass 2): binds the pipeline (created lazily per
//!   render pass), the image's descriptor set and vertex buffer, pushes [`SkyImagePushConstants`]
//!   and draws the triangle list.
use super::*;
use crate::scene::sky_image::{SkyImageEntry, SkyImageGpuData, SkyImageParams, SkyImageVertex};

/// Push constants of `sky_image.vert` / `sky_image.frag` (vertex + fragment stages).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyImagePushConstants {
  /// `proj · view` (rotation only), column major
  pub view_proj: [f32; 16],
  /// opacity, asinh stretch, black, white (normalised units)
  pub params: [f32; 4],
}
static_assertions::const_assert_eq!(core::mem::size_of::<SkyImagePushConstants>(), 80);

/// descriptor sets of the shared pool: images loaded at once, across scenes, plus the retired ones
/// still in flight
const SKY_OVERLAY_MAX_SETS: u32 = 32;

struct SkyOverlayImage {
  generation: u64,
  image: vk::Image,
  image_alloc: vk_mem::Allocation,
  view: vk::ImageView,
  vertices: vk::Buffer,
  vertices_alloc: vk_mem::Allocation,
  vertex_count: u32,
  set: vk::DescriptorSet,
}

/// Resources waiting for the graphics timeline.
#[derive(Default)]
struct Retired {
  buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
  image: Option<SkyOverlayImage>,
}

/// Layouts, pool and sampler shared by every image; created on the first upload.
struct Shared {
  set_layout: vk::DescriptorSetLayout,
  pipeline_layout: vk::PipelineLayout,
  pool: vk::DescriptorPool,
  sampler: vk::Sampler,
  /// pipeline per compatible render pass (the pipelines are owned by the pipeline pool)
  pipelines: Vec<(vk::RenderPass, vk::Pipeline)>,
}

#[derive(Default)]
struct Inner {
  shared: Option<Shared>,
  /// by store key (`scene::sky_image::scene_key`)
  images: Vec<(usize, SkyOverlayImage)>,
  /// what each presentation engine draws this frame
  current: Vec<(PresentationEngineHandle, usize, SkyImageParams)>,
  retired: Vec<(u64, Retired)>,
}

// Vulkan handles and VMA allocations are plain handles, externally synchronised by `inner`.
unsafe impl Send for Inner {}

/// Per-device sky overlay state (a field of `DeviceResources`).
#[derive(Default)]
pub struct SkyOverlayManager {
  inner: spin::Mutex<Inner>,
}

impl SkyOverlayManager {
  fn destroy_image(
    device: &LogicalDevice,
    allocator: &vk_mem::AllocatorView,
    img: SkyOverlayImage,
    pool: Option<vk::DescriptorPool>,
  ) {
    let SkyOverlayImage {
      image,
      mut image_alloc,
      view,
      vertices,
      mut vertices_alloc,
      set,
      ..
    } = img;
    unsafe {
      if let Some(pool) = pool {
        let _ = device.free_descriptor_sets(pool, &[set]);
      }
      device.destroy_image_view(view, None);
      allocator.destroy_image(image, &mut image_alloc);
      allocator.destroy_buffer(vertices, &mut vertices_alloc);
    }
  }

  fn destroy_retired(
    device: &LogicalDevice,
    allocator: &vk_mem::AllocatorView,
    r: Retired,
    pool: Option<vk::DescriptorPool>,
  ) {
    for (b, mut a) in r.buffers {
      unsafe { allocator.destroy_buffer(b, &mut a) };
    }
    if let Some(img) = r.image {
      Self::destroy_image(device, allocator, img, pool);
    }
  }

  /// Destroys everything (device teardown: the device is idle). The pipelines belong to the
  /// pipeline pool.
  pub fn cleanup(&self, device: &LogicalDevice, allocator: &vk_mem::AllocatorView) {
    let mut inner = self.inner.lock();
    let pool = inner.shared.as_ref().map(|s| s.pool);
    for (_, r) in core::mem::take(&mut inner.retired) {
      Self::destroy_retired(device, allocator, r, pool);
    }
    for (_, img) in core::mem::take(&mut inner.images) {
      Self::destroy_image(device, allocator, img, pool);
    }
    inner.current.clear();
    if let Some(s) = inner.shared.take() {
      unsafe {
        device.destroy_descriptor_pool(s.pool, None);
        device.destroy_sampler(s.sampler, None);
        device.destroy_pipeline_layout(s.pipeline_layout, None);
        device.destroy_descriptor_set_layout(s.set_layout, None);
      }
    }
  }
}

fn create_shared(device: &LogicalDevice) -> GpuResult<Shared> {
  let binding = vk::DescriptorSetLayoutBinding::default()
    .binding(0)
    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .descriptor_count(1)
    .stage_flags(vk::ShaderStageFlags::FRAGMENT);
  let set_layout = unsafe {
    device.create_descriptor_set_layout(
      &vk::DescriptorSetLayoutCreateInfo::default().bindings(core::slice::from_ref(&binding)),
      None,
    )?
  };
  let range = vk::PushConstantRange::default()
    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
    .offset(0)
    .size(core::mem::size_of::<SkyImagePushConstants>() as u32);
  let layouts = [set_layout];
  let pipeline_layout = unsafe {
    device.create_pipeline_layout(
      &vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&layouts)
        .push_constant_ranges(core::slice::from_ref(&range)),
      None,
    )
  };
  let pipeline_layout = match pipeline_layout {
    Ok(l) => l,
    Err(e) => {
      unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
      return Err(e.into());
    }
  };
  let size = vk::DescriptorPoolSize::default()
    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
    .descriptor_count(SKY_OVERLAY_MAX_SETS);
  let pool = unsafe {
    device.create_descriptor_pool(
      &vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(SKY_OVERLAY_MAX_SETS)
        .pool_sizes(core::slice::from_ref(&size)),
      None,
    )
  };
  let pool = match pool {
    Ok(p) => p,
    Err(e) => {
      unsafe {
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
      }
      return Err(e.into());
    }
  };
  // bilinear, clamped: the mesh covers exactly the image, edge texels must not wrap
  let sampler = unsafe {
    device.create_sampler(
      &vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(0.0),
      None,
    )
  };
  let sampler = match sampler {
    Ok(s) => s,
    Err(e) => {
      unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
      }
      return Err(e.into());
    }
  };
  Ok(Shared {
    set_layout,
    pipeline_layout,
    pool,
    sampler,
    pipelines: Vec::new(),
  })
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
  bytes
    .chunks_exact(4)
    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    .collect()
}

impl Device {
  /// A host-visible, mapped buffer holding `data`.
  fn sky_overlay_host_buffer(
    &self,
    allocator: &vk_mem::AllocatorView,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
  ) -> GpuResult<(vk::Buffer, vk_mem::Allocation)> {
    let info = vk::BufferCreateInfo::default().size(data.len() as u64).usage(usage);
    #[allow(unused_mut)]
    let mut alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferHost,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_info);
    let (buffer, mut alloc, res) = unsafe { allocator.create_buffer_get_info(&info, &alloc_info) }
      .with_name(&self.device, name)?;
    let mapped = res.mapped_data as *mut u8;
    if mapped.is_null() {
      unsafe { allocator.destroy_buffer(buffer, &mut alloc) };
      return Err(gpu_err!("sky overlay: {name} not mapped"));
    }
    // HOST_COHERENT: no flush
    unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len()) };
    Ok((buffer, alloc))
  }

  /// Creates the texture + mesh of `data`, recording the texture upload on `cmd`. Returns the
  /// image and the staging buffer to retire after the submit.
  fn sky_overlay_upload(
    &self,
    cmd: vk::CommandBuffer,
    allocator: &vk_mem::AllocatorView,
    shared: &Shared,
    data: &SkyImageGpuData,
    generation: u64,
  ) -> GpuResult<(SkyOverlayImage, (vk::Buffer, vk_mem::Allocation))> {
    let device = &self.device;
    let extent = vk::Extent3D {
      width: data.tex_width,
      height: data.tex_height,
      depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .extent(extent)
      .mip_levels(1)
      .array_layers(1)
      .format(vk::Format::R16_SFLOAT)
      .tiling(vk::ImageTiling::OPTIMAL)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1);
    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(alloc_info);
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    let (image, mut image_alloc, _) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &alloc_info) }
        .with_name(device, "VkImage_SkyOverlay")?;
    let subresource = vk::ImageSubresourceRange {
      aspect_mask: vk::ImageAspectFlags::COLOR,
      base_mip_level: 0,
      level_count: 1,
      base_array_layer: 0,
      layer_count: 1,
    };
    let view = unsafe {
      device.create_image_view(
        &vk::ImageViewCreateInfo::default()
          .image(image)
          .view_type(vk::ImageViewType::TYPE_2D)
          .format(vk::Format::R16_SFLOAT)
          .subresource_range(subresource),
        None,
      )
    };
    let view = match view {
      Ok(v) => v,
      Err(e) => {
        unsafe { allocator.destroy_image(image, &mut image_alloc) };
        return Err(e.into());
      }
    };
    let undo_image = |mut a: vk_mem::Allocation| unsafe {
      device.destroy_image_view(view, None);
      allocator.destroy_image(image, &mut a);
    };
    let staging = match self.sky_overlay_host_buffer(
      allocator,
      bytemuck::cast_slice(&data.texels),
      vk::BufferUsageFlags::TRANSFER_SRC,
      "SkyOverlay_Staging",
    ) {
      Ok(s) => s,
      Err(e) => {
        undo_image(image_alloc);
        return Err(e);
      }
    };
    let (vertices, vertices_alloc) = match self.sky_overlay_host_buffer(
      allocator,
      bytemuck::cast_slice::<SkyImageVertex, u8>(&data.vertices),
      vk::BufferUsageFlags::VERTEX_BUFFER,
      "SkyOverlay_Vertices",
    ) {
      Ok(v) => v,
      Err(e) => {
        let (b, mut a) = staging;
        unsafe { allocator.destroy_buffer(b, &mut a) };
        undo_image(image_alloc);
        return Err(e);
      }
    };
    let layouts = [shared.set_layout];
    let set = unsafe {
      device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
          .descriptor_pool(shared.pool)
          .set_layouts(&layouts),
      )
    };
    let set = match set {
      Ok(s) => s[0],
      Err(e) => {
        let (b, mut a) = staging;
        let mut va = vertices_alloc;
        unsafe {
          allocator.destroy_buffer(b, &mut a);
          allocator.destroy_buffer(vertices, &mut va);
        }
        undo_image(image_alloc);
        return Err(e.into());
      }
    };
    let image_desc = vk::DescriptorImageInfo::default()
      .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      .image_view(view)
      .sampler(shared.sampler);
    unsafe {
      device.update_descriptor_sets(
        &[vk::WriteDescriptorSet::default()
          .dst_set(set)
          .dst_binding(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(&image_desc))],
        &[],
      );
      let to_dst = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(subresource);
      device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_dst)),
      );
      device.cmd_copy_buffer_to_image(
        cmd,
        staging.0,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::BufferImageCopy::default()
          .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
          })
          .image_extent(extent)],
      );
      let to_read = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(image)
        .subresource_range(subresource);
      device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_read)),
      );
    }
    Ok((
      SkyOverlayImage {
        generation,
        image,
        image_alloc,
        view,
        vertices,
        vertices_alloc,
        vertex_count: data.vertices.len() as u32,
        set,
      },
      staging,
    ))
  }

  /// Per frame, outside the render pass: syncs the presentation engine `pe` with the store entry of
  /// its scene (`entry`, `None`: no image), uploading on `cmd` when the generation changed.
  pub fn sky_overlay_prepare(
    &self,
    cmd: vk::CommandBuffer,
    pe: PresentationEngineHandle,
    key: usize,
    entry: Option<&SkyImageEntry>,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    let mut inner = res.sky_overlay.inner.lock();
    let pool = inner.shared.as_ref().map(|s| s.pool);

    // 1. retired resources the GPU is done with
    let done = res.timeline_manager.get_cached_value();
    let mut i = 0;
    while i < inner.retired.len() {
      if inner.retired[i].0 <= done {
        let (_, r) = inner.retired.swap_remove(i);
        SkyOverlayManager::destroy_retired(&self.device, &allocator, r, pool);
      } else {
        i += 1;
      }
    }
    let retire_at = res.timeline_manager.get_next_submit_value();

    inner.current.retain(|(p, _, _)| *p != pe);
    let Some(entry) = entry else {
      // removed from the store: retire its image (no presentation engine may draw it any more)
      if let Some(pos) = inner.images.iter().position(|(k, _)| *k == key) {
        let (_, img) = inner.images.swap_remove(pos);
        inner.current.retain(|(_, k, _)| *k != key);
        inner.retired.push((
          retire_at,
          Retired {
            buffers: Vec::new(),
            image: Some(img),
          },
        ));
      }
      return Ok(());
    };

    // 2. upload a new generation
    let Inner {
      shared,
      images,
      retired,
      ..
    } = &mut *inner;
    let have = images.iter().position(|(k, _)| *k == key);
    if have.map(|p| images[p].1.generation) != Some(entry.generation) {
      if shared.is_none() {
        *shared = Some(create_shared(&self.device)?);
      }
      let shared = shared.as_ref().unwrap();
      let (img, staging) =
        self.sky_overlay_upload(cmd, &allocator, shared, &entry.data, entry.generation)?;
      let old = match have {
        Some(p) => Some(core::mem::replace(&mut images[p].1, img)),
        None => {
          images.push((key, img));
          None
        }
      };
      retired.push((
        retire_at,
        Retired {
          buffers: alloc::vec![staging],
          image: old,
        },
      ));
      aethervk_oshal_rlib::log!(
        "[SkyOverlay] uploaded image {}×{} ({} vertices), generation {}",
        entry.data.tex_width,
        entry.data.tex_height,
        entry.data.vertices.len(),
        entry.generation
      );
    }
    inner.current.push((pe, key, entry.params));
    Ok(())
  }

  /// The pipeline of `render_pass` subpass 2 (created on first use, owned by the pipeline pool).
  fn sky_overlay_pipeline(
    &self,
    res: &DeviceResources,
    shared: &mut Shared,
    render_pass: vk::RenderPass,
    subpass: u32,
  ) -> GpuResult<vk::Pipeline> {
    if let Some((_, p)) = shared.pipelines.iter().find(|(rp, _)| *rp == render_pass) {
      return Ok(*p);
    }
    use super::pipelines::{
      FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization, VertexIn,
    };
    let vert = spirv_words(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/../../assets/sky_image.vert.spv"
    )));
    let frag = spirv_words(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/../../assets/sky_image.frag.spv"
    )));
    let vert = unsafe {
      self
        .device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert), None)?
    };
    let frag = match unsafe {
      self
        .device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag), None)
    } {
      Ok(f) => f,
      Err(e) => {
        unsafe { self.device.destroy_shader_module(vert, None) };
        return Err(e.into());
      }
    };
    let stride = core::mem::size_of::<SkyImageVertex>() as u32;
    let mut info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
          .add_binding(0, stride, vk::VertexInputRate::VERTEX)
          .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0)
          .add_attribute(0, 1, vk::Format::R32G32_SFLOAT, 12),
      )
      .with_pre_rasterization(PreRasterization::default().with_vertex_module(vert))
      .with_fragment_shader(FragmentShader::default().with_fragment_module(frag))
      // alpha "over" on the swapchain colour; nothing written to the global depth attachment
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST | PipelineFlags::NO_DEPTH_WRITE);
    info.fragment_shader.viewports.push(vk::Viewport::default());
    info.fragment_shader.scissors.push(vk::Rect2D::default());
    info.fragment_out.color_attachment_formats.push(vk::Format::B8G8R8A8_SRGB);
    info.fragment_out.color_write_masks.push(vk::ColorComponentFlags::RGBA);
    super::pipelines::pad_compositing_outputs(&mut info.fragment_out, subpass);
    info.pipeline_layout = shared.pipeline_layout;
    info.render_pass = render_pass;
    info.subpass = subpass;
    info.debug_name = alloc::string::String::from("SkyOverlay");
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&self.device);
    let pipeline =
      res
        .pipeline_pool
        .get_or_create_graphics_pipeline(&self.device, &info, &mut rollback);
    rollback.defuse();
    unsafe {
      self.device.destroy_shader_module(vert, None);
      self.device.destroy_shader_module(frag, None);
    }
    let pipeline = pipeline?.get();
    shared.pipelines.push((render_pass, pipeline));
    Ok(pipeline)
  }

  /// Draws presentation engine `pe`'s sky image in the current compositing subpass of
  /// `cmd_buffer` (call in subpass 2). `view_proj` = the camera's `proj · view` (rotation only).
  /// Returns whether something was drawn.
  pub fn cmd_draw_sky_overlay(
    &self,
    cmd_buffer: CommandBufferHandle,
    pe: PresentationEngineHandle,
    view_proj: [f32; 16],
  ) -> GpuResult<bool> {
    let (cmd, render_pass, subpass) = {
      let data = self
        .recording_command_buffers
        .get(&(cmd_buffer, QueueRole::Graphics))
        .ok_or(gpu_err!("sky overlay: command buffer not recording"))?;
      let Some(ctx) = data.compositing_ctx.as_ref() else {
        return Ok(false);
      };
      (data.command_buffer.get(), ctx.render_pass, ctx.subpass)
    };
    let res = self.res.read();
    let mut inner = res.sky_overlay.inner.lock();
    let Some(&(_, key, params)) = inner.current.iter().find(|(p, _, _)| *p == pe) else {
      return Ok(false);
    };
    if !params.visible || !(params.opacity > 0.0) {
      return Ok(false);
    }
    let Some((set, vbuf, count)) = inner
      .images
      .iter()
      .find(|(k, _)| *k == key)
      .map(|(_, i)| (i.set, i.vertices, i.vertex_count))
    else {
      return Ok(false);
    };
    let Inner { shared, .. } = &mut *inner;
    let Some(shared) = shared.as_mut() else {
      return Ok(false);
    };
    let pipeline = self.sky_overlay_pipeline(&res, shared, render_pass, subpass)?;
    let push = SkyImagePushConstants {
      view_proj,
      params: [params.opacity, params.stretch, 0.0, 1.0],
    };
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        shared.pipeline_layout,
        0,
        &[set],
        &[],
      );
      self.device.cmd_bind_vertex_buffers(cmd, 0, &[vbuf], &[0]);
      self.device.cmd_push_constants(
        cmd,
        shared.pipeline_layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        bytemuck::bytes_of(&push),
      );
      self.device.cmd_draw(cmd, count, 1, 0, 0);
    }
    Ok(true)
  }
}
