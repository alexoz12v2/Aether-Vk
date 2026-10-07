//! Large texture upload: textures bigger than the 32 MiB per-frame staging arena (a single
//! 4096² RGBA texture is 64 MiB) go through `Image::new_2d_dedicated`. Needs a Vulkan device
//! (any, Lavapipe included).
use super::{
  test_utils::{setup_assets_dir, setup_render_frontend_for_tests},
  *,
};
use crate::simulation::comet::{TexelFormat, Texture};

fn with_device(f: impl FnOnce(&Device)) {
  setup_assets_dir();
  let (_pool, render_frontend, handle, _) = setup_render_frontend_for_tests(false);
  render_frontend
    .with_device(handle, |dyn_device| {
      let device: &Device = dyn_device.as_any().downcast_ref().unwrap();
      f(device);
      unsafe { device.device.device_wait_idle() }.unwrap();
      GpuResult::Ok(())
    })
    .unwrap();
}

/// Copies a `SHADER_READ_ONLY_OPTIMAL` RGBA8 image back to host memory.
fn read_image(device: &Device, image: vk::Image, w: u32, h: u32) -> alloc::vec::Vec<u8> {
  let bytes = (w * h * 4) as u64;
  let allocator = device.res.read().allocator.allocator.as_allocator_view();
  let info = vk::BufferCreateInfo::default()
    .size(bytes)
    .usage(vk::BufferUsageFlags::TRANSFER_DST);
  let mut alloc_info = vk_mem::AllocationCreateInfo::default();
  alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
  alloc_info.flags =
    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
  let (dst, mut dst_alloc, dst_info) =
    unsafe { allocator.create_buffer_get_info(&info, &alloc_info) }.unwrap();
  let range = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
  };
  let mut res = device
    .run_transient_commands(|cmd| {
      let to_src = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image)
        .subresource_range(range);
      let dep = vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_src));
      let copy = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
          aspect_mask: vk::ImageAspectFlags::COLOR,
          mip_level: 0,
          base_array_layer: 0,
          layer_count: 1,
        })
        .image_extent(vk::Extent3D {
          width: w,
          height: h,
          depth: 1,
        });
      unsafe {
        device.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
        device.device.cmd_copy_image_to_buffer(
          cmd,
          image,
          vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
          dst,
          &[copy],
        );
      }
      Ok(())
    })
    .unwrap();
  res.cleanup(&device.device);
  allocator.invalidate_allocation(&dst_alloc, 0, bytes).unwrap();
  let out =
    unsafe { core::slice::from_raw_parts(dst_info.mapped_data.cast::<u8>(), bytes as usize) }
      .to_vec();
  unsafe { allocator.destroy_buffer(dst, &mut dst_alloc) };
  out
}

#[test]
fn texture_larger_than_frame_staging_arena_uploads_via_dedicated_staging() {
  const N: u32 = 4096;
  let mut data = alloc::vec::Vec::with_capacity((N * N * 4) as usize);
  for y in 0..N {
    for x in 0..N {
      data.extend_from_slice(&[
        (x % 251) as u8,
        (y % 241) as u8,
        ((x ^ y) & 0xff) as u8,
        255,
      ]);
    }
  }
  assert!(
    data.len() > 32 * 1024 * 1024,
    "must exceed the per-frame staging arena"
  );
  assert!(data.len() > resources::DedicatedStaging::ARENA_BYPASS_THRESHOLD);
  let texture = Texture {
    data: data.into(),
    format: TexelFormat::R8G8B8A8_UNORM,
    width: N,
    height: N,
    has_mipmaps: false,
  };

  with_device(|device| {
    let allocator = device.res.read().allocator.allocator.as_allocator_view();
    let mut staging = None;
    let mut image = None;
    let mut res = device
      .run_transient_commands(|cmd| {
        let (img, st) = resources::Image::new_2d_dedicated(
          &device.device,
          allocator,
          cmd,
          &texture,
          vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
          "test_big_texture",
        )?;
        image = Some(img);
        staging = Some(st);
        Ok(())
      })
      .unwrap();
    res.cleanup(&device.device);
    // the transient submission has completed: staging can go
    unsafe { staging.unwrap().destroy() };

    let img = image.unwrap();
    let back = read_image(device, img.image.get(), N, N);
    let px = |x: u32, y: u32| {
      let i = ((y * N + x) * 4) as usize;
      [back[i], back[i + 1], back[i + 2], back[i + 3]]
    };
    for (x, y) in [(0, 0), (4095, 0), (1234, 3210), (4095, 4095), (2048, 17)] {
      assert_eq!(
        px(x, y),
        [
          (x % 251) as u8,
          (y % 241) as u8,
          ((x ^ y) & 0xff) as u8,
          255
        ],
        "texel ({x},{y})"
      );
    }

    let mut alloc_h = img.allocation;
    unsafe {
      device.device.destroy_image_view(img.image_view.get(), None);
      allocator.destroy_image(img.image.get(), &mut alloc_h);
    }
  });
}
