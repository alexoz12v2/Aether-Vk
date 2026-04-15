use ash::vk::{
  self, Handle, PFN_vkAllocateDescriptorSets, PFN_vkCreateDescriptorPool, PFN_vkResetDescriptorPool,
};
use alloc::{sync, vec::Vec};
use core::{ptr};

use crate::{
  gpu_backends::vulkan::{
    device::{DeviceResource, resources},
    utils::NonZeroHandle,
  },
  types::{GpuError, GpuResult},
};

const MAX_DESCRIPTOR_SETS: u32 = 1024;
const POOL_SIZES: [vk::DescriptorPoolSize; 8] = [
  vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::INPUT_ATTACHMENT, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 1024 },
  vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 1024 },
];

#[derive(Debug)]
struct DescriptorPoolsInner {
  active_pool: vk::DescriptorPool,
  full_pools: Vec<vk::DescriptorPool>,
  recycled_pools: Vec<vk::DescriptorPool>,
}

#[derive(Debug)]
pub(super) struct DescriptorPools {
  inner: spin::Mutex<DescriptorPoolsInner>,
}

unsafe impl Sync for DescriptorPools {}
unsafe impl Send for DescriptorPools {}

impl DescriptorPools {
  pub(super) fn new(
    device: &ash::Device,
    _fixed_capacity_pow2: usize,
  ) -> GpuResult<sync::Arc<Self>> {
    let mut inner = DescriptorPoolsInner {
      active_pool: vk::DescriptorPool::null(),
      full_pools: Vec::new(),
      recycled_pools: Vec::new(),
    };
    inner.ensure_active_pool(device.handle(), device.fp_v1_0().create_descriptor_pool)?;
    
    Ok(sync::Arc::new(Self {
      inner: spin::Mutex::new(inner),
    }))
  }

  pub(super) fn allocate(
    self: &sync::Arc<Self>,
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
    discard_pool: &resources::DiscardPool,
    timeline_value: u64,
  ) -> GpuResult<NonZeroHandle<vk::DescriptorSet>> {
    let mut inner = self.inner.lock();
    loop {
      let pool = inner.active_pool;
      if pool == vk::DescriptorPool::null() {
        return Err(GpuError::InvalidState);
      }

      let layouts = [layout];
      let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
        
      let mut descriptor_set = vk::DescriptorSet::null();
      let res = unsafe {
        (device.fp_v1_0().allocate_descriptor_sets)(
          device.handle(),
          ptr::from_ref(&alloc_info),
          ptr::from_mut(&mut descriptor_set),
        )
      };

      match res {
        vk::Result::SUCCESS => return Ok(unsafe { NonZeroHandle::new_unchecked(descriptor_set) }),
        vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL => {
          self.discard_active_pool(&mut inner, discard_pool, timeline_value);
          inner.ensure_active_pool(device.handle(), device.fp_v1_0().create_descriptor_pool)?;
          // Loop continues and tries again with new pool
        }
        e => return Err(e.into()),
      }
    }
  }

  pub(super) fn recycle(&self, device: &ash::Device, pool: vk::DescriptorPool) {
    if pool.is_null() { return; }
    if unsafe {
      (device.fp_v1_0().reset_descriptor_pool)(
        device.handle(),
        pool,
        vk::DescriptorPoolResetFlags::empty(),
      ).result()
    }.is_ok() {
      let mut inner = self.inner.lock();
      inner.recycled_pools.push(pool);
    }
  }

  fn discard_active_pool(
    self: &sync::Arc<Self>,
    inner: &mut DescriptorPoolsInner,
    discard_pool: &resources::DiscardPool,
    timeline_value: u64,
  ) {
    if !inner.active_pool.is_null() {
      discard_pool.discard_descriptor_pool(
        inner.active_pool,
        sync::Arc::clone(self),
        timeline_value,
      );
      inner.full_pools.push(inner.active_pool);
      inner.active_pool = vk::DescriptorPool::null();
    }
  }
}

impl DescriptorPoolsInner {
  fn ensure_active_pool(
    &mut self,
    device: vk::Device,
    create_descriptor_pool: PFN_vkCreateDescriptorPool,
  ) -> GpuResult<()> {
    if !self.active_pool.is_null() { return Ok(()); }
    if let Some(pool) = self.recycled_pools.pop() {
      self.active_pool = pool;
      return Ok(());
    }

    let pool_sizes = POOL_SIZES;
    let create_info = vk::DescriptorPoolCreateInfo::default()
      .max_sets(MAX_DESCRIPTOR_SETS)
      .pool_sizes(&pool_sizes);
      
    let mut pool = vk::DescriptorPool::null();
    let res = unsafe {
      (create_descriptor_pool)(
        device,
        ptr::from_ref(&create_info),
        ptr::null(),
        ptr::from_mut(&mut pool),
      )
    };
    
    if res == vk::Result::SUCCESS {
      self.active_pool = pool;
      Ok(())
    } else {
      Err(GpuError::InvalidState)
    }
  }
}

impl DeviceResource for DescriptorPools {
  fn cleanup(&mut self, device: &ash::Device) {
    let mut inner = self.inner.lock();
    if !inner.active_pool.is_null() {
      unsafe { device.destroy_descriptor_pool(inner.active_pool, None) };
    }
    for pool in inner.full_pools.drain(..) {
      unsafe { device.destroy_descriptor_pool(pool, None) };
    }
    for pool in inner.recycled_pools.drain(..) {
      unsafe { device.destroy_descriptor_pool(pool, None) };
    }
  }
}