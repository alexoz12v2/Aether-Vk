use ash::vk::{
  self, Handle, PFN_vkAllocateDescriptorSets, PFN_vkCreateDescriptorPool, PFN_vkResetDescriptorPool,
};
use alloc::{sync, vec::Vec};
use core::{ptr};

use crate::{
  gpu_backends::vulkan::device::{DeviceResource, resources},
  types::{GpuError, GpuResult, SpscQueue},
};

const POOL_SIZE_STORAGE_BUFFER: u32 = 210;
const POOL_SIZE_STORAGE_IMAGE: u32 = 126;
const POOL_SIZE_COMBINED_IMAGE_SAMPLER: u32 = 250;
const POOL_SIZE_SAMPLER: u32 = 32;
const POOL_SIZE_SAMPLED_IMAGE: u32 = 250;
const POOL_SIZE_UNIFORM_BUFFER: u32 = 216;
const POOL_SIZE_UNIFORM_TEXEL_BUFFER: u32 = 32;
const POOL_SIZE_INPUT_ATTACHMENT: u32 = 9;
const MAX_DESCRIPTOR_SETS: u32 = 256;
const POOL_SIZES: [vk::DescriptorPoolSize; 8] = [
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::STORAGE_BUFFER,
    descriptor_count: POOL_SIZE_STORAGE_BUFFER,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::STORAGE_IMAGE,
    descriptor_count: POOL_SIZE_STORAGE_IMAGE,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    descriptor_count: POOL_SIZE_COMBINED_IMAGE_SAMPLER,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::UNIFORM_BUFFER,
    descriptor_count: POOL_SIZE_UNIFORM_BUFFER,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
    descriptor_count: POOL_SIZE_UNIFORM_TEXEL_BUFFER,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::INPUT_ATTACHMENT,
    descriptor_count: POOL_SIZE_INPUT_ATTACHMENT,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::SAMPLER,
    descriptor_count: POOL_SIZE_SAMPLER,
  },
  vk::DescriptorPoolSize {
    ty: vk::DescriptorType::SAMPLED_IMAGE,
    descriptor_count: POOL_SIZE_SAMPLED_IMAGE,
  },
];

#[derive(Debug)]
struct DescriptorPoolsInner {
  recycled_pools: SpscQueue<vk::DescriptorPool>,
  active_pool: vk::DescriptorPool,
}

#[derive(Debug)]
pub(super) struct DescriptorPools {
  inner: spin::RwLock<DescriptorPoolsInner>,
}

unsafe impl Sync for DescriptorPools {}
unsafe impl Send for DescriptorPools {}

impl DescriptorPools {
  pub(super) fn new(
    device: &ash::Device,
    fixed_capacity_pow2: usize,
  ) -> GpuResult<sync::Arc<Self>> {
    let s = sync::Arc::new(Self {
      inner: spin::RwLock::new(DescriptorPoolsInner {
        recycled_pools: SpscQueue::new(fixed_capacity_pow2),
        active_pool: vk::DescriptorPool::null(),
      }),
    });

    // initialize the first pool
    s.inner
      .write()
      .ensure_active_pool(device.handle(), device.fp_v1_0().create_descriptor_pool)?;
    Ok(s)
  }

  pub(super) fn allocate(
    self: &sync::Arc<Self>,
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
    discard_pool: &resources::DiscardPool,
    timeline_value: u64,
  ) -> GpuResult<vk::DescriptorSet> {
    loop {
      let inner = self.inner.read();
      let pool = if inner.active_pool != vk::DescriptorPool::null() {
        Ok(inner.active_pool)
      } else {
        GpuResult::Err(GpuError::InvalidState)
      }?;

      let layouts = [layout];
      let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
      match unsafe {
        let mut descriptor_set = vk::DescriptorSet::null();
        let res = (device.fp_v1_0().allocate_descriptor_sets)(
          device.handle(),
          ptr::from_ref(&alloc_info),
          ptr::from_mut(&mut descriptor_set),
        );
        res.result_with_success(descriptor_set)
      } {
        Ok(sets) => return Ok(sets),
        Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
          let mut inner = self.inner.write();
          self.discard_active_pool(&mut inner, discard_pool, timeline_value);
          inner.ensure_active_pool(device.handle(), device.fp_v1_0().create_descriptor_pool)?;
          // Loop continues to try again with the pool
        }
        Err(e) => return Err(e.into()),
      }
    }
  }

  pub(super) fn recycle(&self, device: &ash::Device, pool: vk::DescriptorPool) {
    if pool.is_null() {
      return;
    }
    // TODO: unsuccessful log
    if unsafe {
      (device.fp_v1_0().reset_descriptor_pool)(
        device.handle(),
        pool,
        vk::DescriptorPoolResetFlags::empty(),
      )
      .result()
    }
    .is_ok()
    {
      let inner = self.inner.write();
      let _ = inner.recycled_pools.try_push(pool);
    }
  }

  fn discard_active_pool(
    self: &sync::Arc<Self>,
    inner: &mut spin::RwLockWriteGuard<DescriptorPoolsInner>,
    discard_pool: &resources::DiscardPool,
    timeline_value: u64,
  ) {
    if !inner.active_pool.is_null() {
      // Pass the Arc of self to discard pool so that DiscardPool can call the `recycle` method when it's ready to give back the Descriptor Pool
      discard_pool.discard_descriptor_pool(
        inner.active_pool,
        sync::Arc::clone(self),
        timeline_value,
      );
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
    if !self.active_pool.is_null() {
      return Ok(());
    }
    if let Some(pool) = self.recycled_pools.try_pop() {
      self.active_pool = pool;
      return Ok(());
    }

    let pool_sizes = POOL_SIZES;
    let create_info = vk::DescriptorPoolCreateInfo::default()
      .max_sets(MAX_DESCRIPTOR_SETS)
      .pool_sizes(&pool_sizes);
    self.active_pool = unsafe {
      let mut pool = vk::DescriptorPool::null();
      let res = (create_descriptor_pool)(
        device,
        ptr::from_ref(&create_info),
        ptr::null(),
        ptr::from_mut(&mut pool),
      );
      res.result_with_success(pool)
    }?;
    Ok(())
  }
}

impl DeviceResource for DescriptorPools {
  fn cleanup(&mut self, device: &ash::Device) {
    let inner = self.inner.write();
    if !inner.active_pool.is_null() {
      unsafe { device.destroy_descriptor_pool(inner.active_pool, None) };
    }
    while let Some(recycled_pool) = inner.recycled_pools.try_pop() {
      unsafe { device.destroy_descriptor_pool(recycled_pool, None) };
    }
  }
}

pub(super) trait DescriptorPoolsCaller {
  fn allocate(&self, layout: vk::DescriptorSetLayout) -> GpuResult<vk::DescriptorSet>;
  fn recycle(&self, pool: vk::DescriptorPool);
}
