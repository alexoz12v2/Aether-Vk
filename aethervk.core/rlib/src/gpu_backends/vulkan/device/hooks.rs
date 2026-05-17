use alloc::collections::BTreeMap;
use ash::vk;
use spin::Mutex;

use crate::gpu_backends::vulkan::device::locks::assert_no_locks_held;

pub static QUEUE_TO_DEVICE: Mutex<BTreeMap<vk::Queue, vk::Device>> = Mutex::new(BTreeMap::new());
pub static CMD_BUF_TO_DEVICE: Mutex<BTreeMap<vk::CommandBuffer, vk::Device>> = Mutex::new(BTreeMap::new());

pub trait DispatchableToDevice {
    fn to_device(&self) -> vk::Device;
}
impl DispatchableToDevice for vk::Device {
    fn to_device(&self) -> vk::Device { *self }
}
impl DispatchableToDevice for vk::Queue {
    fn to_device(&self) -> vk::Device { *QUEUE_TO_DEVICE.lock().get(self).expect("vk::Queue not found in tracking map") }
}
impl DispatchableToDevice for vk::CommandBuffer {
    fn to_device(&self) -> vk::Device { *CMD_BUF_TO_DEVICE.lock().get(self).expect("vk::CommandBuffer not found in tracking map") }
}

pub static GET_DEVICE_QUEUE_PTRS: Mutex<BTreeMap<vk::Device, vk::PFN_vkGetDeviceQueue>> = Mutex::new(BTreeMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    assert_no_locks_held();
    let real_fn = *GET_DEVICE_QUEUE_PTRS.lock().get(&device).expect("vkGetDeviceQueue missing");
    unsafe { real_fn(device, queue_family_index, queue_index, p_queue) };
    QUEUE_TO_DEVICE.lock().insert(unsafe { *p_queue }, device);
}

pub static ALLOCATE_COMMAND_BUFFERS_PTRS: Mutex<BTreeMap<vk::Device, vk::PFN_vkAllocateCommandBuffers>> = Mutex::new(BTreeMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkAllocateCommandBuffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    assert_no_locks_held();
    let real_fn = *ALLOCATE_COMMAND_BUFFERS_PTRS.lock().get(&device).expect("vkAllocateCommandBuffers missing");
    let res = unsafe { real_fn(device, p_allocate_info, p_command_buffers) };
    if res == vk::Result::SUCCESS {
        let count = unsafe { (*p_allocate_info).command_buffer_count } as usize;
        let buffers = unsafe { core::slice::from_raw_parts(p_command_buffers, count) };
        let mut map = CMD_BUF_TO_DEVICE.lock();
        for &cb in buffers {
            map.insert(cb, device);
        }
    }
    res
}

pub static FREE_COMMAND_BUFFERS_PTRS: Mutex<BTreeMap<vk::Device, vk::PFN_vkFreeCommandBuffers>> = Mutex::new(BTreeMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkFreeCommandBuffers(
    device: vk::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    assert_no_locks_held();
    let real_fn = *FREE_COMMAND_BUFFERS_PTRS.lock().get(&device).expect("vkFreeCommandBuffers missing");
    unsafe { real_fn(device, command_pool, command_buffer_count, p_command_buffers) };
    let buffers = unsafe { core::slice::from_raw_parts(p_command_buffers, command_buffer_count as usize) };
    let mut map = CMD_BUF_TO_DEVICE.lock();
    for &cb in buffers {
        map.remove(&cb);
    }
}

macro_rules! define_hook {
    (
        $cmd_name:ident, $pfn_type:ident,
        fn( $first_arg:ident : $first_ty:ty $(, $arg_name:ident : $arg_type:ty )* $(,)? ) $( -> $ret_type:ty )?
    ) => {
        paste::paste! {
            #[allow(non_upper_case_globals)]
            pub static [< $cmd_name _PTRS >]: spin::Mutex<alloc::collections::BTreeMap<vk::Device, vk::$pfn_type>> = spin::Mutex::new(alloc::collections::BTreeMap::new());

            #[allow(non_upper_case_globals)]
            pub static mut [< $cmd_name _HOOK >]: Option<fn( $first_ty $(, $arg_type)* ) $( -> $ret_type )?> = None;

            #[allow(non_snake_case)]
            pub unsafe extern "system" fn [< hooked_ $cmd_name >](
                $first_arg : $first_ty
                $(, $arg_name : $arg_type )*
            ) $( -> $ret_type )? {
                crate::gpu_backends::vulkan::device::locks::assert_no_locks_held();

                unsafe {
                    if let Some(hook) = [< $cmd_name _HOOK >] {
                        let _ = hook( $first_arg $(, $arg_name )* );
                    }
                }

                let device = DispatchableToDevice::to_device(&$first_arg);
                let real_fn = *[< $cmd_name _PTRS >].lock().get(&device).expect(concat!(stringify!($cmd_name), " missing"));
                unsafe { real_fn( $first_arg $(, $arg_name )* ) }
            }
        }
    };
}

define_hook!(
    vkQueueSubmit, PFN_vkQueueSubmit,
    fn(queue: vk::Queue, submit_count: u32, p_submits: *const vk::SubmitInfo, fence: vk::Fence) -> vk::Result
);

define_hook!(
    vkQueueWaitIdle, PFN_vkQueueWaitIdle,
    fn(queue: vk::Queue) -> vk::Result
);

define_hook!(
    vkDeviceWaitIdle, PFN_vkDeviceWaitIdle,
    fn(device: vk::Device) -> vk::Result
);

define_hook!(
    vkCreateBuffer, PFN_vkCreateBuffer,
    fn(device: vk::Device, p_create_info: *const vk::BufferCreateInfo, p_allocator: *const vk::AllocationCallbacks, p_buffer: *mut vk::Buffer) -> vk::Result
);

define_hook!(
    vkDestroyBuffer, PFN_vkDestroyBuffer,
    fn(device: vk::Device, buffer: vk::Buffer, p_allocator: *const vk::AllocationCallbacks)
);

define_hook!(
    vkCreateImage, PFN_vkCreateImage,
    fn(device: vk::Device, p_create_info: *const vk::ImageCreateInfo, p_allocator: *const vk::AllocationCallbacks, p_image: *mut vk::Image) -> vk::Result
);

define_hook!(
    vkDestroyImage, PFN_vkDestroyImage,
    fn(device: vk::Device, image: vk::Image, p_allocator: *const vk::AllocationCallbacks)
);

define_hook!(
    vkCmdDraw, PFN_vkCmdDraw,
    fn(command_buffer: vk::CommandBuffer, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32)
);

define_hook!(
    vkCmdDrawIndexed, PFN_vkCmdDrawIndexed,
    fn(command_buffer: vk::CommandBuffer, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32)
);

define_hook!(
    vkCmdBindPipeline, PFN_vkCmdBindPipeline,
    fn(command_buffer: vk::CommandBuffer, pipeline_bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline)
);

define_hook!(
    vkCmdDispatch, PFN_vkCmdDispatch,
    fn(command_buffer: vk::CommandBuffer, group_count_x: u32, group_count_y: u32, group_count_z: u32)
);

pub unsafe fn load_device_with_hooks(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    create_info: &vk::DeviceCreateInfo,
) -> crate::types::GpuResult<ash::Device> {
    let device_handle = {
        let mut handle = vk::Device::null();
        let res = (instance.fp_v1_0().create_device)(
            physical_device,
            create_info,
            core::ptr::null(),
            &mut handle,
        );
        if res != vk::Result::SUCCESS {
            return Err(res.into());
        }
        handle
    };

    let device = ash::Device::load_with(|name| {
        let name_bytes = name.to_bytes();
        let real_ptr = (instance.fp_v1_0().get_device_proc_addr)(device_handle, name.as_ptr());

        if name_bytes == b"vkGetDeviceQueue" {
            let mut map = GET_DEVICE_QUEUE_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkGetDeviceQueue as *const core::ffi::c_void;
        }
        if name_bytes == b"vkAllocateCommandBuffers" {
            let mut map = ALLOCATE_COMMAND_BUFFERS_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkAllocateCommandBuffers as *const core::ffi::c_void;
        }
        if name_bytes == b"vkFreeCommandBuffers" {
            let mut map = FREE_COMMAND_BUFFERS_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkFreeCommandBuffers as *const core::ffi::c_void;
        }
        if name_bytes == b"vkQueueSubmit" {
            let mut map = vkQueueSubmit_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkQueueSubmit as *const core::ffi::c_void;
        }
        if name_bytes == b"vkQueueWaitIdle" {
            let mut map = vkQueueWaitIdle_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkQueueWaitIdle as *const core::ffi::c_void;
        }
        if name_bytes == b"vkDeviceWaitIdle" {
            let mut map = vkDeviceWaitIdle_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkDeviceWaitIdle as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCreateBuffer" {
            let mut map = vkCreateBuffer_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCreateBuffer as *const core::ffi::c_void;
        }
        if name_bytes == b"vkDestroyBuffer" {
            let mut map = vkDestroyBuffer_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkDestroyBuffer as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCreateImage" {
            let mut map = vkCreateImage_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCreateImage as *const core::ffi::c_void;
        }
        if name_bytes == b"vkDestroyImage" {
            let mut map = vkDestroyImage_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkDestroyImage as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCmdDraw" {
            let mut map = vkCmdDraw_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCmdDraw as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCmdDrawIndexed" {
            let mut map = vkCmdDrawIndexed_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCmdDrawIndexed as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCmdBindPipeline" {
            let mut map = vkCmdBindPipeline_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCmdBindPipeline as *const core::ffi::c_void;
        }
        if name_bytes == b"vkCmdDispatch" {
            let mut map = vkCmdDispatch_PTRS.lock();
            map.insert(device_handle, core::mem::transmute(real_ptr.unwrap()));
            return hooked_vkCmdDispatch as *const core::ffi::c_void;
        }

        real_ptr.map_or(core::ptr::null(), |p| p as *const () as *const core::ffi::c_void)
    }, device_handle);

    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    static QUEUE_SUBMIT_CALL_COUNT: AtomicU32 = AtomicU32::new(0);

    fn custom_submit_hook(
        _queue: vk::Queue,
        _submit_count: u32,
        _p_submits: *const vk::SubmitInfo,
        _fence: vk::Fence,
    ) -> vk::Result {
        QUEUE_SUBMIT_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        vk::Result::SUCCESS
    }

    #[test]
    fn test_custom_hook_registration() {
        unsafe {
            vkQueueSubmit_HOOK = Some(custom_submit_hook);
        }
        
        let is_some = unsafe { core::ptr::addr_of!(vkQueueSubmit_HOOK).read().is_some() };
        assert!(is_some);
    }
}
