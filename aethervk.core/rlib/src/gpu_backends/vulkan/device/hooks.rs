use ash::vk;
use spin::Mutex;
use hashbrown::HashMap;

use crate::gpu_backends::vulkan::device::locks::assert_no_locks_held;

pub static QUEUE_TO_DEVICE: Mutex<HashMap<vk::Queue, vk::Device>> = Mutex::new(HashMap::new());
pub static CMD_BUF_TO_DEVICE: Mutex<HashMap<vk::CommandBuffer, vk::Device>> = Mutex::new(HashMap::new());

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

pub static GET_DEVICE_QUEUE_PTRS: Mutex<HashMap<vk::Device, vk::PFN_vkGetDeviceQueue>> = Mutex::new(HashMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    assert_no_locks_held();
    let real_fn = *GET_DEVICE_QUEUE_PTRS.lock().get(&device).expect("vkGetDeviceQueue missing");
    real_fn(device, queue_family_index, queue_index, p_queue);
    QUEUE_TO_DEVICE.lock().insert(*p_queue, device);
}

pub static ALLOCATE_COMMAND_BUFFERS_PTRS: Mutex<HashMap<vk::Device, vk::PFN_vkAllocateCommandBuffers>> = Mutex::new(HashMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkAllocateCommandBuffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    assert_no_locks_held();
    let real_fn = *ALLOCATE_COMMAND_BUFFERS_PTRS.lock().get(&device).expect("vkAllocateCommandBuffers missing");
    let res = real_fn(device, p_allocate_info, p_command_buffers);
    if res == vk::Result::SUCCESS {
        let count = (*p_allocate_info).command_buffer_count as usize;
        let buffers = core::slice::from_raw_parts(p_command_buffers, count);
        let mut map = CMD_BUF_TO_DEVICE.lock();
        for &cb in buffers {
            map.insert(cb, device);
        }
    }
    res
}

pub static FREE_COMMAND_BUFFERS_PTRS: Mutex<HashMap<vk::Device, vk::PFN_vkFreeCommandBuffers>> = Mutex::new(HashMap::new());

#[allow(non_snake_case)]
pub unsafe extern "system" fn hooked_vkFreeCommandBuffers(
    device: vk::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    assert_no_locks_held();
    let real_fn = *FREE_COMMAND_BUFFERS_PTRS.lock().get(&device).expect("vkFreeCommandBuffers missing");
    real_fn(device, command_pool, command_buffer_count, p_command_buffers);
    let buffers = core::slice::from_raw_parts(p_command_buffers, command_buffer_count as usize);
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
            pub static [< $cmd_name _PTRS >]: spin::Mutex<hashbrown::HashMap<vk::Device, vk::$pfn_type>> = spin::Mutex::new(hashbrown::HashMap::new());

            #[allow(non_upper_case_globals)]
            pub static mut [< $cmd_name _HOOK >]: Option<fn( $first_ty $(, $arg_type)* ) $( -> $ret_type )?> = None;

            #[allow(non_snake_case)]
            pub unsafe extern "system" fn [< hooked_ $cmd_name >](
                $first_arg : $first_ty
                $(, $arg_name : $arg_type )*
            ) $( -> $ret_type )? {
                crate::gpu_backends::vulkan::device::locks::assert_no_locks_held();

                if let Some(hook) = [< $cmd_name _HOOK >] {
                    hook( $first_arg $(, $arg_name )* );
                }

                let device = DispatchableToDevice::to_device(&$first_arg);
                let real_fn = *[< $cmd_name _PTRS >].lock().get(&device).expect(concat!(stringify!($cmd_name), " missing"));
                real_fn( $first_arg $(, $arg_name )* )
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
