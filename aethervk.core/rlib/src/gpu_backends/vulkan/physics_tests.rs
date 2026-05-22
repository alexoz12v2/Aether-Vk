
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backends::vulkan::mock_scene_data::{generate_mock_particles, generate_sparse_collisions};
    use crate::gpu_backends::vulkan::{VulkanCore, LogicalDevice};
    use ash::vk;
    use std::sync::Arc;

    #[test]
    fn test_stream_compact_and_reduce_toi() {
        // Since we need an actual VulkanDevice, we use VulkanCore
        let core = VulkanCore::from_path(None, None).expect("Failed to init Vulkan");
        let handle = core.create_headless_device().expect("Failed to create headless device");
        let device = core.live_devices.get(&handle).unwrap();
        
        let kernels = VulkanComputeKernels::new(device.as_ref()).expect("Failed to init kernels");
        let mut cmd = kernels.create_command_buffer().unwrap();
        let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(device.as_ref());

        let num_elements = 10000;
        let sparse_data = generate_sparse_collisions(num_elements, 0.5);
        
        let sparse_buffer = kernels.allocate_device_buffer::<crate::gpu_backends::vulkan::mock_scene_data::SparseCollisionData>(
            device.as_ref(),
            device.get_vma(),
            num_elements,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            true,
            &mut rollback,
        ).unwrap();
        
        let mut staging = kernels.allocate_device_buffer(
            device.as_ref(),
            device.get_vma(),
            num_elements,
            vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            &mut rollback,
        ).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(sparse_data.as_ptr(), staging.get_mapped_ptr().unwrap() as *mut _, num_elements);
        }
        
        let copy_region = vk::BufferCopy::default().size((num_elements * std::mem::size_of::<crate::gpu_backends::vulkan::mock_scene_data::SparseCollisionData>()) as u64);
        unsafe { device.cmd_copy_buffer(cmd.cmd, staging.buffer, sparse_buffer.buffer, &[copy_region]); }
        
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        
        let packed_out = kernels.allocate_device_buffer::<gpu::CollisionPair>(
            device.as_ref(),
            device.get_vma(),
            num_elements,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            true,
            &mut rollback,
        ).unwrap();
        
        unsafe { device.cmd_fill_buffer(cmd.cmd, packed_out.buffer, 0, 16, 0); }
        
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let pc_compact = StreamCompactPushConstants {
            sparse_in: sparse_buffer.address,
            packed_out: packed_out.address,
            total_elements: num_elements as u32,
            _pad: 0,
        };
        let bytes_compact = unsafe {
            core::slice::from_raw_parts(
                &pc_compact as *const _ as *const u8,
                core::mem::size_of::<StreamCompactPushConstants>(),
            )
        };
        
        unsafe {
            device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, kernels.pipelines.stream_compact);
            device.cmd_push_constants(cmd.cmd, kernels.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_compact);
            device.cmd_dispatch(cmd.cmd, (num_elements as u32 + 127) / 128, 1, 1);
        }

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let mut header_staging = kernels.allocate_device_buffer::<u32>(
            device.as_ref(),
            device.get_vma(),
            4,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            &mut rollback,
        ).unwrap();
        
        let header_copy = vk::BufferCopy::default().size(16);
        unsafe { device.cmd_copy_buffer(cmd.cmd, packed_out.buffer, header_staging.buffer, &[header_copy]); }

        let sync = cmd.submit().unwrap().unwrap();
        kernels.wait_sync(&sync).unwrap();
        
        let header_data = unsafe { std::slice::from_raw_parts(header_staging.get_mapped_ptr().unwrap() as *const u32, 4) };
        println!("Stream Compacted Count: {}", header_data[3]);
        assert!(header_data[3] > 0);
        
        let mut cmd = kernels.create_command_buffer().unwrap();
        let out_toi = kernels.allocate_device_buffer::<u32>(
            device.as_ref(),
            device.get_vma(),
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            true,
            &mut rollback,
        ).unwrap();
        
        unsafe { device.cmd_fill_buffer(cmd.cmd, out_toi.buffer, 0, 4, f32::to_bits(1.0)); }
        
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let particles_data = generate_mock_particles(num_elements);
        let particles = kernels.allocate_device_buffer::<f32>(
            device.as_ref(),
            device.get_vma(),
            num_elements * 10,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            true,
            &mut rollback,
        ).unwrap();
        
        let mut particles_staging = kernels.allocate_device_buffer::<f32>(
            device.as_ref(),
            device.get_vma(),
            num_elements * 10,
            vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            &mut rollback,
        ).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(particles_data.as_ptr(), particles_staging.get_mapped_ptr().unwrap() as *mut _, num_elements * 10);
        }
        
        let p_copy = vk::BufferCopy::default().size((num_elements * 10 * 4) as u64);
        unsafe { device.cmd_copy_buffer(cmd.cmd, particles_staging.buffer, particles.buffer, &[p_copy]); }
        
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let pc_toi = ReduceToiPushConstants {
            particles: particles.address,
            collisions: packed_out.address,
            out_toi: out_toi.address,
            particle_radius: 1.0,
            dt: 0.016,
        };
        let bytes_toi = unsafe {
            core::slice::from_raw_parts(
                &pc_toi as *const _ as *const u8,
                core::mem::size_of::<ReduceToiPushConstants>(),
            )
        };
        
        unsafe {
            device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, kernels.pipelines.reduce_toi);
            device.cmd_push_constants(cmd.cmd, kernels.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_toi);
            device.cmd_dispatch(cmd.cmd, 1, 1, 1);
        }
        
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        
        let mut toi_staging = kernels.allocate_device_buffer::<u32>(
            device.as_ref(),
            device.get_vma(),
            1,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            &mut rollback,
        ).unwrap();
        
        let t_copy = vk::BufferCopy::default().size(4);
        unsafe { device.cmd_copy_buffer(cmd.cmd, out_toi.buffer, toi_staging.buffer, &[t_copy]); }

        let sync = cmd.submit().unwrap().unwrap();
        kernels.wait_sync(&sync).unwrap();
        
        let toi_data = unsafe { std::slice::from_raw_parts(toi_staging.get_mapped_ptr().unwrap() as *const u32, 1) };
        let final_toi = f32::from_bits(toi_data[0]);
        println!("Final TOI: {}", final_toi);
    }
}
