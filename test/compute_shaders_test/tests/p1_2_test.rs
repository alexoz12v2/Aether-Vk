use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use bytemuck::{Pod, Zeroable};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct P1_2_Data {
    dt: f32,
    total_particles: u32,
    input_particles: Vec<f32>,
    expected_particles: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    particles_addr: u64,
    dt: f32,
    total_particles: u32,
}

#[test]
fn test_p1_2_imex_particles() {
    let json_data = fs::read_to_string("test_data/p1_2.json").expect("Failed to read JSON");
    let test_data: P1_2_Data = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let ctx = VulkanContext::new();

    let (buffer, mut alloc, address) = ctx.create_buffer(
        &test_data.input_particles,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let push_constants = PushConstants {
        particles_addr: address,
        dt: test_data.dt,
        total_particles: test_data.total_particles,
    };

    let push_constants_bytes = unsafe {
        std::slice::from_raw_parts(
            &push_constants as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        )
    };

    let spv_path = "../../assets/sim/p1-2_imex_particles.comp.spv";
    let dispatch_x = (test_data.total_particles + 127) / 128;

    run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

    let output_particles: Vec<f32> = ctx.read_buffer(buffer, &mut alloc, test_data.input_particles.len());

    ctx.destroy_buffer(buffer, alloc);

    for (i, (actual, expected)) in output_particles.iter().zip(test_data.expected_particles.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-3, "Mismatch at {}: actual {}, expected {}", i, actual, expected);
    }
}
