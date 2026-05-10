use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct P5Data {
    num_particles: u32,
    num_emitters: u32,
    dt: f32,
    particles: Vec<f32>,
    emitters: Vec<f32>,
    expected_particles: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    particles_addr: u64,
    emitters_addr: u64,
    dt: f32,
    total_particles: u32,
    num_emitters: u32,
}

#[test]
fn test_p5_imex_particles() {
    let json_data = fs::read_to_string("test_data/p5.json").expect("Failed to read JSON");
    let test_data: P5Data = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let ctx = VulkanContext::new();

    let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
        &test_data.particles,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let (emitters_buffer, emitters_alloc, emitters_addr) = ctx.create_buffer(
        &test_data.emitters,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let push_constants = PushConstants {
        particles_addr,
        emitters_addr,
        dt: test_data.dt,
        total_particles: test_data.num_particles,
        num_emitters: test_data.num_emitters,
    };

    let push_constants_bytes = unsafe {
        std::slice::from_raw_parts(
            &push_constants as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        )
    };

    let spv_path = "../../assets/sim/p5_imex_particles.comp.spv";
    let dispatch_x = (test_data.num_particles + 127) / 128;

    run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

    let output_particles: Vec<f32> = ctx.read_buffer(particles_buffer, &mut particles_alloc, test_data.particles.len());

    ctx.destroy_buffer(particles_buffer, particles_alloc);
    ctx.destroy_buffer(emitters_buffer, emitters_alloc);

    for (i, (actual, expected)) in output_particles.iter().zip(test_data.expected_particles.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-3, "Mismatch at {}: actual {}, expected {}", i, actual, expected);
    }
}
