use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct ApplyImpulsesData {
    num_particles: u32,
    particles: Vec<f32>,
    collisions: Vec<u32>,
    impulses: Vec<f32>,
    expected_particles: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    particles_addr: u64,
    collisions_addr: u64,
    impulses_addr: u64,
}

#[test]
fn test_apply_impulses() {
    let json_data = fs::read_to_string("test_data/apply_impulses.json").expect("Failed to read JSON");
    let test_data: ApplyImpulsesData = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let ctx = VulkanContext::new();

    let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
        &test_data.particles,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let (collisions_buffer, collisions_alloc, collisions_addr) = ctx.create_buffer(
        &test_data.collisions,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let (impulses_buffer, impulses_alloc, impulses_addr) = ctx.create_buffer(
        &test_data.impulses,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let push_constants = PushConstants {
        particles_addr,
        collisions_addr,
        impulses_addr,
    };

    let push_constants_bytes = unsafe {
        std::slice::from_raw_parts(
            &push_constants as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        )
    };

    let spv_path = "../../assets/sim/apply_impulses.comp.spv";
    // Dispatch is based on collision count
    let collision_count = test_data.collisions[0];
    let dispatch_x = (collision_count + 255) / 256;

    // Run the compute shader
    run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

    let output_particles: Vec<f32> = ctx.read_buffer(particles_buffer, &mut particles_alloc, test_data.particles.len());

    ctx.destroy_buffer(particles_buffer, particles_alloc);
    ctx.destroy_buffer(collisions_buffer, collisions_alloc);
    ctx.destroy_buffer(impulses_buffer, impulses_alloc);

    for (i, (actual, expected)) in output_particles.iter().zip(test_data.expected_particles.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-4, "Mismatch at {}: actual {}, expected {}", i, actual, expected);
    }
}
