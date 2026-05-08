use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct BarnesHutData {
    total_particles: u32,
    root_index: u32,
    particles: Vec<f32>,
    bvh_nodes: Vec<f32>,
    expected_particles: Vec<f32>,
    theta: f32,
    #[allow(non_snake_case)]
    G: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    particles_addr: u64,
    bvh_addr: u64,
    root_index: u32,
    total_particles: u32,
    theta: f32,
    g: f32,
}

#[test]
fn test_barnes_hut() {
    let json_data = fs::read_to_string("test_data/barnes_hut.json").expect("Failed to read JSON");
    let test_data: BarnesHutData = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let ctx = VulkanContext::new();

    let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
        &test_data.particles,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let (bvh_buffer, bvh_alloc, bvh_addr) = ctx.create_buffer(
        &test_data.bvh_nodes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let push_constants = PushConstants {
        particles_addr,
        bvh_addr,
        root_index: test_data.root_index,
        total_particles: test_data.total_particles,
        theta: test_data.theta,
        g: test_data.G,
    };

    let push_constants_bytes = unsafe {
        std::slice::from_raw_parts(
            &push_constants as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        )
    };

    let spv_path = "../../assets/sim/barnes_hut.comp.spv";
    let dispatch_x = (test_data.total_particles + 255) / 256;

    run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

    let output_particles: Vec<f32> = ctx.read_buffer(particles_buffer, &mut particles_alloc, test_data.particles.len());

    ctx.destroy_buffer(particles_buffer, particles_alloc);
    ctx.destroy_buffer(bvh_buffer, bvh_alloc);

    for (i, (actual, expected)) in output_particles.iter().zip(test_data.expected_particles.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-3, "Mismatch at index {}: actual {}, expected {}", i, actual, expected);
    }
}
