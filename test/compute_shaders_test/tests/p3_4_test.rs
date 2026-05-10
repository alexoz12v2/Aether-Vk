use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct P3_4Data {
    total_bodies: u32,
    num_emitters: u32,
    dt: f32,
    rigid_bodies: Vec<f32>,
    emitters: Vec<f32>,
    expected_rigid_bodies: Vec<f32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
    rigid_bodies_addr: u64,
    emitters_addr: u64,
    total_bodies: u32,
    num_emitters: u32,
    dt: f32,
}

#[test]
fn test_p3_4_imex_rigidbody_imr() {
    let json_data = fs::read_to_string("test_data/p3_4.json").expect("Failed to read JSON");
    let test_data: P3_4Data = serde_json::from_str(&json_data).expect("Failed to parse JSON");

    let ctx = VulkanContext::new();

    let (rb_buffer, mut rb_alloc, rb_addr) = ctx.create_buffer(
        &test_data.rigid_bodies,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let (emitters_buffer, emitters_alloc, emitters_addr) = ctx.create_buffer(
        &test_data.emitters,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );

    let push_constants = PushConstants {
        rigid_bodies_addr: rb_addr,
        emitters_addr,
        total_bodies: test_data.total_bodies,
        num_emitters: test_data.num_emitters,
        dt: test_data.dt,
    };

    let push_constants_bytes = unsafe {
        std::slice::from_raw_parts(
            &push_constants as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>(),
        )
    };

    let spv_path = "../../assets/sim/p3-4_imex_rigidbody_imr.comp.spv";
    let dispatch_x = (test_data.total_bodies + 31) / 32; // local_size_x is 32

    run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

    let output_bodies: Vec<f32> = ctx.read_buffer(rb_buffer, &mut rb_alloc, test_data.rigid_bodies.len());

    ctx.destroy_buffer(rb_buffer, rb_alloc);
    ctx.destroy_buffer(emitters_buffer, emitters_alloc);

    for (i, (actual, expected)) in output_bodies.iter().zip(test_data.expected_rigid_bodies.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-3, "Mismatch at index {}: actual {}, expected {}", i, actual, expected);
    }
}
