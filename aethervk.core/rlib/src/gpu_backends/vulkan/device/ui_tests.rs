use super::*;
use crate::gpu::{RenderFrontend, ScopedCommandBuffer};
use crate::gpu_backends::vulkan::device::test_utils::{setup_assets_dir, setup_render_frontend_for_tests};

#[test]
fn test_ui_billboard_apis() {
    setup_assets_dir();
    let (pool_arc, render_frontend, device_handle, pe_handle) = setup_render_frontend_for_tests(true);
    let pe_handle = pe_handle.unwrap();

    render_frontend.with_device(device_handle, |device| {
        let cmd_buffer = device.get_command_buffer().unwrap();
        device.set_command_buffer_presentation_engine(cmd_buffer, pe_handle).unwrap();
        
        {
            let _scoped_cmd = crate::gpu::ScopedCommandBuffer::new(device, cmd_buffer, None).unwrap();
            
            let create_res = device.create_billboard_resources(cmd_buffer, pe_handle);
            assert!(create_res.is_ok(), "Failed to create billboard resources");
            
            let get_res = device.get_billboard_resources(pe_handle);
            assert!(get_res.is_ok(), "Failed to get billboard resources");
            
            let tex_data = vec![255u8; 4 * 16 * 16];
            let tex = crate::simulation::comet::Texture {
                width: 16,
                height: 16,
                data: bytes::Bytes::from(tex_data),
                format: crate::simulation::comet::TexelFormat::R8G8B8A8_UNORM,
                has_mipmaps: false,
            };
            
            let texture_id = 0;
            let add_res = device.add_billboard_texture(cmd_buffer, texture_id, &tex, 0);
            assert!(add_res.is_ok(), "Failed to add billboard texture");
            
            let internal_tex_id = add_res.unwrap();
            
            let check_res = device.check_billboard_texture_id(internal_tex_id as u64);
            assert!(check_res.is_ok(), "Failed to check billboard texture id");
            
            let bind_res = device.prepare_billboard_archetype_for_render_and_bind_pipeline(cmd_buffer);
            assert!(bind_res.is_ok(), "Failed to bind billboard pipeline");
        }
        
        Ok(())
    }).unwrap();

    render_frontend.with_device(device_handle, |device| {
        device.destroy_presentation_engine(pe_handle).unwrap();
        Ok(())
    }).unwrap();
}
