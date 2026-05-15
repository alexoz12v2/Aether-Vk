import re

def rewrite_gpu_rs():
    with open('src/gpu.rs', 'r') as f:
        content = f.read()

    funcs_to_change = [
        "upload_particle_systems", "upload_particle2_systems", "upload_trajectories", "upload_ui", "upload_text2",
        "draw_particle_indirect", "draw_particle2_indirect", "bind_buffers", "push_constants_raw", "update_sun",
        "prepare_sun_for_render", "prepare_sky_for_render", "render_text", "render_minimap", "render_ui_rect"
    ]

    for func in funcs_to_change:
        # We find `fn func(`
        # If it doesn't have `handle: PresentationEngineHandle`, we add it after `cmd_buffer: CommandBufferHandle` or similar.
        # It's easier to just use regex
        pattern = re.compile(rf'(fn {func}[<A-Za-z0-9_:\s,>]*\(\s*&self,\s*cmd_buffer: (?:crate::)?gpu::CommandBufferHandle,?)')
        replacement = r'\1 handle: PresentationEngineHandle,'
        # wait, sometimes it's super::CommandBufferHandle or CommandBufferHandle
        pattern2 = re.compile(rf'(fn {func}[<A-Za-z0-9_:\s,>]*\(\s*&self,\s*cmd_buffer: CommandBufferHandle,?)')
        
        content = pattern.sub(replacement, content)
        content = pattern2.sub(replacement, content)

    with open('src/gpu.rs', 'w') as f:
        f.write(content)

rewrite_gpu_rs()
