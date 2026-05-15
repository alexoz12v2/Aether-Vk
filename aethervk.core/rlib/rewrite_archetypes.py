import re

def fix():
    with open('src/gpu_backends/vulkan/device.rs', 'r') as f:
        content = f.read()

    # 1. Update `update_all_archetypes_for_presentation_engine`
    old_update = """    let presentation_engine_state = presentation_engine_state_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    let format = presentation_engine_state.format();

    self.archetypes.update_physical_mesh_archetype_for_presentation_engine"""
    
    new_update = """    let mut presentation_engine_state = presentation_engine_state_lock.write();
    let mut write_pipeline = self.pipeline_pool.write();
    let format = presentation_engine_state.format();

    presentation_engine_state.archetypes_mut().update_physical_mesh_archetype_for_presentation_engine"""
    
    content = content.replace(old_update, new_update)
    
    # In that function, there are 16 more calls to self.archetypes. Let's replace them specifically in that function block
    start_idx = content.find('fn update_all_archetypes_for_presentation_engine')
    end_idx = content.find('Ok(())', start_idx) + 10
    
    block = content[start_idx:end_idx]
    block = block.replace('self.archetypes.', 'presentation_engine_state.archetypes_mut().')
    content = content[:start_idx] + block + content[end_idx:]

    # 2. Update `init_archetypes`
    old_init = """  fn init_archetypes(&self, handle: crate::gpu::PresentationEngineHandle) -> GpuResult<()> {
    // TODO: remove all logs
    let res_guard = self.res.read();
    let timeline = res_guard.timeline_manager.get_cached_value() + 1;
    let mut shader_manager = res_guard.shader_manager.write();"""
    
    new_init = """  fn init_archetypes(&self, handle: crate::gpu::PresentationEngineHandle) -> GpuResult<()> {
    // TODO: remove all logs
    let res_guard = self.res.read();
    let timeline = res_guard.timeline_manager.get_cached_value() + 1;
    let mut shader_manager = res_guard.shader_manager.write();
    let live_engines = res_guard.live_presentation_engines.read();
    let mut engine_lock = live_engines.get(&handle).unwrap().write();
    let format = engine_lock.format();"""
    
    content = content.replace(old_init, new_init)
    
    start_idx = content.find('fn init_archetypes')
    end_idx = content.find('Ok(())', start_idx) + 10
    block = content[start_idx:end_idx]
    
    # In `init_archetypes`, we replace:
    # res_guard.archetypes. -> engine_lock.archetypes_mut().
    # self.res.read().archetypes. -> engine_lock.archetypes_mut().
    # and we need to pass `format` directly instead of `{ let pe_lock = ... }`
    
    block = block.replace('res_guard.archetypes.', 'engine_lock.archetypes_mut().')
    block = block.replace('self.res.read().archetypes.', 'engine_lock.archetypes_mut().')
    
    # the format block in init_archetypes looks like:
    # {
    #   let pe_lock = res_guard.live_presentation_engines.read();
    #   pe_lock.get(&handle).unwrap().read().format()
    # }
    format_block = """        {
          let pe_lock = res_guard.live_presentation_engines.read();
          pe_lock.get(&handle).unwrap().read().format()
        }"""
    block = block.replace(format_block, "        format")
    
    content = content[:start_idx] + block + content[end_idx:]

    with open('src/gpu_backends/vulkan/device.rs', 'w') as f:
        f.write(content)

fix()
