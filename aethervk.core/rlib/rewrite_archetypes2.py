import re

def fix_archetypes():
    with open('src/gpu_backends/vulkan/device.rs', 'r') as f:
        content = f.read()

    # 1. Fix create_*_archetype functions
    # They have:
    # let presentation_engine_state = presentation_engine_lock.read();
    # let mut write_pipeline = self.pipeline_pool.write();
    # self.archetypes.create_...

    pattern = re.compile(
        r'let presentation_engine_state = presentation_engine_lock\.read\(\);\n\s*let mut write_pipeline = self\.pipeline_pool\.write\(\);\n\s*self\.archetypes\.create_',
        re.MULTILINE
    )
    
    replacement = """let mut presentation_engine_state = presentation_engine_lock.write();
    let mut write_pipeline = self.pipeline_pool.write();
    presentation_engine_state.archetypes_mut().create_"""

    content = pattern.sub(replacement, content)

    # Some functions might not have `let mut write_pipeline = ...`
    pattern2 = re.compile(
        r'let presentation_engine_state = presentation_engine_lock\.read\(\);\n\s*self\.archetypes\.create_',
        re.MULTILINE
    )
    
    replacement2 = """let mut presentation_engine_state = presentation_engine_lock.write();
    presentation_engine_state.archetypes_mut().create_"""

    content = pattern2.sub(replacement2, content)

    # 2. Fix has_discardables
    old_has = """  fn has_discardables(&self) -> bool {
    self.archetypes.has_discardables()
      || self.physical_mesh_resources.read().is_some()"""
      
    new_has = """  fn has_discardables(&self) -> bool {
    let mut archetypes_have_discardables = false;
    for pe in self.live_presentation_engines.read().values() {
      if pe.read().archetypes().has_discardables() {
        archetypes_have_discardables = true;
        break;
      }
    }
    archetypes_have_discardables
      || self.physical_mesh_resources.read().is_some()"""

    content = content.replace(old_has, new_has)

    # 3. Fix DeviceResources::cleanup
    content = content.replace("self.archetypes.discard(device, &self.discard_pool);", "")

    # 4. Fix other uses of res_guard.archetypes. or res.archetypes.
    # We will search for all `.archetypes.xxx_render_archetype`
    # e.g., `let archetype_guard = res_guard.archetypes.physical_mesh_render_archetype.read();`
    # Since we are inside functions that often DO NOT have `handle`, this is a problem!
    # Let's save the progress first.

    with open('src/gpu_backends/vulkan/device.rs', 'w') as f:
        f.write(content)

fix_archetypes()
