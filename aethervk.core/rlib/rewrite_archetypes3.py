import re

def main():
    with open("src/gpu_backends/vulkan/device.rs", "r") as f:
        content = f.read()

    # 1. Update function signatures in device.rs to match gpu.rs
    funcs_to_change = [
        "upload_particle_systems", "upload_particle2_systems", "upload_trajectories", "upload_ui", "upload_text2",
        "draw_particle_indirect", "draw_particle2_indirect", "bind_buffers", "push_constants_raw", "update_sun",
        "prepare_sun_for_render", "prepare_sky_for_render", "render_text", "render_minimap", "render_ui_rect",
        "prepare_background_archetype_for_render_and_bind_pipeline"
    ]

    for func in funcs_to_change:
        pattern = re.compile(rf'(fn {func}[<A-Za-z0-9_:\s,>]*\(\s*&self,\s*cmd_buffer: (?:crate::)?gpu::CommandBufferHandle,?)')
        replacement = r'\1 handle: PresentationEngineHandle,'
        pattern2 = re.compile(rf'(fn {func}[<A-Za-z0-9_:\s,>]*\(\s*&self,\s*cmd_buffer: CommandBufferHandle,?)')
        content = pattern.sub(replacement, content)
        content = pattern2.sub(replacement, content)
        # deduplicate
        content = re.sub(
            r"cmd_buffer:\s*CommandBufferHandle,\s*handle:\s*PresentationEngineHandle,\n\s*handle:\s*PresentationEngineHandle,",
            r"cmd_buffer: CommandBufferHandle,\n    handle: PresentationEngineHandle,",
            content
        )
        content = content.replace("cmd_buffer: gpu::CommandBufferHandle handle: PresentationEngineHandle,", "cmd_buffer: gpu::CommandBufferHandle, handle: PresentationEngineHandle,")
        content = content.replace("cmd_buffer: CommandBufferHandle handle: PresentationEngineHandle,", "cmd_buffer: CommandBufferHandle, handle: PresentationEngineHandle,")


    # 2. Replace archetype accesses
    # We will do this by looking for `res_guard.archetypes.` or `res.archetypes.` or `self.archetypes.`
    # and replacing it with `archetypes.`
    # And at the top of the function (or right before the access), we inject:
    # let live_engines_xxx = res_guard.live_presentation_engines.read();
    # let pe_xxx = live_engines_xxx.get(&handle).unwrap().read();
    # let archetypes = pe_xxx.archetypes();
    
    # Actually, simpler: replace `res_guard.archetypes.` with `pe_xxx.archetypes().`
    # We can write a regex that matches `res_guard.archetypes.([a-z0-9_]+_render_archetype)`
    # and replaces with `pe_xxx.archetypes().\1`
    
    # We need to make sure `pe_xxx` is defined.
    # To do this safely, let's just do it manually for the file using a line-by-line approach where we track the current function.
    
    lines = content.split('\n')
    out_lines = []
    
    current_func = ""
    for i, line in enumerate(lines):
        if "fn " in line and "{" in line or "fn " in line and line.strip().endswith("("):
            m = re.search(r'fn ([a-z0-9_]+)', line)
            if m:
                current_func = m.group(1)
                
        # Skip create_*_archetype as we already fixed them in rewrite_archetypes2.py
        if current_func.startswith("create_") and current_func.endswith("_archetype"):
            out_lines.append(line)
            continue
            
        if ".archetypes." in line:
            # Handle special global functions first
            if current_func in ["allocate_rasterized_font_atlas", "free_rasterized_font_atlas", "add_billboard_texture"]:
                # For these, we can't easily fix them line-by-line because they need to loop.
                # We will leave a marker and fix them in step 3.
                pass
            else:
                # Normal functions with `handle`
                # If it's a `write()`, we need `pe_xxx.archetypes_mut()`
                
                # Check if we already injected pe_xxx in this function
                # We'll just inject it inline inside a block or assume we can just do:
                # `res_guard.live_presentation_engines.read().get(&handle).unwrap().read().archetypes()`
                # `res_guard.live_presentation_engines.read().get(&handle).unwrap().write().archetypes_mut()`
                
                # What if the variable is `res` instead of `res_guard`?
                if "res." in line:
                    res_var = "res"
                else:
                    res_var = "res_guard"
                    
                # Replace reads
                line = re.sub(
                    rf'{res_var}\.archetypes\.([a-z0-9_]+_render_archetype)\.read\(\)',
                    rf'{res_var}.live_presentation_engines.read().get(&handle).unwrap().read().archetypes().\1.read()',
                    line
                )
                # Replace writes
                line = re.sub(
                    rf'{res_var}\.archetypes\.([a-z0-9_]+_render_archetype)\.write\(\)',
                    rf'{res_var}.live_presentation_engines.read().get(&handle).unwrap().write().archetypes_mut().\1.write()',
                    line
                )
                
                # Replace method calls like update_..._for_presentation_engine
                line = re.sub(
                    rf'{res_var}\.archetypes\.update_',
                    rf'{res_var}.live_presentation_engines.read().get(&handle).unwrap().write().archetypes_mut().update_',
                    line
                )

        out_lines.append(line)

    content = "\n".join(out_lines)

    # 3. Fix the global functions
    
    # allocate_rasterized_font_atlas
    old_alloc = """    let mut archetype1 = res_guard.archetypes.text_render_archetype.write();
    if archetype1.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] allocate_rasterized_font_atlas text archetype not initialized",
      ));
    }
    archetype1.as_mut().unwrap().uploaded_fonts.insert(
      hash,
      UploadedFont {
        texture: uploaded_image1,
        descriptor_index: descriptor_index1,
        last_used_frame: 0,
      },
    );

    let mut archetype2 = res_guard.archetypes.text2_render_archetype.write();
    if archetype2.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] allocate_rasterized_font_atlas text2 archetype not initialized",
      ));
    }
    archetype2.as_mut().unwrap().uploaded_fonts.insert(
      hash,
      UploadedFont {
        texture: uploaded_image2,
        descriptor_index: descriptor_index2,
        last_used_frame: 0,
      },
    );"""
    
    new_alloc = """    for pe in res_guard.live_presentation_engines.read().values() {
      let mut pe = pe.write();
      
      let mut archetype1 = pe.archetypes_mut().text_render_archetype.write();
      if archetype1.is_none() {
        continue;
      }
      archetype1.as_mut().unwrap().uploaded_fonts.insert(
        hash,
        UploadedFont {
          texture: uploaded_image1.clone(),
          descriptor_index: descriptor_index1,
          last_used_frame: 0,
        },
      );

      let mut archetype2 = pe.archetypes_mut().text2_render_archetype.write();
      if archetype2.is_none() {
        continue;
      }
      archetype2.as_mut().unwrap().uploaded_fonts.insert(
        hash,
        UploadedFont {
          texture: uploaded_image2.clone(),
          descriptor_index: descriptor_index2,
          last_used_frame: 0,
        },
      );
    }"""
    content = content.replace(old_alloc, new_alloc)

    # free_rasterized_font_atlas
    old_free = """    let mut archetype1 = res_guard.archetypes.text_render_archetype.write();
    if archetype1.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] free_rasterized_font_atlas text archetype not initialized",
      ));
    }
    if let Some(mut uploaded) = archetype1.as_mut().unwrap().uploaded_fonts.remove(&hash) {
      uploaded.texture.discard(&self.device, &res_guard.discard_pool, u64::MAX);
    }

    let mut archetype2 = res_guard.archetypes.text2_render_archetype.write();
    if archetype2.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] free_rasterized_font_atlas text2 archetype not initialized",
      ));
    }
    if let Some(mut uploaded) = archetype2.as_mut().unwrap().uploaded_fonts.remove(&hash) {
      uploaded.texture.discard(&self.device, &res_guard.discard_pool, u64::MAX);
    }"""

    new_free = """    for pe in res_guard.live_presentation_engines.read().values() {
      let mut pe = pe.write();
      let mut archetype1 = pe.archetypes_mut().text_render_archetype.write();
      if archetype1.is_some() {
        if let Some(mut uploaded) = archetype1.as_mut().unwrap().uploaded_fonts.remove(&hash) {
          uploaded.texture.discard(&self.device, &res_guard.discard_pool, u64::MAX);
        }
      }

      let mut archetype2 = pe.archetypes_mut().text2_render_archetype.write();
      if archetype2.is_some() {
        if let Some(mut uploaded) = archetype2.as_mut().unwrap().uploaded_fonts.remove(&hash) {
          uploaded.texture.discard(&self.device, &res_guard.discard_pool, u64::MAX);
        }
      }
    }"""
    content = content.replace(old_free, new_free)

    # add_billboard_texture
    old_billboard = """    let billboard_render_archetype = res.archetypes.billboard_render_archetype.read();
    if billboard_render_archetype.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] add_billboard_texture billboard archetype not initialized",
      ));
    }

    drop(billboard_render_archetype);
    let mut billboard_render_archetype = res.archetypes.billboard_render_archetype.write();

    let archetype = billboard_render_archetype.as_mut().unwrap();"""
    
    new_billboard = """    for pe in res.live_presentation_engines.read().values() {
      let mut pe = pe.write();
      let mut billboard_render_archetype = pe.archetypes_mut().billboard_render_archetype.write();
      if billboard_render_archetype.is_none() {
        continue;
      }
      let archetype = billboard_render_archetype.as_mut().unwrap();"""
      
    content = content.replace(old_billboard, new_billboard)
    
    # We need to close the for loop for add_billboard_texture
    # Looking at the function `add_billboard_texture`, the archetype variable is used until the end.
    # It ends with `Ok(())`. So we can replace `Ok(())` at the end of that function.
    content = content.replace(
        "archetype.free_descriptor_indices.push(replaced_index);\n      }\n    }\n\n    Ok(())",
        "archetype.free_descriptor_indices.push(replaced_index);\n      }\n    }\n    }\n\n    Ok(())"
    )

    with open("src/gpu_backends/vulkan/device.rs", "w") as f:
        f.write(content)

main()
