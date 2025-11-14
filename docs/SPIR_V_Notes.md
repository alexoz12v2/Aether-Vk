# SPIR-V Notes

## Extensions

### `SPV_KHR_shader_draw_parameters`

This extension adds built-in variables that give the shader access to extra information about the current draw call — things that normally only the CPU or the driver knows.

It’s especially useful when doing multi-draw indirect or instanced rendering.

When you enable this extension, you get three new built-in variables available in the shader:

| Built-in       | Type  | Description                                                                                       |
| -------------- | ----- | ------------------------------------------------------------------------------------------------- |
| `BaseVertex`   | `int` | The value added to vertex indices before fetching vertex data. (Matches `gl_BaseVertex` in GLSL.) |
| `BaseInstance` | `int` | The first instance ID for this draw call. (Matches `gl_BaseInstance` in GLSL.)                    |
| `DrawIndex`    | `int` | The index of the current draw within a multi-draw call. (Matches `gl_DrawID` in GLSL.)            |

To use it, you declare it at the top of your SPIR-V module:

```spirv
OpCapability DrawParameters
OpExtension "SPV_KHR_shader_draw_parameters"
```

Then you can declare the built-ins like so:

```spirv
%int = OpTypeInt 32 1
%baseVertex = OpVariable %_ptr_Input_int Input
OpDecorate %baseVertex BuiltIn BaseVertex
```

### Debugging with Slangc

Slang’s SPIRV backend supports generating debug information using the
NonSemantic Shader DebugInfo Instructions.
To enable debugging information when targeting SPIRV, specify
the -emit-spirv-directly and the -g2 argument when using slangc tool,
or set EmitSpirvDirectly to 1 and DebugInformation to SLANG_DEBUG_INFO_LEVEL_STANDARD when using the
API. Debugging support has been tested with RenderDoc.

### Slang built-in variables

<https://docs.shader-slang.org/en/latest/coming-from-glsl.html#built-in-variables-reference>
