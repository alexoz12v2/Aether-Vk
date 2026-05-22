#extension GL_EXT_debug_printf : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(constant_id = 10) const uint DEBUG_SHADERS = 1;

// Usage in shaders:
//    if (SHOULD_DEBUG_PRINT) {
//        debugPrintfEXT("My values are: %f, %d", floatVal, intVal);
//    }

// Macro that evaluates to true only for the first active invocation in the subgroup
#ifndef SHOULD_DEBUG_PRINT
#define SHOULD_DEBUG_PRINT false
#endif