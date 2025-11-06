// fragment_inline_uniform_block.slang
// Slang (HLSL superset targeting Vulkan SPIR-V)

[[vk::binding(0, 0)]]
cbuffer InlineUniformBlock
{
    float4 color;    // 16 bytes — must obey std140-like packing
};

// specialization constant with a given id
[vk::constant_id(1)]
const int myConst = 1;

[shader("fragment")]
float4 main() : SV_Target0
{
    return color;
}
