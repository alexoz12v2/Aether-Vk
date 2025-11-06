struct VSOutput {
  float4 Position : SV_POSITION;
  float3 Color;
};

// max push constant size = 128 bytes
struct PushConstant {
  float scaleFactor;
  float colorAdd;
};

[[vk::push_constant]]
ConstantBuffer<PushConstant> pc;

[shader("fragment")]
float4 main(VSOutput input) {
  return float4(input.Color + pc.colorAdd.xxx, 1.f);
}