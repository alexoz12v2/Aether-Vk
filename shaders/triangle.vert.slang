struct VSInput {
  float3 Position : POSITION0;
  float3 Color;
};

struct VSOutput {
  float4 Position : SV_POSITION;
  float3 Color;
};

struct ModelViewProj {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ModelViewProj> uboMVP;

[shader("vertex")]
VSOutput main(VSInput input) {
  VSOutput output;
  output.Position = mul(
    uboMVP.proj, mul(uboMVP.view, mul(uboMVP.model, float4(input.Position, 1.f))));
  output.Color = input.Color;
  return output;
}
