
struct VSInput {
  float3 Position : Position0;
}

struct VSOutput {
  float4 Position : SV_Position;
  float3 WorldDir : TEXCOORD0;
}

struct PSOutput {
  float4 Color : SV_Target0;
}

struct CameraPushConstant {
  float4x4 View;
  float4x4 Projection;
}

[vk::push_constant]
ConstantBuffer<CameraPushConstant> pcCamera;

[vk::binding(0,0)]
TextureCube texCubeMap;

[vk::binding(1,0)]
SamplerState sampler;

// Note: Projection matrix should be a orthographic projection, because the cubemap
// is supposed to be far away

[shader("vertex")]
VSOutput vertexEntry(VSInput input) {
  VSOutput output;
  output.Position = mul(pcCamera.Projection, mul(pcCamera.View, float4(input.Position, 1)));
  output.WorldDir = input.Position; // assumes cube is -1,1
  return output;
}

// should execute only if
// 1. Executes before all geometry
// 2. Executes after some geometry, but found stencil buffer clear
[shader("fragment")]
PSOutput fragmentEntry(VSOutput input) {
  PSOutput output;
  float3 dir = normalize(input.WorldDir);
  // TODO: Mipmapping
  output.Color = float4(texCubeMap.SampleLevel(sampler, dir, 0).xyz, 1);
  return output;
}
