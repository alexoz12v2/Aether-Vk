// types for input and output in vertex shader/fragment shader

struct VSInput {
  float3 Position : Position0;
}

struct VSOutput {
  float4 Position : SV_Position;
  float3 Color;
}

struct PSOutput {
  float4 Color : SV_Target0;
}

// types for buffers

struct CubeFaceColorMappingUBO {
  uint FaceMap[8][12];
  float3 Colors[6];
}

struct ModelMatrix {
  float4x4 Model;
}

struct CameraPushConstant {
  float4x4 View;
  float4x4 Projection;
}

[vk::binding(0, 0)]
ConstantBuffer<CubeFaceColorMappingUBO> cbCubeFaceColorMappingUBO;

[vk::binding(1, 0)]
ConstantBuffer<ModelMatrix> cbModelMatrix;

[vk::push_constant]
ConstantBuffer<CameraPushConstant> pcCameraPushConstant;

[shader("vertex")]
VSOutput vertexEntry(VSInput input, uint baseVertex: SV_StartVertexLocation, uint baseInstance : SV_StartInstanceLocation) {
  VSOutput output;
  // careful on buffer access. Out-of-bounds is a very common source of NO_BUFFER_AVAILABLE / GPU hangs.
  output.Color = cbCubeFaceColorMappingUBO.Colors[cbCubeFaceColorMappingUBO.FaceMap[baseVertex % 8][baseInstance % 12]];
  output.Position = mul(pcCameraPushConstant.Projection, mul(pcCameraPushConstant.View, mul(cbModelMatrix.Model, float4(input.Position, 1))));
  return output;
}

[shader("fragment")]
PSOutput fragmentEntry(VSOutput input) {
 PSOutput output;
 output.Color = {input.Color, 1};
 return output;
}