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

// SV_StartVertexLocation -> First Vertex of the Draw call
// SV_StartInstanceLocation -> Always zero on non-instanced draw calls
// SV_PrimitiveID -> Available only on fragment (Geo and Tess)

[shader("vertex")]
VSOutput vertexEntry(VSInput input) {
  VSOutput output;
  // careful on buffer access. Out-of-bounds is a very common source of NO_BUFFER_AVAILABLE / GPU hangs.
  output.Position = mul(pcCameraPushConstant.Projection, mul(pcCameraPushConstant.View, mul(cbModelMatrix.Model, float4(input.Position, 1))));
  return output;
}

[shader("fragment")]
PSOutput fragmentEntry(VSOutput input, float4 pointCoord : SV_PointCoord) {
 PSOutput output;
 int index = int(pointCoord.x + pointCoord.y);
 output.Color = {cbCubeFaceColorMappingUBO.Colors[(index / 32) % 6], 1};
 return output;
}