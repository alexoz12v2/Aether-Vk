struct VSInput {
   float3 Position : POSITION0;
   float3 Color;
};

struct VSOutput {
  float4 Position : SV_Position;
  float3 Color;
};

[shader("vertex")]
VSOutput vertexEntry(VSInput input) {
  VSOutput output;
  output.Position = float4(input.Position, 1.f);
  output.Color = input.Color;
  return output;
}

[shader("fragment")]
float4 fragmentEntry(VSOutput input) {
  return float4(input.Color, 1.f);
}
