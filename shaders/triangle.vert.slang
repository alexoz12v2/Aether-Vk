struct VSInput {
  float3 Position : POSITION0;
  float3 Color;
};

struct VSOutput {
  float4 Position : SV_POSITION;
  float3 Color;
};

[shader("vertex")]
VSOutput main(VSInput input) {
  VSOutput output;
  output.Position = float4(input.Position, 1.f);
  output.Color = input.Color;
  return output;
}
