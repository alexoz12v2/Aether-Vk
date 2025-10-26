struct VSOutput {
  float4 Position : SV_POSITION;
  float3 Color;
};

[shader("fragment")]
float4 main(VSOutput input) {
  return float4(input.Color, 1.f);
}