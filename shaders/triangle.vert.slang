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

struct PushConstant {
  float scaleFactor;
  float colorAdd;
};

[[vk::push_constant]]
ConstantBuffer<PushConstant> pc;

[[vk::binding(0, 0)]]
ConstantBuffer<ModelViewProj> uboMVP;

[shader("vertex")]
VSOutput main(VSInput input) {
  VSOutput output;

  // Create a uniform scale matrix using the push constant
  float4x4 scaleMat = float4x4(
    pc.scaleFactor, 0.0f,         0.0f,         0.0f,
    0.0f,         pc.scaleFactor, 0.0f,         0.0f,
    0.0f,         0.0f,         pc.scaleFactor, 0.0f,
    0.0f,         0.0f,         0.0f,         1.0f
  );

  // Apply scale before model
  float4x4 modelScaled = mul(uboMVP.model, scaleMat);

  output.Position = mul(
    uboMVP.proj, mul(uboMVP.view, mul(modelScaled, float4(input.Position, 1.f)))
  );
  output.Color = input.Color;
  return output;
}
