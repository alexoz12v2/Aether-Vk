[[vk::binding(0, 0)]]// binding = 0, set = 0
[[vk::image_format("rgba32f")]]// VK_FORMAT_R32G32B32A32_FLOAT
RWTexture2D<float4> storageImage;

// implement an image inversion operation
[shader("compute")]
[numthreads(16,16,1)] // 16 x 16 workgroup size
void main(uint3 threadId : SV_DispatchThreadID) {
  uint width, height;
  storageImage.GetDimensions(width, height);

  if (threadId.x >= width || threadId.y >= height)
    return;

  // assumes not HDR
  float4 color = storageImage[threadId.xy];
  storageImage[threadId.xy] = float4(1.f - color.rgb, color.a);
}