# KTX2 File Format (version 4.3.2, standard revision 3)

## Command line creation tool

To create a cubemap (assuming the order is the same as the `basisu` command line tool)
The order must be:
- +X (right)
- -X (left)
- +Y (top)
- -Y (bottom)
- +Z (front)
- -Z (back)

```shell
ktx create --format R8G8B8_SRGB --runtime-mipmap --cubemap posx.png negx.png posy.png negy.png posz.png negz.png starry-night.ktx2
```

This will create an uncompressed version of our texture. We'd like to create 2
compressed versions for consumption in our application

- A Block Compressed Version (`VK_FORMAT_BC*``) (*Desktop*)
- A version using Ericsson Texture Compression (`VK_FORMAT_ETC2_*``) (*Mobile*)

To do that, we need to compress our texture to a common format, from which
both compression formats are easy to derive at runtime

The KTX-Software package provides us with two "Supercompression Schemes",
which are `basis-lz` and `uastc`

```shell
ktx encode --codec basis-lz --clevel 4 <input> <output>
```

```shell
ktx encode --codec uastc --uastc-quality 0 --uastc-rdo --uastc-rdo-d 8192 <input> <output>
```

What we found with a test cubemap whose six faces contain Van Gogh's "Starry Night"

```powershell
Name                          MB
----                          --
starry-night-basis-lz.ktx2  0.87
starry-night-uastc.ktx2     6.00
starry-night.ktx2          18.00
```

Then, once the compressed KTX2 Texture has been created, we can load it inside our
program with one of the `ktxTexture2_Create*` function, and transcode it according
to our Vulkan capable device features. Example:

```cpp
VkPhysicalDeviceFeatures deviceFeatures = getFeatures();
ktxTexture2* texture = nullptr;
ktxTexture2_CreateFromNamedFile("texture.ktx2", KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);

if (deviceFeatures.textureCompressionBC) {
  ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, 0);
} else if (deviceFeatures.textureCompressionETC2) {
  ktxTexture2_TranscodeBasis(texture, KTX_TTF_ETC2_RGBA, 0);
}
```
