# Texture Notes

## Possible Formats to support

| Format                      | Library (vcpkg port) |
| :-------------------------- | :------------------- |
| PNG                         | `libpng`             |
| BMP/TGA/PPM/etc. (fallback) | `stb`                |

Possble todo in the future

| Format      | Library (vcpkg port) |
| :---------- | :------------------- |
| JPEG        | `libwebp-turbo`      |
| WebP        | `libwebp`            |
| HEIF / HEIC | `libheif`            |
| AVIF        | `libavif`            |

## Desktop only

KTX Library (TODO later)

## What happens on vulkan

- load image from disk and decode it (see steps before)
- declare a
