#pragma once

// library
#include <array>
#include <glm/glm.hpp>

namespace avk::hashes {

using namespace avk::literals;
inline constexpr uint64_t Vertex = "Vertex"_hash;
inline constexpr uint64_t Index = "Index"_hash;
inline constexpr uint64_t Model = "Model"_hash;
inline constexpr uint64_t Cube = "Cube"_hash;
inline constexpr uint64_t Staging = "Staging"_hash;

}  // namespace avk::hashes

namespace avk {
// TODO render/basic-types.h
struct alignas(16) float3 {
  glm::vec3 v;
};

// stride 16 uint (note: this has been observed with disassembler)
struct alignas(16) uintArray12 {
  uint32_t i;
};

// outer array has 192 stride
struct uintArrayArray8 {
  uintArray12 is[12];
};
static_assert(sizeof(uintArrayArray8) == 192);

struct CubeFaceMapping {
  uintArrayArray8 faceMap[8];
  float3 colors[6]{};

  CubeFaceMapping(std::array<std::array<uint32_t, 12>, 8> const &_faceMap,
                  std::array<glm::vec4, 6> const &_colors)
      : faceMap{} {
    for (uint32_t i = 0; i < 8; i++) {
      for (uint32_t j = 0; j < 12; j++) {
        faceMap[i].is[j].i = _faceMap[i][j];
      }
    }
    for (uint32_t i = 0; i < 6; ++i) {
      colors[i].v = _colors[i];
    }
  }
};

struct Camera {
  glm::mat4 view;
  glm::mat4 proj;
};

}  // namespace avk

namespace avk::test {

// [vertex][triangle] → faceIndex (0–5) or UINT32_MAX (invalid)
inline void cubePrimitive(std::array<glm::vec3, 8> &vertexBuffer,
                          std::array<glm::uvec3, 12> &indexBuffer,
                          std::array<std::array<uint32_t, 12>, 8> &faceMap) {
  // Define 8 cube corners
  // 0 (-,-,-), 1 (+,-,-), 2 (+,+,-), 3 (-,+,-),
  // 4 (-,-,+), 5 (+,-,+), 6 (+,+,+), 7 (-,+,+)
  vertexBuffer[0] = {-0.5f, -0.5f, -0.5f};  // 0
  vertexBuffer[1] = {0.5f, -0.5f, -0.5f};   // 1
  vertexBuffer[2] = {0.5f, 0.5f, -0.5f};    // 2
  vertexBuffer[3] = {-0.5f, 0.5f, -0.5f};   // 3
  vertexBuffer[4] = {-0.5f, -0.5f, 0.5f};   // 4
  vertexBuffer[5] = {0.5f, -0.5f, 0.5f};    // 5
  vertexBuffer[6] = {0.5f, 0.5f, 0.5f};     // 6
  vertexBuffer[7] = {-0.5f, 0.5f, 0.5f};    // 7

  // Define 12 triangles (two per cube face)
  indexBuffer[0] = {0, 1, 2}, indexBuffer[1] = {2, 3, 0};    // back face (-Z)
  indexBuffer[2] = {4, 5, 6}, indexBuffer[3] = {6, 7, 4};    // front face (+Z)
  indexBuffer[4] = {0, 4, 7}, indexBuffer[5] = {7, 3, 0};    // left face (-X)
  indexBuffer[6] = {1, 5, 6}, indexBuffer[7] = {6, 2, 1};    // right face (+X)
  indexBuffer[8] = {3, 2, 6}, indexBuffer[9] = {6, 7, 3};    // top face (+Y)
  indexBuffer[10] = {0, 1, 5}, indexBuffer[11] = {5, 4, 0};  // bottom face (-Y)

  // Note: Usually, to achieve per face attributes, you duplicate the vertices,
  // here we are instead defining a vertex,index to face mapping
  // Initialize all entries to invalid
  for (auto &row : faceMap) row.fill(UINT32_MAX);

  // Fill valid mappings
  for (uint8_t tri = 0; tri < 12; ++tri) {
    uint8_t face = tri / 2;
    const glm::uvec3 &t = indexBuffer[tri];
    faceMap[t.x][tri] = face;
    faceMap[t.y][tri] = face;
    faceMap[t.z][tri] = face;
  }
}

// vec4 not 3 because of HLSL like alignment
inline void cubeColors(std::array<glm::vec4, 6> &outColors) {
  outColors = {glm::vec4(.5, .5, .5, 0), {.4, .1, .4, 0}, {.1, .2, .6, 0},
               {.6, .1, .0, 0},          {.1, .5, .2, 0}, {0, 1, .1, 0}};
}

}  // namespace avk::test