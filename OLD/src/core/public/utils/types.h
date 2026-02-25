#pragma once

#include <glm/vec3.hpp>

namespace avk {

struct SphericalBB {
  glm::vec3 Center;
  float Radius;
};

}  // namespace avk