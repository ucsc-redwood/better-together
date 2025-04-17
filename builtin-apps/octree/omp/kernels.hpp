#pragma once

#include <libmorton/morton.h>

#include <cfloat>
#include <glm/vec4.hpp>

namespace octree::omp {

// ----------------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------

inline void compute_morton_codes(const glm::vec4* positions, const size_t n, uint32_t* codes_out) {
  // 1a) find bounding box via a parallel reduction
  float xmin = FLT_MAX, ymin = FLT_MAX, zmin = FLT_MAX;
  float xmax = -FLT_MAX, ymax = -FLT_MAX, zmax = -FLT_MAX;

  // #pragma omp for reduction(min : xmin, ymin, zmin) reduction(max : xmax, ymax, zmax)
  // schedule(static)
  for (size_t i = 0; i < n; ++i) {
    const glm::vec4& p = positions[i];
    xmin = fminf(xmin, p.x);
    ymin = fminf(ymin, p.y);
    zmin = fminf(zmin, p.z);
    xmax = fmaxf(xmax, p.x);
    ymax = fmaxf(ymax, p.y);
    zmax = fmaxf(zmax, p.z);
  }

#pragma omp barrier

  const float dx = xmax - xmin;
  const float dy = ymax - ymin;
  const float dz = zmax - zmin;

  // 1b) normalize, quantize to 10 bits, then encode
#pragma omp for
  for (size_t i = 0; i < n; ++i) {
    const glm::vec4& p = positions[i];

    // normalize into [0,1]
    float nx = (dx > 0.0f) ? ((p.x - xmin) / dx) : 0.0f;
    float ny = (dy > 0.0f) ? ((p.y - ymin) / dy) : 0.0f;
    float nz = (dz > 0.0f) ? ((p.z - zmin) / dz) : 0.0f;

    // quantize to 10 bits (0..1023)
    uint16_t xi = uint16_t(nx * 1023.0f);
    uint16_t yi = uint16_t(ny * 1023.0f);
    uint16_t zi = uint16_t(nz * 1023.0f);

    // libmorton call
    codes_out[i] = libmorton::morton3D_32_encode(xi, yi, zi);
  }
}

}  // namespace octree::omp
