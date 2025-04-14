#pragma once

#include <glm/glm.hpp>

namespace tree {

namespace omp {

constexpr auto morton_bits = 30;

// ---------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------

[[nodiscard]] constexpr uint32_t morton3D_SplitBy3bits(const uint32_t a) {
  auto x = static_cast<uint32_t>(a) & 0x000003ff;
  x = (x | x << 16) & 0x30000ff;
  x = (x | x << 8) & 0x0300f00f;
  x = (x | x << 4) & 0x30c30c3;
  x = (x | x << 2) & 0x9249249;
  return x;
}

[[nodiscard]] constexpr uint32_t m3D_e_magicbits(const uint32_t x,
                                                 const uint32_t y,
                                                 const uint32_t z) {
  return morton3D_SplitBy3bits(x) | (morton3D_SplitBy3bits(y) << 1) |
         (morton3D_SplitBy3bits(z) << 2);
}

[[nodiscard]] constexpr uint32_t xyz_to_morton32(const glm::vec4 &xyz,
                                                 const float min_coord,
                                                 const float range) {
  constexpr auto bit_scale = 1024;
  const auto i = static_cast<uint32_t>((xyz.x - min_coord) / range * bit_scale);
  const auto j = static_cast<uint32_t>((xyz.y - min_coord) / range * bit_scale);
  const auto k = static_cast<uint32_t>((xyz.z - min_coord) / range * bit_scale);
  return m3D_e_magicbits(i, j, k);
}

// ---------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------

[[nodiscard]] constexpr uint32_t morton3D_GetThirdBits(const uint32_t m) {
  auto x = m & 0x9249249;
  x = (x ^ (x >> 2)) & 0x30c30c3;
  x = (x ^ (x >> 4)) & 0x0300f00f;
  x = (x ^ (x >> 8)) & 0x30000ff;
  x = (x ^ (x >> 16)) & 0x000003ff;
  return x;
}

inline void m3D_d_magicbits(const uint32_t m, uint32_t *xyz) {
  xyz[0] = morton3D_GetThirdBits(m);
  xyz[1] = morton3D_GetThirdBits(m >> 1);
  xyz[2] = morton3D_GetThirdBits(m >> 2);
}

inline void morton32_to_xyz(glm::vec4 *ret,
                            const uint32_t code,
                            const float min_coord,
                            const float range) {
  constexpr auto bit_scale = 1024.0f;

  uint32_t dec_raw_x[3];
  m3D_d_magicbits(code, dec_raw_x);

  const auto dec_x = (static_cast<float>(dec_raw_x[0]) / bit_scale) * range + min_coord;
  const auto dec_y = (static_cast<float>(dec_raw_x[1]) / bit_scale) * range + min_coord;
  const auto dec_z = (static_cast<float>(dec_raw_x[2]) / bit_scale) * range + min_coord;

  (*ret)[0] = dec_x;
  (*ret)[1] = dec_y;
  (*ret)[2] = dec_z;
  (*ret)[3] = 1.0f;
}

}  // namespace omp

}  // namespace tree