#pragma once

// Minimal, dependency-free NumPy .npy reader (format v1/v2). Supports exactly
// what the exported AlexNetCIFAR weights need: little-endian '<f4' / '<i4',
// C-order (fortran_order rejected). Every failure throws std::runtime_error
// naming the file and the reason -- callers rely on this to fail loud instead
// of silently falling back to synthetic data.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt::npy {

namespace detail {

[[noreturn]] inline void fail(const std::string& path, const std::string& why) {
  throw std::runtime_error("npy: " + path + ": " + why);
}

inline std::string shape_str(const std::vector<size_t>& s) {
  std::string out = "(";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + ")";
}

// Value of a quoted header entry, e.g. key "'descr'" in "{'descr': '<f4', ...}".
inline std::string quoted_value(const std::string& hdr, const char* key, const std::string& path) {
  const size_t k = hdr.find(key);
  if (k == std::string::npos) fail(path, std::string("header missing ") + key);
  const size_t q1 = hdr.find('\'', k + std::strlen(key));
  const size_t q2 = (q1 == std::string::npos) ? std::string::npos : hdr.find('\'', q1 + 1);
  if (q2 == std::string::npos) fail(path, std::string("malformed header value for ") + key);
  return hdr.substr(q1 + 1, q2 - q1 - 1);
}

}  // namespace detail

// Load the payload of a .npy file into `dst`. `descr` is the required numpy
// dtype string ("<f4" or "<i4"); `expected` is the required shape. `dst` must
// hold product(expected) elements of 4 bytes. Throws std::runtime_error (with
// the path and reason) on any IO / format / dtype / order / shape mismatch.
inline void load(const std::string& path,
                 const std::string& descr,
                 const std::vector<size_t>& expected,
                 void* dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f) detail::fail(path, "cannot open file");

  unsigned char pre[8];  // \x93NUMPY + major + minor
  if (!f.read(reinterpret_cast<char*>(pre), 8) || std::memcmp(pre, "\x93NUMPY", 6) != 0)
    detail::fail(path, "not an NPY file (bad magic)");
  const int major = pre[6];
  size_t header_len = 0;
  if (major == 1) {
    unsigned char b[2];
    if (!f.read(reinterpret_cast<char*>(b), 2)) detail::fail(path, "truncated header length");
    header_len = static_cast<size_t>(b[0]) | static_cast<size_t>(b[1]) << 8;
  } else if (major == 2 || major == 3) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char*>(b), 4)) detail::fail(path, "truncated header length");
    header_len = static_cast<size_t>(b[0]) | static_cast<size_t>(b[1]) << 8 |
                 static_cast<size_t>(b[2]) << 16 | static_cast<size_t>(b[3]) << 24;
  } else {
    detail::fail(path, "unsupported NPY version " + std::to_string(major));
  }

  std::string hdr(header_len, '\0');
  if (!f.read(hdr.data(), static_cast<std::streamsize>(header_len)))
    detail::fail(path, "truncated header");

  const std::string got_descr = detail::quoted_value(hdr, "'descr'", path);
  if (got_descr != descr)
    detail::fail(path, "dtype is '" + got_descr + "', expected '" + descr + "'");

  const size_t fo = hdr.find("'fortran_order'");
  if (fo == std::string::npos) detail::fail(path, "header missing 'fortran_order'");
  if (hdr.find("False", fo) == std::string::npos)
    detail::fail(path, "fortran_order arrays are not supported (C-order only)");

  const size_t sk = hdr.find("'shape'");
  const size_t p1 = (sk == std::string::npos) ? std::string::npos : hdr.find('(', sk);
  const size_t p2 = (p1 == std::string::npos) ? std::string::npos : hdr.find(')', p1);
  if (p2 == std::string::npos) detail::fail(path, "header missing 'shape'");
  std::vector<size_t> shape;
  size_t pos = p1 + 1;
  while (pos < p2) {
    if (hdr[pos] >= '0' && hdr[pos] <= '9') {
      size_t end = 0;
      shape.push_back(std::stoull(hdr.substr(pos, p2 - pos), &end));
      pos += end;
    } else {
      ++pos;
    }
  }
  if (shape != expected)
    detail::fail(
        path, "shape is " + detail::shape_str(shape) + ", expected " + detail::shape_str(expected));

  size_t count = 1;
  for (const size_t d : expected) count *= d;
  if (!f.read(static_cast<char*>(dst), static_cast<std::streamsize>(count * 4)))
    detail::fail(path, "truncated payload (expected " + std::to_string(count * 4) + " bytes)");
}

}  // namespace bt::npy
