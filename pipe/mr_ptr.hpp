#pragma once
// Normalize a dispatcher's get_mr() to a memory_resource pointer. CUDA dispatchers
// return get_mr() by REFERENCE, Vulkan dispatchers by POINTER; AppData constructors
// take a pointer. as_mr_ptr() accepts either, so make_dataset / the baseline driver
// are identical across backends without touching the ~40 &disp.get_mr() call sites a
// full get_mr() signature change would require.
namespace bt_pipe {
template <class T>
inline T* as_mr_ptr(T* p) {
  return p;
}
template <class T>
inline T* as_mr_ptr(T& r) {
  return &r;
}
}  // namespace bt_pipe
