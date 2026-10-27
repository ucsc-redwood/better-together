#pragma once
// Test helpers for runtime/placement.hpp (memory-placement spec, design B3), generic over
// the AppData: an app's test instantiates them with its own types.
//
// ExpectCpuBuffersStayBoundary<AppData>(n): for every schedule that runs one contiguous
// stage range on the GPU and the rest on CPU chunks, every buffer a CPU stage (or host
// code) is declared to touch is allocated from the boundary class. Sparse host reads
// (bt::kHostSparse) do not count, unless no stage touches the buffer.
//
// ExpectPlacementUnchanged<AppData>(n): for an AppData that declares no sparse host reads,
// every buffer under every such schedule gets the class of the rule before bt::kHostSparse
// existed (GPU class iff every accessor is a GPU stage).
//
// RecordingResource: tags each allocation with the stages its buffer declared, so a
// hardware test can compare the declarations with what a dispatcher actually binds.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory_resource>
#include <set>
#include <string>
#include <vector>

#include "runtime/placement.hpp"

namespace bt_placement_test {

struct Allocation {
  bt::BufferClass cls = bt::BufferClass::kBoundary;
  bt::BufferKind kind = bt::BufferKind::kActivation;
  std::vector<int> stages;
};

// Forwards to `upstream`, remembering which class/kind/stages each pointer was placed for.
class RecordingResource final : public std::pmr::memory_resource {
 public:
  RecordingResource(std::pmr::memory_resource* upstream,
                    const bt::BufferClass cls,
                    std::map<const void*, Allocation>* log)
      : upstream_(upstream), cls_(cls), log_(log) {}

  // Set by the placement right before the buffer it returns this resource for allocates.
  bt::BufferKind next_kind = bt::BufferKind::kActivation;
  std::vector<int> next_stages;

 private:
  void* do_allocate(const std::size_t bytes, const std::size_t alignment) override {
    void* p = upstream_->allocate(bytes, alignment);
    (*log_)[p] = Allocation{cls_, next_kind, next_stages};
    return p;
  }
  void do_deallocate(void* p, const std::size_t bytes, const std::size_t alignment) override {
    upstream_->deallocate(p, bytes, alignment);
  }
  [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  std::pmr::memory_resource* upstream_;
  bt::BufferClass cls_;
  std::map<const void*, Allocation>* log_;
};

// A dispatcher stand-in with the three buffer classes, each a RecordingResource over
// `upstream` (default: plain host memory).
struct RecordingDispatcher {
  explicit RecordingDispatcher(
      std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      : boundary(upstream, bt::BufferClass::kBoundary, &log),
        weights(upstream, bt::BufferClass::kWeights, &log),
        gpu_local(upstream, bt::BufferClass::kGpuLocal, &log) {}

  std::pmr::memory_resource* get_mr(const bt::BufferClass c) { return &of(c); }
  RecordingResource& of(const bt::BufferClass c) {
    return c == bt::BufferClass::kWeights    ? weights
           : c == bt::BufferClass::kGpuLocal ? gpu_local
                                             : boundary;
  }

  // schedule_placement, also telling the chosen resource what it is about to allocate.
  bt::PlaceBuffer placement(const std::set<int>& gpu_stages) {
    const bt::PlaceBuffer inner = schedule_placement(*this, gpu_stages);
    return [this, inner](const bt::BufferKind kind, const std::initializer_list<int> stages) {
      std::pmr::memory_resource* mr = inner(kind, stages);
      for (RecordingResource* r : {&boundary, &weights, &gpu_local}) {
        if (r == mr) {
          r->next_kind = kind;
          r->next_stages.assign(stages);
        }
      }
      return mr;
    };
  }

  std::map<const void*, Allocation> log;
  RecordingResource boundary, weights, gpu_local;
};

// Calls check(gpu, alloc, where) for every buffer of an AppData placed for each schedule
// that runs one contiguous stage range on the GPU.
template <class AppData, class Check>
void ForEachOneRangePlacement(const int n_stages, const Check& check) {
  for (int a = 1; a <= n_stages; ++a) {
    for (int b = a; b <= n_stages; ++b) {
      const std::set<int> gpu = [&] {
        std::set<int> g;
        for (int s = a; s <= b; ++s) g.insert(s);
        return g;
      }();
      RecordingDispatcher disp;
      {
        const AppData app(disp.placement(gpu));
      }  // allocate, log, free
      ASSERT_FALSE(disp.log.empty());
      const std::string where = "GPU stages " + std::to_string(a) + "-" + std::to_string(b);
      for (const auto& [ptr, alloc] : disp.log) check(gpu, alloc, where);
    }
  }
}

[[nodiscard]] inline bt::BufferClass gpu_class_of(const bt::BufferKind kind) {
  return kind == bt::BufferKind::kWeights ? bt::BufferClass::kWeights : bt::BufferClass::kGpuLocal;
}

template <class AppData>
void ExpectCpuBuffersStayBoundary(const int n_stages) {
  ForEachOneRangePlacement<AppData>(
      n_stages, [](const std::set<int>& gpu, const Allocation& alloc, const std::string& where) {
        bool cpu_touched = false, gpu_touched = false, sparse = false;
        for (const int s : alloc.stages) {
          if (s == bt::kHostSparse) {
            sparse = true;
          } else {
            (gpu.contains(s) ? gpu_touched : cpu_touched) = true;
          }
        }
        if (cpu_touched || (sparse && !gpu_touched)) {
          EXPECT_EQ(alloc.cls, bt::BufferClass::kBoundary) << where;
        } else {
          EXPECT_EQ(alloc.cls, gpu_class_of(alloc.kind)) << where;
        }
      });
}

template <class AppData>
void ExpectPlacementUnchanged(const int n_stages) {
  ForEachOneRangePlacement<AppData>(
      n_stages, [](const std::set<int>& gpu, const Allocation& alloc, const std::string& where) {
        ASSERT_EQ(std::ranges::count(alloc.stages, bt::kHostSparse), 0) << "declares the marker";
        const bool gpu_only =
            std::ranges::all_of(alloc.stages, [&](const int s) { return gpu.contains(s); });
        EXPECT_EQ(alloc.cls, gpu_only ? gpu_class_of(alloc.kind) : bt::BufferClass::kBoundary)
            << where;
      });
}

}  // namespace bt_placement_test
