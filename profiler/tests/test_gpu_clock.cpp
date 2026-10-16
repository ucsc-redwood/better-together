// read_gpu_clock_mhz (profiler/gpu_clock.hpp) on fake sysfs trees: the contents and units of
// each driver's node, as read on the fleet (Pixel 7a and Samsung: 2026-10-16). Hardware-free.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "profiler/gpu_clock.hpp"

namespace {

namespace fs = std::filesystem;

// A fresh, empty fake root per test; files are written relative to it.
class GpuClock : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() /
            ("bt-gpu-clock-" +
             std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(root_);
    fs::create_directories(root_);
  }
  void TearDown() override { fs::remove_all(root_); }

  void write(const std::string& path, const std::string& contents) const {
    const fs::path p = root_ / path.substr(1);
    fs::create_directories(p.parent_path());
    std::ofstream(p) << contents;
  }
  [[nodiscard]] int read() const { return bt_prof::read_gpu_clock_mhz(root_.string()); }

  fs::path root_;
};

TEST_F(GpuClock, NoSourceIsUnknown) { EXPECT_EQ(read(), -1); }

TEST_F(GpuClock, AmdActiveDpmLine) {
  write("/sys/class/drm/card1/device/pp_dpm_sclk", "0: 800Mhz\n1: 1500Mhz *\n2: 2700Mhz\n");
  EXPECT_EQ(read(), 1500);
}

TEST_F(GpuClock, TegraDevfreqInHz) {
  write("/sys/devices/platform/gpu/devfreq/gpu/cur_freq", "1020000000\n");
  EXPECT_EQ(read(), 1020);
}

TEST_F(GpuClock, MaliCurFreqInKhz) {  // Pixel 7a, idle
  write("/sys/class/misc/mali0/device/cur_freq", "202000\n");
  EXPECT_EQ(read(), 202);
}

TEST_F(GpuClock, ExynosGpuClockInKhz) {  // Samsung, busy
  write("/sys/kernel/gpu/gpu_clock", "1095000\n");
  EXPECT_EQ(read(), 1095);
}

TEST_F(GpuClock, ExynosSgpuDevfreqInKhz) {
  write("/sys/kernel/gpu/gpu_clock", "0\n");                               // idle reading, skipped
  write("/sys/class/devfreq/17000010.devfreq_mif/cur_freq", "2093000\n");  // not the GPU
  write("/sys/class/devfreq/22200000.sgpu/cur_freq", "545000\n");
  EXPECT_EQ(read(), 545);
}

TEST_F(GpuClock, IdlePowerGatedZeroIsUnknown) {  // Samsung, GPU idle
  write("/sys/kernel/gpu/gpu_clock", "0\n");
  write("/sys/class/devfreq/22200000.sgpu/cur_freq", "0\n");
  EXPECT_EQ(read(), -1);
}

TEST_F(GpuClock, GarbageIsUnknown) {
  write("/sys/class/misc/mali0/device/cur_freq", "n/a\n");
  EXPECT_EQ(read(), -1);
}

// Not a check: the cost of one read on the device this runs on (design D5's stride), printed.
TEST(GpuClockDevice, ReadCost) {
  if (bt_prof::read_gpu_clock_mhz() < 0) GTEST_SKIP() << "no readable GPU clock here";
  constexpr int kReads = 2000;
  const auto t0 = std::chrono::steady_clock::now();
  int sink = 0;
  for (int i = 0; i < kReads; ++i) sink += bt_prof::read_gpu_clock_mhz();
  const double us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
      kReads;
  std::printf("read_gpu_clock_mhz: %.2f us per call (%d reads, last %d MHz)\n",
              us,
              kReads,
              bt_prof::read_gpu_clock_mhz());
  EXPECT_GT(sink, 0);
}

}  // namespace
