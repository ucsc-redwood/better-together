// Pinned multi-threaded streaming-read bandwidth (llm-phase-affinity, design D4).
// usage: cpu_readbw <core,core,...> <total_MiB> <duration_s>
// One thread per listed core, each pinned to its core with sched_setaffinity (the Samsung
// scheduler packs unpinned threads, see diag-ems-thread-placement.txt), each streaming its own
// share of the buffer with 128-bit loads. Every 500 ms it prints "<epoch_s> <GB/s over the
// interval>"; at the end "median_gbps <x>" over all intervals.
// Build: $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang -O3 -pthread
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static atomic_ullong bytes_read;
static atomic_int stop;
static size_t share;

struct arg {
  int core;
  uint64_t sink;
};

static void *reader(void *p) {
  struct arg *a = p;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(a->core, &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) {
    perror("sched_setaffinity");
    exit(1);
  }
  uint64_t *buf = aligned_alloc(64, share);
  memset(buf, 1, share);
  uint64x2_t s0 = vdupq_n_u64(0), s1 = s0, s2 = s0, s3 = s0;
  const size_t n = share / sizeof(uint64_t);
  while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
    for (size_t i = 0; i < n; i += 8) {
      s0 = vaddq_u64(s0, vld1q_u64(buf + i));
      s1 = vaddq_u64(s1, vld1q_u64(buf + i + 2));
      s2 = vaddq_u64(s2, vld1q_u64(buf + i + 4));
      s3 = vaddq_u64(s3, vld1q_u64(buf + i + 6));
    }
    atomic_fetch_add_explicit(&bytes_read, share, memory_order_relaxed);
  }
  uint64x2_t s = vaddq_u64(vaddq_u64(s0, s1), vaddq_u64(s2, s3));
  a->sink = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
  free(buf);
  return NULL;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <core,core,...> <total_MiB> <duration_s>\n", argv[0]);
    return 2;
  }
  int cores[16], nt = 0;
  for (char *tok = strtok(argv[1], ","); tok && nt < 16; tok = strtok(NULL, ","))
    cores[nt++] = atoi(tok);
  share = (size_t)atoi(argv[2]) * 1024 * 1024 / nt;
  share -= share % 64;
  double duration = atof(argv[3]);

  pthread_t th[16];
  struct arg args[16];
  for (int i = 0; i < nt; i++) {
    args[i].core = cores[i];
    pthread_create(&th[i], NULL, reader, &args[i]);
  }
  usleep(1000000); // buffers allocated and touched, clocks ramped

  double samples[4096];
  int ns = 0;
  double t0 = now(), tp = t0;
  unsigned long long bp = atomic_load(&bytes_read);
  while (now() - t0 < duration && ns < 4096) {
    usleep(500000);
    double t = now();
    unsigned long long b = atomic_load(&bytes_read);
    double gbps = (b - bp) / (t - tp) / 1e9;
    printf("%.3f %.3f\n", t, gbps);
    fflush(stdout);
    samples[ns++] = gbps;
    tp = t;
    bp = b;
  }
  atomic_store(&stop, 1);
  uint64_t sink = 0;
  for (int i = 0; i < nt; i++) {
    pthread_join(th[i], NULL);
    sink += args[i].sink;
  }
  qsort(samples, ns, sizeof(double), cmp);
  printf("median_gbps %.3f threads %d share_bytes %zu sink %llu\n", samples[ns / 2], nt, share,
         (unsigned long long)sink);
  return 0;
}
