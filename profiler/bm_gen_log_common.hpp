#pragma once
// ---------------------------------------------------------------------------
// bm_gen_log_common -- shared driver for the schedule EXECUTOR that emits the
// per-task `### Python ###` timing log (consumed by 04_parse_schedules ->
// data/sched_logs). The 6 pipe/<cell>/bm_gen_log.* were ~95% identical forks; the
// only per-cell variation is the OMP dispatch namespace and which ExecutionModel
// is "the GPU" for this binary (kCuda for *-cu, kVulkan for *-vk).
//
// Each cell's bm_gen_log.{cpp,cu} is now a thin main() that supplies its types
// (const.hpp, included first) + its GPU ExecutionModel + the OMP dispatch closure.
// ---------------------------------------------------------------------------

#include <omp.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "platform/registry/device_registry.hpp"
#include "runtime/config_reader.hpp"
#include "runtime/placement.hpp"
#include "runtime/schedule_source.hpp"

namespace bt_gen_log {

// Spawn one worker per chunk: the GPU chunk (em == gpu_em) runs on the dispatcher;
// an OMP chunk runs the cell's omp_dispatch pinned to the chunk's CPU tier. A chunk
// targeting the OTHER GPU backend is rejected up front (see run()), so the spawn
// loop here only ever sees gpu_em or kOMP.

// A Vulkan GPU chunk's thread: its own Sequence, so a schedule may place the GPU in
// several chunks (design C2), and up to BT_VK_INFLIGHT items in flight (design D3).
// on_submit/on_complete(item index) bracket each item's time in the chunk.
static std::thread vk_gpu_chunk_thread(DispatcherT& disp,
                                       QueueT& q_in,
                                       QueueT& q_out,
                                       const int start,
                                       const int end,
                                       const size_t n_items,
                                       const bool is_last,
                                       std::function<void(size_t)> on_submit,
                                       std::function<void(size_t)> on_complete) {
  const auto seq = disp.make_seq();
  std::function<uint64_t(AppDataT*)> submit = [&disp, seq, start, end](AppDataT* app) {
    return disp.submit_multi_stage(*app, start, end, *seq);
  };
  std::function<void(AppDataT*, uint64_t)> complete = [&disp, seq, start, end](
                                                          AppDataT* app, const uint64_t ticket) {
    disp.complete_multi_stage(*app, start, end, *seq, ticket);
  };
  return std::thread([&q_in,
                      &q_out,
                      submit = std::move(submit),
                      complete = std::move(complete),
                      n_items,
                      is_last,
                      depth = vk_inflight_depth(),
                      on_submit = std::move(on_submit),
                      on_complete = std::move(on_complete)] {
    inflight_worker<QueueT, AppDataT>(
        q_in, q_out, submit, complete, n_items, is_last, depth, on_submit, on_complete);
  });
}

template <class OmpDispatch>
static void warmup(const Schedule& schedule,
                   const ExecutionModel gpu_em,
                   const OmpDispatch& omp_dispatch) {
  const auto n_chunks = schedule.n_chunks();
  DispatcherT disp;
  const std::vector<AppDataPtr> dataset =
      make_placed_dataset<AppDataT>(disp, kPoolSize, gpu_stages_of(schedule, gpu_em));

  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) queues[0].enqueue(dataset[i].get());

  constexpr size_t num_warmup_items = 5;
  std::vector<std::thread> threads;
  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueT& q_in = queues[chunk_id];
    QueueT& q_out = queues[(chunk_id + 1) % n_chunks];
    const int start = schedule.start_stage(chunk_id);
    const int end = schedule.end_stage(chunk_id);
    const ExecutionModel em = schedule.chunks[chunk_id].exec_model;
    const bool is_last = chunk_id == n_chunks - 1;

    if (em == gpu_em) {
      if constexpr (requires { disp.make_seq(); }) {
        threads.push_back(
            vk_gpu_chunk_thread(disp, q_in, q_out, start, end, num_warmup_items, is_last, {}, {}));
      } else {
        threads.emplace_back(
            worker<QueueT, AppDataT>,
            std::ref(q_in),
            std::ref(q_out),
            [&disp, start, end](AppDataT* app) { disp.dispatch_multi_stage(*app, start, end); },
            num_warmup_items,
            is_last);
      }
    } else {  // kOMP (the warmup is constructed with only gpu_em + kOMP chunks)
      const ProcessorType cpu_pt = get_processor_type_from_chunk_config(schedule.chunks[chunk_id]);
      threads.emplace_back(
          worker<QueueT, AppDataT>,
          std::ref(q_in),
          std::ref(q_out),
          [&omp_dispatch, cpu_pt, start, end](AppDataT* app) {
            auto& cores = get_cores_by_type(cpu_pt);
            omp_dispatch(cores, cores.size(), *app, start, end);
          },
          num_warmup_items,
          is_last);
    }
  }
  for (auto& t : threads) t.join();
}

template <class OmpDispatch>
static void run_schedule(const Schedule& schedule,
                         const ExecutionModel gpu_em,
                         const OmpDispatch& omp_dispatch) {
  const auto n_chunks = schedule.n_chunks();
  DispatcherT disp;
  const std::vector<AppDataPtr> dataset =
      make_placed_dataset<AppDataT>(disp, kPoolSize, gpu_stages_of(schedule, gpu_em));

  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) queues[0].enqueue(dataset[i].get());

  Logger<kNumToProcess> logger;
  std::vector<std::thread> threads;
  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueT& q_in = queues[chunk_id];
    QueueT& q_out = queues[(chunk_id + 1) % n_chunks];
    const int start = schedule.start_stage(chunk_id);
    const int end = schedule.end_stage(chunk_id);
    const ExecutionModel em = schedule.chunks[chunk_id].exec_model;
    const bool is_last = chunk_id == n_chunks - 1;

    if (em == gpu_em) {
      if constexpr (requires { disp.make_seq(); }) {
        const int id = static_cast<int>(chunk_id);
        threads.push_back(vk_gpu_chunk_thread(
            disp,
            q_in,
            q_out,
            start,
            end,
            kNumToProcess,
            is_last,
            [&logger, id](const size_t item) { logger.start_tick(item, id); },
            [&logger, id](const size_t item) { logger.end_tick(item, id); }));
      } else {
        threads.emplace_back(
            worker_with_record<QueueT, AppDataT, kNumToProcess>,
            static_cast<int>(chunk_id),
            std::ref(logger),
            std::ref(q_in),
            std::ref(q_out),
            [&disp, start, end](AppDataT* app) { disp.dispatch_multi_stage(*app, start, end); },
            kNumToProcess,
            is_last);
      }
    } else {  // kOMP (wrong-backend GPU chunks are filtered before we get here)
      const ProcessorType cpu_pt = get_processor_type_from_chunk_config(schedule.chunks[chunk_id]);
      threads.emplace_back(
          worker_with_record<QueueT, AppDataT, kNumToProcess>,
          static_cast<int>(chunk_id),
          std::ref(logger),
          std::ref(q_in),
          std::ref(q_out),
          [&omp_dispatch, cpu_pt, start, end](AppDataT* app) {
            auto& cores = get_cores_by_type(cpu_pt);
            omp_dispatch(cores, cores.size(), *app, start, end);
          },
          kNumToProcess,
          is_last);
    }
  }
  for (auto& t : threads) t.join();

  logger.dump_records_for_python(schedule);
}

// Every chunk must be runnable by THIS binary: an OMP chunk, or the GPU chunk for
// this backend. A chunk targeting the other GPU backend (a cross-backend schedule)
// can't run here -> caller skips it instead of deadlocking a worker ring.
[[nodiscard]] static inline bool backend_matches(const Schedule& schedule,
                                                 const ExecutionModel gpu_em) {
  for (const auto& c : schedule.chunks) {
    if (c.exec_model != ExecutionModel::kOMP && c.exec_model != gpu_em) return false;
  }
  return true;
}

// A portable warmup schedule that contiguously covers [1, kNumStages]: the first
// half on the device's first-present CPU tier, the rest on the GPU. Replaces the old
// hand-coded magic schedule (which hardcoded a tier and, for tree, even dropped the
// last stage). Validated via validate_schedule_coverage below.
[[nodiscard]] static inline Schedule make_warmup_schedule(const ExecutionModel gpu_em) {
  const int n = static_cast<int>(kNumStages);
  const int mid = n > 1 ? n / 2 : 1;
  Schedule s;
  s.uid = "warmup";
  s.chunks.push_back({ExecutionModel::kOMP, 1, mid, first_present_cpu_type()});
  if (mid < n) s.chunks.push_back({gpu_em, mid + 1, n, std::nullopt});
  return s;
}

template <class OmpDispatch>
inline int run(int argc,
               char** argv,
               const ExecutionModel gpu_em,
               const OmpDispatch& omp_dispatch) {
  PARSE_ARGS_BEGIN

  std::string schedule_file;
  app.add_option("--schedule-file", schedule_file, "Schedule JSON file path");
  size_t n_schedules_to_run = 0;  // 0 means run all schedules
  app.add_option(
      "--n-schedules-to-run", n_schedules_to_run, "Number of schedules to run (0 means run all)");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::off);  // don't log during warmup

  const Schedule warmup_schedule = make_warmup_schedule(gpu_em);
  validate_schedule_coverage(warmup_schedule, kNumStages);  // defensive: full coverage
  warmup(warmup_schedule, gpu_em, omp_dispatch);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (schedule_file.empty()) {
    spdlog::info("No schedule file provided. Running only warmup phase.");
    return 0;
  }

  std::vector<Schedule> schedules;
  try {
    const auto json = load_schedule_json(schedule_file);
    schedules = readSchedulesFromJson(json);
  } catch (const std::exception& e) {
    spdlog::error("Failed to read or parse schedules: {}", e.what());
    spdlog::warn("Running only warmup phase.");
    return 0;
  }

  // Fail-fast: a machine-generated schedule must contiguously cover all stages.
  for (size_t vi = 0; vi < schedules.size(); ++vi) {
    try {
      validate_schedule_coverage(schedules[vi], kNumStages);
    } catch (const std::exception& e) {
      spdlog::error("Schedule {} fails coverage validation: {}", vi, e.what());
      return 1;
    }
  }

  const auto n_schedules = schedules.size();
  if (n_schedules == 0) {
    spdlog::info("No schedules found.");
    return 0;
  }

  spdlog::info("Loaded {} schedules", n_schedules);
  for (size_t i = 0; i < n_schedules; ++i) {
    std::cout << "--------------------------------" << std::endl;
    schedules[i].print(i);
  }

  if (n_schedules_to_run == 0 || n_schedules_to_run > n_schedules) {
    n_schedules_to_run = n_schedules;
  }
  spdlog::info("Running {}/{} schedules", n_schedules_to_run, n_schedules);

  for (size_t i = 0; i < n_schedules_to_run; ++i) {
    if (const auto reason = schedule_unrunnable_reason(schedules[i])) {
      spdlog::warn("Skipping schedule {} [{}]: {}", i, schedules[i].uid, *reason);
      continue;
    }
    if (const auto reason = first_concurrent_gpu_chunk(schedules[i])) {
      spdlog::warn("Skipping schedule {} [{}]: {}", i, schedules[i].uid, *reason);
      continue;
    }
    if constexpr (requires(DispatcherT& d) { d.make_seq(); }) {
      size_t gpu_chunks = 0;
      for (const auto& c : schedules[i].chunks) gpu_chunks += c.exec_model == gpu_em ? 1 : 0;
      const size_t need =
          min_pool_size(gpu_chunks, schedules[i].n_chunks() - gpu_chunks, vk_inflight_depth());
      if (need > kPoolSize) {
        spdlog::warn("Schedule {} [{}]: a pool of {} caps BT_VK_INFLIGHT={} (needs {} items)",
                     i,
                     schedules[i].uid,
                     kPoolSize,
                     vk_inflight_depth(),
                     need);
      }
    }
    if (!backend_matches(schedules[i], gpu_em)) {
      spdlog::warn("Skipping schedule {} [{}]: targets the other GPU backend", i, schedules[i].uid);
      continue;
    }
    std::cout << "\n--------------------------------" << std::endl;
    run_schedule(schedules[i], gpu_em, omp_dispatch);
  }

  return 0;
}

}  // namespace bt_gen_log
