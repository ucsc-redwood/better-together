// #include <spdlog/spdlog.h>

// #include "engine.hpp"

// template <typename T>
// using std::pmr::vector = std::pmr::vector<T>;

// static void run_hello_vector_add(const vulkan::Engine& engine, const vulkan::Sequence* seq) {
//   constexpr auto n = 1024;
//   std::pmr::vector<float> input_a(n, engine.get_mr());
//   std::pmr::vector<float> input_b(n, engine.get_mr());
//   std::pmr::vector<float> output(n, engine.get_mr());

//   std::ranges::fill(input_a, 1.0f);
//   std::ranges::fill(input_b, 2.0f);
//   std::ranges::fill(output, 0.0f);

//   struct Ps {
//     uint32_t n;
//   };

//   auto algo = engine.make_algo("hello_vector_add")
//                   ->work_group_size(256, 1, 1)
//                   ->num_sets(1)
//                   ->num_buffers(3)
//                   ->push_constant<Ps>()
//                   ->build();

//   algo->update_push_constant(Ps{
//       .n = n,
//   });

//   algo->update_descriptor_set(0,
//                               {
//                                   engine.get_buffer_info(input_a),
//                                   engine.get_buffer_info(input_b),
//                                   engine.get_buffer_info(output),
//                               });

//   seq->cmd_begin();
//   algo->record_bind_core(seq->get_handle(), 0);
//   algo->record_bind_push(seq->get_handle());
//   algo->record_dispatch(seq->get_handle(), {vulkan::div_ceil(n, 256), 1, 1});
//   seq->cmd_end();

//   seq->launch_kernel_async();
//   seq->sync();

//   // print 10 output elements
//   for (auto i = 0; i < 10; i++) {
//     spdlog::info("output[{}] = {}", i, output[i]);
//   }
// }

// static void run_multiple_steps(const vulkan::Engine& engine, const vulkan::Sequence* seq) {
//   constexpr auto n = 1024;
//   std::pmr::vector<float> buf_a(n, engine.get_mr());
//   std::pmr::vector<float> buf_b(n, engine.get_mr());
//   std::pmr::vector<float> buf_c(n, engine.get_mr());
//   std::pmr::vector<float> buf_d(n, engine.get_mr());
//   std::pmr::vector<float> buf_e(n, engine.get_mr());
//   std::pmr::vector<float> buf_f(n, engine.get_mr());

//   std::ranges::fill(buf_a, 1.0f);
//   std::ranges::fill(buf_c, 2.0f);
//   std::ranges::fill(buf_e, 3.0f);

//   struct Ps {
//     uint32_t n;
//   };

//   auto algo = engine.make_algo("hello_multiple_steps")
//                   ->work_group_size(256, 1, 1)
//                   ->num_sets(3)  // we want 3 descriptor sets
//                   ->num_buffers(2)
//                   ->push_constant<Ps>()
//                   ->build();

//   algo->update_push_constant(Ps{
//       .n = n,
//   });

//   algo->update_descriptor_set(0,
//                               {
//                                   engine.get_buffer_info(buf_a),
//                                   engine.get_buffer_info(buf_b),
//                               });
//   algo->update_descriptor_set(1,
//                               {
//                                   engine.get_buffer_info(buf_c),
//                                   engine.get_buffer_info(buf_d),
//                               });
//   algo->update_descriptor_set(2,
//                               {
//                                   engine.get_buffer_info(buf_e),
//                                   engine.get_buffer_info(buf_f),
//                               });
//   seq->cmd_begin();

//   algo->record_bind_core(seq->get_handle(), 0);
//   algo->record_bind_push(seq->get_handle());
//   algo->record_dispatch(seq->get_handle(),
//                         {vulkan::div_ceil(n, 256), 1, 1});  // whatever your grid is

//   algo->record_bind_core(seq->get_handle(), 1);
//   algo->record_bind_push(seq->get_handle());
//   algo->record_dispatch(seq->get_handle(), {vulkan::div_ceil(n, 256), 1, 1});

//   algo->record_bind_core(seq->get_handle(), 2);
//   algo->record_bind_push(seq->get_handle());
//   algo->record_dispatch(seq->get_handle(), {vulkan::div_ceil(n, 256), 1, 1});

//   seq->cmd_end();

//   seq->launch_kernel_async();
//   seq->sync();

//   // print 10 output elements
//   for (auto i = 0; i < 10; i++) {
//     spdlog::info("buf_a[{}] = {}", i, buf_a[i]);
//     spdlog::info("buf_b[{}] = {}", i, buf_b[i]);
//     spdlog::info("buf_c[{}] = {}", i, buf_c[i]);
//     spdlog::info("buf_d[{}] = {}", i, buf_d[i]);
//     spdlog::info("buf_e[{}] = {}", i, buf_e[i]);
//     spdlog::info("buf_f[{}] = {}", i, buf_f[i]);
//   }
// }

// // static void run_multiple_steps_chained(const vulkan::Engine& engine, const vulkan::Sequence* seq) {
// //   constexpr auto n = 1024;
// //   std::pmr::vector<float> buf_a(n, engine.get_mr());
// //   std::pmr::vector<float> buf_b(n, engine.get_mr());
// //   std::pmr::vector<float> buf_c(n, engine.get_mr());
// //   std::pmr::vector<float> buf_d(n, engine.get_mr());

// //   std::ranges::fill(buf_a, 1.0f);
// //   std::ranges::fill(buf_c, 2.0f);

// //   struct Ps {
// //     uint32_t n;
// //   };

// //   auto algo = engine.make_algo("hello_multiple_steps")
// //                   ->work_group_size(256, 1, 1)
// //                   ->num_sets(3)  // we want 3 descriptor sets
// //                   ->num_buffers(2)
// //                   ->push_constant<Ps>()
// //                   ->build();

// //   algo->update_push_constant(Ps{
// //       .n = n,
// //   });

// //   algo->update_descriptor_set(0,
// //                               {
// //                                   engine.get_buffer_info(buf_a),
// //                                   engine.get_buffer_info(buf_b),
// //                               });
// //   algo->update_descriptor_set(1,
// //                               {
// //                                   engine.get_buffer_info(buf_b),
// //                                   engine.get_buffer_info(buf_c),
// //                               });
// //   algo->update_descriptor_set(2,
// //                               {
// //                                   engine.get_buffer_info(buf_c),
// //                                   engine.get_buffer_info(buf_d),
// //                               });
// //   seq->cmd_begin();

// //   algo->record_bind_core(seq->get_handle(), 0);
// //   algo->record_bind_push(seq->get_handle());
// //   algo->record_dispatch(seq->get_handle(),
// //                         {vulkan::div_ceil(n, 256), 1, 1});  // whatever your grid is

// //   seq->insert_compute_memory_barrier();

// //   algo->record_bind_core(seq->get_handle(), 1);
// //   algo->record_bind_push(seq->get_handle());
// //   algo->record_dispatch(seq->get_handle(), {vulkan::div_ceil(n, 256), 1, 1});

// //   seq->insert_compute_memory_barrier();

// //   algo->record_bind_core(seq->get_handle(), 2);
// //   algo->record_bind_push(seq->get_handle());
// //   algo->record_dispatch(seq->get_handle(), {vulkan::div_ceil(n, 256), 1, 1});

// //   seq->cmd_end();

// //   seq->launch_kernel_async();
// //   seq->sync();

// //   // print 10 output elements
// //   for (auto i = 0; i < 10; i++) {
// //     spdlog::info("buf_a[{}] = {}", i, buf_a[i]);
// //     spdlog::info("buf_b[{}] = {}", i, buf_b[i]);
// //     spdlog::info("buf_c[{}] = {}", i, buf_c[i]);
// //     spdlog::info("buf_d[{}] = {}", i, buf_d[i]);
// //   }
// // }

// int main() {
//   spdlog::set_level(spdlog::level::trace);

//   vulkan::Engine engine;
//   auto seq = engine.make_seq();

//   run_hello_vector_add(engine, seq.get());

//   run_multiple_steps(engine, seq.get());

//   run_multiple_steps_chained(engine, seq.get());

//   spdlog::info("done!");
//   return 0;
// }
