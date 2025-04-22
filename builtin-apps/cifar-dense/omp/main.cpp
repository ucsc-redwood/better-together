#include <omp.h>
#include <spdlog/spdlog.h>

#include "../../app.hpp"
#include "../../hex_dump.hpp"
#include "dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  dumpCompressed(appdata.u_input.pmr_vec());

  spdlog::info("u_conv1_out before");
  dumpCompressed(appdata.u_conv1_out.pmr_vec());

  cifar_dense::omp::dispatch_stage(appdata, 1);

  spdlog::info("u_conv1_out after");
  dumpCompressed(appdata.u_conv1_out.pmr_vec());


  return 0;
}
