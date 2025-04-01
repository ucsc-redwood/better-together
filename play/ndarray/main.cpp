#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  std::string input_file;
  app.add_option("-i,--input", input_file, "Input filename")
      ->default_val("cifar10_images/img_00005.npy");

  PARSE_ARGS_END;

  cifar_dense::AppData appdata(input_file);

  omp::dispatch_multi_stage_unrestricted(8, appdata, 1, 9);

  // Print result
  int predicted_class = cifar_dense::arg_max(appdata.linear_out.raw());
  cifar_dense::print_prediction(predicted_class);

  return 0;
}
