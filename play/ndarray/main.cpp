#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  std::string input_file;
  app.add_option("-i,--input", input_file, "Input filename")
      ->default_val("cifar10_images/img_00005.npy");

  PARSE_ARGS_END;

  AppData appdata(input_file);

  omp::dispatch_multi_stage(appdata, 1, 9, 8);

  // Print result
  int predicted_class = arg_max(appdata.linear_out.raw());
  print_prediction(predicted_class);

  return 0;
}
