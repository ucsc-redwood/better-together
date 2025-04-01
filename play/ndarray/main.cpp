#include <cnpy.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstring>
#include <string>

#include "appdata.hpp"
#include "kernels.hpp"
#include "ndarray.hpp"

int main(int argc, char** argv) {
  CLI::App app{"CIFAR-10 Dense App"};

  std::string input_file;
  app.add_option("-i,--input", input_file, "Input filename")->required();

  CLI11_PARSE(app, argc, argv);

  Appdata appdata(input_file);

#pragma omp parallel
  {
    conv2d_omp(
        appdata.input, appdata.conv1_weights, appdata.conv1_bias, 1, 0, true, appdata.conv1_out);
    maxpool2d_omp(appdata.conv1_out, 2, 2, appdata.pool1_out);
    conv2d_omp(appdata.pool1_out,
               appdata.conv2_weights,
               appdata.conv2_bias,
               1,
               0,
               true,
               appdata.conv2_out);
    maxpool2d_omp(appdata.conv2_out, 2, 2, appdata.pool2_out);
    conv2d_omp(appdata.pool2_out,
               appdata.conv3_weights,
               appdata.conv3_bias,
               1,
               0,
               true,
               appdata.conv3_out);
    conv2d_omp(appdata.conv3_out,
               appdata.conv4_weights,
               appdata.conv4_bias,
               1,
               0,
               true,
               appdata.conv4_out);
    conv2d_omp(appdata.conv4_out,
               appdata.conv5_weights,
               appdata.conv5_bias,
               1,
               0,
               true,
               appdata.conv5_out);
    maxpool2d_omp(appdata.conv5_out, 2, 2, appdata.pool3_out);

    linear_omp(appdata.pool3_out.flatten(),
               appdata.linear_weights,
               appdata.linear_bias,
               appdata.linear_out);
  }

  // use argmax to get the predicted class
  int predicted_class = arg_max(appdata.linear_out.raw());
  print_prediction(predicted_class);

  return 0;
}
