// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/cls.h>

namespace PaddleClas {

void Classifier::LoadModel(const std::string &model_path,
                           const std::string &params_path) {
  paddle_infer::Config config;
  config.SetModel(model_path, params_path);

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    if (this->use_tensorrt_) {
      config.EnableTensorRtEngine(
          1 << 20, 1, 3,
          this->use_fp16_ ? paddle_infer::Config::Precision::kHalf
                          : paddle_infer::Config::Precision::kFloat32,
          false, false);
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}

double Classifier::Run(cv::Mat &img, std::vector<double> *times) {
  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);

  auto preprocess_start = std::chrono::steady_clock::now();
  this->resize_op_.Run(img, resize_img, this->resize_short_size_);

  this->crop_op_.Run(resize_img, this->crop_size_);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);
  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());

  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
  auto preprocess_end = std::chrono::system_clock::now();

  auto infer_start = std::chrono::system_clock::now();
  input_t->CopyFromCpu(input.data());
  this->predictor_->Run();

  std::vector<float> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  auto infer_end = std::chrono::system_clock::now();

  auto postprocess_start = std::chrono::system_clock::now();
  int maxPosition =
      max_element(out_data.begin(), out_data.end()) - out_data.begin();
  auto postprocess_end = std::chrono::system_clock::now();

  // std::chrono::duration<float> preprocess_diff = preprocess_end -
  // preprocess_start;
  // times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = infer_end - infer_start;
  double inference_cost_time = double(inference_diff.count() * 1000);
  times->push_back(inference_cost_time);
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));

  std::cout << "result: " << std::endl;
  std::cout << "\tclass id: " << maxPosition << std::endl;
  std::cout << std::fixed << std::setprecision(10)
            << "\tscore: " << double(out_data[maxPosition]) << std::endl;

  return inference_cost_time;
}

} // namespace PaddleClas
