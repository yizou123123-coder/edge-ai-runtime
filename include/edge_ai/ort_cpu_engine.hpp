#pragma once

#include <edge_ai/inference_engine.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace edge_ai
{

class OrtCpuEngine final : public IInferenceEngine
{
  public:
    explicit OrtCpuEngine(const std::string &model_path, int intra_op_threads = 0);

    std::vector<float> infer(const std::vector<float> &input) override;

  private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;
};

} // namespace edge_ai
