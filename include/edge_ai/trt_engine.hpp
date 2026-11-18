#pragma once

#include <edge_ai/inference_engine.hpp>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace edge_ai
{

// TensorRT 日志实现
class TrtLogger final : public nvinfer1::ILogger
{
  public:
    void log(Severity severity, const char *msg) noexcept override;
};

class TrtEngine final : public IInferenceEngine
{
  public:
    explicit TrtEngine(const std::string &engine_path);

    ~TrtEngine() override;

    // 统一 infer() 接口
    std::vector<float> infer(const std::vector<float> &input) override;

  private:
    TrtLogger logger_;

    // 三个 TensorRT 对象
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;

    // GPU 显存指针
    void *device_input_{nullptr};
    void *device_output_{nullptr};

    // CUDA 工作队列：存放操作命令而非数据
    cudaStream_t stream_{nullptr};
};

} // namespace edge_ai
