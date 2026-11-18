#include <edge_ai/ort_cpu_engine.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace edge_ai
{

namespace // 匿名命名空间
{
// 辅助函数：设置 ORT 中的线程数量，并返回 options
Ort::SessionOptions make_session_options(int intra_op_threads)
{
    if (intra_op_threads < 0)
    {
        throw std::invalid_argument("intra_op_threads must be >= 0");
    }

    Ort::SessionOptions options;

    if (intra_op_threads > 0)
    {
        options.SetIntraOpNumThreads(intra_op_threads);
    }

    return options;
}
} // namespace

OrtCpuEngine::OrtCpuEngine(const std::string &model_path, int intra_op_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "edge_detector"),
      session_options_(make_session_options(intra_op_threads)),
      session_(env_, model_path.c_str(), session_options_)
{
    Ort::AllocatorWithDefaultOptions allocator;

    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}

std::vector<float> OrtCpuEngine::infer(const std::vector<float> &input)
{
    constexpr std::size_t expected_input_size = 1 * 3 * 640 * 640;
    if (input.size() != expected_input_size)
    {
        throw std::invalid_argument("Input tensor size mismatch");
    }

    // 四维矩阵 规定了输入的格式
    const std::array<std::int64_t, 4> input_shape{1, 3, 640, 640};

    std::vector<float> writable_input = input;

    // 向 ORT 描述 Tensor 所使用的内存位于哪里，以及相关内存属性：默认cpu内存
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ONNX Runtime 能识别和使用的输入对象：
    // tensor_data：真正的 float 数据  input_shape：按 [1,3,640,640] 维度解释数据
    // MemoryInfo：描述输入数据所在的内存
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, writable_input.data(), writable_input.size(),
                                                        input_shape.data(), input_shape.size());

    // 创建 input_names 和 output_names
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    if (outputs.size() != 1)
    {
        throw std::runtime_error("Unexpected output tensor count");
    }
    if (!outputs[0].IsTensor())
    {
        throw std::runtime_error("output Tensor");
    }

    // 获取 outputs[0] 的类型和形状信息
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();

    // 验证输出的形状
    const auto output_shape = output_info.GetShape();
    const std::vector<std::int64_t> expected_shape{1, 84, 8400};
    if (output_shape != expected_shape)
    {
        throw std::runtime_error("Output shape mismatch");
    }

    const std::size_t element_count = output_info.GetElementCount();
    constexpr std::size_t expected_elements = 705600;
    if (element_count != expected_elements)
    {
        std::cerr << "Output element count mismatch: " << element_count << '\n';
        throw std::runtime_error("Output element count mismatch");
    }

    const float *output_data = outputs[0].GetTensorData<float>();
    if (output_data == nullptr)
    {
        std::cerr << "Failed to get output tensor data pointer!\n";
        throw std::runtime_error("Failed to get output tensor data pointer!");
    }

    return std::vector<float>(output_data, output_data + element_count);
}

} // namespace edge_ai
