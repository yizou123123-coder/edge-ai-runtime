#include <edge_ai/trt_engine.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace edge_ai
{

void TrtLogger::log(Severity severity,
                    const char *msg) noexcept // noexcept：这个函数承诺不让异常传播到调用者
{
    // 只输出警告及更严重的日志
    if (severity <= Severity::kWARNING)
    {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

// 构造函数：输入：Builder将ONNX优化成Engine文件；过程：Runtime加载Engine文件；Engine表示优化好的模型；Context是给GPU执行的对象
TrtEngine::TrtEngine(const std::string &engine_path)
{
    // 1.初始化file对象：打开并读取二进制 engine文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open TensorRT engine: " + engine_path);
    }

    // 2.获取文件大小
    const std::streamsize file_size = file.tellg();
    if (file_size <= 0)
    {
        throw std::runtime_error("TensorRT engine file is empty or invalid");
    }

    // 3.申请CPU一块连续的内存存放Engine数据
    std::vector<char> engine_data(static_cast<std::size_t>(file_size));

    // 4.将文件内容读到CPU内存中
    file.seekg(0, std::ios::beg);
    if (!file.read(engine_data.data(), file_size))
    {
        throw std::runtime_error("Failed to read TensorRT engine file");
    }

    //======================== TensorRT api=========================================
    // 1.创建IRuntime 让runtime_管理IRuntime指针
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
    {
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    // 2.反序列化Engine：加载已有的执行计划、创建运行时所需的内部结构与资源
    // runtime_先反序列化，再让engine_管理指针
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_)
    {
        throw std::runtime_error("Failed to deserialize TensorRT Engine");
    }

    // 3.创建执行上下文：用于实际推理
    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // 4.获取TRT engine的输入和输出信息
    // 当前 YOLOv8n 模型预期只有 1 个输入、1 个输出
    if (engine_->getNbIOTensors() != 2)
    {
        throw std::runtime_error("Expected exactly 2 I/O tensors");
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        if (name == nullptr)
        {
            throw std::runtime_error("Failed to get tensor name");
        }

        const auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_name_ = name;
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            output_name_ = name;
    }

    if (input_name_.empty() || output_name_.empty())
    {
        throw std::runtime_error("Tensor I/O name error");
    }

    // 5.检查Engine的输入输出Tensor形状和数据类型
    const auto in_shape = engine_->getTensorShape(input_name_.c_str());
    const auto out_shape = engine_->getTensorShape(output_name_.c_str());
    // 判断输入输出格式
    if (in_shape.nbDims != 4 || in_shape.d[0] != 1 || in_shape.d[1] != 3 || in_shape.d[2] != 640 ||
        in_shape.d[3] != 640)
    {
        throw std::runtime_error("Unsupported input shape");
    }

    if (out_shape.nbDims != 3 || out_shape.d[0] != 1 || out_shape.d[1] != 84 ||
        out_shape.d[2] != 8400)
    {
        throw std::runtime_error("Unsupported output shape");
    }
    // 当前统一接口使用 float，所以先要求 I/O 为FP32
    if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
        engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT)
    {
        throw std::runtime_error("Unsupported TensorRT I/O data type");
    }
}

// 每一帧使用 GPU 进行推理
std::vector<float> TrtEngine::infer(const std::vector<float> &input)
{
    constexpr std::size_t expected_elements = 1 * 3 * 640 * 640;
    if (input.size() != expected_elements)
    {
        throw std::invalid_argument("Input tensor size mismatch");
    }

    // 1.第一次推理时申请显存，以后复用
    // 输入显存申请
    const std::size_t input_bytes = input.size() * sizeof(float);
    if (device_input_ == nullptr)
    {
        const cudaError_t err = cudaMalloc(&device_input_, input_bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }
    // 输出显存申请
    const std::size_t output_bytes = 1 * 84 * 8400 * sizeof(float);
    if (device_output_ == nullptr)
    {
        const cudaError_t err = cudaMalloc(&device_output_, output_bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }

    // 2.第一次推理时创建CUDA Stream，以后复用
    if (stream_ == nullptr)
    {
        const cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    // 3.创建CPU输出数组
    constexpr std::size_t output_elements = 1 * 84 * 8400;
    std::vector<float> output(output_elements);

    // 4.通过stream命令从CPU向GPU复制传输input数据（输入张量数组）
    cudaError_t err =
        cudaMemcpyAsync(device_input_, input.data(), input_bytes, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("H2D copy failed: ") + cudaGetErrorString(err));
    }

    // 5.绑定模型Tensor名称和GPU显存地址
    if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
        !context_->setTensorAddress(output_name_.c_str(), device_output_))
    {
        throw std::runtime_error("setTensorAddress failed");
    }

    // 6.提交TensorRT GPU推理任务
    if (!context_->enqueueV3(stream_))
    {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    // 7.通过stream命令从GPU向CPU复制传输output数据（输出张量数组）
    err = cudaMemcpyAsync(output.data(), device_output_, output_bytes, cudaMemcpyDeviceToHost,
                          stream_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("D2H copy failed: ") + cudaGetErrorString(err));
    }

    // 8.等待当前 Stream 中的任务执行完毕
    err = cudaStreamSynchronize(stream_);

    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA synchronization failed: ") +
                                 cudaGetErrorString(err));
    }

    return output;
}

TrtEngine::~TrtEngine()
{
    if (stream_ != nullptr)
    {
        // 等待当前 Stream 中的任务执行完毕，然后销毁
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (device_input_ != nullptr)
    {
        cudaFree(device_input_);
    }

    if (device_output_ != nullptr)
    {
        cudaFree(device_output_);
    }
}
} // namespace edge_ai
