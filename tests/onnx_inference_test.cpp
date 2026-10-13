#include <edge_ai/preprocess.hpp>
#include <edge_ai/postprocess.hpp>

#include <onnxruntime_cxx_api.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <cstring> 
#include <fstream>
#include <stdexcept>
#include <string>

// 二进制导出函数
void save_f32_binary(
    const std::string& path,
    const float* data,
    std::size_t count)
{
    // 以二进制且清空旧内容的方式，打开或创建一个用于写入数据的文件, 专门用于向磁盘写数据
    std::ofstream file(path, std::ios::binary | std::ios::trunc);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // 写入file  count * sizeof(float) 个字节
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));

    if(!file.good())
    {
        throw std::runtime_error("Failed to write binary data to file: " + path);
    }
}

int main()
{
    const std::string model_path = "models/yolov8n.onnx";

    const std::string video_path = "/home/yizou/datasets/edge-ai-runtime/test.mp4";


    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference_test");

    Ort::SessionOptions session_options;

    Ort::Session session(env, model_path.c_str(), session_options);

    cv::VideoCapture capture(video_path);
    if(!capture.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << '\n';
        return 1;
    }

    cv::Mat frame;
    if(!capture.read(frame) || frame.empty())
    {
        std::cerr << "Failed to read first frame!\n";
        return 1;
    }
    if (!cv::imwrite("build/reference/frame_cpp.png", frame))
    {
        throw std::runtime_error("Failed to save reference frame");
    }
    auto preprocess_result = edge_ai::preprocess_yolo(frame, 640, 640);

    save_f32_binary("build/reference/input_f32.bin", preprocess_result.tensor_data.data(), preprocess_result.tensor_data.size());

    std::cout << "Tensor data size: " << preprocess_result.tensor_data.size() << " floats\n";

    // 四维矩阵 规定了输入的格式
    const std::array<std::int64_t, 4> input_shape{1, 3, 640, 640};

    // 向 ORT 描述 Tensor 所使用的内存位于哪里，以及相关内存属性：默认cpu内存
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ONNX Runtime 能识别和使用的输入对象：
    // tensor_data：真正的 float 数据  input_shape：按 [1,3,640,640] 维度解释数据  MemoryInfo：描述输入数据所在的内存
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocess_result.tensor_data.data(), preprocess_result.tensor_data.size(),
                                                        input_shape.data(), input_shape.size());

    if(!input_tensor.IsTensor())
    {
        std::cerr << "Failed to be Tensor " << '\n';
        return 1;
    }

    // GetTensorTypeAndShapeInfo()：读取已经创建的 Tensor 的数据类型和形状描述
    // GetShape()：从形状描述中取得具体维度数组
    if(input_tensor.GetTensorTypeAndShapeInfo().GetShape() != std::vector<std::int64_t>{1, 3, 640, 640})
    {
        std::cerr << "Tensor shape error" << '\n';
        return 1;
    }

    // 创建一个 ONNX Runtime 默认的内存分配器对象（内存管家）,查询输入/输出节点的名字
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::cout << "Input name: " << input_name.get() << '\n';
    std::cout << "Onput name: " << output_name.get() << '\n';
    if (!input_name || !output_name)
    {
        std::cerr << "Failed to get node names\n";
        return 1;
    }
    if(std::strcmp(input_name.get(), "images") !=0 ||  std::strcmp(output_name.get(), "output0"))
    {
        std::cerr << "Unexpected input/output names\n";
        return 1;
    }

    // run
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    if(outputs.size() != 1)
    {
        std::cerr << "output size\n";
        return 1;
    }
    if (!outputs[0].IsTensor())
    {
        std::cerr << "output Tensor\n";
        return 1;
    }

    // 获取 outputs[0] 的类型和形状信息
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo(); 
    // 验证输出的形状
    const auto output_shape = output_info.GetShape();
    const std::vector<std::int64_t> expected_shape{1, 84, 8400};
    if (output_shape != expected_shape) //
    {
        std::cerr << "Output shape mismatch!\n";
        return 1;
    }
    // 验证输出的数量
    const std::size_t element_count = output_info.GetElementCount(); //
    constexpr std::size_t expected_elements = 705600; 
    if (element_count != expected_elements)
    {
        std::cerr << "Output element count mismatch: " << element_count << '\n';
        return 1;
    }

    // 通过 GetTensorData<float>() 获取数据指针
    const float* output_data = outputs[0].GetTensorData<float>();
    if (output_data == nullptr)
    {
        std::cerr << "Failed to get output tensor data pointer!\n";
        return 1;
    }

    save_f32_binary("build/reference/cpp_output_f32.bin", output_data, element_count);

    std::cout << "First 10 output values: "; 
    for (std::size_t i = 0; i < 10; ++i) 
    {
        std::cout << output_data[i] << ' ';
    }
    std::cout << '\n';


    const std::size_t num_candidates = static_cast<std::size_t>(output_shape[2]);
    const auto detections = edge_ai::postprocess_yolo(
        output_data,
        num_candidates,
        preprocess_result,
        0.25F,
        0.45F
    );

    std::cout << "Detections after NMS: " << detections.size() << '\n';

    std::size_t printed = 0;

    for (const auto& det : detections)
    {
        if (printed >= 10)
        {
            break;
        }

        std::cout << "Class ID: " << det.class_id
                  << " Confidence: " << det.confidence
                  << " Box: ("
                  << det.box.x << ", "
                  << det.box.y << ", "
                  << det.box.width << ", "
                  << det.box.height << ")\n";

        ++printed;
    }
    std::cout << "ONNX inference execution: PASS\n";
    return 0;
}

