#include <onnxruntime_cxx_api.h>

#include <iostream>

int main()
{
  
    // 创建环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_session_test");

    // 配置“这个 Session 怎么运行”
    Ort::SessionOptions session_options;

    // 加载一个 ONNX 模型后的推理会话
    Ort::Session session(env, "models/yolov8n.onnx", session_options);

    std::cout << "ONNX Runtime environment: PASS\n";

    const std::size_t input_count = session.GetInputCount();
    const std::size_t output_count = session.GetOutputCount();

    std::cout << "Input count: " << input_count << '\n';
    std::cout << "Output count: " << output_count << '\n';
    
    // 创建一个 ONNX Runtime 默认的内存分配器对象（“内存管家”）,查询输入/输出节点的名字
    Ort::AllocatorWithDefaultOptions allocator;

    // 查询第 0 个输入节点的名称
    auto input0 = session.GetInputNameAllocated(0, allocator); 

    // 获取第 0 个输入的类型信息
    auto input_type_info = session.GetInputTypeInfo(0);
    // 从类型信息中提取张量类型与形状描述
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    // 获取具体的维度数值数组
    const std::vector<int64_t> input_shape = input_tensor_info.GetShape(); 

    std::cout << "Input name: " << input0.get() << '\n';
    std::cout << "Input shape: ";
    for (const auto a : input_shape)
    {
        std::cout << a << " ";
    }
    std::cout << '\n';

    auto output0 = session.GetOutputNameAllocated(0, allocator);
    auto output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> output_shape = output_tensor_info.GetShape();
    std::cout << "Onput name: " << output0.get() << '\n';
    std::cout << "Output shape: ";
    for (const auto a : output_shape)
    {
        std::cout << a << " ";
    }
    std::cout << '\n';

    return 0;
}