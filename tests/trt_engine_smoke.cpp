#include <edge_ai/trt_engine.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: trt_engine_smoke " << "<engine_path> <input.bin> <output.bin>\n";
        return 2;
    }

    try
    {
        // 加载 TensorRT Engine
        edge_ai::TrtEngine trt(argv[1]);

        // 创建一份测试输入
        constexpr std::size_t input_elements = 1 * 3 * 640 * 640;
        std::vector<float> input(input_elements);

        std::ifstream file(argv[2], std::ios::binary);

        if (!file)
        {
            throw std::runtime_error("Cannot open input.bin");
        }

        const auto input_bytes = static_cast<std::streamsize>(input.size() * sizeof(float));

        if (!file.read(reinterpret_cast<char *>(input.data()), input_bytes))
        {
            throw std::runtime_error("Failed to read input.bin");
        }

        if (file.peek() != std::char_traits<char>::eof())
        {
            throw std::runtime_error("Unexpected extra input data");
        }

        // 真正执行 GPU 推理
        auto output = trt.infer(input);

        if (output.size() != 1 * 84 * 8400)
        {
            throw std::runtime_error("Output size mismatch");
        }

        const bool finite = std::all_of(output.begin(), output.end(),
                                        [](float value) { return std::isfinite(value); });
                                        
        std::ofstream out_file(argv[3], std::ios::binary);

        if (!out_file)
        {
            throw std::runtime_error("Cannot create output.bin");
        }

        out_file.write(reinterpret_cast<const char *>(output.data()),
                       static_cast<std::streamsize>(output.size() * sizeof(float)));

        if (!out_file)
        {
            throw std::runtime_error("Failed to save output.bin");
        }


        if (!finite)
        {
            throw std::runtime_error("Output contains NaN or Inf");
        }

        std::cout << "TensorRT GPU inference OK\n";
        std::cout << "Output elements: " << output.size() << '\n';
        std::cout << "First value: " << output[0] << '\n';

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "TensorRT error: " << e.what() << '\n';
        return 1;
    }
}
