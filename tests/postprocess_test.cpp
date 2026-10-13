
#include <edge_ai/postprocess.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

bool near(float a, float b)
{
    return std::fabs(a - b) < 1e-3F;
}

// 按 YOLO [84, N] 的排列方式填入一个候选框
void set_candidate(
    std::vector<float>& data,
    std::size_t n,
    std::size_t i,
    float cx, float cy, float w, float h,
    int class_id,
    float confidence)
{
    data[0 * n + i] = cx;
    data[1 * n + i] = cy;
    data[2 * n + i] = w;
    data[3 * n + i] = h;

    data[(4 + class_id) * n + i] = confidence;
}

bool contains(
    const std::vector<edge_ai::Detection>& detections,
    int class_id,
    float confidence,
    float x, float y, float w, float h)
{
    for (const auto& det : detections)
    {
        if (det.class_id == class_id &&
            near(det.confidence, confidence) &&
            near(det.box.x, x) &&
            near(det.box.y, y) &&
            near(det.box.width, w) &&
            near(det.box.height, h))
        {
            return true;
        }
    }
    return false;
}

void print_detections(
    const std::vector<edge_ai::Detection>& detections)
{
    for (const auto& det : detections)
    {
        std::cout
            << "Class: " << det.class_id
            << " Score: " << det.confidence
            << " Box: ("
            << det.box.x << ", "
            << det.box.y << ", "
            << det.box.width << ", "
            << det.box.height << ")\n";
    }
}

} // namespace

int main()
{
    // =========================================
    // TEST 1: Decode + Confidence Filter + NMS
    // =========================================
    {
        constexpr std::size_t n = 5;
        std::vector<float> data(84 * n, 0.0F);

        // A: person
        set_candidate(data, n, 0, 100, 100, 40, 40, 0, 0.95F);

        // B: 与 A 高度重叠，同类别，应抑制
        set_candidate(data, n, 1, 102, 102, 40, 40, 0, 0.80F);

        // C: 远处的另一人，应保留
        set_candidate(data, n, 2, 300, 300, 40, 40, 0, 0.90F);

        // D: 与 A 重叠，但属于 car，应保留
        set_candidate(data, n, 3, 101, 101, 40, 40, 2, 0.85F);

        // E: 低置信度，应过滤
        set_candidate(data, n, 4, 450, 450, 40, 40, 0, 0.10F);

        edge_ai::PreprocessResult prep;
        prep.original_width = 640;
        prep.original_height = 640;
        prep.scale = 1.0F;
        prep.pad_x = 0;
        prep.pad_y = 0;

        auto result = edge_ai::postprocess_yolo(
            data.data(), n, prep, 0.25F, 0.45F
        );

        if (result.size() != 3 ||
            !contains(result, 0, 0.95F, 80, 80, 40, 40) ||
            !contains(result, 0, 0.90F, 280, 280, 40, 40) ||
            !contains(result, 2, 0.85F, 81, 81, 40, 40))
        {
            std::cerr << "TEST 1 FAIL: Decode/NMS\n";
            print_detections(result);
            return 1;
        }

        std::cout << "TEST 1 Decode + NMS: PASS\n";
    }

    // =========================================
    // TEST 2: Letterbox Coordinate Restore
    // =========================================
    {
        constexpr std::size_t n = 1;
        std::vector<float> data(84 * n, 0.0F);

        // 模型空间：
        // center=(120,240), size=(40,80)
        // 左上角=(100,200)
        set_candidate(data, n, 0, 120, 240, 40, 80, 0, 0.90F);

        edge_ai::PreprocessResult prep;
        prep.original_width = 1920;
        prep.original_height = 1080;
        prep.scale = 1.0F / 3.0F;
        prep.pad_x = 0;
        prep.pad_y = 140;

        auto result = edge_ai::postprocess_yolo(
            data.data(), n, prep, 0.25F, 0.45F
        );

        // 还原后：
        // x=300, y=180, w=120, h=240
        if (result.size() != 1 ||
            !contains(result, 0, 0.90F, 300, 180, 120, 240))
        {
            std::cerr << "TEST 2 FAIL: Letterbox restore\n";
            print_detections(result);
            return 2;
        }

        std::cout << "TEST 2 Letterbox restore: PASS\n";
    }

    // =========================================
    // TEST 3: Boundary Clipping
    // =========================================
    {
        constexpr std::size_t n = 1;
        std::vector<float> data(84 * n, 0.0F);

        // 框部分超出原图左侧、上侧
        set_candidate(data, n, 0, 5, 150, 30, 60, 0, 0.90F);

        edge_ai::PreprocessResult prep;
        prep.original_width = 1920;
        prep.original_height = 1080;
        prep.scale = 1.0F / 3.0F;
        prep.pad_x = 0;
        prep.pad_y = 140;

        auto result = edge_ai::postprocess_yolo(
            data.data(), n, prep, 0.25F, 0.45F
        );

        // 裁剪前：
        // x=-30, y=-60, x2=60, y2=120
        //
        // 裁剪到原图后：
        // x=0, y=0, w=60, h=120
        if (result.size() != 1 ||
            !contains(result, 0, 0.90F, 0, 0, 60, 120))
        {
            std::cerr << "TEST 3 FAIL: Boundary clipping\n";
            print_detections(result);
            return 3;
        }

        std::cout << "TEST 3 Boundary clipping: PASS\n";
    }

    std::cout << "\nP3-03 Postprocess Tests: ALL PASS\n";
    return 0;
}
