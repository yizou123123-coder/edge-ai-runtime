#include <edge_ai/postprocess.hpp>

#include <algorithm>
#include <algorithm>
#include <vector>

namespace edge_ai
{
// 辅助函数：计算iou
float compute_iou(
    const cv::Rect2f& a,
    const cv::Rect2f& b
)
{
    const float inter_left = std::max(a.x, b.x);
    const float inter_top = std::max(a.y, b.y);
    const float inter_right = std::min(a.x + a.width, b.x + b.width);
    const float inter_bottom = std::min(a.y + a.height, b.y + b.height);

    const float inter_width = std::max(inter_right - inter_left, 0.0F);
    const float inter_height = std::max(inter_bottom - inter_top, 0.0F);

    const float inter_area = inter_width * inter_height;

    const float area_a = a.width * a.height;
    const float area_b = b.width * b.height;
    const float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0F)
    {
        return 0.0F;
    }
    
    return inter_area / union_area;
}


// 辅助函数：通过nms比较8400个框的iou，并返回
std::vector<Detection> apply_nms(
    std::vector<Detection> candidates,
    float iou_threshold
)
{
    // 排序，按Lambda 表达式排序
    std::sort(candidates.begin(), 
        candidates.end(), 
        [](const Detection& lhs, const Detection& rhs)
        {
            return lhs.confidence > rhs.confidence;
        });
    // 
    std::vector<Detection> kept_detections;
    for(const auto& candidate : candidates)
    {
        // 是否被高分同类框抑制
        bool should_suppress = false;
        for (const auto& kept : kept_detections)
        {
            if (candidate.class_id == kept.class_id)
            {
                if (compute_iou(candidate.box, kept.box) > iou_threshold)
                {
                    should_suppress = true;
                    break;
                }
            }
        }
        // 未被高分同类框抑制，则保留该候选框
        if (!should_suppress)
        {
            kept_detections.push_back(candidate);
        }
    }
    return kept_detections;
}


std::vector<Detection> postprocess_yolo(
    const float* output_data,
    std::size_t num_candidates,
    const PreprocessResult& preprocess,
    float confidence_threshold,
    float nms_iou_threshold)
{
    std::vector<Detection> candidates;
    if (output_data == nullptr || num_candidates == 0)
    {
        return candidates;
    }
    constexpr int num_classes = 80;
    for (std::size_t i = 0; i < num_candidates; ++i)
    {
        const float cx = output_data[0 * num_candidates + i];
        const float cy = output_data[1 * num_candidates + i];
        const float w  = output_data[2 * num_candidates + i];
        const float h  = output_data[3 * num_candidates + i];

        float max_score = 0.0F;
        int best_class_id{-1};

        for (int c = 0; c < num_classes; ++c)
        {
            const float score = output_data[(4 + c) * num_candidates + i];
            if (score > max_score)
            {
                max_score = score;
                best_class_id = c;
            }
        }
        
        if (max_score < confidence_threshold)
        {
            continue;
        }
        
        if(w<=0 || h<=0)
        {
            continue;
        }

        const float x1 = cx - w * 0.5F;
        const float y1 = cy - h * 0.5F;

        Detection det;
        det.class_id = best_class_id;
        det.confidence = max_score;
        det.box = cv::Rect2f(x1, y1, w, h);
        candidates.push_back(det);
    }


    auto final_detections = apply_nms(candidates, nms_iou_threshold);

    const float original_w = static_cast<float>(preprocess.original_width);
    const float original_h = static_cast<float>(preprocess.original_height);

    std::vector<Detection> valid_detections;
    // 预留空间给 valid_detections
    valid_detections.reserve(final_detections.size());

    for (const auto& candidate : final_detections)
    {
        // 逆向反算还原到未裁剪的原图坐标
        const float raw_x1 = (candidate.box.x - preprocess.pad_x) / preprocess.scale;
        const float raw_y1 = (candidate.box.y - preprocess.pad_y) / preprocess.scale;
        const float raw_w  = candidate.box.width / preprocess.scale;
        const float raw_h  = candidate.box.height / preprocess.scale;

        // TODO 1: 取得还原后框的 x1、y1、x2、y2
        const float raw_x2 = raw_x1 + raw_w;
        const float raw_y2 = raw_y1 + raw_h;

        // TODO 2: 使用 std::clamp() 限制在图像有效区域内
        // 将 x 限制在 [0, original_width]，y 限制在 [0, original_height]
        const float x1 = std::clamp(raw_x1, 0.0F, original_w);
        const float y1 = std::clamp(raw_y1, 0.0F, original_h);
        const float x2 = std::clamp(raw_x2, 0.0F, original_w);
        const float y2 = std::clamp(raw_y2, 0.0F, original_h);

        // TODO 3: 检查裁剪后 x2 > x1、y2 > y1
        // 如果一个框完全位于灰色填充区外部，裁剪后宽或高会塌陷为 0，必须丢弃
        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }

        // TODO 4: 重新构造 Rect2f
        Detection det;
        det.class_id   = candidate.class_id;
        det.confidence = candidate.confidence;
        det.box        = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

        valid_detections.push_back(det);
    }

    return valid_detections;
}
} // namespace edge_ai