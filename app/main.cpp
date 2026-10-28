#include <edge_ai/runtime.hpp>
#include <edge_ai/config_loader.hpp>
#include <edge_ai/frame_source_factory.hpp>
#include <edge_ai/inference_result.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <exception>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[])
{
    if (argc != 3 || std::string(argv[1]) != "--config")
    {
        std::cerr << "Usage: " << argv[0] << " --config <yaml_path>\n";
        return 1;
    }

    const std::string mode = argv[1];

    if (mode == "--config")
    {
        try
        {
            edge_ai::RuntimeConfig config = edge_ai::load_config(argv[2]);

            edge_ai::validate_startup_config(config);

            auto source = edge_ai::create_frame_source(config.source);

            edge_ai::Runtime runtime(std::move(source), config.runtime.queue_policy, config.runtime.queue_capacity, config.model.path, 
                            config.runtime.deadline_ms, config.model.confidence_threshold, config.model.nms_iou_threshold, config.display.enabled);

            if (!runtime.start())
            {
                return runtime.exit_code();
            }

            if (config.display.enabled)
            {
                // 新建一个窗口
                cv::namedWindow("Edge AI", cv::WINDOW_NORMAL);

                while (true)
                {
                    edge_ai::InferenceResult result;

                    
                    if (runtime.try_pop_display(result))
                    {
                        cv::Mat display_image = result.frame.image.clone();

                        // 1. 遍历当前帧的所有检测结果
                        for (const auto& detection : result.detections)
                        {
                            // 2. 将检测框限制在图像范围内
                            const cv::Rect2f image_bounds(
                                0.0F,
                                0.0F,
                                static_cast<float>(display_image.cols),
                                static_cast<float>(display_image.rows)
                            );

                            const cv::Rect2f box = detection.box & image_bounds;
                                
                            if (box.width <= 0.0F || box.height <= 0.0F)
                            {
                                continue;
                            }

                            // 3. 绘制绿色检测框
                            cv::rectangle(display_image,  cv::Point(cvRound(box.x), cvRound(box.y)),
                                          cv::Point(cvRound(box.x + box.width), 
                                          cvRound(box.y + box.height)),
                                          cv::Scalar(0, 255, 0), 2);

                            // 4. 生成类别和置信度文本
                            std::ostringstream label;
                            label << "Class " << detection.class_id << " " << std::fixed << std::setprecision(2) << detection.confidence;

                            // 5. 在检测框附近绘制标签
                            cv::putText(display_image, label.str(),
                                        cv::Point(cvRound(box.x), std::max(20, cvRound(box.y) - 8)),
                                        cv::FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        cv::Scalar(0, 255, 0),
                                        2);
                        }

                        // 6. 显示本帧检测数量
                        cv::putText(display_image,
                                    "Detections: " + std::to_string(result.detections.size()),
                                    cv::Point(20, 40),
                                    cv::FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    cv::Scalar(0, 255, 0),
                                    2);

                        // 7. 显示绘制后的图像
                        cv::imshow("Edge AI", display_image);
                    }

                    else if (runtime.display_finished())
                    {
                        break;
                    }

                    // 主线程处理 GUI 事件
                    const int key = cv::waitKey(10);

                    if (key == 27) // ESC
                    {
                        runtime.stop();
                        break;
                    }
                    // 用户点击窗口右上角 ×
                    double visible = -1.0;
                    try
                    {
                        visible = cv::getWindowProperty("Edge AI", cv::WND_PROP_VISIBLE);
                    }
                    catch (const cv::Exception&)
                    {
                        // 某些 GUI 后端在窗口已销毁时会抛异常
                        visible = -1.0;
                    }
                    if (visible < 1.0)
                    {
                        runtime.stop();
                        break;
                    }
                }
                cv::destroyWindow("Edge AI");
            }
            
            runtime.join();
            
            return runtime.exit_code();

        }

        
        catch (const std::exception& e)
        {
            std::cerr << "[ERROR] Configuration startup failed: " << e.what() << '\n';
            return 1;
        }
    }
}   