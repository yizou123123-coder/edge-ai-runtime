
#include <edge_ai/blocking_queue.hpp>
#include <edge_ai/inference_result.hpp>

#include <opencv2/core.hpp>

#include <iostream>
#include <stdexcept>
#include <utility>

using namespace edge_ai;

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

InferenceResult make_result(std::uint64_t id)
{
    InferenceResult result;

    result.frame.id = id;
    result.frame.image = cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));

    Detection detection;
    detection.class_id = 15;
    detection.confidence = 0.80F;
    detection.box = cv::Rect2f(1, 1, 4, 4);

    result.detections.push_back(detection);

    return result;
}

void test_latest_frame()
{
    BlockingQueue<InferenceResult> queue(1, QueuePolicy::LatestFrame);

    require(queue.push(make_result(1)), "Push 1 failed");
    require(queue.push(make_result(2)), "Push 2 failed");

    require(queue.peak_size() == 1, "Capacity exceeded");
    require(queue.dropped_count() == 1, "Drop count mismatch");

    InferenceResult result;

    require(queue.try_pop(result), "Pop failed");
    require(result.frame.id == 2, "Did not keep latest frame");

    require(!result.frame.image.empty(), "Image was lost");
    require(result.detections.size() == 1, "Detections lost");
    require(result.detections[0].class_id == 15, "Detection data mismatch");

    require(!queue.try_pop(result), "Empty queue returned data");

    std::cout << "[PASS] LatestFrame display queue\n";
}

void test_close_and_drain()
{
    BlockingQueue<InferenceResult> queue(1, QueuePolicy::LatestFrame);

    require(queue.push(make_result(7)), "Push failed");

    queue.close();

    require(!queue.push(make_result(8)), "Closed queue accepted a new result");

    InferenceResult result;

    require(queue.try_pop(result), "Closed queue failed to drain");

    require(result.frame.id == 7, "Drained frame ID mismatch");

    require(!queue.try_pop(result), "Empty closed queue returned data");

    std::cout << "[PASS] Close and drain\n";
}

int main()
{
    try
    {
        test_latest_frame();
        test_close_and_drain();

        std::cout << "\nDisplayQueue: ALL PASS\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
