#pragma once

#include <vector>

namespace edge_ai
{

class IInferenceEngine
{
  public:
    virtual ~IInferenceEngine() = default;

    virtual std::vector<float> infer(const std::vector<float> &input) = 0;
};

} // namespace edge_ai
