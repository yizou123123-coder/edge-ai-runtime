#include <edge_ai/logger.hpp>

#include <iostream>

namespace edge_ai
{

void Logger::info(const std::string &message) { std::cout << "[INFO] " << message << '\n'; }

void Logger::warn(const std::string &message) { std::cout << "[WARN] " << message << '\n'; }

void Logger::error(const std::string &message) { std::cerr << "[ERROR] " << message << '\n'; }

} // namespace edge_ai