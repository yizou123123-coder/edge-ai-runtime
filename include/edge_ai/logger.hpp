#pragma once

#include <string>

namespace edge_ai
{

class Logger
{
public:

    static void info(const std::string& message);

    static void warn(const std::string& message);

    static void error(const std::string& message);

};


} // namespace edge_ai