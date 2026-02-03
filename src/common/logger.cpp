#include "common/logger.hpp"

namespace gpusssp
{
namespace common
{

Logger &Logger::get()
{
    static Logger instance;
    return instance;
}

} // namespace common
} // namespace gpusssp
