#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vulkan/vulkan.hpp>
#include <limits>

// Placeholder test to verify Catch2 is working
TEST_CASE("Catch2 is working", "[sanity]") {
    REQUIRE(true);
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("Vulkan headers are available", "[sanity]") {
    // Just verify we can use Vulkan types
    vk::ApplicationInfo appInfo;
    REQUIRE(appInfo.apiVersion == 0);
}
