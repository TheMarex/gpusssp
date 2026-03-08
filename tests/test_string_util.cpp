#include "common/string_util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

TEST_CASE("String split", "[string_util]")
{
    std::vector<std::string> tokens;

    SECTION("Split by single character")
    {
        gpusssp::common::detail::split(tokens, "a,b,c", ",");
        REQUIRE(tokens.size() == 3);
        CHECK(tokens[0] == "a");
        CHECK(tokens[1] == "b");
        CHECK(tokens[2] == "c");
    }

    SECTION("Split with empty tokens")
    {
        gpusssp::common::detail::split(tokens, "a,,c", ",");
        REQUIRE(tokens.size() == 3);
        CHECK(tokens[0] == "a");
        CHECK(tokens[1].empty());
        CHECK(tokens[2] == "c");
    }
}

TEST_CASE("String join", "[string_util]")
{
    SECTION("Join multiple elements")
    {
        std::vector<std::string> elements = {"a", "b", "c"};
        CHECK(gpusssp::common::detail::join(elements, ",") == "a,b,c");
    }

    SECTION("Join empty vector")
    {
        std::vector<std::string> elements;
        CHECK(gpusssp::common::detail::join(elements, ",").empty());
    }
}
