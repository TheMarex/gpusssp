#include "common/circular_vector.hpp"
#include "common/constants.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test range after inserts and pops", "[circular_vector]")
{
    gpusssp::common::CircularVector<unsigned> vec(100, gpusssp::common::INVALID_ID);
    vec.update(200, 5);
    vec.update(299, 1);
    vec.update(210, 3);
    vec.update(200, 2);

    REQUIRE(vec.front() == 2);
    REQUIRE(vec.back() == 1);
    REQUIRE(vec.front_index() == 200);
    REQUIRE(vec.back_index() == 299);

    vec.pop_front();
    REQUIRE(vec.front() == 3);
    vec.pop_front();
    REQUIRE(vec.front() == 1);
    vec.pop_front();
    REQUIRE(vec.empty());

    vec.update(399, 1);
    vec.update(300, 2);

    REQUIRE(vec.front() == 2);
    REQUIRE(vec.back() == 1);
}

TEST_CASE("Full to empty to full", "[circular_vector]")
{
    gpusssp::common::CircularVector<unsigned> vec(100, gpusssp::common::INVALID_ID);
    vec.update(200, 5);
    vec.update(299, 1);
    vec.update(210, 3);
    vec.update(200, 2);
    vec.pop_front();
    vec.pop_front();
    vec.pop_front();

    REQUIRE(vec.empty());

    vec.update(3000, 3);
    vec.update(3099, 1);

    REQUIRE(vec.size() == 100);
}

TEST_CASE("Grow preserves existing values", "[circular_vector]")
{
    const unsigned bound = 128;
    gpusssp::common::CircularVector<unsigned> vec(bound, gpusssp::common::INVALID_ID);
    vec.update(10, 1);
    vec.update(20, 2);
    vec.update(50, 3);
    vec.update(bound - 1, 4);

    vec.update((bound * 2) + 1, 99);

    REQUIRE(vec.peek(10) == 1);
    REQUIRE(vec.peek(20) == 2);
    REQUIRE(vec.peek(50) == 3);
    REQUIRE(vec.peek(bound - 1) == 4);
    REQUIRE(vec.peek((bound * 2) + 1) == 99);
}

TEST_CASE("Grow size", "[circular_vector]")
{
    const unsigned bound = 64;
    gpusssp::common::CircularVector<unsigned> vec(bound, gpusssp::common::INVALID_ID);
    vec.update(0, 7);

    const auto far_index = bound * 10;
    vec.update(far_index, 55);

    REQUIRE(vec.peek(0) == 7);
    REQUIRE(vec.peek(far_index) == 55);
    REQUIRE(vec.front_index() == 0);
    REQUIRE(vec.back_index() == far_index);
}
