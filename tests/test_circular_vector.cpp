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
