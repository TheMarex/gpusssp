#include "common/coordinate.hpp"
#include "common/zorder.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("z-order bit interleaving", "[zorder]")
{
    // Test basic bit interleaving
    SECTION("interleave zero values")
    {
        auto result = gpusssp::common::morton_encode(0, 0);
        REQUIRE(result == 0);
    }

    SECTION("interleave simple values")
    {
        // x = 1 = 0b01, y = 2 = 0b10
        // x[0]=1, y[0]=0, x[1]=0, y[1]=1
        // Expected interleaving: x[0]y[0]x[1]y[1] = 1001 binary = 9
        auto result = gpusssp::common::morton_encode(1, 2);
        REQUIRE(result == 9);
    }

    SECTION("interleave alternating bits")
    {
        // x = 5 = 0b0101, y = 10 = 0b1010
        // x[0]=1, y[0]=0, x[1]=0, y[1]=1, x[2]=1, y[2]=0, x[3]=0, y[3]=1
        // Expected: x[0]y[0]x[1]y[1]x[2]y[2]x[3]y[3] = 10011001 binary = 153
        auto result = gpusssp::common::morton_encode(5, 10);
        REQUIRE(result == 153);
    }
}

TEST_CASE("coordinate to z-order conversion", "[zorder]")
{
    using namespace gpusssp::common;

    SECTION("origin coordinate")
    {
        auto coord = Coordinate{0, 0};
        auto zorder = coordinate_to_zorder(coord);
        REQUIRE(zorder > 0);
    }

    SECTION("positive coordinates")
    {
        auto coord1 = Coordinate::from_floating(10.0, 20.0);
        auto coord2 = Coordinate::from_floating(20.0, 10.0);

        auto z1 = coordinate_to_zorder(coord1);
        auto z2 = coordinate_to_zorder(coord2);

        // Different coordinates should produce different z-order values
        REQUIRE(z1 != z2);
    }

    SECTION("negative coordinates")
    {
        auto coord1 = Coordinate::from_floating(-10.0, -20.0);
        auto coord2 = Coordinate::from_floating(-20.0, -10.0);

        auto z1 = coordinate_to_zorder(coord1);
        auto z2 = coordinate_to_zorder(coord2);

        // Should handle negative coordinates correctly
        REQUIRE(z1 != z2);
        REQUIRE(z1 > 0);
        REQUIRE(z2 > 0);
    }

    SECTION("extreme coordinates")
    {
        auto coord_max = Coordinate::from_floating(180.0, 90.0);
        auto coord_min = Coordinate::from_floating(-180.0, -90.0);

        auto z_max = coordinate_to_zorder(coord_max);
        auto z_min = coordinate_to_zorder(coord_min);

        // Both should be valid
        REQUIRE(z_max > 0);
        REQUIRE(z_max != z_min);
    }

    SECTION("nearby coordinates have similar z-order values")
    {
        auto coord1 = Coordinate::from_floating(10.0, 20.0);
        auto coord2 = Coordinate::from_floating(10.1, 20.1);
        auto coord3 = Coordinate::from_floating(50.0, 60.0);

        auto z1 = coordinate_to_zorder(coord1);
        auto z2 = coordinate_to_zorder(coord2);
        auto z3 = coordinate_to_zorder(coord3);

        // Nearby coordinates should have closer z-order values than distant ones
        auto diff_nearby = std::abs(static_cast<std::int64_t>(z1 - z2));
        auto diff_far = std::abs(static_cast<std::int64_t>(z1 - z3));

        REQUIRE(diff_nearby < diff_far);
    }
}
