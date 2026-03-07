#include "common/bucket_queue.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test push and pop", "[bucket_queue]")
{
    gpusssp::common::BucketQueue queue(10, 100);
    queue.push({.id = 5, .key = 200});
    queue.push({.id = 1, .key = 299});
    queue.push({.id = 3, .key = 210});
    queue.push({.id = 2, .key = 200});
    queue.push({.id = 4, .key = 200});

    REQUIRE(queue.pop().id == 4);
    REQUIRE(queue.pop().id == 2);
    REQUIRE(queue.pop().id == 5);
    REQUIRE(queue.peek().id == 3);

    queue.push({.id = 9, .key = 309});
    queue.push({.id = 8, .key = 309});

    REQUIRE(queue.pop().id == 3);
    REQUIRE(queue.peek().id == 1);

    queue.push({.id = 7, .key = 398});
}

TEST_CASE("Decrease key", "[bucket_queue]")
{
    gpusssp::common::BucketQueue queue(10, 100);
    queue.push({.id = 5, .key = 200});
    queue.push({.id = 1, .key = 299});
    queue.push({.id = 3, .key = 210});
    queue.push({.id = 2, .key = 200});
    queue.push({.id = 4, .key = 200});

    queue.decrease_key({.id = 1, .key = 200});

    REQUIRE(queue.peek().id == 1);
    queue.pop();
    REQUIRE(queue.peek().id == 4);

    queue.decrease_key({.id = 3, .key = 109});
    REQUIRE(queue.peek().id == 3);
}

TEST_CASE("Double decrease_key corrupts linked list via stale prev pointer", "[bucket_queue]")
{
    gpusssp::common::BucketQueue queue(10, 100);

    queue.push({.id = 0, .key = 10});
    queue.push({.id = 1, .key = 10});
    queue.push({.id = 2, .key = 10});

    queue.decrease_key({.id = 1, .key = 5});
    queue.decrease_key({.id = 1, .key = 3});

    REQUIRE(queue.size() == 3);

    auto first = queue.pop();
    REQUIRE(first.id == 1);
    REQUIRE(first.key == 3);

    auto second = queue.pop();
    REQUIRE(second.key == 10);

    auto third = queue.pop();
    REQUIRE(third.key == 10);

    REQUIRE(queue.empty());
}

TEST_CASE("Full empty full", "[bucket_queue]")
{
    gpusssp::common::BucketQueue queue(10, 100);

    for (uint32_t offset = 1; offset < (1 << 20); offset = offset << 1)
    {
        queue.push({.id = 5, .key = 200 + offset});
        queue.push({.id = 1, .key = 299 + offset});
        queue.push({.id = 3, .key = 210 + offset});
        queue.push({.id = 2, .key = 200 + offset});
        queue.push({.id = 4, .key = 200 + offset});
        REQUIRE(queue.peek().id == 4);

        queue.pop();
        queue.pop();
        queue.pop();
        queue.pop();
        queue.pop();
        REQUIRE(queue.size() == 0);

        queue.push({.id = 5, .key = 200});
        queue.push({.id = 1, .key = 299});
        REQUIRE(queue.peek().id == 5);

        queue.pop();
        queue.pop();
        REQUIRE(queue.size() == 0);
    }
}
