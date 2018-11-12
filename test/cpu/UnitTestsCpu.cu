#include "CudaMPMCQueue.cuh"
#include <gtest/gtest.h>

using namespace CudaMPMCQueue;

class MPMCCpuTests : public ::testing::TestWithParam<std::tuple<uint64_t, bool>> {

};

TEST_P(MPMCCpuTests, PushIncreaseSize) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    for(uint64_t i = 0; i < capacity; i++) {
        ASSERT_TRUE(queue.try_push(i));
        queue.try_sync();
        ASSERT_EQ(queue.size_approx(), i + 1);
    }
}

TEST_P(MPMCCpuTests, PopDecreaseSize) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    for(uint64_t i = 0; i < capacity; i++) {
        ASSERT_TRUE(queue.try_push(i));
    }

    optional<uint64_t> out;
    for(uint64_t i = capacity; i > 0; i--) {
        ASSERT_EQ(queue.size_approx(), i);
        ASSERT_TRUE(out = queue.try_pop());
    }
    ASSERT_EQ(queue.size_approx(), 0);
}

TEST_P(MPMCCpuTests, PushPopOrdering) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    for(uint64_t i = 0; i < capacity; i++) {
        ASSERT_TRUE(queue.try_push(i));
    }

    ASSERT_FALSE(queue.try_push(capacity));

    queue.try_sync();
    ASSERT_EQ(queue.size_approx(), capacity);

    optional<uint64_t> out;
    for(uint64_t i = 0; i < capacity; i++) {
        ASSERT_TRUE(out = queue.try_pop());
        ASSERT_EQ(out.value(), i);
    }

    queue.try_sync();
    ASSERT_EQ(queue.size_approx(), 0);

    ASSERT_FALSE(out = queue.try_pop());

    queue.try_sync();
    ASSERT_EQ(queue.size_approx(), 0);
}

TEST_P(MPMCCpuTests, UsePoppedSpaces) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    // Fill then empty same queue multiple times
    for(uint64_t j = 0; j < 5; j++) {
        for(uint64_t i = 0; i < capacity; i++) {
            ASSERT_TRUE(queue.try_push(i));
        }

        ASSERT_FALSE(queue.try_push(capacity));

        queue.try_sync();
        ASSERT_EQ(queue.size_approx(), capacity);

        optional<uint64_t> out;
        for(uint64_t i = 0; i < capacity; i++) {
            ASSERT_TRUE(out = queue.try_pop());
            ASSERT_EQ(out.value(), i);
        }

        queue.try_sync();
        ASSERT_EQ(queue.size_approx(), 0);

        ASSERT_FALSE(out = queue.try_pop());

        queue.try_sync();
        ASSERT_EQ(queue.size_approx(), 0);
    }
}

TEST_P(MPMCCpuTests, PopHalfFull) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    // Fill then empty same queue multiple times
    for(uint64_t i = 0; i < ceil_div(capacity, 2ul); i++) {
        ASSERT_TRUE(queue.try_push(i));
    }

    queue.try_sync();
    uint64_t size = queue.size_approx();
    ASSERT_EQ(size, ceil_div(capacity, 2ul));


    optional<uint64_t> out;
    for(uint64_t i = 0; i < size; i++) {
        ASSERT_TRUE(out = queue.try_pop());
        ASSERT_EQ(out.value(), i);
    }

    ASSERT_FALSE(out = queue.try_pop());

    queue.try_sync();
    ASSERT_EQ(queue.size_approx(), 0);
}

TEST_P(MPMCCpuTests, PushPopDecreasingNum) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    MPMCQueue<uint64_t> queue(capacity, h_mem);

    uint64_t amount = capacity;

    uint64_t last_push = 0;
    uint64_t last_pop = 0;
    while(amount > 0) {
        // Fill then empty same queue multiple times
        for(uint64_t i = 0; i < amount; i++) {
            ASSERT_TRUE(queue.try_push(last_push++));
        }

        queue.try_sync();

        amount /= 2;

        optional<uint64_t> out;
        for(uint64_t i = 0; i < amount; i++) {
            ASSERT_TRUE(out = queue.try_pop());
            ASSERT_EQ(out.value(), last_pop++);
        }

        queue.try_sync();
    }
}

TEST_P(MPMCCpuTests, HeterogeneousAlloc) {
    const uint64_t capacity = std::get<0>( GetParam() );
    const bool h_mem = std::get<1>( GetParam() );
    auto queue = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity, h_mem);
    ASSERT_EQ(queue->capacity(), capacity);
    MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(queue);
}

INSTANTIATE_TEST_CASE_P(MPMCCpuTests,
                        MPMCCpuTests,
                        ::testing::Combine(::testing::Values(1, 10, 1024, 65536, 65537, 1024*1024), ::testing::Bool()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
