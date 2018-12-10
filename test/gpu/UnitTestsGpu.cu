#include "GpuTestHeader.cuh"

GPU_TEST_P_NO_SYNC(MPMCGpuTests, PushIncreaseSize) {
    assert(num_threads == 1);

    for(uint64_t i = 0; i < capacity; i++) {
        GPU_ASSERT_TRUE(queue->try_push(i), "Failed to push");
        queue->try_sync();
        GPU_ASSERT_EQ(queue->size_approx(), i + 1, "Incorrect size");
    }
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, PopDecreaseSize) {
    assert(num_threads == 1);

    for(uint64_t i = 0; i < capacity; i++) {
        GPU_ASSERT_TRUE(queue->try_push(i));
    }

    optional<uint64_t> out;
    for(uint64_t i = capacity; i > 0; i--) {
        GPU_ASSERT_EQ(queue->size_approx(), i);
        GPU_ASSERT_TRUE((out = queue->try_pop()));
    }
    GPU_ASSERT_EQ(queue->size_approx(), 0);
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, PushPopOrdering) {
    assert(num_threads == 1);

    for(uint64_t i = 0; i < capacity; i++) {
        GPU_ASSERT_TRUE(queue->try_push(i));
    }

    GPU_ASSERT_FALSE(queue->try_push(capacity));

    queue->try_sync();
    GPU_ASSERT_EQ(queue->size_approx(), capacity);

    optional<uint64_t> out;
    for(uint64_t i = 0; i < capacity; i++) {
        GPU_ASSERT_TRUE(out = queue->try_pop());
        GPU_ASSERT_EQ(out.value(), i);
    }

    queue->try_sync();
    GPU_ASSERT_EQ(queue->size_approx(), 0);

    GPU_ASSERT_FALSE((out = queue->try_pop()));

    queue->try_sync();
    GPU_ASSERT_EQ(queue->size_approx(), 0);
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, UsePoppedSpaces) {
    assert(num_threads == 1);

    // Fill then empty same q multiple times
    for(uint64_t j = 0; j < 5; j++) {
        for(uint64_t i = 0; i < capacity; i++) {
            GPU_ASSERT_TRUE(queue->try_push(i));
        }

        GPU_ASSERT_FALSE(queue->try_push(capacity));

        queue->try_sync();
        GPU_ASSERT_EQ(queue->size_approx(), capacity);

        optional<uint64_t> out;
        for(uint64_t i = 0; i < capacity; i++) {
            GPU_ASSERT_TRUE(out = queue->try_pop());
            GPU_ASSERT_EQ(out.value(), i);
        }

        queue->try_sync();
        GPU_ASSERT_EQ(queue->size_approx(), 0);

        GPU_ASSERT_FALSE((out = queue->try_pop()));

        queue->try_sync();
        GPU_ASSERT_EQ(queue->size_approx(), 0);
    }
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, PopHalfFull) {
    assert(num_threads == 1);

    // Fill then empty same q multiple times
    for(uint64_t i = 0; i < ceil_div(capacity, 2ul); i++) {
        GPU_ASSERT_TRUE(queue->try_push(i));
    }

    queue->try_sync();
    uint64_t size = queue->size_approx();
    GPU_ASSERT_EQ(size, ceil_div(capacity, 2ul));


    optional<uint64_t> out;
    for(uint64_t i = 0; i < size; i++) {
        GPU_ASSERT_TRUE(out = queue->try_pop());
        GPU_ASSERT_EQ(out.value(), i);
    }

    GPU_ASSERT_FALSE(out = queue->try_pop());

    queue->try_sync();
    GPU_ASSERT_EQ(queue->size_approx(), 0);
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, PushPopDecreasingNum) {
    assert(num_threads == 1);

    uint64_t amount = capacity;

    uint64_t last_push = 0;
    uint64_t last_pop = 0;
    while(amount > 0) {
        // Fill then empty same q multiple times
        for(uint64_t i = 0; i < amount; i++) {
            GPU_ASSERT_TRUE(queue->try_push(last_push++));
        }

        queue->try_sync();

        amount /= 2;

        optional<uint64_t> out;
        for(uint64_t i = 0; i < amount; i++) {
            GPU_ASSERT_TRUE(out = queue->try_pop());
            GPU_ASSERT_EQ(out.value(), last_pop++);
        }

        queue->try_sync();
    }
}

INSTANTIATE_TEST_CASE_P(MPMCGpuTests,
                        MPMCGpuTests,
                        ::testing::Combine(::testing::Values(1, 10, 1024, 65536, 65537, 128*1024), ::testing::Values(1u), ::testing::Bool(), ::testing::Values(false)));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
