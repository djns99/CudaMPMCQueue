#include "CpuTestHeader.cuh"

TEST_P(MPMCThreadedCpuTests, PushPopCapacityEach) {
    init();
    auto thread_lambda = [&](uint64_t id) {
        for(uint64_t i = 0; i < capacity; i++) {
            queue->push(id);
            auto val = queue->pop();
            ASSERT_LT(val, num_threads) << "Invalid value popped";
        }
    };

    runThreads(thread_lambda, 10000ul);
}

TEST_P(MPMCThreadedCpuTests, PushPartCapacityEach) {
    init();
    const auto op_per_thread = capacity / (uint64_t)num_threads;
    const auto extra_op_thresh = capacity - op_per_thread * num_threads;

    auto push_lambda = [&](uint64_t id, uint64_t start_val) {
        uint32_t num = op_per_thread;
        num += (id < extra_op_thresh);
        for(uint64_t i = 0; i < num; i++) {
            queue->push(start_val + id + num_threads*i);
        }
    };
    auto pop_lambda = [&](uint64_t id, uint64_t min, uint64_t max) {
        uint32_t num = op_per_thread;
        num += (id < extra_op_thresh);
        for(uint64_t i = 0; i < num; i++) {
            auto val = queue->pop();
            EXPECT_GE(val, min) << "Value was less than min";
            EXPECT_LE(val, max) << "Value was greater than max";
        }
    };

    for(uint32_t j = 0; j < 5; j++) {

        runThreads(std::bind(push_lambda, std::placeholders::_1, capacity * j), 1000ul);

        queue->try_sync();
        ASSERT_EQ(queue->size_approx(), capacity) << "Not enough values pushed";

        runThreads(std::bind(pop_lambda, std::placeholders::_1, capacity * j, capacity * (j+1) - 1), 1000ul);

        queue->try_sync();
        ASSERT_EQ(queue->size_approx(), 0) << "Not enough values popped";
    }
}

TEST_P(MPMCThreadedCpuTests, PushSyncPop) {
    init();
    if(capacity == 1)
        return;
    uint32_t half_capacity = capacity / 2;
    const auto op_per_thread = half_capacity / (uint64_t)num_threads;
    const auto extra_op_thresh = half_capacity - op_per_thread * num_threads;

    auto push_lambda = [&](uint64_t id, uint64_t start_val) {
        uint32_t num = op_per_thread;
        num += (id < extra_op_thresh);
        for(uint64_t i = 0; i < num; i++) {
            queue->push(start_val + id + num_threads*i);
        }
    };
    auto pop_lambda = [&](uint64_t id, uint64_t min, uint64_t max) {
        uint32_t num = op_per_thread;
        num += (id < extra_op_thresh);
        for(uint64_t i = 0; i < num; i++) {
            auto val = queue->pop();
            EXPECT_GE(val, min) << "Value was less than min";
            EXPECT_LE(val, max) << "Value was greater than max";
        }
    };

    std::vector<std::thread> threads;

    runThreads(std::bind(push_lambda, std::placeholders::_1, 0), 1000ul);

    queue->try_sync();
    ASSERT_EQ(queue->size_approx(), half_capacity) << "Not enough values pushed";

    runThreads(std::bind(push_lambda, std::placeholders::_1, capacity), 1000ul);

    queue->try_sync();
    ASSERT_EQ(queue->size_approx(), half_capacity * 2) << "Not enough values pushed";

    // Relative ordering not guaranteed. Check popped values are from first batch
    runThreads(std::bind(pop_lambda, std::placeholders::_1, 0, capacity - 1), 1000ul);

    queue->try_sync();
    ASSERT_EQ(queue->size_approx(), half_capacity) << "Not enough values popped";

    runThreads(std::bind(pop_lambda, std::placeholders::_1, capacity, capacity * 2 - 1), 1000ul);

    queue->try_sync();
    ASSERT_EQ(queue->size_approx(), 0) << "Not enough values popped";
}


INSTANTIATE_TEST_CASE_P(MPMCThreadedCpuTests,
                        MPMCThreadedCpuTests,
                        ::testing::Combine(::testing::Values(1, 10, 65536, 65537, 1024*1024), ::testing::Values(4, 10), ::testing::Bool()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
