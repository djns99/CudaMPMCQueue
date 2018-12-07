#include "GpuTestHeader.cuh"

__device__ void push_func(MPMCQueue<uint64_t>* queue, uint64_t id, uint64_t start_val, uint32_t num, uint32_t step) {
    for(uint64_t i = 0; i < num; i++) {
        queue->push(start_val + id + step*i);
    }
}

__device__ void pop_func(MPMCQueue<uint64_t>* queue, uint64_t id, uint64_t min, uint64_t max, uint32_t num) {
    for(uint64_t i = 0; i < num; i++) {
        auto val = queue->pop();
        GPU_ASSERT_GE(val, min, "Value was less than min");
        GPU_ASSERT_LE(val, max, "Value was greater than max");
    }
}

GPU_TEST_P(MPMCGpuTests, PushPartCapacityEach) {
    const auto op_per_thread = capacity / (uint64_t)num_threads;
    const auto extra_op_thresh = capacity - op_per_thread * num_threads;

    uint32_t num = op_per_thread;
    num += (tid < extra_op_thresh);
    for(uint32_t j = 0; j < 5; j++) {

        push_func(queue, tid, capacity * j, num, num_threads);

        grid.sync();
        queue->try_sync();
        grid.sync();

        GPU_ASSERT_EQ(queue->size_approx(), capacity, "Not enough values pushed");

        pop_func(queue, tid, capacity * j, capacity * (j+1) - 1, num);

        grid.sync();
        queue->try_sync();
        grid.sync();

        GPU_ASSERT_EQ(queue->size_approx(), 0, "Not enough values popped");

        grid.sync();
    }
}

GPU_TEST_P(MPMCGpuTests, PushSyncPop) {
    if(capacity == 1)
        return;
    uint32_t half_capacity = capacity / 2;
    const auto op_per_thread = half_capacity / (uint64_t)num_threads;
    const auto extra_op_thresh = half_capacity - op_per_thread * num_threads;

    uint32_t num = op_per_thread;
    num += (tid < extra_op_thresh);

    push_func(queue, tid, 0, num, num_threads);

    grid.sync();
    queue->try_sync();
    grid.sync();

    GPU_ASSERT_EQ(queue->size_approx(), half_capacity, "Not enough values pushed");

    push_func(queue, tid, half_capacity, num, num_threads);

    grid.sync();
    queue->try_sync();
    grid.sync();

    GPU_ASSERT_EQ(queue->size_approx(), half_capacity * 2, "Not enough values pushed");

    // Relative ordering not guaranteed. Just check popped values are from first batch
    pop_func(queue, tid, 0, half_capacity - 1, num);

    grid.sync();
    queue->try_sync();
    grid.sync();

    GPU_ASSERT_EQ(queue->size_approx(), half_capacity, "Not enough values popped");

    pop_func(queue, tid, half_capacity, half_capacity * 2 - 1, num);

    grid.sync();
    queue->try_sync();
    grid.sync();

    GPU_ASSERT_EQ(queue->size_approx(), 0, "Not enough values popped");
}

/*
 * Test high contention scenario where all threads are trying to produce/consume capacity elements.
 */
GPU_TEST_P(MPMCGpuTests, PushPopCapacityEach) {
    bool pushing = true;

    for(uint64_t i = 0; i < capacity; ) {
        // Use try_* methods to allow threads to re-converge and avoid deadlocks
        if(pushing && queue->try_push(tid)) {
            pushing = false;
        }

        optional<uint64_t> val;
        if(!pushing && (val = queue->try_pop())) {
            pushing = true;
            GPU_ASSERT_LT(val, num_threads, "Invalid value popped");
            i++;
        }
    }
}

INSTANTIATE_TEST_CASE_P(MPMCGpuTests,
                        MPMCGpuTests,
                        ::testing::Combine(::testing::Values(1, 10, 65536, 65537, 128*1024), ::testing::Values(256, 1024), ::testing::Bool()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
