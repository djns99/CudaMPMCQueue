#include "GpuTestHeader.cuh"

__device__ void check_different(MPMCQueue<uint64_t>* queue, size_t tid) {
    size_t capacity = queue->capacity();

    for(size_t i = 0; i < capacity; i++) {
        GPU_ASSERT_EQ(queue->size_approx(), i, "Size did not match expected");
        bool push = queue->try_push(tid*capacity + i);
        GPU_ASSERT_TRUE(push, "Failed to push");
        queue->try_sync();
    }

    for(size_t i = 0; i < capacity; i++) {
        GPU_ASSERT_EQ(queue->size_approx(), capacity - i, "Size did not match expected");
        auto res = queue->try_pop();
        GPU_ASSERT_TRUE(res.has_value(), "Failed to pop");;
        GPU_ASSERT_EQ(res.value(), tid*capacity + i, "Incorrect value popped");
        queue->try_sync();
    }

    GPU_ASSERT_EQ(queue->size_approx(), 0, "Size did not match expected");
}

/*
 * If we aren't careful warp cooperation could end up mutating the wrong object
 * Do a basic test to verify locally owned objects are not leaking into other threads
 */
GPU_TEST_P_NO_SYNC(MPMCGpuTests, LocalIsLocal) {
    MPMCQueue<uint64_t> local(capacity);
    check_different(&local, tid);
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, DifferentIsDifferent) {

    assert(blockDim.x == 256);
    __shared__ MPMCQueue<uint64_t>* queues[256];

    queues[tid % blockDim.x] = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity);

    // Check our queue is local
    check_different(queues[tid % blockDim.x], tid);

    __syncthreads();

    // Check neighbour's queue is local when we use it
    check_different(queues[(tid + 1) % blockDim.x], tid);
}


INSTANTIATE_TEST_CASE_P(MPMCGpuTests,
                        MPMCGpuTests,
                        ::testing::Combine(::testing::Values(1024ul), ::testing::Values(256u), ::testing::Values(true)));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
