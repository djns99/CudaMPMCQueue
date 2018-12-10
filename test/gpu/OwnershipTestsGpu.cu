#include "GpuTestHeader.cuh"

__device__ void check_different(MPMCQueue<uint64_t>* queue, size_t tid, uint32_t active_mask) {
    size_t capacity = queue->capacity();

    if(active_mask != 0) {
        // Not using the default queue for these tests - use our own
        active_mask = queue->register_warp();
    }

    for(size_t i = 0; i < capacity; i++) {
        GPU_ASSERT_EQ(queue->size_approx(), i, "Size did not match expected");
        bool push = queue->try_push(tid*capacity + i, active_mask);
        GPU_ASSERT_TRUE(push, "Failed to push");
        queue->try_sync(active_mask);
    }

    for(size_t i = 0; i < capacity; i++) {
        GPU_ASSERT_EQ(queue->size_approx(), capacity - i, "Size did not match expected");
        auto res = queue->try_pop(active_mask);
        GPU_ASSERT_TRUE(res.has_value(), "Failed to pop");;
        GPU_ASSERT_EQ(res.value(), tid*capacity + i, "Incorrect value popped");
        queue->try_sync(active_mask);
    }

    GPU_ASSERT_EQ(queue->size_approx(), 0, "Size did not match expected");
}

/*
 * If we aren't careful warp cooperation could end up mutating the wrong object
 * Do a basic test to verify locally owned objects are not leaking into other threads
 */
GPU_TEST_P_NO_SYNC(MPMCGpuTests, LocalIsLocal) {
    MPMCQueue<uint64_t> local(capacity);
    check_different(&local, tid, active_mask);
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, DifferentIsDifferent) {

    assert(blockDim.x == 256);
    __shared__ MPMCQueue<uint64_t>* queues[256];

    queues[threadIdx.x] = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity);

    __syncthreads();

    // Check our queue is not shared
    check_different(queues[threadIdx.x], tid, active_mask);

    __syncthreads();

    // Check neighbour's queue is not shared when we use it
    check_different(queues[(threadIdx.x + 1) % blockDim.x], tid, active_mask);

    __syncthreads();

    MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue( queues[threadIdx.x] );
}

GPU_TEST_P_NO_SYNC(MPMCGpuTests, SharedIsShared) {
    __shared__ MPMCQueue<uint64_t>* queues[1];
    __shared__ uint32_t successes[1];
    if(threadIdx.x == 0) {
        successes[0] = 0;
        queues[0] = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity);
    }
    __syncthreads();

    MPMCQueue<uint64_t>* q = queues[0];
    if(active_mask != 0 ) {
        active_mask = q->register_warp();
    }

    if(q->try_push(tid, active_mask)) {
        atomicAdd(successes, 1ul);
    }

    __syncthreads();

    q->try_sync(active_mask);

    __syncthreads();

    GPU_ASSERT_EQ(successes[0], q->size_approx(), "Some pushes were lost");

    __syncthreads();

    if(threadIdx.x == 0) {
        MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(queues[0]);
    }
}


INSTANTIATE_TEST_CASE_P(MPMCGpuTests,
                        MPMCGpuTests,
                        ::testing::Combine(::testing::Values(1024ul), ::testing::Values(256u), ::testing::Values(true), ::testing::Bool()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
