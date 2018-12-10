#include "CudaMPMCQueue.cuh"
#include <gtest/gtest.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace CudaMPMCQueue;
using namespace cooperative_groups;

__managed__ cuAtomic<uint64_t>* GPU_GTEST_FAILURE = nullptr;


// Prints first failure
#define GPU_ASSERT_TRUE_BASE(expr, ...) if(!(expr)) { \
		                                    if(!GPU_GTEST_FAILURE->atomic_add(1ul)) { \
                                                printf("%s:%d: %s\n", __FILE__, __LINE__, __VA_ARGS__); \
                                            } \
                                            return; \
                                        } else if (GPU_GTEST_FAILURE->value()) { \
                                            return; \
                                        }
#define GPU_ASSERT_TRUE(expr, ...) GPU_ASSERT_TRUE_BASE(expr, #expr " was not true" "\n" __VA_ARGS__)
#define GPU_ASSERT_FALSE(expr, ...) GPU_ASSERT_TRUE_BASE(!(expr), #expr " was not false" "\n" __VA_ARGS__)
#define GPU_ASSERT_EQ(a, b, ...) GPU_ASSERT_TRUE_BASE((a) == (b), #a " == " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_NE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) != (b), #a " != " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_LE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) <= b, #a " <= " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_LT(a, b, ...) GPU_ASSERT_TRUE_BASE((a) < (b), #a " < " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_GE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) >= (b), #a " >= " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_GT(a, b, ...) GPU_ASSERT_TRUE_BASE((a) > (b), #a " > " #b " Failed" "\n" __VA_ARGS__)

#define RUN_KERNEL(kernel, sync) \
    if(GPU_GTEST_FAILURE) {cudaFree(GPU_GTEST_FAILURE);} \
    cudaMallocManaged(&GPU_GTEST_FAILURE, sizeof(cuAtomic<uint64_t>)); \
    new(GPU_GTEST_FAILURE) cuAtomic<uint64_t>(0); \
    if (sync) { \
        void* args[] = {(void*)&queue, (void*)&capacity, (void*)&num_threads, (void*)&use_warp }; \
        int occupancy; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, threads, 0); \
        int dev; cudaGetDevice(&dev); \
        cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, dev); \
        ASSERT_NE(occupancy, 0) << "Cannot run kernel with " << threads << " threads"; \
        ASSERT_LE(ceil_div<uint64_t>(blocks, occupancy), deviceProp.multiProcessorCount) << "Too many threads: " << blocks << "/" << occupancy; \
        ASSERT_EQ(cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, args), cudaSuccess) << "Launch failed"; \
    } else { \
        kernel<<<blocks, threads>>>(queue, capacity, num_threads, use_warp); \
    } \
    auto err = cudaDeviceSynchronize(); /* Check error after checking ASSERT flag */ \
    ASSERT_EQ(GPU_GTEST_FAILURE->value(), 0) << GPU_GTEST_FAILURE->value() << " threads failed"; \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error";

// Declare test running as a device function
#define GPU_TEST_P(cls, test) \
__device__ void device_##test(MPMCQueue<uint64_t>*, uint64_t, uint32_t, grid_group, uint32_t, uint32_t);\
__device__ MPMCQueue<uint64_t>* q_##test; \
__global__ void kernel_##test(MPMCQueue<uint64_t>* q, uint64_t c, uint32_t n, bool w){ \
    grid_group g = this_grid(); \
    uint32_t t = g.thread_rank(); \
	if(t >= n) {return;} \
	bool alloc = false; \
	if(q == nullptr) { \
		if(t == 0) { \
		    q_##test = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(c); \
		} \
		g.sync(); \
		q = q_##test; \
		alloc = true; \
	} \
	uint32_t m = 0; \
    if(w) { m = q->register_warp(); } \
    device_##test(q, c, n, g, t, m); \
    g.sync(); \
    if(alloc && t == 0) { MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(q); } \
} \
TEST_P(cls, test) { \
    init(); \
    RUN_KERNEL(kernel_##test, true) \
} \
__device__ void device_##test(MPMCQueue<uint64_t>* queue, uint64_t capacity, uint32_t num_threads, grid_group grid, uint32_t tid, uint32_t active_mask)

#define GPU_TEST_P_NO_SYNC(cls, test) \
__device__ void device_##test(MPMCQueue<uint64_t>*, uint64_t, uint32_t, uint32_t, uint32_t);\
__global__ void kernel_##test(MPMCQueue<uint64_t>* q, uint64_t c, uint32_t n, bool w){ \
    uint32_t t = threadIdx.x + blockDim.x * blockIdx.x; \
    if(t >= n) {return;} \
    bool alloc = false; \
    if(q == nullptr) { q = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(c); alloc = true; } \
    uint32_t m = 0; \
    if(w) { m = q->register_warp(); } \
    device_##test(q, c, n, t, m); \
    if(alloc) { MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(q); } \
}\
TEST_P(cls, test) { \
    init(); \
    RUN_KERNEL(kernel_##test, false) \
} \
__device__ void device_##test(MPMCQueue<uint64_t>* queue, uint64_t capacity, uint32_t num_threads, uint32_t tid, uint32_t active_mask)

class MPMCGpuTests : public ::testing::TestWithParam<std::tuple<uint64_t, uint32_t, bool, bool>> {
protected:
    void init() {
        capacity = std::get<0>(GetParam());
        num_threads = std::get<1>(GetParam());
        blocks = ceil_div(num_threads, threads);

        host_alloc = std::get<2>(GetParam());

        use_warp = std::get<3>(GetParam());

        if(host_alloc)
            queue = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity);
    }

    void TearDown() {
        if(queue)
            MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(queue);
    }

    MPMCQueue<uint64_t>* queue = nullptr;
    uint64_t capacity;
    uint32_t num_threads;

    // Not availible in kernel
    const uint32_t threads = 256;
    uint32_t blocks;

    // queue param will be nullptr on test start
    bool host_alloc;

    bool use_warp;
};
