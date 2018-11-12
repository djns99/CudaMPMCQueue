#include "CudaMPMCQueue.cuh"
#include <gtest/gtest.h>

using namespace CudaMPMCQueue;

__managed__ cuAtomic<uint64_t>* GPU_GTEST_FAILURE = nullptr;


// Prints first failure
#define GPU_ASSERT_TRUE_BASE(expr, ...) if(!(expr)) { if(!GPU_GTEST_FAILURE->atomic_add(1ul)) { printf("%s:%d: %s\n", __FILE__, __LINE__, __VA_ARGS__); } return; }
#define GPU_ASSERT_TRUE(expr, ...) GPU_ASSERT_TRUE_BASE(expr, #expr " was not true" "\n" __VA_ARGS__)
#define GPU_ASSERT_FALSE(expr, ...) GPU_ASSERT_TRUE_BASE(!(expr), #expr " was not false" "\n" __VA_ARGS__)
#define GPU_ASSERT_EQ(a, b, ...) GPU_ASSERT_TRUE_BASE((a) == (b), #a " == " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_NE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) != (b), #a " != " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_LE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) <= b, #a " <= " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_LT(a, b, ...) GPU_ASSERT_TRUE_BASE((a) < (b), #a " < " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_GE(a, b, ...) GPU_ASSERT_TRUE_BASE((a) >= (b), #a " >= " #b " Failed" "\n" __VA_ARGS__)
#define GPU_ASSERT_GT(a, b, ...) GPU_ASSERT_TRUE_BASE((a) > (b), #a " > " #b " Failed" "\n" __VA_ARGS__)

#define RUN_KERNEL(kernel) \
if(GPU_GTEST_FAILURE) {cudaFree(GPU_GTEST_FAILURE);} \
cudaMallocManaged(&GPU_GTEST_FAILURE, sizeof(cuAtomic<uint64_t>)); \
new(GPU_GTEST_FAILURE) cuAtomic<uint64_t>(0); \
kernel<<<blocks, threads>>>(queue, capacity, num_threads); \
ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA error"; \
ASSERT_EQ(GPU_GTEST_FAILURE->value(), 0) << GPU_GTEST_FAILURE->value() << " threads failed";

// Declare test running as a device function
#define GPU_TEST_P(cls, test) \
__device__ void device_##test(MPMCQueue<uint64_t>*, uint64_t, uint32_t);\
__global__ void kernel_##test(MPMCQueue<uint64_t>* q, uint64_t c, uint32_t n){ \
	if(threadIdx.x + blockIdx.x * blockDim.x >= n) {return;} \
	bool alloc = false; \
	if(q == nullptr) { q = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(c); alloc = true; } \
    device_##test(q, c, n); \
    if(alloc) { MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(q); } \
}\
TEST_P(cls, test) { \
    init(); \
    RUN_KERNEL(kernel_##test) \
} \
__device__ void device_##test(MPMCQueue<uint64_t>* queue, uint64_t capacity, uint32_t num_threads)

class MPMCGpuTests : public ::testing::TestWithParam<std::tuple<uint64_t, uint32_t, bool>> {
protected:
    void init() {
        capacity = std::get<0>(GetParam());
        num_threads = std::get<1>(GetParam());
        blocks = ceil_div(num_threads, threads);

        host_alloc = std::get<2>(GetParam());

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
};
