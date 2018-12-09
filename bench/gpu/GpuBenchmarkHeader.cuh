#include <benchmark/benchmark.h>
#include "CudaMPMCQueue.cuh"

#define SETUP_BENCHMARK \
    const size_t capacity = state.range(0); \
    CudaMPMCQueue::MPMCQueue<uint8_t>* const queue = CudaMPMCQueue::MPMCQueue<uint8_t>::allocHeterogeneousMPMCQueue(capacity, true); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);

#define SETUP_BENCHMARK_THREADED \
    SETUP_BENCHMARK \
    const size_t num_threads = state.range(1); \
    const bool use_warps = state.range(2); \
    const size_t threads_per_block = use_warps ? std::min(256ul, num_threads) : 1; \
    const size_t num_blocks = use_warps ? ((num_threads + (threads_per_block - 1)) / threads_per_block) : num_threads;

#define CLEANUP_BENCHMARK \
    state.counters["Capacity"] = capacity; \
    state.SetItemsProcessed(state.iterations() * capacity); \
    CudaMPMCQueue::MPMCQueue<uint8_t>::freeHeterogeneousMPMCQueue(queue);

#define CLEANUP_BENCHMARK_THREADED \
    state.counters["Num Threads"] = num_threads; \
    if(use_warps) { state.SetLabel("Using Warp Optimizations"); } \
    CLEANUP_BENCHMARK

#define START_TIMING \
    queue->try_sync(); \
    cudaEventRecord(start);

#define STOP_TIMING \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaDeviceSynchronize(); \
    float milliseconds = 0; \
    cudaEventElapsedTime(&milliseconds, start, stop); \
    state.SetIterationTime(milliseconds / 1000.0);

#define CLEAR_QUEUE \
        queue->try_sync(); \
        while(queue->size_approx()) { \
            queue->try_pop(); \
            queue->try_sync(); \
        }

#define FILL_QUEUE \
        queue->try_sync(); \
        while(queue->size_approx() != queue->capacity()) { \
            queue->push(0); \
            queue->try_sync(); \
        }
