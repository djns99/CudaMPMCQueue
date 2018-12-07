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
    const size_t threads_per_block = std::min(256ul, num_threads); \
    const size_t num_blocks = (threads_per_block + (num_threads - 1)) / num_threads;

#define CLEANUP_BENCHMARK \
    state.counters["Capacity"] = capacity; \
    state.SetItemsProcessed(state.iterations() * capacity); \
    CudaMPMCQueue::MPMCQueue<uint8_t>::freeHeterogeneousMPMCQueue(queue);

#define CLEANUP_BENCHMARK_THREADED \
    state.counters["Num Threads"] = num_threads; \
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
