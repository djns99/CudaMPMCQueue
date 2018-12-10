#include "GpuBenchmarkHeader.cuh"

__global__ void FillQueue(CudaMPMCQueue::MPMCQueue<uint8_t>* queue) {
    assert(queue->size_approx() == 0);
    const size_t capacity = queue->capacity();
    for( size_t i = 0; i < capacity; i++) {
        queue->push(0, 0x1);
    }
}

__global__ void EmptyQueue(CudaMPMCQueue::MPMCQueue<uint8_t>* queue) {
    assert(queue->size_approx() == queue->capacity());
    const size_t capacity = queue->capacity();
    for( size_t i = 0; i < capacity; i++) {
        queue->pop(0x1);
    }
}

__global__ void FillEmptyQueue(CudaMPMCQueue::MPMCQueue<uint8_t>* queue) {
    assert(queue->size_approx() == 0);
    const size_t capacity = queue->capacity();
    for( size_t i = 0; i < capacity; i++) {
        queue->push(0, 0x1);
    }

    for( size_t i = 0; i < capacity; i++) {
        queue->pop(0x1);
    }
}

__global__ void InterleavedFillEmptyQueue(CudaMPMCQueue::MPMCQueue<uint8_t>* queue) {
    const size_t capacity = queue->capacity();
    for( size_t i = 0; i < capacity; i++) {
        queue->push(0, 0x1);
        queue->pop(0x1);
    }
}

/*
 * Benchmarks filling queue
 */
void FillQueue(benchmark::State& state) {

    SETUP_BENCHMARK

    for(auto _ : state) {
        START_TIMING
        FillQueue<<<1, 1>>>(queue);
        STOP_TIMING

        CLEAR_QUEUE
    }

    CLEANUP_BENCHMARK
}

/*
 * Benchmarks emptying full queue
 */
void EmptyQueue(benchmark::State& state) {

    SETUP_BENCHMARK

    for(auto _ : state) {
        FILL_QUEUE

        START_TIMING
        EmptyQueue<<<1, 1>>>(queue);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK
}

/*
 * Benchmark filling then emptying queue
 */
void FillEmptyQueue(benchmark::State& state) {

    SETUP_BENCHMARK

    for(auto _ : state) {
        START_TIMING
        FillEmptyQueue<<<1, 1>>>(queue);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK
}

/*
 * Benchmark consecutive push and pop
 */
void InterleavedFillEmptyQueue(benchmark::State& state) {

    SETUP_BENCHMARK

    for(auto _ : state) {
        START_TIMING
        InterleavedFillEmptyQueue<<<1, 1>>>(queue);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK
}

/*
 * Benchmark consecutive push and pop, with a half full queue
 */
void InterleavedOffsetFillEmptyQueue(benchmark::State& state) {

    SETUP_BENCHMARK

    // Partially fill queue
    for(size_t i = 0; i < capacity / 2; i++) {
        queue->push(0);
    }

    for(auto _ : state) {
        START_TIMING
        InterleavedFillEmptyQueue<<<1, 1>>>(queue);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK
}

BENCHMARK(FillQueue)->UseManualTime()->RangeMultiplier(16)->Range(1, 1<<20);
BENCHMARK(EmptyQueue)->UseManualTime()->RangeMultiplier(16)->Range(1, 1<<20);
BENCHMARK(FillEmptyQueue)->UseManualTime()->RangeMultiplier(16)->Range(1, 1<<20);
BENCHMARK(InterleavedFillEmptyQueue)->UseManualTime()->RangeMultiplier(16)->Range(1, 1<<20);
BENCHMARK(InterleavedOffsetFillEmptyQueue)->UseManualTime()->RangeMultiplier(16)->Range(16, 1<<20);

BENCHMARK_MAIN();
