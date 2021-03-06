#include "GpuBenchmarkHeader.cuh"

__global__ void PushPartEach(CudaMPMCQueue::MPMCQueue<uint8_t>* queue, size_t num_ops) {
    for(size_t i = 0; i < num_ops; i++) {
        queue->push(0, 0xffffffff);
    }
}

__global__ void PopPartEach(CudaMPMCQueue::MPMCQueue<uint8_t>* queue, size_t num_ops) {
    for(size_t i = 0; i < num_ops; i++) {
        queue->pop(0xffffffff);
    }
}

__global__ void PushPopPartEach(CudaMPMCQueue::MPMCQueue<uint8_t>* queue, size_t num_ops) {
    for(size_t i = 0; i < num_ops; i++) {
        queue->push(0, 0xffffffff);
    }

    for(size_t i = 0; i < num_ops; i++) {
        queue->pop(0xffffffff);
    }
}

__global__ void InterleavedPushPopPartEach(CudaMPMCQueue::MPMCQueue<uint8_t>* queue, size_t num_ops) {
    for(size_t i = 0; i < num_ops; i++) {
        queue->push(0, 0xffffffff);
        queue->pop(0xffffffff);
    }
}

__global__ void PushPopFullContention(CudaMPMCQueue::MPMCQueue<uint8_t>* queue) {
    const size_t capacity = queue->capacity();

    bool pushing = true;
    for(uint64_t i = 0; i < capacity; ) {
        // Use try_* methods to allow threads to avoid deadlocks
        if(pushing && queue->try_push( 0, 0xffffffff )) {
            pushing = false;
        }

        if(!pushing && queue->try_pop( 0xffffffff )) {
            pushing = true;
            i++;
        }
    }
}

/*
 * Benchmarks filling queue with multiple threads working in parallel
 */
void PushPartEach(benchmark::State& state) {

    SETUP_BENCHMARK_THREADED

    const size_t num_ops = capacity / num_threads;
    for(auto _ : state) {
        START_TIMING
        PushPartEach<<<num_blocks, threads_per_block>>>(queue, num_ops);
        STOP_TIMING

        CLEAR_QUEUE
    }

    CLEANUP_BENCHMARK_THREADED
}

/*
 * Benchmarks emptying queue with multiple threads working in parallel
 */
void PopPartEach(benchmark::State& state) {

    SETUP_BENCHMARK_THREADED

    const size_t num_ops = capacity / num_threads;
    for(auto _ : state) {

        FILL_QUEUE

        START_TIMING
        PopPartEach<<<num_blocks, threads_per_block>>>(queue, num_ops);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK_THREADED
}

/*
 * Benchmarks filling and emptying queue with multiple threads working in parallel
 */
void PushPopPartEach(benchmark::State& state) {

    SETUP_BENCHMARK_THREADED

    const size_t num_ops = capacity / num_threads;
    for(auto _ : state) {

        START_TIMING
        PushPopPartEach<<<num_blocks, threads_per_block>>>(queue, num_ops);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK_THREADED
}

/*
 * Benchmarks filling and emptying queue with multiple threads working in parallel
 */
void InterleavedPushPopPartEach(benchmark::State& state) {

    SETUP_BENCHMARK_THREADED

    const size_t num_ops = capacity / num_threads;
    for(auto _ : state) {

        START_TIMING
        InterleavedPushPopPartEach<<<num_blocks, threads_per_block>>>(queue, num_ops);
        STOP_TIMING
    }

    CLEANUP_BENCHMARK_THREADED
}

/*
 * Benchmarks filling and emptying queue with multiple threads contending for space
 */
void PushPopFullContention(benchmark::State& state) {

    SETUP_BENCHMARK_THREADED

    if((use_warps && (num_threads * capacity <= (1ul<<31))) || (!use_warps && (num_threads * capacity <= (1u<<26)))){
        for(auto _ : state) {
            START_TIMING
            PushPopFullContention<<<num_blocks, threads_per_block>>>(queue);
            STOP_TIMING
        }
    } else {
        state.SetLabel("Too Slow. Skipped");
    }

    CLEANUP_BENCHMARK_THREADED

    state.SetItemsProcessed(state.iterations() * capacity * num_threads);
}

void ArgsGenerator(benchmark::internal::Benchmark* b) {
    std::vector<uint32_t> num_threads_vec = { 1u, 32u, 256u, 1024u, 65536u };
    std::vector<uint32_t> capacities_vec = { 1u, 1024u, 65536u, 1u<<20 };
    for( const uint32_t num_threads : num_threads_vec ) {
        for( const uint32_t capacity : capacities_vec ) {
            if(num_threads <= capacity) {
                b->Args({capacity, num_threads, 0});
                if(num_threads >= 1) {
                    b->Args({capacity, num_threads, 1});
                }
            }
        }
    }
}

BENCHMARK(PushPartEach)->UseManualTime()->Apply(ArgsGenerator);
BENCHMARK(PopPartEach)->UseManualTime()->Apply(ArgsGenerator);
BENCHMARK(PushPopPartEach)->UseManualTime()->Apply(ArgsGenerator);
BENCHMARK(InterleavedPushPopPartEach)->UseManualTime()->Apply(ArgsGenerator);
BENCHMARK(PushPopFullContention)->UseManualTime()->RangeMultiplier(32)->Ranges({{1, 1<<20}, {1, 65536}, {0, 1}});

BENCHMARK_MAIN();
