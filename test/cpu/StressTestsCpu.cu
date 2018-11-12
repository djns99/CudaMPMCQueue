#include "CudaMPMCQueue.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <future>
#include <atomic>

#define TEST_TIMEOUT_BEGIN   {std::promise<bool> promisedFinished; \
                              auto futureResult = promisedFinished.get_future(); \
                              std::thread([&](std::promise<bool>& finished) {

#define TEST_TIMEOUT_FAIL_END(X)  finished.set_value(true); \
                                   }, std::ref(promisedFinished)).detach(); \
                                   ASSERT_TRUE(futureResult.wait_for(std::chrono::milliseconds(X)) != std::future_status::timeout);}

#define TEST_TIMEOUT_SUCCESS_END(X)  finished.set_value(true); \
                                      }, std::ref(promisedFinished)).detach(); \
                                      ASSERT_FALSE(futureResult.wait_for(std::chrono::milliseconds(X)) != std::future_status::timeout);}

using namespace CudaMPMCQueue;

class MPMCCpuTests : public ::testing::TestWithParam<std::tuple<uint64_t, uint32_t, bool>> {
protected:
    void init() {
        capacity = std::get<0>( GetParam() );
        num_threads = std::get<1>( GetParam() );
        h_mem = std::get<2>( GetParam() );
        queue = MPMCQueue<uint64_t>::allocHeterogeneousMPMCQueue(capacity, h_mem);
    }

    void TearDown() {
        MPMCQueue<uint64_t>::freeHeterogeneousMPMCQueue(queue);
    }

    uint64_t capacity;
    uint32_t num_threads;
    bool h_mem;
    MPMCQueue<uint64_t>* queue;
};

TEST_P(MPMCCpuTests, StressTest) {
    init();
    bool running = true;
    std::atomic<uint64_t> push_fails{0}, push_success{0};
    std::atomic<uint64_t> pop_fails{0}, pop_success{0};
    std::atomic<uint64_t> zero_pops{0};

    auto thread_lambda = [&](uint64_t id) {
        bool nonzero = false;
        while(running) {
            bool push = rand() % 2;
            if(push)
                if(!queue->try_push(id + 1))
                    push_fails++;
                else
                    push_success++;
            else {
                auto val = queue->try_pop();
                if(val.has_value()) {
                    ASSERT_LE(val.value(), num_threads) << "Invalid value popped";
                    if(val.value() == 0) {
                        zero_pops++;
                        ASSERT_FALSE(nonzero) << "Out of order pop";
                    } else {
                        nonzero = true;
                    }
                    pop_success++;
                } else {
                    pop_fails++;
                }
            }
        }
        optional<uint64_t> val;
        while((val = queue->try_pop()).has_value()) {
            ASSERT_LE(val.value(), num_threads) << "Invalid value popped";
            if(val.value() == 0) {
                zero_pops++;
                ASSERT_FALSE(nonzero) << "Out of order pop";
            } else {
                nonzero = true;
            }
            pop_success++;
        }
    };

    // Initialise to 50% capacity with 0s
    for(uint64_t i = 0; i < capacity / 2; i++) {
        queue->push(0);
        push_success++;
    }

    std::vector<std::thread> threads;
    for(uint32_t i = 0; i < num_threads; i++) {
        auto id = i;
        threads.emplace_back(std::bind(thread_lambda, id));
    }
    uint64_t sleep_seconds = 60ul;
    sleep(sleep_seconds);
    running = false;

    // Asserts all threads complete within reasonable time
    TEST_TIMEOUT_BEGIN
    for(auto& thread : threads) {
        thread.join();
    }
    TEST_TIMEOUT_FAIL_END(1000ul);

    queue->try_sync();
    ASSERT_EQ(queue->size_approx(), 0) << "Values left in queue";

    ASSERT_EQ(push_success, pop_success) << "Push was lost";

    ASSERT_EQ(zero_pops, capacity / 2);

    uint64_t total_pop = pop_success + pop_fails;
    uint64_t total_push = push_success + push_fails;
    std::cout << "Pops failed: " << pop_fails << "/" << total_pop << std::endl;
    std::cout << "Push failed: " << push_fails << "/" << total_push << std::endl;
}

INSTANTIATE_TEST_CASE_P(MPMCCpuTests,
                        MPMCCpuTests,
                        // Try both a high contention and low contention scenario
                        ::testing::Combine(::testing::Values(1024ul, 1024*1024ul), ::testing::Values(10u), ::testing::Values(true)));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
