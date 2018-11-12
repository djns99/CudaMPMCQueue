#include "CudaMPMCQueue.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <future>

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

class MPMCThreadedCpuTests : public ::testing::TestWithParam<std::tuple<uint64_t, uint32_t, bool>> {
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

    void runThreads(std::function<void(uint64_t)> func, uint64_t timeout) {
        std::vector<std::thread> threads;
        for(uint32_t i = 0; i < num_threads; i++) {
            auto id = i;
            threads.emplace_back(std::bind(func, id));
        }

        // Asserts all threads complete within reasonable time
        TEST_TIMEOUT_BEGIN
        for(auto& thread : threads) {
            thread.join();
        }
        TEST_TIMEOUT_FAIL_END(timeout);
    }

    uint64_t capacity;
    uint32_t num_threads;
    bool h_mem;
    MPMCQueue<uint64_t>* queue;
};
