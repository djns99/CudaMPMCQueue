#pragma once

#include "HeterogeneousFunctions.cuh"

namespace CudaMPMCQueue {

/*
 * Divides and rounds up
 */
template<typename T>
__host__ __device__
constexpr T ceil_div(const T& val, const T& divisor)
{
    return (val + divisor - 1) / divisor;
}

/*
 * Custom CUDA compatible implementation of optional return type
 * Can probably be improved
 */
template<typename T>
class optional {
private:
    bool set;

    union{
        T val;
        bool empty_val;
    }; // Avoid initialising new object if not set.

public:
    // Set val
    __host__ __device__
    optional(T val) : set(true), val(val) {
    }

    // Set empty
    __host__ __device__
    optional() : set(false), empty_val(true) {
    }

    // Check if value is set
    __host__ __device__
    bool has_value() const {
        return set;
    }

    // Retrieve value
    __host__ __device__
    T value() const {
        assert(set);
        return val; // only should be called after checking has_value()
    }

    // Implicit bool operator
    __host__ __device__
    operator bool() const {
        return has_value();
    }
};

/*
 * Custom CUDA compatible mutex that supports try_lock operation
 * Not intended to be used to implement blocking lock
 */
class cuMutex {
private:
    uint32_t _lock;
public:
    __host__ __device__
    cuMutex() : _lock(false) {
    }

    /*
     * Try to lock mutex
     */
    __host__ __device__
    bool try_lock() {
        // TODO Potential optimization for warps - Only one thread attempts to obtain lock
        return !_lock && !heterogeneousExch(&_lock, true); // Fail without atomic op if possible
    }

    /*
     * Release the mutex
     * NOTE: Should only be called by one thread per successful lock operation.
     * Calling by multiple parallel threads may allow multiple threads to obtain lock.
     */
    __host__ __device__
    void unlock() {
        _lock = false;
    }
};

template<class T>
class cuAtomic {
private:
    T ctr;
public:
    __host__ __device__
    cuAtomic(T val) : ctr(val) {
    }

//    __host__ __device__
//    T operator ++() {
//        return 1 + heterogeneousAdd(&ctr, 1);
//    }
//
//    __host__ __device__
//    T operator++(int) {
//        return heterogeneousAdd(&ctr, 1);
//    }
//
//    __host__ __device__
//    T atomic_or(T val) {
//        return heterogeneousOr(&ctr, val);
//    }
//
//    __host__ __device__
//    T atomic_and(T val) {
//        return heterogeneousAnd(&ctr, val);
//    }

    __host__ __device__
    T atomic_add(T val) {
        T out = heterogeneousAdd(&ctr, val);
        // Crude catch for wrapping
        assert(out < INT64_MAX - val || out > INT64_MAX);
        return out;
    }

    __host__ __device__
    T atomic_sub(T val) {
        T out = heterogeneousAdd(&ctr, -val);
        // Crude catch for wrapping.
        assert(out > INT64_MAX + val || out < INT64_MAX);
        return out;
    }

    __host__ __device__
    T value() const { return ctr; }
};

} // CudaMPMCQueue
