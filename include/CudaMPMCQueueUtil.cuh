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
    const bool _is_local;
public:
    __host__ __device__
    cuMutex() :
        _lock(false),
        _is_local(isLocal(this)){

    }

    /*
     * Try to lock mutex
     */
    __host__ __device__
    bool try_lock() {
        if(_is_local)
            return true; // Local __device__ variable always succeeds
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
    T _ctr;
    const bool _is_local;
public:
    __host__ __device__
    cuAtomic(T val) :
        _ctr(val),
        _is_local(isLocal(this))
    {
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
        T out;
        if(!_is_local) {
            out = heterogeneousAdd(&_ctr, val);
        } else {
            // Local var. No atomics
            out = _ctr;
            _ctr += val;
        }
        // Crude catch for wrapping
        assert(out < INT64_MAX - val || out > INT64_MAX);
        return out;
    }

    __host__ __device__
    T atomic_sub(T val) {
        T out;
        if(!_is_local) {
            out = heterogeneousAdd(&_ctr, -val);
        } else {
            // Local var. No atomics
            out = _ctr;
            _ctr -= val;
        }
        // Crude catch for wrapping.
        assert(out > INT64_MAX + val || out < INT64_MAX);
        return out;
    }

    __host__ __device__
    T value() const { return _ctr; }
};


typedef uint32_t bitmap_entry;
typedef uint32_t bitmap_mask;

class cuBitmap {
private:
   const size_t _bitmap_len;
   bitmap_entry* const _bitmap_buffer;
   const bool _using_heterogeneous_mem;
   const bool _is_local;
public:

   __host__ __device__
    cuBitmap(size_t len, bool use_heterogeneous_mem) :
       _bitmap_len(len),
       _bitmap_buffer(heterogeneousAlloc<bitmap_entry>(_bitmap_len, use_heterogeneous_mem)),
       _using_heterogeneous_mem(use_heterogeneous_mem),
       _is_local(isLocal(this))
    {
        if(_bitmap_buffer == nullptr) {
            printf("Failed to allocate %lu bitmap entries\n", _bitmap_len);
            assert(false);
            return;
        }
        heterogeneousMemset(_bitmap_buffer, 0x0, _bitmap_len * sizeof(bitmap_entry), _using_heterogeneous_mem);
    }

   __host__ __device__
    ~cuBitmap() {
        heterogeneousFree(_bitmap_buffer, _using_heterogeneous_mem);
    }

    /*
     * Sets specified bit
     */
   __host__ __device__
    void setBit(size_t bit_idx) {
       if(_is_local)
           _bitmap_buffer[bit_idx >> 5] |= 1 << (bit_idx & 31);
       else
           heterogeneousOr(_bitmap_buffer + (bit_idx >> 5), 1 << (bit_idx & 31));
    }

    /*
     * Clears specified bit
     */
   __host__ __device__
    void clearBit(size_t bit_idx) {
       if(_is_local)
           _bitmap_buffer[bit_idx >> 5] &= ~(1 << (bit_idx & 31));
       else
           heterogeneousAnd(_bitmap_buffer + (bit_idx >> 5), ~(1 << (bit_idx & 31)));
    }

    /*
     * Sets all bits set in mask
     *
     * entry_idx The bitmap entry to operate on (bit_idx / 32)
     */
   __host__ __device__
    void setMask(size_t entry_idx, bitmap_mask mask) {
       if(_is_local)
           _bitmap_buffer[entry_idx] |= mask;
       else
           heterogeneousOr(_bitmap_buffer + entry_idx, mask);
    }

    /*
     * Clear all bits cleared in mask
     *
     * entry_idx The bitmap entry to operate on (bit_idx / 32)
     */
   __host__ __device__
    void clearMask(size_t entry_idx, bitmap_mask mask) {
       if(_is_local)
           _bitmap_buffer[entry_idx] &= mask;
       else
           heterogeneousAnd(_bitmap_buffer + entry_idx, mask);
    }

   __host__ __device__
    bitmap_entry operator[](size_t idx) const {
        assert(idx < _bitmap_len);
        return _bitmap_buffer[idx];
    }
};

} // CudaMPMCQueue
