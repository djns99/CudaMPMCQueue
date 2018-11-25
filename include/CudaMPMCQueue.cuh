#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "HeterogeneousFunctions.cuh"
#include "CudaMPMCQueueUtil.cuh"

namespace CudaMPMCQueue {

// Enable if the queue will be in prolonged usage
// Avoids uint64_t overflows by reducing the counters when clearing head/tail
// #define AVOID_OVERFLOWS 1

/*
 * CUDA Compatible MPMCQueue
 * Supports parallel threads pushing and popping simultaneously
 * Any push/pop call that occurs after a previous push/pop call has completed is guaranteed to be placed after that call in the queue
 * Any push/pop call that happens before a parallel push/pop call has completed has no ordering guarantee
 */
template<typename T>
class MPMCQueue
{
public:

    /*
     * Constructor
     *
     * capacity             The number of elements the queue can hold. Set to a power of 2 for best performance.
     * use_device_mem       Set to false to use host-only memory. Only use this if you know the object will never be used on the device
     */
    __host__ __device__ MPMCQueue(uint64_t capacity, bool use_heterogeneous_mem = true);
    __host__ __device__ MPMCQueue(MPMCQueue&& other) = delete;
    __host__ __device__ MPMCQueue(MPMCQueue& other) = delete;
    __host__ __device__ MPMCQueue() = delete;

    __host__ __device__ MPMCQueue& operator=(const MPMCQueue& other) = delete;


    __host__ __device__ ~MPMCQueue();

    /*
     * Try to push a value onto the queue
     *
     * val      Value to push onto the queue
     *
     * Returns true if we successfully found a free slot
     */
    __host__ __device__ bool try_push(T val);
    /*
     * Blocks until method can successfully push value onto the queue.
     *
     * val      Value to push onto the queue
     *
     * WARNING: Only use this if you know all calling threads in the warp will succeed.
     *          Otherwise this may result in deadlock.
     */
    __host__ __device__ void push(T val);
    /*
     * Try to pop a value from the queue
     *
     * Returns optional<> that has value set if a value was popped
     */
    __host__ __device__ optional<T> try_pop();
    /*
     * Blocks until method can successfully pop a value from the queue
     *
     * Returns value popped from the queue
     *
     * WARNING: Only use this if you know all calling threads in the warp will succeed.
     *          Otherwise this may result in deadlock.
     */
    __host__ __device__ T pop();

    /*
     * Get the capacity of the queue
     */
    __host__ __device__ uint64_t capacity() const;

    /*
     * Get the best guess number of elements in the queue
     * Returns the number of elements in the queue when method looked
     */
    __host__ __device__ uint64_t size_approx() const;
    /*
     * Attempts to resolve all uncertainties regarding sizes
     * If called from a single thread (i.e. with no other threads mutating the object),
     * it is guaranteed that subsequent calls to size_approx() will be correct until the next push or pop
     */
    __host__ __device__ bool try_sync();

    /*
     * Heterogeneously allocates the object using cudaMallocManaged from the host
     *
     * capacity                 Capacity of allocated queue
     * use_device_memory        Whether the queue will be allocated using heterogeneous memory
     *
     * Returns the newly allocated queue
     */
    __host__ __device__ static MPMCQueue<T>* allocHeterogeneousMPMCQueue(uint64_t capacity, bool use_heterogeneous_memory=true);
    /*
     * Frees a queue allocated with allocHeterogeneousMPMCQueue
     *
     * queue        The pointer to the queue to free
     */
    __host__ __device__ static void freeHeterogeneousMPMCQueue(MPMCQueue<T>* queue);
private:
    __host__ __device__ uint64_t mod_capacity(uint64_t val) const;
    __host__ __device__ uint64_t div_capacity(uint64_t val) const;

    __host__ __device__ bool try_push_internal(T val);
    __host__ __device__ optional<T> try_pop_internal();

    template<bool flip>
    __host__ __device__ optional<uint64_t> try_clear_range(uint64_t start, uint64_t end) const;

    __host__ __device__ bool try_clear_head();
    __host__ __device__ bool try_clear_tail();

    __device__ uint32_t get_active_mask() const;

    __device__ bool try_push_internal_warp(T val, uint32_t active_mask);

    __device__ optional<T> try_pop_internal_warp(uint32_t active_mask);

    template<bool flip>
    __device__ optional<uint64_t> try_clear_range_warp(uint64_t start, uint64_t end, uint32_t active_mask) const;

    __device__ bool try_clear_head_warp();
    __device__ bool try_clear_tail_warp();

    const uint64_t _capacity;
    const uint64_t _capacity_log;
    const bool _can_fast_mod;
    const uint64_t _fast_mod_mask;

    T* const _data;
    bool _using_heterogeneous_mem;
    uint64_t _head_safe;
    cuAtomic<uint64_t> _head_unsafe;
    uint64_t _tail_safe;
    cuAtomic<uint64_t> _tail_unsafe;
    cuAtomic<uint64_t> _size;
    cuAtomic<uint64_t> _free;
    const uint64_t _bitmap_len;
    cuBitmap _write_bitmap;

    cuMutex _updating_head{};
    cuMutex _updating_tail{};

    const bool _is_local;

    // Tuning parameter
    // The number of active threads in a warp before using warp based methods
    const uint32_t _use_warp = 4;
};


template<typename T>
__host__ __device__ MPMCQueue<T>::MPMCQueue(uint64_t capacity, bool use_heterogeneous_mem) :
                _capacity(capacity),
                _capacity_log(heterogeneousLog2(capacity)),
                _can_fast_mod(!(capacity & (capacity - 1))),
                _fast_mod_mask((1 << _capacity_log) - 1),
                _data(heterogeneousAlloc<T>(_capacity, use_heterogeneous_mem)),
                _using_heterogeneous_mem(use_heterogeneous_mem),
                _head_safe(0),
                _head_unsafe(0),
                _tail_safe(0),
                _tail_unsafe(0),
                _size(0),
                _free(_capacity),
                _bitmap_len(ceil_div<uint64_t>(_capacity, 32ull)),
                _write_bitmap(_bitmap_len, use_heterogeneous_mem),
                _is_local(isLocal(this))
{
    if (_data == nullptr)
    {
        // Not much we can do here. CUDA doesn't support exceptions
        printf("Failed to allocate %lu entries\n", _capacity);
        assert(false);
        return;
    }
}

template<typename T>
__host__ __device__ MPMCQueue<T>::~MPMCQueue()
{
    heterogeneousFree(_data, _using_heterogeneous_mem);
}

template<typename T>
__host__ __device__
bool MPMCQueue<T>::try_push(T val)
{
#ifdef __CUDA_ARCH__
    if(!_is_local) {
        uint32_t mask = get_active_mask();
        uint32_t active_count = __popc(mask);
        if (active_count >= _use_warp)
        {
            return try_push_internal_warp(val, mask);
        }
    }
#endif
    return try_push_internal(val);
}

template<typename T>
__host__ __device__
void MPMCQueue<T>::push(T val)
{
    while (!try_push(val))
        ;
}

template<typename T>
__host__ __device__
optional<T> MPMCQueue<T>::try_pop()
{
#ifdef __CUDA_ARCH__
    if(!_is_local) {
        uint32_t mask = get_active_mask();
        uint32_t active_count = __popc(mask);
        if (active_count >= _use_warp)
        {
            return try_pop_internal_warp(mask);
        }
    }
#endif
    return try_pop_internal();
}

template<typename T>
__host__ __device__
T MPMCQueue<T>::pop()
{
    optional<T> res;
    while (!(res = try_pop()))
        ;

    return res.value();
}

template<typename T>
__host__  __device__
uint64_t MPMCQueue<T>::capacity() const
{
    return _capacity;
}

template<typename T>
__host__  __device__
uint64_t MPMCQueue<T>::size_approx() const
{
#ifdef AVOID_OVERFLOWS
    auto head = mod_capacity(_head_unsafe.value());
    auto tail = mod_capacity(_tail_unsafe.value());

    if(head > tail)
        return head - tail;
    else if(head < tail)
        return head + _capacity - tail;

    // head == tail. Could be 0 or _capacity
    if(_free.value() > _size.value()) // Educated guess
        return 0;
    else
        return _capacity;
#else
    // Difference <= capacity
    assert(_head_unsafe.value() - _tail_unsafe.value() <= _capacity);
    return _head_unsafe.value() - _tail_unsafe.value();
#endif
}

template<typename T>
__host__ __device__ bool MPMCQueue<T>::try_sync()
{
    bool head = try_clear_head();
    bool tail = try_clear_tail();
    return head || tail;
}

template<typename T>
__host__ __device__
MPMCQueue<T>* MPMCQueue<T>::allocHeterogeneousMPMCQueue(uint64_t capacity, bool use_heterogeneous_mem)
{
    MPMCQueue<T>* queue = heterogeneousAlloc<MPMCQueue<T>>(1, use_heterogeneous_mem);
    if(queue) {
        new(queue) MPMCQueue<T>(capacity, use_heterogeneous_mem);
    }
    return queue;
}

template<typename T>
__host__ __device__
void MPMCQueue<T>::freeHeterogeneousMPMCQueue(MPMCQueue<T>* queue)
{
    queue->~MPMCQueue<T>();
    heterogeneousFree(queue, queue->_using_heterogeneous_mem);
}

__host__ __device__
uint32_t ffs(uint32_t val)
{
#ifdef __CUDA_ARCH__
    return __ffs(val);
#else
    return __builtin_ffs(val);
#endif
}

template<typename T>
__host__ __device__
uint64_t MPMCQueue<T>::mod_capacity(const uint64_t val) const {
    return (_can_fast_mod) ? (val & _fast_mod_mask) : (val % _capacity);
}

template<typename T>
__host__ __device__
uint64_t MPMCQueue<T>::div_capacity(uint64_t val) const {
    return (_can_fast_mod) ? (val >> _capacity_log) : (val / _capacity);
}

template<typename T>
__host__ __device__ bool MPMCQueue<T>::try_push_internal(T val)
{
    uint64_t free_slots = _free.atomic_sub(1ul); // Consume one free slot
    if (free_slots == 0 || free_slots > INT64_MAX) // If free slots are 0 or 'negative'
    {
        _free.atomic_add(1ul); // Release free slot
        if (try_clear_tail()) // See if we can advance tail
            return try_push(val); // Re attempt push if we made space
        return false;
    }
    // We are guaranteed to have a slot if we reach here
    // Get index of next available slot
    uint64_t idx = mod_capacity(_head_unsafe.atomic_add(1ul));
    _data[idx] = val; // Write data
    _write_bitmap.setBit(idx); // Set bit
    try_clear_head(); // Try to advance head pointer
    return true;
}

template<typename T>
__host__ __device__
optional<T> MPMCQueue<T>::try_pop_internal()
{
    uint64_t size = _size.atomic_sub(1ul); // Try consume an element
    if (size == 0 || size > INT64_MAX) // If size is 0 or 'negative'
    {
        _size.atomic_add(1ul); // Release spot
        if (try_clear_head()) // Try advance head
            return try_pop(); // Reattempt if we made space
        return {};
    }
    // We are guaranteed to have an element to pop if we reach here
    // Get index of next available element
    uint64_t idx = mod_capacity(_tail_unsafe.atomic_add(1ul));
    T out = _data[idx]; // Read data
    _write_bitmap.clearBit(idx); // Unset bit
    try_clear_tail(); // Try to advance tail pointer
    return {out};
}

template<typename T>
template<bool flip>
__host__ __device__
optional<uint64_t> MPMCQueue<T>::try_clear_range(const uint64_t start, const uint64_t end) const
{
    if(start == end)
        return {};

    // Assumes exclusive access to bitmap by warp

    const uint64_t start_mod = mod_capacity(start);
    const uint64_t end_mod = mod_capacity(end);

    const uint64_t end_prev = end_mod == 0 ? _capacity - 1 : end_mod - 1;
    const uint64_t end_idx = end_prev >> 5;
    uint64_t idx = start_mod >> 5;

    // Mask for ranges starting partway through entry
    const uint32_t start_off = start_mod & 31;
    const bitmap_mask start_mask = (~0) << start_off;

    // Load initial bitmap value
    bitmap_entry bitmap_val = _write_bitmap[idx];
    if(flip) // Expect 1s - flip bits so 1 -> 0
        bitmap_val = ~bitmap_val;

    // Mask after flip so correct bits are dropped
    bitmap_val &= start_mask;

    // Mask final (partial) bitmap entry for non-multiple 32 capacity
    const uint64_t final_bit_off = (_capacity - 1) & 31;
    const bitmap_mask last_entry_mask = (~0u) >> (31u - final_bit_off);
    if(idx == _bitmap_len - 1)
        bitmap_val &= last_entry_mask;

    // Loop until incorrect bit or end of range
    while(!bitmap_val && idx != end_idx) {
        // Inc and wrap
        idx = (idx == _bitmap_len - 1) ? 0 : idx + 1;
        // Load new bitmap value
        bitmap_val = _write_bitmap[idx];
        if(flip)
            bitmap_val = ~bitmap_val;
        if(idx == _bitmap_len - 1)
            bitmap_val &= last_entry_mask;
    }

    if(idx == end_idx) {
        const uint32_t end_off = end_mod & 31;
        const bitmap_mask end_mask = (~0u) >> (31u - end_off);
        bitmap_val &= end_mask;
        if(bitmap_val == 0) // No bits were set. Clear full range
            return {end};
    }

    // Location of first incorrectly set bit
    uint64_t new_start = idx * 32 + (ffs(bitmap_val) - 1);
    if(new_start < start_mod)
        new_start += _capacity;

    // Add number of spots advanced
    return {start + (new_start - start_mod)};
}

template<typename T>
__host__ __device__
bool MPMCQueue<T>::try_clear_head()
{
    // Early exit if no pending requests
    if(_head_safe == _head_unsafe.value())
        return false;

#ifdef __CUDA_ARCH__
    if(!_is_local) {
        return try_clear_head_warp();
    }
#endif

    if (!_updating_head.try_lock())
        return false;
    uint64_t old = _head_safe;
    optional<uint64_t> new_head_opt = try_clear_range<true>(_head_safe, _head_unsafe.value());
    if(new_head_opt) {
        uint64_t new_head_value = new_head_opt.value();
        _head_safe = new_head_value;
        _size.atomic_add(new_head_value - old);

#ifdef AVOID_OVERFLOWS
        // Reduce counters to avoid overflows
        if(new_head_value > _capacity) {
            uint64_t div = div_capacity(new_head_value);
            // Values should only decrease here.
            _head_unsafe.atomic_sub(_capacity * div);
            _head_safe -= _capacity * div;
        }
#endif
    }
    _updating_head.unlock();
    return new_head_opt.has_value();
}

template<typename T>
__host__ __device__
bool MPMCQueue<T>::try_clear_tail()
{
    // Early exit if no pending requests
    if(_tail_safe == _tail_unsafe.value())
        return false;

#ifdef __CUDA_ARCH__
    if(!_is_local) {
        return try_clear_tail_warp();
    }
#endif

    if (!_updating_tail.try_lock())
        return false;
    uint64_t old = _tail_safe;
    optional<uint64_t> new_tail_opt = try_clear_range<false>(_tail_safe, _tail_unsafe.value());
    if(new_tail_opt) {
        uint32_t new_tail_value = new_tail_opt.value();
        _tail_safe = new_tail_value;
        _free.atomic_add(new_tail_value - old);

#ifdef AVOID_OVERFLOWS
        // Reduce counters to avoid overflows
        if(new_tail_value > _capacity) {
            uint64_t div = div_capacity(new_tail_value);
            // Values should only decrease here.
            _tail_unsafe.atomic_sub(_capacity * div);
            _tail_safe -= _capacity * div;
        }
#endif
    }
    _updating_tail.unlock();
    return new_tail_opt.has_value();
}

__device__ uint32_t match_any_sync(uint32_t mask, uint64_t val) {
#if __CUDA_ARCH__ >= 700
    return __match_any_sync(mask, val);
#else
    uint32_t out_mask = 0x0;
    for(uint32_t i = 0; i < 32; i++) { // TODO 32 - __clz(mask)?
        if(mask & (1 << i)) {
            out_mask |= (__shfl_sync(mask, val, i) == val) << i;
        }
    }
    return out_mask;
#endif
}

template<typename T>
__device__
uint32_t MPMCQueue<T>::get_active_mask() const {
    if(_is_local) {
        return 1 << (threadIdx.x & 31);
    }

    uint32_t active_mask = __activemask();
    return match_any_sync(active_mask, (uint64_t)this);
}

template<typename T>
__device__
bool MPMCQueue<T>::try_push_internal_warp(T val, uint32_t active_mask)
{
    assert(!_is_local);

    // Get thread/warp info
    uint32_t active_threads = __popc(active_mask);
    uint32_t lane = threadIdx.x & 31;
    uint32_t tid0 = ffs(active_mask) - 1;

    // Try to get a slot for all active threads
    uint64_t free_slots;
    if (tid0 == lane)
    {
        free_slots = _free.atomic_sub(active_threads);
    }

    free_slots = __shfl_sync(active_mask, free_slots, tid0); // Share number of slots to all threads

    // No slots obtained
    if (free_slots == 0 || free_slots > INT64_MAX)
    {
        if (tid0 == lane)
        {
            _free.atomic_add(active_threads); // Release slots
        }
        __syncwarp(active_mask); // Sync to ensure cooperation in clear_tail
        if (try_clear_tail()) // Try to advance tail pointer
        {
            // Re attempt push if we made space
            return try_push(val);
        }
        return false;
    }

    uint32_t tid = __popc(active_mask & ((1 << lane) - 1));
    // Use active_threads worth of slots
    uint32_t successful = (free_slots > active_threads) ? active_threads : free_slots;

    uint64_t idx;
    if (tid == 0)
    {
        // Cleanup unsuccessful allocations
        _free.atomic_add(active_threads - successful);
        // Obtain start index for writes
        idx = mod_capacity(_head_unsafe.atomic_add(successful));
    }
    idx = __shfl_sync(active_mask, idx, tid0); // Share index

    uint64_t wrap_tid = _capacity - idx; // How far are we from the end

    // Dont terminate failed threads
    // Keep around them to help with clearing head
    if (tid < successful)
    {
        // Wrap if necessary
        uint32_t myidx = (tid >= wrap_tid) ? tid - wrap_tid : idx + tid;
        _data[myidx] = val;
    }
    __syncwarp(active_mask);

    // try write 'successful' bits
    // if we overflow bitmap entry move on (includes partial last entry of bitmap)
    // try write 'successful - used' bits
    // if we overflow bitmap entry move on (only if partial last entry of bitmap)
    // try write 'successful - used' bits
    // this must always succeed
    // NOTE: Two writes may be to same memory address for small queues
    uint32_t overflows = 0;
    uint32_t off = idx & 31;
    overflows += (off + successful > 32) || (wrap_tid < successful); // 2 entries or off the end
    uint64_t b_idx = idx >> 5;
    overflows += (b_idx == _bitmap_len - 2) && (wrap_tid < successful); // Second write wraps

    uint32_t entry = tid;
    while (entry <= overflows)
    {
        // Can never overflow by more than one
        uint64_t my_b_idx = (b_idx + entry >= _bitmap_len) ? 0 : b_idx + entry;
        uint32_t b_off = off * !entry; // Second and third overflow have offset of 0
        uint32_t num;
        if (tid == 0)
            num = successful; // Will get truncated by shift
        else if (tid == 1)
            num = successful - (32 - off); // Off the end bits can be ignored (for double overflow)
        else if (tid == 2)
            num = successful - wrap_tid; // Will always be < 32

        bitmap_mask mask = (1 << num) - 1;
        _write_bitmap.setMask(my_b_idx, mask << b_off);

        entry += active_threads; // Loops for < 3 threads
    }
    __syncwarp(active_mask); // Sync to ensure cooperation in clear_head
    try_clear_head();
    return tid < successful;
}

template<typename T>
__device__
optional<T> MPMCQueue<T>::try_pop_internal_warp(uint32_t active_mask)
{
    assert(!_is_local);

    // Get thread/warp info
    uint32_t active_threads = __popc(active_mask);
    uint32_t lane = threadIdx.x & 31;
    uint32_t tid0 = ffs(active_mask) - 1;

    // Try to get a slot for all active threads
    uint64_t size;
    if (tid0 == lane)
    {
        size = _size.atomic_sub(active_threads);
    }

    size = __shfl_sync(active_mask, size, tid0); // Share number of slots to all threads

    // No slots obtained
    if (size == 0 || size > INT64_MAX)
    {
        if (tid0 == lane)
        {
            _size.atomic_add(active_threads); // Release slots
        }
        __syncwarp(active_mask); // Sync to ensure cooperation in clear_head
        if (try_clear_head()) // Try to advance head
        {
            // Re attempt pop if we made space
            return try_pop();
        }
        return {};
    }

    uint32_t tid = __popc(active_mask & ((1 << lane) - 1));
    // Use active_threads worth of slots
    uint64_t successful = (size > active_threads) ? active_threads : size;

    uint64_t idx;
    if (tid == 0)
    {
        // Cleanup unsuccessful allocations
        _size.atomic_add(active_threads - successful);
        // Obtain start index for writes
        idx = mod_capacity(_tail_unsafe.atomic_add(successful));
    }
    idx = __shfl_sync(active_mask, idx, tid0); // Share index

    uint64_t wrap_tid = _capacity - idx; // How far are we from the end

    // Dont terminate failed threads
    // Keep around them to help with clearing tail
    T out;
    if (tid < successful)
    {
        // Wrap if necessary
        uint32_t myidx = (tid >= wrap_tid) ? tid - wrap_tid : idx + tid;
        assert(myidx < _capacity);
        out = _data[myidx];
    }
    __syncwarp(active_mask);

    // try write 'successful' bits
    // if we overflow bitmap entry move on (includes partial last entry of bitmap)
    // try write 'successful - used' bits
    // if we overflow bitmap entry move on (only if partial last entry of bitmap)
    // try write 'successful - used' bits
    // this must always succeed
    // NOTE: Two writes may be to same memory address for small queues
    uint32_t overflows = 0;
    uint32_t off = idx & 31;
    overflows += (off + successful > 32) || (wrap_tid < successful); // 2 entries or off the end
    uint64_t b_idx = idx >> 5;
    overflows += (b_idx == _bitmap_len - 2) && (wrap_tid < successful); // Second write wraps

    uint32_t entry = tid;
    while (entry <= overflows)
    {
        // Can never overflow by more than one
        uint64_t my_b_idx = (b_idx + entry >= _bitmap_len) ? 0 : b_idx + entry;
        uint32_t b_off = off * !entry; // Second and third overflow have offset of 0
        uint32_t num;
        if (tid == 0)
            num = successful; // Will get truncated by shift
        else if (tid == 1)
            num = successful - (32 - off); // Off the end bits can be ignored (for double overflow)
        else if (tid == 2)
            num = successful - wrap_tid; // Will always be < 32

        bitmap_mask mask = (1 << num) - 1;
        _write_bitmap.clearMask(my_b_idx, ~(mask << b_off)); // Set bit

        entry += active_threads; // Loops for < 3 threads
    }
    __syncwarp(active_mask); // Sync to ensure cooperation in clear_tail
    try_clear_tail();
    return tid < successful ? optional<T>{out} : optional<T>{};
}

template<typename T>
template<bool flip>
__device__ optional<uint64_t> MPMCQueue<T>::try_clear_range_warp(const uint64_t start, const uint64_t end,
        uint32_t active_mask) const
{
    assert(!_is_local);

    // Assumes exclusive access to bitmap by warp
    if(start == end)
        return {};

    // Warp/Thread info
    uint32_t active_threads = __popc(active_mask);
    uint32_t lane = threadIdx.x & 31;
    uint32_t tid = __popc(active_mask & ((1 << lane) - 1));

    const uint64_t start_mod = mod_capacity(start);
    const uint64_t end_mod = mod_capacity(end);

    // Convert start and end to bitmap idxs
    const uint64_t start_idx = start_mod >> 5;
    const uint64_t end_prev = end_mod == 0 ? (_capacity - 1) : (end_mod - 1);
    const uint64_t end_idx = end_prev >> 5;

    bool wrap = start_mod > end_prev;
    // Calculate the number of entries to try and clear
    uint64_t num_entries = (!wrap ? end_idx - start_idx : (end_idx + _bitmap_len) - start_idx) + 1;

    // Terminate unneeded threads
    bool exit = tid >= num_entries;
    active_mask = __ballot_sync(active_mask, !exit);
    if (exit)
    {
        return {};
    }

    // Calculate which bitmap entry to look at
    uint64_t idx = start_idx + tid;
    idx -= _bitmap_len * (idx >= _bitmap_len);
    uint32_t start_off = start_mod & 31;
    bitmap_mask start_mask = (~0) << start_off;
    bitmap_mask last_entry_mask = (1 << (_capacity & 31)) - 1;

    bitmap_entry bitmap_val = _write_bitmap[idx];
    if (flip) // We are looking for 0s - flip bits so nonzero entries are 1
        bitmap_val = ~bitmap_val;
    // Mask start byte if needed
    bitmap_val &= (tid == 0) * start_mask + (tid != 0) * 0xFFFFFFFF;
    // Handle partial final index
    bitmap_val &= (idx == _bitmap_len - 1) * last_entry_mask + (idx != _bitmap_len - 1 && idx == _bitmap_len - 1) * 0xFFFFFFFF;

    uint32_t entry = tid;
    uint32_t ballot = 0;
    bool overshoot = false;
    // Loop until non-zero bitmap entry is found
    while (!(ballot = __ballot_sync(active_mask, bitmap_val)))
    {
        entry += active_threads;
        if (entry >= num_entries)
        {
            // Reached end
            // Set to non-zero to pass ballot if all bits are cleared
            bitmap_val = 0xFFFFFFFF;
            overshoot = true;
        }
        else
        {
            // Wrap if necessary
            idx += active_threads;
            idx -= _bitmap_len * (idx >= _bitmap_len);

            bitmap_val = _write_bitmap[idx];
            if (flip)
                bitmap_val = ~bitmap_val;

            // Handle partial final index
            if(idx == _bitmap_len - 1 && last_entry_mask)
                bitmap_val &= last_entry_mask;
        }
    }
    // Exit if we didn't find first non-zero entry
    if (ffs(ballot) - 1 != lane)
        return {};

    assert(__popc(get_active_mask()) == 1);

    if(overshoot)
        return {end};

    if (idx == end_idx)
    {
        // Mask end index
        uint32_t end_off = end_prev & 31;
        bitmap_val &= (~0u) >> (31u - end_off);
        if(bitmap_val == 0)
            return {end};
    }

    // Index of first incorrectly set bit
    uint32_t new_start = idx * 32 + (ffs(bitmap_val) - 1);
    if(new_start < start_mod)
        new_start += _capacity;

    // Add number of places advanced
    return {start + (new_start - start_mod)};
}

template<typename T>
__device__
bool MPMCQueue<T>::try_clear_head_warp()
{
    assert(!_is_local);

    uint32_t active_mask = get_active_mask();
    if (!__any_sync(active_mask, _updating_head.try_lock())) // Did this warp obtain the bitmap
        return false;
    // Warp has exclusive access to bitmap between head_safe and head_unsafe
    uint64_t old = _head_safe;
    optional<uint64_t> new_head_opt = try_clear_range_warp<true>(_head_safe, _head_unsafe.value(), active_mask);
    if (new_head_opt)
    {
        uint64_t new_head_value = new_head_opt.value();
        // This thread found the new pos
        _head_safe = new_head_value;
        // Make new entries available to pop
        _size.atomic_add(new_head_value - old);
#ifdef AVOID_OVERFLOWS
        // Reduce counters to avoid overflows
        if(new_head_value > _capacity) {
            uint64_t div = div_capacity(new_head_value);
            // Values should only decrease here.
            _head_unsafe.atomic_sub(_capacity * div);
            _head_safe -= _capacity * div;
        }
#endif
    }

    if ((ffs(active_mask) - 1) == (threadIdx.x & 31))
        _updating_head.unlock();

    // All threads return if we successfully advanced head
    return __any_sync(active_mask, new_head_opt.has_value());
}

template<typename T>
__device__
bool MPMCQueue<T>::try_clear_tail_warp()
{
    assert(!_is_local);

    uint32_t active_mask = get_active_mask();
    if (!__any_sync(active_mask, _updating_tail.try_lock())) // Did this warp obtain the bitmap
        return false;
    // Warp has exclusive access to bitmap between tail_safe and tail_unsafe
    uint64_t old = _tail_safe;
    optional<uint64_t> new_tail_opt = try_clear_range_warp<false>(_tail_safe, _tail_unsafe.value(), active_mask);
    if (new_tail_opt)
    {
        uint32_t new_tail_value = new_tail_opt.value();
        // This thread found the new pos
        _tail_safe = new_tail_value;
        // Make new slots available to push to
        _free.atomic_add(new_tail_value - old);

#ifdef AVOID_OVERFLOWS
        // Reduce counters to avoid potential overflows
        if(new_tail_value > _capacity) {
            uint64_t div = div_capacity(new_tail_value);
            // Values should only decrease here.
            _tail_unsafe.atomic_sub(_capacity * div);
            _tail_safe -= _capacity * div;
        }
#endif
    }
    if ((ffs(active_mask) - 1) == (threadIdx.x & 31))
        _updating_tail.unlock();
    // All threads return if we successfully advanced tail
    return __any_sync(active_mask, new_tail_opt.has_value());
}

} // CudaMPMCQueue

