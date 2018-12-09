#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

namespace CudaMPMCQueue {

__host__ __device__
uint32_t isLocal(const void* const ptr) {
#ifdef __CUDA_ARCH__
    uint32_t islocal = 0;
    asm volatile ("{ \n\t"
                  "    .reg .pred p; \n\t"
                  "    isspacep.local p, %1; \n\t"
                  "    selp.u32 %0, 1, 0, p;  \n\t"
                  "} \n\t" : "=r"(islocal) : "l"(ptr));
    return islocal;
#else
    return false;
#endif
}

__host__ __device__
uint32_t heterogeneousLog2(uint32_t val) {
#ifdef __CUDA_ARCH__
        return (31 - __clz(val));
#else
        return log2l(val);
#endif
}

template<class T>
__host__ __device__
T* heterogeneousAlloc(size_t len, bool use_heterogeneous_mem) {
#ifdef __CUDA_ARCH__
		return static_cast<T*>(malloc(len * sizeof(T)));
#else
		if(use_heterogeneous_mem) {
            T* ptr;
            cudaMallocManaged(&ptr, len * sizeof(T));
            return ptr;
		} else {
		    return static_cast<T*>(malloc(len * sizeof(T)));
		}
#endif
}

template<class T>
__host__ __device__
void heterogeneousFree(T* ptr, bool use_heterogeneous_mem) {
#ifdef __CUDA_ARCH__
		free(ptr);
#else
		if(use_heterogeneous_mem) {
		    cudaFree(ptr);
		} else {
		    free(ptr);
		}
#endif
}

template<class T>
__host__ __device__
void heterogeneousMemset(T* ptr, uint8_t val, size_t len, bool use_heterogeneous_mem) {
#ifdef __CUDA_ARCH__
		memset(ptr, val, len);
#else
		if(use_heterogeneous_mem) {
		    cudaMemset(ptr, val, len);
		} else {
		    memset(ptr, val, len);
		}
#endif
}

__host__ __device__
uint32_t heterogeneousCAS(uint32_t* dst, uint32_t oldVal, uint32_t newVal) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicCAS_system(dst, oldVal, newVal);
#elif __CUDA_ARCH__
	return atomicCAS(dst, oldVal, newVal);
#else
	return __sync_val_compare_and_swap(dst, oldVal, newVal);
#endif
}

__host__ __device__
uint64_t heterogeneousCAS(uint64_t* dst, uint64_t oldVal, uint64_t newVal) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicCAS_system((unsigned long long*)dst, (unsigned long long)oldVal, (unsigned long long)newVal);
#elif __CUDA_ARCH__
	return atomicCAS((unsigned long long*)dst, (unsigned long long)oldVal, (unsigned long long)newVal);
#else
	return __sync_val_compare_and_swap(dst, oldVal, newVal);
#endif
}

__host__ __device__
uint32_t heterogeneousExch(uint32_t* dst, uint32_t val) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicExch_system(dst, val);
#elif __CUDA_ARCH__
	return atomicExch(dst, val);
#else
	return __sync_lock_test_and_set(dst, val);
#endif
}

__host__ __device__
uint64_t heterogeneousAdd(uint64_t* dst, uint64_t val) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "Expected uint64_t == ull");
	return atomicAdd_system((unsigned long long int*)dst, (unsigned long long int)val);
#elif __CUDA_ARCH__
	static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "Expected uint64_t == ull");
	return atomicAdd((unsigned long long int*)dst, (unsigned long long int)val);
#else
	return __sync_fetch_and_add(dst, val);
#endif
}

//__host__ __device__
//uint32_t heterogeneousAddMod(uint32_t* dst, uint32_t val, uint32_t modulo) {
//	assert(val <= modulo);
//	uint32_t assumed, old, new_val;
//	old = *dst;
//	do {
//		assumed = old;
//		new_val = old + val;
//		if(new_val > modulo) new_val -= modulo;
//	} while((old = heterogeneousCAS(dst, old, new_val)) != assumed);
//	return old;
//}

__host__ __device__
uint32_t heterogeneousInc(uint32_t* dst, uint32_t modulo) {
	uint32_t mod=modulo-1;
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicInc_system(dst, mod);
#elif __CUDA_ARCH__
	return atomicInc(dst, mod);
#else
	uint32_t old, val;
	do {
		old = *dst;
		val = (old >= mod) ? 0 : (old+1);
	} while(heterogeneousCAS(dst, old, val) != old);
	return old;
#endif
}


__host__ __device__
uint32_t heterogeneousDec(uint32_t* dst, uint32_t modulo) {
	uint32_t mod= modulo-1;
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicDec_system(dst, mod);
#elif __CUDA_ARCH__
	return atomicDec(dst, mod);
#else
	uint32_t old, val;
	do {
		old = *dst;
		val = ((old == 0) | (old > mod)) ? mod : (old-1);
	} while(heterogeneousCAS(dst, old, val) != old);
	return old;
#endif
}

__host__ __device__
uint32_t heterogeneousOr(uint32_t* dst, uint32_t orVal) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicOr_system(dst, orVal);
#elif __CUDA_ARCH__
	return atomicOr(dst, orVal);
#else
	return __sync_fetch_and_or(dst, orVal);
#endif
}

__host__ __device__
uint32_t heterogeneousAnd(uint32_t* dst, uint32_t mask) {
#if defined(__CUDA_ARCH__) && defined(USE_SYSTEM_ATOMICS)
	return atomicAnd_system(dst, mask);
#elif __CUDA_ARCH__
	return atomicAnd(dst, mask);
#else
	return __sync_fetch_and_and(dst, mask);
#endif
}

} // CudaMPMCQueue
