cmake_minimum_required(VERSION 3.7.2)

project(CudaMPMCQueue LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

find_package(CUDA 10.0 REQUIRED)
set( CUDA_SEPARABLE_COMPILATION ON )
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

option(USE_SYSTEM_ATOMICS "If atomic functions in tests should use atomic*_system" OFF)
option(AVOID_OVERFLOWS "If tests should have overflow avoidance on" ON)

if(USE_SYSTEM_ATOMICS)
	add_definitions(-DUSE_SYSTEM_ATOMICS)
endif()

if(AVOID_OVERFLOWS)
	add_definitions(-DAVOID_OVERFLOWS)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS -std=c++11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror -O3")

include_directories(include)

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
message("Using compute ${ARCH_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

add_subdirectory(test)
add_subdirectory(bench)