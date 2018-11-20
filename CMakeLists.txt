cmake_minimum_required(VERSION 3.7.2)

project(CudaMPMCQueue LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

find_package(CUDA 10.0 REQUIRED)
set( CUDA_SEPARABLE_COMPILATION ON )
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS -std=c++11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror -03")

include_directories(include)

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
message("Using compute ${ARCH_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} )

add_subdirectory(test)