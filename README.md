# CudaMPMCQueue

CudaMPMCQueue is a CUDA Compatible MPMC Queue that supports both host and device operations. It comes with Unified Memory support allowing the same object to be used on both the host and the device.

## Dependencies

In order to use CudaMPMCQueue you should have cuda 10 installed. It has only been tested for compute 6.x however, the code _should_ be compatible with older versions (back to 3.x).

In order to build the tests you will require cmake, gtest and google benchmark.

For CMake:

https://cmake.org/install/

For gtest:

https://github.com/google/googletest

