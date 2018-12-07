# CudaMPMCQueue

CudaMPMCQueue is a CUDA Compatible MPMC Queue that supports both host and device operations. It comes with Unified Memory support allowing the same object to be used on both the host and the device.

## Dependencies

In order to use CudaMPMCQueue you should have cuda 10 installed. It has only been tested for compute 6.x however, the code _should_ be compatible with older versions (back to 3.x).

In order to build the tests/benchmarks you will require cmake, gtest and google benchmark.

For CMake:

https://cmake.org/install/

For gtest:

https://github.com/google/googletest

## Building

To build tests and benchmarks run the following commands from the base directory of the repo

```
mkdir build && cd build
cmake ..
make -j $(nproc)
```

## Running tests

From the build directory run:

```
ctest
```

This will run the complete test suite. Individual executables can be found in build/bin.

NOTE: Some tests may not be able to run on all GPUs. These will fail with an error 'Too many threads'. This can be avoided by changing the test parameters in the corresponding test file.

## Running benchmarks

To run a benchmark execute the corresponding executable in build/bin/

