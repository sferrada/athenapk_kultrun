# AthenaPK compilation and usage in KULTRUN
More details on **AthenaPK** and its installation can be found on [its official GitHub page](https://github.com/parthenon-hpc-lab/athenapk/tree/main).

## Installation
Most of this guide has been taken from the original GitHub readme. Only what is pertinent to KULTRUN is new.
### Dependencies

#### Required

* CMake 3.13 or greater
* C++17 compatible compiler (gcc or clang; gcc is recommended)
* Parthenon (using the submodule version provided by AthenaPK)
* Kokkos (using the submodule version provided by AthenaPK)

#### Optional

* MPI
* OpenMP (for host parallelism. Note that MPI is the recommended option for on-node parallelism.)
* HDF5 (for outputs)
* Python3 (for regressions tests with numpy, scipy, matplotlib, unyt, and h5py modules)
* Ascent (for in situ visualization and analysis)

### Building AthenaPK

Obtain all (AthenaPK, Parthenon, and Kokkos) sources

    git clone https://github.com/parthenon-hpc-lab/athenapk.git athenapk
    cd athenapk

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update

Most of the general build instructions and options for Parthenon (see [here](https://parthenon-hpc-lab.github.io/parthenon/develop/src/building.html)) also apply to AthenaPK.
The following applies for KULTRUN, remember to change **<user_name>** to your actual username.

> Before configuring the compiler, make sure you have the following modules loaded through `module load`:
> - gcc/12.2.0
> - openmpi/4.1.5
> - hdf5/1.14.1-2_openmpi-4.1.5_parallel
> - cuda/12.2

To configure the compiler on the **login node** and Intel CPUs, use the following command:
```
cmake -S. -Bbuild-host -DKokkos_ENABLE_CUDA=ON -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF -DCMAKE_CXX_COMPILER=/home/<user_name>/athenapk/external/Kokkos/bin/nvcc_wrapper -DKokkos_ARCH_AMPERE80=ON -DCMAKE_INSTALL_PREFIX=/opt/athenapk/athenapk
```

To configure it directly on the **GPU node** (*WARNING: this might give permission errors and has not been thoroughly tested*), use the following command:
```
cmake -S. -Bbuild-host -DKokkos_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=/home/<user_name>/athena_project/athenapk/external/Kokkos/bin/nvcc_wrapper -DKokkos_ARCH_ZEN2=ON -DCMAKE_INSTALL_PREFIX=/opt/athenapk/athenapk
```
here, the flag `-DKokkos_ARCH_ZEN2=ON` is probably not even needed as `cmake` should automatically detect the CPU architecture.

### Compile the code
> Again, make sure you have the following modules loaded with `module load` before compiling, plus a `Python interpreter` module:
> - gcc/12.2.0
> - openmpi/4.1.5
> - hdf5/1.14.1-2_openmpi-4.1.5_parallel
> - cuda/12.2

After the compiler has been configured, to compile the code simply execute
```
cd build-host && make
```
or
```
cmake --build build-host
```

If `cmake` has troubling finding the HDF5 library (which is required for writing analysis outputs or
restartings simulation) an additional hint to the location of the library can be provided via
`-DHDF5_ROOT=/path/to/local/hdf5` on the first `cmake` command for configuration.

## Analysing a run

## To-do

## Known errors and warnings
1. The flag `-DCMAKE_CXX_COMPILER=/home/<user_name>/athena_project/athenapk/external/Kokkos/bin/nvcc_wrapper` might not be needed at all when configuring the build. It can help, however, when the following error shows up:
```
CMake Error at external/Kokkos/cmake/kokkos_test_cxx_std.cmake:132 (MESSAGE):
  Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang or
  NVC++ or use kokkos_launch_compiler, but compiler ID was GNU
```

2. For the following error:
```
-- Check for working CXX compiler: /home/sferrada/athena_project/athenapk/external/Kokkos/bin/nvcc_wrapper - broken
CMake Error at /opt/cmake/cmake-3.21.1/share/cmake-3.21/Modules/CMakeTestCXXCompiler.cmake:62 (message):
The C++ compiler

 "/home/sferrada/athena_project/athenapk/external/Kokkos/bin/nvcc_wrapper"
```
Updating/roll-backing Kokkos might help. For this, `cd` to the Kokkos directory (`athenapk/external/Kokkos`) and run `git checkout 4.0.01`.


3. Compilation error by 2 Python libraries installed, avoided by using the `-DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF` flag at configuration time.
```
-- Architectures:
--  AMPERE80
-- Found CUDAToolkit: /usr/local/cuda-12.2/include (found version "12.2.91")
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found TPLCUDA: TRUE
-- Found TPLLIBDL: /usr/include
-- Using internal desul_atomics copy
-- Kokkos Devices: CUDA;SERIAL, Kokkos Backends: CUDA;SERIAL
-- Using Kokkos source from Kokkos_ROOT=/home/sferrada/athena_project/athenapk/external/Kokkos
-- CUDA: ON
-- PAR_LOOP_LAYOUT='MANUAL1D_LOOP' (default par_for wrapper layout)
-- PAR_LOOP_INNER_LAYOUT='TVR_INNER_LOOP' (default par_for_inner wrapper layout)
-- Found Python3: /opt/Python/Python-3.11.4/bin/python3 (found version "3.11.4") found components: Interpreter
-- Found Git: /usr/bin/git (found version "2.39.3")
-- Checking for Python modules (numpy;unyt;matplotlib;h5py;scipy) required for regression tests.
   Check can be disabled by setting PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF but then
   tests are not guaranteed to work anymore.
-- Found Python3: /usr/bin/python3.9 (found version "3.9.16") found components: Interpreter
CMake Error at external/parthenon/cmake/PythonModuleCheck.cmake:44 (message):
  Required python module(s) numpy;unyt;matplotlib;h5py;scipy not found.
Call Stack (most recent call first):
  external/parthenon/cmake/TestSetup.cmake:32 (python_modules_found)
  tst/regression/CMakeLists.txt:14 (include)
```
