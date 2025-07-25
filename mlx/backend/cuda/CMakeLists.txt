# Filename rules in cuda backend:
#
# * Use .cu/.cuh if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/maxpooling.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/topk_to_uint64.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_CUDA)

# Embed kernel sources in binary for JIT compilation.
file(
  GLOB MLX_JIT_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.cuh")
string(JOIN ":" MLX_JIT_SOURCES_ARG ${MLX_JIT_SOURCES})
add_custom_command(
  OUTPUT gen/cuda_jit_sources.h
  COMMAND
    ${CMAKE_COMMAND} -DMLX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
    -DMLX_JIT_SOURCES=${MLX_JIT_SOURCES_ARG} -P
    "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
  DEPENDS bin2h.cmake ${MLX_JIT_SOURCES})
add_custom_target(cuda_jit_sources DEPENDS gen/cuda_jit_sources.h)
add_dependencies(mlx cuda_jit_sources)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Enable defining device lambda functions.
target_compile_options(mlx
                       PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

# Enable calling host constexpr functions from device. This is needed because
# the constexpr version of isnan is host only.
target_compile_options(
  mlx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# CUDA 12.8 emits warning #20280-D for copy kernels which is a false positive.
# Explicitly pass this flag to suppress the warning, it is safe to set it to
# true but the warning wouldn't be suppressed.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
  target_compile_options(
    mlx
    PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>")
endif()

# Suppress warning when building for compute capability 7 used by V100.
target_compile_options(
  mlx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>")

# Compute capability 7 is required for synchronization between CPU/GPU with
# managed memory. TODO: Add more architectures for potential performance gain.
set(MLX_CUDA_ARCHITECTURES
    "70;80"
    CACHE STRING "CUDA architectures")
message(STATUS "CUDA architectures: ${MLX_CUDA_ARCHITECTURES}")
set_target_properties(mlx PROPERTIES CUDA_ARCHITECTURES
                                     "${MLX_CUDA_ARCHITECTURES}")

# Use fixed version of CCCL.
FetchContent_Declare(
  cccl
  URL "https://github.com/NVIDIA/cccl/releases/download/v2.8.1/cccl-v2.8.1.zip")
FetchContent_MakeAvailable(cccl)
target_include_directories(mlx BEFORE PRIVATE "${cccl_SOURCE_DIR}/include")

# Use fixed version of NVTX.
FetchContent_Declare(
  nvtx3
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
  GIT_TAG v3.1.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR c EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nvtx3)
target_link_libraries(mlx PUBLIC $<BUILD_INTERFACE:nvtx3-cpp>)

# Make cuda runtime APIs available in non-cuda files.
find_package(CUDAToolkit REQUIRED)
target_include_directories(mlx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Use cublasLt.
target_link_libraries(mlx PRIVATE CUDA::cublasLt)

# Use NVRTC and driver APIs.
target_link_libraries(mlx PRIVATE CUDA::nvrtc CUDA::cuda_driver)

# Suppress nvcc warnings on MLX headers.
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                                   --diag_suppress=997>)
