cmake_minimum_required(VERSION 3.16)
project(nn_operator_optimization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

option(GEMM_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(GEMM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# same flags for every kernel so the speedups come from the code, not the flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # not -march=native, so the binary still runs on other AVX2 machines (CI)
  set(GEMM_FLAGS -O3 -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  # NEON + FMA are always there on 64-bit ARM, no flags needed
  set(GEMM_FLAGS -O3)
else()
  message(FATAL_ERROR "Unsupported CPU '${CMAKE_SYSTEM_PROCESSOR}': this project targets x86-64 with AVX2 or 64-bit ARM")
endif()
string(REPLACE ";" " " GEMM_FLAGS_STR "${GEMM_FLAGS}")
message(STATUS "Building for ${CMAKE_SYSTEM_PROCESSOR} with ${GEMM_FLAGS_STR}")
set(WARNING_FLAGS -Wall -Wextra -Wpedantic)

add_library(gemm
  src/gemm_naive.cpp
  src/gemm_reordered.cpp
  src/gemm_blocked.cpp
  src/gemm_blocked_mt.cpp
  src/gemm_simd.cpp
  src/kernels.cpp
  src/threads.cpp
)
target_include_directories(gemm PUBLIC include PRIVATE src)
target_compile_options(gemm PRIVATE ${GEMM_FLAGS} ${WARNING_FLAGS})
target_link_libraries(gemm PUBLIC Threads::Threads)

# optional block sizes for stage 5, e.g. -DGEMM_KC=384
foreach(param GEMM_MC GEMM_KC GEMM_NC)
  if(DEFINED ${param})
    target_compile_definitions(gemm PRIVATE ${param}=${${param}})
  endif()
endforeach()

add_executable(bench bench/bench.cpp)
target_compile_options(bench PRIVATE -O2 ${WARNING_FLAGS})
target_link_libraries(bench PRIVATE gemm)

add_executable(test_gemm tests/test_gemm.cpp)
target_compile_options(test_gemm PRIVATE -O2 ${WARNING_FLAGS})
target_link_libraries(test_gemm PRIVATE gemm)

enable_testing()
add_test(NAME gemm_correctness COMMAND test_gemm)
