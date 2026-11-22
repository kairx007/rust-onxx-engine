cmake_minimum_required(VERSION 3.16)
project(RustLOBCppBench LANGUAGES CXX)

# Same toolchain settings as cpp-engine/cpp_inference/CMakeLists.txt.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime C++ SDK root (include/, lib/)")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to the ONNX Runtime 1.31.0 C++ SDK (see bench/README.md)")
endif()

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(rustlob_cpp_bench rustlob_cpp_bench.cpp)
target_include_directories(rustlob_cpp_bench PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(rustlob_cpp_bench PRIVATE ${ONNXRUNTIME_LIBRARY})
target_compile_options(rustlob_cpp_bench PRIVATE -Wall -Wextra)
set_target_properties(rustlob_cpp_bench PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
