cmake_minimum_required(VERSION 3.16)
project(DeepLOBCppInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ONNX Runtime discovery strategy:
# 1) Prefer ONNXRUNTIME_ROOT env var or cmake -DONNXRUNTIME_ROOT=...
# 2) Fallback to local third_party SDK folder.
# 3) Fallback to CMAKE_PREFIX_PATH-based discovery.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime installation root")

if(NOT ONNXRUNTIME_ROOT)
  file(GLOB _ORT_LOCAL_SDKS "${CMAKE_CURRENT_LIST_DIR}/third_party/onnxruntime-win-*")
  list(LENGTH _ORT_LOCAL_SDKS _ORT_LOCAL_COUNT)
  if(_ORT_LOCAL_COUNT GREATER 0)
    list(GET _ORT_LOCAL_SDKS 0 ONNXRUNTIME_ROOT)
  endif()
endif()

if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

  set(_ORT_LIB_HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib/Release")

  if(WIN32)
    find_file(ONNXRUNTIME_RUNTIME_DLL
      NAMES onnxruntime.dll
      PATHS ${_ORT_LIB_HINTS}
      NO_DEFAULT_PATH
    )
  endif()

  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime onnxruntime.dll.a
    PATHS ${_ORT_LIB_HINTS}
    NO_DEFAULT_PATH
  )

  # MinGW cannot always link MSVC .lib import libraries directly.
  # If needed, generate a GNU import library from onnxruntime.dll.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(WIN32)
      set(ONNXRUNTIME_DLL ${ONNXRUNTIME_RUNTIME_DLL})

      if(ONNXRUNTIME_DLL AND (NOT ONNXRUNTIME_LIBRARY OR ONNXRUNTIME_LIBRARY MATCHES "\\.lib$"))
        find_program(GENDEF_EXECUTABLE NAMES gendef)
        find_program(DLLTOOL_EXECUTABLE NAMES dlltool)

        if(GENDEF_EXECUTABLE AND DLLTOOL_EXECUTABLE)
          set(_ORT_DEF_FILE "${CMAKE_BINARY_DIR}/onnxruntime.def")
          set(_ORT_IMPORT_LIB "${CMAKE_BINARY_DIR}/libonnxruntime.dll.a")

          add_custom_command(
            OUTPUT ${_ORT_DEF_FILE} ${_ORT_IMPORT_LIB}
            COMMAND ${GENDEF_EXECUTABLE} ${ONNXRUNTIME_DLL}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.def" ${_ORT_DEF_FILE}
            COMMAND ${DLLTOOL_EXECUTABLE} -d ${_ORT_DEF_FILE} -D onnxruntime.dll -l ${_ORT_IMPORT_LIB}
            DEPENDS ${ONNXRUNTIME_DLL}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating MinGW import library for ONNX Runtime"
            VERBATIM
          )

          add_custom_target(onnxruntime_importlib ALL DEPENDS ${_ORT_IMPORT_LIB})
          set(ONNXRUNTIME_LIBRARY ${_ORT_IMPORT_LIB})
        endif()
      endif()
    endif()
  endif()
else()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "Could not find ONNX Runtime SDK. Set ONNXRUNTIME_ROOT to your ONNX Runtime install folder "
    "or run cpp_inference/setup_onnxruntime.ps1 to download it."
  )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

add_executable(inference inference.cpp)
target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(inference PRIVATE ${ONNXRUNTIME_LIBRARY})
if(TARGET onnxruntime_importlib)
  add_dependencies(inference onnxruntime_importlib)
endif()

add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(benchmark PRIVATE ${ONNXRUNTIME_LIBRARY})
if(TARGET onnxruntime_importlib)
  add_dependencies(benchmark onnxruntime_importlib)
endif()

add_executable(full_inference full_inference.cpp)
target_include_directories(full_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(full_inference PRIVATE ${ONNXRUNTIME_LIBRARY})
if(TARGET onnxruntime_importlib)
  add_dependencies(full_inference onnxruntime_importlib)
endif()

if(WIN32)
  # Copy selected onnxruntime.dll beside executables so they run with matching API version.
  if(ONNXRUNTIME_RUNTIME_DLL AND EXISTS "${ONNXRUNTIME_RUNTIME_DLL}")
    add_custom_command(TARGET inference POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_RUNTIME_DLL}" $<TARGET_FILE_DIR:inference>
    )
    add_custom_command(TARGET benchmark POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_RUNTIME_DLL}" $<TARGET_FILE_DIR:benchmark>
    )
    add_custom_command(TARGET full_inference POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_RUNTIME_DLL}" $<TARGET_FILE_DIR:full_inference>
    )
  endif()
endif()
