cmake_minimum_required(VERSION 3.16)
project(rapt_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at an extracted onnxruntime release, e.g.
#   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.22.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an onnxruntime release directory")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime (download a release from "
    "https://github.com/microsoft/onnxruntime/releases)")
endif()

# Release tarballs sometimes ship only versioned libs (libonnxruntime.so.1.x)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIB)
  file(GLOB _ort_libs "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*"
                      "${ONNXRUNTIME_ROOT}/lib/libonnxruntime*.dylib")
  list(SORT _ort_libs)
  list(GET _ort_libs 0 ONNXRUNTIME_LIB)
endif()
if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "No onnxruntime library found under ${ONNXRUNTIME_ROOT}/lib")
endif()

add_executable(rapt_replay main.cpp)
target_include_directories(rapt_replay PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(rapt_replay PRIVATE ${ONNXRUNTIME_LIB})
set_target_properties(rapt_replay PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
