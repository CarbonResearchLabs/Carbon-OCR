cmake_minimum_required(VERSION 3.24)
project(carbon
  VERSION 0.1.0
  DESCRIPTION "Fast line-based OCR on CPU (ONNX Runtime) and GPU (TensorRT)"
  LANGUAGES CXX)

option(CARBON_WITH_ONNXRUNTIME "Build the ONNX Runtime (CPU) backend" ON)
option(CARBON_WITH_TENSORRT "Build the TensorRT (CUDA) backend; needs TensorRT_ROOT" OFF)
option(CARBON_WITH_CUDA "Build the CUDA kernels: the crop rule and CTC decoding on the GPU" OFF)
option(CARBON_BUILD_CLI "Build the carbon-ocr command line" ON)
option(CARBON_WITH_PDFIUM "Read PDFs: their pages and text layers, through a prebuilt PDFium" ON)
option(CARBON_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(CARBON_BUILD_PYTHON "Build the Python module carbon_ocr (pip install ./python does this)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CARBON_WITH_CUDA)
  # The GPU this machine has unless told otherwise; set CMAKE_CUDA_ARCHITECTURES
  # (89 for Ada, 86 for Ampere, ...) to build for others.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

if(CARBON_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static library goes into a shared module
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CarbonOptions)
include(CarbonDependencies)

add_subdirectory(carbon)
if(CARBON_BUILD_CLI)
  add_subdirectory(cli)
endif()
if(CARBON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(CARBON_BUILD_PYTHON)
  add_subdirectory(python)
endif()

message(STATUS "Carbon ${PROJECT_VERSION}: ONNX Runtime ${CARBON_WITH_ONNXRUNTIME}, "
               "TensorRT ${CARBON_WITH_TENSORRT}, CUDA ${CARBON_WITH_CUDA}, "
               "CLI ${CARBON_BUILD_CLI}, tests ${CARBON_BUILD_TESTS}")
