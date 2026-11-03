# Third-party libraries, pinned.  Downloads are checked against their SHA-256.
# FETCHCONTENT_BASE_DIR moves the download cache (CI caches it).

include(FetchContent)

# ----------------------------------------------------------------- ONNX Runtime
#
# A prebuilt release.  ONNXRUNTIME_ROOT points at an unpacked release instead
# (offline builds, another version, a GPU build).

set(CARBON_ORT_VERSION 1.31.0)

if(CARBON_WITH_ONNXRUNTIME)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Unpacked ONNX Runtime release (empty: download ${CARBON_ORT_VERSION})")
  if(NOT ONNXRUNTIME_ROOT)
    if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
      set(_ort_asset onnxruntime-win-x64-${CARBON_ORT_VERSION}.zip)
      set(_ort_sha256 3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195)
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_ort_asset onnxruntime-osx-arm64-${CARBON_ORT_VERSION}.tgz)
      set(_ort_sha256 1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc)
    elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(_ort_asset onnxruntime-linux-x64-${CARBON_ORT_VERSION}.tgz)
      set(_ort_sha256 cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57)
    else()
      message(FATAL_ERROR "No pinned ONNX Runtime download for ${CMAKE_SYSTEM_NAME} "
                          "${CMAKE_SYSTEM_PROCESSOR}: set ONNXRUNTIME_ROOT to an unpacked release")
    endif()
    FetchContent_Declare(onnxruntime
      URL https://github.com/microsoft/onnxruntime/releases/download/v${CARBON_ORT_VERSION}/${_ort_asset}
      URL_HASH SHA256=${_ort_sha256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime)
    set(_ort_root ${onnxruntime_SOURCE_DIR})
  else()
    set(_ort_root ${ONNXRUNTIME_ROOT})
  endif()

  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS ${_ort_root}/include ${_ort_root}/include/onnxruntime
    NO_DEFAULT_PATH REQUIRED)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR})
  if(WIN32)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION ${_ort_root}/lib/onnxruntime.dll
      IMPORTED_IMPLIB ${_ort_root}/lib/onnxruntime.lib)
  else()
    find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${_ort_root}/lib NO_DEFAULT_PATH REQUIRED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY})
  endif()
endif()

# ----------------------------------------------------------------------- PDFium
#
# PDF pages and their text layers (carbon/pdf.h): a prebuilt PDFium
# (bblanchon/pdfium-binaries; PDFium is BSD-3-Clause, and the code it carries
# comes under its own licenses, in the archive's licenses/), pinned to a
# Chromium branch.  PDFIUM_ROOT points at an unpacked build instead.

set(CARBON_PDFIUM_BRANCH 8076)  # chromium/8076, 29 September 2026

if(CARBON_WITH_PDFIUM)
  set(PDFIUM_ROOT "" CACHE PATH "Unpacked PDFium build (empty: download chromium/${CARBON_PDFIUM_BRANCH})")
  if(NOT PDFIUM_ROOT)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)
    if(WIN32 AND _proc MATCHES "^(amd64|x86_64)$")
      set(_pdfium_asset pdfium-win-x64.tgz)
      set(_pdfium_sha256 808d36da9bc5a3104315fb307c80998121f565ee53953633bf33e80d7429e5ac)
    elseif(APPLE AND _proc MATCHES "^(arm64|aarch64)$")
      set(_pdfium_asset pdfium-mac-arm64.tgz)
      set(_pdfium_sha256 0d6781fe08906baff3d82c90953e519fbc4eb253fe76431e5ed53b157763b97c)
    elseif(UNIX AND NOT APPLE AND _proc MATCHES "^(amd64|x86_64)$")
      set(_pdfium_asset pdfium-linux-x64.tgz)
      set(_pdfium_sha256 d9d67bc40af03aef4fe28a60b19b1086f28ace019c8c9caf19cb7fe3d14ceca3)
    elseif(UNIX AND NOT APPLE AND _proc MATCHES "^(arm64|aarch64)$")
      set(_pdfium_asset pdfium-linux-arm64.tgz)
      set(_pdfium_sha256 d7247b33ae5545615a5e877235dd97afc879e3a8805689684f528cae3339d352)
    else()
      message(FATAL_ERROR "No pinned PDFium download for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}: "
                          "set PDFIUM_ROOT to an unpacked build, or CARBON_WITH_PDFIUM=OFF")
    endif()
    FetchContent_Declare(pdfium
      URL https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${CARBON_PDFIUM_BRANCH}/${_pdfium_asset}
      URL_HASH SHA256=${_pdfium_sha256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(pdfium)
    set(_pdfium_root ${pdfium_SOURCE_DIR})
  else()
    set(_pdfium_root ${PDFIUM_ROOT})
  endif()
  add_library(pdfium::pdfium SHARED IMPORTED GLOBAL)
  set_target_properties(pdfium::pdfium PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${_pdfium_root}/include)
  if(WIN32)
    set_target_properties(pdfium::pdfium PROPERTIES
      IMPORTED_LOCATION ${_pdfium_root}/bin/pdfium.dll
      IMPORTED_IMPLIB ${_pdfium_root}/lib/pdfium.dll.lib)
  elseif(APPLE)
    set_target_properties(pdfium::pdfium PROPERTIES IMPORTED_LOCATION ${_pdfium_root}/lib/libpdfium.dylib)
  else()
    set_target_properties(pdfium::pdfium PROPERTIES IMPORTED_LOCATION ${_pdfium_root}/lib/libpdfium.so)
  endif()
endif()

# -------------------------------------------------------------------- stb_image
#
# One public-domain header that reads PNG, JPEG, BMP and PNM pages: the
# pipeline's image input until libpng and libjpeg-turbo come.
# PNG goes to Carbon's own decoder first (carbon/src/png.cpp), which gives
# stb_image's bytes faster and leaves it the files it does not read itself.

set(CARBON_STB_COMMIT 013ac3beddff3dbffafd5177e7972067cd2b5083)  # stb_image v2.30, 31 May 2024
FetchContent_Declare(stb_image
  URL https://raw.githubusercontent.com/nothings/stb/${CARBON_STB_COMMIT}/stb_image.h
  URL_HASH SHA256=594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(stb_image)
set(CARBON_STB_INCLUDE_DIR ${stb_image_SOURCE_DIR})

# -------------------------------------------------------------------- libdeflate
#
# libdeflate (MIT): deflate, adler-32 and crc-32 with SIMD: the searchable PDF's
# images and the PNG pages it reads.

enable_language(C)
set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_GZIP OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_GZIP_SUPPORT ON CACHE BOOL "" FORCE)  # it carries crc32, which PNG needs
FetchContent_Declare(libdeflate
  URL https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.24.tar.gz
  URL_HASH SHA256=ad8d3723d0065c4723ab738be9723f2ff1cb0f1571e8bfcf0301ff9661f475e8
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(libdeflate)

# --------------------------------------------------------------------- TensorRT
#
# Not downloaded: the release needs NVIDIA's license terms and matches a CUDA
# version.  Unpack one and set TensorRT_ROOT (or the environment variable).
# TensorRT 11 builds strongly typed networks only: precision comes from the ONNX
# file (an FP16 export, Q/DQ nodes for INT8), not from builder flags.

if(CARBON_WITH_TENSORRT)
  find_package(CUDAToolkit 12 REQUIRED)
  set(TensorRT_ROOT "$ENV{TensorRT_ROOT}" CACHE PATH "Unpacked TensorRT release")
  find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)
  find_library(TensorRT_NVINFER NAMES nvinfer_11 nvinfer_10 nvinfer
    HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64)
  find_library(TensorRT_ONNXPARSER NAMES nvonnxparser_11 nvonnxparser_10 nvonnxparser
    HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib lib64)
  if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER OR NOT TensorRT_ONNXPARSER)
    message(FATAL_ERROR "CARBON_WITH_TENSORRT needs TensorRT: set TensorRT_ROOT to an unpacked "
                        "TensorRT release (found include: ${TensorRT_INCLUDE_DIR}, "
                        "nvinfer: ${TensorRT_NVINFER}, nvonnxparser: ${TensorRT_ONNXPARSER})")
  endif()
  # UNKNOWN, not SHARED: the DLLs (with the builder resources, about 1 GB) stay in
  # TensorRT's folder and are found through PATH, not copied next to each binary.
  foreach(_lib NVINFER ONNXPARSER)
    string(TOLOWER ${_lib} _name)
    add_library(TensorRT::${_name} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(TensorRT::${_name} PROPERTIES
      IMPORTED_LOCATION ${TensorRT_${_lib}}
      INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR})
  endforeach()
  get_filename_component(TensorRT_LIBRARY_DIR ${TensorRT_NVINFER} DIRECTORY)
  set(TensorRT_RUNTIME_DIRS ${TensorRT_LIBRARY_DIR} ${TensorRT_ROOT}/bin)
endif()

# ------------------------------------------------------------------- GoogleTest

if(CARBON_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # same MSVC runtime as Carbon
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
