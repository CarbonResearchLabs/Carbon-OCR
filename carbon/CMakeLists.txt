configure_file(include/carbon/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/carbon/version.h @ONLY)

add_library(carbon
  src/tensor.cpp
  src/session.cpp
  src/lines.cpp
  src/detect.cpp
  src/det_head.cpp
  src/engine.cpp
  src/pipeline.cpp
  src/image.cpp
  src/png.cpp
  src/parallel.cpp
  src/strips.cpp
  src/cpu/features.cpp
  src/cpu/kernels.cpp
  src/cpu/turns.cpp
  src/textlayer.cpp
  src/order.cpp
  src/score.cpp
  src/deskew.cpp
  src/output.cpp
  src/pdf_writer.cpp
  src/net/onnx.cpp
  src/net/net.cpp
  src/net/order_net.cpp)
add_library(carbon::carbon ALIAS carbon)
target_include_directories(carbon
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE src)
target_compile_features(carbon PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
# stb_image's header, compiled into src/image.cpp only; libdeflate for the searchable PDF's
# images and the PNG pages (src/png.cpp)
target_include_directories(carbon SYSTEM PRIVATE ${CARBON_STB_INCLUDE_DIR})
target_link_libraries(carbon PUBLIC Threads::Threads PRIVATE libdeflate::libdeflate_static)
carbon_warnings(carbon)
carbon_cpu_kernels(carbon)
# The crop rule's coefficients must round as they do on the GPU: no fused multiply-adds (MSVC does
# not contract under /fp:precise).
if(NOT MSVC)
  set_source_files_properties(src/lines.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# carbon-ocr serve (carbon/server.h): HTTP/1.1 over the engine, apart from the library so the library and the
# Python module carry no sockets.
add_library(carbon_serve STATIC src/serve/http.cpp src/serve/server.cpp)
add_library(carbon::serve ALIAS carbon_serve)
target_link_libraries(carbon_serve PUBLIC carbon)
if(WIN32)
  target_link_libraries(carbon_serve PRIVATE ws2_32)
endif()
carbon_warnings(carbon_serve)

if(CARBON_WITH_CUDA)
  target_sources(carbon PRIVATE src/cuda/lines.cu src/cuda/ctc_head.cu src/cuda/detect.cu
                 src/cuda/pipeline.cu)
  # -fmad=false: the same rounding as the CPU code, so the crops are the same bytes.
  set_source_files_properties(src/cuda/lines.cu PROPERTIES COMPILE_OPTIONS -fmad=false)
  target_link_libraries(carbon PRIVATE CUDA::cudart_static)
  target_compile_definitions(carbon PUBLIC CARBON_WITH_CUDA=1)
endif()

# PDFs: the library asks helper processes (PdfPool); PDFium itself is only in
# carbon_pdf, which the helper (carbon-pdf) and its tests link, so no process of
# the engine loads it.
if(CARBON_WITH_PDFIUM)
  target_sources(carbon PRIVATE src/pdf/pool.cpp)
  target_compile_definitions(carbon PUBLIC CARBON_WITH_PDFIUM=1)
  add_library(carbon_pdf STATIC src/pdf/pdf.cpp src/pdf/helper.cpp)
  add_library(carbon::pdf ALIAS carbon_pdf)
  target_include_directories(carbon_pdf PRIVATE src)
  target_link_libraries(carbon_pdf PUBLIC carbon PRIVATE pdfium::pdfium)
  carbon_warnings(carbon_pdf)
endif()

if(CARBON_WITH_ONNXRUNTIME)
  target_sources(carbon PRIVATE src/ort/ort_session.cpp)
  target_link_libraries(carbon PRIVATE onnxruntime::onnxruntime)
  target_compile_definitions(carbon PRIVATE CARBON_WITH_ONNXRUNTIME=1)
endif()

if(CARBON_WITH_TENSORRT)
  target_sources(carbon PRIVATE src/trt/trt_session.cpp)
  # Static cudart: no CUDA runtime DLL to ship next to the binaries.
  target_link_libraries(carbon PRIVATE TensorRT::nvinfer TensorRT::onnxparser CUDA::cudart_static)
  target_compile_definitions(carbon PRIVATE CARBON_WITH_TENSORRT=1)
  if(CARBON_WITH_CUDA)  # the batched pipeline's GPU models (carbon/pipeline.h)
    target_sources(carbon PRIVATE src/cuda/pipeline_cuda.cpp)
  endif()
endif()
