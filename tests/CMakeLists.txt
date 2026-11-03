include(GoogleTest)

add_executable(carbon_tests cpp/tensor_test.cpp cpp/lines_test.cpp cpp/parallel_test.cpp cpp/strips_test.cpp cpp/detect_test.cpp
  cpp/pipeline_test.cpp cpp/cpu_test.cpp cpp/net_test.cpp cpp/textlayer_test.cpp cpp/output_test.cpp cpp/deskew_test.cpp cpp/order_test.cpp
  cpp/score_test.cpp cpp/order_net_test.cpp cpp/image_test.cpp
  cpp/png_test.cpp)
# png_test writes PNGs with libdeflate and holds the decoder to stb_image, whose
# functions carbon carries (src/image.cpp)
target_link_libraries(carbon_tests PRIVATE carbon::carbon GTest::gtest_main libdeflate::libdeflate_static)
target_include_directories(carbon_tests SYSTEM PRIVATE ${CARBON_STB_INCLUDE_DIR})
target_compile_definitions(carbon_tests PRIVATE
  CARBON_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
carbon_warnings(carbon_tests)

if(CARBON_WITH_ONNXRUNTIME)
  target_sources(carbon_tests PRIVATE cpp/ort_session_test.cpp cpp/engine_test.cpp cpp/batch_pipeline_test.cpp)
endif()
# PDFium in a test executable of its own (carbon_pdf), as in the engine's processes.
if(CARBON_WITH_PDFIUM)
  add_executable(carbon_pdf_tests cpp/pdf_test.cpp)
  target_link_libraries(carbon_pdf_tests PRIVATE carbon::pdf GTest::gtest_main)
  target_compile_definitions(carbon_pdf_tests PRIVATE CARBON_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
  if(TARGET carbon-pdf)  # the helper processes' program, for PdfPool
    target_compile_definitions(carbon_pdf_tests PRIVATE CARBON_PDF_HELPER="$<TARGET_FILE:carbon-pdf>")
    add_dependencies(carbon_pdf_tests carbon-pdf)
  endif()
  carbon_warnings(carbon_pdf_tests)
  carbon_copy_runtime_dlls(carbon_pdf_tests)
  gtest_discover_tests(carbon_pdf_tests DISCOVERY_MODE PRE_TEST)
endif()
if(CARBON_WITH_CUDA)
  target_sources(carbon_tests PRIVATE cpp/cuda_lines_test.cpp cpp/cuda_ctc_head_test.cpp)
  target_link_libraries(carbon_tests PRIVATE CUDA::cudart_static)
endif()
if(CARBON_WITH_TENSORRT)
  target_sources(carbon_tests PRIVATE cpp/trt_session_test.cpp)
  if(CARBON_WITH_CUDA AND CARBON_WITH_ONNXRUNTIME)  # the batched pipeline on a GPU, against the CPU's
    target_sources(carbon_tests PRIVATE cpp/cuda_pipeline_test.cpp)
  endif()
  target_link_libraries(carbon_tests PRIVATE CUDA::cudart_static)
endif()

carbon_copy_runtime_dlls(carbon_tests)

if(CARBON_BUILD_CLI AND CARBON_WITH_ONNXRUNTIME)
  # carbon-ocr ocr on a page whose path is not ASCII
  add_test(NAME Cli.OcrPathAnyCharacters
    COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:carbon-ocr> -DDATA=${CMAKE_CURRENT_SOURCE_DIR}/data
            -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cli -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_test.cmake)
  # carbon-ocr ocr --precision: the files it picks, and a set without them
  add_test(NAME Cli.OcrPrecision
    COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:carbon-ocr> -DDATA=${CMAKE_CURRENT_SOURCE_DIR}/data
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_precision_test.cmake)
  if(CARBON_WITH_PDFIUM)
    # carbon-ocr ocr on a PDF: its pages in helper processes, read from their text layer
    add_test(NAME Cli.OcrPdfTextLayer
      COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:carbon-ocr> -DDATA=${CMAKE_CURRENT_SOURCE_DIR}/data
              -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cli-pdf -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_pdf_test.cmake)
  endif()
endif()

# The files built for an instruction set share no code with the others
# (carbon/src/cpu/kernels.h), in an optimized build: their symbols from nm, or from
# MSVC's link /dump.
unset(_symbols)
if(MSVC AND CMAKE_LINKER MATCHES "[/\\\\]link\\.exe$")
  set(_symbols -DDUMP=${CMAKE_LINKER})
elseif(NOT MSVC AND CMAKE_NM)
  set(_symbols -DNM=${CMAKE_NM})
endif()
set(_configs "")
if(CMAKE_CONFIGURATION_TYPES)
  set(_configs CONFIGURATIONS Release RelWithDebInfo MinSizeRel)
elseif(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
  unset(_symbols)
endif()
if(DEFINED _symbols)
  add_test(NAME Cpu.IsaFilesShareNoCode
    COMMAND ${CMAKE_COMMAND} "-DOBJECTS=$<JOIN:$<TARGET_OBJECTS:carbon>,|>" ${_symbols}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/isa_objects_test.cmake
    ${_configs})
endif()

set(_env "")
if(CARBON_WITH_TENSORRT)
  foreach(_dir ${TensorRT_RUNTIME_DIRS})
    list(APPEND _env "PATH=path_list_prepend:${_dir}")
  endforeach()
endif()
# PRE_TEST: tests are listed when ctest runs, after the DLLs are in place.
gtest_discover_tests(carbon_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT_MODIFICATION "${_env}")
