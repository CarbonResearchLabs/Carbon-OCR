add_executable(carbon-ocr main.cpp)
if(WIN32)
  # UTF-8 arguments, so a path with any characters opens
  target_sources(carbon-ocr PRIVATE carbon-ocr.manifest)
endif()
target_link_libraries(carbon-ocr PRIVATE carbon::carbon)
if(CARBON_WITH_TENSORRT)
  # bench also times the run on device buffers
  target_link_libraries(carbon-ocr PRIVATE CUDA::cudart_static)
  target_compile_definitions(carbon-ocr PRIVATE CARBON_WITH_TENSORRT=1)
endif()
carbon_warnings(carbon-ocr)
carbon_copy_runtime_dlls(carbon-ocr)

# carbon-pdf: the helper processes PDF pages render in (carbon::PdfPool), beside
# carbon-ocr, where the pool looks for it
if(CARBON_WITH_PDFIUM)
  add_executable(carbon-pdf pdf_main.cpp)
  if(WIN32)
    target_sources(carbon-pdf PRIVATE carbon-ocr.manifest)
  endif()
  target_link_libraries(carbon-pdf PRIVATE carbon::pdf)
  carbon_warnings(carbon-pdf)
  carbon_copy_runtime_dlls(carbon-pdf)
  add_dependencies(carbon-ocr carbon-pdf)
endif()
