# carbon_ocr, the Python module (nanobind), built by scikit-build-core from
# python/pyproject.toml: `pip install ./python`.

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NB_STATIC: nanobind's runtime linked in, so the module is one file.
nanobind_add_module(carbon_ocr NB_STATIC src/carbon_ocr.cpp)
target_link_libraries(carbon_ocr PRIVATE carbon::carbon)
carbon_warnings(carbon_ocr)
install(TARGETS carbon_ocr LIBRARY DESTINATION .)

# ONNX Runtime's shared library goes next to the module, which finds it there:
# Windows looks in the module's folder first, and on Linux the module's RPATH
# is its own folder.  The library is installed under its soname.
if(CARBON_WITH_ONNXRUNTIME)
  if(WIN32)
    install(FILES $<TARGET_FILE:onnxruntime::onnxruntime> DESTINATION .)
  else()
    get_target_property(_ort_lib onnxruntime::onnxruntime IMPORTED_LOCATION)
    file(REAL_PATH "${_ort_lib}" _ort_real)
    install(FILES "${_ort_real}" DESTINATION . RENAME libonnxruntime.so.1)
    set_target_properties(carbon_ocr PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
  endif()
endif()

