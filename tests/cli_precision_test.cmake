# carbon-ocr ocr --precision on the fake set, which has its models in FP32 alone (its
# orient.fp16.onnx and det.fp16.onnx are the GPU's, with no recognizer beside them):
# fp32 reads the page as auto does, int8 finds no file, fp16 does not open, and a
# precision that is not one is a usage error.
#   cmake -DCLI=<carbon-ocr> -DDATA=<tests/data> -P cli_precision_test.cmake
set(page "${DATA}/fake-set/page.png")
foreach(precision auto fp32)
  execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --precision ${precision} --workers 1
                  RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
  string(REPLACE "\r" "" out "${out}")  # Windows' text mode
  if(NOT code EQUAL 0 OR NOT out STREQUAL "aaa\naaaaa\naa\n")
    message(FATAL_ERROR "--precision ${precision}: ${code}, '${out}', ${err}")
  endif()
endforeach()
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --precision int8 --workers 1
                RESULT_VARIABLE code ERROR_VARIABLE err)
if(code EQUAL 0 OR NOT err MATCHES "orient\\.int8\\.onnx: the set has no int8 file")
  message(FATAL_ERROR "--precision int8 on a set without int8 files: ${code}, ${err}")
endif()
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --precision fp16 --workers 1
                RESULT_VARIABLE code ERROR_VARIABLE err)
if(code EQUAL 0 OR NOT err MATCHES "orient\\.fp16\\.onnx")
  message(FATAL_ERROR "--precision fp16 on the GPU's FP16 files: ${code}, ${err}")
endif()
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --precision half
                RESULT_VARIABLE code ERROR_VARIABLE err)
if(NOT code EQUAL 2 OR NOT err MATCHES "unknown precision 'half'")
  message(FATAL_ERROR "--precision half: ${code}, ${err}")
endif()
