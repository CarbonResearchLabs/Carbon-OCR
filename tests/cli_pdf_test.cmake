# carbon-ocr ocr on a PDF (tests/data/text.pdf, by tools/make_test_pdfs.py), its
# pages rendered in helper processes: read from their text layer, which passes
# the checks, and by OCR with --text-layer off, which says so; written as hOCR,
# and as PAGE XML a file a page.  figure.pdf: a hybrid page.
#   cmake -DCLI=<carbon-ocr> -DDATA=<tests/data> -DOUT=<dir> -P cli_pdf_test.cmake
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format json --workers 2
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr failed (${code}): ${err}")
endif()
string(JSON n LENGTH "${out}" pages)
if(NOT n EQUAL 2)
  message(FATAL_ERROR "expected the PDF's two pages, got: ${out}")
endif()
foreach(i 0 1)
  string(JSON page GET "${out}" pages ${i} page)
  string(JSON source GET "${out}" pages ${i} source)
  string(JSON text GET "${out}" pages ${i} lines 0 text)
  math(EXPR want "${i} + 1")
  if(NOT page EQUAL want OR NOT source STREQUAL "text layer" OR NOT text STREQUAL "Hello PDF")
    message(FATAL_ERROR "page ${i}: page ${page}, source '${source}', first line '${text}': ${out}")
  endif()
endforeach()
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --text-layer off
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr --text-layer off failed (${code}): ${err}")
endif()
string(JSON source GET "${out}" pages 0 source)
if(NOT source STREQUAL "ocr")
  message(FATAL_ERROR "--text-layer off: source '${source}'")
endif()

# hOCR, and PAGE XML a file a page
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format hocr --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(FIND "${out}" "class=\"ocrx_word\"" word)
string(FIND "${out}" ">Hello</span>" hello)
if(NOT code EQUAL 0 OR word EQUAL -1 OR hello EQUAL -1)
  message(FATAL_ERROR "--format hocr (${code}): ${out}${err}")
endif()
file(REMOVE_RECURSE "${OUT}")
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format page --out-dir "${OUT}"
                        --workers 1
                RESULT_VARIABLE code ERROR_VARIABLE err)
if(NOT code EQUAL 0 OR NOT EXISTS "${OUT}/text-1.xml" OR NOT EXISTS "${OUT}/text-2.xml")
  message(FATAL_ERROR "--format page (${code}): ${err}")
endif()
file(READ "${OUT}/text-2.xml" page2)
string(FIND "${page2}" "<Unicode>Hello PDF</Unicode>" line)
if(line EQUAL -1)
  message(FATAL_ERROR "text-2.xml: ${page2}")
endif()

# a page read from its text layer whose picture holds text of its own (figure.pdf: a bar the fake
# set reads as "aaa"): its picture read by OCR, its lines joining the layer's; --hybrid off, not
execute_process(COMMAND "${CLI}" ocr "${DATA}/figure.pdf" --set "${DATA}/fake-set" --format json --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr on figure.pdf failed (${code}): ${err}")
endif()
string(JSON source GET "${out}" pages 0 source)
string(JSON n LENGTH "${out}" pages 0 lines)
string(JSON pictures LENGTH "${out}" pages 0 pictures)
string(FIND "${out}" "\"text\": \"Hello PDF\"" hello)
string(FIND "${out}" "\"text\": \"aaa\"" bar)
if(NOT source STREQUAL "hybrid" OR NOT n EQUAL 2 OR NOT pictures EQUAL 1 OR hello EQUAL -1 OR bar EQUAL -1)
  message(FATAL_ERROR "figure.pdf: source '${source}', ${n} lines, ${pictures} pictures: ${out}")
endif()
execute_process(COMMAND "${CLI}" ocr "${DATA}/figure.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --hybrid off
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(JSON source GET "${out}" pages 0 source)
string(JSON n LENGTH "${out}" pages 0 lines)
if(NOT code EQUAL 0 OR NOT source STREQUAL "text layer" OR NOT n EQUAL 1)
  message(FATAL_ERROR "figure.pdf --hybrid off (${code}): source '${source}', ${n} lines: ${out}${err}")
endif()
