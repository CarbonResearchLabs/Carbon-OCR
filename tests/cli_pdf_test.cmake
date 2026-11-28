# carbon-ocr ocr on a PDF (tests/data/text.pdf, by tools/make_test_pdfs.py), its
# pages rendered in helper processes: read from their text layer, which passes
# the checks, and by OCR with --text-layer off, which says so; written as hOCR,
# and as PAGE XML a file a page.  figure.pdf: a hybrid page.
#   cmake -DCLI=<carbon-ocr> -DDATA=<tests/data> -DOUT=<dir> -P cli_pdf_test.cmake
# --layer-lines layer: the layer's own lines (PDFium's), each page's "Hello PDF"
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format json --workers 2
                        --layer-lines layer
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
  string(JSON rotation GET "${out}" pages ${i} rotation)
  math(EXPR want "${i} + 1")
  math(EXPR turned "${i} * 90")  # the second page shows its text down the page: read upright, rotation 90
  if(NOT page EQUAL want OR NOT source STREQUAL "text layer" OR NOT text STREQUAL "Hello PDF"
     OR NOT rotation EQUAL turned)
    message(FATAL_ERROR "page ${i}: page ${page}, source '${source}', first line '${text}', rotation ${rotation}: ${out}")
  endif()
endforeach()
# a text layer drawn turned (turned.pdf: up the page by its content, shown upright by /Rotate, upside down) read
# upright, each page's rotation the degrees its text is turned clockwise as shown
execute_process(COMMAND "${CLI}" ocr "${DATA}/turned.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --layer-lines layer
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr on turned.pdf failed (${code}): ${err}")
endif()
set(rotations 270 0 180)
foreach(i 0 1 2)
  string(JSON source GET "${out}" pages ${i} source)
  string(JSON text GET "${out}" pages ${i} lines 0 text)
  string(JSON rotation GET "${out}" pages ${i} rotation)
  list(GET rotations ${i} want)
  if(NOT source STREQUAL "text layer" OR NOT text STREQUAL "Hello PDF" OR NOT rotation EQUAL want)
    message(FATAL_ERROR "turned.pdf page ${i}: source '${source}', first line '${text}', rotation ${rotation}: ${out}")
  endif()
endforeach()
string(JSON text GET "${out}" pages 3 lines 0 text)  # a line of three runs, upside down: read in their order
string(JSON rotation GET "${out}" pages 3 rotation)
if(NOT text STREQUAL "Hello PDF 12" OR NOT rotation EQUAL 180)
  message(FATAL_ERROR "turned.pdf page 3: first line '${text}', rotation ${rotation}: ${out}")
endif()
# a layer whose unreadable codes are a run of their own (badruns.pdf): its four sound lines kept, the run's line read
# by OCR with the pictures; --hybrid off, the page by OCR as before
execute_process(COMMAND "${CLI}" ocr "${DATA}/badruns.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --layer-lines layer
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(JSON source GET "${out}" pages 0 source)
string(REGEX MATCHALL "\"text\": \"a line of sound text\"" sound "${out}")
list(LENGTH sound n)
if(NOT code EQUAL 0 OR source STREQUAL "ocr" OR NOT n EQUAL 4)
  message(FATAL_ERROR "badruns.pdf (${code}): source '${source}', ${n} sound lines: ${out}${err}")
endif()
execute_process(COMMAND "${CLI}" ocr "${DATA}/badruns.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --hybrid off
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(JSON source GET "${out}" pages 0 source)
string(JSON note GET "${out}" pages 0 text_layer)
string(FIND "${note}" "not readable" why)
if(NOT code EQUAL 0 OR NOT source STREQUAL "ocr" OR why EQUAL -1)
  message(FATAL_ERROR "badruns.pdf --hybrid off (${code}): source '${source}', '${note}'")
endif()
# a text layer's sub- and superscripts (scripts.pdf) in the HTML
execute_process(COMMAND "${CLI}" ocr "${DATA}/scripts.pdf" --set "${DATA}/fake-set" --format html --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(FIND "${out}" "H<sub>2</sub>O" sub)
string(FIND "${out}" "mc<sup>2</sup>" sup)
if(NOT code EQUAL 0 OR sub EQUAL -1 OR sup EQUAL -1)
  message(FATAL_ERROR "scripts.pdf as HTML (${code}): ${out}${err}")
endif()
# by default a page's words go into the detector's lines (the fake set's detector parts the two words: its lines
# are its own), every word of the layer kept and in order; the page whose text runs down the page too, read upright
execute_process(COMMAND "${CLI}" ocr "${DATA}/text.pdf" --set "${DATA}/fake-set" --format text --workers 2
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr (the detector's lines) failed (${code}): ${err}")
endif()
string(REGEX REPLACE "[ 
]+" " " flat "${out}")
string(STRIP "${flat}" flat)
if(NOT flat STREQUAL "Hello PDF Hello PDF")
  message(FATAL_ERROR "the detector's lines: '${flat}'")
endif()
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
                        --workers 1 --layer-lines layer
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
# set reads as "aaa"): its picture read by OCR, its lines joining the layer's (PDFium's lines, --layer-lines layer:
# the fake set's detector would part the words); --hybrid off, not
execute_process(COMMAND "${CLI}" ocr "${DATA}/figure.pdf" --set "${DATA}/fake-set" --format json --workers 1
                        --layer-lines layer
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
                        --hybrid off --layer-lines layer
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
string(JSON source GET "${out}" pages 0 source)
string(JSON n LENGTH "${out}" pages 0 lines)
if(NOT code EQUAL 0 OR NOT source STREQUAL "text layer" OR NOT n EQUAL 1)
  message(FATAL_ERROR "figure.pdf --hybrid off (${code}): source '${source}', ${n} lines: ${out}${err}")
endif()
# the detector's lines (the default): the picture's bar read from the line the detector found on the whole page
# (no second detection of the picture's page), and with --hybrid detect from that page detected again; both the
# same hybrid page
foreach(mode auto detect)
  execute_process(COMMAND "${CLI}" ocr "${DATA}/figure.pdf" --set "${DATA}/fake-set" --format json --workers 2
                          --hybrid ${mode}
                  RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
  string(JSON source GET "${out}" pages 0 source)
  string(FIND "${out}" "\"text\": \"aaa\"" bar)
  string(FIND "${err}" "1 of those with pictures read" counted)
  if(NOT code EQUAL 0 OR NOT source STREQUAL "hybrid" OR bar EQUAL -1 OR counted EQUAL -1)
    message(FATAL_ERROR "figure.pdf --hybrid ${mode} (${code}): source '${source}': ${out}${err}")
  endif()
endforeach()

# Pages of many files, PDFs' and images', written as they finish in the order given, the same as each file
# read alone; --text-out beside them
set(files "${DATA}/figure.pdf" "${DATA}/fake-set/page.png" "${DATA}/text.pdf" "${DATA}/fake-set/page.png")
execute_process(COMMAND "${CLI}" ocr ${files} --set "${DATA}/fake-set" --format text --workers 2
                        --text-out "${OUT}/all.txt"
                RESULT_VARIABLE code OUTPUT_VARIABLE all ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr on four files failed (${code}): ${err}")
endif()
set(want "")
set(first TRUE)
string(ASCII 12 ff)
foreach(f ${files})
  execute_process(COMMAND "${CLI}" ocr "${f}" --set "${DATA}/fake-set" --format text --workers 1
                  RESULT_VARIABLE code OUTPUT_VARIABLE one ERROR_VARIABLE err ENCODING UTF-8)
  if(NOT first)
    string(APPEND want "${ff}\n")
  endif()
  string(APPEND want "${one}")
  set(first FALSE)
endforeach()
file(READ "${OUT}/all.txt" text_out)
if(NOT all STREQUAL want OR NOT text_out STREQUAL want)
  message(FATAL_ERROR "four files' pages:\n${all}\nagainst each alone:\n${want}\n--text-out:\n${text_out}")
endif()

# each file read by what its first bytes say, not its name: the PDF named .png is a PDF of two pages from its text
# layer, the page's PNG named .pdf an image
file(MAKE_DIRECTORY "${OUT}")
file(COPY_FILE "${DATA}/text.pdf" "${OUT}/named.png")
file(COPY_FILE "${DATA}/fake-set/page.png" "${OUT}/named.pdf")
execute_process(COMMAND "${CLI}" ocr "${OUT}/named.png" "${OUT}/named.pdf" --set "${DATA}/fake-set" --format json
                        --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "files named for another format (${code}): ${err}")
endif()
string(JSON n LENGTH "${out}" pages)
string(JSON first GET "${out}" pages 0 source)
string(JSON last GET "${out}" pages 2 source)
if(NOT n EQUAL 3 OR NOT first STREQUAL "text layer" OR NOT last STREQUAL "ocr")
  message(FATAL_ERROR "files named for another format: ${n} pages, sources '${first}', '${last}': ${out}")
endif()
