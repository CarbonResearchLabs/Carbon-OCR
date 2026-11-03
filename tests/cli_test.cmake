# carbon-ocr ocr on the fake set's page, copied to a path that is not ASCII: on
# Windows the arguments arrive as UTF-8 only through carbon-ocr.manifest.  Then as
# the OCRmyPDF plugin calls it.
#   cmake -DCLI=<carbon-ocr> -DDATA=<tests/data> -DOUT=<dir> -P cli_test.cmake
set(page "${OUT}/café 漢字/página.png")
file(MAKE_DIRECTORY "${OUT}/café 漢字")
file(COPY_FILE "${DATA}/fake-set/page.png" "${page}")
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --format json --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "carbon-ocr failed (${code}): ${err}")
endif()
string(JSON file GET "${out}" pages 0 file)
if(NOT file STREQUAL page)
  message(FATAL_ERROR "the page's name came back as '${file}', not '${page}'")
endif()
set(texts "")
string(JSON n LENGTH "${out}" pages 0 lines)
math(EXPR last "${n} - 1")
foreach(i RANGE ${last})
  string(JSON t GET "${out}" pages 0 lines ${i} text)
  list(APPEND texts "${t}")
endforeach()
if(NOT texts STREQUAL "aaa;aaaaa;aa")
  message(FATAL_ERROR "expected the page's three lines, got: ${out}")
endif()

# as OCRmyPDF's plugin calls it: the page as given, its text layer alone at the image's dpi,
# the text beside
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --format pdf --text-only --image-dpi 150
                        --as-given --output "${OUT}/text-only.pdf" --text-out "${OUT}/text-only.txt" --workers 1
                RESULT_VARIABLE code ERROR_VARIABLE err)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "--format pdf --text-only failed (${code}): ${err}")
endif()
file(READ "${OUT}/text-only.txt" text)
if(NOT text STREQUAL "aaa\naaaaa\naa\n")
  message(FATAL_ERROR "--text-out wrote '${text}'")
endif()
# file(STRINGS): the printable runs of a binary file (file(READ) stops at its first NUL)
file(STRINGS "${OUT}/text-only.pdf" image REGEX "/Subtype /Image")
file(STRINGS "${OUT}/text-only.pdf" box REGEX "/MediaBox \\[0 0 384 491\\.52\\]")  # 800 x 1024 px at 150 dpi
file(STRINGS "${OUT}/text-only.pdf" font REGEX "/GlyphLessFont")  # the words' (their stream is deflated)
if(image OR NOT box OR NOT font)
  message(FATAL_ERROR "text-only.pdf: an image ('${image}'), not the page's size ('${box}') or no text font")
endif()

# --format html: an HTML5 document, the page a section of its paragraphs (the fake set has no
# order model, so its lines make paragraphs by their gaps), every line's text in them, in order
execute_process(COMMAND "${CLI}" ocr "${page}" --set "${DATA}/fake-set" --format html --workers 1
                RESULT_VARIABLE code OUTPUT_VARIABLE html ERROR_VARIABLE err ENCODING UTF-8)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "--format html failed (${code}): ${err}")
endif()
string(FIND "${html}" "<!DOCTYPE html>" doctype)
string(FIND "${html}" "<section class=\"page\" id=\"page-1\">" section)
string(REGEX MATCHALL "<p>[a ]+</p>" paras "${html}")
string(REPLACE "</p>;<p>" " " paras "${paras}")
if(NOT doctype EQUAL 0 OR section LESS 0 OR NOT paras STREQUAL "<p>aaa aaaaa aa</p>")
  message(FATAL_ERROR "--format html wrote: ${html}")
endif()
