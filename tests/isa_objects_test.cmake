# The files built for an instruction set (carbon/src/cpu/avx2.cpp, png_avx2.cpp,
# avxvnni.cpp, avx512.cpp, avx512fp16.cpp, neondot.cpp, neonfp16.cpp) define no code
# another file could share (kernels.h): an inline function or a template's instance
# they emit is one copy the linker keeps for the whole program, and that copy could
# run AVX-512 code on a CPU without it.  Code only: data shared that way is the same
# bytes in every file.  In an optimized build, where the standard library's one-line
# accessors are inlined.
#
#   cmake -DOBJECTS=<object>|<object>... (-DNM=<nm> | -DDUMP=<link.exe>) -P isa_objects_test.cmake

cmake_minimum_required(VERSION 3.24)
string(REPLACE "|" ";" objects "${OBJECTS}")
list(FILTER objects INCLUDE REGEX "[/\\\\](png_)?(avx2|avxvnni|avx512|avx512fp16|neondot|neonfp16)(\\.cpp)?\\.(o|obj)$")
if(NOT objects)
  message(STATUS "no file built for an instruction set: nothing to check")
  return()
endif()

set(shared "")
foreach(obj IN LISTS objects)
  get_filename_component(name "${obj}" NAME)
  if(DUMP)
    # COFF (MSVC): an external function in a COMDAT section the linker may pick from
    # any file (selection 2, 3, 4 or 6; 1 is a function defined once).  Names in an
    # anonymous namespace (?A0x) and lambdas are this file's own.
    execute_process(COMMAND "${DUMP}" /dump /symbols "${obj}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "${DUMP} /dump /symbols ${obj}: ${rc}")
    endif()
    string(REGEX REPLACE "[][;]" "_" out "${out}")  # whole lines for the list
    string(REGEX MATCHALL "[^\n]+" lines "${out}")
    set(section "")
    set(picked "")  # this file's sections the linker picks from any file
    set(funcs "")
    foreach(line IN LISTS lines)
      if(line MATCHES "^[0-9A-F]+ [0-9A-F]+ (SECT[0-9A-F]+) +notype +Static +\\| \\.")
        set(section "${CMAKE_MATCH_1}")  # a section; its details on the next line
        continue()
      endif()
      if(section)
        if(line MATCHES "selection +([2346]) ")
          list(APPEND picked "${section}")
        endif()
        set(section "")
      endif()
      if(line MATCHES "^[0-9A-F]+ [0-9A-F]+ (SECT[0-9A-F]+) +notype \\(\\) +External +\\| ([^ ]+)")
        list(APPEND funcs "${CMAKE_MATCH_1} ${CMAKE_MATCH_2}")
      endif()
    endforeach()
    foreach(f IN LISTS funcs)
      string(REGEX MATCH "^([^ ]+) (.*)$" _ "${f}")
      set(sect "${CMAKE_MATCH_1}")
      set(sym "${CMAKE_MATCH_2}")
      if(sect IN_LIST picked AND NOT sym MATCHES "\\?A0x|<lambda_")
        list(APPEND shared "${name}: ${sym}")
      endif()
    endforeach()
  else()
    # ELF: a weak function (W, w).  Clang's __clang_call_terminate calls std::terminate.
    execute_process(COMMAND "${NM}" --defined-only -C "${obj}" OUTPUT_VARIABLE out RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "${NM} --defined-only ${obj}: ${rc}")
    endif()
    string(REGEX REPLACE "[][;]" "_" out "${out}")  # whole lines for the list
    string(REGEX MATCHALL "[^\n]+" lines "${out}")
    foreach(line IN LISTS lines)
      if(line MATCHES "^[0-9a-fA-F]+ [Ww] (.+)$" AND NOT CMAKE_MATCH_1 STREQUAL "__clang_call_terminate")
        list(APPEND shared "${name}: ${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endif()
endforeach()

list(LENGTH objects n)
if(shared)
  list(JOIN shared "\n  " text)
  message(FATAL_ERROR "code the files built for an instruction set share with others (kernels.h):\n  ${text}")
endif()
message(STATUS "${n} files built for an instruction set define no code another file could share")
