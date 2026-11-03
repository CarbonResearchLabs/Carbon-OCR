# Compiler settings shared by every Carbon target.

# Warnings for Carbon's own code.  Third-party headers come in through imported
# targets, which CMake marks as system headers, so they do not add warnings.
function(carbon_warnings target)
  if(MSVC)
    # /utf-8: sources and string literals are UTF-8 on every code page (files
    # read as cp1252 break on Windows).  CUDA sources go through
    # nvcc, which hands MSVC's options on with -Xcompiler.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8 /Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
  endif()
endfunction()

# The CPU kernels (carbon/cpu.h): one source file per instruction set, compiled
# with that set's flags and dispatched at run time, so the rest of the library
# stays baseline x86-64 or Armv8-A.  A kernel is built when the compiler takes
# its intrinsics; CARBON_KERNEL_<ISA> tells the dispatch it is there.  No
# contraction into fused multiply-adds (GCC and Clang lower the intrinsics to
# plain vector arithmetic, which they would fuse): every version and the scalar
# reference must round alike.  MSVC does not contract under /fp:precise.
include(CheckCXXSourceCompiles)
function(carbon_cpu_kernels target)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" proc)
  set(kernels)
  if(proc MATCHES "^(x86_64|amd64)$")
    if(MSVC)
      set(avx2_flags /arch:AVX2)
      set(avxvnni_flags /arch:AVX2)
      set(avx512_flags /arch:AVX512)
    else()
      set(avx2_flags -mavx2 -mfma)
      set(avxvnni_flags -mavx2 -mfma -mavxvnni)
      set(avx512_flags -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni)
    endif()
    set(avx2_probe "__m256 v = _mm256_fmadd_ps(_mm256_set1_ps(float(x)), _mm256_set1_ps(1), _mm256_setzero_ps())")
    set(avxvnni_probe "__m256i v = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_set1_epi32(x), _mm256_set1_epi32(1))")
    set(avx512_probe "__m512i v = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_set1_epi32(x), _mm512_set1_epi32(1))")
    set(kernels avx2 avxvnni avx512)
    set(header immintrin.h)
  elseif(proc MATCHES "^(aarch64|arm64)$")
    set(neon_flags)
    if(MSVC)
      set(neondot_flags)
    else()
      set(neondot_flags -march=armv8.2-a+dotprod)
    endif()
    set(neon_probe "float32x4_t v = vfmaq_f32(vdupq_n_f32(0), vdupq_n_f32(float(x)), vdupq_n_f32(1))")
    set(neondot_probe "int32x4_t v = vdotq_laneq_s32(vdupq_n_s32(0), vdupq_n_s8(1), vdupq_n_s8(static_cast<int8_t>(x)), 1)")
    set(kernels neon neondot)
    set(header arm_neon.h)
  endif()
  set(no_contract)
  if(NOT MSVC)
    set(no_contract -ffp-contract=off)
    set_source_files_properties(src/cpu/kernels.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
  endif()
  foreach(isa ${kernels})
    string(TOUPPER ${isa} up)
    set(CMAKE_REQUIRED_FLAGS "${${isa}_flags}")
    list(JOIN CMAKE_REQUIRED_FLAGS " " CMAKE_REQUIRED_FLAGS)
    check_cxx_source_compiles("
      #include <${header}>
      #include <cstdint>
      int main(int argc, char**) { int x = argc; ${${isa}_probe}; (void)v; return 0; }"
      CARBON_COMPILER_HAS_${up})
    if(CARBON_COMPILER_HAS_${up})
      set(src src/cpu/${isa}.cpp)
      target_sources(${target} PRIVATE ${src})
      set(options ${${isa}_flags} ${no_contract})
      set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "${options}")
      target_compile_definitions(${target} PRIVATE CARBON_KERNEL_${up}=1)
    endif()
  endforeach()
endfunction()

# Windows finds a DLL next to the executable first.  Copy the shared libraries a
# target links (ONNX Runtime) there after each build: System32 has an older
# onnxruntime.dll of its own, which would otherwise load and fail.
function(carbon_copy_runtime_dlls target)
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E
        $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${target}>>,copy_if_different,true>
        $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS)
  endif()
endfunction()
