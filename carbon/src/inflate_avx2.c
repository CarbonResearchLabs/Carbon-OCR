/* libdeflate's decompressor compiled again, in a file of its own, with /arch:AVX2 (MSVC only, which has no
 * per-function target: the library's own BMI2 variant is compiled for baseline x86-64 bar its bzhi intrinsics, so
 * its shifts are not shlx and shrx). Under __AVX2__ the library's cpu_features.h has BMI2 native, so this copy is
 * the BMI2 decoder whole, with no dispatch; png.cpp calls it where the CPU has AVX2 and BMI2. Its public names
 * renamed, it shares the library's tables, allocator and adler-32 (their own files). Inflating is exact: the same
 * bytes as the library's. */
#define libdeflate_alloc_decompressor carbon_inflate_avx2_alloc
#define libdeflate_alloc_decompressor_ex carbon_inflate_avx2_alloc_ex
#define libdeflate_free_decompressor carbon_inflate_avx2_free
#define libdeflate_deflate_decompress carbon_inflate_avx2_deflate
#define libdeflate_deflate_decompress_ex carbon_inflate_avx2_deflate_ex
#define libdeflate_zlib_decompress carbon_inflate_avx2_zlib
#define libdeflate_zlib_decompress_ex carbon_inflate_avx2_zlib_ex
#include "lib/deflate_decompress.c"
#include "lib/zlib_decompress.c"
