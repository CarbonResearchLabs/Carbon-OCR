// What the CPU can run (carbon/cpu.h): cpuid and xgetbv on x86-64, the OS on arm64.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "carbon/cpu.h"

#if defined(_M_X64) || defined(__x86_64__)
#define CARBON_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(_M_ARM64) || defined(__aarch64__)
#define CARBON_ARM64 1
#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#endif

namespace carbon::cpu {
namespace {

#if defined(CARBON_X86)
void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int x[4];
  __cpuidex(x, leaf, sub);
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(x[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}
#endif

#if defined(CARBON_ARM64) && defined(__APPLE__)
bool sysctl_flag(const char* name) {
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, nullptr, 0) == 0 && v;
}
#endif

Features detect() {
  Features f;
#if defined(CARBON_X86)
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  char vendor[13] = {};
  std::memcpy(vendor, &r[1], 4);
  std::memcpy(vendor + 4, &r[3], 4);
  std::memcpy(vendor + 8, &r[2], 4);
  f.vendor = vendor;
  cpuid(0x80000000, 0, r);
  if (r[0] >= 0x80000004) {
    char brand[49] = {};
    for (int i = 0; i < 3; ++i) {
      cpuid(0x80000002 + i, 0, r);
      std::memcpy(brand + 16 * i, r, 16);
    }
    f.brand = brand;
    while (!f.brand.empty() && f.brand.back() == ' ') f.brand.pop_back();
    while (!f.brand.empty() && f.brand.front() == ' ') f.brand.erase(0, 1);
  }
  cpuid(1, 0, r);
  const bool osxsave = r[2] >> 27 & 1, avx = r[2] >> 28 & 1;
  f.sse41 = r[2] >> 19 & 1;
  const uint64_t xcr0 = osxsave ? xgetbv0() : 0;
  const bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xe6) == 0xe6, tiles = (xcr0 & 0x60000) == 0x60000;
  f.fma = ymm && avx && (r[2] >> 12 & 1);
  f.f16c = ymm && avx && (r[2] >> 29 & 1);
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    const unsigned ebx = r[1], ecx = r[2], edx = r[3];
    f.avx2 = ymm && (ebx >> 5 & 1);
    f.bmi2 = ebx >> 8 & 1;
    f.avx512f = zmm && (ebx >> 16 & 1);
    f.avx512dq = zmm && (ebx >> 17 & 1);
    f.avx512bw = zmm && (ebx >> 30 & 1);
    f.avx512vl = zmm && (ebx >> 31 & 1);
    f.avx512vnni = zmm && (ecx >> 11 & 1);
    f.avx512fp16 = zmm && (edx >> 23 & 1);
    f.amx_bf16 = tiles && (edx >> 22 & 1);
    f.amx_int8 = tiles && (edx >> 25 & 1);
    f.amx_tile = tiles && (edx >> 24 & 1);  // the tiles themselves (CPUID 7.0 EDX[24])
    cpuid(7, 1, r);
    f.avx_vnni = ymm && (r[0] >> 4 & 1);
    f.avx512bf16 = zmm && (r[0] >> 5 & 1);
    f.amx_fp16 = tiles && (r[0] >> 21 & 1);  // CPUID 7.1 EAX[21]
  }
  if (max_leaf >= 4) {  // the L1 data cache, from the deterministic cache parameters
    for (int sub = 0; sub < 16; ++sub) {
      cpuid(4, sub, r);
      const unsigned type = r[0] & 31, level = (r[0] >> 5) & 7;
      if (type == 0) break;
      if (type == 1 && level == 1) {  // data, L1
        const unsigned ways = (r[1] >> 22) + 1, parts = ((r[1] >> 12) & 0x3ff) + 1, line = (r[1] & 0xfff) + 1;
        f.l1d_bytes = static_cast<int>(ways * parts * line * (r[2] + 1));
        break;
      }
    }
  }
#elif defined(CARBON_ARM64)
  f.vendor = "arm64";
  f.neon = true;  // Advanced SIMD is part of Armv8-A
#if defined(__linux__)
  const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
  f.fp16 = hw >> 10 & 1;     // HWCAP_ASIMDHP: FP16 arithmetic on vectors
  f.dotprod = hw >> 20 & 1;  // HWCAP_ASIMDDP
  f.sve = hw >> 22 & 1;      // HWCAP_SVE
  f.sve2 = hw2 >> 1 & 1;     // HWCAP2_SVE2
  f.i8mm = hw2 >> 13 & 1;    // HWCAP2_I8MM
  f.bf16 = hw2 >> 14 & 1;    // HWCAP2_BF16
  if (FILE* c = std::fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r")) {  // L1D, "48K"
    int kb = 0;
    if (std::fscanf(c, "%d", &kb) == 1 && kb > 0) f.l1d_bytes = kb << 10;
    std::fclose(c);
  }
#elif defined(__APPLE__)
  {
    int64_t v = 0;
    size_t n = sizeof v;
    if (sysctlbyname("hw.perflevel0.l1dcachesize", &v, &n, nullptr, 0) == 0 && v > 0) f.l1d_bytes = static_cast<int>(v);
  }
  f.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
  f.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
  f.fp16 = sysctl_flag("hw.optional.arm.FEAT_FP16");
  f.bf16 = sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(_WIN32)
  f.dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE);
#endif
#endif
  return f;
}

}  // namespace

const Features& features() {
  static const Features f = detect();
  return f;
}

const char* isa_name(Isa isa) {
  switch (isa) {
    case Isa::kScalar: return "scalar";
    case Isa::kNeon: return "neon";
    case Isa::kAvx2: return "avx2";
    case Isa::kAvxVnni: return "avx-vnni";
    case Isa::kAvx512: return "avx512";
    case Isa::kNeonDot: return "neon-dot";
    case Isa::kAvx512Fp16: return "avx512-fp16";
    case Isa::kNeonFp16: return "neon-fp16";
  }
  return "?";
}

bool isa_available(Isa isa) {
  [[maybe_unused]] const Features& f = features();
  switch (isa) {
    case Isa::kScalar: return true;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: return f.avx2 && f.fma;
#endif
#if defined(CARBON_KERNEL_AVX2) && defined(CARBON_KERNEL_AVXVNNI)
    case Isa::kAvxVnni: return f.avx2 && f.fma && f.avx_vnni;  // FP32 as avx2
#endif
#if defined(CARBON_KERNEL_AVX2) && defined(CARBON_KERNEL_AVX512)
    case Isa::kAvx512:  // depthwise as avx2
      return f.avx2 && f.fma && f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.avx512vnni;
#endif
#if defined(CARBON_KERNEL_AVX2) && defined(CARBON_KERNEL_AVX512) && defined(CARBON_KERNEL_AVX512FP16)
    case Isa::kAvx512Fp16:  // int8 and FP32 as avx512
      return f.avx2 && f.fma && f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.avx512vnni && f.avx512fp16;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: return f.neon;
#endif
#if defined(CARBON_KERNEL_NEON) && defined(CARBON_KERNEL_NEONDOT)
    case Isa::kNeonDot: return f.neon && f.dotprod;  // FP32 as neon
#endif
#if defined(CARBON_KERNEL_NEON) && defined(CARBON_KERNEL_NEONDOT) && defined(CARBON_KERNEL_NEONFP16)
    case Isa::kNeonFp16: return f.neon && f.dotprod && f.fp16;  // int8 as neon-dot, FP32 as neon
#endif
    default: return false;
  }
}

std::vector<Isa> isas_available() {
  std::vector<Isa> out;
  for (Isa i : {Isa::kScalar, Isa::kNeon, Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512, Isa::kAvx512Fp16, Isa::kNeonDot,
                Isa::kNeonFp16}) {
    if (isa_available(i)) out.push_back(i);
  }
  return out;
}

Isa best_isa() {
  static const Isa best = [] {
    const auto all = isas_available();
    std::string want;
#if defined(_MSC_VER)
    char* v = nullptr;
    size_t len = 0;
    if (_dupenv_s(&v, &len, "CARBON_ISA") == 0 && v) want = v;
    std::free(v);
#else
    if (const char* v = std::getenv("CARBON_ISA")) want = v;
#endif
    for (Isa i : all) {
      if (want == isa_name(i)) return i;
    }
    return all.back();
  }();
  return best;
}

}  // namespace carbon::cpu
