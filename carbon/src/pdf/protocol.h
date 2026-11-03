// What PdfPool (pool.cpp, in carbon) and its helpers (helper.cpp, in carbon_pdf)
// say to each other, little-endian, one request at a time on the helper's standard
// input and output:
//
//   request  u32 op, i32 page, f64 scale, u32 bytes, the file's path in UTF-8
//   answer   u32 status: 0 and the op's answer, or 1, u32 bytes and an error's text
//
//   op 1 pages   i32 pages
//   op 2 render  i32 width, i32 height, the gray pixels
//   op 3 exit    none: the helper exits at once, as a crash would (tests)
//   op 4 chars     u32 count, then each character as WireChar
//   op 5 pictures  u32 count, then each box as 4 f64 (x0, y0, x1, y1)
//   op 6 structure u32 elements, each i32 parent, u32 bytes and its type; u32 runs, each
//                  i32 element, i32 MCID
//   op 7 render into the shared region   i32 width, i32 height, u32 where: 1 the pixels are
//                  in the region, 0 they follow (a page larger than the region)
//   op 8 shared    none: the helper maps the region the pool made for it, `page` its handle
//                  (Windows: duplicated into the helper) or file descriptor (POSIX: 3), `scale`
//                  its bytes; the pool asks it once, after starting the helper
#pragma once

#include <algorithm>
#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace carbon::pdf_wire {

enum Op : uint32_t {
  kPages = 1,
  kRender = 2,
  kExit = 3,
  kChars = 4,
  kPictures = 5,
  kStructure = 6,
  kRenderShared = 7,
  kShared = 8,
};

#pragma pack(push, 1)
struct WireChar {  // a PdfChar
  uint32_t code;
  float x0, y0, x1, y1, lx0, ly0, lx1, ly1, size;
  int32_t mcid;
  uint8_t flags;  // 1 generated, 2 bad map, 4 invisible
};
#pragma pack(pop)

#if defined(_WIN32)
using Handle = HANDLE;

inline bool read_all(Handle h, void* out, size_t n) {
  auto* p = static_cast<char*>(out);
  while (n) {
    DWORD got = 0;
    if (!ReadFile(h, p, static_cast<DWORD>(std::min<size_t>(n, size_t{1} << 30)), &got, nullptr) || got == 0) {
      return false;
    }
    p += got;
    n -= got;
  }
  return true;
}

inline bool write_all(Handle h, const void* in, size_t n) {
  const auto* p = static_cast<const char*>(in);
  while (n) {
    DWORD put = 0;
    if (!WriteFile(h, p, static_cast<DWORD>(std::min<size_t>(n, size_t{1} << 30)), &put, nullptr) || put == 0) {
      return false;
    }
    p += put;
    n -= put;
  }
  return true;
}
#else
using Handle = int;

inline bool read_all(Handle fd, void* out, size_t n) {
  auto* p = static_cast<char*>(out);
  while (n) {
    const ssize_t got = ::read(fd, p, n);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    p += got;
    n -= static_cast<size_t>(got);
  }
  return true;
}

// A socket's end, without SIGPIPE: a peer that died is an error, not a signal
// (Linux: MSG_NOSIGNAL; macOS: SO_NOSIGPIPE, set on the socket).
inline bool write_all(Handle fd, const void* in, size_t n) {
  const auto* p = static_cast<const char*>(in);
  while (n) {
#if defined(MSG_NOSIGNAL)
    const ssize_t put = ::send(fd, p, n, MSG_NOSIGNAL);
#else
    const ssize_t put = ::write(fd, p, n);
#endif
    if (put < 0 && errno == EINTR) continue;
    if (put <= 0) return false;
    p += put;
    n -= static_cast<size_t>(put);
  }
  return true;
}
#endif

}  // namespace carbon::pdf_wire
