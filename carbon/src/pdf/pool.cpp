// PDF pages in helper processes (carbon/pdf.h): PdfPool, in the caller.  The
// helpers are carbon-pdf (helper.cpp, in carbon_pdf); what they say to each other
// is protocol.h.  PDFium is in the helpers only, never in this process.
//
// Windows: two anonymous pipes, only the child's ends inherited (an attribute
// list names them, so a process another thread starts at the same moment gets
// none).  POSIX: a socket pair, closed on exec but for the child's copies, and
// written without SIGPIPE, so a helper that died is an error, not a signal.
//
// Each helper renders into a shared region of its own, and only the page's size
// comes down the pipe (op 7): Windows, a section backed by the paging file, its
// handle duplicated into the helper; Linux, a memfd, other POSIX systems a POSIX
// shared memory object unlinked at once, given to the helper as its descriptor 3.
// A page larger than the region comes down the pipe.

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "carbon/parallel.h"
#include "carbon/pdf.h"
#include "protocol.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <thread>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
extern char** environ;
#endif

namespace carbon {

using namespace pdf_wire;

namespace {

std::string utf8(const std::filesystem::path& p) {
  const std::u8string s = p.u8string();
  return std::string(s.begin(), s.end());
}

}  // namespace

// ---------------------------------------------------------------- a helper process

struct PdfPool::Helper {
#if defined(_WIN32)
  HANDLE process = nullptr, to = nullptr, from = nullptr;
#else
  pid_t pid = -1;
  int fd = -1;
#endif
  std::string open;  // the file it has open, in UTF-8
  bool busy = false;
  // the region it renders into, mapped here read-only; null: pages come down the pipe
  const uint8_t* shared = nullptr;
  size_t shared_bytes = 0;
  bool shared_on = false;  // the helper has mapped it
#if defined(_WIN32)
  HANDLE mapping = nullptr;
#else
  int shm_fd = -1;
#endif

  ~Helper() {
    stop();
    unshare();
  }

  // The region, once: kept across restarts of the helper.
  void make_shared(size_t bytes) {
    if (bytes == 0) return;
#if defined(_WIN32)
    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(bytes >> 32),
                                 static_cast<DWORD>(bytes & 0xffffffffu), nullptr);
    if (!mapping) return;
    shared = static_cast<const uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, bytes));
    if (!shared) {
      CloseHandle(mapping);
      mapping = nullptr;
      return;
    }
#else
#if defined(__linux__) && defined(MFD_CLOEXEC)
    shm_fd = ::memfd_create("carbon-pdf", MFD_CLOEXEC);
#else
    const std::string name = "/carbon-pdf-" + std::to_string(::getpid()) + "-" +
                             std::to_string(reinterpret_cast<uintptr_t>(this));
    shm_fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (shm_fd >= 0) {
      ::shm_unlink(name.c_str());
      fcntl(shm_fd, F_SETFD, FD_CLOEXEC);
    }
#endif
    if (shm_fd < 0) return;
    if (shm_fd <= 3) {  // above 3, so the spawn's dup2 to 3 makes a copy (and clears close-on-exec)
      const int high = fcntl(shm_fd, F_DUPFD_CLOEXEC, 10);
      ::close(shm_fd);
      shm_fd = high;
      if (shm_fd < 0) return;
    }
    if (::ftruncate(shm_fd, static_cast<off_t>(bytes)) != 0) {
      ::close(shm_fd);
      shm_fd = -1;
      return;
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (p == MAP_FAILED) {
      ::close(shm_fd);
      shm_fd = -1;
      return;
    }
    shared = static_cast<const uint8_t*>(p);
#endif
    shared_bytes = bytes;
  }

  void unshare() {
    if (!shared) return;
#if defined(_WIN32)
    UnmapViewOfFile(shared);
    CloseHandle(mapping);
    mapping = nullptr;
#else
    ::munmap(const_cast<uint8_t*>(shared), shared_bytes);
    ::close(shm_fd);
    shm_fd = -1;
#endif
    shared = nullptr;
    shared_bytes = 0;
  }

  // The helper maps the region (op 8): its handle duplicated into it (Windows), or its
  // descriptor 3 (POSIX, set at spawn).
  void give_shared() {
    shared_on = false;
    if (!shared) return;
    int32_t id = 3;
#if defined(_WIN32)
    HANDLE remote = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), mapping, process, &remote, 0, FALSE, DUPLICATE_SAME_ACCESS)) return;
    id = static_cast<int32_t>(reinterpret_cast<uintptr_t>(remote));
#endif
    uint32_t status = 1;
    if (!ask(kShared, id, static_cast<double>(shared_bytes), "") || !get(&status, 4)) return;
    if (status != 0) {  // its error's text, then pages down the pipe
      uint32_t len = 0;
      if (!get(&len, 4)) return;
      std::string msg(len, '\0');
      if (!get(msg.data(), len)) return;
      return;
    }
    shared_on = true;
  }

#if defined(_WIN32)
  bool put(const void* p, size_t n) { return write_all(to, p, n); }
  bool get(void* p, size_t n) { return read_all(from, p, n); }
#else
  bool put(const void* p, size_t n) { return write_all(fd, p, n); }
  bool get(void* p, size_t n) { return read_all(fd, p, n); }
#endif

  bool ask(Op op, int32_t page, double scale, const std::string& file) {
    const uint32_t o = op, len = static_cast<uint32_t>(file.size());
    return put(&o, 4) && put(&page, 4) && put(&scale, 8) && put(&len, 4) && put(file.data(), file.size());
  }

  void start(const std::filesystem::path& exe) {
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE child_in = nullptr, parent_out = nullptr, parent_in = nullptr, child_out = nullptr;
    if (!CreatePipe(&child_in, &parent_out, &sa, 0)) throw Error("PdfPool: CreatePipe failed");
    if (!CreatePipe(&parent_in, &child_out, &sa, 0)) {
      CloseHandle(child_in);
      CloseHandle(parent_out);
      throw Error("PdfPool: CreatePipe failed");
    }
    SetHandleInformation(parent_out, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(parent_in, HANDLE_FLAG_INHERIT, 0);
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<char> attr_buf(size);
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
    InitializeProcThreadAttributeList(attrs, 1, 0, &size);
    HANDLE inherit[2] = {child_in, child_out};
    UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof inherit, nullptr, nullptr);
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = child_in;
    si.StartupInfo.hStdOutput = child_out;
    si.lpAttributeList = attrs;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + exe.wstring() + L"\"";
    const BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                   EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr, nullptr, &si.StartupInfo,
                                   &pi);
    const DWORD err = ok ? 0 : GetLastError();
    DeleteProcThreadAttributeList(attrs);
    CloseHandle(child_in);
    CloseHandle(child_out);
    if (!ok) {
      CloseHandle(parent_out);
      CloseHandle(parent_in);
      throw Error("PdfPool: cannot start " + exe.string() + " (error " + std::to_string(err) + ")");
    }
    CloseHandle(pi.hThread);
    process = pi.hProcess;
    to = parent_out;
    from = parent_in;
#else
    int sv[2];
#if defined(SOCK_CLOEXEC)
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) throw Error("PdfPool: socketpair failed");
#else
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw Error("PdfPool: socketpair failed");
    fcntl(sv[0], F_SETFD, FD_CLOEXEC);
    fcntl(sv[1], F_SETFD, FD_CLOEXEC);
#endif
#if defined(SO_NOSIGPIPE)
    const int one = 1;
    setsockopt(sv[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, sv[1], 0);  // dup2 clears close-on-exec on the copies
    posix_spawn_file_actions_adddup2(&fa, sv[1], 1);
    if (shm_fd >= 0) posix_spawn_file_actions_adddup2(&fa, shm_fd, 3);  // the shared region
    const std::string path = exe.string();
    char* argv[] = {const_cast<char*>(path.c_str()), nullptr};
    const int rc = posix_spawn(&pid, path.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(sv[1]);
    if (rc != 0) {
      ::close(sv[0]);
      pid = -1;
      throw Error("PdfPool: cannot start " + path + " (" + std::strerror(rc) + ")");
    }
    fd = sv[0];
#endif
    open.clear();
    give_shared();
  }

  // Its input closed, it finishes and exits; one that does not within 2 s is killed.
  void stop() {
#if defined(_WIN32)
    if (to) CloseHandle(to);
    if (from) CloseHandle(from);
    to = from = nullptr;
    if (process) {
      if (WaitForSingleObject(process, 2000) != WAIT_OBJECT_0) TerminateProcess(process, 1);
      CloseHandle(process);
      process = nullptr;
    }
#else
    if (fd >= 0) ::close(fd);
    fd = -1;
    if (pid > 0) {
      int status = 0;
      for (int i = 0; i < 200 && ::waitpid(pid, &status, WNOHANG) == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (::waitpid(pid, &status, WNOHANG) == 0) {
        ::kill(pid, SIGKILL);
        ::waitpid(pid, &status, 0);
      }
      pid = -1;
    }
#endif
  }
};

struct PdfPool::Wait {
  std::condition_variable free;
};

PdfPool::PdfPool(int processes, std::filesystem::path helper, size_t shared_bytes) : wait_(new Wait) {
  if (helper.empty()) {
    helper = running_program().parent_path() /
#if defined(_WIN32)
             "carbon-pdf.exe";
#else
             "carbon-pdf";
#endif
  }
  if (!std::filesystem::is_regular_file(helper)) throw Error("PdfPool: no helper program " + helper.string());
  helper_ = helper;
  if (processes <= 0) processes = cpu_topology().default_workers();
  for (int i = 0; i < processes; ++i) {
    helpers_.push_back(std::make_unique<Helper>());
    helpers_.back()->make_shared(shared_bytes);
    helpers_.back()->start(helper_);
  }
}

PdfPool::~PdfPool() = default;

int PdfPool::processes() const { return static_cast<int>(helpers_.size()); }

PdfPool::Helper& PdfPool::acquire(const std::string& file) {
  std::unique_lock<std::mutex> g(m_);
  for (;;) {
    Helper* any = nullptr;
    for (auto& h : helpers_) {
      if (h->busy) continue;
      if (h->open == file) {  // the document is open there already
        h->busy = true;
        return *h;
      }
      if (!any) any = h.get();
    }
    if (any) {
      any->busy = true;
      return *any;
    }
    wait_->free.wait(g);
  }
}

void PdfPool::release(Helper& h) {
  {
    std::lock_guard<std::mutex> g(m_);
    h.busy = false;
  }
  wait_->free.notify_one();
}

namespace {

// One request on a helper, taken and given back: on success the answer's body is
// the helper's to read; a failure answer throws its text; a helper gone is
// started again, and throws.
struct Call {
  PdfPool::Helper& h;
  const std::filesystem::path& exe;
  std::string what;

  [[noreturn]] void dead() {
    h.stop();
    h.start(exe);
    throw Error("PdfPool: the helper died on " + what);
  }

  void get(void* p, size_t n) {
    if (!h.get(p, n)) dead();
  }

  void run(Op op, int32_t page, double scale, const std::string& file) {
    uint32_t status = 0;
    if (!h.ask(op, page, scale, file) || !h.get(&status, 4)) dead();
    if (status != 0) {
      uint32_t len = 0;
      get(&len, 4);
      std::string msg(len, '\0');
      get(msg.data(), len);
      h.open.clear();
      throw Error(msg);
    }
    h.open = file;
  }
};

}  // namespace

template <typename F>
auto PdfPool::with_helper(const std::filesystem::path& file, F&& f) {
  const std::string key = utf8(file);
  Helper& h = acquire(key);
  struct Release {
    PdfPool* p;
    Helper& h;
    ~Release() { p->release(h); }
  } r{this, h};
  return f(h, key);
}

int PdfPool::pages(const std::filesystem::path& file) {
  return with_helper(file, [&](Helper& h, const std::string& key) {
    Call c{h, helper_, file.string()};
    c.run(kPages, 0, 0, key);
    int32_t n = 0;
    c.get(&n, 4);
    return static_cast<int>(n);
  });
}

GrayImage PdfPool::render(const std::filesystem::path& file, int page, double scale) {
  return with_helper(file, [&](Helper& h, const std::string& key) {
    Call c{h, helper_, file.string() + " page " + std::to_string(page)};
    const bool shared = h.shared_on;
    c.run(page < 0 ? kExit : shared ? kRenderShared : kRender, page, scale, key);  // a negative page: a crash
    int32_t wh[2] = {0, 0};
    c.get(wh, 8);
    uint32_t there = 0;
    if (shared) c.get(&there, 4);
    GrayImage img;
    img.width = wh[0];
    img.height = wh[1];
    img.pixels.resize(static_cast<size_t>(img.width) * img.height);
    if (there) {
      std::memcpy(img.pixels.data(), h.shared, img.pixels.size());
    } else {
      c.get(img.pixels.data(), img.pixels.size());
    }
    return img;
  });
}

std::vector<PdfChar> PdfPool::chars(const std::filesystem::path& file, int page, double scale) {
  return with_helper(file, [&](Helper& h, const std::string& key) {
    Call c{h, helper_, file.string() + " page " + std::to_string(page)};
    c.run(kChars, page, scale, key);
    uint32_t n = 0;
    c.get(&n, 4);
    std::vector<WireChar> wire(n);
    c.get(wire.data(), wire.size() * sizeof(WireChar));
    std::vector<PdfChar> out(n);
    for (uint32_t i = 0; i < n; ++i) {
      const WireChar& w = wire[i];
      PdfChar& o = out[i];
      o.code = static_cast<char32_t>(w.code);
      o.x0 = w.x0;
      o.y0 = w.y0;
      o.x1 = w.x1;
      o.y1 = w.y1;
      o.lx0 = w.lx0;
      o.ly0 = w.ly0;
      o.lx1 = w.lx1;
      o.ly1 = w.ly1;
      o.size = w.size;
      o.mcid = w.mcid;
      o.generated = w.flags & 1;
      o.bad_map = w.flags & 2;
      o.invisible = w.flags & 4;
    }
    return out;
  });
}

std::vector<LineBox> PdfPool::pictures(const std::filesystem::path& file, int page, double scale) {
  return with_helper(file, [&](Helper& h, const std::string& key) {
    Call c{h, helper_, file.string() + " page " + std::to_string(page)};
    c.run(kPictures, page, scale, key);
    uint32_t n = 0;
    c.get(&n, 4);
    std::vector<LineBox> out(n);
    for (LineBox& b : out) {
      double v[4];
      c.get(v, sizeof v);
      b = {v[0], v[1], v[2], v[3]};
    }
    return out;
  });
}

PdfStructure PdfPool::structure(const std::filesystem::path& file, int page) {
  return with_helper(file, [&](Helper& h, const std::string& key) {
    Call c{h, helper_, file.string() + " page " + std::to_string(page)};
    c.run(kStructure, page, 1.0, key);
    PdfStructure out;
    uint32_t ne = 0;
    c.get(&ne, 4);
    out.elements.resize(ne);
    for (PdfStructElement& e : out.elements) {
      int32_t parent = -1;
      uint32_t len = 0;
      c.get(&parent, 4);
      c.get(&len, 4);
      e.parent = parent;
      e.type.resize(len);
      if (len) c.get(e.type.data(), len);
    }
    uint32_t nr = 0;
    c.get(&nr, 4);
    out.runs.resize(nr);
    for (auto& r : out.runs) {
      int32_t v[2];
      c.get(v, sizeof v);
      r = {v[0], v[1]};
    }
    return out;
  });
}

std::filesystem::path running_program() {
#if defined(_WIN32)
  std::wstring buf(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0) throw Error("running_program: GetModuleFileNameW failed");
    if (n < buf.size()) {
      buf.resize(n);
      return std::filesystem::path(buf);
    }
    buf.resize(buf.size() * 2);
  }
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buf(size, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) throw Error("running_program: _NSGetExecutablePath failed");
  return std::filesystem::canonical(buf.c_str());
#else
  return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

}  // namespace carbon
