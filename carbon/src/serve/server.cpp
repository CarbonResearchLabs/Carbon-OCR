#include "carbon/server.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "carbon/document.h"
#include "carbon/error.h"
#include "carbon/http.h"
#include "carbon/image.h"
#include "carbon/output.h"
#include "carbon/pdf.h"
#include "carbon/version.h"

namespace carbon {

namespace {

// ---------------------------------------------------------------- sockets, on both systems

#if defined(_WIN32)
using Sock = SOCKET;
constexpr Sock kNoSock = INVALID_SOCKET;
using PollFd = WSAPOLLFD;
int last_error() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK || e == WSAEINTR; }
void close_sock(Sock s) { closesocket(s); }
int poll_socks(PollFd* fds, size_t n, int ms) { return WSAPoll(fds, static_cast<ULONG>(n), ms); }
void set_nonblocking(Sock s) {
  u_long on = 1;
  ioctlsocket(s, FIONBIO, &on);
}
constexpr int kSendFlags = 0;
struct Net {  // Winsock, started once a server, stopped with it
  Net() {
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw Error("serve: Winsock did not start");
  }
  ~Net() { WSACleanup(); }
};
#else
using Sock = int;
constexpr Sock kNoSock = -1;
using PollFd = pollfd;
int last_error() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
void close_sock(Sock s) { ::close(s); }
int poll_socks(PollFd* fds, size_t n, int ms) { return ::poll(fds, static_cast<nfds_t>(n), ms); }
void set_nonblocking(Sock s) { fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK); }
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;  // a client gone is an error here, not a SIGPIPE
#else
constexpr int kSendFlags = 0;
#endif
struct Net {};
#endif

std::string error_text(int e) {
#if defined(_WIN32)
  char buf[256] = {};
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, static_cast<DWORD>(e), 0, buf,
                 sizeof buf, nullptr);
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s + " (" + std::to_string(e) + ")";
#else
  return std::string(std::strerror(e)) + " (" + std::to_string(e) + ")";
#endif
}

void no_delay(Sock s) {
  int one = 1;
  setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
#if defined(SO_NOSIGPIPE)
  setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
}

// The listening socket on host:port.
Sock listen_on(const std::string& host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  addrinfo* found = nullptr;
  const std::string service = std::to_string(port);
  if (const int r = getaddrinfo(host.c_str(), service.c_str(), &hints, &found); r != 0) {
    throw Error("serve: no address " + host + ": " + gai_strerror(r));
  }
  std::string why = "no address";
  Sock s = kNoSock;
  for (addrinfo* a = found; a && s == kNoSock; a = a->ai_next) {
    s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (s == kNoSock) continue;
#if !defined(_WIN32)
    int one = 1;  // a restart binds again at once (Windows' SO_REUSEADDR would let another process share the port)
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
    if (bind(s, a->ai_addr, static_cast<int>(a->ai_addrlen)) != 0 || listen(s, SOMAXCONN) != 0) {
      why = error_text(last_error());
      close_sock(s);
      s = kNoSock;
    }
  }
  freeaddrinfo(found);
  if (s == kNoSock) throw Error("serve: cannot listen on " + host + ":" + std::to_string(port) + ": " + why);
  set_nonblocking(s);
  return s;
}

int port_of(Sock s) {
  sockaddr_storage a{};
  socklen_t n = sizeof a;
  getsockname(s, reinterpret_cast<sockaddr*>(&a), &n);
  if (a.ss_family == AF_INET6) return ntohs(reinterpret_cast<sockaddr_in6*>(&a)->sin6_port);
  return ntohs(reinterpret_cast<sockaddr_in*>(&a)->sin_port);
}

// Two connected sockets on the loopback: a byte written to the second wakes poll on the first (the network
// thread's), from the threads that answer and from stop().
std::pair<Sock, Sock> wake_pair() {
  const Sock l = listen_on("127.0.0.1", 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port_of(l)));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const Sock w = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (w == kNoSock || connect(w, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    close_sock(l);
    throw Error("serve: no loopback socket: " + error_text(last_error()));
  }
  Sock r = kNoSock;
  for (int tries = 0; r == kNoSock && tries < 1000; ++tries) {  // the listener is non-blocking
    r = accept(l, nullptr, nullptr);
    if (r == kNoSock) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  close_sock(l);
  if (r == kNoSock) {
    close_sock(w);
    throw Error("serve: no loopback socket");
  }
  set_nonblocking(r);
  set_nonblocking(w);  // a full buffer means a wake already waiting: the byte is not needed
  no_delay(w);
  return {r, w};
}

// ---------------------------------------------------------------- what a page's answer is

// The formats, and their media types.
const std::map<std::string, std::string, std::less<>>& formats() {
  static const std::map<std::string, std::string, std::less<>> f = {
      {"json", "application/json"},
      {"text", "text/plain; charset=utf-8"},
      {"hocr", "application/xhtml+xml; charset=utf-8"},
      {"alto", "application/xml; charset=utf-8"},
      {"page", "application/xml; charset=utf-8"},
      {"markdown", "text/markdown; charset=utf-8"},
      {"html", "text/html; charset=utf-8"},
      {"pdf", "application/pdf"},
  };
  return f;
}

// What the body's first bytes say it is (carbon::file_kind): an image decode_image reads, a PDF, or neither.
enum class Kind { kImage, kPdf, kOther };
Kind sniff(std::string_view b) {
  const FileKind k = file_kind(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(b.data()), b.size()));
  return k == FileKind::kPdf ? Kind::kPdf : k == FileKind::kUnknown ? Kind::kOther : Kind::kImage;
}

std::string json_string(std::string_view s) {
  std::string q = "\"";
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      q += '\\';
      q += ch;
    } else if (c < 0x20) {
      char e[8];
      std::snprintf(e, sizeof e, "\\u%04x", c);
      q += e;
    } else {
      q += ch;
    }
  }
  return q + "\"";
}

// A JSON value (RFC 8259), as a /parse body holds them: objects, arrays, strings, numbers, booleans and null.
struct Json {
  enum Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = kNull;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> fields;
  const Json* get(std::string_view key) const {
    for (const auto& [k, v] : fields) {
      if (k == key) return &v;
    }
    return nullptr;
  }
};

// Reads one JSON value, the whole text; throws http::BadRequest (400) where it is not JSON.
class JsonReader {
 public:
  explicit JsonReader(std::string_view t) : t_(t) {}
  Json document() {
    Json v = value(0);
    space();
    if (i_ != t_.size()) bad("text after the value");
    return v;
  }

 private:
  [[noreturn]] void bad(const std::string& why) const {
    throw http::BadRequest(400, "the body is not JSON: " + why + " at byte " + std::to_string(i_));
  }
  void space() {
    while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) ++i_;
  }
  bool word(std::string_view w) {
    if (t_.substr(i_, w.size()) != w) return false;
    i_ += w.size();
    return true;
  }
  static void put_utf8(std::string& out, uint32_t c) {
    if (c < 0x80) {
      out += static_cast<char>(c);
    } else if (c < 0x800) {
      out += static_cast<char>(0xC0 | (c >> 6));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      out += static_cast<char>(0xE0 | (c >> 12));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (c >> 18));
      out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    }
  }
  uint32_t hex4() {
    if (i_ + 4 > t_.size()) bad("a short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = t_[i_++];
      v = v * 16 + (c >= '0' && c <= '9'   ? c - '0'
                    : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                           : (bad("a bad \\u escape"), 0));
    }
    return v;
  }
  std::string string() {
    ++i_;  // the opening quote
    std::string out;
    for (;;) {
      const size_t run = i_;
      while (i_ < t_.size() && t_[i_] != '"' && t_[i_] != '\\' && static_cast<unsigned char>(t_[i_]) >= 0x20) ++i_;
      out.append(t_.substr(run, i_ - run));
      if (i_ >= t_.size()) bad("an unterminated string");
      const char c = t_[i_++];
      if (c == '"') return out;
      if (c != '\\') bad("a control character in a string");
      if (i_ >= t_.size()) bad("an unterminated escape");
      const char e = t_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t u = hex4();
          if (u >= 0xD800 && u <= 0xDBFF && t_.substr(i_, 2) == "\\u") {  // a surrogate pair
            i_ += 2;
            const uint32_t lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) bad("a lone surrogate");
            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
          }
          put_utf8(out, u);
          break;
        }
        default:
          bad("a bad escape");
      }
    }
  }
  Json value(int depth) {
    if (depth > 64) bad("nesting too deep");
    space();
    if (i_ >= t_.size()) bad("no value");
    Json v;
    const char c = t_[i_];
    if (c == '{') {
      v.kind = Json::kObject;
      ++i_;
      space();
      if (i_ < t_.size() && t_[i_] == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        space();
        if (i_ >= t_.size() || t_[i_] != '"') bad("an object's key is not a string");
        std::string key = string();
        space();
        if (i_ >= t_.size() || t_[i_] != ':') bad("no ':' after a key");
        ++i_;
        v.fields.emplace_back(std::move(key), value(depth + 1));
        space();
        if (i_ < t_.size() && t_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < t_.size() && t_[i_] == '}') {
          ++i_;
          return v;
        }
        bad("no ',' or '}' in an object");
      }
    }
    if (c == '[') {
      v.kind = Json::kArray;
      ++i_;
      space();
      if (i_ < t_.size() && t_[i_] == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.items.push_back(value(depth + 1));
        space();
        if (i_ < t_.size() && t_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < t_.size() && t_[i_] == ']') {
          ++i_;
          return v;
        }
        bad("no ',' or ']' in an array");
      }
    }
    if (c == '"') {
      v.kind = Json::kString;
      v.s = string();
      return v;
    }
    if (word("true") || word("false")) {
      v.kind = Json::kBool;
      v.b = t_[i_ - 1] == 'e' && t_[i_ - 2] == 'u';
      return v;
    }
    if (word("null")) return v;
    const std::string num(t_.substr(i_, std::min<size_t>(64, t_.size() - i_)));
    char* end = nullptr;
    v.n = std::strtod(num.c_str(), &end);
    if (end == num.c_str() || !(c == '-' || (c >= '0' && c <= '9'))) bad("no value");
    v.kind = Json::kNumber;
    i_ += static_cast<size_t>(end - num.c_str());
    return v;
  }
  std::string_view t_;
  size_t i_ = 0;
};

// Base64 (RFC 4648, its standard alphabet; whitespace skipped, the padding optional); throws http::BadRequest.
std::string from_base64(std::string_view s) {
  auto at = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  std::string out;
  out.reserve(s.size() / 4 * 3);
  uint32_t acc = 0;
  int bits = 0;
  bool ended = false;
  for (const char c : s) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (c == '=') {
      ended = true;
      continue;
    }
    const int v = at(c);
    if (v < 0 || ended) throw http::BadRequest(400, "the page is not base64");
    acc = (acc << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out += static_cast<char>((acc >> bits) & 0xFF);
    }
  }
  return out;
}

// The bytes of a model set's files (its .onnx, .bin and .json, by name), as one FNV-1a hash: what a /parse
// answer's "served" names as its revision, so a run records which weights answered.
std::string set_revision(const std::filesystem::path& set) {
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(set, ec)) {
    const std::string x = e.path().extension().string();
    if (e.is_regular_file() && (x == ".onnx" || x == ".bin" || x == ".json")) files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  uint64_t h = 1469598103934665603ull;
  std::vector<char> buf(size_t{1} << 20);
  for (const auto& f : files) {
    for (const char c : f.filename().string()) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    std::ifstream in(f, std::ios::binary);
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      for (std::streamsize k = 0; k < in.gcount(); ++k) {
        h = (h ^ static_cast<unsigned char>(buf[static_cast<size_t>(k)])) * 1099511628211ull;
      }
    }
  }
  char hex[32];
  std::snprintf(hex, sizeof hex, "fnv1a64:%016llx", static_cast<unsigned long long>(h));
  return hex;
}

using Clock = std::chrono::steady_clock;

}  // namespace

// ---------------------------------------------------------------- the server

struct Server::Impl {
  // A request that needs the engine, from the network thread to the document stream and back: an image (one page),
  // a PDF (each of its pages), or a /parse body holding either.
  struct Work {
    uint64_t conn = 0;
    std::shared_ptr<const std::string> body;  // the page's file: the image's or the PDF's bytes
    bool pdf = false;
    bool parse = false;  // /parse: its JSON answer, every page with its html and markdown, and what answers
    std::string format, name;
    bool text_only = false, html = false, keep_alive = true;
    double dpi = 200;  // a PDF's pages
    Clock::time_point t0;
    // its pages, once the PDF is counted: their results as they finish, and how many are left
    std::vector<OcrPage> pages;
    std::vector<std::shared_ptr<const GrayImage>> images;  // format=pdf of a PDF: each page as rendered
    size_t left = 0;
    std::filesystem::path file;  // a PDF's bytes on disk, for the helpers, until its pages are rendered
    size_t rendered = 0;
    std::exception_ptr error;    // the first page's that met one
    bool unreadable = false;
  };
  // A PDF page to render, by the feeders, between requests.
  struct Render {
    uint64_t work = 0;
    size_t page = 0;
  };
  // An answer, from the threads that write them to the network thread.
  struct Answer {
    uint64_t conn = 0;
    std::string bytes;
    bool close = false;
  };
  struct Conn {
    Sock fd = kNoSock;
    std::string in, out;  // read and not yet a whole request; to write
    size_t sent = 0;
    bool waiting = false;      // its request is being read: nothing more read from it until answered
    bool close_after = false;  // closed once `out` is written
    bool continued = false;    // 100 Continue written for the head at the start of `in`
    bool dead = false;         // its client gone, or its socket broken: closed this round
  };

  std::filesystem::path set;
  ServerOptions o;
  Net net;
  std::unique_ptr<DocumentStream> stream;
  std::string revision;  // the set's files' hash, for /parse's and /served's "served"
  int in_flight = 0;
  Sock listener = kNoSock, wake_r = kNoSock, wake_w = kNoSock;
  int bound_port = 0;
  std::atomic<bool> stopping{false};

  // the requests waiting for the engine, and the PDF pages to render
  std::mutex q_m;
  std::condition_variable q_cv;
  std::deque<Work> queue;
  std::deque<Render> renders;
  // The requests held, from when one is read until its answer goes out: waiting here, on a feeder (which holds
  // one while the stream is full), in the stream, or being written. Past the stream's pages in flight and
  // o.queue more, a request is answered 503 at once (route). Under q_m.
  size_t held = 0;
  // the requests given to the stream, by number; a page's tag is its request's number and its page
  static constexpr int kPageBits = 24;
  std::mutex given_m;
  std::condition_variable given_cv;
  std::unordered_map<uint64_t, Work> given;
  uint64_t next_work = 0;
  size_t pages_given = 0, pages_taken = 0;
  int feeders_left = 0;  // the threads still giving pages
#if defined(CARBON_WITH_PDFIUM)
  std::mutex pdf_m;
  std::unique_ptr<PdfPool> pdf_pool;  // made at the first PDF
#endif
  // the answers for the network thread
  std::mutex out_m;
  std::vector<Answer> outbox;

  Impl(const std::filesystem::path& s, ServerOptions opts) : set(s), o(std::move(opts)) {
    DocumentOptions d;
    d.engine = o.engine;
    d.cuda = o.cuda;
    d.pipeline = o.pipeline;
    d.text_layer = o.text_layer;
    d.pictures = o.pictures;
    d.layer_lines = o.layer_lines;
    stream = std::make_unique<DocumentStream>(set, d);
    in_flight = stream->in_flight();
    revision = set_revision(set);
    listener = listen_on(o.host, o.port);
    bound_port = port_of(listener);
    try {
      std::tie(wake_r, wake_w) = wake_pair();
    } catch (...) {
      close_sock(listener);
      throw;
    }
  }
  ~Impl() {
    for (Sock s : {listener, wake_r, wake_w}) {
      if (s != kNoSock) close_sock(s);
    }
  }

  void log(const std::string& line) const {
    if (o.log) o.log(line);
  }

  void wake() const {
    const char b = 1;
    send(wake_w, &b, 1, kSendFlags);
  }

  void post(Answer a) {
    {
      std::lock_guard lk(out_m);
      outbox.push_back(std::move(a));
    }
    wake();
  }

  // ------------------------------------------------ the engine's side

  // The GPU pipeline's runs so far: how full its batches are, so a load test can see whether its pages in
  // flight fill them. Empty for the CPU engine.
  std::string pipeline_stats() const {
    const Pipeline* pipe = stream->pipeline();
    if (!pipe) return "";
    const PipelineStats st = pipe->stats();
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  ", \"pipeline\": {\"pages\": %lld, \"lines\": %lld, \"detect_runs\": %lld, "
                  "\"recognize_runs\": %lld, \"pages_per_detect_run\": %.2f, \"lines_per_recognize_run\": %.1f}",
                  static_cast<long long>(st.pages), static_cast<long long>(st.lines),
                  static_cast<long long>(st.detect_runs), static_cast<long long>(st.recognize_runs),
                  st.pages_per_detect_run(), st.lines_per_recognize_run());
    return buf;
  }

  std::string served() const {
    return "{\"weights\": " + json_string(set.filename().empty() ? set.parent_path().filename().generic_string()
                                                                 : set.filename().generic_string()) +
           ", \"revision\": " + json_string(revision) + "}";
  }

  std::string info() const {
    return "{\"version\": " + json_string(kVersion) + ", \"set\": " + json_string(set.generic_string()) +
           ", \"device\": " + (o.cuda ? "\"cuda\"" : "\"cpu\"") + ", \"precision\": " +
           json_string(precision_name(o.engine.precision)) + ", \"workers\": " + std::to_string(stream->workers()) +
           ", \"in_flight\": " + std::to_string(in_flight) + ", \"queue\": " + std::to_string(o.queue) +
           ", \"pdf\": " + (pdf_supported() ? "true" : "false") + ", \"served\": " + served() + pipeline_stats() +
           "}\n";
  }

  // A page's JSON object as carbon-ocr ocr writes it: OcrPage::json, its page in a PDF first where it has one, and
  // its html body (to_html_body) and markdown where asked.
  static std::string page_json(const OcrPage& page, int number, bool html, bool markdown) {
    std::string s = page.json();
    if (html) s.insert(s.size() - 1, ", \"html\": " + json_string(to_html_body(page)));
    if (markdown) s.insert(s.size() - 1, ", \"markdown\": " + json_string(to_markdown(std::span(&page, 1))));
    if (number > 0) s = "{\"page\": " + std::to_string(number) + ", " + s.substr(1);
    return s;
  }

  // A request's answer, in its format, as carbon-ocr ocr writes its pages: an image's page alone (json: the
  // page's object), a PDF's pages together (json: {"pages": [...]}, each with its page number), /parse's JSON.
  void answer(Work& w) {
    int status = 200;
    std::string type, body;
    std::vector<std::pair<std::string, std::string>> extra;
    if (w.error) {
      try {
        std::rethrow_exception(w.error);
      } catch (const http::BadRequest& e) {
        status = e.status;
        body = std::string(e.what()) + "\n";
      } catch (const std::exception& e) {
        status = w.unreadable ? 422 : 500;
        body = std::string(e.what()) + "\n";
      } catch (...) {
        status = 500;
        body = "unknown error\n";
      }
      type = "text/plain; charset=utf-8";
    } else {
      try {
        const std::span<const OcrPage> pages(w.pages);
        std::vector<std::string> names;
        for (size_t i = 0; i < w.pages.size(); ++i) {
          names.push_back(w.pdf ? w.name + "#page=" + std::to_string(i + 1) : w.name);
        }
        type = w.parse ? "application/json" : formats().find(w.format)->second;
        if (w.parse) {  // every page with its html and markdown, and what answered
          body = "{\"pages\": [";
          for (size_t i = 0; i < w.pages.size(); ++i) {
            body += (i ? ", " : "") + page_json(w.pages[i], w.pdf ? static_cast<int>(i) + 1 : 0, true, true);
          }
          body += "], \"served\": " + served() + ", \"vendor\": {\"carbon-ocr\": " + json_string(kVersion) + "}}\n";
        } else if (w.format == "json") {  // html=1: the page's html body inside it, as carbon_ocr's html=True
          if (!w.pdf) {
            body = page_json(w.pages[0], 0, w.html, false) + "\n";
          } else {
            body = "{\"pages\": [";
            for (size_t i = 0; i < w.pages.size(); ++i) {
              body += (i ? ", " : "") + page_json(w.pages[i], static_cast<int>(i) + 1, w.html, false);
            }
            body += "]}\n";
          }
        } else if (w.format == "text") {
          for (size_t i = 0; i < w.pages.size(); ++i) body += (i ? "\f\n" : "") + w.pages[i].text();
        } else if (w.format == "hocr") {
          body = to_hocr(pages, names);
        } else if (w.format == "alto") {
          body = to_alto(pages, names);
        } else if (w.format == "page") {
          if (w.pages.size() != 1) {
            throw http::BadRequest(400, "format page is one page's PAGE XML; this PDF has " +
                                            std::to_string(w.pages.size()));
          }
          body = to_page_xml(w.pages[0], w.name);
        } else if (w.format == "markdown") {
          body = to_markdown(pages);
        } else if (w.format == "html") {
          body = to_html(pages);
        } else {  // pdf: the pages' images under their words
          std::vector<GrayImage> decoded;
          std::vector<GrayView> views;
          std::vector<double> dpis;
          if (w.pdf) {
            for (const auto& img : w.images) views.push_back(img->view());
            dpis.assign(w.pages.size(), w.dpi);
          } else {
            decoded.push_back(decode_image(
                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(w.body->data()), w.body->size())));
            views.push_back(decoded[0].view());
            dpis.push_back(o.image_dpi);
          }
          body = to_pdf(pages, views, dpis, w.text_only);
        }
        double ms = 0;
        size_t lines = 0;
        for (const OcrPage& p : w.pages) {
          ms += p.ms.total;
          lines += p.lines.size();
        }
        char t[32];
        std::snprintf(t, sizeof t, "%.1f", ms);
        extra.emplace_back("X-Carbon-Ms", t);
        extra.emplace_back("X-Carbon-Lines", std::to_string(lines));
        if (w.pdf) extra.emplace_back("X-Carbon-Pages", std::to_string(w.pages.size()));
      } catch (const http::BadRequest& e) {
        status = e.status;
        type = "text/plain; charset=utf-8";
        body = std::string(e.what()) + "\n";
        extra.clear();
      } catch (const std::exception& e) {
        status = 500;
        type = "text/plain; charset=utf-8";
        body = std::string(e.what()) + "\n";
        extra.clear();
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - w.t0).count();
    char line[200];
    std::snprintf(line, sizeof line, "POST %s %d %s %zu bytes in, %zu page%s, %zu out, %.1f ms",
                  w.parse ? "/parse" : "/ocr", status, w.parse ? "json" : w.format.c_str(),
                  w.body ? w.body->size() : size_t{0}, w.pages.size(), w.pages.size() == 1 ? "" : "s", body.size(),
                  ms);
    log(line);
    {  // its place given up as its answer goes out
      std::lock_guard lk(q_m);
      --held;
    }
    post({w.conn, http::response(status, type, body, w.keep_alive, extra), !w.keep_alive});
  }

  // A request's pages all in (or its first error): its answer, by the writers.
  void hand_on(Work w) {
    if (!w.file.empty()) {  // a PDF failed before every page was rendered
      std::error_code ec;
      std::filesystem::remove(w.file, ec);
    }
    hand_to_writers(std::move(w));
  }

  // The next request waiting or PDF page to render (the pages first: a PDF's pages keep the stream fed), or
  // nullopt once stopping.
  std::optional<std::variant<Work, Render>> take_waiting() {
    std::unique_lock lk(q_m);
    q_cv.wait(lk, [&] { return stopping || !queue.empty() || !renders.empty(); });
    if (stopping) return std::nullopt;
    if (!renders.empty()) {
      Render r = renders.front();
      renders.pop_front();
      return r;
    }
    Work w = std::move(queue.front());
    queue.pop_front();
    return w;
  }

  void feeder_done() {
    {
      std::lock_guard lk(given_m);
      --feeders_left;
    }
    given_cv.notify_all();
  }

  // A request's page `p` given to the stream, counted first (the collector waits for pages given).
  template <class Give>
  void give_page(uint64_t id, size_t p, Give&& give) {
    {
      std::lock_guard lk(given_m);
      ++pages_given;
    }
    given_cv.notify_all();
    give((id << kPageBits) | p);
  }

  // The feeders: each request's pages to the stream (an image's bytes, decoded by the worker that takes the
  // page, on a GPU here; a PDF's pages counted, then rendered by the feeders one at a time).
  void feed() {
    while (auto item = take_waiting()) {
      if (Render* r = std::get_if<Render>(&*item)) {
        render(*r);
        continue;
      }
      Work& w = std::get<Work>(*item);
      uint64_t id = 0;
      if (!w.pdf) {
        w.pages.resize(1);
        w.left = 1;
        const std::shared_ptr<const std::string> body = w.body;
        {
          std::lock_guard lk(given_m);
          id = next_work++;
          given.emplace(id, std::move(w));
        }
        give_page(id, 0, [&](uint64_t tag) { stream->encoded(body, tag); });
        continue;
      }
      start_pdf(std::move(w));
    }
    feeder_done();
  }

  // A PDF's bytes on disk for the helpers, its pages counted, and a render task for each.
  void start_pdf(Work w) {
#if defined(CARBON_WITH_PDFIUM)
    try {
      static std::atomic<uint64_t> files{0};
      w.file = std::filesystem::temp_directory_path() /
               ("carbon-serve-" + std::to_string(bound_port) + "-" + std::to_string(files++) + ".pdf");
      {
        std::ofstream f(w.file, std::ios::binary);
        f.write(w.body->data(), static_cast<std::streamsize>(w.body->size()));
        if (!f) throw Error("serve: cannot write the PDF to " + w.file.string());
      }
      int n = 0;
      {
        std::lock_guard lk(pdf_m);
        if (!pdf_pool) pdf_pool = std::make_unique<PdfPool>(o.engine.pool.workers, o.pdf_helper);
      }
      try {
        n = pdf_pool->pages(w.file);
      } catch (...) {
        w.unreadable = true;
        throw;
      }
      if (n <= 0) throw http::BadRequest(422, "the PDF has no pages");
      if (n >= (1 << kPageBits)) throw http::BadRequest(413, "the PDF has " + std::to_string(n) + " pages");
      w.pages.resize(static_cast<size_t>(n));
      if (w.format == "pdf" && !w.parse) w.images.resize(static_cast<size_t>(n));
      w.left = static_cast<size_t>(n);
      uint64_t id = 0;
      {
        std::lock_guard lk(given_m);
        id = next_work++;
        given.emplace(id, std::move(w));
      }
      {
        std::lock_guard lk(q_m);
        for (int p = 0; p < n; ++p) renders.push_back({id, static_cast<size_t>(p)});
      }
      q_cv.notify_all();
    } catch (...) {
      w.error = std::current_exception();
      hand_on(std::move(w));
    }
#else
    w.error = std::make_exception_ptr(http::BadRequest(415, "this build reads no PDFs"));
    hand_on(std::move(w));
#endif
  }

  // One page of a PDF rendered and given to the stream; its error, its request's.
  void render(const Render& r) {
#if defined(CARBON_WITH_PDFIUM)
    std::filesystem::path file;
    double dpi = 200;
    bool keep = false;
    {
      std::lock_guard lk(given_m);
      auto it = given.find(r.work);
      if (it == given.end()) return;
      file = it->second.file;
      dpi = it->second.dpi;
      keep = !it->second.images.empty();
    }
    bool last = false;
    try {
      PdfPage pg = pdf_pool->page(file, static_cast<int>(r.page), dpi / 72.0, stream->pdf_want());
      // format=pdf: each page's image as shown (one read turned upright, turned back; one turned that goes to
      // OCR rendered again as shown, below)
      const auto keep_image = [&](GrayImage image) {
        std::lock_guard lk(given_m);
        auto it = given.find(r.work);
        if (it != given.end()) it->second.images[r.page] = std::make_shared<const GrayImage>(std::move(image));
      };
      if (keep) keep_image(pg.turn ? rotate_quarter(pg.image.view(), 4 - pg.turn) : GrayImage(pg.image));
      const double scale = pg.scale;
      give_page(r.work, r.page, [&](uint64_t tag) {
        stream->pdf_page(std::move(pg), tag, [&] {
          GrayImage shown = pdf_pool->page(file, static_cast<int>(r.page), scale, kPdfImage).image;
          if (keep) keep_image(shown);
          return shown;
        });
      });
      {  // rendered once the stream has it: its file no more read for it
        std::lock_guard lk(given_m);
        auto it = given.find(r.work);
        if (it != given.end()) last = ++it->second.rendered == it->second.pages.size();
      }
    } catch (...) {  // not rendered: the page is taken as failed with it
      Done d;
      d.tag = (r.work << kPageBits) | r.page;
      d.error = std::current_exception();
      {
        std::lock_guard lk(given_m);
        ++pages_given;
        ++pages_taken;
        auto it = given.find(r.work);
        if (it != given.end()) last = ++it->second.rendered == it->second.pages.size();
      }
      page_done(d);
    }
    if (last) {
      std::lock_guard lk(given_m);
      auto it = given.find(r.work);
      if (it != given.end()) {
        std::error_code ec;
        std::filesystem::remove(it->second.file, ec);  // every page rendered: its file no more needed
        it->second.file.clear();
      }
    }
#else
    (void)r;
#endif
  }

  // ------------------------------------------------ the writers: each finished request's answer written in
  // its format (a PDF's deflate, an hOCR's XML) on a thread of their own, so the collect thread only takes pages.
  using Done = DocumentStream::Done;
  std::mutex w_m;
  std::condition_variable w_cv;
  std::deque<Work> to_write;
  bool writing_done = false;

  void hand_to_writers(Work w) {
    {
      std::lock_guard lk(w_m);
      to_write.push_back(std::move(w));
    }
    w_cv.notify_one();
  }

  void write_answers() {
    for (;;) {
      Work w;
      {
        std::unique_lock lk(w_m);
        w_cv.wait(lk, [&] { return writing_done || !to_write.empty(); });
        if (to_write.empty()) return;  // collect is done and every answer written
        w = std::move(to_write.front());
        to_write.pop_front();
      }
      answer(w);
    }
  }

  // collect, then the writers told that no more pages come.
  void collect_then_done() {
    collect();
    {
      std::lock_guard lk(w_m);
      writing_done = true;
    }
    w_cv.notify_all();
  }

  // A page out: into its request, which goes to the writers once its last page is in.
  void page_done(Done& d) {
    const uint64_t id = d.tag >> kPageBits;
    const size_t p = static_cast<size_t>(d.tag & ((uint64_t{1} << kPageBits) - 1));
    std::optional<Work> ready;
    {
      std::lock_guard lk(given_m);
      auto it = given.find(id);
      if (it == given.end()) return;
      Work& w = it->second;
      if (d.error && !w.error) {
        w.error = d.error;
        w.unreadable = d.unreadable;
      }
      if (!d.error && p < w.pages.size()) w.pages[p] = std::move(d.page);
      if (--w.left == 0) {
        ready = std::move(w);
        given.erase(it);
      }
    }
    if (ready) hand_on(std::move(*ready));
  }

  // The pages as they finish (a slow page holds back no other), each into its request.
  void collect() {
    for (;;) {
      {
        std::unique_lock lk(given_m);
        given_cv.wait(lk, [&] { return pages_given > pages_taken || feeders_left == 0; });
        if (pages_given == pages_taken) return;  // every feeder gone, every page answered
      }
      std::optional<Done> d = stream->next();
      if (!d) return;
      {
        std::lock_guard lk(given_m);
        ++pages_taken;
      }
      page_done(*d);
    }
  }

  // ------------------------------------------------ the network's side

  std::unordered_map<uint64_t, Conn> conns;
  uint64_t next_conn = 1;

  void reply(Conn& c, int status, std::string_view type, std::string_view body, bool keep_alive,
             const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    c.out += http::response(status, type, body, keep_alive, extra);
    if (!keep_alive) c.close_after = true;
  }

  void error_reply(Conn& c, int status, const std::string& why, bool keep_alive,
                   const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    reply(c, status, "text/plain; charset=utf-8", why + "\n", keep_alive, extra);
    log(std::to_string(status) + " " + why);
  }

  // Answers what the connection's buffer holds, a request at a time, until one needs the engine or more bytes.
  void handle(uint64_t id, Conn& c) {
    while (!c.waiting && !c.close_after && !c.in.empty()) {
      std::optional<std::pair<size_t, http::Request>> parsed;
      try {
        parsed = http::parse_head(c.in);
      } catch (const http::BadRequest& e) {
        error_reply(c, e.status, e.what(), false);
        return;
      }
      if (!parsed) return;
      auto& [head, req] = *parsed;
      if (req.content_length > o.max_body) {
        error_reply(c, 413, "the body is " + std::to_string(req.content_length) + " bytes, past the server's " +
                                std::to_string(o.max_body), false);
        return;
      }
      if (c.in.size() < head + req.content_length) {  // the body still coming
        if (req.expect_continue && !c.continued) {
          c.out += "HTTP/1.1 100 Continue\r\n\r\n";
          c.continued = true;
        }
        return;
      }
      std::string body = c.in.substr(head, req.content_length);
      c.in.erase(0, head + req.content_length);
      c.continued = false;
      route(id, c, req, std::move(body));
    }
  }

  void route(uint64_t id, Conn& c, const http::Request& req, std::string body) {
    const bool keep = req.keep_alive;
    if (req.path == "/health" || req.path == "/info" || req.path == "/served") {
      const Pipeline* pipe = stream->pipeline();
      if (req.method != "GET") {
        error_reply(c, 405, req.path + " takes GET", keep, {{"Allow", "GET"}});
      } else if (req.path == "/health") {
        if (pipe && pipe->stopped()) {  // its device broken: nothing more will be read
          error_reply(c, 503, "the pipeline stopped: its device can run nothing more", keep);
        } else {
          reply(c, 200, "text/plain; charset=utf-8", "ok\n", keep);
        }
      } else if (req.path == "/served") {
        reply(c, 200, "application/json", served() + "\n", keep);
      } else {
        reply(c, 200, "application/json", info(), keep);
      }
      return;
    }
    if (req.path != "/ocr" && req.path != "/parse") {
      error_reply(c, 404, "no " + req.path + ": POST /ocr, POST /parse, GET /health, GET /info, GET /served", keep);
      return;
    }
    if (req.method != "POST") {
      error_reply(c, 405,
                  req.path == "/ocr" ? "/ocr takes POST, the page's image or PDF as its body"
                                     : "/parse takes POST, a JSON object of the page and its settings",
                  keep, {{"Allow", "POST"}});
      return;
    }
    if (!req.has_length) {
      error_reply(c, 411, req.path + " needs a Content-Length", false);
      return;
    }
    Work w;
    if (req.path == "/parse") {  // {"page": base64 file, "suffix": ".pdf" or the image's, "settings": {"dpi": N}}
      w.parse = true;
      w.format = "json";
      w.name = "page";
      try {
        const Json doc = JsonReader(body).document();
        const Json* page = doc.get("page");
        if (doc.kind != Json::kObject || !page || page->kind != Json::kString) {
          throw http::BadRequest(400, "/parse takes {\"page\": the file in base64, \"suffix\", \"settings\"}");
        }
        if (const Json* s = doc.get("settings"); s && s->kind != Json::kNull) {
          if (s->kind != Json::kObject) throw http::BadRequest(400, "settings is an object");
          for (const auto& [k, v] : s->fields) {
            if (k != "dpi") throw http::BadRequest(400, "setting " + k + ": the settings are dpi");
            if (v.kind != Json::kNumber || !(v.n >= 36 && v.n <= 1200)) {
              throw http::BadRequest(400, "dpi takes a number from 36 to 1200");
            }
            w.dpi = v.n;
          }
        }
        body = from_base64(page->s);
      } catch (const http::BadRequest& e) {
        error_reply(c, e.status, e.what(), keep);
        return;
      }
    } else {
      w.format = std::string(req.param("format").value_or("json"));
      if (!formats().count(w.format)) {
        error_reply(c, 400, "format " + w.format + ": json, text, hocr, alto, page, markdown, html or pdf", keep);
        return;
      }
      w.name = std::string(req.param("name").value_or("page"));
      const std::string_view to = req.param("text_only").value_or("0");
      w.text_only = to == "1" || to == "true";
      const std::string_view html = req.param("html").value_or("0");
      w.html = html == "1" || html == "true";
      if (const auto dpi = req.param("dpi")) {
        const std::string v(*dpi);
        char* end = nullptr;
        w.dpi = std::strtod(v.c_str(), &end);
        if (end == v.c_str() || *end || !(w.dpi >= 36 && w.dpi <= 1200)) {
          error_reply(c, 400, "dpi takes a number from 36 to 1200", keep);
          return;
        }
      }
    }
    const Kind kind = sniff(body);
    if (body.empty() || kind == Kind::kOther || (kind == Kind::kPdf && !pdf_supported())) {
      error_reply(c, 415,
                  kind == Kind::kPdf ? "a PDF, which this build does not read (CMake option CARBON_WITH_PDFIUM)"
                                     : "the body is no page the server reads: PNG, JPEG, BMP, PNM, GIF, WebP or PDF",
                  keep);
      return;
    }
    w.pdf = kind == Kind::kPdf;
    w.conn = id;
    w.keep_alive = keep;
    w.t0 = Clock::now();
    w.body = std::make_shared<const std::string>(std::move(body));
    {  // every request held counts, not only those still here: each feeder holds one while the stream is full
      std::lock_guard lk(q_m);
      const size_t limit = static_cast<size_t>(in_flight) + static_cast<size_t>(std::max(0, o.queue));
      if (held >= limit) {
        error_reply(c, 503,
                    "busy: " + std::to_string(held) + " requests held, " + std::to_string(in_flight) +
                        " pages in flight and " + std::to_string(o.queue) + " waiting",
                    keep, {{"Retry-After", "1"}});
        return;
      }
      ++held;
      queue.push_back(std::move(w));
    }
    q_cv.notify_one();
    c.waiting = true;
  }

  void read_from(uint64_t id, Conn& c, bool& closed) {
    char buf[65536];
    for (;;) {
      const auto n = recv(c.fd, buf, static_cast<int>(sizeof buf), 0);
      if (n > 0) {
        c.in.append(buf, static_cast<size_t>(n));
        if (c.in.size() > o.max_body + 65536) break;  // handle() answers 413 or 431 for it
        continue;
      }
      if (n == 0 || !would_block(last_error())) closed = true;  // the client gone, or its connection broken
      break;
    }
    if (!closed) handle(id, c);
  }

  void write_to(uint64_t id, Conn& c, bool& closed) {
    while (c.sent < c.out.size()) {
      const auto n = send(c.fd, c.out.data() + c.sent, static_cast<int>(c.out.size() - c.sent), kSendFlags);
      if (n > 0) {
        c.sent += static_cast<size_t>(n);
        continue;
      }
      if (!would_block(last_error())) closed = true;
      return;
    }
    c.out.clear();
    c.sent = 0;
    if (c.close_after) {
      closed = true;
      return;
    }
    handle(id, c);  // a request it sent while this one was being read
  }

  void run() {
    std::vector<std::thread> threads;
    // On the GPU, PNG decoding (about 10 ms a page a core) is the CPU's share of a page: half the physical cores
    // by default, at least 4, where 4 alone capped a server near 370 pages a second. On the
    // CPU an image's bytes are only handed to the engine, whose workers decode them; a PDF's pages are rendered by
    // the feeders (a helper process each at a time): half the physical cores, at least 2.
    const int cores = cpu_topology().default_workers();
    const int feeders = o.decoders > 0 ? o.decoders : o.cuda ? std::max(4, cores / 2) : std::max(2, cores / 2);
    feeders_left = feeders;
    for (int i = 0; i < feeders; ++i) threads.emplace_back([this] { feed(); });
    threads.emplace_back([this] { collect_then_done(); });
    const int writers = o.writers > 0 ? o.writers : (o.cuda ? std::max(2, feeders / 2) : 2);
    for (int i = 0; i < writers; ++i) threads.emplace_back([this] { write_answers(); });

    std::vector<PollFd> fds;
    std::vector<uint64_t> ids;
    while (!stopping) {
      fds.clear();
      ids.clear();
      fds.push_back({listener, POLLIN, 0});
      fds.push_back({wake_r, POLLIN, 0});
      for (auto& [id, c] : conns) {
        short ev = 0;
        if (!c.waiting && !c.close_after) ev |= POLLIN;
        if (c.sent < c.out.size()) ev |= POLLOUT;
        fds.push_back({c.fd, ev, 0});
        ids.push_back(id);
      }
      if (poll_socks(fds.data(), fds.size(), 1000) < 0) {
        if (would_block(last_error())) continue;
        log("serve: poll failed: " + error_text(last_error()));
        break;
      }
      if (fds[1].revents) {  // answers ready, or stop()
        char b[256];
        while (recv(wake_r, b, sizeof b, 0) > 0) {
        }
        std::vector<Answer> ready;
        {
          std::lock_guard lk(out_m);
          ready.swap(outbox);
        }
        for (Answer& a : ready) {
          auto it = conns.find(a.conn);
          if (it == conns.end()) continue;  // its client left before its answer
          it->second.out += a.bytes;
          it->second.waiting = false;
          if (a.close) it->second.close_after = true;
        }
      }
      for (size_t k = 0; k < ids.size(); ++k) {  // what came in
        const short re = fds[k + 2].revents;
        Conn& c = conns.at(ids[k]);
        if (re & POLLIN) read_from(ids[k], c, c.dead);
        else if (re & (POLLERR | POLLHUP | POLLNVAL)) c.dead = true;
      }
      for (auto& [id, c] : conns) {  // what is to go out: replies made just now, and the answers that came
        if (!c.dead && c.sent < c.out.size()) write_to(id, c, c.dead);
      }
      for (auto it = conns.begin(); it != conns.end();) {
        Conn& c = it->second;
        if (c.dead || (c.close_after && c.sent >= c.out.size())) {
          close_sock(c.fd);
          it = conns.erase(it);
        } else {
          ++it;
        }
      }
      if (fds[0].revents & POLLIN) {
        for (;;) {
          const Sock s = accept(listener, nullptr, nullptr);
          if (s == kNoSock) break;
          set_nonblocking(s);
          no_delay(s);
          Conn c;
          c.fd = s;
          conns.emplace(next_conn++, std::move(c));
        }
      }
    }
    stopping = true;
    q_cv.notify_all();
    {
      std::lock_guard lk(q_m);
      queue.clear();
      renders.clear();
    }
    for (std::thread& t : threads) t.join();
    for (auto& [id, c] : conns) close_sock(c.fd);
    conns.clear();
    std::lock_guard lk(given_m);  // the PDFs whose pages were left unrendered: their files
    for (auto& [n, w] : given) {
      std::error_code ec;
      if (!w.file.empty()) std::filesystem::remove(w.file, ec);
    }
  }
};

Server::Server(const std::filesystem::path& set, ServerOptions options)
    : impl_(std::make_unique<Impl>(set, std::move(options))) {}

Server::~Server() = default;

int Server::port() const { return impl_->bound_port; }

void Server::run() { impl_->run(); }

void Server::stop() {
  impl_->stopping = true;
  impl_->wake();
}

}  // namespace carbon
