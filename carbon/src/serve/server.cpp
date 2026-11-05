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
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "carbon/error.h"
#include "carbon/http.h"
#include "carbon/image.h"
#include "carbon/output.h"
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

// What the body's first bytes say it is: an image decode_image reads, a PDF, or neither.
enum class Kind { kImage, kPdf, kOther };
Kind sniff(std::string_view b) {
  auto starts = [&](std::string_view m) { return b.substr(0, m.size()) == m; };
  if (starts("\x89PNG\r\n\x1a\n") || starts("\xff\xd8\xff") || starts("BM")) return Kind::kImage;
  if (b.size() >= 2 && b[0] == 'P' && b[1] >= '1' && b[1] <= '6') return Kind::kImage;  // PNM
  if (starts("%PDF-")) return Kind::kPdf;
  return Kind::kOther;
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

using Clock = std::chrono::steady_clock;

}  // namespace

// ---------------------------------------------------------------- the server

struct Server::Impl {
  // A page asked for, from the network thread to the engine and back.
  struct Work {
    uint64_t conn = 0;
    std::shared_ptr<const std::string> body;
    std::string format, name;
    bool text_only = false, html = false, keep_alive = true;
    Clock::time_point t0;
    std::shared_ptr<GrayImage> image;  // on a GPU with format=pdf: the decoded page, for the PDF's image
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
  std::unique_ptr<Engine> engine;
  std::unique_ptr<Pipeline> pipe;
  int in_flight = 0;
  Sock listener = kNoSock, wake_r = kNoSock, wake_w = kNoSock;
  int bound_port = 0;
  std::atomic<bool> stopping{false};

  // the requests waiting for the engine
  std::mutex q_m;
  std::condition_variable q_cv;
  std::deque<Work> queue;
  // the pages given to the engine, by number
  std::mutex given_m;
  std::condition_variable given_cv;
  std::unordered_map<size_t, Work> given;
  size_t next_number = 0, submitted = 0, taken = 0;
  int feeders_left = 0;  // the threads still giving pages
  std::mutex submit_m;   // on a GPU: a page's number and its place in the pipeline, together
  // the answers for the network thread
  std::mutex out_m;
  std::vector<Answer> outbox;

  Impl(const std::filesystem::path& s, ServerOptions opts) : set(s), o(std::move(opts)) {
    if (o.cuda) {
      PipelineOptions po = o.pipeline;
      po.engine = o.engine;
      po.device = Device::kCuda;
      pipe = std::make_unique<Pipeline>(set, po);
      in_flight = po.in_flight;
    } else {
      engine = std::make_unique<Engine>(set, o.engine);
      in_flight = o.engine.in_flight > 0 ? o.engine.in_flight : 2 * engine->workers();
    }
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

  std::string info() const {
    const int workers = engine ? engine->workers() : 0;
    return "{\"version\": " + json_string(kVersion) + ", \"set\": " + json_string(set.generic_string()) +
           ", \"device\": " + (o.cuda ? "\"cuda\"" : "\"cpu\"") + ", \"precision\": " +
           json_string(precision_name(o.engine.precision)) + ", \"workers\": " + std::to_string(workers) +
           ", \"in_flight\": " + std::to_string(in_flight) + ", \"queue\": " + std::to_string(o.queue) +
           pipeline_stats() + "}\n";
  }

  // A page's answer, in its format, as carbon-ocr ocr writes the page.
  void answer(Work& w, OcrPage* page, std::exception_ptr error, bool unreadable) {
    int status = 200;
    std::string type, body;
    std::vector<std::pair<std::string, std::string>> extra;
    if (error) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception& e) {
        body = std::string(e.what()) + "\n";
      } catch (...) {
        body = "unknown error\n";
      }
      status = unreadable ? 422 : 500;
      type = "text/plain; charset=utf-8";
    } else {
      try {
        const std::span<const OcrPage> pages(page, 1);
        const std::vector<std::string> names = {w.name};
        type = formats().find(w.format)->second;
        if (w.format == "json") {  // html=1: the page's html body inside it, as carbon_ocr's html=True gives it
          body = page->json();
          if (w.html) body.insert(body.size() - 1, ", \"html\": " + json_string(to_html_body(*page)));
          body += "\n";
        } else if (w.format == "text") {
          body = page->text();
        } else if (w.format == "hocr") {
          body = to_hocr(pages, names);
        } else if (w.format == "alto") {
          body = to_alto(pages, names);
        } else if (w.format == "page") {
          body = to_page_xml(*page, w.name);
        } else if (w.format == "markdown") {
          body = to_markdown(pages);
        } else if (w.format == "html") {
          body = to_html(pages);
        } else {  // pdf: the page's image under its words
          const GrayImage img = w.image ? std::move(*w.image)
                                        : decode_image(std::span<const uint8_t>(
                                              reinterpret_cast<const uint8_t*>(w.body->data()), w.body->size()));
          const GrayView view = img.view();
          body = to_pdf(pages, std::span<const GrayView>(&view, 1), o.image_dpi, w.text_only);
        }
        char ms[32];
        std::snprintf(ms, sizeof ms, "%.1f", page->ms.total);
        extra.emplace_back("X-Carbon-Ms", ms);
        extra.emplace_back("X-Carbon-Lines", std::to_string(page->lines.size()));
      } catch (const std::exception& e) {
        status = 500;
        type = "text/plain; charset=utf-8";
        body = std::string(e.what()) + "\n";
        extra.clear();
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - w.t0).count();
    char line[160];
    std::snprintf(line, sizeof line, "POST /ocr %d %s %zu bytes in, %zu out, %.1f ms", status, w.format.c_str(),
                  w.body ? w.body->size() : size_t{0}, body.size(), ms);
    log(line);
    post({w.conn, http::response(status, type, body, w.keep_alive, extra), !w.keep_alive});
  }

  // The next request waiting, or nullopt once stopping.
  std::optional<Work> take_waiting() {
    std::unique_lock lk(q_m);
    q_cv.wait(lk, [&] { return stopping || !queue.empty(); });
    if (stopping) return std::nullopt;
    Work w = std::move(queue.front());
    queue.pop_front();
    return w;
  }

  void given_one() {
    {
      std::lock_guard lk(given_m);
      ++submitted;
    }
    given_cv.notify_all();
  }

  void feeder_done() {
    {
      std::lock_guard lk(given_m);
      --feeders_left;
    }
    given_cv.notify_all();
  }

  // The CPU: each page to the engine's stream as its bytes, decoded by the worker that takes it.
  void feed_engine() {
    while (auto w = take_waiting()) {
      const size_t n = next_number++;  // this thread alone gives pages, so they are numbered as given
      const std::shared_ptr<const std::string> body = w->body;
      {
        std::lock_guard lk(given_m);
        given.emplace(n, std::move(*w));
      }
      try {
        const size_t got = engine->submit_encoded(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(body->data()), body->size()), body);
        if (got != n) throw Error("serve: the engine numbered a page " + std::to_string(got) + ", not " +
                                  std::to_string(n));
      } catch (...) {
        Work lost;
        {
          std::lock_guard lk(given_m);
          lost = std::move(given.at(n));
          given.erase(n);
        }
        --next_number;
        answer(lost, nullptr, std::current_exception(), false);
        continue;
      }
      given_one();
    }
    feeder_done();
  }

  // On a GPU: decoded on these threads, then given to the pipeline, a page's number and its place in it
  // taken together.
  void feed_pipeline() {
    while (auto w = take_waiting()) {
      GrayImage img;
      try {
        img = decode_image(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(w->body->data()), w->body->size()));
      } catch (...) {
        answer(*w, nullptr, std::current_exception(), true);
        continue;
      }
      if (w->format == "pdf") w->image = std::make_shared<GrayImage>(img);
      std::exception_ptr failed;
      {
        std::lock_guard sub(submit_m);  // submit waits while the pipeline is full: the other decoders here
        const size_t n = next_number;
        {
          std::lock_guard lk(given_m);
          given.emplace(n, std::move(*w));
        }
        try {
          pipe->submit(std::move(img));
          ++next_number;
        } catch (...) {
          failed = std::current_exception();
          std::lock_guard lk(given_m);
          *w = std::move(given.at(n));
          given.erase(n);
        }
      }
      if (failed) {
        answer(*w, nullptr, failed, false);
        continue;
      }
      given_one();
    }
    feeder_done();
  }

  // ------------------------------------------------ the writers: each finished page's answer written in its
  // format (a PDF's deflate, an hOCR's XML) on a thread of their own, so the collect thread only takes pages.
  struct Done {
    Work w;
    OcrPage page;
    std::exception_ptr error;
    bool unreadable = false;
  };
  std::mutex w_m;
  std::condition_variable w_cv;
  std::deque<Done> to_write;
  bool writing_done = false;

  void hand_to_writers(Done d) {
    {
      std::lock_guard lk(w_m);
      to_write.push_back(std::move(d));
    }
    w_cv.notify_one();
  }

  void write_answers() {
    for (;;) {
      Done d;
      {
        std::unique_lock lk(w_m);
        w_cv.wait(lk, [&] { return writing_done || !to_write.empty(); });
        if (to_write.empty()) return;  // collect is done and every answer written
        d = std::move(to_write.front());
        to_write.pop_front();
      }
      answer(d.w, d.error ? nullptr : &d.page, d.error, d.unreadable);
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

  // The pages as they finish, each handed to the writers as soon as it is: the CPU engine's next_finished, and the
  // GPU pipeline's (a slow page holds back no other).
  void collect() {
    for (;;) {
      {
        std::unique_lock lk(given_m);
        given_cv.wait(lk, [&] { return submitted > taken || feeders_left == 0; });
        if (submitted == taken) return;  // every feeder gone, every page answered
      }
      size_t n = 0;
      OcrPage page;
      std::exception_ptr error;
      bool unreadable = false;
      if (engine) {
        std::optional<Engine::Finished> f = engine->next_finished();
        if (!f) continue;
        n = f->number;
        page = std::move(f->page);
        error = f->error;
        unreadable = f->unreadable;
      } else {
        std::optional<Pipeline::Finished> f;
        try {
          f = pipe->next_finished();
        } catch (...) {  // the pipeline stopped (a broken device): the oldest page waiting takes its error
          error = std::current_exception();
          std::lock_guard lk(given_m);
          if (given.empty()) continue;
          n = std::min_element(given.begin(), given.end(),
                               [](const auto& a, const auto& b) { return a.first < b.first; })->first;
        }
        if (f) {
          n = f->number;
          page = std::move(f->page);
          error = f->error;
        } else if (!error) {
          continue;
        }
      }
      Work w;
      {
        std::lock_guard lk(given_m);
        auto it = given.find(n);
        if (it == given.end()) continue;
        w = std::move(it->second);
        given.erase(it);
        ++taken;
      }
      hand_to_writers({std::move(w), std::move(page), error, unreadable});
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
    if (req.path == "/health" || req.path == "/info") {
      if (req.method != "GET") {
        error_reply(c, 405, req.path + " takes GET", keep, {{"Allow", "GET"}});
      } else if (req.path == "/health") {
        if (pipe && pipe->stopped()) {  // its device broken: nothing more will be read
          error_reply(c, 503, "the pipeline stopped: its device can run nothing more", keep);
        } else {
          reply(c, 200, "text/plain; charset=utf-8", "ok\n", keep);
        }
      } else {
        reply(c, 200, "application/json", info(), keep);
      }
      return;
    }
    if (req.path != "/ocr") {
      error_reply(c, 404, "no " + req.path + ": POST /ocr, GET /health, GET /info", keep);
      return;
    }
    if (req.method != "POST") {
      error_reply(c, 405, "/ocr takes POST, the page's image as its body", keep, {{"Allow", "POST"}});
      return;
    }
    if (!req.has_length) {
      error_reply(c, 411, "/ocr needs a Content-Length", false);
      return;
    }
    Work w;
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
    const Kind kind = sniff(body);
    if (body.empty() || kind != Kind::kImage) {
      error_reply(c, 415,
                  kind == Kind::kPdf ? "a PDF: the server reads page images (PNG, JPEG, BMP, PNM); carbon-ocr ocr "
                                       "reads PDFs"
                                     : "the body is no image the server reads: PNG, JPEG, BMP or PNM",
                  keep);
      return;
    }
    w.conn = id;
    w.keep_alive = keep;
    w.t0 = Clock::now();
    w.body = std::make_shared<const std::string>(std::move(body));
    {
      std::lock_guard lk(q_m);
      if (queue.size() >= static_cast<size_t>(o.queue)) {
        error_reply(c, 503, "busy: " + std::to_string(o.queue) + " requests waiting", keep, {{"Retry-After", "1"}});
        return;
      }
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
    // by default, at least 4, where 4 alone capped a server near 370 pages a second.
    const int feeders = o.cuda ? std::max(1, o.decoders > 0 ? o.decoders
                                                            : std::max(4, cpu_topology().default_workers() / 2))
                               : 1;
    feeders_left = feeders;
    for (int i = 0; i < feeders; ++i) {
      threads.emplace_back([this] { o.cuda ? feed_pipeline() : feed_engine(); });
    }
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
    }
    for (std::thread& t : threads) t.join();
    for (auto& [id, c] : conns) close_sock(c.fd);
    conns.clear();
    if (pipe) pipe->close();
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
