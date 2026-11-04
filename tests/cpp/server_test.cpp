// carbon::Server end to end on tests/data/fake-set: a server on a free port, clients on real sockets. Its
// answers are what the engine reads and the writers write for the same page (every format), on one connection
// after another or many at once; what it cannot read it refuses with the status it says; past its queue it
// answers 503 and goes on.
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
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "carbon/image.h"
#include "carbon/output.h"
#include "carbon/pipeline.h"

namespace {

using carbon::GrayImage;
using carbon::OcrPage;

const std::filesystem::path kSet = std::filesystem::path(CARBON_TEST_DATA) / "fake-set";

#if defined(_WIN32)
using Sock = SOCKET;
constexpr Sock kNoSock = INVALID_SOCKET;
void close_sock(Sock s) { closesocket(s); }
struct Net {
  Net() {
    WSADATA d;
    WSAStartup(MAKEWORD(2, 2), &d);
  }
  ~Net() { WSACleanup(); }
};
#else
using Sock = int;
constexpr Sock kNoSock = -1;
void close_sock(Sock s) { ::close(s); }
struct Net {};
#endif

struct Reply {
  int status = 0;
  std::string head, body;
  bool has(const std::string& header_line) const { return head.find(header_line + "\r\n") != std::string::npos; }
};

// A blocking client on one connection: requests written whole, replies read one at a time.
class Client {
 public:
  explicit Client(int port) {
    s_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s_, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) throw std::runtime_error("no connection");
#if defined(_WIN32)
    DWORD ms = 20000;  // no test waits forever
    setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms);
#else
    timeval tv{20, 0};
    setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
  }
  ~Client() { close_sock(s_); }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void write(const std::string& data) {
    size_t at = 0;
    while (at < data.size()) {
#if defined(MSG_NOSIGNAL)
      constexpr int flags = MSG_NOSIGNAL;  // a server that closed first is an error here, not a SIGPIPE
#else
      constexpr int flags = 0;
#endif
      const auto n = send(s_, data.data() + at, static_cast<int>(data.size() - at), flags);
      if (n <= 0) throw std::runtime_error("send failed");
      at += static_cast<size_t>(n);
    }
  }
  // The next reply, or nullopt where the server closed the connection first.
  std::optional<Reply> read() {
    for (;;) {
      const size_t end = buf_.find("\r\n\r\n");
      if (end != std::string::npos) {
        Reply r;
        r.head = buf_.substr(0, end + 2);
        r.status = std::stoi(r.head.substr(9, 3));
        size_t length = 0;
        if (const size_t at = r.head.find("Content-Length: "); at != std::string::npos) {
          length = std::stoul(r.head.substr(at + 16));
        }
        if (buf_.size() >= end + 4 + length) {
          r.body = buf_.substr(end + 4, length);
          buf_.erase(0, end + 4 + length);
          return r;
        }
      }
      char b[65536];
      const auto n = recv(s_, b, static_cast<int>(sizeof b), 0);
      if (n <= 0) return std::nullopt;
      buf_.append(b, static_cast<size_t>(n));
    }
  }
  Reply ask(const std::string& request) {
    write(request);
    std::optional<Reply> r = read();
    if (!r) throw std::runtime_error("the server closed the connection");
    return *r;
  }

 private:
  Sock s_ = kNoSock;
  std::string buf_;
};

std::string get(const std::string& target) { return "GET " + target + " HTTP/1.1\r\nHost: test\r\n\r\n"; }
std::string post(const std::string& target, const std::string& body, const std::string& headers = "") {
  return "POST " + target + " HTTP/1.1\r\nHost: test\r\nContent-Length: " + std::to_string(body.size()) + "\r\n" +
         headers + "\r\n" + body;
}

std::string file_bytes(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// A page's JSON without its timings, which differ from run to run.
std::string without_ms(const std::string& json) { return json.substr(0, json.rfind(", \"ms\": {")); }

// The page tiled n by n, as a PGM: a larger page, read more slowly.
std::string tiled_pgm(const GrayImage& p, int n) {
  std::string s = "P5\n" + std::to_string(p.width * n) + " " + std::to_string(p.height * n) + "\n255\n";
  for (int ty = 0; ty < n; ++ty) {
    for (int y = 0; y < p.height; ++y) {
      for (int tx = 0; tx < n; ++tx) s.append(reinterpret_cast<const char*>(&p.pixels[y * p.width]), p.width);
    }
  }
  return s;
}

// A server on a free port, serving on a thread of its own until the test ends.
struct Running {
  Net net;
  carbon::Server server;
  std::thread thread;
  explicit Running(carbon::ServerOptions o) : server(kSet, std::move(o)), thread([this] { server.run(); }) {}
  ~Running() {
    server.stop();
    thread.join();
  }
  int port() const { return server.port(); }
};

carbon::ServerOptions options(int workers) {
  carbon::ServerOptions o;
  o.port = 0;
  o.engine.pool.workers = workers;
  return o;
}

}  // namespace

TEST(Server, AnswersAPageAsTheEngineAndTheWritersRead) {
  const GrayImage image = carbon::load_image(kSet / "page.png");
  const std::string png = file_bytes(kSet / "page.png");
  carbon::ServerOptions o = options(2);
  const OcrPage want = carbon::Engine(kSet, o.engine).ocr(image.view());
  ASSERT_EQ(want.lines.size(), 3u);
  const std::vector<OcrPage> pages = {want};
  Running r(o);
  Client c(r.port());

  Reply h = c.ask(get("/health"));
  EXPECT_EQ(h.status, 200);
  EXPECT_EQ(h.body, "ok\n");
  h = c.ask(get("/info"));
  EXPECT_EQ(h.status, 200);
  EXPECT_NE(h.body.find("\"device\": \"cpu\", \"precision\": \"auto\", \"workers\": 2"), std::string::npos)
      << h.body;

  // every format on one connection, one request after another
  Reply j = c.ask(post("/ocr", png));
  EXPECT_EQ(j.status, 200);
  EXPECT_TRUE(j.has("Content-Type: application/json"));
  EXPECT_TRUE(j.has("X-Carbon-Lines: 3"));
  EXPECT_EQ(without_ms(j.body), without_ms(want.json())) << j.body;
  // html=1: the same object with the page's html body in it, as carbon_ocr's html=True gives it
  const Reply jh = c.ask(post("/ocr?format=json&html=1", png));
  const size_t at = jh.body.rfind(", \"html\": \"");
  ASSERT_NE(at, std::string::npos) << jh.body;
  EXPECT_EQ(without_ms(jh.body.substr(0, at)), without_ms(want.json()));
  std::string html = "\"";
  for (const char ch : carbon::to_html_body(want)) {
    if (ch == '"' || ch == '\\') {
      html += '\\';
      html += ch;
    } else if (static_cast<unsigned char>(ch) < 0x20) {  // a control character as \u00XX, as the server writes it
      char u[8];
      std::snprintf(u, sizeof u, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(ch)));
      html += u;
    } else {
      html += ch;
    }
  }
  EXPECT_EQ(jh.body.substr(at + 10), html + "\"}\n");
  EXPECT_EQ(c.ask(post("/ocr?format=text", png)).body, want.text());
  EXPECT_EQ(c.ask(post("/ocr?format=hocr", png)).body, carbon::to_hocr(pages, std::vector<std::string>{"page"}));
  EXPECT_EQ(c.ask(post("/ocr?format=alto&name=scan%201.png", png)).body,
            carbon::to_alto(pages, std::vector<std::string>{"scan 1.png"}));
  EXPECT_EQ(c.ask(post("/ocr?format=page&name=a.png", png)).body, carbon::to_page_xml(want, "a.png"));
  EXPECT_EQ(c.ask(post("/ocr?format=markdown", png)).body, carbon::to_markdown(pages));
  EXPECT_EQ(c.ask(post("/ocr?format=html", png)).body, carbon::to_html(pages));
  const Reply pdf = c.ask(post("/ocr?format=pdf", png));
  EXPECT_EQ(pdf.status, 200);
  EXPECT_TRUE(pdf.has("Content-Type: application/pdf"));
  const carbon::GrayView view = image.view();
  EXPECT_EQ(pdf.body, carbon::to_pdf(pages, std::span<const carbon::GrayView>(&view, 1), 300));

  // the client's 100-continue: the server asks for the body, then reads it
  c.write("POST /ocr?format=text HTTP/1.1\r\nHost: test\r\nContent-Length: " + std::to_string(png.size()) +
          "\r\nExpect: 100-continue\r\n\r\n");
  const std::optional<Reply> go = c.read();
  ASSERT_TRUE(go.has_value());
  EXPECT_EQ(go->status, 100);
  c.write(png);
  EXPECT_EQ(c.read()->body, want.text());
}

TEST(Server, ReadsManyClientsPagesAtOnce) {
  const std::string png = file_bytes(kSet / "page.png");
  carbon::ServerOptions o = options(3);
  const std::string want = carbon::Engine(kSet, o.engine).ocr(carbon::load_image(kSet / "page.png").view()).text();
  Running r(o);
  std::atomic<int> right{0};
  std::vector<std::thread> clients;
  for (int t = 0; t < 6; ++t) {
    clients.emplace_back([&] {
      Client c(r.port());
      for (int k = 0; k < 5; ++k) {
        const Reply a = c.ask(post("/ocr?format=text", png));
        if (a.status == 200 && a.body == want) ++right;
      }
    });
  }
  for (auto& t : clients) t.join();
  EXPECT_EQ(right.load(), 30);
}

TEST(Server, RefusesWhatItCannotRead) {
  const std::string png = file_bytes(kSet / "page.png");
  carbon::ServerOptions o = options(1);
  o.max_body = png.size() * 4;
  Running r(o);
  {
    Client c(r.port());
    EXPECT_EQ(c.ask(post("/ocr", "plain text, no image")).status, 415);
    const Reply pdf = c.ask(post("/ocr", "%PDF-1.7 and the rest"));
    EXPECT_EQ(pdf.status, 415);
    EXPECT_NE(pdf.body.find("carbon-ocr ocr reads PDFs"), std::string::npos);
    const Reply cut = c.ask(post("/ocr", png.substr(0, 64)));  // a PNG's start, and nothing after
    EXPECT_EQ(cut.status, 422) << cut.body;
    EXPECT_EQ(c.ask(post("/ocr?format=docx", png)).status, 400);
    EXPECT_EQ(c.ask(get("/nothing-here")).status, 404);
    const Reply m = c.ask(get("/ocr"));
    EXPECT_EQ(m.status, 405);
    EXPECT_TRUE(m.has("Allow: POST"));
    EXPECT_EQ(c.ask(post("/health", "")).status, 405);
    EXPECT_EQ(c.ask(post("/ocr?format=text", png)).status, 200);  // the connection still serves
  }
  {
    Client c(r.port());
    EXPECT_EQ(c.ask("POST /ocr HTTP/1.1\r\nHost: test\r\n\r\n").status, 411);
    EXPECT_FALSE(c.read().has_value());  // and closed
  }
  {
    Client c(r.port());
    EXPECT_EQ(c.ask(post("/ocr", std::string(png.size() * 5, 'x'))).status, 413);
  }
  {
    Client c(r.port());
    EXPECT_EQ(c.ask("GET  /health HTTP/1.1\r\n\r\n").status, 400);
    EXPECT_FALSE(c.read().has_value());
  }
  {
    Client c(r.port());
    EXPECT_EQ(c.ask(post("/ocr?format=text", png)).status, 200);  // the server itself goes on
  }
}

TEST(Server, AnswersBusyPastItsQueueAndGoesOn) {
  const std::string big = tiled_pgm(carbon::load_image(kSet / "page.png"), 4);
  carbon::ServerOptions o = options(1);
  o.engine.in_flight = 1;
  o.queue = 1;
  Running r(o);
  std::vector<std::unique_ptr<Client>> clients;
  for (int i = 0; i < 24; ++i) {  // every request in before any is answered: past one page and one waiting, 503
    clients.push_back(std::make_unique<Client>(r.port()));
    clients.back()->write(post("/ocr?format=text", big));
  }
  int ok = 0, busy = 0;
  std::string first;
  for (auto& c : clients) {
    const std::optional<Reply> a = c->read();
    ASSERT_TRUE(a.has_value());
    if (a->status == 200) {
      if (!ok++) first = a->body;
      EXPECT_EQ(a->body, first);
    } else {
      EXPECT_EQ(a->status, 503);
      EXPECT_TRUE(a->has("Retry-After: 1"));
      ++busy;
    }
  }
  EXPECT_GE(ok, 1);
  EXPECT_GE(busy, 1);
  EXPECT_EQ(ok + busy, 24);
  Client c(r.port());
  EXPECT_EQ(c.ask(post("/ocr?format=text", big)).status, 200);
}

// On a GPU: the pages decoded by the server's decoders, read by the batched pipeline in its batches, every client's
// pages among them, each answered as the CPU reads it; at fp32 and int8 too. Skipped without a GPU.
TEST(Server, ReadsOnTheGpuAsTheCpuReads) {
  if (!carbon::cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const std::string png = file_bytes(kSet / "page.png");
  const GrayImage image = carbon::load_image(kSet / "page.png");
  const std::string want = carbon::Engine(kSet, options(2).engine).ocr(image.view()).text();
  for (const carbon::Precision p : {carbon::Precision::kAuto, carbon::Precision::kFp32, carbon::Precision::kInt8}) {
    SCOPED_TRACE(std::string(carbon::precision_name(p)));
    carbon::ServerOptions o = options(2);
    o.cuda = true;
    o.engine.precision = p;
    o.pipeline.page_batch = 4;
    o.pipeline.engine_cache = std::filesystem::temp_directory_path() / "carbon-test-engines";
    Running r(o);
    std::atomic<int> right{0};
    std::vector<std::thread> clients;
    for (int t = 0; t < 3; ++t) {
      clients.emplace_back([&] {
        Client c(r.port());
        for (int k = 0; k < 4; ++k) {
          const Reply a = c.ask(post("/ocr?format=text", png));
          if (a.status == 200 && a.body == want) ++right;
        }
      });
    }
    for (auto& t : clients) t.join();
    EXPECT_EQ(right.load(), 12);
    Client c(r.port());
    EXPECT_NE(c.ask(get("/info")).body.find("\"device\": \"cuda\""), std::string::npos);
    EXPECT_EQ(c.ask(post("/ocr", png.substr(0, 64))).status, 422);  // decoded on the server's decoders
  }
}
